#!/usr/bin/env tsx
// zkMetal CLI — generic Noir prover with pluggable adapters.
//
// Usage:
//   tsx cli.ts prove    --circuit ./target/my_circuit.json --witness witness.json
//   tsx cli.ts verify   --proof proof.json
//   tsx cli.ts watch    --circuit ./target/my_circuit.json --adapter persistia --node http://localhost:8787
//   tsx cli.ts bench    --circuit ./target/my_circuit.json --adapter persistia --node http://localhost:8787 --block 5

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { resolve, join } from "path";
import { execSync } from "child_process";
import { ProverEngine, extractInnerProof } from "./engine.js";
import { watchSequential, watchPipelined, watchParallelMsgpack } from "./watcher.js";
import type { DataSource, WitnessBuilder, ProofSink, ProofOutput } from "./types.js";

// --- CLI Helpers ---

const args = process.argv.slice(2);
const command = args[0];

function getArg(name: string, defaultVal?: string): string {
  const idx = args.indexOf(`--${name}`);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  if (defaultVal !== undefined) return defaultVal;
  throw new Error(`Missing required argument: --${name}`);
}

function hasArg(name: string): boolean {
  return args.includes(`--${name}`);
}

const DEFAULT_CIRCUIT = resolve(
  import.meta.dirname ?? ".",
  "../../target/persistia_state_proof.json",
);

function getEngine(): ProverEngine {
  const circuitPath = getArg("circuit", DEFAULT_CIRCUIT);
  return new ProverEngine({
    circuitPath,
    threads: parseInt(getArg("threads", "8")),
  });
}

// --- Adapter Loading ---

async function loadAdapter(nodeBase: string): Promise<{
  dataSource: DataSource<any>;
  witnessBuilder: WitnessBuilder<any>;
  proofSink: ProofSink;
}> {
  const adapterName = getArg("adapter", "persistia");

  switch (adapterName) {
    case "persistia": {
      const { createPersistiaAdapter } = await import("./adapters/persistia/index.js");
      return createPersistiaAdapter(nodeBase);
    }
    default:
      throw new Error(
        `Unknown adapter: ${adapterName}. Available: persistia\n` +
        `To add a custom adapter, implement DataSource, WitnessBuilder, and ProofSink from zkmetal/types.`,
      );
  }
}

// --- Commands ---

async function cmdProve() {
  const engine = getEngine();
  const nodeBase = getArg("node", "http://localhost:8787");
  const blockNumber = parseInt(getArg("block"));
  const outputPath = getArg("output", "proof.json");
  const useNative = hasArg("native");
  const prevProofPath = hasArg("prev-proof") ? getArg("prev-proof") : undefined;

  console.log(`Generating proof for block ${blockNumber}...`);
  const { dataSource, witnessBuilder } = await loadAdapter(nodeBase);

  const block = await dataSource.fetchBlock(blockNumber);

  let recursiveOpts: any;
  if (prevProofPath && existsSync(prevProofPath)) {
    const prevData = JSON.parse(readFileSync(prevProofPath, "utf-8"));
    const prevProofBytes = Buffer.from(prevData.proof, "base64");
    const innerProof = extractInnerProof(new Uint8Array(prevProofBytes));

    let vkAsFields: string[];
    let vkHash: string;
    if (prevData.vkAsFields && prevData.vkHash) {
      vkAsFields = prevData.vkAsFields;
      vkHash = prevData.vkHash;
    } else {
      const artifacts = await engine.generateRecursiveArtifacts({
        proof: prevProofBytes,
        publicInputs: prevData.publicInputs,
      });
      vkAsFields = artifacts.vkAsFields;
      vkHash = artifacts.vkHash;
    }

    recursiveOpts = {
      prevProvenBlocks: prevData.provenBlocks ?? prevData.proven_blocks ?? 0,
      prevGenesisRoot: prevData.meta?.genesis_root ?? "0",
      prevProof: innerProof,
      prevVk: vkAsFields,
      prevKeyHash: vkHash,
      prevPublicInputs: prevData.publicInputs,
    };
    console.log(`Previous proof loaded (${recursiveOpts.prevProvenBlocks} blocks). Recursive chaining enabled.`);
  }

  const witness = await witnessBuilder.buildWitness(block, blockNumber, recursiveOpts);
  const { witness: solvedWitness } = await engine.execute(witness);

  const start = performance.now();
  let proof: { proof: Uint8Array; publicInputs: string[] };
  let vkBytes: Uint8Array | undefined;

  if (useNative && engine.nativeBbAvailable()) {
    console.log("Using native bb CLI...");
    const result = engine.nativeProve(solvedWitness);
    proof = result;
    vkBytes = result.vk;
  } else {
    if (useNative) console.warn("Native bb not available, falling back to WASM");
    proof = await engine.prove(solvedWitness);
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(2);
  const artifacts = await engine.generateRecursiveArtifacts(proof);

  const proofOutput: ProofOutput = {
    proof: Buffer.from(proof.proof).toString("base64"),
    publicInputs: proof.publicInputs,
    vkAsFields: artifacts.vkAsFields,
    vkHash: artifacts.vkHash,
    vk: vkBytes ? Buffer.from(vkBytes).toString("base64") : undefined,
    blockNumber,
    provenBlocks: recursiveOpts ? recursiveOpts.prevProvenBlocks + 1 : 1,
    prover: useNative ? "zkmetal-native" : "zkmetal-wasm",
    timestamp: new Date().toISOString(),
  };

  const dir = resolve(outputPath, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(outputPath, JSON.stringify(proofOutput, null, 2));

  console.log(`Proof generated in ${elapsed}s -> ${outputPath}`);
  console.log(`  Proven blocks: ${proofOutput.provenBlocks}`);
  console.log(`  Proof size: ${proof.proof.length} bytes`);
  console.log(`  Recursive: ${prevProofPath ? "yes (chained)" : "no (genesis)"}`);

  await engine.destroy();
}

async function cmdVerify() {
  const proofPath = getArg("proof");
  const useNative = hasArg("native");
  const engine = getEngine();

  console.log(`Verifying proof from ${proofPath}...`);
  const proofData = JSON.parse(readFileSync(proofPath, "utf-8"));
  const proofBytes = Buffer.from(proofData.proof, "base64");

  let valid: boolean;
  const start = performance.now();

  if (useNative && proofData.vk && engine.nativeBbAvailable()) {
    console.log("Using native bb CLI...");
    const vkBytes = Buffer.from(proofData.vk, "base64");
    valid = engine.nativeVerify(new Uint8Array(proofBytes), proofData.publicInputs, new Uint8Array(vkBytes));
  } else {
    if (useNative && !proofData.vk) console.warn("No VK in proof file, falling back to WASM");
    valid = await engine.verify({ proof: proofBytes, publicInputs: proofData.publicInputs });
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(2);

  if (valid) {
    console.log(`Proof VALID (verified in ${elapsed}s)`);
    console.log(`  Block: ${proofData.blockNumber ?? proofData.block_number}`);
    console.log(`  Proven blocks: ${proofData.provenBlocks ?? proofData.proven_blocks}`);
  } else {
    console.error("Proof INVALID");
    process.exit(1);
  }

  await engine.destroy();
}

async function cmdWatch() {
  const engine = getEngine();
  const nodeBase = getArg("node", "http://localhost:8787");
  const { dataSource, witnessBuilder, proofSink } = await loadAdapter(nodeBase);

  const mode = getArg("mode", "sequential");
  const opts = {
    intervalSec: parseInt(getArg("interval", "10")),
    startBlock: hasArg("start") ? parseInt(getArg("start")) : undefined,
    useNative: hasArg("native"),
    recursive: hasArg("recursive"),
    proofDir: getArg("proof-dir", "./proofs"),
  };

  switch (mode) {
    case "sequential":
      await watchSequential(engine, dataSource, witnessBuilder, proofSink, opts);
      break;
    case "pipelined":
      await watchPipelined(engine, dataSource, witnessBuilder, proofSink, opts);
      break;
    case "parallel":
      await watchParallelMsgpack(engine, dataSource, witnessBuilder, proofSink, {
        ...opts,
        workers: parseInt(getArg("workers", "6")),
      });
      break;
    default:
      throw new Error(`Unknown watch mode: ${mode}. Available: sequential, pipelined, parallel`);
  }
}

async function cmdBench() {
  const engine = getEngine();
  const nodeBase = getArg("node", "http://localhost:8787");
  const blockNumber = parseInt(getArg("block"));
  const { dataSource, witnessBuilder } = await loadAdapter(nodeBase);

  console.log("=== zkMetal Circuit Benchmark ===\n");

  const block = await dataSource.fetchBlock(blockNumber);
  const witness = await witnessBuilder.buildWitness(block, blockNumber);

  // Warmup
  console.log("Warming up...");
  await engine.execute(witness);

  // Benchmark execute
  console.log("\n--- Execute (witness solving) ---");
  const execTimes: number[] = [];
  for (let i = 0; i < 3; i++) {
    const start = performance.now();
    await engine.execute(witness);
    execTimes.push(performance.now() - start);
  }
  const avgExec = execTimes.reduce((a, b) => a + b) / execTimes.length;
  console.log(`  Avg: ${(avgExec / 1000).toFixed(3)}s`);

  // Benchmark prove
  console.log("\n--- Prove (full proof generation) ---");
  const start = performance.now();
  const { witness: solved } = await engine.execute(witness);
  const proof = await engine.prove(solved);
  const proveTime = performance.now() - start;
  console.log(`  Time: ${(proveTime / 1000).toFixed(3)}s`);
  console.log(`  Proof size: ${proof.proof.length} bytes`);

  // Benchmark verify
  console.log("\n--- Verify ---");
  const verifyStart = performance.now();
  const valid = await engine.verify(proof);
  const verifyTime = performance.now() - verifyStart;
  console.log(`  Time: ${(verifyTime / 1000).toFixed(3)}s`);
  console.log(`  Valid: ${valid}`);

  console.log("\n=== Summary ===");
  console.log(`  Execute:  ${(avgExec / 1000).toFixed(3)}s`);
  console.log(`  Prove:    ${(proveTime / 1000).toFixed(3)}s`);
  console.log(`  Verify:   ${(verifyTime / 1000).toFixed(3)}s`);
  console.log(`  Proof:    ${proof.proof.length} bytes`);

  await engine.destroy();
}

async function cmdBbVersion() {
  const engine = getEngine();
  const config = engine.getConfig();
  try {
    const current = execSync(`${config.bbPath} --version`, { encoding: "utf-8" }).trim();
    console.log(`bb version: ${current}`);
    console.log(`bb path:    ${config.bbPath}`);
  } catch {
    console.log("bb CLI not found. Install with:");
    console.log("  curl -L https://raw.githubusercontent.com/AztecProtocol/aztec-packages/master/barretenberg/bbup/install | bash");
    console.log("  bbup -v 4.1.2");
  }
}

// --- Main ---

switch (command) {
  case "prove":
    cmdProve();
    break;
  case "verify":
    cmdVerify();
    break;
  case "watch":
    cmdWatch();
    break;
  case "bench":
    cmdBench();
    break;
  case "bb-version":
    cmdBbVersion();
    break;
  case "gpu-info": {
    const engine = getEngine();
    const gpuInfo = engine.metalGpuInfo();
    if (gpuInfo) {
      console.log(`Metal GPU: ${gpuInfo.gpu}`);
      console.log(`Unified memory: ${gpuInfo.unified_memory}`);
      console.log(`MSM acceleration: available`);
      // Quick correctness check: 2*G should give known BN254 2G
      const result = engine.metalMsm(
        [["0x1", "0x2"]],
        ["0x2"],
      );
      if (result && result.x === "0x030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3") {
        console.log(`MSM correctness: verified (2*G matches known value)`);
      } else {
        console.log(`MSM correctness: FAILED`);
      }
    } else {
      console.log("Metal GPU: not available (zkmsm binary not found)");
      console.log("Install: cd metal && swift build -c release && mkdir -p ~/.zkmsm/shaders && cp .build/release/zkmsm ~/.zkmsm/ && cp Sources/zkmsm/shaders/bn254.metal ~/.zkmsm/shaders/");
    }
    const bbAvailable = engine.nativeBbAvailable();
    console.log(`Native bb: ${bbAvailable ? "available" : "not found"}`);
    break;
  }
  default:
    console.log(`zkMetal — Generic Noir Prover

Usage:
  tsx cli.ts prove    --block <n> [--node <url>] [--output proof.json] [--prev-proof <path>] [--native]
  tsx cli.ts verify   --proof <path> [--native]
  tsx cli.ts watch    --node <url> [--mode sequential|pipelined|parallel] [--recursive] [--native]
  tsx cli.ts bench    --block <n> [--node <url>]
  tsx cli.ts bb-version

Options:
  --circuit <path>    Path to compiled Noir circuit JSON (default: target/persistia_state_proof.json)
  --adapter <name>    Data adapter: persistia (default: persistia)
  --node <url>        Node URL for data source
  --native            Use native bb CLI instead of WASM (faster)
  --recursive         Enable recursive IVC proof chaining
  --mode <mode>       Watch mode: sequential, pipelined, parallel
  --workers <n>       Number of parallel workers (default: 6)
  --interval <n>      Poll interval in seconds (default: 10)
  --threads <n>       WASM threads (default: 8)

Adapters:
  persistia           Fetches blocks from a Persistia node (reference implementation)
  (custom)            Implement DataSource, WitnessBuilder, ProofSink from zkmetal/types`);
  }
