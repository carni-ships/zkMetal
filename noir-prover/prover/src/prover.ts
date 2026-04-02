#!/usr/bin/env tsx
// Persistia Noir Prover -- Host-side proof generation and verification.
//
// Drop-in replacement for the SP1 prover (contracts/zk/prover/).
// Uses @noir-lang/noir_js + Barretenberg UltraHonk backend.
//
// Usage:
//   tsx prover.ts prove   --node http://localhost:8787 --block 5
//   tsx prover.ts execute --node http://localhost:8787 --block 5
//   tsx prover.ts verify  --proof proof.bin
//   tsx prover.ts watch   --node http://localhost:8787
//   tsx prover.ts bench   --block 5

import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from "fs";
import { join, resolve } from "path";
import { exec, execSync, spawn, fork, type ChildProcess } from "child_process";
import { gunzipSync } from "zlib";
import { Encoder, Decoder } from "msgpackr";
import { buildSingleBlockWitness, buildTestWitness, destroyBb, buildMutationWitness, emptyMutation, computePoseidon2MerkleRoot, type CircuitWitness } from "./witness.js";

// --- Proof Field Conversion ---
// UltraHonk proofs are serialized as concatenated 32-byte big-endian field elements.
// The first 51 fields are overhead (pairing points etc), the remaining 449 are the
// "inner proof" needed for recursive verification inside the circuit.

const PROOF_OVERHEAD_FIELDS = 51;

function proofToFields(proofBytes: Uint8Array): string[] {
  const fields: string[] = [];
  for (let i = 0; i < proofBytes.length; i += 32) {
    const chunk = proofBytes.slice(i, i + 32);
    const hex = "0x" + Array.from(chunk).map(b => b.toString(16).padStart(2, "0")).join("");
    fields.push(hex);
  }
  return fields;
}

// --- Circuit Loading ---

const CIRCUIT_PATH = resolve(import.meta.dirname ?? ".", "../../target/persistia_state_proof.json");

function loadCircuit() {
  if (!existsSync(CIRCUIT_PATH)) {
    console.error(`Circuit not found at ${CIRCUIT_PATH}`);
    console.error("Run 'nargo compile' in contracts/zk-noir/ first.");
    process.exit(1);
  }
  return JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
}

async function createProver() {
  const circuit = loadCircuit();
  const api = await Barretenberg.new({ threads: 8 });
  const backend = new UltraHonkBackend(circuit.bytecode, api);
  const noir = new Noir(circuit);
  return { backend, noir, circuit, api };
}

// --- Native bb CLI ---

const BB_PATH = join(process.env.HOME ?? "~", ".bb", "bb");

function nativeBbAvailable(): boolean {
  try {
    execSync(`${BB_PATH} --version`, { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// Precomputed VK caches — bb write_vk outputs to a directory containing vk + vk_hash.
// We cache both noir-recursive and evm VKs alongside the circuit artifact.
type VerifierTarget = "noir-recursive-no-zk" | "evm-no-zk";

const VK_CACHE_DIRS: Record<VerifierTarget, string> = {
  "noir-recursive-no-zk": resolve(import.meta.dirname ?? ".", "../../target/bb_vk"),
  "evm-no-zk": resolve(import.meta.dirname ?? ".", "../../target/bb_vk_evm"),
};

function ensureVkCached(target: VerifierTarget = "noir-recursive-no-zk"): string | null {
  const cacheDir = VK_CACHE_DIRS[target];
  const cacheFile = join(cacheDir, "vk");
  if (existsSync(cacheFile)) return cacheFile;
  try {
    mkdirSync(cacheDir, { recursive: true });
    execSync(
      `${BB_PATH} write_vk -b ${CIRCUIT_PATH} -o ${cacheDir} -t ${target}`,
      { stdio: "pipe" },
    );
    if (existsSync(cacheFile)) return cacheFile;
  } catch {}
  return null;
}

function nativeBbProve(
  witnessBytes: Uint8Array,
  target: VerifierTarget = "noir-recursive-no-zk",
): { proof: Uint8Array; publicInputs: string[] } {
  const tmpDir = "/tmp/persistia_bb_prove";
  rmSync(tmpDir, { recursive: true, force: true });

  // Use precomputed VK if available (saves ~700ms per prove)
  const vkPath = ensureVkCached(target);
  const vkFlag = vkPath ? ` -k ${vkPath}` : " --write_vk";

  writeFileSync("/tmp/persistia_bb_witness.gz", witnessBytes);
  execSync(
    `${BB_PATH} prove -b ${CIRCUIT_PATH} -w /tmp/persistia_bb_witness.gz -o ${tmpDir}${vkFlag} -t ${target}`,
    { stdio: "pipe" },
  );

  const proof = readFileSync(join(tmpDir, "proof"));
  const piRaw = readFileSync(join(tmpDir, "public_inputs"));

  // Public inputs are concatenated 32-byte fields
  const publicInputs: string[] = [];
  for (let i = 0; i < piRaw.length; i += 32) {
    const hex = "0x" + Array.from(piRaw.subarray(i, i + 32)).map(b => b.toString(16).padStart(2, "0")).join("");
    publicInputs.push(hex);
  }

  return { proof: new Uint8Array(proof), publicInputs };
}

function nativeBbVerify(proof: Uint8Array, publicInputs: string[], vk: Uint8Array): boolean {
  const tmpDir = "/tmp/persistia_bb_verify";
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(tmpDir, { recursive: true });

  writeFileSync(join(tmpDir, "proof"), proof);
  writeFileSync(join(tmpDir, "vk"), vk);

  // Write public inputs as concatenated 32-byte fields
  const piBytes = Buffer.alloc(publicInputs.length * 32);
  for (let i = 0; i < publicInputs.length; i++) {
    const val = BigInt(publicInputs[i]);
    const hex = val.toString(16).padStart(64, "0");
    for (let j = 0; j < 32; j++) {
      piBytes[i * 32 + j] = parseInt(hex.substring(j * 2, j * 2 + 2), 16);
    }
  }
  writeFileSync(join(tmpDir, "public_inputs"), piBytes);

  try {
    execSync(
      `${BB_PATH} verify -p ${tmpDir}/proof -k ${tmpDir}/vk -i ${tmpDir}/public_inputs -t noir-recursive-no-zk`,
      { stdio: "pipe" },
    );
    return true;
  } catch {
    return false;
  }
}

// --- Node API ---

function nodeUrl(base: string, path: string): string {
  try {
    const u = new URL(base);
    // Split path from its query string so pathname doesn't encode the '?'
    const [pathname, query] = path.split("?", 2);
    u.pathname = u.pathname.replace(/\/$/, "") + pathname;
    if (query) {
      for (const param of query.split("&")) {
        const [k, v] = param.split("=", 2);
        u.searchParams.set(k, v ?? "");
      }
    }
    return u.toString();
  } catch {
    return `${base}${path}`;
  }
}

async function fetchBlock(nodeBase: string, blockNumber: number) {
  const url = nodeUrl(nodeBase, `/proof/block?block=${blockNumber}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch block ${blockNumber}: ${res.status}`);
  return res.json();
}

async function fetchLatestBlock(nodeBase: string): Promise<number> {
  const url = nodeUrl(nodeBase, "/proof/zk/status");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ZK status: ${res.status}`);
  const data = await res.json() as any;
  return data.last_committed_round ?? data.latest_block ?? data.latestBlock ?? 0;
}

async function submitProof(nodeBase: string, proofData: any) {
  const url = nodeUrl(nodeBase, "/proof/zk/submit");
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(proofData),
  });
  if (!res.ok) throw new Error(`Failed to submit proof: ${res.status}`);
  return res.json();
}

// --- State Root Computation ---
// The Noir circuit's state_root is a Poseidon2 Merkle root of the block's mutations.
// prev_state_root for block N = new_state_root of block N-1.
// For block 1, prev_state_root is a genesis value ("0").

const GENESIS_STATE_ROOT = "0";

async function computeBlockStateRoot(block: any): Promise<string> {
  if (!block.mutations || block.mutations.length === 0) {
    return GENESIS_STATE_ROOT;
  }
  const muts = block.mutations.map(buildMutationWitness);
  while (muts.length < 32) muts.push(emptyMutation());
  return computePoseidon2MerkleRoot(muts.slice(0, 32));
}

async function fetchPrevStateRoot(nodeBase: string, blockNumber: number): Promise<string> {
  if (blockNumber <= 1) return GENESIS_STATE_ROOT;
  try {
    const prevBlock = await fetchBlock(nodeBase, blockNumber - 1);
    return computeBlockStateRoot(prevBlock);
  } catch {
    // Previous block not available — use genesis
    return GENESIS_STATE_ROOT;
  }
}

// --- Commands ---

async function cmdExecute(nodeBase: string, blockNumber: number) {
  console.log(`Executing circuit for block ${blockNumber} (no proof)...`);
  const { noir } = await createProver();

  const block = await fetchBlock(nodeBase, blockNumber);
  const prevStateRoot = await fetchPrevStateRoot(nodeBase, blockNumber);
  const witness = await buildSingleBlockWitness(block, prevStateRoot);

  const start = performance.now();
  const result = await noir.execute(witness as any);
  const elapsed = ((performance.now() - start) / 1000).toFixed(2);

  console.log(`Execution OK in ${elapsed}s`);
  console.log("Public output:", result.returnValue);
}

async function cmdProve(
  nodeBase: string,
  blockNumber: number,
  outputPath: string,
  prevProofPath?: string,
  useNative = false,
  target: VerifierTarget = "noir-recursive-no-zk",
) {
  console.log(`Generating proof for block ${blockNumber}...`);
  const { noir, backend } = await createProver();

  const block = await fetchBlock(nodeBase, blockNumber);

  let opts: any = {};
  if (prevProofPath && existsSync(prevProofPath)) {
    const prevProofData = JSON.parse(readFileSync(prevProofPath, "utf-8"));

    // Extract inner proof fields (skip 51-field overhead)
    const prevProofBytes = Buffer.from(prevProofData.proof, "base64");
    const allProofFields = proofToFields(new Uint8Array(prevProofBytes));
    const innerProof = allProofFields.slice(PROOF_OVERHEAD_FIELDS);

    // Use pre-stored VK fields if available, otherwise extract them
    let vkAsFields: string[];
    let vkHash: string;
    if (prevProofData.vkAsFields && prevProofData.vkHash) {
      vkAsFields = prevProofData.vkAsFields;
      vkHash = prevProofData.vkHash;
    } else {
      const artifacts = await backend.generateRecursiveProofArtifacts(
        { proof: prevProofBytes, publicInputs: prevProofData.publicInputs },
        prevProofData.publicInputs.length,
      );
      vkAsFields = artifacts.vkAsFields;
      vkHash = artifacts.vkHash;
    }

    opts = {
      prevProvenBlocks: prevProofData.proven_blocks,
      prevGenesisRoot: prevProofData.genesis_root,
      prevProof: innerProof,
      prevVk: vkAsFields,
      prevKeyHash: vkHash,
      prevPublicInputs: prevProofData.publicInputs,
    };
    console.log(`Previous proof loaded (${prevProofData.proven_blocks} blocks). Recursive chaining enabled.`);
  }

  const prevStateRoot = await fetchPrevStateRoot(nodeBase, blockNumber);
  const witness = await buildSingleBlockWitness(block, prevStateRoot, opts);

  const start = performance.now();
  const { witness: solvedWitness } = await noir.execute(witness as any);

  let proof: { proof: Uint8Array; publicInputs: string[] };
  let vkBytes: Uint8Array | undefined;

  if (useNative && nativeBbAvailable()) {
    console.log(`Using native bb CLI for proving (${target})...`);
    proof = nativeBbProve(solvedWitness, target);
    vkBytes = readFileSync("/tmp/persistia_bb_prove/vk");
  } else {
    if (useNative) console.warn("Native bb not available, falling back to WASM");
    proof = await backend.generateProof(solvedWitness);
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(2);

  // Extract recursive artifacts for this proof so the next block can chain
  const artifacts = await backend.generateRecursiveProofArtifacts(
    proof,
    proof.publicInputs.length,
  );

  const proofData: any = {
    proof: Buffer.from(proof.proof).toString("base64"),
    publicInputs: proof.publicInputs,
    // Store VK fields and hash directly for the next proof in the chain
    vkAsFields: artifacts.vkAsFields,
    vkHash: artifacts.vkHash,
    block_number: blockNumber,
    proven_blocks: prevProofPath ? (opts.prevProvenBlocks + 1) : 1,
    genesis_root: opts.prevGenesisRoot ?? prevStateRoot,
    state_root: witness.new_state_root,
    prover: useNative ? "noir-ultrahonk-native" : "noir-ultrahonk",
    timestamp: new Date().toISOString(),
  };

  if (vkBytes) {
    proofData.vk = Buffer.from(vkBytes).toString("base64");
  }

  const dir = resolve(outputPath, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(outputPath, JSON.stringify(proofData, null, 2));

  console.log(`Proof generated in ${elapsed}s -> ${outputPath}`);
  console.log(`  State root: ${witness.new_state_root}`);
  console.log(`  Proven blocks: ${proofData.proven_blocks}`);
  console.log(`  Proof size: ${proof.proof.length} bytes`);
  console.log(`  Recursive: ${prevProofPath ? "yes (chained)" : "no (genesis)"}`);

  if (typeof backend.destroy === "function") await backend.destroy();
}

async function cmdVerify(proofPath: string, useNative = false) {
  console.log(`Verifying proof from ${proofPath}...`);

  const proofData = JSON.parse(readFileSync(proofPath, "utf-8"));
  const proofBytes = Buffer.from(proofData.proof, "base64");

  let valid: boolean;
  let elapsed: string;

  if (useNative && proofData.vk && nativeBbAvailable()) {
    console.log("Using native bb CLI for verification...");
    const vkBytes = Buffer.from(proofData.vk, "base64");
    const start = performance.now();
    valid = nativeBbVerify(new Uint8Array(proofBytes), proofData.publicInputs, new Uint8Array(vkBytes));
    elapsed = ((performance.now() - start) / 1000).toFixed(2);
  } else {
    if (useNative && !proofData.vk) console.warn("No VK in proof file, falling back to WASM");
    const { backend } = await createProver();
    const start = performance.now();
    valid = await backend.verifyProof({ proof: proofBytes, publicInputs: proofData.publicInputs });
    elapsed = ((performance.now() - start) / 1000).toFixed(2);
    if (typeof backend.destroy === "function") await backend.destroy();
  }

  if (valid) {
    console.log(`Proof VALID (verified in ${elapsed}s)`);
    console.log(`  Block: ${proofData.block_number}`);
    console.log(`  State root: ${proofData.state_root}`);
    console.log(`  Proven blocks: ${proofData.proven_blocks}`);
    console.log(`  Genesis root: ${proofData.genesis_root}`);
  } else {
    console.error("Proof INVALID");
    process.exit(1);
  }
}

async function cmdWatch(
  nodeBase: string,
  proofDir: string,
  intervalSec: number,
  useNative = false,
  startBlock?: number,
) {
  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  // Default: start from current committed round (skip historical blocks without mutations)
  let lastProvenBlock = startBlock ?? (await fetchLatestBlock(nodeBase));
  console.log(`Watching ${nodeBase} (interval=${intervalSec}s, starting after block ${lastProvenBlock})`);
  let prevProofPath: string | undefined;

  while (true) {
    try {
      const latestBlock = await fetchLatestBlock(nodeBase);

      // Catch up to latest committed block
      while (latestBlock > lastProvenBlock) {
        const blockNum = lastProvenBlock + 1;
        const outPath = join(proofDir, `block_${blockNum}.json`);
        let proved = false;
        try {
          await cmdProve(nodeBase, blockNum, outPath, prevProofPath, useNative);
          prevProofPath = outPath;
          proved = true;

          // Submit to node
          try {
            const proofData = JSON.parse(readFileSync(outPath, "utf-8"));
            await submitProof(nodeBase, proofData);
            console.log("Proof submitted to node");
          } catch (e: any) {
            console.warn(`Submit failed (non-fatal): ${e.message}`);
          }
        } catch (e: any) {
          // Skip blocks that don't exist (pre-deploy blocks without mutations)
          if (e.message?.includes("404") || e.message?.includes("not found") || e.message?.includes("not committed") || e.message?.includes("Too many mutations")) {
            console.log(`Block ${blockNum} not available, skipping`);
          } else {
            throw e;
          }
        }
        lastProvenBlock = blockNum;
      }
    } catch (e: any) {
      console.error(`Error: ${e.message}`);
    }

    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
}

async function cmdBench(nodeBase: string, blockNumber: number) {
  console.log("=== Noir Circuit Benchmark ===\n");
  const { noir, backend } = await createProver();

  const block = await fetchBlock(nodeBase, blockNumber);
  const prevStateRoot = await fetchPrevStateRoot(nodeBase, blockNumber);
  const witness = await buildSingleBlockWitness(block, prevStateRoot);

  // Warmup
  console.log("Warming up...");
  await noir.execute(witness as any);

  // Benchmark execute (witness solving)
  console.log("\n--- Execute (witness solving) ---");
  const execTimes: number[] = [];
  for (let i = 0; i < 3; i++) {
    const start = performance.now();
    await noir.execute(witness as any);
    execTimes.push(performance.now() - start);
  }
  const avgExec = execTimes.reduce((a, b) => a + b) / execTimes.length;
  console.log(`  Avg: ${(avgExec / 1000).toFixed(3)}s`);

  // Benchmark proof generation
  console.log("\n--- Prove (full proof generation) ---");
  const start = performance.now();
  const { witness: solved } = await noir.execute(witness as any);
  const proof = await backend.generateProof(solved);
  const proveTime = performance.now() - start;
  console.log(`  Time: ${(proveTime / 1000).toFixed(3)}s`);
  console.log(`  Proof size: ${proof.proof.length} bytes`);

  // Benchmark verification
  console.log("\n--- Verify ---");
  const verifyStart = performance.now();
  const valid = await backend.verifyProof(proof);
  const verifyTime = performance.now() - verifyStart;
  console.log(`  Time: ${(verifyTime / 1000).toFixed(3)}s`);
  console.log(`  Valid: ${valid}`);

  console.log("\n=== Summary ===");
  console.log(`  Execute:  ${(avgExec / 1000).toFixed(3)}s`);
  console.log(`  Prove:    ${(proveTime / 1000).toFixed(3)}s`);
  console.log(`  Verify:   ${(verifyTime / 1000).toFixed(3)}s`);
  console.log(`  Proof:    ${proof.proof.length} bytes`);
  console.log(`  Speedup vs SP1 (~70s): ~${(70000 / proveTime).toFixed(1)}x`);

  if (typeof backend.destroy === "function") await backend.destroy();
}

// --- Parallel Prover ---
// When catching up, proves multiple blocks concurrently using separate bb processes.
// Each block gets its own worker with limited threads to avoid CPU contention.

async function cmdWatchParallel(
  nodeBase: string,
  proofDir: string,
  intervalSec: number,
  workers: number,
) {
  // Optimal: 2 threads per worker gives best throughput (803 blocks/min at 6 workers)
  // More threads per worker has diminishing returns above 6 threads.
  const threadsPerWorker = Math.max(2, Math.floor(12 / workers));
  console.log(`Parallel prover: ${workers} workers, ${threadsPerWorker} threads each`);
  console.log(`Watching ${nodeBase} (interval=${intervalSec}s)\n`);
  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  let lastProvenBlock = 0;

  // Check what's already proven
  try {
    const statusRes = await fetch(nodeUrl(nodeBase, "/proof/zk/latest"));
    if (statusRes.ok) {
      const status = await statusRes.json() as any;
      lastProvenBlock = status.block_number ?? 0;
      console.log(`Resuming from block ${lastProvenBlock}`);
    }
  } catch {}

  while (true) {
    try {
      const latestBlock = await fetchLatestBlock(nodeBase);
      const gap = latestBlock - lastProvenBlock;

      if (gap <= 0) {
        await new Promise((r) => setTimeout(r, intervalSec * 1000));
        continue;
      }

      // Determine how many blocks to prove in parallel
      const batchSize = Math.min(gap, workers);
      const blockNumbers = Array.from(
        { length: batchSize },
        (_, i) => lastProvenBlock + 1 + i,
      );

      console.log(`\n--- Proving blocks ${blockNumbers.join(", ")} in parallel (gap=${gap}) ---`);
      const batchStart = performance.now();

      // Spawn parallel prove processes
      const results = await Promise.allSettled(
        blockNumbers.map(async (blockNum) => {
          const outPath = join(proofDir, `block_${blockNum}.json`);
          await proveBlockNative(nodeBase, blockNum, outPath, threadsPerWorker);
          return { blockNum, outPath };
        }),
      );

      const batchElapsed = ((performance.now() - batchStart) / 1000).toFixed(2);
      let proved = 0;

      for (const result of results) {
        if (result.status === "fulfilled") {
          const { blockNum, outPath } = result.value;
          proved++;
          // Submit to node
          try {
            const proofData = JSON.parse(readFileSync(outPath, "utf-8"));
            await submitProof(nodeBase, proofData);
            console.log(`  Block ${blockNum}: submitted`);
          } catch (e: any) {
            console.warn(`  Block ${blockNum}: submit failed (${e.message})`);
          }
        } else {
          console.error(`  Block prove failed: ${result.reason}`);
        }
      }

      lastProvenBlock += proved;
      const throughput = (proved / parseFloat(batchElapsed) * 60).toFixed(1);
      console.log(`Batch done: ${proved}/${batchSize} blocks in ${batchElapsed}s (${throughput} blocks/min)`);

    } catch (e: any) {
      console.error(`Error: ${e.message}`);
    }

    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
}

/** Prove a single block using native bb CLI with limited threads. */
async function proveBlockNative(
  nodeBase: string,
  blockNumber: number,
  outputPath: string,
  threads: number,
): Promise<void> {
  const block = await fetchBlock(nodeBase, blockNumber);
  const prevStateRoot = await fetchPrevStateRoot(nodeBase, blockNumber);
  const circuit = loadCircuit();
  const noir = new Noir(circuit);

  const witness = await buildSingleBlockWitness(block, prevStateRoot);
  const { witness: solvedWitness } = await noir.execute(witness as any);

  // Write witness to a unique temp file
  const witnessPath = `/tmp/persistia_bb_witness_${blockNumber}.gz`;
  const outDir = `/tmp/persistia_bb_prove_${blockNumber}`;
  writeFileSync(witnessPath, solvedWitness);
  rmSync(outDir, { recursive: true, force: true });

  // Use precomputed VK (saves ~700ms per prove)
  const vkPath = ensureVkCached();
  const vkFlag = vkPath ? ` -k ${vkPath}` : " --write_vk";

  // Prove with limited threads via OMP_NUM_THREADS
  execSync(
    `${BB_PATH} prove -b ${CIRCUIT_PATH} -w ${witnessPath} -o ${outDir}${vkFlag} -t noir-recursive-no-zk`,
    {
      stdio: "pipe",
      env: { ...process.env, OMP_NUM_THREADS: String(threads) },
    },
  );

  const proof = readFileSync(join(outDir, "proof"));
  const piRaw = readFileSync(join(outDir, "public_inputs"));
  const vk = readFileSync(join(outDir, "vk"));

  const publicInputs: string[] = [];
  for (let i = 0; i < piRaw.length; i += 32) {
    const hex = "0x" + Array.from(piRaw.subarray(i, i + 32)).map(b => b.toString(16).padStart(2, "0")).join("");
    publicInputs.push(hex);
  }

  const proofData = {
    proof: Buffer.from(proof).toString("base64"),
    publicInputs,
    vk: Buffer.from(vk).toString("base64"),
    block_number: blockNumber,
    proven_blocks: 1,
    genesis_root: prevStateRoot,
    state_root: witness.new_state_root,
    prover: "noir-ultrahonk-native-parallel",
    timestamp: new Date().toISOString(),
  };

  const dir = resolve(outputPath, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(outputPath, JSON.stringify(proofData, null, 2));

  // Cleanup temp files
  rmSync(witnessPath, { force: true });
  rmSync(outDir, { recursive: true, force: true });
}

// --- Pipelined Watch ---
// Overlaps witness solving (JS, ~200ms) with native bb proving (~3.5s).
// While bb proves block N, we fetch + solve witness for block N+1.

async function cmdWatchPipelined(
  nodeBase: string,
  proofDir: string,
  intervalSec: number,
) {
  if (!nativeBbAvailable()) {
    console.error("Pipelined mode requires native bb CLI. Install with: bbup -v 4.1.2");
    process.exit(1);
  }

  console.log(`Pipelined prover watching ${nodeBase} (interval=${intervalSec}s)`);
  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  const circuit = loadCircuit();
  const noir = new Noir(circuit);

  // Ensure VK is cached before the loop (one-time ~700ms cost)
  ensureVkCached();

  let lastProvenBlock = 0;
  let prevProofPath: string | undefined;

  // Pre-solved witness for the next block (pipelining buffer)
  let prefetchedWitness: { blockNum: number; solved: Uint8Array; block: any; prevStateRoot: string; stateRoot: string } | null = null;

  while (true) {
    try {
      const latestBlock = await fetchLatestBlock(nodeBase);

      if (latestBlock <= lastProvenBlock) {
        // No new blocks — speculatively prefetch if we don't already have one
        await new Promise((r) => setTimeout(r, intervalSec * 1000));
        continue;
      }

      const blockNum = lastProvenBlock + 1;

      // --- Phase 1: Get solved witness (from prefetch or solve now) ---
      let solvedWitness: Uint8Array;
      let block: any;
      let blockPrevStateRoot: string;
      let blockStateRoot: string;

      if (prefetchedWitness && prefetchedWitness.blockNum === blockNum) {
        solvedWitness = prefetchedWitness.solved;
        block = prefetchedWitness.block;
        blockPrevStateRoot = prefetchedWitness.prevStateRoot;
        blockStateRoot = prefetchedWitness.stateRoot;
        console.log(`  Block ${blockNum}: using prefetched witness`);
        prefetchedWitness = null;
      } else {
        block = await fetchBlock(nodeBase, blockNum);
        blockPrevStateRoot = await fetchPrevStateRoot(nodeBase, blockNum);
        const witness = await buildSingleBlockWitness(block, blockPrevStateRoot);
        blockStateRoot = witness.new_state_root;
        const result = await noir.execute(witness as any);
        solvedWitness = result.witness;
      }

      // --- Phase 2: Prove block N AND prefetch witness for block N+1 ---
      const witnessPath = `/tmp/persistia_bb_witness_${blockNum}.gz`;
      const outDir = `/tmp/persistia_bb_prove_${blockNum}`;
      writeFileSync(witnessPath, solvedWitness);
      rmSync(outDir, { recursive: true, force: true });

      const vkPath = ensureVkCached();
      const vkFlag = vkPath ? ` -k ${vkPath}` : " --write_vk";

      const proveStart = performance.now();

      // Start native proving as a child process (non-blocking)
      const provePromise = new Promise<void>((resolve, reject) => {
        exec(
          `${BB_PATH} prove -b ${CIRCUIT_PATH} -w ${witnessPath} -o ${outDir}${vkFlag} -t noir-recursive-no-zk`,
          { env: process.env },
          (err) => { if (err) reject(err); else resolve(); },
        );
      });

      // Simultaneously prefetch next block's witness if available
      const nextBlockNum = blockNum + 1;
      let prefetchPromise: Promise<void> = Promise.resolve();
      if (nextBlockNum <= latestBlock) {
        prefetchPromise = (async () => {
          try {
            const nextBlock = await fetchBlock(nodeBase, nextBlockNum);
            const nextPrevSR = await fetchPrevStateRoot(nodeBase, nextBlockNum);
            const nextWitness = await buildSingleBlockWitness(nextBlock, nextPrevSR);
            const result = await noir.execute(nextWitness as any);
            prefetchedWitness = { blockNum: nextBlockNum, solved: result.witness, block: nextBlock, prevStateRoot: nextPrevSR, stateRoot: nextWitness.new_state_root };
            console.log(`  Block ${nextBlockNum}: witness prefetched`);
          } catch (e: any) {
            // Prefetch failure is non-fatal — we'll solve on demand
            prefetchedWitness = null;
          }
        })();
      }

      // Wait for prove to complete (prefetch runs concurrently)
      await Promise.all([provePromise, prefetchPromise]);
      const proveElapsed = ((performance.now() - proveStart) / 1000).toFixed(2);

      // Read and package proof
      const proof = readFileSync(join(outDir, "proof"));
      const piRaw = readFileSync(join(outDir, "public_inputs"));
      const vk = existsSync(join(outDir, "vk")) ? readFileSync(join(outDir, "vk")) : null;

      const publicInputs: string[] = [];
      for (let i = 0; i < piRaw.length; i += 32) {
        const hex = "0x" + Array.from(piRaw.subarray(i, i + 32)).map(b => b.toString(16).padStart(2, "0")).join("");
        publicInputs.push(hex);
      }

      const outPath = join(proofDir, `block_${blockNum}.json`);
      const proofData = {
        proof: Buffer.from(proof).toString("base64"),
        publicInputs,
        vk: vk ? Buffer.from(vk).toString("base64") : undefined,
        block_number: blockNum,
        proven_blocks: 1,
        genesis_root: blockPrevStateRoot,
        state_root: blockStateRoot,
        prover: "noir-ultrahonk-native-pipelined",
        timestamp: new Date().toISOString(),
      };

      writeFileSync(outPath, JSON.stringify(proofData, null, 2));
      lastProvenBlock = blockNum;
      prevProofPath = outPath;

      console.log(`Block ${blockNum}: proved in ${proveElapsed}s -> ${outPath}`);

      // Submit to node
      try {
        await submitProof(nodeBase, proofData);
        console.log("  Proof submitted to node");
      } catch (e: any) {
        console.warn(`  Submit failed (non-fatal): ${e.message}`);
      }

      // Cleanup temp files
      rmSync(witnessPath, { force: true });
      rmSync(outDir, { recursive: true, force: true });

    } catch (e: any) {
      console.error(`Error: ${e.message}`);
      await new Promise((r) => setTimeout(r, intervalSec * 1000));
    }
  }
}

// --- Persistent bb Msgpack API ---
// Keeps bb processes resident across proves, eliminating ~40ms startup overhead per prove.

const msgpackEncoder = new Encoder({ useRecords: false });
const msgpackDecoder = new Decoder({ useRecords: false });

class PersistentBb {
  private proc: ChildProcess;
  private pendingResolves: Array<{ resolve: (v: any) => void; reject: (e: Error) => void }> = [];
  private buffer = Buffer.alloc(0);
  private readingLength = true;
  private expectedLength = 0;

  constructor(threads: number) {
    this.proc = spawn(BB_PATH, ["msgpack", "run"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, OMP_NUM_THREADS: String(threads) },
    });
    this.proc.stdout!.on("data", (d: Buffer) => this.handleData(d));
    this.proc.on("error", (e) => {
      if (this.pendingResolves.length > 0) this.pendingResolves.shift()!.reject(e);
    });
  }

  private handleData(data: Buffer) {
    this.buffer = Buffer.concat([this.buffer, data]);
    while (true) {
      if (this.readingLength) {
        if (this.buffer.length >= 4) {
          this.expectedLength = this.buffer.readUInt32LE(0);
          this.buffer = this.buffer.subarray(4);
          this.readingLength = false;
        } else break;
      } else {
        if (this.buffer.length >= this.expectedLength) {
          const payload = this.buffer.subarray(0, this.expectedLength);
          this.buffer = this.buffer.subarray(this.expectedLength);
          this.readingLength = true;
          const resp = msgpackDecoder.unpack(payload);
          if (this.pendingResolves.length > 0) this.pendingResolves.shift()!.resolve(resp);
        } else break;
      }
    }
  }

  prove(bytecode: Buffer, vk: Buffer, witness: Buffer): Promise<any> {
    return new Promise((resolve, reject) => {
      this.pendingResolves.push({ resolve, reject });
      const cmd = [["CircuitProve", {
        circuit: { name: "persistia_state_proof", bytecode, verification_key: vk },
        witness,
        settings: {
          ipa_accumulation: false,
          oracle_hash_type: "poseidon2",
          disable_zk: true,
          optimized_solidity_verifier: false,
        },
      }]];
      const packed = msgpackEncoder.pack(cmd);
      const lenBuf = Buffer.alloc(4);
      lenBuf.writeUInt32LE(packed.length, 0);
      this.proc.stdin!.write(lenBuf);
      this.proc.stdin!.write(packed);
    });
  }

  shutdown(): Promise<void> {
    const packed = msgpackEncoder.pack([["Shutdown", {}]]);
    const lenBuf = Buffer.alloc(4);
    lenBuf.writeUInt32LE(packed.length, 0);
    this.proc.stdin!.write(lenBuf);
    this.proc.stdin!.write(packed);
    this.proc.stdin!.end();
    return new Promise((r) => this.proc.on("close", r));
  }
}

async function cmdWatchParallelMsgpack(
  nodeBase: string,
  proofDir: string,
  intervalSec: number,
  workers: number,
) {
  if (!nativeBbAvailable()) {
    console.error("Msgpack mode requires native bb CLI. Install with: bbup -v 4.1.2");
    process.exit(1);
  }

  const threadsPerWorker = Math.max(2, Math.floor(12 / workers));
  console.log(`Msgpack parallel prover: ${workers} persistent bb workers, ${threadsPerWorker} threads each`);
  console.log(`Watching ${nodeBase} (interval=${intervalSec}s)\n`);
  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  // Load circuit and prepare bytecode/VK
  const circuit = loadCircuit();
  const noir = new Noir(circuit);
  const bytecode = gunzipSync(Buffer.from(circuit.bytecode, "base64"));
  const vkPath = ensureVkCached();
  if (!vkPath) {
    console.error("Failed to compute VK");
    process.exit(1);
  }
  const vkBytes = readFileSync(vkPath);

  // Spawn persistent bb workers
  const bbs = Array.from({ length: workers }, () => new PersistentBb(threadsPerWorker));

  // Warmup each worker
  console.log("Warming up workers...");
  const testWitness = await (async () => {
    const { buildTestWitness } = await import("./witness.js");
    const tw = await buildTestWitness();
    const { witness: sw } = await noir.execute(tw as any);
    let decompressed: Buffer;
    try { decompressed = gunzipSync(Buffer.from(sw)); } catch { decompressed = Buffer.from(sw); }
    return decompressed;
  })();

  await Promise.all(bbs.map((bb) => bb.prove(bytecode, vkBytes, testWitness)));
  console.log("Workers ready.\n");

  let lastProvenBlock = 0;

  // Check what's already proven
  try {
    const statusRes = await fetch(nodeUrl(nodeBase, "/proof/zk/latest"));
    if (statusRes.ok) {
      const status = (await statusRes.json()) as any;
      lastProvenBlock = status.block_number ?? 0;
      console.log(`Resuming from block ${lastProvenBlock}`);
    }
  } catch {}

  while (true) {
    try {
      const latestBlock = await fetchLatestBlock(nodeBase);
      const gap = latestBlock - lastProvenBlock;

      if (gap <= 0) {
        await new Promise((r) => setTimeout(r, intervalSec * 1000));
        continue;
      }

      const batchSize = Math.min(gap, workers);
      const blockNumbers = Array.from({ length: batchSize }, (_, i) => lastProvenBlock + 1 + i);
      console.log(`\n--- Proving blocks ${blockNumbers.join(", ")} via msgpack (gap=${gap}) ---`);
      const batchStart = performance.now();

      // Solve witnesses in parallel
      const witnessPromises = blockNumbers.map(async (blockNum) => {
        const block = await fetchBlock(nodeBase, blockNum);
        const prevSR = await fetchPrevStateRoot(nodeBase, blockNum);
        const witness = await buildSingleBlockWitness(block, prevSR);
        const stateRoot = witness.new_state_root;
        const { witness: sw } = await noir.execute(witness as any);
        let decompressed: Buffer;
        try { decompressed = gunzipSync(Buffer.from(sw)); } catch { decompressed = Buffer.from(sw); }
        return { blockNum, block, witness: decompressed, prevSR, stateRoot };
      });

      const witnessResults = await Promise.all(witnessPromises);

      // Prove via persistent bb workers
      const proveResults = await Promise.allSettled(
        witnessResults.map(({ blockNum, block, witness, prevSR, stateRoot }, i) =>
          (async () => {
            const resp = await bbs[i % bbs.length].prove(bytecode, vkBytes, witness);
            const [tag, data] = resp;
            if (tag === "ErrorResponse") throw new Error(data.message);

            // Extract proof and public inputs from msgpack response
            const publicInputs: string[] = data.public_inputs.map((pi: Buffer) =>
              "0x" + Array.from(pi).map((b: number) => b.toString(16).padStart(2, "0")).join(""),
            );
            const proofFields = data.proof.map((f: Buffer) =>
              "0x" + Array.from(f).map((b: number) => b.toString(16).padStart(2, "0")).join(""),
            );
            // Convert proof fields back to concatenated bytes
            const proofBytes = Buffer.alloc(proofFields.length * 32);
            for (let j = 0; j < proofFields.length; j++) {
              const val = BigInt(proofFields[j]);
              const hex = val.toString(16).padStart(64, "0");
              for (let k = 0; k < 32; k++) {
                proofBytes[j * 32 + k] = parseInt(hex.substring(k * 2, k * 2 + 2), 16);
              }
            }

            const outPath = join(proofDir, `block_${blockNum}.json`);
            const proofData = {
              proof: proofBytes.toString("base64"),
              publicInputs,
              block_number: blockNum,
              proven_blocks: 1,
              genesis_root: prevSR,
              state_root: stateRoot,
              prover: "noir-ultrahonk-native-msgpack",
              timestamp: new Date().toISOString(),
            };

            const dir = resolve(outPath, "..");
            if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
            writeFileSync(outPath, JSON.stringify(proofData, null, 2));
            return { blockNum, outPath };
          })(),
        ),
      );

      const batchElapsed = ((performance.now() - batchStart) / 1000).toFixed(2);
      let proved = 0;

      for (const result of proveResults) {
        if (result.status === "fulfilled") {
          const { blockNum, outPath } = result.value;
          proved++;
          try {
            const proofData = JSON.parse(readFileSync(outPath, "utf-8"));
            await submitProof(nodeBase, proofData);
            console.log(`  Block ${blockNum}: submitted`);
          } catch (e: any) {
            console.warn(`  Block ${blockNum}: submit failed (${e.message})`);
          }
        } else {
          console.error(`  Block prove failed: ${result.reason}`);
        }
      }

      lastProvenBlock += proved;
      const throughput = (proved / parseFloat(batchElapsed) * 60).toFixed(1);
      console.log(`Batch done: ${proved}/${batchSize} blocks in ${batchElapsed}s (${throughput} blocks/min)`);
    } catch (e: any) {
      console.error(`Error: ${e.message}`);
    }

    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
}

// --- bb Version Manager ---

function cmdBbVersion() {
  try {
    const current = execSync(`${BB_PATH} --version`, { encoding: "utf-8" }).trim();
    console.log(`bb version: ${current}`);
    console.log(`bb path:    ${BB_PATH}`);

    // Check latest available
    try {
      const latest = execSync(
        `npm view @aztec/bb.js version 2>/dev/null`,
        { encoding: "utf-8" },
      ).trim();
      console.log(`Latest bb.js: ${latest}`);
      if (latest !== current) {
        console.log(`\nUpdate available! Run:`);
        console.log(`  ~/.bb/bbup -v ${latest}`);
        console.log(`  npm install @aztec/bb.js@${latest}`);
      } else {
        console.log("Up to date.");
      }
    } catch {
      console.log("Could not check latest version.");
    }
  } catch {
    console.log("bb CLI not found. Install with:");
    console.log("  curl -L https://raw.githubusercontent.com/AztecProtocol/aztec-packages/master/barretenberg/bbup/install | bash");
    console.log("  bbup -v 4.1.2");
  }
}

function cmdBbUpgrade() {
  try {
    const latest = execSync(`npm view @aztec/bb.js version 2>/dev/null`, { encoding: "utf-8" }).trim();
    const current = execSync(`${BB_PATH} --version`, { encoding: "utf-8" }).trim();

    if (latest === current) {
      console.log(`Already on latest: ${current}`);
      return;
    }

    console.log(`Upgrading bb: ${current} -> ${latest}`);
    execSync(`${BB_PATH}up -v ${latest}`, { stdio: "inherit" });
    execSync(`npm install @aztec/bb.js@${latest}`, { stdio: "inherit", cwd: resolve(import.meta.dirname ?? ".", "..") });
    console.log("Done. Re-compile circuit if needed: cd contracts/zk-noir && nargo compile");
  } catch (e: any) {
    console.error(`Upgrade failed: ${e.message}`);
  }
}

// --- CLI ---

const args = process.argv.slice(2);
const command = args[0];

function getArg(name: string, defaultVal?: string): string {
  const idx = args.indexOf(`--${name}`);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  if (defaultVal !== undefined) return defaultVal;
  throw new Error(`Missing required argument: --${name}`);
}

const useNative = args.includes("--native");
const verifierTarget: VerifierTarget = args.includes("--evm") ? "evm-no-zk" : "noir-recursive-no-zk";

switch (command) {
  case "execute":
    cmdExecute(getArg("node", "http://localhost:8787"), parseInt(getArg("block")));
    break;
  case "prove":
    cmdProve(
      getArg("node", "http://localhost:8787"),
      parseInt(getArg("block")),
      getArg("output", "proof.json"),
      args.includes("--prev-proof") ? getArg("prev-proof") : undefined,
      useNative,
      verifierTarget,
    );
    break;
  case "verify":
    cmdVerify(getArg("proof"), useNative);
    break;
  case "watch":
    cmdWatch(
      getArg("node", "http://localhost:8787"),
      getArg("proof-dir", "./proofs"),
      parseInt(getArg("interval", "10")),
      useNative,
      args.includes("--start") ? parseInt(getArg("start")) : undefined,
    );
    break;
  case "bench":
    cmdBench(getArg("node", "http://localhost:8787"), parseInt(getArg("block")));
    break;
  case "watch-parallel":
    cmdWatchParallel(
      getArg("node", "http://localhost:8787"),
      getArg("proof-dir", "./proofs"),
      parseInt(getArg("interval", "5")),
      parseInt(getArg("workers", "6")),
    );
    break;
  case "watch-pipelined":
    cmdWatchPipelined(
      getArg("node", "http://localhost:8787"),
      getArg("proof-dir", "./proofs"),
      parseInt(getArg("interval", "5")),
    );
    break;
  case "watch-parallel-msgpack":
    cmdWatchParallelMsgpack(
      getArg("node", "http://localhost:8787"),
      getArg("proof-dir", "./proofs"),
      parseInt(getArg("interval", "5")),
      parseInt(getArg("workers", "6")),
    );
    break;
  case "bb-version":
    cmdBbVersion();
    break;
  case "bb-upgrade":
    cmdBbUpgrade();
    break;
  default:
    console.log(`Persistia Noir Prover

Usage:
  tsx prover.ts execute  --node <url> --block <n>      Execute without proof (test)
  tsx prover.ts prove    --node <url> --block <n>      Generate proof
  tsx prover.ts verify   --proof <path>                Verify a proof file
  tsx prover.ts watch    --node <url>                  Watch and prove continuously
  tsx prover.ts watch-parallel --node <url>            Parallel proving (catch-up mode)
  tsx prover.ts watch-pipelined --node <url>           Pipelined proving (overlaps witness + prove)
  tsx prover.ts watch-parallel-msgpack --node <url>    Persistent bb workers (6-9% faster parallel)
  tsx prover.ts bench    --node <url> --block <n>      Benchmark proving times
  tsx prover.ts bb-version                             Check bb CLI version
  tsx prover.ts bb-upgrade                             Upgrade bb CLI + bb.js

Options:
  --native     Use native bb CLI instead of WASM (faster, requires bb installed)
  --evm        Generate EVM-optimized proofs (7.8KB, 1.7M gas vs 14KB, 3.3M gas)
  --workers N  Number of parallel workers (default: 6, for watch-parallel)
  --interval N Poll interval in seconds (default: 5 for parallel, 10 for sequential)`);
}
