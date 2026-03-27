// Watcher — generic watch loop for continuous proving.
//
// Polls a DataSource for new blocks, builds witnesses via WitnessBuilder,
// generates proofs via ProverEngine, and optionally submits via ProofSink.
// Works with any Noir circuit — no application-specific logic.

import { Noir } from "@noir-lang/noir_js";
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from "fs";
import { join, resolve } from "path";
import { gunzipSync } from "zlib";
import { ProverEngine, extractInnerProof, type VerifierTarget } from "./engine.js";
import { PersistentBb } from "./persistent-bb.js";
import type {
  DataSource,
  WitnessBuilder,
  ProofSink,
  ProofOutput,
  WatchOptions,
  RecursiveProofInputs,
} from "./types.js";

/** Sequential watcher — proves one block at a time with optional IVC chaining. */
export async function watchSequential<B>(
  engine: ProverEngine,
  dataSource: DataSource<B>,
  witnessBuilder: WitnessBuilder<B>,
  sink?: ProofSink,
  opts: WatchOptions = {},
): Promise<never> {
  const intervalSec = opts.intervalSec ?? 10;
  const proofDir = opts.proofDir ?? "./proofs";
  const useNative = opts.useNative ?? false;
  const recursive = opts.recursive ?? false;

  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  let lastProvenBlock = opts.startBlock ?? await dataSource.fetchLatestBlockNumber();
  let prevProofPath: string | undefined;

  console.log(`[zkMetal] Watching (interval=${intervalSec}s, starting after block ${lastProvenBlock})`);

  while (true) {
    try {
      const latestBlock = await dataSource.fetchLatestBlockNumber();

      while (latestBlock > lastProvenBlock) {
        const blockNum = lastProvenBlock + 1;
        const outPath = join(proofDir, `block_${blockNum}.json`);

        try {
          const block = await dataSource.fetchBlock(blockNum);

          // Build recursive inputs from previous proof if chaining
          let recursiveOpts: RecursiveProofInputs | undefined;
          if (recursive && prevProofPath && existsSync(prevProofPath)) {
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
          }

          const witness = await witnessBuilder.buildWitness(block, blockNum, recursiveOpts);
          const { witness: solvedWitness } = await engine.execute(witness);

          const start = performance.now();
          let proof: { proof: Uint8Array; publicInputs: string[] };
          let vkBytes: Uint8Array | undefined;

          if (useNative && engine.nativeBbAvailable()) {
            const result = engine.nativeProve(solvedWitness, "noir-recursive-no-zk", `/tmp/zkmetal_bb_${blockNum}`);
            proof = result;
            vkBytes = result.vk;
          } else {
            proof = await engine.prove(solvedWitness);
          }

          const elapsed = ((performance.now() - start) / 1000).toFixed(2);

          // Generate recursive artifacts for chaining
          const artifacts = await engine.generateRecursiveArtifacts(proof);

          const proofOutput: ProofOutput = {
            proof: Buffer.from(proof.proof).toString("base64"),
            publicInputs: proof.publicInputs,
            vkAsFields: artifacts.vkAsFields,
            vkHash: artifacts.vkHash,
            vk: vkBytes ? Buffer.from(vkBytes).toString("base64") : undefined,
            blockNumber: blockNum,
            provenBlocks: recursiveOpts ? recursiveOpts.prevProvenBlocks + 1 : 1,
            prover: useNative ? "zkmetal-native" : "zkmetal-wasm",
            timestamp: new Date().toISOString(),
          };

          const dir = resolve(outPath, "..");
          if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
          writeFileSync(outPath, JSON.stringify(proofOutput, null, 2));
          prevProofPath = outPath;

          console.log(`[zkMetal] Block ${blockNum}: proved in ${elapsed}s (${proofOutput.provenBlocks} in chain)`);

          if (sink) {
            try {
              await sink.submitProof(proofOutput);
              console.log(`[zkMetal] Block ${blockNum}: submitted`);
            } catch (e: any) {
              console.warn(`[zkMetal] Block ${blockNum}: submit failed (${e.message})`);
            }
          }
        } catch (e: any) {
          if (e.message?.includes("404") || e.message?.includes("not found") || e.message?.includes("not committed")) {
            console.log(`[zkMetal] Block ${blockNum}: not available, skipping`);
          } else {
            throw e;
          }
        }

        lastProvenBlock = blockNum;
      }
    } catch (e: any) {
      console.error(`[zkMetal] Error: ${e.message}`);
    }

    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
}

/** Pipelined watcher — overlaps witness solving with native proving. */
export async function watchPipelined<B>(
  engine: ProverEngine,
  dataSource: DataSource<B>,
  witnessBuilder: WitnessBuilder<B>,
  sink?: ProofSink,
  opts: WatchOptions = {},
): Promise<never> {
  const intervalSec = opts.intervalSec ?? 5;
  const proofDir = opts.proofDir ?? "./proofs";

  if (!engine.nativeBbAvailable()) {
    throw new Error("Pipelined mode requires native bb CLI.");
  }

  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });
  engine.ensureVkCached();

  const noir = new Noir(engine.getCircuit());
  let lastProvenBlock = opts.startBlock ?? 0;
  let prefetchedWitness: { blockNum: number; solved: Uint8Array } | null = null;

  console.log(`[zkMetal] Pipelined watcher (interval=${intervalSec}s)`);

  while (true) {
    try {
      const latestBlock = await dataSource.fetchLatestBlockNumber();
      if (latestBlock <= lastProvenBlock) {
        await new Promise((r) => setTimeout(r, intervalSec * 1000));
        continue;
      }

      const blockNum = lastProvenBlock + 1;

      // Phase 1: Get solved witness
      let solvedWitness: Uint8Array;
      if (prefetchedWitness && prefetchedWitness.blockNum === blockNum) {
        solvedWitness = prefetchedWitness.solved;
        prefetchedWitness = null;
      } else {
        const block = await dataSource.fetchBlock(blockNum);
        const witness = await witnessBuilder.buildWitness(block, blockNum);
        const result = await noir.execute(witness as any);
        solvedWitness = result.witness;
      }

      // Phase 2: Prove + prefetch next witness concurrently
      const provePromise = engine.nativeProveAsync(
        solvedWitness,
        "noir-recursive-no-zk",
        `/tmp/zkmetal_bb_${blockNum}`,
      );

      const nextBlockNum = blockNum + 1;
      let prefetchPromise: Promise<void> = Promise.resolve();
      if (nextBlockNum <= latestBlock) {
        prefetchPromise = (async () => {
          try {
            const nextBlock = await dataSource.fetchBlock(nextBlockNum);
            const nextWitness = await witnessBuilder.buildWitness(nextBlock, nextBlockNum);
            const result = await noir.execute(nextWitness as any);
            prefetchedWitness = { blockNum: nextBlockNum, solved: result.witness };
          } catch {
            prefetchedWitness = null;
          }
        })();
      }

      const [proveResult] = await Promise.all([provePromise, prefetchPromise]);

      const outPath = join(proofDir, `block_${blockNum}.json`);
      const proofOutput: ProofOutput = {
        proof: Buffer.from(proveResult.proof).toString("base64"),
        publicInputs: proveResult.publicInputs,
        vk: proveResult.vk ? Buffer.from(proveResult.vk).toString("base64") : undefined,
        blockNumber: blockNum,
        provenBlocks: 1,
        prover: "zkmetal-native-pipelined",
        timestamp: new Date().toISOString(),
      };

      writeFileSync(outPath, JSON.stringify(proofOutput, null, 2));
      lastProvenBlock = blockNum;

      console.log(`[zkMetal] Block ${blockNum}: proved -> ${outPath}`);

      if (sink) {
        try { await sink.submitProof(proofOutput); } catch {}
      }
    } catch (e: any) {
      console.error(`[zkMetal] Error: ${e.message}`);
      await new Promise((r) => setTimeout(r, intervalSec * 1000));
    }
  }
}

/** Parallel watcher using persistent bb msgpack workers. */
export async function watchParallelMsgpack<B>(
  engine: ProverEngine,
  dataSource: DataSource<B>,
  witnessBuilder: WitnessBuilder<B>,
  sink?: ProofSink,
  opts: WatchOptions & { workers?: number } = {},
): Promise<never> {
  const intervalSec = opts.intervalSec ?? 5;
  const proofDir = opts.proofDir ?? "./proofs";
  const workers = opts.workers ?? 6;

  if (!engine.nativeBbAvailable()) {
    throw new Error("Msgpack mode requires native bb CLI.");
  }

  const threadsPerWorker = Math.max(2, Math.floor(12 / workers));
  console.log(`[zkMetal] Parallel msgpack: ${workers} workers, ${threadsPerWorker} threads each`);

  if (!existsSync(proofDir)) mkdirSync(proofDir, { recursive: true });

  const circuit = engine.getCircuit();
  const noir = new Noir(circuit);
  const bytecode = gunzipSync(Buffer.from(circuit.bytecode, "base64"));
  const vkPath = engine.ensureVkCached();
  if (!vkPath) throw new Error("Failed to compute VK");
  const vkBytes = readFileSync(vkPath);

  const config = engine.getConfig();
  const bbs = Array.from(
    { length: workers },
    () => new PersistentBb(threadsPerWorker, config.bbPath),
  );

  let lastProvenBlock = opts.startBlock ?? 0;

  while (true) {
    try {
      const latestBlock = await dataSource.fetchLatestBlockNumber();
      const gap = latestBlock - lastProvenBlock;
      if (gap <= 0) {
        await new Promise((r) => setTimeout(r, intervalSec * 1000));
        continue;
      }

      const batchSize = Math.min(gap, workers);
      const blockNumbers = Array.from({ length: batchSize }, (_, i) => lastProvenBlock + 1 + i);
      const batchStart = performance.now();

      // Solve witnesses in parallel
      const witnessResults = await Promise.all(
        blockNumbers.map(async (blockNum) => {
          const block = await dataSource.fetchBlock(blockNum);
          const witness = await witnessBuilder.buildWitness(block, blockNum);
          const { witness: sw } = await noir.execute(witness as any);
          let decompressed: Buffer;
          try { decompressed = gunzipSync(Buffer.from(sw)); } catch { decompressed = Buffer.from(sw); }
          return { blockNum, witness: decompressed };
        }),
      );

      // Prove via persistent bb workers
      const proveResults = await Promise.allSettled(
        witnessResults.map(({ blockNum, witness }, i) =>
          (async () => {
            const resp = await bbs[i % bbs.length].prove(bytecode, vkBytes, witness);
            const [tag, data] = resp;
            if (tag === "ErrorResponse") throw new Error(data.message);

            const publicInputs: string[] = data.public_inputs.map((pi: Buffer) =>
              "0x" + Array.from(pi).map((b: number) => b.toString(16).padStart(2, "0")).join(""),
            );
            const proofFields = data.proof.map((f: Buffer) =>
              "0x" + Array.from(f).map((b: number) => b.toString(16).padStart(2, "0")).join(""),
            );
            const proofBytes = Buffer.alloc(proofFields.length * 32);
            for (let j = 0; j < proofFields.length; j++) {
              const val = BigInt(proofFields[j]);
              const hex = val.toString(16).padStart(64, "0");
              for (let k = 0; k < 32; k++) {
                proofBytes[j * 32 + k] = parseInt(hex.substring(k * 2, k * 2 + 2), 16);
              }
            }

            const outPath = join(proofDir, `block_${blockNum}.json`);
            const proofOutput: ProofOutput = {
              proof: proofBytes.toString("base64"),
              publicInputs,
              blockNumber: blockNum,
              provenBlocks: 1,
              prover: "zkmetal-native-msgpack",
              timestamp: new Date().toISOString(),
            };

            const dir = resolve(outPath, "..");
            if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
            writeFileSync(outPath, JSON.stringify(proofOutput, null, 2));
            return proofOutput;
          })(),
        ),
      );

      const batchElapsed = ((performance.now() - batchStart) / 1000).toFixed(2);
      let proved = 0;

      for (const result of proveResults) {
        if (result.status === "fulfilled") {
          proved++;
          if (sink) {
            try { await sink.submitProof(result.value); } catch {}
          }
        } else {
          console.error(`[zkMetal] Prove failed: ${result.reason}`);
        }
      }

      lastProvenBlock += proved;
      const throughput = (proved / parseFloat(batchElapsed) * 60).toFixed(1);
      console.log(`[zkMetal] Batch: ${proved}/${batchSize} in ${batchElapsed}s (${throughput} blocks/min)`);
    } catch (e: any) {
      console.error(`[zkMetal] Error: ${e.message}`);
    }

    await new Promise((r) => setTimeout(r, intervalSec * 1000));
  }
}
