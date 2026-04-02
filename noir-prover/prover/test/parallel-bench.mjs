#!/usr/bin/env node
// Benchmark: Sequential vs Parallel native bb proving
import { Noir } from "@noir-lang/noir_js";
import { Barretenberg } from "@aztec/bb.js";
import { readFileSync, writeFileSync, rmSync } from "fs";
import { execSync, exec } from "child_process";
import { createHash } from "crypto";

const BB = process.env.HOME + "/.bb/bb";
const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;
const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
const noir = new Noir(circuit);
const api = await Barretenberg.new({ threads: 4 });

// Generate test witness (reuse from native-bench)
const blockMsg = Buffer.from("block:1");
const msgHash = new Uint8Array(createHash("sha256").update(blockMsg).digest());
const privateKey = new Uint8Array(32); privateKey[31] = 1;
const { publicKey } = await api.schnorrComputePublicKey({ privateKey });
const { s, e } = await api.schnorrConstructSignature({ message: msgHash, privateKey });
const dummyKey = new Uint8Array(32); dummyKey[31] = 0xff;
const dummyMsg = new Uint8Array(32);
const { publicKey: dPk } = await api.schnorrComputePublicKey({ privateKey: dummyKey });
const { s: dS, e: dE } = await api.schnorrConstructSignature({ message: dummyMsg, privateKey: dummyKey });

function bytesToHex(bytes) { return "0x" + Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join(""); }
function fieldToBytes(v) {
  const n = BigInt(v);
  const hex = n.toString(16).padStart(64, "0");
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  return bytes;
}

const { hash: leafHash } = await api.poseidon2Hash({ inputs: [fieldToBytes(1), fieldToBytes(1), fieldToBytes(100)] });
const stateRoot = bytesToHex(leafHash);

const sigs = [
  { pubkey_x: bytesToHex(publicKey.x), pubkey_y: bytesToHex(publicKey.y), signature: [...Array.from(s), ...Array.from(e)], msg: Array.from(msgHash), enabled: true },
];
while (sigs.length < 4) {
  sigs.push({ pubkey_x: bytesToHex(dPk.x), pubkey_y: bytesToHex(dPk.y), signature: [...Array.from(dS), ...Array.from(dE)], msg: Array.from(dummyMsg), enabled: false });
}
const muts = [{ key: "1", new_value: "100", is_delete: false, enabled: true }];
while (muts.length < 32) muts.push({ key: "0", new_value: "0", is_delete: false, enabled: false });

const witness = {
  mutations: muts, mutation_count: 1,
  signatures: sigs, sig_count: 1,
  recursive: false, prev_proven_blocks: 0, prev_genesis_root: "0",
  prev_proof: new Array(449).fill("0"), prev_vk: new Array(115).fill("0"),
  prev_key_hash: "0", prev_public_inputs: new Array(8).fill("0"),
  prev_state_root: "0xaa", new_state_root: stateRoot, block_number: 1, active_nodes: 1,
};

console.log("Solving witness...");
const { witness: solvedWitness } = await noir.execute(witness);

const WORKERS = 3;
for (let i = 0; i < WORKERS; i++) {
  writeFileSync(`/tmp/bb_par_w${i}.gz`, solvedWitness);
}

function bbProve(id, threads) {
  return new Promise((resolve, reject) => {
    rmSync(`/tmp/bb_par_out${id}`, { recursive: true, force: true });
    const cmd = `${BB} prove -b ${CIRCUIT_PATH} -w /tmp/bb_par_w${id}.gz -o /tmp/bb_par_out${id} --write_vk -t noir-recursive-no-zk`;
    exec(cmd, { env: { ...process.env, OMP_NUM_THREADS: String(threads) } }, (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

console.log(`\n=== Sequential vs Parallel (${WORKERS} workers) ===\n`);

// Sequential (all threads per prove)
console.log("--- Sequential (12 threads, 1 at a time) ---");
const seqStart = performance.now();
for (let i = 0; i < WORKERS; i++) {
  await bbProve(i, 12);
}
const seqTime = (performance.now() - seqStart) / 1000;
console.log(`Total: ${seqTime.toFixed(2)}s (${(seqTime / WORKERS).toFixed(2)}s/block)`);
console.log(`Throughput: ${(WORKERS / seqTime * 60).toFixed(0)} blocks/min`);

// Parallel (limited threads per prove)
const threadsPerWorker = Math.floor(12 / WORKERS);
console.log(`\n--- Parallel (${WORKERS} concurrent, ${threadsPerWorker} threads each) ---`);
const parStart = performance.now();
await Promise.all(
  Array.from({ length: WORKERS }, (_, i) => bbProve(i, threadsPerWorker))
);
const parTime = (performance.now() - parStart) / 1000;
console.log(`Total: ${parTime.toFixed(2)}s (wall clock for ${WORKERS} blocks)`);
console.log(`Throughput: ${(WORKERS / parTime * 60).toFixed(0)} blocks/min`);

console.log(`\n=== Summary ===`);
console.log(`Sequential: ${(WORKERS / seqTime * 60).toFixed(0)} blocks/min`);
console.log(`Parallel:   ${(WORKERS / parTime * 60).toFixed(0)} blocks/min`);
console.log(`Speedup:    ${(seqTime / parTime).toFixed(2)}x`);
console.log(`Mutations/min (parallel): ${(WORKERS / parTime * 60 * 32).toFixed(0)} (at 32 mutations/block)`);

// Cleanup
for (let i = 0; i < WORKERS; i++) {
  rmSync(`/tmp/bb_par_w${i}.gz`, { force: true });
  rmSync(`/tmp/bb_par_out${i}`, { recursive: true, force: true });
}

await api.destroy();
