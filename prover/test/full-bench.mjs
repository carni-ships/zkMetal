#!/usr/bin/env node
// Full benchmark: sequential vs parallel, with/without VK cache
import { Noir } from "@noir-lang/noir_js";
import { Barretenberg } from "@aztec/bb.js";
import { readFileSync, writeFileSync, rmSync, existsSync, mkdirSync } from "fs";
import { execSync, exec } from "child_process";
import { createHash } from "crypto";

const BB = process.env.HOME + "/.bb/bb";
const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;
const VK_DIR = new URL("../../target/bb_vk", import.meta.url).pathname;
const VK_PATH = VK_DIR + "/vk";

// Ensure VK is precomputed
if (!existsSync(VK_PATH)) {
  console.log("Precomputing VK...");
  mkdirSync(VK_DIR, { recursive: true });
  execSync(`${BB} write_vk -b ${CIRCUIT_PATH} -o ${VK_DIR} -t noir-recursive-no-zk`, { stdio: "pipe" });
}

const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
const noir = new Noir(circuit);
const api = await Barretenberg.new({ threads: 4 });

// Generate test witness
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
  const n = BigInt(v), hex = n.toString(16).padStart(64, "0"), bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  return bytes;
}
const { hash: leafHash } = await api.poseidon2Hash({ inputs: [fieldToBytes(1), fieldToBytes(1), fieldToBytes(100)] });
const stateRoot = bytesToHex(leafHash);
const sigs = [{ pubkey_x: bytesToHex(publicKey.x), pubkey_y: bytesToHex(publicKey.y), signature: [...Array.from(s), ...Array.from(e)], msg: Array.from(msgHash), enabled: true }];
while (sigs.length < 4) sigs.push({ pubkey_x: bytesToHex(dPk.x), pubkey_y: bytesToHex(dPk.y), signature: [...Array.from(dS), ...Array.from(dE)], msg: Array.from(dummyMsg), enabled: false });
const muts = [{ key: "1", new_value: "100", is_delete: false, enabled: true }];
while (muts.length < 32) muts.push({ key: "0", new_value: "0", is_delete: false, enabled: false });
const witness = {
  mutations: muts, mutation_count: 1, signatures: sigs, sig_count: 1,
  recursive: false, prev_proven_blocks: 0, prev_genesis_root: "0",
  prev_proof: new Array(449).fill("0"), prev_vk: new Array(115).fill("0"),
  prev_key_hash: "0", prev_public_inputs: new Array(8).fill("0"),
  prev_state_root: "0xaa", new_state_root: stateRoot, block_number: 1, active_nodes: 1,
};

console.log("Solving witness...");
const { witness: solvedWitness } = await noir.execute(witness);

const N = 3;
for (let i = 0; i < N; i++) writeFileSync(`/tmp/bb_fb_w${i}.gz`, solvedWitness);

function bbProve(id, threads, useVk) {
  return new Promise((resolve, reject) => {
    rmSync(`/tmp/bb_fb_out${id}`, { recursive: true, force: true });
    const vkFlag = useVk ? `-k ${VK_PATH}` : "--write_vk";
    const cmd = `${BB} prove -b ${CIRCUIT_PATH} -w /tmp/bb_fb_w${id}.gz -o /tmp/bb_fb_out${id} ${vkFlag} -t noir-recursive-no-zk`;
    exec(cmd, { env: { ...process.env, OMP_NUM_THREADS: String(threads) } }, (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

console.log(`\n${"═".repeat(60)}`);
console.log("  Persistia Noir Prover — Full Benchmark");
console.log(`${"═".repeat(60)}\n`);

// 1. Sequential, no VK cache
console.log("1. Sequential (12 threads, no VK cache)");
let t = performance.now();
for (let i = 0; i < N; i++) await bbProve(i, 12, false);
const seq_noVk = (performance.now() - t) / 1000;
console.log(`   ${seq_noVk.toFixed(2)}s total, ${(seq_noVk/N).toFixed(2)}s/block, ${(N/seq_noVk*60).toFixed(0)} blocks/min\n`);

// 2. Sequential, with VK cache
console.log("2. Sequential (12 threads, VK cached)");
t = performance.now();
for (let i = 0; i < N; i++) await bbProve(i, 12, true);
const seq_vk = (performance.now() - t) / 1000;
console.log(`   ${seq_vk.toFixed(2)}s total, ${(seq_vk/N).toFixed(2)}s/block, ${(N/seq_vk*60).toFixed(0)} blocks/min\n`);

// 3. Parallel (4 threads each), no VK cache
console.log(`3. Parallel ${N}x (4 threads each, no VK cache)`);
t = performance.now();
await Promise.all(Array.from({length: N}, (_, i) => bbProve(i, 4, false)));
const par_noVk = (performance.now() - t) / 1000;
console.log(`   ${par_noVk.toFixed(2)}s total, ${(N/par_noVk*60).toFixed(0)} blocks/min\n`);

// 4. Parallel (4 threads each), with VK cache
console.log(`4. Parallel ${N}x (4 threads each, VK cached)`);
t = performance.now();
await Promise.all(Array.from({length: N}, (_, i) => bbProve(i, 4, true)));
const par_vk = (performance.now() - t) / 1000;
console.log(`   ${par_vk.toFixed(2)}s total, ${(N/par_vk*60).toFixed(0)} blocks/min\n`);

console.log(`${"═".repeat(60)}`);
console.log("  Summary");
console.log(`${"═".repeat(60)}`);
console.log(`  Sequential, no VK:    ${(N/seq_noVk*60).toFixed(0)} blocks/min  (${(seq_noVk/N).toFixed(2)}s/block)`);
console.log(`  Sequential, VK cache: ${(N/seq_vk*60).toFixed(0)} blocks/min  (${(seq_vk/N).toFixed(2)}s/block)`);
console.log(`  Parallel, no VK:      ${(N/par_noVk*60).toFixed(0)} blocks/min`);
console.log(`  Parallel, VK cache:   ${(N/par_vk*60).toFixed(0)} blocks/min  ★ best`);
console.log(`  VK cache speedup:     ${(seq_noVk/seq_vk).toFixed(2)}x`);
console.log(`  Parallel speedup:     ${(seq_vk/par_vk).toFixed(2)}x`);
console.log(`  Combined speedup:     ${(seq_noVk/par_vk).toFixed(2)}x`);
console.log(`  Max mutations/min:    ${(N/par_vk*60*32).toFixed(0)} (at 32/block)`);
console.log(`${"═".repeat(60)}`);

// Cleanup
for (let i = 0; i < N; i++) {
  rmSync(`/tmp/bb_fb_w${i}.gz`, { force: true });
  rmSync(`/tmp/bb_fb_out${i}`, { recursive: true, force: true });
}
await api.destroy();
