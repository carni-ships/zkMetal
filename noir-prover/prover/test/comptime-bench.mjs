#!/usr/bin/env node
// Benchmark the compile-time optimized circuit (ENABLE_RECURSIVE=false)
// Expected: ~16x fewer gates (46K vs 769K), significantly faster proving
import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync, writeFileSync, rmSync, existsSync, mkdirSync, statSync } from "fs";
import { execSync, exec } from "child_process";
import { createHash } from "crypto";

const BB = process.env.HOME + "/.bb/bb";
const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;
const VK_PATH = new URL("../../target/bb_vk/vk", import.meta.url).pathname;
const VK_EVM_PATH = new URL("../../target/bb_vk_evm/vk", import.meta.url).pathname;

const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
const noir = new Noir(circuit);
const api = await Barretenberg.new({ threads: 4 });

// Generate witness
const blockMsg = Buffer.from("block:1");
const msgHash = new Uint8Array(createHash("sha256").update(blockMsg).digest());
const pk = new Uint8Array(32); pk[31] = 1;
const { publicKey } = await api.schnorrComputePublicKey({ privateKey: pk });
const { s, e } = await api.schnorrConstructSignature({ message: msgHash, privateKey: pk });
const dk = new Uint8Array(32); dk[31] = 0xff;
const dm = new Uint8Array(32);
const { publicKey: dpk } = await api.schnorrComputePublicKey({ privateKey: dk });
const { s: ds, e: de } = await api.schnorrConstructSignature({ message: dm, privateKey: dk });
function bh(b) { return "0x" + Array.from(b).map(x => x.toString(16).padStart(2, "0")).join(""); }
function fb(v) { const n = BigInt(v), h = n.toString(16).padStart(64, "0"), b = new Uint8Array(32); for (let i = 0; i < 32; i++) b[i] = parseInt(h.substring(i*2, i*2+2), 16); return b; }
const { hash: lh } = await api.poseidon2Hash({ inputs: [fb(1), fb(1), fb(100)] });
const sr = bh(lh);
const sigs = [{ pubkey_x: bh(publicKey.x), pubkey_y: bh(publicKey.y), signature: [...Array.from(s), ...Array.from(e)], msg: Array.from(msgHash), enabled: true }];
while (sigs.length < 4) sigs.push({ pubkey_x: bh(dpk.x), pubkey_y: bh(dpk.y), signature: [...Array.from(ds), ...Array.from(de)], msg: Array.from(dm), enabled: false });
const muts = [{ key: "1", new_value: "100", is_delete: false, enabled: true }];
while (muts.length < 32) muts.push({ key: "0", new_value: "0", is_delete: false, enabled: false });
const w = {
  mutations: muts, mutation_count: 1, signatures: sigs, sig_count: 1,
  prev_proven_blocks: 0, prev_genesis_root: "0",
  prev_proof: new Array(449).fill("0"), prev_vk: new Array(115).fill("0"),
  prev_key_hash: "0", prev_public_inputs: new Array(8).fill("0"),
  prev_state_root: "0xaa", new_state_root: sr, block_number: 1, active_nodes: 1,
};

// Warmup
await noir.execute(w);

console.log(`${"═".repeat(60)}`);
console.log("  Compile-Time Optimized Circuit Benchmark");
console.log("  ENABLE_RECURSIVE=false → 46K gates (was 769K)");
console.log(`${"═".repeat(60)}\n`);

// Witness solve timing
const solveTimes = [];
for (let i = 0; i < 5; i++) {
  const t = performance.now();
  await noir.execute(w);
  solveTimes.push(performance.now() - t);
}
const avgSolve = solveTimes.reduce((a, b) => a + b) / solveTimes.length;
console.log(`Witness solve: avg=${avgSolve.toFixed(0)}ms, min=${Math.min(...solveTimes).toFixed(0)}ms\n`);

const { witness: sw } = await noir.execute(w);
writeFileSync("/tmp/bb_ct_witness.gz", sw);

// Native prove - multiple targets
function bbProve(witnessPath, outDir, vkPath, target, threads) {
  return new Promise((resolve, reject) => {
    rmSync(outDir, { recursive: true, force: true });
    const cmd = `${BB} prove -b ${CIRCUIT_PATH} -w ${witnessPath} -o ${outDir} -k ${vkPath} -t ${target}`;
    exec(cmd, { env: { ...process.env, OMP_NUM_THREADS: String(threads) }, timeout: 30000 }, (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

// Single-threaded and multi-threaded benchmarks
for (const [target, vk] of [["noir-recursive-no-zk", VK_PATH], ["evm-no-zk", VK_EVM_PATH]]) {
  console.log(`--- ${target} ---`);
  for (const threads of [1, 2, 6, 12]) {
    const out = "/tmp/bb_ct_out";
    const times = [];
    for (let i = 0; i < 3; i++) {
      const t = performance.now();
      await bbProve("/tmp/bb_ct_witness.gz", out, vk, target, threads);
      times.push(performance.now() - t);
    }
    const avg = times.reduce((a, b) => a + b) / times.length;
    const proofSize = statSync(`${out}/proof`).size;
    console.log(`  ${String(threads).padStart(2)} threads: ${(avg/1000).toFixed(3)}s  proof=${proofSize}B  ${(60000/avg).toFixed(0)} blocks/min`);
    rmSync(out, { recursive: true, force: true });
  }
  console.log();
}

// Parallel benchmark: 6 workers × 2 threads
console.log("--- Parallel: 6 workers × 2 threads ---");
for (let i = 0; i < 6; i++) writeFileSync(`/tmp/bb_ct_w${i}.gz`, sw);

const t = performance.now();
await Promise.all(Array.from({length: 6}, (_, i) =>
  bbProve(`/tmp/bb_ct_w${i}.gz`, `/tmp/bb_ct_out${i}`, VK_PATH, "noir-recursive-no-zk", 2)
));
const parTime = (performance.now() - t) / 1000;
console.log(`  ${parTime.toFixed(2)}s for 6 blocks = ${(6/parTime*60).toFixed(0)} blocks/min = ${(6/parTime*60*32).toFixed(0)} mutations/min`);

// Sustained: 5 batches of 6
const sustainedStart = performance.now();
for (let batch = 0; batch < 5; batch++) {
  await Promise.all(Array.from({length: 6}, (_, i) =>
    bbProve(`/tmp/bb_ct_w${i}.gz`, `/tmp/bb_ct_out${i}`, VK_PATH, "noir-recursive-no-zk", 2)
  ));
}
const sustainedTime = (performance.now() - sustainedStart) / 1000;
console.log(`  Sustained (30 blocks): ${sustainedTime.toFixed(2)}s = ${(30/sustainedTime*60).toFixed(0)} blocks/min = ${(30/sustainedTime*60*32).toFixed(0)} mutations/min`);

// WASM comparison
console.log("\n--- WASM comparison ---");
const backend = new UltraHonkBackend(circuit.bytecode, api);
const { witness: sw2 } = await noir.execute(w);
const t2 = performance.now();
const wasmProof = await backend.generateProof(sw2);
const wasmTime = (performance.now() - t2) / 1000;
console.log(`  WASM prove: ${wasmTime.toFixed(3)}s, proof=${wasmProof.proof.length}B`);

console.log(`\n${"═".repeat(60)}`);

// Cleanup
for (let i = 0; i < 6; i++) {
  rmSync(`/tmp/bb_ct_w${i}.gz`, { force: true });
  rmSync(`/tmp/bb_ct_out${i}`, { recursive: true, force: true });
}
rmSync("/tmp/bb_ct_witness.gz", { force: true });
await api.destroy();
