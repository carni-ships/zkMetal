#!/usr/bin/env node
// Benchmark: stock bb vs custom M3-optimized bb build
import { Noir } from "@noir-lang/noir_js";
import { Barretenberg } from "@aztec/bb.js";
import { readFileSync, writeFileSync, rmSync, existsSync, mkdirSync, statSync } from "fs";
import { execSync, exec } from "child_process";
import { createHash } from "crypto";

const BB_STOCK = process.env.HOME + "/.bb/bb";
const BB_M3 = "/tmp/bb_build/barretenberg/cpp/build_m3/bin/bb";
const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;
const VK_DIR = new URL("../../target/bb_vk", import.meta.url).pathname;
const VK_PATH = VK_DIR + "/vk";

if (!existsSync(VK_PATH)) {
  mkdirSync(VK_DIR, { recursive: true });
  execSync(`${BB_STOCK} write_vk -b ${CIRCUIT_PATH} -o ${VK_DIR} -t noir-recursive-no-zk`, { stdio: "pipe" });
}

const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
const noir = new Noir(circuit);
const api = await Barretenberg.new({ threads: 4 });

// Generate test witness
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

const { witness: sw } = await noir.execute(w);
writeFileSync("/tmp/bb_bench_witness.gz", sw);

// Also write VK for M3 build
const M3_VK_DIR = "/tmp/bb_m3_vk";
mkdirSync(M3_VK_DIR, { recursive: true });
execSync(`${BB_M3} write_vk -b ${CIRCUIT_PATH} -o ${M3_VK_DIR} -t noir-recursive-no-zk`, { stdio: "pipe" });
const M3_VK_PATH = M3_VK_DIR + "/vk";

console.log(`${"═".repeat(60)}`);
console.log("  Stock bb vs M3-Optimized bb Build");
console.log(`${"═".repeat(60)}\n`);

async function benchBb(bbPath, vkPath, label, threads) {
  function prove() {
    return new Promise((resolve, reject) => {
      rmSync("/tmp/bb_bench_out", { recursive: true, force: true });
      exec(`${bbPath} prove -b ${CIRCUIT_PATH} -w /tmp/bb_bench_witness.gz -o /tmp/bb_bench_out -k ${vkPath} -t noir-recursive-no-zk`, {
        env: { ...process.env, OMP_NUM_THREADS: String(threads) },
        timeout: 30000,
      }, (err) => { if (err) reject(err); else resolve(); });
    });
  }

  // Warmup (2 runs)
  await prove(); await prove();

  const times = [];
  for (let i = 0; i < 7; i++) {
    const t = performance.now();
    await prove();
    times.push(performance.now() - t);
    rmSync("/tmp/bb_bench_out", { recursive: true, force: true });
  }

  // Drop best and worst
  times.sort((a, b) => a - b);
  const trimmed = times.slice(1, -1);
  const avg = trimmed.reduce((a, b) => a + b) / trimmed.length;
  const min = trimmed[0];
  const max = trimmed[trimmed.length - 1];

  console.log(`${label} (${threads} threads):`);
  console.log(`  avg=${avg.toFixed(0)}ms  min=${min.toFixed(0)}ms  max=${max.toFixed(0)}ms  (${(60000/avg).toFixed(0)} blocks/min)\n`);
  return avg;
}

// Also print_bench for detailed comparison
console.log("--- Detailed profile: Stock bb ---");
execSync(`${BB_STOCK} prove -b ${CIRCUIT_PATH} -w /tmp/bb_bench_witness.gz -o /tmp/bb_bench_out -k ${VK_PATH} -t noir-recursive-no-zk --print_bench`, {
  env: { ...process.env, OMP_NUM_THREADS: "2" },
  stdio: "inherit",
});
rmSync("/tmp/bb_bench_out", { recursive: true, force: true });

console.log("\n--- Detailed profile: M3-optimized bb ---");
execSync(`${BB_M3} prove -b ${CIRCUIT_PATH} -w /tmp/bb_bench_witness.gz -o /tmp/bb_bench_out -k ${M3_VK_PATH} -t noir-recursive-no-zk --print_bench`, {
  env: { ...process.env, OMP_NUM_THREADS: "2" },
  stdio: "inherit",
});
rmSync("/tmp/bb_bench_out", { recursive: true, force: true });

console.log("\n--- Throughput benchmark ---");

// Single-threaded
const stock1 = await benchBb(BB_STOCK, VK_PATH, "Stock bb", 1);
const m3_1 = await benchBb(BB_M3, M3_VK_PATH, "M3-optimized bb", 1);

// 2 threads
const stock2 = await benchBb(BB_STOCK, VK_PATH, "Stock bb", 2);
const m3_2 = await benchBb(BB_M3, M3_VK_PATH, "M3-optimized bb", 2);

// 6 threads
const stock6 = await benchBb(BB_STOCK, VK_PATH, "Stock bb", 6);
const m3_6 = await benchBb(BB_M3, M3_VK_PATH, "M3-optimized bb", 6);

// 12 threads
const stock12 = await benchBb(BB_STOCK, VK_PATH, "Stock bb", 12);
const m3_12 = await benchBb(BB_M3, M3_VK_PATH, "M3-optimized bb", 12);

console.log(`${"═".repeat(60)}`);
console.log("  Summary");
console.log(`${"═".repeat(60)}`);
console.log(`  1 thread:  stock=${stock1.toFixed(0)}ms  m3=${m3_1.toFixed(0)}ms  speedup=${(stock1/m3_1).toFixed(3)}x`);
console.log(`  2 threads: stock=${stock2.toFixed(0)}ms  m3=${m3_2.toFixed(0)}ms  speedup=${(stock2/m3_2).toFixed(3)}x`);
console.log(`  6 threads: stock=${stock6.toFixed(0)}ms  m3=${m3_6.toFixed(0)}ms  speedup=${(stock6/m3_6).toFixed(3)}x`);
console.log(`  12 threads: stock=${stock12.toFixed(0)}ms  m3=${m3_12.toFixed(0)}ms  speedup=${(stock12/m3_12).toFixed(3)}x`);
console.log(`${"═".repeat(60)}`);

// Cleanup
rmSync("/tmp/bb_bench_witness.gz", { force: true });
rmSync("/tmp/bb_bench_out", { recursive: true, force: true });
rmSync(M3_VK_DIR, { recursive: true, force: true });
await api.destroy();
