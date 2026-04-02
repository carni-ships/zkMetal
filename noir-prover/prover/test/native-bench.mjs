#!/usr/bin/env node
// Benchmark: Native bb CLI vs WASM proving
import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync, writeFileSync } from "fs";
import { execSync } from "child_process";
import { createHash } from "crypto";

const BB = process.env.HOME + "/.bb/bb";
const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;

const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
const noir = new Noir(circuit);
const api = await Barretenberg.new({ threads: 4 });

// Generate test witness (1 validator, 1 mutation)
const blockMsg = Buffer.from("block:1");
const msgHash = new Uint8Array(createHash("sha256").update(blockMsg).digest());

const privateKey = new Uint8Array(32);
privateKey[31] = 1;
const { publicKey } = await api.schnorrComputePublicKey({ privateKey });
const { s, e } = await api.schnorrConstructSignature({ message: msgHash, privateKey });

const dummyKey = new Uint8Array(32);
dummyKey[31] = 0xff;
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

console.log("=== Native bb vs WASM Benchmark ===\n");

// Solve witness
console.log("Solving witness...");
const t0 = performance.now();
const { witness: solvedWitness } = await noir.execute(witness);
console.log(`Witness solved in ${((performance.now() - t0) / 1000).toFixed(2)}s`);

// Write witness for bb CLI
writeFileSync("/tmp/bb_witness.gz", solvedWitness);

// WASM prove
console.log("\n--- WASM Prove ---");
const backend = new UltraHonkBackend(circuit.bytecode, api);
const t1 = performance.now();
const wasmProof = await backend.generateProof(solvedWitness);
const wasmTime = (performance.now() - t1) / 1000;
console.log(`Time: ${wasmTime.toFixed(2)}s`);
console.log(`Proof: ${wasmProof.proof.length} bytes`);

// Native bb prove (bb v4.x outputs to a directory: proof, public_inputs, vk, vk_hash)
console.log("\n--- Native bb Prove ---");
execSync("rm -rf /tmp/bb_native_out", { stdio: "pipe" });
const t2 = performance.now();
try {
  execSync(`${BB} prove -b ${CIRCUIT_PATH} -w /tmp/bb_witness.gz -o /tmp/bb_native_out -t noir-recursive-no-zk --write_vk`, { stdio: "pipe" });
  const nativeTime = (performance.now() - t2) / 1000;
  const nativeProof = readFileSync("/tmp/bb_native_out/proof");
  console.log(`Time: ${nativeTime.toFixed(2)}s`);
  console.log(`Proof: ${nativeProof.length} bytes`);

  // Verify natively
  const t3 = performance.now();
  execSync(`${BB} verify -p /tmp/bb_native_out/proof -k /tmp/bb_native_out/vk -i /tmp/bb_native_out/public_inputs -t noir-recursive-no-zk`, { stdio: "pipe" });
  const verifyTime = (performance.now() - t3) / 1000;
  console.log(`Verify: ${verifyTime.toFixed(2)}s`);

  console.log(`\n=== Summary ===`);
  console.log(`WASM prove:   ${wasmTime.toFixed(2)}s`);
  console.log(`Native prove: ${nativeTime.toFixed(2)}s`);
  console.log(`Speedup:      ${(wasmTime / nativeTime).toFixed(1)}x`);
} catch (err) {
  console.log("Native prove failed:", err.stderr?.toString() || err.message);
}

await api.destroy();
