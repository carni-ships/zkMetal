#!/usr/bin/env node
// Integration test: Pure JS Schnorr → Noir circuit → Prove → Verify
//
// Tests the full pipeline without any bb.js Schnorr dependency:
// 1. Generate Grumpkin keys and Schnorr signatures using pure JS (@noble/curves)
// 2. Compute Poseidon2 Merkle root using bb.js
// 3. Build circuit witness
// 4. Execute Noir circuit (witness solving)
// 5. Generate UltraHonk proof
// 6. Verify proof

import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync } from "fs";
import { createHash } from "crypto";

// Import pure JS Grumpkin Schnorr (same code that runs in CF Workers)
import { weierstrass } from "@noble/curves/abstract/weierstrass";
import { Field } from "@noble/curves/abstract/modular";
import { blake2s } from "@noble/hashes/blake2.js";
import { concatBytes, randomBytes } from "@noble/hashes/utils.js";
import { hmac } from "@noble/hashes/hmac.js";
import { sha256 as nobleSha256 } from "@noble/hashes/sha2.js";

// ─── Grumpkin/Schnorr (copied from src/grumpkin-schnorr.ts for standalone test) ───

const p = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const n = 21888242871839275222246405745257275088696311157297823662689037894645226208583n;
const Fp = Field(p);

const Grumpkin = weierstrass({
  a: 0n, b: Fp.neg(17n), Fp, n,
  Gx: 1n, Gy: BigInt("0x02cf135e7506a45d632d270d45f1181294833fc48d823f272c"),
  h: 1n, hash: nobleSha256,
  hmac: (key, ...msgs) => hmac(nobleSha256, key, concatBytes(...msgs)),
  randomBytes,
});

const GENS = [
  Grumpkin.ProjectivePoint.fromAffine({
    x: BigInt("0x083e7911d835097629f0067531fc15cafd79a89beecb39903f69572c636f4a5a"),
    y: BigInt("0x1a7f5efaad7f315c25a918f30cc8d7333fccab7ad7c90f14de81bcc528f9935d"),
  }),
  Grumpkin.ProjectivePoint.fromAffine({
    x: BigInt("0x054aa86a73cb8a34525e5bbed6e43ba1198e860f5f3950268f71df4591bde402"),
    y: BigInt("0x209dcfbf2cfb57f9f6046f44d71ac6faf87254afc7407c04eb621a6287cac126"),
  }),
  Grumpkin.ProjectivePoint.fromAffine({
    x: BigInt("0x1c44f2a5207c81c28a8321a5815ce8b1311024bbed131819bbdaf5a2ada84748"),
    y: BigInt("0x03aaee36e6422a1d0191632ac6599ae9eba5ac2c17a8c920aa3caf8b89c5f8a8"),
  }),
];
const H_LEN_X = BigInt("0x2df8b940e5890e4e1377e05373fae69a1d754f6935e6a780b666947431f2cdcd");
const H_LEN_Y2 = Fp.add(Fp.mul(Fp.mul(H_LEN_X, H_LEN_X), H_LEN_X), Fp.neg(17n));
const H_LEN = Grumpkin.ProjectivePoint.fromAffine({ x: H_LEN_X, y: Fp.sqrt(H_LEN_Y2) });

function fieldToBytes32(f) {
  const hex = f.toString(16).padStart(64, "0");
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  return bytes;
}

function bytesToField(bytes) {
  let val = 0n;
  for (const b of bytes) val = (val << 8n) | BigInt(b);
  return val;
}

function pedersenHash(inputs) {
  let result = H_LEN.multiply(BigInt(inputs.length));
  for (let i = 0; i < inputs.length; i++) {
    if (inputs[i] !== 0n) result = result.add(GENS[i].multiply(inputs[i]));
  }
  return result.toAffine().x;
}

function schnorrSign(privKeyBigInt, message) {
  const G = Grumpkin.ProjectivePoint.BASE;
  const P = G.multiply(privKeyBigInt).toAffine();
  const kBytes = randomBytes(32);
  let k = bytesToField(kBytes) % n;
  if (k === 0n) k = 1n;
  const R = G.multiply(k).toAffine();
  const compressed = pedersenHash([R.x, P.x, P.y]);
  const eInput = concatBytes(fieldToBytes32(compressed), message);
  const eBytes = blake2s(eInput);
  let e = bytesToField(eBytes) % n;
  let s = (k - privKeyBigInt * e) % n;
  if (s < 0n) s += n;
  return { s: fieldToBytes32(s), e: eBytes, pubkey: P };
}

// ─── Test ───

const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;

console.log("=== Persistia Integration Test: Pure JS Schnorr → Noir Circuit ===\n");

// 1. Load circuit
console.log("1. Loading compiled circuit...");
const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));

// 2. Generate Schnorr signatures (pure JS, no bb.js)
console.log("2. Generating Schnorr signatures (pure JS @noble/curves)...");
const blockNumber = 42;
const numValidators = 2;
const numMutations = 3;

const blockMsg = Buffer.from(`block:${blockNumber}`);
const msgHash = new Uint8Array(createHash("sha256").update(blockMsg).digest());

const validators = [];
for (let i = 0; i < numValidators; i++) {
  const privKey = BigInt(i + 1);
  const sig = schnorrSign(privKey, msgHash);
  validators.push({
    pubkey_x: "0x" + sig.pubkey.x.toString(16).padStart(64, "0"),
    pubkey_y: "0x" + sig.pubkey.y.toString(16).padStart(64, "0"),
    signature: [...Array.from(sig.s), ...Array.from(sig.e)],
    msg: Array.from(msgHash),
    enabled: true,
  });
  console.log(`   Validator ${i}: signed ✓`);
}

// Fill disabled slots with valid dummy sigs (using bb.js for now)
const api = await Barretenberg.new({ threads: 4 });
const dummyKey = new Uint8Array(32);
dummyKey[31] = 0xff;
const dummyMsg = new Uint8Array(32);
const { publicKey: dPk } = await api.schnorrComputePublicKey({ privateKey: dummyKey });
const { s: dS, e: dE } = await api.schnorrConstructSignature({ message: dummyMsg, privateKey: dummyKey });

while (validators.length < 4) {
  validators.push({
    pubkey_x: "0x" + Array.from(dPk.x).map(b => b.toString(16).padStart(2, "0")).join(""),
    pubkey_y: "0x" + Array.from(dPk.y).map(b => b.toString(16).padStart(2, "0")).join(""),
    signature: [...Array.from(dS), ...Array.from(dE)],
    msg: Array.from(dummyMsg),
    enabled: false,
  });
}

// 3. Compute Poseidon2 Merkle root (using bb.js)
console.log("3. Computing Poseidon2 Merkle root...");

async function poseidon2Hash(inputs) {
  const fieldInputs = inputs.map(v => {
    const nn = BigInt(v);
    const hex = nn.toString(16).padStart(64, "0");
    const bytes = new Uint8Array(32);
    for (let i = 0; i < 32; i++) bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
    return bytes;
  });
  const { hash } = await api.poseidon2Hash({ inputs: fieldInputs });
  return "0x" + Array.from(hash).map(b => b.toString(16).padStart(2, "0")).join("");
}

const mutations = [];
for (let i = 0; i < numMutations; i++) {
  mutations.push({ key: `${i + 1}`, new_value: `${(i + 1) * 100}`, is_delete: false, enabled: true });
}

// Build leaf hashes
const leafHashes = [];
for (const m of mutations) {
  leafHashes.push(await poseidon2Hash(["1", m.key, m.new_value]));
}
// Pad to power of 2
let p2 = 1;
while (p2 < leafHashes.length) p2 *= 2;
const last = leafHashes[leafHashes.length - 1];
while (leafHashes.length < p2) leafHashes.push(last);

// Binary reduction
let sz = p2;
while (sz > 1) {
  const half = sz / 2;
  for (let j = 0; j < half; j++) {
    leafHashes[j] = await poseidon2Hash(["2", leafHashes[j * 2], leafHashes[j * 2 + 1]]);
  }
  sz = half;
}
const stateRoot = leafHashes[0];
console.log(`   State root: ${stateRoot.substring(0, 18)}...`);

// 4. Build witness
console.log("4. Building circuit witness...");
const mutsPadded = [...mutations];
while (mutsPadded.length < 32) mutsPadded.push({ key: "0", new_value: "0", is_delete: false, enabled: false });

const witness = {
  mutations: mutsPadded,
  mutation_count: numMutations,
  signatures: validators,
  sig_count: numValidators,
  recursive: false,
  prev_proven_blocks: 0,
  prev_genesis_root: "0",
  prev_proof: new Array(449).fill("0"),
  prev_vk: new Array(115).fill("0"),
  prev_key_hash: "0",
  prev_public_inputs: new Array(8).fill("0"),
  prev_state_root: "0xaa",
  new_state_root: stateRoot,
  block_number: blockNumber,
  active_nodes: numValidators,
};

// 5. Execute
console.log("5. Executing circuit (witness solving)...");
const noir = new Noir(circuit);
const execStart = performance.now();
const { witness: solvedWitness, returnValue } = await noir.execute(witness);
const execTime = performance.now() - execStart;
console.log(`   Execution OK in ${(execTime / 1000).toFixed(2)}s`);
console.log(`   Output: ${JSON.stringify(returnValue)}`);

// 6. Prove
console.log("6. Generating UltraHonk proof...");
const backend = new UltraHonkBackend(circuit.bytecode, api);
const proveStart = performance.now();
const proof = await backend.generateProof(solvedWitness);
const proveTime = performance.now() - proveStart;
console.log(`   Proof generated in ${(proveTime / 1000).toFixed(2)}s`);
console.log(`   Proof size: ${proof.proof.length} bytes`);

// 7. Verify
console.log("7. Verifying proof...");
const verifyStart = performance.now();
const valid = await backend.verifyProof(proof);
const verifyTime = performance.now() - verifyStart;
console.log(`   Proof valid: ${valid}`);
console.log(`   Verified in ${(verifyTime / 1000).toFixed(2)}s`);

// Summary
console.log("\n=== Results ===");
console.log(`  Validators: ${numValidators} (pure JS Schnorr)`);
console.log(`  Mutations:  ${numMutations}`);
console.log(`  Block:      ${blockNumber}`);
console.log(`  Execute:    ${(execTime / 1000).toFixed(2)}s`);
console.log(`  Prove:      ${(proveTime / 1000).toFixed(2)}s`);
console.log(`  Verify:     ${(verifyTime / 1000).toFixed(2)}s`);
console.log(`  Proof size: ${proof.proof.length} bytes`);
console.log(`  Status:     ${valid ? "✓ ALL PASS" : "✗ FAILED"}`);

await api.destroy();
process.exit(valid ? 0 : 1);
