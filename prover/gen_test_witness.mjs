#!/usr/bin/env node
// Generate a Prover.toml with valid Schnorr signatures and Poseidon2 Field mutations.
// Usage: node gen_test_witness.mjs

import { Barretenberg } from "@aztec/bb.js";
import { createHash } from "crypto";
import { writeFileSync } from "fs";

function sha256(data) {
  return createHash("sha256").update(data).digest();
}

function toTomlArray(arr) {
  return "[" + arr.map(b => `"0x${b.toString(16).padStart(2, "0")}"`).join(", ") + "]";
}

function fieldToHex(bytes) {
  return "0x" + Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
}

function fieldToBytes(value) {
  const n = BigInt(value);
  const hex = n.toString(16).padStart(64, "0");
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) {
    bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

async function poseidon2Hash(bb, inputs) {
  const fieldInputs = inputs.map(v => fieldToBytes(v));
  const { hash } = await bb.poseidon2Hash({ inputs: fieldInputs });
  return fieldToHex(hash);
}

async function computeMerkleRoot(bb, mutations) {
  const hashes = [];
  for (const mut of mutations) {
    if (mut.enabled && !mut.isDelete) {
      hashes.push(await poseidon2Hash(bb, ["1", mut.key, mut.value]));
    }
  }
  if (hashes.length === 0) return poseidon2Hash(bb, ["0"]);
  if (hashes.length === 1) return hashes[0];

  let p2 = 1;
  while (p2 < hashes.length) p2 *= 2;
  const last = hashes[hashes.length - 1];
  while (hashes.length < p2) hashes.push(last);

  let sz = p2;
  while (sz > 1) {
    const half = sz / 2;
    for (let j = 0; j < half; j++) {
      hashes[j] = await poseidon2Hash(bb, ["2", hashes[j * 2], hashes[j * 2 + 1]]);
    }
    sz = half;
  }
  return hashes[0];
}

async function main() {
  const bb = await Barretenberg.new();

  const blockNumber = 1;
  const numValidators = 1;
  const numMutations = 2;

  const VK_SIZE = 115;
  const PROOF_SIZE = 449;
  const PUBLIC_INPUTS_SIZE = 8;

  const blockMsg = Buffer.from(`block:${blockNumber}`);
  const msgHash = sha256(blockMsg);

  // Build mutations for Merkle root computation
  const mutationData = [];
  for (let i = 0; i < numMutations; i++) {
    mutationData.push({ key: `${i + 1}`, value: `${(i + 1) * 100}`, enabled: true, isDelete: false });
  }
  const computedRoot = await computeMerkleRoot(bb, mutationData);

  const lines = [];

  lines.push(`mutation_count = ${numMutations}`);
  lines.push(`sig_count = ${numValidators}`);
  lines.push(`recursive = false`);
  lines.push(`prev_proven_blocks = 0`);
  lines.push(`prev_genesis_root = "0x00"`);
  lines.push(`prev_state_root = "0xaa"`);
  lines.push(`new_state_root = "${computedRoot}"`);
  lines.push(`block_number = ${blockNumber}`);
  lines.push(`active_nodes = ${numValidators}`);
  lines.push(`prev_key_hash = "0x00"`);
  lines.push("");

  // Recursive proof arrays (zeroed for non-recursive)
  lines.push(`prev_proof = [${new Array(PROOF_SIZE).fill('"0x00"').join(", ")}]`);
  lines.push(`prev_vk = [${new Array(VK_SIZE).fill('"0x00"').join(", ")}]`);
  lines.push(`prev_public_inputs = [${new Array(PUBLIC_INPUTS_SIZE).fill('"0x00"').join(", ")}]`);
  lines.push("");

  // Generate dummy signature for disabled slots
  const dummyKey = new Uint8Array(32);
  dummyKey[31] = 0xff;
  const dummyMsg = new Uint8Array(32);
  const { publicKey: dPk } = await bb.schnorrComputePublicKey({ privateKey: dummyKey });
  const { s: dS, e: dE } = await bb.schnorrConstructSignature({ message: dummyMsg, privateKey: dummyKey });

  // Signatures
  for (let i = 0; i < 4; i++) {
    lines.push(`[[signatures]]`);

    if (i < numValidators) {
      const privateKey = new Uint8Array(32);
      privateKey[31] = i + 1;

      const { publicKey } = await bb.schnorrComputePublicKey({ privateKey });
      const { s, e } = await bb.schnorrConstructSignature({
        message: new Uint8Array(msgHash),
        privateKey,
      });

      lines.push(`pubkey_x = "${fieldToHex(publicKey.x)}"`);
      lines.push(`pubkey_y = "${fieldToHex(publicKey.y)}"`);
      lines.push(`signature = ${toTomlArray([...s, ...e])}`);
      lines.push(`msg = ${toTomlArray(Array.from(msgHash))}`);
      lines.push(`enabled = true`);
    } else {
      lines.push(`pubkey_x = "${fieldToHex(dPk.x)}"`);
      lines.push(`pubkey_y = "${fieldToHex(dPk.y)}"`);
      lines.push(`signature = ${toTomlArray([...dS, ...dE])}`);
      lines.push(`msg = ${toTomlArray(Array.from(dummyMsg))}`);
      lines.push(`enabled = false`);
    }
    lines.push("");
  }

  // Mutations -- now Field-typed key/value
  for (let i = 0; i < 32; i++) {
    lines.push(`[[mutations]]`);
    if (i < numMutations) {
      lines.push(`key = "${i + 1}"`);
      lines.push(`new_value = "${(i + 1) * 100}"`);
      lines.push(`is_delete = false`);
      lines.push(`enabled = true`);
    } else {
      lines.push(`key = "0"`);
      lines.push(`new_value = "0"`);
      lines.push(`is_delete = false`);
      lines.push(`enabled = false`);
    }
    lines.push("");
  }

  const toml = lines.join("\n");
  writeFileSync("Prover.toml", toml);
  console.log("Generated Prover.toml");
  console.log(`  Validators: ${numValidators}`);
  console.log(`  Mutations: ${numMutations}`);

  await bb.destroy();
}

main().catch(e => { console.error(e); process.exit(1); });
