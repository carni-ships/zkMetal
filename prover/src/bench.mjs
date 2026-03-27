// Benchmark Noir circuit proving with Barretenberg backend.

import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync } from "fs";

const CIRCUIT_PATH = new URL("../../target/persistia_state_proof.json", import.meta.url).pathname;

console.log("=== Persistia Noir Circuit Benchmark ===\n");

// Load compiled circuit
console.log("Loading circuit...");
const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
console.log(`  ACIR opcodes: ${circuit.bytecode ? "present" : "missing"}`);

// Create Barretenberg API + backend
console.log("Initializing Barretenberg UltraHonk backend...");
const api = await Barretenberg.new({ threads: 8 });
const backend = new UltraHonkBackend(circuit.bytecode, api);
const noir = new Noir(circuit);

// Build a test witness using the witness module (Schnorr + Grumpkin)
console.log("Building test witness with Schnorr signatures...");
const { buildTestWitness, destroyBb } = await import("./witness.ts");
const witness = await buildTestWitness({
  numValidators: 1,
  numMutations: 2,
  blockNumber: 1,
});

// Execute (witness solving)
console.log("\n--- Execute (witness solving) ---");
const execStart = performance.now();
const { witness: solvedWitness } = await noir.execute(witness);
const execTime = performance.now() - execStart;
console.log(`  Time: ${(execTime / 1000).toFixed(2)}s`);

// Proof generation
console.log("\n--- Proof Generation (UltraHonk) ---");
const proveStart = performance.now();
const proof = await backend.generateProof(solvedWitness);
const proveTime = performance.now() - proveStart;
console.log(`  Time: ${(proveTime / 1000).toFixed(2)}s`);
console.log(`  Proof size: ${proof.proof.length} bytes`);

// Verification
console.log("\n--- Verification ---");
const verifyStart = performance.now();
const valid = await backend.verifyProof(proof);
const verifyTime = performance.now() - verifyStart;
console.log(`  Time: ${(verifyTime / 1000).toFixed(2)}s`);
console.log(`  Valid: ${valid}`);

// Summary
console.log("\n=== Summary ===");
console.log(`  Execute:      ${(execTime / 1000).toFixed(2)}s`);
console.log(`  Prove:        ${(proveTime / 1000).toFixed(2)}s`);
console.log(`  Verify:       ${(verifyTime / 1000).toFixed(2)}s`);
console.log(`  Proof size:   ${proof.proof.length} bytes`);
console.log(`  vs SP1 (~70s): ~${(70000 / proveTime).toFixed(1)}x faster`);

await destroyBb();
await api.destroy();
