#!/usr/bin/env tsx
// Generate a Solidity verifier contract from the compiled Noir circuit.
// This contract can be deployed to Berachain for on-chain proof verification.
//
// Usage: tsx gen-verifier.ts [--output ../contracts/PersistiaVerifier.sol]

import { BarretenbergBackend } from "@noir-lang/backend_barretenberg";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve } from "path";

const CIRCUIT_PATH = resolve(import.meta.dirname ?? ".", "../../target/persistia_state_proof.json");

async function main() {
  const outputPath = process.argv.includes("--output")
    ? process.argv[process.argv.indexOf("--output") + 1]
    : resolve(import.meta.dirname ?? ".", "../../contracts/PersistiaVerifier.sol");

  if (!existsSync(CIRCUIT_PATH)) {
    console.error("Circuit not compiled. Run 'nargo compile' first.");
    process.exit(1);
  }

  const circuit = JSON.parse(readFileSync(CIRCUIT_PATH, "utf-8"));
  const backend = new BarretenbergBackend(circuit);

  console.log("Generating Solidity verifier...");
  const solidity = await backend.getSolidityVerifier();

  writeFileSync(outputPath, solidity);
  console.log(`Verifier written to ${outputPath}`);
  console.log("\nDeploy to Berachain and call verify(proof, publicInputs) to verify anchors on-chain.");

  await backend.destroy();
}

main().catch(console.error);
