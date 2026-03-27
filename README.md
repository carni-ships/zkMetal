# zkMetal

Zero-knowledge state transition proofs for [Persistia](https://github.com/carni-ships/Persistia) using Noir circuits and Barretenberg's UltraHonk proving system.

## What it proves

Each proof attests that:
1. **BFT quorum** — A quorum of validators signed the block (Schnorr on Grumpkin curve)
2. **State integrity** — Mutations produce the declared Poseidon2 Merkle root
3. **Chain continuity** — (Recursive) The previous proof in the IVC chain is valid

A single proof verifies the entire chain of state transitions back to genesis.

## Performance

| Metric | Value |
|--------|-------|
| Proof time | ~6s per block (Apple Silicon, native ARM64) |
| Circuit size | ~42K gates (non-recursive) / ~769K gates (recursive IVC) |
| Proof size | 16 KB |
| Max mutations/proof | 256 |
| Max validators/proof | 4 |
| Sustained throughput | ~10 proofs/min |

See [RESEARCH.md](RESEARCH.md) for detailed performance analysis, bottleneck findings, and scaling paths.

## Project Structure

```
src/main.nr              # Noir circuit (Schnorr + Poseidon2 Merkle + recursive IVC)
Nargo.toml               # Noir project config
Prover.toml              # Test witness inputs

prover/
  src/prover.ts           # CLI: prove, verify, watch, bench
  src/witness.ts          # Witness generation (Schnorr signing, Poseidon2 hashing)
  src/gen-verifier.ts     # Solidity verifier contract generator
  src/bench.mjs           # Benchmark harness
  gen_test_witness.mjs    # Generate Prover.toml from test data
  test/                   # Integration tests and benchmarks

target/
  persistia_state_proof.json   # Compiled circuit
  PersistiaVerifier.sol        # Generated Solidity verifier (~300K gas)
  PersistiaVerifier_evm.sol    # EVM-optimized variant

test/
  PersistiaVerifier.t.sol      # Foundry tests for on-chain verification
  EvmVerifier.t.sol            # EVM verifier tests
```

## Quick Start

### Prerequisites

- [Noir](https://noir-lang.org/) (nargo)
- [Barretenberg](https://github.com/AztecProtocol/barretenberg) (bb) — native binary for your platform
- Node.js 20+

### Compile the circuit

```bash
nargo compile
```

### Run circuit tests

```bash
nargo test
```

### Generate a proof

```bash
cd prover
npm install
npx tsx src/prover.ts prove --node https://your-persistia-node.com?shard=node-1 --block 100
```

### Watch mode (continuous proving)

```bash
npx tsx src/prover.ts watch --node https://your-persistia-node.com?shard=node-1 --interval 10
```

The watcher polls for new committed blocks and proves them sequentially, building a recursive IVC chain.

### Verify a proof

```bash
npx tsx src/prover.ts verify --proof proofs/block_100.json
```

### Generate Solidity verifier

```bash
npx tsx src/gen-verifier.ts
```

### Run Foundry tests

```bash
cd test
forge install
forge test
```

## Architecture

### Hash Function: Poseidon2

The state tree uses Poseidon2 — a field-native hash function that is ~100x cheaper in ZK circuits than SHA-256.

- Leaf: `Poseidon2([1, key, value])`
- Branch: `Poseidon2([2, left, right])`
- Empty: `Poseidon2([0])`

### Signatures: Schnorr on Grumpkin

Validators sign blocks with Schnorr signatures on the Grumpkin curve (BN254's embedded curve), natively supported in Noir at ~3K gates per verification.

### Proving System: UltraHonk

Proofs use Barretenberg's UltraHonk in non-ZK mode. State transitions are public — the proof attests to correct computation, not data privacy.

### Recursive IVC

Each proof optionally verifies the previous proof, building an Incremental Verifiable Computation chain. A light client needs only the latest proof to verify the entire history from genesis.

## License

MIT
