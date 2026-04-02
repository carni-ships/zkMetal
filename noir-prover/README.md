# zkMetal

Generic Noir prover SDK — load any compiled circuit, generate UltraHonk proofs, verify, and run continuous proving pipelines.

Built on [Barretenberg](https://github.com/AztecProtocol/barretenberg) with support for WASM, native CLI, and persistent msgpack worker pools.

## Features

- **Any Noir circuit** — point at a compiled JSON, start proving
- **Multiple proving modes** — WASM, native bb CLI, pipelined, parallel msgpack workers
- **Recursive IVC** — chain proofs for incremental verifiable computation
- **Pluggable architecture** — implement `DataSource`, `WitnessBuilder`, `ProofSink` for your app
- **Solidity verifiers** — generate on-chain verification contracts

## Quick Start

```bash
cd prover
npm install

# Prove a single block (using Persistia adapter)
npx tsx src/cli.ts prove --node https://your-node.com?shard=node-1 --block 100

# Watch mode (continuous proving)
npx tsx src/cli.ts watch --node https://your-node.com?shard=node-1 --mode sequential --recursive

# Parallel proving with persistent bb workers
npx tsx src/cli.ts watch --node https://your-node.com?shard=node-1 --mode parallel --workers 6

# Verify a proof
npx tsx src/cli.ts verify --proof proofs/block_100.json

# Benchmark
npx tsx src/cli.ts bench --node https://your-node.com?shard=node-1 --block 100
```

## SDK Usage

```typescript
import { ProverEngine, watchSequential } from "zkmetal";
import type { DataSource, WitnessBuilder, ProofSink } from "zkmetal/types";

// 1. Point at your compiled circuit
const engine = new ProverEngine({
  circuitPath: "./target/my_circuit.json",
  threads: 8,
});

// 2. Implement the three interfaces for your application
const dataSource: DataSource<MyBlock> = {
  fetchBlock: async (n) => { /* fetch block n from your chain/API */ },
  fetchLatestBlockNumber: async () => { /* return latest block number */ },
};

const witnessBuilder: WitnessBuilder<MyBlock> = {
  buildWitness: async (block, blockNumber, recursiveOpts?) => {
    // Transform your block data into the witness map your circuit expects
    return { field1: "value1", field2: "value2", ... };
  },
};

const proofSink: ProofSink = {
  submitProof: async (proof) => { /* POST proof to your verifier */ },
};

// 3. Start proving
await watchSequential(engine, dataSource, witnessBuilder, proofSink, {
  intervalSec: 10,
  recursive: true,
});
```

### Using the Persistia Adapter

```typescript
import { ProverEngine, watchSequential } from "zkmetal";
import { createPersistiaAdapter } from "zkmetal/adapters/persistia";

const engine = new ProverEngine({
  circuitPath: "./target/persistia_state_proof.json",
});

const { dataSource, witnessBuilder, proofSink } = createPersistiaAdapter(
  "https://your-persistia-node.com?shard=node-1"
);

await watchSequential(engine, dataSource, witnessBuilder, proofSink, {
  recursive: true,
  useNative: true,
});
```

## Performance

| Metric | Value |
|--------|-------|
| Proof time | ~6s per block (Apple Silicon, native ARM64) |
| Circuit size | ~42K gates (non-recursive) / ~769K gates (recursive IVC) |
| Proof size | 16 KB |
| Sustained throughput | ~10 proofs/min (sequential) |

See [RESEARCH.md](RESEARCH.md) for detailed performance analysis and scaling paths.

## Architecture

### Proving Engine (`ProverEngine`)

Core class that wraps Noir + Barretenberg. Handles circuit loading, witness execution, proof generation, and verification. Supports both WASM and native bb CLI backends.

### Watch Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `sequential` | One block at a time, supports IVC chaining | Production with recursive proofs |
| `pipelined` | Overlaps witness solving with native proving | Single-prover catch-up |
| `parallel` | Multiple persistent bb workers via msgpack | High-throughput catch-up |

### Adapter Pattern

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ DataSource   │────>│ ProverEngine │────>│ ProofSink     │
│ (fetch data) │     │ (prove)      │     │ (submit)      │
└─────────────┘     └──────────────┘     └───────────────┘
       ↑                    ↑                     ↑
       │            ┌──────────────┐              │
       │            │WitnessBuilder│              │
       │            │(data→witness)│              │
       │            └──────────────┘              │
       │                                          │
   Your chain API                          Your verifier
```

Implement `DataSource`, `WitnessBuilder`, and `ProofSink` from `zkmetal/types` to connect any data source to the proving pipeline.

### Project Structure

```
src/main.nr                    # Noir circuit (reference: Persistia state proof)
Nargo.toml                     # Noir project config

prover/
  src/
    index.ts                   # SDK exports
    types.ts                   # Core interfaces (DataSource, WitnessBuilder, ProofSink)
    engine.ts                  # ProverEngine (circuit loading, proving, verification)
    watcher.ts                 # Watch loops (sequential, pipelined, parallel)
    persistent-bb.ts           # PersistentBb msgpack worker
    witness.ts                 # Persistia-specific witness generation
    cli.ts                     # CLI entrypoint
    adapters/
      persistia/index.ts       # Persistia adapter (reference implementation)
  test/                        # Integration tests and benchmarks

target/
  persistia_state_proof.json   # Compiled circuit
  PersistiaVerifier.sol        # Generated Solidity verifier

test/
  PersistiaVerifier.t.sol      # Foundry tests for on-chain verification
```

## Writing a Custom Adapter

To prove a different Noir circuit with your own data source:

1. **Compile your circuit**: `nargo compile` in your circuit directory
2. **Implement `DataSource<B>`**: fetch your blocks/units of work
3. **Implement `WitnessBuilder<B>`**: transform blocks into the witness map your circuit expects
4. **Implement `ProofSink`** (optional): submit proofs to your verifier
5. **Wire it up**:

```typescript
const engine = new ProverEngine({ circuitPath: "./target/your_circuit.json" });
await watchSequential(engine, yourDataSource, yourWitnessBuilder, yourProofSink);
```

## Hash Function: Poseidon2

The reference circuit uses Poseidon2 — a field-native hash function ~100x cheaper in ZK circuits than SHA-256.

## Signatures: Schnorr on Grumpkin

The reference circuit verifies Schnorr signatures on the Grumpkin curve (BN254's embedded curve), natively supported in Noir at ~3K gates per verification.

## Proving System: UltraHonk

Proofs use Barretenberg's UltraHonk in non-ZK mode. Suitable for applications where the proof attests to correct computation rather than data privacy.

## License

MIT
