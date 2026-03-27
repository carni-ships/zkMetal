# Noir ZK Prover: Research Findings & Performance Analysis

## Overview

The Persistia Noir ZK prover generates UltraHonk proofs for DAG-BFT consensus state transitions. Each proof attests that:
1. A BFT quorum of validators signed the block (Schnorr on Grumpkin curve)
2. State mutations produce the declared Poseidon2 Merkle root
3. (Optional) The previous proof in the IVC chain is valid

This document captures performance findings, bottlenecks discovered, and architectural decisions made during production deployment and stress testing.

---

## Prover Performance

### Proof Generation Time

| Metric | Value |
|--------|-------|
| Proof time (per block) | ~6 seconds |
| Backend | Barretenberg UltraHonk (native ARM64) |
| Circuit size | ~42K gates (non-recursive) / ~769K gates (recursive IVC) |
| Proof size | 16,000 bytes |
| Max mutations per proof | 1024 (circuit compile-time constant) |
| Max validators per proof | 4 (circuit compile-time constant) |

### Sustained Throughput

| Configuration | Proof Rate | Block Rate | Prover Keeps Up? |
|---------------|-----------|------------|-----------------|
| 12s blocks, 22 events/vertex, alarm-only | ~10 proofs/min | ~5 even blocks/min | **Yes** (2x headroom) |
| 12s blocks, 80 events/vertex | ~10 proofs/min | ~5 even blocks/min | Yes, mutations within 1024 limit |
| 10s reactive alarm | ~10 proofs/min | ~78 blocks/min | **No** (gap grows ~100/90s) |
| 5s reactive alarm | ~12 proofs/min | ~97 blocks/min | **No** (gap grows ~128/90s) |
| 100ms reactive alarm (runaway) | ~10 proofs/min | ~1080 blocks/min | **No** (catastrophically behind) |

### Key Finding: Proof Time is Constant

The prover takes ~6-8s per block **regardless of mutation count** (1-1024). The Noir circuit uses fixed-size arrays padded with disabled entries. This means:

> **Optimal throughput comes from packing more events per block, not producing more blocks.**

---

## Bottleneck Analysis

### 1. Block Production Rate vs Proof Rate

**Problem:** The reactive alarm system (`scheduleReactiveAlarm`) was designed for multi-node quorum convergence. In single-node or low-node configurations, it creates a tight feedback loop:
- Vertex created -> round advances -> reactive alarm (100ms) -> vertex created -> ...
- Result: ~18 blocks/second, overwhelming the prover (~0.17 proofs/second)

**Solution:** Made the reactive alarm delay configurable (`reactive_alarm_delay_ms`, default 5000ms) and moved vertex creation to alarm-only (removed event-triggered `maybeCreateVertex`).

**Impact:** Block rate dropped from ~1080/min to ~5/min, matching prover capacity.

### 2. Mutation Accumulation Across Blocks

**Problem:** `stateTree.getDirtyMutations()` returned ALL mutations since the last `computeCommitment()` call. But `commitAnchor()` never called `computeCommitment()`, so dirty keys accumulated indefinitely. Blocks routinely contained 10,000-15,000 mutations, far exceeding the circuit's 256-slot limit.

**Solution:** Added `clearDirtyKeys()` call after capturing mutations in `commitAnchor()`. Each block now captures only mutations from events finalized in that specific block.

**Impact:** Mutation count per block dropped from ~15,000 to ~200 (at 22 events/vertex).

### 3. Mutations Per Event

Each game event type produces a different number of state mutations:

| Event Type | Mutations |
|-----------|-----------|
| `place` | ~5 (tile, inventory, token balance, player stats, world meta) |
| `break` | ~3 (tile removal, inventory, token balance) |
| `craft` | ~4 (inventory changes, recipe tracking) |

With 2 vertices per commit cycle and variable mutations per event, the safe limit is:

> **22 events/vertex x 2 vertices/block x ~5 mutations/event = ~220 mutations/block** (under 256 limit)

### 4. Event-Triggered Vertex Creation

**Problem:** Every accepted event called `maybeCreateVertex()`, which created a vertex and advanced the round. Under load (50+ agents, ~50 events/s), this caused ~26 rounds/second — identical to the reactive alarm problem but triggered by HTTP requests instead.

**Solution:** Disabled event-triggered vertex creation entirely. Vertices are now created exclusively by the alarm timer, giving deterministic block production rate.

**Impact:** Round advancement became predictable: exactly 2 rounds per alarm cycle.

### 5. Single-Node Commit Rule

**Problem:** `checkCommit()` in consensus.ts had a hardcoded `if (activeCount < 3) return false` guard that prevented commits with fewer than 3 active nodes. The self-commit fallback (added for solo operation) was unreachable because the function returned early.

**Solution:** Moved the 3-node check after the anchor lookup and made it only gate the quorum path, allowing the self-commit fallback to execute for any node count.

**Impact:** Single-node networks can now commit every even round without manual intervention.

### 6. Missing `tryCommitRounds()` in Alarm

**Problem:** The DO alarm created vertices and advanced rounds but never called `tryCommitRounds()`. Commits only happened during gossip sync or vertex reception — neither of which fires in single-node mode with no gossip peers.

**Solution:** Added `tryCommitRounds()` call in the alarm between vertex creation and pruning.

**Impact:** Commits now advance every alarm cycle instead of stalling indefinitely.

### 7. SQLite Bind Parameter Limit

**Problem:** Bulk INSERT of block mutations with 200+ rows exceeded Cloudflare DO SQLite's bind parameter limit (lower than standard SQLite's 999). Initial fix with chunked inserts (50 rows) also failed.

**Solution:** Single-row INSERT in a loop. Performance impact is negligible since the loop runs synchronously within the DO's single-threaded runtime.

**Impact:** Blocks with any number of mutations (up to 256) can be persisted without SQL errors.

---

## Optimal Configuration

Based on stress testing with 50 concurrent event agents:

```
round_interval_ms = 12000        # 12-second block time
max_events_per_vertex = 22       # ~200 mutations/block (under 256 limit)
reactive_alarm_delay_ms = 30000  # Effectively disabled (alarm-only vertices)
min_nodes_for_consensus = 1      # Allow single-node operation
```

**Resulting throughput:**
- ~3.7 finalized events/second
- ~5 provable blocks/minute
- Prover keeps pace with ~2x headroom
- Recursive IVC proof chain extends indefinitely

---

## Scaling Paths

### Short-term: Increase Circuit Mutation Capacity

The Noir circuit's mutation array is a compile-time constant. Increasing from 256 to 1024 allows more events per block, but gate count scaling is super-linear due to Poseidon2 Merkle tree computation:

| Mutation Slots | Events/Block | Finalized Events/s | Actual Gate Count |
|---------------|-------------|--------------------|--------------------|
| 1 | ~2 | ~0.2 | 46K |
| 256 | ~44 | ~3.7 | 632K |
| 1024 (current) | ~200 | ~16.7 | **9.13M** |

**Gate count profiling (2026-03):** Detailed component isolation revealed that the Poseidon2 Merkle tree computation accounts for 99.7% of all gates. Schnorr signature verification (4 validators) contributes only ~41K gates (0.45%). Function signature overhead (array inputs) is ~5K gates. Noir compiles all conditional block contents into constraints regardless of runtime values, so loop unrolling and conditional guards do not reduce gate count. The only paths to meaningful reduction are: (1) reducing MAX_MUTATIONS, (2) switching to incremental Merkle proofs (verify paths instead of full tree recomputation), or (3) GPU-accelerated proving to make the large circuit tractable.

### Medium-term: Parallel Proving

Multiple prover instances can prove non-overlapping blocks simultaneously. The IVC chain requires serial dependency for recursive proofs, but independent lineages can run in parallel.

### Long-term: Proof Aggregation

Batch multiple blocks into a single aggregate proof. Instead of proving each block individually, prove a range of N blocks with one circuit execution. This amortizes the fixed cost (Schnorr verification, IVC overhead) across many blocks.

### Medium-term: Metal GPU Acceleration

A Metal compute shader for BN254 multi-scalar multiplication (MSM) has been implemented in `metal/`. Initial benchmarks on M3 Pro:

| Points | GPU Time | Notes |
|--------|----------|-------|
| 1,024 | 169ms | Shader compilation overhead dominates |
| 4,096 | 358ms | Sublinear scaling begins |
| 16,384 | 2,542ms | Memory bandwidth limited |
| 65,536 | 6,843ms | ~10K points/sec sustained |

The MSM kernel uses Pippenger's bucket method with configurable window size. Current implementation uses per-thread bucket accumulation with host-side reduction. Next steps: atomic bucket accumulation, SIMD-group reductions, and integration with Barretenberg's proving pipeline to replace CPU MSM.

### Long-term: Cloudflare Free Tier Prover

Explore whether the proving pipeline can run within the resource constraints of Cloudflare's free tier (Workers, Durable Objects, R2). Key constraints to investigate: Worker CPU time limits (10ms free / 30s paid), memory caps (128MB), no native binary execution (WASM only). Would require the Barretenberg WASM backend and potentially chunked proving across multiple Worker invocations.

---

## Recursive IVC Chain

The prover supports recursive Incremental Verifiable Computation (IVC):

- **Genesis proof:** `ENABLE_RECURSIVE=false`, ~42K gates, ~6s
- **Chained proof:** `ENABLE_RECURSIVE=true`, ~769K gates, ~6s (with cached VK)
- **Chain length:** Unlimited — each proof references the previous, building a cryptographic chain from genesis

The `proven_blocks` counter in each proof tracks chain length. A light client needs only the latest proof to verify the entire history.

**Current production chain:** 40+ blocks proven with recursive chaining from genesis root.

---

## Architecture Notes

### Hash Function: Poseidon2

The state tree uses Poseidon2 (field-native, ~100x cheaper in ZK circuits than SHA-256). This is a genesis-time decision — changing hash functions invalidates all existing state roots and proofs.

- Leaf hash: `Poseidon2([1, key, value])`
- Branch hash: `Poseidon2([2, left, right])`
- Empty hash: `Poseidon2([0])`

### Signature Scheme: Schnorr on Grumpkin

Validators sign blocks with Schnorr signatures on the Grumpkin curve (BN254's embedded curve). This is natively supported in Noir via `schnorr::verify_signature` and costs ~3K gates per signature verification.

### Proof Format: UltraHonk (non-ZK)

Proofs use Barretenberg's UltraHonk proving system in non-ZK mode (`noir-recursive-no-zk`). This is faster than ZK mode and appropriate since the state transitions are public. The proof attests to correct computation, not data privacy.

### On-Chain Verification

Generated Solidity verifier contracts (`PersistiaVerifier.sol`) enable on-chain proof verification on any EVM chain. The verifier is ~24KB deployed and costs ~300K gas per verification.
