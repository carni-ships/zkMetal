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
| Max mutations per proof | 256 (circuit compile-time constant) |
| Max validators per proof | 4 (circuit compile-time constant) |

### Sustained Throughput

| Configuration | Proof Rate | Block Rate | Prover Keeps Up? |
|---------------|-----------|------------|-----------------|
| 12s blocks, 22 events/vertex, alarm-only | ~10 proofs/min | ~5 even blocks/min | **Yes** (2x headroom) |
| 12s blocks, 80 events/vertex | ~10 proofs/min | ~5 even blocks/min | Yes, but mutations exceed 256 limit |
| 10s reactive alarm | ~10 proofs/min | ~78 blocks/min | **No** (gap grows ~100/90s) |
| 5s reactive alarm | ~12 proofs/min | ~97 blocks/min | **No** (gap grows ~128/90s) |
| 100ms reactive alarm (runaway) | ~10 proofs/min | ~1080 blocks/min | **No** (catastrophically behind) |

### Key Finding: Proof Time is Constant

The prover takes ~6s per block **regardless of mutation count** (1-256). The Noir circuit uses fixed-size arrays padded with disabled entries. This means:

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

## Hardware Benchmarks: Before vs After Optimization

**Test hardware:** Apple M3 Pro (12-core), 18 GB RAM, macOS
**Load profile:** 50 concurrent event-generating agents, sustained for 90+ seconds
**Prover backend:** Barretenberg UltraHonk, native ARM64 bb CLI

### Before Optimization

The unoptimized system had multiple compounding issues that made ZK proving infeasible under load:

| Metric | Value |
|--------|-------|
| Block production rate | ~1,080 blocks/min (100ms reactive alarm) |
| Mutations per block | 10,000–15,000 (dirty key accumulation bug) |
| Proof generation rate | ~10 proofs/min (~6s each) |
| Prover gap growth | ~100 blocks behind every 90s |
| Effective provability | **0%** — every block exceeded the 256-mutation circuit limit |
| Event-triggered rounds | ~26 rounds/s under 50-agent load |
| Commits advancing? | **No** — `checkCommit()` blocked self-commit, alarm never called `tryCommitRounds()` |

Under the default configuration, the system entered a runaway state where blocks were produced 100x faster than the prover could consume them, and every block was unprovable due to mutation overflow. The prover fell behind catastrophically and could never recover.

### After Optimization

Seven fixes (documented above) brought the system to a stable, provable steady state:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Block production rate | ~1,080/min | ~5/min | 216x reduction (matched to prover) |
| Mutations per block | 10,000–15,000 | ~200 | 60x reduction (under 256 limit) |
| Prover gap (steady state) | ∞ (growing) | 2–4 blocks | **Converges to zero** |
| Provable blocks | 0% | 100% | All blocks within circuit capacity |
| Events finalized/sec | ~50 (but unprovable) | ~3.7 (all proven) | Provable throughput: 0 → 3.7/s |
| Prover headroom | None | **2x** | Prover at 50% capacity |
| Recursive IVC chain | Impossible | 40+ blocks from genesis | Unlimited chain length |
| Time to prove 14 blocks | N/A | ~3 minutes | ~6s/block sustained |

### Proving Pipeline Performance (Optimized)

Detailed timing breakdown for a single proof on M3 Pro:

| Stage | Time | Notes |
|-------|------|-------|
| Fetch block from node API | ~50ms | HTTP round-trip to Cloudflare DO |
| Build witness (Schnorr + Poseidon2) | ~200ms | Barretenberg WASM, includes dummy sig generation |
| Solve witness (noir.execute) | ~200ms | ACIR constraint solving |
| Generate proof (native bb) | ~5.2s | UltraHonk, 8 threads, VK pre-cached |
| Generate recursive artifacts | ~300ms | VK fields + hash for IVC chaining |
| Submit proof to node | ~100ms | HTTP POST to /proof/zk/submit |
| **Total per block** | **~6s** | Pipelined mode overlaps stages |

### Throughput by Watch Mode (Optimized)

| Mode | Throughput | Latency | Use Case |
|------|-----------|---------|----------|
| Sequential | ~10 proofs/min | ~6s/proof | Production with IVC chaining |
| Pipelined | ~12 proofs/min | ~5.2s/proof | Single-prover catch-up (overlaps witness + prove) |
| Parallel (6 workers) | ~36 proofs/min | ~10s/batch of 6 | Bulk catch-up without IVC |
| Parallel msgpack (6 workers) | ~38 proofs/min | ~9.5s/batch of 6 | Bulk catch-up, 6% faster (persistent bb) |

### Key Takeaway

The prover itself was never the bottleneck — it consistently generates proofs in ~6s regardless of configuration. The bottleneck was the **data pipeline**: block production rate, mutation accumulation, and commit advancement. All optimizations were on the consensus/data side, not the proving side.

---

## Scaling Paths

### Short-term: Increase Circuit Mutation Capacity

The Noir circuit's mutation array is a compile-time constant. Increasing from 256 to 512 or 1024 would allow more events per block:

| Mutation Slots | Events/Block | Finalized Events/s | Gate Count Impact |
|---------------|-------------|--------------------|--------------------|
| 256 (current) | ~44 | ~3.7 | ~42K gates |
| 512 | ~100 | ~8.3 | ~50K gates (est.) |
| 1024 | ~200 | ~16.7 | ~65K gates (est.) |

The proof time increase is sublinear — doubling mutations does not double proof time because the Poseidon2 tree hashing is O(n log n) but dominates less than Schnorr verification.

### Medium-term: Parallel Proving

Multiple prover instances can prove non-overlapping blocks simultaneously. The IVC chain requires serial dependency for recursive proofs, but independent lineages can run in parallel.

### Long-term: Proof Aggregation

Batch multiple blocks into a single aggregate proof. Instead of proving each block individually, prove a range of N blocks with one circuit execution. This amortizes the fixed cost (Schnorr verification, IVC overhead) across many blocks.

---

## Recursive IVC Chain

The prover supports recursive Incremental Verifiable Computation (IVC):

- **Genesis proof:** `ENABLE_RECURSIVE=false`, ~42K gates, ~6s
- **Chained proof:** `ENABLE_RECURSIVE=true`, ~769K gates, ~6s (with cached VK)
- **Chain length:** Unlimited — each proof references the previous, building a cryptographic chain from genesis

The `proven_blocks` counter in each proof tracks chain length. A light client needs only the latest proof to verify the entire history.

**Current production chain:** 40+ blocks proven with recursive chaining from genesis root.

---

## Future Investigation Directions

Concrete optimization paths and open questions for continuing the prover optimization work.

### 1. Gate Count Reduction

Profile which circuit operations dominate gate count (Schnorr verification at ~3K gates/sig vs Poseidon2 Merkle tree). Investigate if batching multiple Schnorr verifications or using aggregate signatures could reduce per-proof overhead.

### 2. Adaptive Mutation Ceiling

The network's adaptive parameter tuning (EIP-1559-style) doesn't account for the circuit's 256-mutation hard limit. Need a mutation-aware cap that prevents the adaptive system from scaling `max_events_per_vertex` above the point where mutations would exceed circuit capacity.

Formula: `max_safe_events = floor(CIRCUIT_MUTATION_SLOTS / (AVG_MUTATIONS_PER_EVENT * VERTICES_PER_BLOCK))`

### 3. Witness Generation Parallelism

Currently witness building (Schnorr signing, Poseidon2 hashing via Barretenberg WASM) runs single-threaded at ~200ms. Investigate batching Poseidon2 hash calls or using the native bb for witness-stage hashing.

### 4. VK Precomputation for Recursive Circuits

VK generation adds ~700ms on first prove. For recursive IVC circuits (769K gates), investigate whether the VK can be embedded in the compiled circuit artifact to eliminate this cost entirely.

### 5. Proof Compression

Current proofs are 16KB. Investigate Barretenberg's proof compression modes and whether EVM-optimized proofs (7.8KB) can also be used for recursive chaining, not just on-chain verification.

### 6. Dynamic Circuit Sizing

The fixed 256-mutation, 4-validator arrays waste gates when blocks are small. Investigate Noir's comptime generics or conditional compilation to support multiple circuit variants (small/medium/large) selected at prove time.

### 7. GPU Acceleration

Barretenberg supports GPU proving via CUDA. Benchmark proof times on GPU vs M3 Pro CPU. Determine if the ~6s proof time is CPU-bound (multi-scalar multiplication) or memory-bound.

### 8. Cross-Block State Root Continuity

Currently each block's `prev_state_root` is fetched by re-computing the previous block's Merkle root. Investigate embedding the state root in the recursive proof's public inputs so it chains automatically without re-fetching.

### 9. Proof Aggregation Circuits

Design a Noir aggregation circuit that verifies N individual block proofs and produces a single aggregate proof. This would amortize the Schnorr and IVC overhead across many blocks, potentially reducing per-block cost to sub-second.

### 10. Prover Market Integration

The generic SDK architecture enables third-party provers. Investigate economic models for a proving marketplace where provers bid on blocks (similar to MEV-Boost but for ZK proofs).

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
