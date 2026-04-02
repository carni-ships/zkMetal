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
| Proof time (per block, cold) | ~855ms (with precomputed VK) |
| Proof time (per block, cached) | ~795ms (with PrecomputedCache) |
| Backend | Barretenberg UltraHonk (native ARM64 + Metal GPU) |
| Circuit variant | Incremental Merkle (428K gates) |
| Circuit size (non-recursive) | ~42K gates |
| Circuit size (recursive IVC) | ~769K gates |
| Proof size | 16,000 bytes |
| Max mutations per proof | 64 (incremental) / 1024 (full-tree) |
| Max validators per proof | 4 (circuit compile-time constant) |

> **Note:** Proof time reduced from ~6s to ~855ms through 14 optimization sessions
> targeting Metal GPU offloading, memory allocation, CPU compute, and circuit building.
> See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for full details.

### Sustained Throughput

| Configuration | Proof Rate | Block Rate | Prover Keeps Up? |
|---------------|-----------|------------|-----------------|
| 12s blocks, 22 events/vertex, alarm-only | ~70 proofs/min | ~5 even blocks/min | **Yes** (14x headroom) |
| 12s blocks, 80 events/vertex | ~70 proofs/min | ~5 even blocks/min | Yes, mutations within 64 limit |
| 6s blocks (incremental circuit) | ~70 proofs/min | ~10 blocks/min | **Yes** (7x headroom) |
| 10s reactive alarm | ~70 proofs/min | ~78 blocks/min | **Yes** (near parity) |
| 5s reactive alarm | ~70 proofs/min | ~97 blocks/min | **No** (gap grows slowly) |
| 100ms reactive alarm (runaway) | ~70 proofs/min | ~1080 blocks/min | **No** (catastrophically behind) |

> **Updated 2026-03-31:** With Metal GPU acceleration and 14 optimization sessions,
> proof rate increased from ~10/min to ~70/min (855ms per proof vs ~6s).

### Key Finding: Proof Time is Constant

The prover takes ~855ms per block **regardless of mutation count** (1-64 for incremental circuit). The Noir circuit uses fixed-size arrays padded with disabled entries. This means:

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

**Gate count profiling (profiled):** Detailed component isolation revealed that the Poseidon2 Merkle tree computation accounts for 99.7% of all gates. Schnorr signature verification (4 validators) contributes only ~41K gates (0.45%). Function signature overhead (array inputs) is ~5K gates. Noir compiles all conditional block contents into constraints regardless of runtime values, so loop unrolling and conditional guards do not reduce gate count. The only paths to meaningful reduction are: (1) reducing MAX_MUTATIONS, (2) switching to incremental Merkle proofs (verify paths instead of full tree recomputation), or (3) GPU-accelerated proving to make the large circuit tractable.

### Short-term: Incremental Merkle Proof Circuit (Implemented)

An alternative circuit (`circuits/incremental/`) replaces full tree recomputation with sparse Merkle tree inclusion/update proofs. Each mutation provides a sibling path and the circuit verifies the old root, then computes the new root by replacing the leaf.

| Circuit Variant | MAX_MUTATIONS | ACIR Opcodes | Gate Count | vs Full-Tree |
|----------------|--------------|-------------|------------|--------------|
| Full-tree | 1024 | 2,137,893 | 9,129,173 | baseline |
| Incremental (D=20) | 64 | 21,312 | 428,709 | **21.3x smaller** |

Key tradeoffs:
- **Gate count:** O(M * D) vs O(M * M). With M=64 and D=20, this is 21x fewer gates.
- **State model:** Persistent sparse Merkle tree across blocks (supports 2^20 ≈ 1M keys) vs per-block ephemeral tree.
- **Witness complexity:** Requires Merkle sibling paths for each mutation (provided by the data source).
- **Throughput:** Fewer mutations per proof (64 vs 1024), but proofs are ~21x faster to generate. Net effect depends on mutation batching.

### Medium-term: Parallel Proving

Multiple prover instances can prove non-overlapping blocks simultaneously. The IVC chain requires serial dependency for recursive proofs, but independent lineages can run in parallel.

### Long-term: Proof Aggregation

Batch multiple blocks into a single aggregate proof. Instead of proving each block individually, prove a range of N blocks with one circuit execution. This amortizes the fixed cost (Schnorr verification, IVC overhead) across many blocks.

### Implemented: Metal GPU Acceleration (Complete)

Metal GPU acceleration has been fully integrated into the Barretenberg UltraHonk prover. The GPU handles all MSM operations (54% of prove time), while CPU handles field arithmetic (41%). See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for the full 14-session optimization history.

**Final results on M3 Pro (428K-gate incremental circuit):**

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Prove time (cold, precomputed VK) | 3,850ms | ~855ms | **-78%** |
| GPU MSM (201K points) | 267ms | 55ms | **-79%** |
| create_circuit | 252ms | 93ms | **-63%** |
| Peak memory | ~500 MiB | ~308 MiB | **-38%** |

The prover is now within 15-35% of theoretical hardware limits on M3 Pro. Further gains require M4 Pro/Max hardware (2x GPU cores) or protocol-level changes (fewer commitment rounds).

### Long-term: M4 Pro/Max Hardware Upgrade

The M4 Pro/Max with 2x more GPU cores is expected to roughly halve GPU MSM time (~514ms -> ~257ms), bringing total prove time to ~600ms. This is the highest-confidence path to further improvement.

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

---

## Metal MSM Optimization Report

### Summary

The Metal GPU MSM implementation for BN254 was developed across two phases:

1. **Standalone optimization** (4 sessions): ~1100ms to ~61ms (18x speedup) at 524K points
2. **Barretenberg integration** (10 sessions): Integrated into UltraHonk prover, achieving 78% end-to-end prove time reduction

The implementation uses Pippenger's bucket method with GLV endomorphism, GPU counting sort, fused gather/reduce kernels, and GPU parallel bucket accumulation. In the integrated prover, GPU MSMs account for 54% of the ~855ms prove time.

### Hardware

- **GPU:** Apple M3 Pro (14 cores, unified memory, SIMD width 32)
- **Curve:** BN254 (254-bit prime field, y^2 = x^3 + 3)
- **Point count:** 524,288 (representative of MSM sizes in UltraHonk proving)
- **Field arithmetic:** 256-bit Montgomery CIOS multiplication (8x32-bit limbs)

### Performance Timeline

| Session | Before | After | Speedup | Key Changes |
|---------|--------|-------|---------|-------------|
| 1 | ~1100ms | ~535ms | 2.1x | Pippenger bucket method, GPU reduce/bucketSum/combine |
| 2 | ~535ms | ~165ms | 3.2x | GLV endomorphism, zero-alloc scalar decomposition |
| 3 | ~165ms | ~85ms | 1.9x | GPU GLV decompose, 512-segment combine, merged pipeline |
| 4 | ~85ms | ~61ms | 1.4x | Count-sorted reduce, precomputed indices, batch overlap |
| **Total** | **~1100ms** | **~61ms** | **18x** | |

### Current Timing Breakdown (524K points, ~61ms)

```
glv:        3-5ms    GPU scalar decomposition (256-bit → 2x 128-bit)
sort:       7-9ms    CPU counting sort (overlapped with batch 1 reduce)
reduce:    ~37ms     GPU bucket accumulation (60% of total)
bucketSum:  ~6.5ms   GPU segmented weighted sum
combine:    ~0.6ms   GPU tree reduction of segment results
horner:     ~0.5ms   CPU window combination (16 doublings per window)
```

### Session 1: Pippenger Bucket Method (~1100ms → ~535ms)

Replaced naive scalar-times-point accumulation with Pippenger's bucket method:

1. **Window decomposition:** Split each 256-bit scalar into w=16-bit windows (16 windows per scalar). Each window digit selects a bucket index.
2. **Bucket accumulation (GPU):** For each window, accumulate points into 2^16 = 65,536 buckets. Each GPU thread handles one bucket, summing all assigned points using Jacobian mixed addition.
3. **Bucket sum (GPU):** Compute weighted sum of buckets per window using running-sum trick: `result = sum(i * bucket[i])`. Segmented across 512 GPU threads per window.
4. **Window combination (CPU):** Horner's method combines window results: `result = W[n-1] * 2^(16*(n-1)) + ... + W[1] * 2^16 + W[0]`.

CPU counting sort orders points by bucket index before GPU dispatch, enabling lock-free sequential reads per bucket (no atomics needed).

### Session 2: GLV Endomorphism (~535ms → ~165ms)

Exploited the BN254 curve's efficient endomorphism to halve scalar bit-length:

1. **Scalar decomposition:** For each scalar k, compute k1, k2 where k = k1 + lambda * k2 (mod r). Both k1, k2 are ~128 bits (half the original 256 bits). Uses Babai's rounding with precomputed lattice vectors.
2. **Point doubling:** For each point P, compute Q = beta * P.x (the endomorphism image) at near-zero cost (one field multiply).
3. **Window reduction:** 128-bit scalars need only 8 windows instead of 16, halving GPU reduce work.
4. **Zero-alloc implementation:** Inline 128x128-bit multiply and flat scalar buffers avoid heap allocation overhead for 1M+ scalars.

### Session 3: GPU Pipeline Optimization (~165ms → ~85ms)

Moved compute-heavy operations to GPU and improved pipeline overlap:

1. **GPU GLV decomposition:** Moved scalar decomposition from CPU to a Metal compute kernel. Processes all 524K scalars in parallel (~3-5ms vs ~15ms on CPU).
2. **GPU endomorphism:** Compute endomorphism points (beta * P.x) on GPU, overlapped with CPU sort.
3. **512-segment bucket sum:** Increased parallelism from 256 to 512 segments per window, improving GPU occupancy.
4. **Unified GPU pipeline:** Reduce, bucketSum, and combine dispatched in a single command buffer, eliminating inter-kernel launch overhead.

### Session 4: SIMD Divergence Elimination (~85ms → ~61ms)

Three optimizations targeting the CPU-GPU pipeline:

#### 1. Count-Sorted Reduce (reduce: 55ms → 37ms)

**Problem:** Bucket sizes follow a Poisson distribution (mean ~16 for 524K points / 32K buckets). Within a SIMD group of 32 threads, the slowest thread (processing the largest bucket) stalls all others. With Poisson(16), the max of 32 samples averages ~28, wasting ~40% of SIMD cycles.

**Solution:** After counting sort, sort bucket IDs by their point count (descending) within each window. Build a permutation mapping (`count_sorted_map`) so the GPU kernel reads bucket data through indirection. Adjacent threads in a SIMD group now process buckets of similar size, achieving near-zero divergence.

The GPU kernel indexes through the mapping:
```metal
uint orig_pos = count_sorted_map[tid];       // sorted → original position
uint count = bucket_counts[orig_pos];         // indirect read
uint offset = bucket_offsets[orig_pos];       // indirect read
// ... accumulate count points sequentially
buckets[orig_pos] = acc;                      // write to original position
```

The CPU builds the mapping with a counting sort on bucket counts (O(n) in the max count, which is ~40 for Poisson(16)):
```
for each window:
    histogram bucket counts → prefix sum (descending) → scatter mapping
```

#### 2. Precomputed Bucket Indices (sort: 14ms → 7ms)

**Problem:** During counting sort, extracting a bucket index requires bit shifting and masking across 32-byte scalar arrays. This is done twice per point per window (once for counting, once for scattering).

**Solution:** Merge bucket index extraction with the counting phase. Store all indices as compact uint16 arrays (2 bytes vs 32 bytes per scalar access). The scatter phase reads from the precomputed uint16 array instead of re-extracting from scalars.

#### 3. Sort-Reduce Batch Overlap (wall time: ~5ms savings)

**Problem:** CPU sort and GPU reduce are sequential — sort all windows, then dispatch all reduces. The CPU is idle during GPU reduce.

**Solution:** Split windows into 2 batches. Sort batch 1, dispatch its GPU reduce (non-blocking), then sort batch 2 while batch 1 reduce runs on GPU. The second batch's CPU sort (~4ms) overlaps with the first batch's GPU reduce (~20ms).

```
sort(windows 0..4)  →  dispatch reduce(0..4)  →  sort(windows 4..8)  →  dispatch reduce(4..8)
                        [GPU running]              [overlapped with GPU]
```

### Ideas Tried and Rejected

| Idea | Expected Gain | Actual Result | Why |
|------|--------------|---------------|-----|
| Branchless `point_add_mixed` | Fewer branches | Incorrect results | Metal compiler register allocation sensitivity |
| Dedicated `fp_sqr` (SOS) | Fewer multiplies | Slower | Register spilling on Metal GPU (tried twice) |
| Pre-gather points into sorted order | Better locality | No change | Compute-bound, not memory-bound |
| SoA scalar transposition | Cache-friendly sort | Slower | Transposition overhead > cache benefit |
| 1024 segments for bucketSum | 2x parallelism | 10.9ms vs 11.7ms | Scalar multiply dominates, not running sum |
| w=13 with GLV | Fewer buckets | 10-30x slower | M3 Pro GPU pathology (see below) |
| w=15 with GLV | Fewer buckets | 5x slower | Same GPU pathology, even with count-sorting |
| 64-bit limb fp_mul | 4 limbs vs 8 | Slower | Metal 64-bit multiply is 1/4 throughput of 32-bit |
| XYZZ coordinates | -1 squaring | Net +1 operation | Maintenance cost offsets the saving |
| Batch affine accumulation | Fewer multiplies | Break-even at 61 pts | Average bucket has 16 points; Jacobian 2.3x faster |
| Signed bucket decomposition | Half bucket count | 67-71ms (worse) | fp_neg per point in reduce hot loop + CPU overhead |
| Fused reduce+bucketSum kernel | Less launch overhead | Slower | Separate kernels have better occupancy |
| Max threadgroup (1024) for reduce | Better occupancy | 65ms vs 62ms | 256 is optimal for register pressure |

### M3 Pro GPU Pathology

The M3 Pro GPU exhibits catastrophic slowdowns at certain Pippenger window sizes due to SIMD thread divergence amplification:

- **Fast:** w=13 (8K buckets), w=16 (64K buckets)
- **Catastrophic (10-30x slower):** w=14, w=15, w=17
- **Slow (3-5x):** w=12, w=18

Root cause: when capping the reduce inner loop to 16 iterations, ALL window sizes become fast. The GPU handles variable-length loops poorly when some threads in a SIMD group run much longer than others, likely due to register spilling for long-running threads under high register pressure (~100+ registers for 256-bit field arithmetic). Count-sorting mitigates but does not fully resolve this for affected window sizes.

**Recommendation:** Always use w=16 for n > 32K points on M3 Pro.

### Theoretical Limits

The reduce kernel at 37ms is near the theoretical compute bound:

- **Mixed addition cost:** 7M + 4S = 11 fp_mul equivalent (optimal for Jacobian coordinates across all known coordinate systems)
- **fp_mul cost:** 8-iteration CIOS with 8x32-bit limbs = 136 multiply-accumulate operations per field multiply
- **Total per point:** 11 * 136 = 1,496 multiply-accumulate operations
- **Points processed:** 524K points * 8 windows = ~4.2M point additions
- **SIMD utilization:** Near-perfect after count-sorting (measured ~95%+ vs ~60% before)

Further improvement requires either fewer field operations per point addition (hard — 11 is the known minimum for all projective coordinate systems) or a fundamentally different MSM algorithm.

### Remaining Standalone MSM Opportunities

1. **GPU radix sort** (~5ms savings): Replace 7-9ms CPU counting sort with a 4-pass 4-bit GPU radix sort for 16-bit keys. Significant implementation effort.
2. **Merge GLV decompose with bucket extraction** (~1-2ms): Fuse the GPU scalar decomposition with CPU-side bucket index extraction to eliminate one buffer read pass.

---

## Barretenberg Prover Integration (14 Sessions)

After proving the Metal MSM viable in standalone benchmarks, 10 additional sessions integrated it into the Barretenberg UltraHonk prover and optimized every phase of the proving pipeline. The full technical report is in [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md).

### Integration Timeline

| Session | Focus | Key Result |
|---------|-------|------------|
| 5-6 | GPU integration, GLV in bb | GPU MSM replaces CPU MSM in OinkProver, Gemini, KZG |
| 7 | GPU counting sort, async overlap | Three Metal kernels; CPU/GPU overlap in Oink |
| 8 | Fused gather/reduce, GPU sparsity | Eliminates 1.15GB buffer per MSM; auto CPU fallback |
| 9 | Timing removal, threadgroup fix | CSM kernel exceeded 32KB limit (silent 3x regression) |
| 10 | Identity removal, Elastic MSM | Micro-opt kept; Elastic MSM rejected (<0.1% match) |
| 11 | PGO, grand_product fusion | Profile-guided optimization -46ms; step fusion -15ms |
| 12 | Horner evaluate, DontZeroMemory | CPU polynomial evaluate -32ms; zero-init skip -19ms |
| 13 | Precomputed VK, SRS bug fix | Revealed real prove time ~1s (not 2.3s); critical bug fix |
| 14 | PrecomputedCache, preallocation, batch Poseidon2 | Circuit building 252ms -> 93ms (-63%) |

### Optimizations by Category

**GPU Metal Kernels (8 shaders):**
- Bucket accumulation, GLV decompose, counting sort (histogram/prefix_sum/scatter)
- Fused gather+reduce kernel, GPU sort sparsity check, Metal prewarm + SRS caching

**CPU Compute:**
- PGO (-46ms sumcheck), Horner's method (-32ms polynomial evaluate)
- Grand product step fusion, factor_roots fusion, parallel Shplonk/Gemini

**Memory Allocation:**
- DontZeroMemory across sumcheck, wires, selectors, Gemini/Shplonk (-84ms total)
- Circuit builder preallocation (-143ms, eliminates ~20 reallocation doublings)
- Batch Poseidon2 permutation (-14ms, eliminates 1.76M function calls)

**Application-Level:**
- PrecomputedCache (-59ms/proof for repeated proofs)
- Precomputed VK path (-1,300ms by separating VK computation)
- skip_imbalance_check (-142ms, keeps benign 18% GPU imbalance)

### Rejected Optimizations (11 serious attempts)

| Idea | Result | Root Cause |
|------|--------|------------|
| ARM64 asm field multiply | 9% regression | Clang __int128 already near-optimal |
| Fused compute_batched | 2x slower | Cache locality kills fused pass |
| GPU compute_univariate | 3-6x slower | Metal 32-bit ALUs vs CPU 64-bit |
| GPU zero-copy partial_evaluate | SIGSEGV | Metal alignment requirements |
| Multi-MSM pipelining | Impossible | Fiat-Shamir sequential dependencies |
| XYZZ coordinates | 30% regression | 128 vs 96 bytes kills GPU occupancy |
| DontZeroMemory Gemini folds | Incorrect proofs | Reads beyond written range |
| Atomic thread pool | 3x regression/crash | Spin burns CPU; condvar races |
| Elastic MSM | <0.1% match rate | Not viable for random scalars |
| Full LTO | Linker crash | Requires >4GB memory |
| logderiv+z_perm batch | 12% slower | Batch overhead > launch savings |

### Critical Bug Fixes

1. **Lazy Montgomery [0, 2p) correctness** — BN254 field elements in [0, 2p) not [0, p); GPU assumed reduced. Fixed with fp_reduce() at boundary.
2. **Metal threadgroup overflow** — msm_compute_csm exceeded 32KB limit, silently disabled ALL GPU MSMs. Fixed by in-place prefix sum.
3. **SRS size calculation** — max_end_index() vs dyadic_size() caused crashes with precomputed VK.
4. **Silent GPU kernel deletion** — Accidental removal of msm_reduce_cooperative during revert. Added defensive logging.

### Optimization Floor

All phases within 15-35% of theoretical hardware limits:

| Component | Current | Theoretical Min | Gap |
|---|---:|---:|---:|
| GPU MSMs (total) | 514ms | ~400ms | 22% |
| Sumcheck | 145ms | ~120ms | 17% |
| create_circuit | 93ms | ~60ms | 35% |
| Shplonk | 62ms | ~50ms | 19% |

**Key lesson:** On Apple Silicon, Clang + LLVM generates near-optimal code for 256-bit field arithmetic. Assembly, SIMD (NEON), and GPU approaches all fail for field math. Gains come from algorithmic changes, memory layout, and GPU offloading of inherently parallel operations (MSM bucket accumulation).

---

## Memory Optimization: Fitting Root Rollup in 18 GiB

### Problem

The Aztec root rollup circuit (13M gates, dyadic 2^24) is the final epoch proof that settles on Ethereum L1. Standard UltraHonk proving requires ~29 GiB peak memory during sumcheck — far exceeding the 18 GiB available on Apple M3 Pro.

The bottleneck: UltraHonk allocates 36 polynomials (28 precomputed + 8 witness), each 512 MiB at 2^24. During sumcheck round 1, all 36 source polys (18 GiB) coexist with 41 half-size partial evaluation destination polys (10 GiB), plus the SRS (1 GiB).

### Solution: Disk-Backed Streaming Proving

Six optimizations, activated by the `--slow_low_memory` flag, reduce peak memory from ~29 GiB to ~10 GiB:

#### 1. Streaming Sumcheck (Option C)

All sumcheck rounds stream polynomials from disk instead of keeping them in memory:

- **Before sumcheck:** Serialize all `get_all()` polys (36 polys) to disk, free from memory
- **compute_univariate:** Memory-map all 42 polys once, process in 2M-row chunks via pointer arithmetic. Peak: ~1.3 GiB
- **partial_evaluate:** Memory-map each source poly one at a time, evaluate to half-size, write back to same path. Peak: 1 source + 1 dest poly
- **After all rounds:** Load tiny (1-element) final polys for evaluation extraction

#### 2. Progressive Oink Memory Management (Option D)

Calls OinkProver rounds individually, freeing precomputed polynomials between rounds:

1. **Before Oink:** Serialize 28 precomputed polys to disk (immutable, safe to serialize early)
2. **Free 11 unused polys** after serialization (q_l, q_4, q_arith, q_delta_range, q_elliptic, etc.)
3. **After log-derivative round:** Free q_lookup, table_1..4, q_m, q_c, q_r, q_o (9 polys)
4. **After grand product round:** Free sigma_1..4, id_1..4 (8 polys)
5. **After Oink:** Serialize witness polys (post-Oink, with correct w_4/lookup_inverses/z_perm)

**Critical insight:** Witness polys MUST be serialized AFTER Oink because Oink modifies w_4 (RAM/ROM records), computes lookup_inverses, and computes z_perm. Precomputed polys are immutable and safe to serialize before Oink.

#### 3. Memory-Mapped Polynomial I/O

Replaced per-chunk `fread` with `mmap` for zero-copy disk access:

- `Polynomial::mmap_from_file()` opens serialized poly, mmaps entire file, returns Polynomial backed by mmap'd memory with `MADV_SEQUENTIAL` hint
- OS handles page faults, prefetch, and eviction transparently
- Chunk size increased from 2^20 to 2^21 (fewer iterations)

#### 4. Memory-Mapped SRS

SRS points written to a native-format temp file and mmap'd instead of stored in `std::vector`:

- After deserialization from .dat (big-endian + Montgomery conversion), writes native format to temp file
- `MAP_PRIVATE` + `MADV_RANDOM` (SRS accessed semi-randomly during Pippenger MSM)
- Pages are OS-evictable under memory pressure (file-backed vs anonymous heap)

**Note:** The .dat file cannot be mmap'd directly because it uses big-endian encoding + Montgomery conversion. A per-process native cache file is required.

#### 5. File-Backed Polynomial Allocation (Discovered)

When `slow_low_memory=true`, the existing `BackingMemory::allocate()` creates mmap'd temp files for ALL polynomial allocations during ProverInstance construction. At 2^24, this means 41 polys × 512 MiB = 20.5 GiB is virtual (file-backed), not physical. The OS transparently manages page eviction.

#### 6. Disk-Backed PCS (Gemini/Shplonk)

`PolynomialBatcher::compute_batched_from_disk()` memory-maps polys one at a time for batching. Each poly goes out of scope (munmap'd) before the next is loaded. Peak: 1 poly (512 MiB) + accumulators (1 GiB).

### Benchmark Results

**Parity-base circuit (2.27M gates, 2^22) — controlled comparison:**

| Mode | Time | Peak RSS | Peak Footprint |
|------|------|----------|----------------|
| Normal | 12.2s | 5.1 GiB | 8.4 GiB |
| Low-memory (all opts) | 31.6s | 5.4 GiB | 2.0 GiB |

**Root rollup circuit (13M gates, 2^24) — the real target:**

| Metric | Value |
|--------|-------|
| Total time | 306.8s (~5.1 min) |
| Peak RSS | **10.36 GiB** |
| Peak memory footprint | **8.07 GiB** |
| Circuit construction | 39.8s |
| Page faults | 7.5M (disk-backed I/O) |

**Result: The 13M-gate root rollup fits comfortably in 18 GiB with ~7.6 GiB headroom.** Without these optimizations, proving would require ~29 GiB and fail with OOM.

### Memory Trajectory at 2^24

```
Phase                                    Memory
─────────────────────────────────────────────────
Circuit construction                     10.4 GiB (file-backed polys, evictable)
After Oink frees precomputed polys       ~4 GiB
Serialize witness polys to disk          ~3 GiB
Streaming sumcheck (all 24 rounds)       ~1.3 GiB (flat, no partial eval spike)
PCS: stream from disk, 1 poly at a time  ~1.5 GiB
```

### Tradeoffs

- **2.6x slower** at 2^22 (31.6s vs 12.2s) — acceptable for a flag that explicitly opts in to memory-constrained proving
- **7.5M page faults** at 2^24 — heavy disk I/O, but M3 Pro SSD delivers ~7 GB/s
- **GPU disabled** in low-memory mode — CPU-only MSMs avoid a known bucket imbalance bug at n≥2^22
- **~18 GiB temp disk space** required for serialized polynomials

### Files Modified

| File | Changes |
|------|---------|
| `polynomial.hpp/cpp` | `serialize_to_file`, `deserialize_from_file`, `deserialize_range_from_file`, `mmap_from_file` |
| `sumcheck.hpp` | `streaming_compute_univariate`, `streaming_partial_eval_to_disk`, `accumulate_edges` (chunk support) |
| `sumcheck_round.hpp` | `accumulate_edges` with chunk_start/chunk_end, `finalize_univariate` |
| `ultra_prover.cpp` | `construct_proof_low_memory()` — progressive Oink, serialize/free, streaming sumcheck |
| `commitment_key.hpp` | Disable GPU MSM when `slow_low_memory` |
| `mem_bn254_crs_factory.cpp` | mmap-backed SRS storage |
| `gemini.hpp` | `compute_batched_from_disk()` — disk-backed PCS batching |

### Key Technical Decisions

1. **mmap vs fread:** mmap is 32% faster for streaming sumcheck (47s → 32s at 2^22). OS prefetch with `MADV_SEQUENTIAL` eliminates per-chunk syscall overhead.

2. **MAP_PRIVATE for SRS:** Although SRS is read-only, `MAP_PRIVATE` allows the `std::span` type to be non-const. Any writes would COW but never occur in practice.

3. **AssertMode::WARN for benchmarking:** The root rollup circuit verifies two checkpoint rollup proofs internally. Real checkpoint proofs only exist transiently during Aztec's prover pipeline — they're never exposed via any public API. Used barretenberg's built-in `AssertMode::WARN` to suppress witness-validity assertions during benchmarking. Circuit size and proving workload are identical regardless of whether inner proofs are real.

4. **No lazy ProverInstance needed:** The existing `BackingMemory::FileBackedData` infrastructure already creates file-backed mmap allocations for all polynomials when `slow_low_memory=true`. The OS manages paging transparently, achieving the same effect as lazy allocation without code restructuring.
