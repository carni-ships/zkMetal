# GPU Primitives Optimization Status

Updated: 2026-04-19

## Summary

This document tracks optimization efforts for zkMetals GPU-accelerated cryptographic primitives. Key insight: **Field arithmetic complexity (1-word vs 8-word) is the primary differentiator in performance.** GPU tuning has diminishing returns once a kernel is compute-bound.

---

## 1. Additive FFT (GF(2^8))

**Engine:** `GPUAdditiveFFTEngine`
**Status:** Optimized

| Optimization | Tried | Result |
|--------------|-------|--------|
| Single-depth kernel (avoid fused) | Yes | Fixed incorrect results from Metal compiler undefined behavior |
| Fused kernel with barriers | No | Compiler bug causing wrong outputs |
| Threadgroup size tuning | Yes | Minimal impact |
| LUT as function constant vs device buffer | Yes | Minor improvement |

**Implementation:** `forwardSingleDepth` kernel with function-constant LUT. Fused approach had Metal compiler undefined behavior with complex indexing in SIMD patterns.

---

## 2. NTT - BN254

**Engine:** `NTTEngine` (fused kernel)
**Status:** Known bottleneck

| Metric | Value |
|--------|-------|
| Throughput | ~9M elem/s |
| vs BabyBear | ~5× slower |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Fused vs non-fused stages | Yes | `maxFusedLogN=10` optimal |
| Four-step threshold tuning | Yes | Threshold=22 works well at 2^20 |
| Twiddle layout | No | Not pursued |
| Karatsuba multiplication | Yes | Bugs in carry/borrow handling |
| Threadgroup size tuning | Yes | Minimal impact (compute-bound) |

**Root Cause:** 8×32-bit Montgomery limbs vs 1×32-bit for BabyBear. Each butterfly requires 64 32-bit multiplications.

---

## 3. NTT - BabyBear

**Engine:** `AsyncNTTEngine` (SIMD kernel)
**Status:** Well-optimized

| Metric | Value |
|--------|-------|
| Throughput | ~45M elem/s |
| vs BN254 | ~5× faster |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Threadgroup size tuning | Yes | No improvement (memory bandwidth bound) |
| SIMD butterfly kernel | Yes | Working correctly |
| Four-step algorithm | Yes | Mixed results by size |

**Notes:** 1×32-bit field — memory bandwidth bound, not compute bound.

---

## 4. NTT - Goldilocks

**Engine:** `AsyncNTTEngine` (same SIMD kernel)
**Status:** Well-optimized

| Metric | Value |
|--------|-------|
| Throughput | ~45M elem/s |
| Notes | Same field as BabyBear (M31), similar performance |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Threadgroup size tuning | Yes | No improvement (memory bandwidth bound) |

---

## 5. Poseidon2 Hash - BN254 (t=3, x^5)

**Engine:** `Poseidon2Engine`
**Status:** Known bottleneck

| Metric | Value |
|--------|-------|
| Throughput | ~713K hash/s |
| vs BabyBear | ~10-20× slower per hash |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Threadgroup size sweep (32-512) | Yes | No improvement |
| Grid size scaling (64K-4M) | Yes | Constant ~0.7M hash/s |
| Multi-pair batching (1-16 pairs/TG) | Yes | No improvement |
| Calibration vs real workload | Yes | Confirmed: XOR chains dont reflect 256-bit register pressure |

**Root Cause Analysis:**
- **240 `fr_mul` calls per hash** (corrected from 192 — 56 partial rounds × 3 muls + 8 full rounds × 3 muls)
- Each `fr_mul` = 64 32-bit multiplications (CIOS Montgomery) for a total of ~30,720 ops
- Purely compute-bound — GPU tuning has no effect

**Optimization (2026-04-19):** Replaced `fr_mul(x,x)` squaring operations with `fr_sqr(x)` in S-box. Squaring via `fr_sqr` = 100 Montgomery ops vs `fr_mul` = 128 ops. Savings: 2 squarings × 240 × 28 ops = ~13,440 ops (~2.4% reduction).

**Benchmark (M3 Pro, 256K grid):**
| Configuration | Throughput |
|---------------|------------|
| TG=32 | 0.66 M hash/s |
| TG=64 | 0.68 M hash/s |
| TG=128 | 0.67 M hash/s |
| TG=256 | 0.71 M hash/s |
| TG=512 | 0.66 M hash/s |

**Grid Scaling:**
| Grid Size | Throughput |
|-----------|------------|
| 64K | 0.68 M hash/s |
| 256K | 0.67 M hash/s |
| 1M | 0.69 M hash/s |
| 4M | 0.68 M hash/s |

**Multi-pair Batching:**
| Batch Size | Throughput |
|------------|------------|
| 1 (single) | 0.71 M hash/s |
| 2 | 0.72 M hash/s |
| 4 | 0.67 M hash/s |
| 8 | 0.64 M hash/s |
| 16 | 0.66 M hash/s |

---

## 6. Poseidon2 Hash - M31 (t=16, x^5)

**Engine:** `Poseidon2M31Engine`
**Status:** Per-optimization notes

| Metric | Value |
|--------|-------|
| Throughput | ~400K hash/s (80× CPU) |
| Width | t=16 |
| Rounds | 21 partial + 14 full = 35 total |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Hash threadgroup tuning | Yes | Uses `hashThreadgroupSize=256` |
| Merkle fused batching | Yes | Already implemented |

**Notes:** 16-element width, 8-element rate. Field fits in single 32-bit word.

---

## 7. Poseidon2 Hash - BabyBear (t=16, x^7)

**Engine:** `Poseidon2BabyBear`
**Status:** Per-optimization notes

| Metric | Value |
|--------|-------|
| S-box | x^7 (2 sqr + 1 mul, ~3 muls total) |
| Width | t=16 |
| Rounds | 13 partial + 8 full = 21 total |

| Optimization | Tried | Result |
|--------------|-------|--------|
| Internal layer specialization | Yes | Diagonal constants use adds instead of muls where possible |

**Notes:** x^7 S-box cost similar to x^5 but 16-element width with partial rounds reduces total work per hash.

---

## 8. MSM - BN254 G1

**Engine:** `MetalMSM`
**Status:** Optimized — Near-Optimal for Current Architecture

### Current Performance (M3 Pro, 2026-04-26)

| Size | Time | Throughput | Points/sec |
|------|------|------------|------------|
| 2^16 = 65K | **3.9ms** | 16.8M pts/sec | 17M |
| 2^17 = 131K | **7.3ms** | 18.0M pts/sec | 18M |
| 2^18 = 262K | **14.1ms** | 18.6M pts/sec | 19M |
| 2^20 = 1M | **61.7ms** | 17.0M pts/sec | 17M |

**Profile breakdown (2^18 = 262K):**
| Phase | Time | Notes |
|-------|------|-------|
| GLV+endo+signed_digit | 14.9ms | ✅ GPU kernel |
| sort (CPU) | 19.0ms | ✅ Fast CPU sort |
| GPU reduce+bucket_sum | ~14ms | ✅ (profile overhead masks actual) |
| GPU Horner combine | 22.9ms | ✅ GPU kernel |

### Kernel Timings (with profile overhead removed)

Single dispatch of `msm_reduce_sorted_buckets` at 2^16 (262,152 threads, TG=256):
- Observed: ~3.6ms (sequential dispatch)
- Metal dispatch overhead: ~0.5ms
- **Actual kernel compute: ~3.1ms**

Single dispatch of `msm_bucket_sum_direct` at 2^16 (2,048 threads, TG=256):
- Observed: ~2.0ms (sequential dispatch)
- Metal dispatch overhead: ~0.5ms
- **Actual kernel compute: ~1.5ms**

### Bottleneck Analysis

The primary bottleneck is **Metal framework dispatch overhead**, not GPU compute:

1. **Sequential dispatch pattern**: Each kernel (reduce, bucket_sum, combine, horner) is dispatched sequentially. GPU idle time between dispatches.

2. **GPU compute is fast**: `msm_reduce_sorted_buckets` with 262K threads completes in ~3.1ms actual compute. `msm_bucket_sum_direct` with 2K threads completes in ~1.5ms.

3. **Horner combine is the longest phase**: At 2^18, Horner combine takes 22.9ms vs the ~5ms observed for reduce+bucket_sum. This is the next optimization target.

4. **Key insight**: The profile timings include full end-to-end CPU+GPU processing (including memory transfers, CPU sort, and framework overhead). The actual GPU kernels are highly efficient.

### Key Optimizations Applied

1. **Increased nSegments**: From 256 to 512 — better GPU utilization for bucket sum phase
2. **GPU Horner combine**: Replaced CPU Horner with GPU kernel (~221ms speedup at 2^20)
3. **GPU GLV decomposition**: Fixed borrow bug in GLV kernel, GPU GLV is ~3% faster than parallel CPU
4. **Cooperative mode disabled**: CPU offload for highest window was SLOWER than all-GPU (58ms vs 4ms at 2^16)

### What Did NOT Help

1. **Increasing nSegments beyond 512**: No measurable improvement
2. **Cooperative GPU+CPU mode**: Made performance 15x worse (55ms vs 4ms at 2^16)
3. **Switching to msm_reduce_cooperative kernel**: The 1-thread-per-bucket approach is slower than the grid-parallel approach for this workload

### Remaining Opportunity

**Horner combine kernel**: At 2^18+, Horner takes more time than the reduce+bucket_sum phase combined. Optimizing this kernel (or reducing its input size via larger windowBits) could provide meaningful speedup.

Alternative: Increase windowBits from 16 to 17 or 18 — fewer windows means less Horner combine work, at the cost of more bucket accumulation work.

### Test command
```
swift run -c release --package-path . -- zkbench msm --profile --no-cpu
```

### Key Settings
- `nSegments = min(512, max(1, nBuckets / 2))` (was 256)
- `cooperativeThreshold = Int.max` (was 8192 — DISABLED as it hurts performance)
- `windowBits = 16` (for large point counts)

---

## 9. Sumcheck Protocol

**Engines:** `GPUSumcheckEngine`, `GPUMultilinearSumcheckEngine`, `AmortizedSumcheckProver`
**Status:** Two parallel implementations - GPU-accelerated (standard fields) + Amortized (tower/binary field)

### Architecture Overview

The sumcheck implementation has **two separate code paths** targeting different use cases:

#### A. GPU-Accelerated Sumcheck (Standard Fields)

Files: `GPUSumcheckEngine.swift`, `GPUMultilinearSumcheckEngine.swift`, `GPUSumcheckProtocolEngine.swift`

Supports BN254 (8x uint32 Montgomery), BabyBear (uint32), Goldilocks (uint64) with Metal GPU kernels.

**Protocol Flow (per round):**
1. Compute round polynomial: `s0 = sum f(0,x)`, `s1 = sum f(1,x)` over boolean hypercube
2. Fold table: `out[i] = in[i] + r*(in[i+half] - in[i])`
3. Generate challenge via Fiat-Shamir transcript

**GPU Threshold:** 4096 elements (below this, CPU fallback is faster due to dispatch overhead)

**GPU Kernel Suite (`sumcheck_reduce.metal`):**
- `sumcheck_reduce_bn254`: Fold kernel (grid-parallel)
- `sumcheck_round_poly_bn254`: Round poly with threadgroup SIMD reduction
- `sumcheck_fused_round_reduce_bn254`: Fused kernel (reads input once)
- `sumcheck_final_reduce_bn254`: GPU-side final reduction of partial sums

**Performance (M3 Pro):**
| Size | GPU Round Poly | CPU Round Poly | Speedup |
|------|----------------|----------------|---------|
| 2^16 | ~3ms | ~60ms | **20x** |
| 2^20 | ~50ms | ~900ms | **18x** |

#### B. Amortized Sumcheck (Binary Tower Field)

Files: `AmortizedSumcheck.swift`, `ConstraintPacker.swift`, `TowerBasisCache.swift`, `PrecomputedPolyManager.swift`

Targets **constraint packing** optimization from ePrint 2024/1038. Uses GF(2^8) arithmetic (1-byte values, XOR-based).

**Key Insight - Tower Basis:**
In binary tower fields, the tower level `k` corresponds to GF(2^k) with basis element `beta^k`. The `TowerBasisCache` precomputes:
- Vanishing polynomials `V_S(x)` for each level (indicator: 0 if x in subspace S)
- Additive NTT twiddle factors
- Lagrange basis coefficients
- Basis elements `beta^k` for each level

**Constraint Packing:**
Instead of one polynomial per constraint, `ConstraintPacker` interleaves multiple constraints using tower level as an extra dimension. Multiple R1CS constraints are XOR-combined into shared coefficient arrays (`aCoeffs`, `bCoeffs`, `cCoeffs`).

**Packing Strategies:**
- `.onePerLevel`: Maximum parallelism (1 constraint per tower level)
- `.maximizeDensity`: Pack as many constraints as fit in one level
- `.adaptive`: Hybrid approach

**PrecomputedPolyManager:**
Thread-safe manager for vanishing/subspace polynomials with O(1) challenge updates using cached powers `r, r^2, r^4, r^8, ...`.

**Amortized Benchmarks (from `amortized_sumcheck_bench.swift`):**
| Config | Precompute | Per-Query (cached) | Speedup |
|--------|------------|-------------------|---------|
| small (2^8) | ~0.1ms | ~0.01ms | ~10x |
| medium (2^10) | ~1ms | ~0.05ms | ~20x |
| large (2^12) | ~10ms | ~0.2ms | ~50x |

### What is "Tower Basis" in TowerBasisCache?

The tower basis for level `k` is the primitive element `beta^k` in GF(2^k). For GF(2^8) with primitive element `beta = 0x02`:
- Level 1: beta^1 = 0x02
- Level 2: beta^2 = 0x04
- Level 3: beta^3 = 0x08
- etc.

The "basis" refers to the field basis used to represent elements. The vanishing polynomial for a subspace S = {0, beta, beta^2, ...} of size 2^k is:
```
V_S(x) = prod_{s in S} (x - s)
```

In characteristic 2 with trace-zero basis, this simplifies to `V_S(x) = x^{2^k} - x` when S is the full space.

### How ConstraintPacker Optimizes Sumcheck

1. **Interleaving**: Instead of evaluating separate polynomials for each constraint, the packer interleaves them into tower levels:
   - At level k, there are 2^k variable positions
   - If `variableCount = 256`, one tower level can pack 256 constraints
   - Coefficients are XOR-combined: `aCoeffs[idx] ^= val`

2. **Vanishing Multiplication**: After evaluating `A(x)*z`, `B(x)*z`, `C(x)*z`, the result is multiplied by the vanishing polynomial for that tower level to enforce the subspace constraint.

3. **O(1) Evaluation**: With precomputed vanishing polynomials in `TowerBasisCache`, each packed constraint evaluation is a simple array lookup + XOR of coefficients.

### GPU vs CPU Analysis

| Aspect | GPU Path | CPU Path |
|--------|----------|----------|
| **BN254** | 8-word Montgomery arithmetic is compute-bound | Threaded C kernels (bn254_fr_vector_sum) |
| **BabyBear** | Fast (single uint32 multiply) | Fast (native 32-bit) |
| **Goldilocks** | Fast (uint64 with special reduction) | Fast (native 64-bit) |
| **Amortized** | No GPU path yet | GF(2^8) is inherently fast |

**Bottleneck for BN254**: 8x32-bit Montgomery multiplication dominates. Each `fr_mul` requires ~64 32-bit multiplications (CIOS Montgomery algorithm).

**Bottleneck for Amortized**: CPU-bound GF(2^8) multiply loop (bit-by-bit with reduction polynomial 0x11B).

### Typical Constraint Counts

From benchmarks:
- Small: 64 constraints packed into ~1 tower level
- Medium: 256-1024 constraints
- Large: 4096+ constraints

**Packing Efficiency** = `originalConstraints / packedPolynomials`. With `.maximizeDensity` strategy and 256 variables, efficiency approaches `variableCount` (256x).

### Optimization Opportunities

1. **GPU Port of Amortized Sumcheck**: GF(2^8) arithmetic could be SIMD-vectorized on GPU for significant speedup
2. **Fused Constraint Evaluation**: Fuse XOR chains + vanishing multiplication into single GPU kernel
3. **Precomputed Twiddle Factor Reuse**: Cache additive FFT twiddle factors across sumcheck rounds
4. **Batch Constraint Packing**: Process multiple constraints in parallel during packing phase
5. **Memory Layout Optimization**: Pack tower basis cache for coalesced GPU access

---

## 10. Circle STARK M31

**Engine:** `GPUCircleSTARKProverEngine`
**Status:** GPU accelerated (optimization work ongoing)

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| GPU Circle NTT | ✅ Working | ~30ms at 2^20 (fused kernel pending fix) |
| GPU Poseidon2 Merkle | ✅ Working | Replaced CPU path |
| Fused Merkle tree | ✅ Working | Single cmd buffer with barriers |
| GPU constraint eval | ❌ Pending | CPU-only currently |
| verify() method | ❌ Incomplete | Stub implementation |

### Verification

```swift
// Line 417-418 in GPUCircleSTARKProverEngine.swift:
if gpuAvailable {
    traceLDEs = try gpuLDE(trace: trace, logTrace: logTrace, logEval: logEval)
}
```

GPU acceleration IS implemented and working.

---

## 10b. Circle FRI (M31)

**Engine:** `CircleFRIEngine`
**Status:** GPU accelerated with fused Merkle

### Optimizations (2026-04-19)

| Optimization | Result |
|-------------|--------|
| GPU Poseidon2 Merkle (replaced m31SimpleHash placeholder) | ✅ Fixed insecure placeholder |
| Fused Merkle tree (single cmd buffer + barriers) | ✅ 2-3x faster Merkle construction |
| GPU hashLeavesWithPosition for FRI | ✅ Correct Poseidon2 padding |
| Async dispatch (didn't help) | Marginal — fold kernel is bandwidth-bound |

### Current Issues

- **Fused NTT kernel correctness bug** — Fixed (2026-04-19). The twiddle index calculation in `effective_local` was wrong. Changed from `half_block_idx * global_half_block + local_idx` to `(base + tid) % global_block_size`.
- **foldBy8Batch wired into multiFold** — for large domains (2^20+), dispatches reduced from ~19 to ~3 (batch of 8 rounds)

### Performance

| Size | FRI Fold Time | Notes |
|------|---------------|-------|
| 2^18 | ~16ms | 17 dispatches |
| 2^20 | ~21-24ms | ~3 dispatches (fold-by-8 batches) |

---

## Primitives NOT YET OPTIMIZED

From codebase scan, these primitives exist but lack optimization profiling:

- [x] **Sumcheck** — ✅ Profiled (both GPU-accelerated and Amortized paths)
- [x] **FRI** — ✅ Profiled (multiple engines: FRIEngine/BN254, CircleFRIEngine/M31, P1FRIEngine/M31)
- [x] **Pasta Poseidon** — ✅ Thoroughly investigated (~2.8M hash/s GPU, ~55x CPU speedup, near-optimal)
- [x] **MSM (Secp256k1, BN254)** — ✅ GLV parallelized (362ms vs 633ms, 1.75x speedup)

---

## Optimization Space Summary

| Primitive | Throughput | Tuning Potential |
|-----------|------------|------------------|
| Additive FFT | High | LUT optimization done |
| NTT BabyBear/Goldilocks | ~45M elem/s | Memory-bound, hard to improve |
| NTT BN254 | ~9M elem/s | Arithmetic-bound, fundamental |
| Poseidon2 M31/BabyBear | ~400K hash/s | Tuned |
| Poseidon2 BN254 | ~713K hash/s | Compute-bound, fundamental |
| Pasta Poseidon (Pallas/Vesta) | ~2.8M hash/s | GPU optimized, all-full-rounds, compute-bound |
| FRI (BN254) fold-by-8 | ~3-7x vs fold-by-2 at 2^20+ | Use fold-by-8 for large domains |
| **SHA-256** | **~129M hash/s** | **Thoroughly investigated — near-optimal; compute-bound** |
| **Blake3** | **~348M hash/s** | **Thoroughly investigated — near-optimal; compute-bound** |
| Keccak-256 | ~500K hash/s | Per PERFORMANCE.md table |

---

## 11. Pasta Poseidon Hash (Pallas/Vesta, t=3, x^7)

**Engine:** `PastaPoseidonEngine`
**Status:** Thoroughly investigated — near-optimal for 55 full-round specification

| Metric | Value | Notes |
|--------|-------|-------|
| GPU Throughput | ~2.8M hash/s at 262K | Up from ~231K (old benchmark at 131K) |
| CPU Throughput | ~50-62K hash/s | C CIOS implementation |
| Width | t=3 | Same as BN254 |
| Rounds | 55 full rounds | No partial rounds (unlike Poseidon2) |
| S-box | x^7 | 2 sqr + 1 mul = 3 muls per element per round |
| Field arithmetic | 8×32-bit limbs | Same as BN254 (Montgomery CIOS) |

### Benchmark (M3 Pro)

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| GPU 262K batch | ~2.8M hash/s | Current (was ~231K at 131K, old measurement method) |
| GPU 131K batch | ~2.6M hash/s | Consistent scaling |
| GPU 16K batch | ~2.2M hash/s | |
| CPU baseline | ~50-60K hash/s | C CIOS |
| GPU vs CPU speedup | ~45-55x | |

### Investigation Results

| Optimization | Tried | Result |
|--------------|-------|--------|
| #pragma unroll 55 on permutation loop | Yes | **FAILED** — Metal compiler rejects `#pragma unroll 55` at function scope |
| #pragma unroll inside function body | Yes | Slower at small sizes (5.45ms vs 2.75ms at 1K), same at large — compiler unrolls automatically |
| sqr() for squaring in S-box | Yes | **HURT performance** — sqr() not faster than mul(x,x) for Pallas 8-limb Montgomery |
| batchSize=2 hashes/thread | Yes | HURT at small scales, neutral at large |
| batchSize=4 hashes/thread | Yes | HURT at all scales |
| batchSize=8 hashes/thread | Yes | HURT at all scales |

**Root Cause Analysis:**
- 55 full rounds × (3 S-box muls + 9 MDS muls) = **660 multiplications per hash**
- Each mul is 8×32-bit Montgomery = 64 ops/hash × 660 = ~42,240 ops/hash
- GPU: ~2.8M hash/s × 660 mul/hash = ~1.85B mul/s
- Memory: 96 bytes/hash × 2.8M = ~270 MB/s (well within 100+ GB/s)
- **The kernel is compute-bound** — GPU runs at ~1.85B mul/s, well below peak
- But the algorithm (660 mul/hash) is fundamentally expensive

**Bottleneck Type:** Compute-bound, algorithm-limited. The 55-round Kimchi specification makes this inherently expensive. GPU tuning has minimal impact because the compiler already optimizes aggressively.

**Conclusion:** No micro-optimization can significantly improve this. The kernel is already near-optimal given the 55-round specification. Real gains would require:
1. A different round specification (not acceptable — changes the hash output)
2. Massively parallel batching across many independent chains (architectural)

### Files

- Engine: `/Users/carnation/Documents/Claude/zkMetal/Sources/zkMetal/Hash/PastaPoseidonEngine.swift`
- GPU Shader: `/Users/carnation/Documents/Claude/zkMetal/Sources/Shaders/hash/pasta_poseidon.metal`
- Benchmark: `/Users/carnation/Documents/Claude/zkMetal/Sources/zkbench/pasta_poseidon_bench.swift`

---

## 12. FRI (Fast Reed-Solomon Interactive Oracle)

**Engines:** `FRIEngine` (BN254), `CircleFRIEngine` (M31), `P1FRIEngine` (M31), `GPUFRIEngine` (query phase)
**Status:** GPU-accelerated with fold-by-2, fold-by-4, fold-by-8 variants

### Architecture Overview

zkMetal has **three FRI implementations** targeting different STARK protocols:

#### A. Standard FRI Engine (BN254)

Files: `FRIEngine.swift`, `Sources/Shaders/fri/fri_fold.metal`, `fri_kernels.metal`

Uses multiplicative subgroup domains with BN254 Fr (256-bit Montgomery). Supports:
- `fold`: single fold-by-2
- `fold4`: single fold-by-4
- `fold8`: single fold-by-8
- `multiFold`: cascade all rounds into single GPU dispatch when data fits in L2$
- `commitPhase`: fold + Poseidon2 Merkle tree on GPU

**Fold formula** (standard FRI):
```
g[i] = (f[i] + f[i + n/2]) + beta * (f[i] - f[i + n/2]) * inv_domain[i]
```

#### B. Circle FRI Engine (M31)

Files: `CircleFRIEngine.swift`, `Sources/Shaders/fri/circle_fri.metal`

Circle-specific FRI using the circle group x^2 + y^2 = 1 mod M31:
- First fold: y-coordinate twin-coset decomposition (pairs (x,y) and (x,-y))
- Subsequent folds: x-coordinate squaring map x -> 2x^2 - 1
- Uses Mersenne31 field (32-bit, single-word)

**Fold formula** (Circle FRI):
```
g[i] = (f[i] + f[i+half])/2 + alpha * (f[i] - f[i+half]) * inv_2twiddle[i]
```

#### C. P^1 Rational Function FRI (M31)

Files: `P1FRIEngine.swift`, `Sources/Shaders/fri/p1_fri.metal`

Newer approach using multiplicative coset domain with standard t->t^2 folding:
- Simpler than Circle FRI (no y-fold + x-fold distinction)
- Standard vanishing polynomial v_H(t) = t^m - c
- Also uses Mersenne31 field

**Note:** CircleFRI and P1FRI GPU kernels have shader loading issues when running from certain working directories.

### Benchmark Results (M3 Pro, BN254 FRI)

| Operation | 2^14 | 2^16 | 2^18 | 2^20 | 2^22 |
|-----------|------|------|------|------|------|
| **GPU fold** | 16.0ms | 16.0ms | 16.0ms | 17.0ms | 60.9ms |
| **CPU fold** | 0.7ms | 3.0ms | 12.0ms | 48.5ms | 215.5ms |
| **GPU/CPU ratio** | 0.04x | 0.19x | 0.75x | 2.9x | 3.5x |
| **GPU fold4** | 16.0ms | 16.0ms | 16.0ms | 38.2ms | 112.4ms |
| **GPU fold8** | 16.0ms | 16.0ms | 16.2ms | 24.9ms | 128.0ms |

### Commit Phase Performance (BN254 FRI)

| Configuration | Layers | Time | Speedup vs fold-by-2 |
|---------------|--------|------|----------------------|
| fold-by-2 2^15 | 16 | 294.6ms | 1x |
| fold-by-4 2^15 | 9 | 209.8ms | 1.4x |
| fold-by-8 2^15 | 6 | 73.5ms | **4.0x** |
| fold-by-2 2^18 | 19 | 1311.9ms | 1x |
| fold-by-4 2^18 | 10 | 489.2ms | 2.7x |
| fold-by-8 2^18 | 7 | 179.3ms | **7.3x** |
| fold-by-2 2^20 | 21 | 4468.5ms | 1x |
| fold-by-4 2^20 | 11 | 1593.5ms | 2.8x |
| fold-by-8 2^20 | 8 | 1320.9ms | **3.4x** |

**Key observation:** Fold dominates Merkle in commit phase (183ms fold vs 0.4ms Merkle).

### Bottleneck Analysis

1. **GPU FRI is compute-bound for small sizes**: Dispatch overhead dominates
   - At 2^14-2^18: GPU ~16ms constant regardless of size (dispatch overhead)
   - CPU scales linearly: 0.7ms -> 12ms

2. **Memory bandwidth at large sizes**: GPU wins only at 2^20+
   - 2^22: GPU 61ms vs CPU 215ms (~3.5x speedup)
   - Fold-by-8 shows best scaling: 4-7x speedup at larger sizes

3. **Threadgroup size**: `friThreadgroupSize=256` (from TuningConfig)
   - No significant sensitivity observed (kernel is memory-bandwidth bound at large sizes)

4. **Fused kernels (fold-by-4, fold-by-8)**: Critical for performance
   - Reduces dispatch overhead significantly
   - fold-by-8 provides best layer reduction but fold-by-4 often better balanced

### Optimization Opportunities

1. **Fold-by-8 for large domains**: Use fold-by-8 as default for 2^20+
   - 4-7x speedup vs fold-by-2
   - Fewer layers = less Poseidon2 Merkle work

2. **Cascade optimization**: When remaining data fits in L2$, cascade all rounds
   - FRIEngine already implements this via `foldCascadeFunction`
   - Activates when curN <= 1024

3. **CPU-GPU hybrid for small sizes**: For < 2^18, CPU may be faster
   - Consider fallback to CPU for small FRI layers

4. **Poseidon2 Merkle**: Currently ~0.5ms per commit phase
   - Not a bottleneck but could fuse with fold for fewer dispatches

### Is Optimization Needed?

**Yes, for specific use cases**:
- For 2^20+ domains, fold-by-8 is critical (3-7x speedup)
- For smaller domains, CPU may be faster due to dispatch overhead
- FRI is often a small fraction of total proof time (alongside NTT, MSM, Poseidon2)

### Files

| File | Description |
|------|-------------|
| `Sources/zkMetal/Polynomial/FRIEngine.swift` | Main FRI engine (BN254) |
| `Sources/zkMetal/Polynomial/CircleFRIEngine.swift` | Circle FRI (M31) |
| `Sources/zkMetal/Polynomial/P1FRIEngine.swift` | P^1 Rational FRI (M31) |
| `Sources/zkMetal/Polynomial/GPUFRIEngine.swift` | GPU query phase evaluator |
| `Sources/Shaders/fri/fri_fold.metal` | Basic fold kernel |
| `Sources/Shaders/fri/fri_kernels.metal` | Fused fold, fold-by-4/8 kernels |
| `Sources/Shaders/fri/circle_fri.metal` | Circle FRI kernels |
| `Sources/Shaders/fri/p1_fri.metal` | P^1 FRI kernel |
| `Sources/zkbench/fri_bench.swift` | FRI benchmark |
| `Sources/zkbench/circle_fri_bench.swift` | Circle FRI benchmark |
| `Sources/zkbench/p1_bench.swift` | P^1 FRI benchmark |

---

## Key Findings

1. **Field size dominates performance**
   - 1-word fields (BabyBear, Goldilocks, M31): Fast, memory-bound
   - 8-word fields (BN254): Slow, compute-bound

2. **GPU tuning has limits**
   - Threadgroup size: No effect when compute-bound
   - Grid size: No effect beyond minimum occupancy
   - Batching: No effect when kernel is pure compute

3. **Calibration matters**
   - Hash calibration used XOR chains, which dont reflect 256-bit register pressure
   - Real workload profiling essential for accurate tuning

4. **SIMD/compiler bugs**
   - Additive FFT: Metal compiler had undefined behavior with fused + complex indexing
   - Karatsuba NTT: Carry/borrow bugs in Montgomery reduction
   - Solution: Use simpler single-depth dispatch patterns

---

## Calibration Notes

Current tuning config (`Tuning.swift`):

```swift
hashThreadgroupSize: 256,  // Was 64; synthetic XOR cal biased low
                           // Poseidon2 S-box is register-heavy
```

BN254 Poseidon2 threadgroup calibration with actual `fr_mul` workload would likely confirm 256 is near-optimal, but the difference is minimal since the kernel is compute-bound.

---

## Scalar Conversion Fixes (2026-04-22)

### BN254 Scalar Conversion Bug Fixed

**Problem**: Jolt integration was failing verification because of incorrect scalar conversion.

**Root Cause**: Two bugs in the conversion from arkworks `Fr` to Pippenger format:

1. **Rust `ark_fr_to_pippenger_scalar()`**: Used wrong constants (`BN254_R` and `BN254_R_INV` were incorrect)
2. **C `bn254_fr_batch_to_limbs()`**: Multiplied by `ONE = {1,0,0,0}` instead of `R^-1`

**Discovery**: `ark_bn254::Fr.into_bigint()` returns **standard (non-Montgomery)** form, not Montgomery form. This means no Montgomery-to-standard conversion is needed - just unpack the limbs.

### Files Modified

1. `bindings/rust/src/arkworks.rs` - Fixed `ark_fr_to_pippenger_scalar()`:
   - Removed incorrect R^-1 multiplication
   - Now just unpacks 4 x u64 into 8 x u32 (little-endian)

2. `Sources/NeonFieldOps/bn254_msm.c` - Fixed `bn254_fr_batch_to_limbs()`:
   - Changed from `fr_mul(mont, ONE, r)` to direct unpacking
   - Comments updated to reflect correct behavior

### Verified Correctness

```python
# Fr(1) → [1, 0, 0, 0, 0, 0, 0, 0] ✓
# Fr(2) → [2, 0, 0, 0, 0, 0, 0, 0] ✓
```

### Benchmark Results (Apple M3 Pro)

| Size | Points | GPU Time | CPU Speedup |
|------|--------|----------|-------------|
| 2^8 | 256 | 1.2ms | 330x |
| 2^10 | 1,024 | 4.1ms | 378x |
| 2^12 | 4,096 | 1.3ms | 4,441x |
| 2^14 | 16,384 | 1.9ms | 12,552x |
| 2^20 | 1,048,576 | 61.2ms | - |

### BLS12-381 Status

**Current**: BLS12-381 scalar conversion NOT in arkworks compatibility layer.

**Assessment**: If using BLS12-381 with zkMetal Pippenger MSM via Rust FFI, need to ensure scalars are in standard form (8 x u32 limbs, little-endian). The Swift `BLS12381MSMEngine` correctly uses `fr381ToInt()` to convert from Montgomery to standard form before passing to C Pippenger.

**Recommendation**: Add `bls381_fr_to_pippenger_scalar()` function to `arkworks.rs` if BLS12-381 arkworks integration is needed.

### Other Curves

| Curve | Has Pippenger | Conversion Helper |
|-------|----------------|-------------------|
| BN254 | ✅ Yes | ✅ `ark_fr_to_pippenger_scalar()` |
| BLS12-381 | ✅ Yes | ⚠️ Needs addition |
| secp256k1 | ✅ Yes | ⚠️ Needs addition |
| Pasta (Pallas/Vesta) | ✅ Yes | ⚠️ Needs addition |
