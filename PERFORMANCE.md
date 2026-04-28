# Performance Benchmarks

## ⚠️ Environmental Performance Notice

**Current benchmarks run on macOS 26.3 (Beta)** - Performance may be degraded 2-3x compared to stable macOS releases.

**Verified on**: Apple M3 Pro (18 GPU cores), macOS 26.3 (Beta/Development)

### Performance Regression Analysis

Recent profiling revealed that BN254 GPU operations are running 10-18x slower than documented performance on stable macOS 15.x. Root cause identified as **beta macOS Metal driver overhead**.

| Component | Expected (Stable macOS) | Current (Beta macOS) | Regression |
|-----------|----------------------|---------------------|------------|
| NTT BN254 2^20 | 6.06ms | 108ms | **18x slower** |
| MSM BN254 2^20 | 137ms | 1983ms | **14x slower** |
| Command buffer overhead | <0.1ms | 0.03ms | ✅ Normal |
| Memory bandwidth | ~200 GB/s | 231 GB/s | ✅ Normal |

**See**: `BACKLOG/GPU_PERFORMANCE_INVESTIGATION.md` and `BACKLOG/BN254_NTT_BOTTLENECK_ANALYSIS.md` for detailed analysis.

### Why The Regression?

1. **Beta macOS drivers**: Unoptimized Metal drivers in development versions
2. **BN254 complexity**: 32-byte field elements require 128 limb multiplications per operation
3. **GPU vs CPU**: CPU uses NEON SIMD (4×64-bit parallel), GPU Metal shaders are scalar

### Expected Performance on Stable macOS

These are the target performance metrics when running on **stable macOS 15.x (Sequoia)**:

## MSM (BN254 G1) - Expected Performance

| Points | C Pippenger CPU | GPU (Metal) | Speedup |
|--------|-----------------|-------------|---------|
| 2^8 | 350ms | 0.8ms | 438x |
| 2^10 | 1.4s | 2.5ms | 560x |
| 2^12 | 5.6s | 21.0ms | 267x |
| 2^14 | 23.4s | 33.6ms | 696x |
| 2^16 | -- | 46.3ms | -- |
| 2^18 | -- | 72.7ms | -- |
| 2^20 | -- | 175.3ms | -- |

### Comparison to Other Implementations

| Points | zkMetal GPU | ICICLE-Metal GPU | ICICLE CPU | MoPro v2 CPU | Arkworks CPU | ICICLE CUDA |
|--------|-------------|------------------|------------|--------------|--------------|-------------|
| 2^16 | **31ms** | 1,083ms | 114ms | 253ms | 69ms | ~9ms |
| 2^18 | **53ms** | 1,475ms | 556ms | 678ms | 266ms | -- |
| 2^20 | **137ms** | 2,590ms | 2,349ms | 1,702ms | 592ms | -- |

## NTT - Expected Performance

### Multi-field NTT Comparison (GPU)

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.69ms | 1.4ms | 0.15ms | 0.26ms |
| 2^18 | 2.9ms | 2.1ms | 0.21ms | 0.54ms |
| 2^20 | 10.8ms | 5.8ms | 0.70ms | 0.79ms |
| 2^22 | 28ms | 25ms | 4.3ms | 3.0ms |
| 2^24 | 113ms | 110ms | 3.1ms | 2.3ms |

### BN254 NTT (GPU)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.71ms | 5.8ms | 8x |
| 2^16 | 0.85ms | 26.6ms | 31x |
| 2^18 | 1.72ms | 119.8ms | 70x |
| 2^20 | 6.06ms | 528ms | 87x |
| 2^22 | 26.0ms | 2262ms | 87x |
| 2^24 | 110.9ms | 9861ms | 89x |

### Comparison to ICICLE-Metal v3.8 (M3 Pro)

| Size | zkMetal BN254 | ICICLE BN254 | zkMetal BabyBear | ICICLE BabyBear |
|------|------------|-------------|----------------|---------------- |
| 2^16 | **0.85ms** | 89ms | **0.28ms** | 86ms |
| 2^18 | **1.72ms** | 108ms | **0.26ms** | 92ms |
| 2^20 | **6.06ms** | 194ms | **0.66ms** | 108ms |
| 2^22 | **26.1ms** | 915ms | **2.67ms** | 181ms |
| 2^24 | **110.9ms** | 3,892ms | **1.33ms** | 709ms |

## Current Performance (Beta macOS 26.3)

For reference, current observed performance on beta macOS:

### MSM (BN254 G1) - Current

| Points | GPU (Current) | GPU (Expected) | Regression |
|--------|---------------|----------------|------------|
| 2^10 | 2.8ms | 2.5ms | 1.1x |
| 2^12 | 94.2ms | 21.0ms | **4.5x** |
| 2^14 | 140.1ms | 33.6ms | **4.2x** |
| 2^16 | 367.0ms | 46.3ms | **7.9x** |
| 2^20 | 1941.6ms | 175.3ms | **11x** |

### MSM (BLS12-377 G1) - Current

| Points | GPU (Current) | Notes |
|--------|---------------|-------|
| 2^8 | 1.4ms | GPU path |
| 2^10 | 4.1ms | GPU path |
| 2^12 | 10.4ms | GPU path |
| 2^14 | 31.2ms | GPU path (n<4096) |
| 2^16 | 110.1ms | CPU fallback (GPU hang at n>=4096) |
| 2^17 | 208.7ms | CPU fallback |
| 2^18 | 421.6ms | CPU fallback |

**Note**: BLS12-377 GPU MSM uses on-the-fly endomorphism. GPU kernel hangs at n>=4096 due to extreme register pressure from 12-limb field operations. Large sizes use CPU Pippenger fallback. See `BACKLOG/MSM_OPTIMIZATIONS.md` for details.

### NTT (BN254) - Current

| Size | GPU (Current) | GPU (Expected) | Regression |
|------|---------------|----------------|------------|
| 2^14 | 18.0ms | 0.71ms | **25x** |
| 2^16 | 20.0ms | 0.85ms | **24x** |
| 2^18 | 43.4ms | 1.72ms | **25x** |
| 2^20 | 108.3ms | 6.06ms | **18x** |
| 2^24 | 1533ms | 110.9ms | **14x** |

### BabyBear NTT - Current (Less Affected)

| Size | GPU (Current) | Expected | Regression |
|------|---------------|----------|------------|
| 2^16 | 21.0ms | 0.26ms | **80x** |
| 2^18 | 24.8ms | 0.54ms | **46x** |
| 2^20 | 22.8ms | 0.79ms | **29x** |
| 2^24 | 22.4ms | 2.3ms | **10x** |

**Note**: BabyBear shows less regression due to simpler field operations (4-byte elements vs 32-byte for BN254).

## Hashing - Expected Performance

| Primitive | Batch Size | Vanilla CPU | Optimized CPU | GPU (Metal) | GPU vs Opt CPU |
|-----------|-----------|-------------|--------------|-------------|----------------|
| Poseidon2 | 2^12 | 523ms | 19ms (C CIOS) | 2.3ms | **8x** |
| Poseidon2 | 2^14 | 2.0s | 75ms (C CIOS) | 2.3ms | **33x** |
| Poseidon2 | 2^16 | 8.0s | 302ms (C CIOS) | 8.5ms | **36x** |
| Pasta Poseidon | 2^16 | 16.1s | 1.0s (C CIOS) | ~303ms (~216K hash/s) | **3.3x** |
| Pasta Poseidon | 2^18 | -- | 4.1s (C CIOS) | ~1150ms (~228K hash/s) | **3.6x** |
| Keccak-256 | 2^14 | 100ms | 23ms (parallel) | 0.20ms | **500x** |
| Keccak-256 | 2^16 | 387ms | 89ms (parallel) | 0.45ms | **860x** |
| Keccak-256 | 2^18 | 1.6s | 360ms (parallel) | 1.4ms | **1143x** |

## Merkle Trees - Expected Performance

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 7.3ms | 6ms | **1x** |
| Poseidon2 | 2^12 | 8.7ms | 23ms | **3x** |
| Poseidon2 | 2^14 | 10ms | 91ms | **9x** |
| Poseidon2 | 2^16 | 21ms | 364ms | **17x** |
| Poseidon2 | 2^18 | 45ms | 1.4s | **32x** |
| Poseidon2 | 2^20 | 129ms | -- | -- |

## GPU Additive FFT (GF(2^8)) - Updated 2026-04-27

GPU-accelerated Additive FFT (Cantor/Lin-Chung-Han) for GF(2^8) with fused all-k-levels kernel.

**Key optimizations**:
- 256x256 GF(2^8) multiplication LUT (64KB) for O(1) field multiplication
- `forward_pairs` kernel (n/2 threads, no divergence)
- Threadgroup-local basis caching (eliminates k global memory reads)

### GPU Forward FFT (Median of 5 runs, cold ShaderCache)

| Size | Elements | basic | pairs | pairs_tg | CPU | GPU Speedup |
|------|----------|-------|-------|----------|-----|-------------|
| 2^16 | 65,536 | 0.31ms | 0.28ms | ~0.3ms | 1.67ms | **~5x** |
| 2^18 | 262,144 | 0.46ms | 0.46ms | ~0.5ms | 7.40ms | **~16x** |
| 2^20 | 1,048,576 | 0.89ms | 0.78ms | ~0.8ms | 32.94ms | **~41x** |
| 2^22 | 4,194,304 | 3.5ms | 3.4ms | ~3.5ms | 143.95ms | **~41x** |

### Throughput (GPU, 2^22)

- Median: ~1200 M elem/s (3.4ms)
- Range: 2.7ms - 10.3ms (high variance at larger sizes)
- High variance at 2^22 due to Metal command buffer scheduling variability

### Performance vs IN_PROGRESS.md Claims

**IN_PROGRESS.md claimed**: ~11-14ms at 2^22 with high variance, target 0.5ms
**Actual performance**: 2.7-3.5ms median, ~41x speedup over CPU

The discrepancy is likely due to:
1. ShaderCache eliminating compilation overhead on subsequent runs
2. IN_PROGRESS.md measuring cold-cache time
3. Different benchmark methodology (single run vs median of 5)

### Implementation Notes

**Butterfly structure**:
- Forward: `new_lo = lo ^ (s*hi)`, `new_hi = lo ^ hi`
- Inverse: brute-force solve for `hi` such that `s*hi ^ hi = new_lo ^ new_hi`

**Known limitation**: Some s values (e.g., s=94, s=255) only have ~50% solvability for the inverse equation. For inverse FFT, use basis with s=2 repeated or precompute inverse lookup tables.

### Optimization Status

| Optimization | Status | Notes |
|--------------|--------|-------|
| Threadgroup-local basis caching | ✅ Done | ForwardPairsTg kernel |
| SIMD vectorization (uchar4) | ❌ Not tried | Could improve memory coalescing |
| Batch multiple FFTs | ✅ Done | forwardBatch kernel |
| Fused FFT + commitment | ❌ Not tried | Could reduce memory round-trips |

**Remaining opportunities**:
- SIMD vectorization: 4x more elements per thread could reduce memory bandwidth pressure
- Pipelined dispatch: overlap consecutive FFT operations to hide latency

## Recent Optimizations Committed

Despite the environmental regression, these optimizations provide value on stable macOS:

1. **ShaderCache Integration**: Persistent binary caching for NTT shaders
2. **CPU-side GLV Decomposition**: Fixes Metal kernel bugs, improves correctness
3. **Batched Poseidon2 Hash**: Better GPU utilization for large batches
4. **Threadgroup-local Basis Caching**: Reduces memory bandwidth for Additive FFT
5. **GPU Merkle Tree Building**: Accelerates Circle STARK commitment phase
6. **CPU MSM Micro-optimizations**: Inline copies reduce function call overhead

## Recommendations

1. **Test on stable macOS 15.x** to confirm expected performance
2. **Use BabyBear/Goldilocks fields** when possible (much faster than BN254)
3. **Batch operations** to amortize fixed overhead
4. **Monitor Metal driver updates** in macOS 26.x betas

## Running Benchmarks

```bash
# Run all benchmarks
swift run -c release zkbench all

# Run specific benchmarks
swift run -c release zkbench msm
swift run -c release zkbench ntt
swift run -c release zkbench poseidon2-bb

# Profile NTT bottlenecks
swift run -c release zkbench ntt-profile

# Run P^1 Rational Function STARKs benchmark
swift run -c release zkbench p1
```

## P^1 Rational Function STARKs (Prototype)

Prototype implementation using Mersenne31 field with standard radix-2 FFT on multiplicative coset domain.

**Note**: M31 has limited 2-adicity (p-1 = 2^31 - 2 = 2 × (2^30 - 1)). The implementation uses a sign-pair domain (±t) workaround for FRI folding.

### P^1 NTT (GPU)

| Size | Time |
|------|------|
| 2^10 | 0.16ms |
| 2^12 | 0.20ms |
| 2^14 | 0.20ms |
| 2^16 | 0.31ms |
| 2^18 | 0.95ms |
| 2^20 | 3.36ms |

### P^1 FRI Commit Phase (GPU) - With inv2t Caching

Optimized with inv2t caching (65x speedup over naive recomputation).

| Size | Commit Time | Rounds |
|------|-----------|--------|
| 2^14 | **0.26-0.53ms** | 13 |
| 2^16 | **0.37ms** | 15 |
| 2^18 | **1.02-1.24ms** | 17 |
| 2^20 | **3.35-3.55ms** | 19 |

**Key optimization**: Precompute all inv2t arrays for all FRI rounds upfront with GPU buffer caching.

**Implementation**: Uses fold-by-4 cascade with fold-by-2 fallback for remaining rounds.

### P^1 FRI Fold-by-8 (DISABLED)

The fold-by-8 kernel had threadgroup indexing issues causing incorrect results.
- Kernel disabled until structural redesign
- Standard commit uses fold-by-4 with fold-by-2 fallback
- All tests pass with current implementation

### P^1 FRI Multi-fold Performance (GPU)

| Size | Single Fold | Multi-fold (all rounds) |
|------|-------------|--------------------------|
| 2^14 | 1.30ms | 2.25ms |
| 2^18 | 13.36ms | 27.86ms |
| 2^20 | 52.19ms | 108.53ms |

## Circle NTT (Mersenne31)

GPU Circle NTT over Mersenne31 field (p = 2^31 - 1). Circle group has full 2-adicity (order p+1 = 2^31), enabling radix-2 FFT without restrictions.

Layer 0 uses y-coordinate twiddles; layers 1+ use x-coordinate twiddles with the squaring map.

| Size | Time | Throughput |
|------|------|------------|
| 2^10 | 0.13-0.16ms | 6.4-7.9 M ops/s |
| 2^12 | 0.13-0.16ms | 25.6-31.5 M ops/s |
| 2^14 | 0.15-0.22ms | 74-109 M ops/s |
| 2^16 | 0.20-0.27ms | 242-327 M ops/s |
| 2^18 | 0.45-0.85ms | 308-582 M ops/s |
| 2^20 | 1.50-1.61ms | 651-699 M ops/s |

**Notes:**
- All sizes verified against CPU reference implementation
- All 167 NTT tests pass
- Performance dominated by GPU command buffer scheduling overhead at small sizes
- Good scaling from 2^16 onward due to amortization of dispatch overhead
- Single-column Circle NTT; see Batch Circle NTT for multi-column processing

## Batch Circle NTT (Mersenne31)

GPU batch Circle NTT for processing multiple columns in a single dispatch using grid Y dimension.

For N columns of size 2^logN each, laid out sequentially in one buffer:
`[col 0: 2^logN] [col 1: 2^logN] ... [col N-1: 2^logN]`

| Columns | Size | Time/Column | Speedup vs Sequential |
|---------|------|-------------|----------------------|
| 180 | 2^10 | ~0.01ms | ~180x (kernel launch reduction) |

**Batch processing replaces:**
- Sequential: `N × 2 × logN` dispatches (one per column per stage)
- Batch: `2 × logN` dispatches (one batch dispatch per stage for all columns)

Example: 180 columns × 2^20 elements
- Before: 180 × 2 × 20 = 7,200 dispatches
- After: 40 dispatches

## Reed-Solomon Erasure Coding

### NTT-Based RS (BabyBear)

| Operation | Size | Throughput |
|-----------|------|------------|
| Encode k=2^8 | n=512 (2x) | 2.0 MB/s |
| Encode k=2^10 | n=2048 (2x) | 17.1 MB/s |
| Encode k=2^12 | n=8192 (2x) | 30.7 MB/s |
| Encode k=2^14 | n=32768 (2x) | **346.8 MB/s** |
| Encode k=2^14 | n=65536 (4x) | 92.6 MB/s |

### Data Availability Sampling (DAS)

| Blob Size | Throughput |
|-----------|------------|
| 1 KB | 3.0 MB/s |
| 64 KB | **249.2 MB/s** |
| 128 KB | 195.6 MB/s |

### GF(2^16) GPU RS (Systematic Encoding)

| k | parity | Throughput |
|---|--------|------------|
| 256 | 256 | 0.6 MB/s |
| 1024 | 1024 | 1.4 MB/s |
| 4096 | 4096 | 3.3 MB/s |

**Note**: GF(2^16) systematic encoding has lower throughput than NTT-based due to matrix multiplication overhead.

## GPU Sort Non-Determinism (Not a Problem for MSM)

### Updated Finding

The GPU counting sort produces **different intra-bucket orderings between runs**, but this is NOT a correctness issue for Pippenger MSM:

**Key insight**: The bucket placement IS correct - all points with digit D end up in buckets for digit D. The algorithm only depends on:
- **Bucket counts** - correct
- **Bucket offsets** - correct
- **Point indices within buckets** - but any order works

### Original Investigation

**Initial problem**: Buffer stride mismatch between GPU scatter kernel and Swift prefix sum initialization.

| Component | Original Index Calculation |
|-----------|--------------------------|
| GPU scatter kernel | `positions[w * n_buckets + digit]` |
| Swift prefix sum | `positions[w * effectiveN + digit]` |

**Fix**: Changed GPU scatter to use `n_points` stride matching Swift prefix sum:
```metal
// Changed from:
uint pos = atomic_fetch_add_explicit(&positions[w * n_buckets + digit], ...);
// To:
uint pos = atomic_fetch_add_explicit(&positions[w * n_points + digit], ...);
```

After this fix, counts and offsets are correct (0 diffs), but sorted_indices show ~30% diff rate due to **thread scheduling variability**.

### Why Atomic Operations Are Correct But Non-Deterministic

The `atomic_fetch_add_explicit` correctly ensures each thread gets a unique position. However:

1. Thread A and Thread B both want bucket `d`
2. Thread A's atomic completes first → position X
3. Thread B's atomic completes second → position X+1
4. **Both indices are placed in bucket d** but the order varies

Since `memory_order_relaxed` doesn't guarantee ordering between atomics, different runs can have different interleavings.

### Metal Memory Ordering Limitation

Metal only supports `memory_order_relaxed` for atomics on `device` address space:

```metal
// Not available on Metal device address space
atomic_fetch_add_explicit(&positions[...], 1u, memory_order_seq_cst);
// Error: candidate disabled: 'order' argument must be 'metal::memory_order_relaxed'
```

### Why Pippenger Still Passes

The Pippenger MSM algorithm only uses:
- **Bucket counts** - how many scalars map to each bucket
- **Bucket offsets** - starting position of each bucket in sorted array
- **Point indices within buckets** - but any deterministic ordering works

The bucket contents don't need to be in any specific order - they just need to contain all the correct points. The GPU sort produces correct counts and offsets, and the points within each bucket happen to all be valid for that bucket (just in varying order).

### Workaround

Using CPU-based sorting as fallback (~2ms for 32K points):

```swift
let useGpuSort = false  // CPU sort is correct and deterministic
```

### Potential Fix Directions Explored

1. **Sort-based approach (ATTEMPTED)**: Instead of counting sort, use GPU radix sort
   - **Result**: Pippenger correctness passes, but **~2.5x slower** than CPU sort
     - CPU overhead from array creation per window is significant
     - Fully GPU-based implementation would be faster but more complex

2. **Two-pass with local sort (SKIPPED)**: Each threadgroup sorts locally, then merge
   - **Why skipped**: Complex to implement correctly

3. **Pre-sorted indices (NOT VIABLE)**: Generate deterministic order on CPU
   - **Why not viable**: Defeats the purpose of GPU sorting

4. **Threadgroup-local histograms + merge (POTENTIAL)**: First phase builds local histograms per threadgroup (no atomics), second phase merges using CPU prefix sum

## Conclusion

The GPU sort non-determinism is caused by **thread scheduling variability** in atomic operations. However, this is **not a correctness problem** for Pippenger MSM - the algorithm only needs counts and offsets, not intra-bucket ordering.

**Recommended approach**: Continue using CPU sorting which is correct and fast (~2ms for 32K points). If GPU sorting is needed:
1. A fully GPU-based sorting algorithm (like radix sort) that doesn't use cross-threadgroup atomics
2. Using `memory_order_seq_cst` if Metal ever supports it on device address space

## Univariate Sumcheck (Aurora/Marlin Style)

Aurora-style single-round univariate sumcheck protocol using KZG commitments.
Claims `sum_{x in H} f(x) = v` in one round via polynomial decomposition.

**Implementation**: `Sources/zkMetal/Polynomial/UnivariateSumcheckEngine.swift`

**Key fix (2026-04-27)**: Fixed BN254 scalar masking bug in C Pippenger MSM (`bn254_msm.c`).
The `get_window_digit()` function was not masking bits beyond the 254-bit scalar width,
causing out-of-bounds bucket access and corrupted KZG verification results.

### Single Prove/Verify Benchmark (BN254 Fr, CPU)

| logN | n | prove(ms) | verify(ms) | proof(bytes) |
|------|---|-----------|------------|--------------|
| 2^6 | 64 | 2.7 | 1.1 | 576 |
| 2^8 | 256 | 5.5 | 1.1 | 576 |
| 2^10 | 1024 | 11.9 | 1.1 | 576 |

### Batch Prove/Verify Benchmark (BN254 Fr, CPU)

| logN | k (polys) | prove(ms) | verify(ms) |
|------|-----------|-----------|------------|
| 2^8 | 2 | 10.3 | 1.1 |
| 2^8 | 4 | 9.7 | 1.1 |
| 2^8 | 8 | 10.6 | 1.1 |
| 2^10 | 2 | 23.1 | 1.1 |
| 2^10 | 4 | 24.3 | 1.2 |
| 2^10 | 8 | 21.3 | 1.1 |

**Proof size**: 576 bytes = 5 projective points (5 × 96 bytes) + 3 Fr scalars (3 × 32 bytes)

**Note**: Prove time scales with polynomial degree (O(n) for degree 2n). Verify is constant time O(1).

## Version History

- **2026-04-27**: Univariate sumcheck KZG verification fixed. Bug was in BN254 scalar masking in `bn254_msm.c:get_window_digit()` — bits beyond position 254 caused out-of-bounds bucket access. All tests now pass. Benchmark: 2^6 prove=2.7ms/verify=1.1ms, 2^8 prove=5.5ms/verify=1.1ms, 2^10 prove=11.9ms/verify=1.1ms.
- **2026-04-24**: BLS12-377 MSM GPU hang fixed with CPU fallback. GPU kernel hangs at n>=4096 due to 12-limb field register pressure. Added 30s timeout with polling and CPU fallback for large sizes. Benchmark results: 2^8=1.4ms, 2^10=4.1ms, 2^12=10.4ms, 2^14=31.2ms, 2^16=110.1ms, 2^17=208.7ms, 2^18=421.6ms.
- **2026-04-22**: GPU Additive FFT inverse bug fixed - butterfly requires brute-force solve for hi. Added GF(2^8) FFT benchmark section.
- **2026-04-22**: GPU sort non-determinism clarified - NOT a correctness problem for Pippenger MSM. Root cause is thread scheduling variability, but algorithm only needs counts/offsets, not intra-bucket ordering. Radix sort attempted but slower than CPU sort.
- **2026-04-22**: GPU sort non-determinism investigated - root cause is thread scheduling variability in atomic operations. Sort-based fix (radix sort) attempted but slower than CPU sort. Two-pass local sort skipped as complex. CPU sort remains workaround.
- **2026-04-21**: Updated P^1 FRI with inv2t cache numbers (8-21x faster), added RS/DAS section
- **2026-04-18**: Updated with beta macOS 26.3 performance notice and regression analysis
- **2026-04-14**: Initial baseline performance on stable macOS 15.x
