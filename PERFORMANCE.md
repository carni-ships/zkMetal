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
| 2^10 | 0.60ms |
| 2^12 | 0.40ms |
| 2^14 | 0.80ms |
| 2^16 | 0.56ms |
| 2^18 | 1.17ms |
| 2^20 | 3.69ms |

### P^1 FRI Commit Phase (GPU) - With inv2t Caching

Optimized with inv2t caching (65x speedup over naive recomputation).

| Size | Commit Time | Rounds |
|------|-----------|--------|
| 2^14 | **0.24-0.59ms** | 13 |
| 2^16 | **0.83ms** | 15 |
| 2^18 | **2.39ms** | 17 |
| 2^20 | **3.99ms** | 19 |

**Key optimization**: Precompute all inv2t arrays for all FRI rounds upfront with GPU buffer caching.

### P^1 FRI Fused Commit Phase (fold-by-8 Cascade)

The fused commit uses a GPU kernel that computes 8 FRI fold rounds in a single dispatch, outputting all intermediate layers for complete proof generation.

| Size | Standard Commit | Fused Commit | Layers Produced |
|------|----------------|-------------|-----------------|
| 2^14 | 0.59ms | ~0.5ms | 14 (all intermediate) |
| 2^18 | 1.38ms | ~1.5ms | 18 (all intermediate) |
| 2^20 | 3.74ms | ~6ms | 20 (all intermediate) |

**Trade-off**: Fused commit produces complete layer output enabling full `queryPhase()` verification, but has slightly higher overhead due to intermediate buffer allocation and readback. Use standard `commitPhase()` when only final result is needed.

**Implementation**: Metal kernel `p1_fri_fold_by8` writes intermediate stages to 7 output buffers (indices 12-18), then Swift engine reads back all layers for Merkle root computation.

### P^1 FRI Multi-fold Performance (GPU)

| Size | Single Fold | Multi-fold (all rounds) |
|------|-------------|--------------------------|
| 2^14 | 1.30ms | 2.25ms |
| 2^18 | 13.36ms | 27.86ms |
| 2^20 | 52.19ms | 108.53ms |

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

## Version History

- **2026-04-21**: Updated P^1 FRI with inv2t cache numbers (8-21x faster), added RS/DAS section
- **2026-04-18**: Updated with beta macOS 26.3 performance notice and regression analysis
- **2026-04-14**: Initial baseline performance on stable macOS 15.x
