# NTT BN254 Optimization Ideas

This document tracks optimization ideas for the GPU-accelerated NTT on BN254 Fr.

## Priority 1: Threadgroup Twiddle Cache

**Status**: Attempted but caused correctness issues - needs debugging

**Idea**: Cache twiddle factors in threadgroup memory to avoid redundant global memory reads.

**Implementation attempt** (reverted due to bug):
```metal
// Threadgroup twiddle cache
if (tid < half_block) {
    uint global_block_size = 1u << (stage + 1);
    uint twiddle_idx = tid * (n / global_block_size);
    twiddle_cache[tid] = twiddles[twiddle_idx];
}
```

**Issue**: The caching logic had a subtle bug where the cache was indexed by `local_idx` but loaded by `tid`, causing incorrect twiddle factor selection.

**Required fix**: Properly verify that the twiddle factor is the same for all threads with the same `local_idx` before enabling the cache.

## Priority 2: Batch NTT for Multiple Transforms

**Status**: Not started

**Idea**: Process multiple independent NTTs in a single kernel dispatch to amortize command buffer overhead.

**Implementation**:
- Process K NTTs of size N simultaneously
- Each thread processes elements from all K transforms
- Share kernel launch overhead across K transforms
- Twiddle factors can be broadcast to all transforms

**Expected impact**: ~10-20% improvement for scenarios with many small NTTs

## Priority 3: Lazy Reduction

**Status**: Not started

**Idea**: Delay modular reduction in BN254 field operations.

**Implementation**:
- Use `fr_add_lazy` which skips modular reduction
- Accumulate 2-3 operations before reducing
- Reduces branch mispredictions and reduction overhead

**Expected impact**: ~5-10% improvement in field-heavy code

## Priority 4: Fuse Row+Twiddle+Transpose

**Status**: Not started

**Idea**: In four-step FFT, fuse the row FFT with twiddle multiplication and transpose into a single kernel.

**Current flow**:
1. Column FFT
2. Transpose
3. Row FFT + twiddle multiply
4. Transpose back

**Optimized flow**:
1. Column FFT
2. Fused row+twiddle+transpose (single kernel)

**Expected impact**: ~10-15% improvement for large transforms

## Priority 5: Higher Radix (Radix-8)

**Status**: Not started

**Idea**: Process 3 stages at once using radix-8 butterflies.

**Implementation**:
- Radix-8 butterfly processes 8 elements with 3 twiddle multiplications
- Reduces loop overhead and memory access by ~33%
- More complex twiddle factor computation

**Expected impact**: ~15-20% improvement

## Priority 6: Vectorized Montgomery Multiplication

**Status**: Not started

**Idea**: Use 2x or 4x vectorized loads/stores for Fr elements.

**Implementation**:
- Pack 2 Fr elements into 64 bytes for vector load
- Use Metal's simdgroup operations
- Batch Montgomery multiplications

**Expected impact**: Depends on memory bandwidth vs compute

## Rejected Ideas

| Idea | Reason |
|------|--------|
| Precomputed twiddle table on GPU | Memory cost exceeds benefit |
| Different FFT algorithm (Cooley-Tukey vs Stockham) | Current algorithm is optimal for GPU |
| fp_montmul_* optimizations from MSM | Different use pattern in NTT |

## Benchmark Baseline

| Size | Current (stable macOS) | Floor | Headroom |
|------|-------------------------|-------|----------|
| 2^16 | 0.85ms | ~0.5ms | 1.7x |
| 2^18 | 1.72ms | ~1ms | 1.7x |
| 2^20 | 6.06ms | ~0.7ms | 8.7x |
| 2^22 | 26.0ms | ~3ms | 8.7x |

## References

- Original NTT kernel: `Sources/Shaders/ntt/ntt_kernels.metal`
- NTT Engine: `Sources/zkMetal/NTT/NTTEngine.swift`
- Four-step FFT paper: Good for large N where data doesn't fit in L2
