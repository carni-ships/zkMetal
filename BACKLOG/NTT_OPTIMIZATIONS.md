# NTT BN254 Optimization Ideas

This document tracks optimization ideas for the GPU-accelerated NTT on BN254 Fr.

## Priority 1: Threadgroup Twiddle Cache

**Status**: ❌ Rejected - Exceeds threadgroup memory limit

**Idea**: Cache twiddle factors in threadgroup memory to avoid redundant global memory reads.

**Challenge**: The twiddle index pattern within a block is `twiddle_idx = local_idx * (n / global_block_size)`. All threads with the same `local_idx` within a block need the same twiddle factor, so `half_block` twiddle values could be cached. However, the cache indexing must use `local_idx` (not `tid`), and the benefit is limited since each twiddle is only read once per stage anyway.

**Issue**: Attempted implementation required adding `threadgroup Fr twiddle_cache[512]` to kernels that already use `threadgroup Fr shared[1024]`. This exceeded the 32KB threadgroup memory limit on Apple M1/M2 GPUs, causing runtime failures: "Threadgroup memory size (49152) exceeds the maximum threadgroup memory allowed (32768)".

**Alternative**: The fused kernels already read twiddles from global memory only once per stage, and the bandwidth savings from caching are minimal since each twiddle is used by only one thread per stage.

## Priority 2: Batch NTT for Multiple Transforms

**Status**: ✅ Implemented (commits 3ebb378e, ff0b6290, 85c58138, bf647abf)

The `BatchNTTEngine` exists and supports:
- Batch processing of K transforms in a single GPU dispatch
- Grid Y dimension for transform index
- Four-step FFT with fused row+twiddle+transpose kernels
- Test coverage: `GPUPlonkWireAssignTests.testBatchNTTRoundTrip`

## Priority 3: Lazy Reduction

**Status**: ❌ Rejected - Too complex/high-risk

**Idea**: Delay modular reduction in BN254 field operations.

**Issue**: Requires ensuring `a + b < 2^256` to avoid overflow. Butterfly operations have complex data dependencies. Risk of subtle correctness bugs.

**Expected impact**: ~5-10% improvement

## Priority 4: Fuse Row+Twiddle+Transpose

**Status**: ✅ Implemented

**Kernels**:
- `ntt_row_fused_twiddle_transpose` (line 1155) - single transform
- `ntt_row_fused_twiddle_transpose_batch` (line 2288) - batch transforms
- `intt_row_fused_twiddle_transpose_batch` (line 2407) - inverse batch

These kernels combine steps 2+3+4 of four-step FFT into a single dispatch.

## Priority 5: Higher Radix (Radix-8)

**Status**: ❌ Rejected - Complex, four-step FFT provides similar benefits

**Issue**: Requires 7 twiddle multiplications per butterfly with complex data access patterns. Four-step FFT already provides ~20% improvement with lower complexity.

## Priority 6: Vectorized Montgomery Multiplication

**Status**: Not implemented

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

## Benchmark Baseline (stable macOS 15.x)

| Size | Current | Floor | Headroom |
|------|---------|-------|----------|
| 2^16 | 0.85ms | ~0.5ms | 1.7x |
| 2^18 | 1.72ms | ~1ms | 1.7x |
| 2^20 | 6.06ms | ~0.7ms | 8.7x |
| 2^22 | 26.0ms | ~3ms | 8.7x |

## References

- Original NTT kernel: `Sources/Shaders/ntt/ntt_kernels.metal`
- NTT Engine: `Sources/zkMetal/NTT/NTTEngine.swift`
- Batch NTT Engine: `Sources/zkMetal/NTT/BatchNTTEngine.swift`
- Four-step FFT paper: Good for large N where data doesn't fit in L2
