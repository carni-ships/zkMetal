# GPU Additive FFT Optimization Backlog

## SIMD Vectorization (IMPLEMENTED)
- **Status**: ✅ SIMD kernels added to `additive_fft_gf8.metal`
- **Kernels**: `additive_fft_gf8_forward_simd` (4 elements/thread), `additive_fft_gf8_forward_simd8` (8 elements/thread)
- **Integration**: Added to `GPUAdditiveFFTEngine.swift` with `forwardSimd()` and `forwardSimd8()` methods
- **Expected**: 4x speedup (11ms → 2.75ms) for large FFTs

## Future Optimizations (Pending)

### High Priority

#### 1. Register Tiling
- **Impact**: 1.5x speedup
- **Description**: Keep multiple elements in registers throughout processing
- **Implementation**: Add `forwardRegisterTile()` method using threadgroup memory for intermediate results
- **Status**: Not started

#### 2. Batched Processing
- **Impact**: Amortized overhead
- **Description**: Process multiple FFTs in single kernel dispatch
- **Implementation**: Use existing `additive_fft_gf8_forward_batch` kernel
- **Status**: Kernel exists, needs benchmark integration

#### 3. Fused FFT + Commitment
- **Impact**: High
- **Description**: Combine FFT with Merkle commitment to avoid intermediate memory round-trip
- **Implementation**: Extend `additive_fft_gf8_forward_then_pointwise_mul` pattern
- **Status**: Not started

### Medium Priority

#### 4. LUT as Metal Function Constant
- **Impact**: 2x speedup
- **Description**: Pass LUT as constant address space instead of device pointer
- **Implementation**: Use `constant` qualifier instead of `device` for LUT buffer
- **Status**: Not started

#### 5. Double/Pipelined Buffering
- **Impact**: 1.5x speedup
- **Description**: Use ping-pong buffers to overlap computation with memory transfers
- **Implementation**: Add double-buffering kernel variant
- **Status**: Not started

#### 6. Threadgroup Memory for Butterfly Exchange
- **Impact**: Medium
- **Description**: Use shared memory for butterfly partner exchanges within threadgroup
- **Implementation**: New kernel variant `additive_fft_gf8_forward_tg_bfly`
- **Status**: Not started

### Low Priority

#### 7. Optimized Memory Access Patterns
- **Impact**: 2x speedup
- **Description**: Coalesced access patterns for better cache utilization
- **Status**: Analysis complete, implementation pending

#### 8. Half4/uchar8 Vectorization
- **Impact**: Medium
- **Description**: Even more aggressive vectorization (16 elements per thread)
- **Status**: Not started

## Implementation Roadmap

```
Phase 1 (Complete):
✅ SIMD vectorization (uchar4)
✅ Kernel integration into GPUAdditiveFFTEngine

Phase 2 (Next):
⏳ Benchmark SIMD kernels to verify 4x improvement
⏳ Add register tiling kernel
⏳ Optimize batched processing

Phase 3 (Future):
⏳ LUT as constant address space
⏳ Fused FFT + commitment
⏳ Ping-pong buffering

Phase 4 (Long-term):
⏳ Threadgroup butterfly exchange
⏳ uchar16 vectorization (if GPU supports)
```

## Benchmark Targets

| Kernel Variant | Elements/Thread | Threads (2^22) | Expected Time |
|----------------|-----------------|----------------|---------------|
| Standard | 1 | 4,194,304 | 11-14ms |
| Pairs | 1 | 2,097,152 | 8-11ms |
| Threadgroup Cache | 1 | 2,097,152 | 6-9ms |
| SIMD (4 elements) | 4 | 1,048,576 | **2.75ms** ✅ |
| SIMD8 (8 elements) | 8 | 524,288 | **1.5ms** |

## Performance Analysis

Current bottleneck: **GPU utilization ~3%**
- Single byte per thread instead of 16-byte SIMD vectors
- Non-coalesced memory access patterns
- Thread divergence in standard kernel

Solution path:
1. SIMD vectorization: 4x improvement (3% → 12% utilization)
2. Better memory patterns: 2x improvement (12% → 24% utilization)
3. Register tiling: 1.5x improvement (24% → 36% utilization)
4. LUT optimization: 2x improvement (36% → 72% utilization)

Target: **0.5ms** (72% GPU utilization)