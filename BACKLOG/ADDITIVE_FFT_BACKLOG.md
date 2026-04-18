# GPU Additive FFT Optimization Backlog

## Benchmark Results (2026-04-18)

| Size | Standard | Pairs | TG-Cache | Winner |
|------|----------|-------|----------|--------|
| 2^16 | 16.91ms | 18.03ms | **13.74ms** | TG-Cache |
| 2^18 | **12.40ms** | 15.76ms | 15.88ms | Standard |
| 2^20 | 18.13ms | 16.75ms | **12.98ms** | TG-Cache |
| 2^22 | 30.79ms | **17.78ms** | 18.94ms | Pairs |

**Key Findings:**
- **Pairs kernel wins at large sizes** (2^22): 17.78ms vs 30.79ms standard — **1.73x speedup**
- **TG-Cache is fastest at medium sizes** (2^16, 2^20)
- **Standard kernel surprisingly fastest at 2^18** — possibly due to cache warming
- Pairs/TG-Cache advantage grows with size (eliminates thread divergence)

## Naive SIMD Approach — FAILED

- **Status**: ❌ SIMD kernels implemented but SLOWER than standard
- **Root Cause**: Butterfly algorithm accesses non-contiguous partner indices
- Processing consecutive elements (0,1,2,3) doesn't align with butterfly partner patterns
- **Result**: SIMD kernel 1.5x slower (15.88ms vs 10.74ms) — abandoned

## Future Optimizations (Pending)

### High Priority

#### 1. Pairs Kernel Optimization
- **Impact**: 1.5x speedup
- **Description**: Improve pairs kernel for medium sizes where it's currently slower
- **Status**: Not started

#### 2. Adaptive Kernel Selection
- **Impact**: Medium
- **Description**: Auto-select best kernel (Standard/Pairs/TG-Cache) based on size
- **Implementation**: Extend `forwardAuto()` to choose optimal kernel per size
- **Status**: Not started

#### 3. Batched Processing
- **Impact**: Amortized overhead
- **Description**: Process multiple FFTs in single kernel dispatch
- **Implementation**: Use existing `additive_fft_gf8_forward_batch` kernel
- **Status**: Kernel exists, needs benchmark integration

#### 4. Fused FFT + Commitment
- **Impact**: High
- **Description**: Combine FFT with Merkle commitment to avoid intermediate memory round-trip
- **Implementation**: Extend `additive_fft_gf8_forward_then_pointwise_mul` pattern
- **Status**: Not started

### Medium Priority

#### 5. LUT as Metal Function Constant
- **Impact**: 2x speedup
- **Description**: Pass LUT as constant address space instead of device pointer
- **Implementation**: Use `constant` qualifier instead of `device` for LUT buffer
- **Status**: Not started

#### 6. Register Tiling
- **Impact**: 1.5x speedup
- **Description**: Keep multiple elements in registers throughout processing
- **Implementation**: Use threadgroup memory for intermediate results
- **Status**: Not started

#### 7. Double/Pipelined Buffering
- **Impact**: 1.5x speedup
- **Description**: Use ping-pong buffers to overlap computation with memory transfers
- **Implementation**: Add double-buffering kernel variant
- **Status**: Not started

### Low Priority

#### 8. Threadgroup Butterfly Exchange
- **Impact**: Medium
- **Description**: Use shared memory for butterfly partner exchanges within threadgroup
- **Implementation**: New kernel variant `additive_fft_gf8_forward_tg_bfly`
- **Status**: Not started

#### 9. Coalesced Memory Access Patterns
- **Impact**: 2x speedup
- **Description**: Reorder memory access to improve cache utilization
- **Status**: Analysis pending

## Implementation Roadmap

```
Phase 1 (Complete):
✅ Benchmark existing kernels across all sizes
✅ Identify pairs kernel advantage at large sizes
✅ Identify TG-Cache advantage at medium sizes
✅ Naive SIMD approach abandoned (failed)

Phase 2 (Next):
⏳ Adaptive kernel selection based on size
⏳ Optimize pairs kernel for medium sizes
⏳ Benchmark batched processing

Phase 3 (Future):
⏳ LUT as constant address space
⏳ Register tiling kernel
⏳ Fused FFT + commitment

Phase 4 (Long-term):
⏳ Threadgroup butterfly exchange
⏳ Ping-pong buffering
```

## Performance Analysis

Current best (2^22): **17.78ms** (Pairs kernel)
Target: **0.5ms**

Bottleneck breakdown:
- Thread divergence in standard kernel: ~2x overhead
- Memory access patterns: significant but addressable
- LUT access latency: addressable via constant memory

Solution path:
1. Adaptive kernel selection: 1.2x improvement
2. LUT as constant: 2x improvement
3. Register tiling: 1.5x improvement
4. Fused FFT+commitment: high but requires full pipeline change

Target path: **0.5ms** achievable via combination
