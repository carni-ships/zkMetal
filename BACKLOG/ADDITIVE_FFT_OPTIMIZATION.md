// GPU Additive FFT Optimization Analysis
//
// Performance Issue: Current implementation gets ~11-14ms for 2^22, target is ~0.5ms
//
// Root Cause Analysis:
// 1. Memory access patterns - each level reads from global memory
// 2. No SIMD vectorization - processing single bytes instead of vectors
// 3. LUT lookup latency - each multiplication requires memory lookup
// 4. Sequential processing - not充分利用 GPU parallelism

## Current Bottlenecks

### Memory Bandwidth Analysis

For 2^22 elements (4,194,304 bytes = 4MB):
- k=22 levels
- Each level: n/2 butterfly operations
- Total butterflies: n * k / 2 = ~46 million
- Each butterfly: 1 read + 1 write = 2 memory ops
- Total memory ops: ~92 million

At 231 GB/s bandwidth:
- Theoretical best case: 4MB / 0.231GB/s = 0.017ms per round-trip
- 22 rounds × 0.017ms = 0.37ms (theoretical floor)

Current: 11-14ms = **30-40x slower than theoretical floor**

### Why The Gap?

1. **Non-coalesced memory access**: Random access patterns
2. **LUT lookup overhead**: Each multiplication requires cache miss
3. **Lack of SIMD**: Processing 1 byte instead of 16 bytes (SIMD width)
4. **Sequential processing**: Not enough threads to hide latency

## Optimization Strategies

### 1. SIMD Vectorization (uchar4) - 4x Potential

Process 4 elements per thread using uchar4:

```metal
thread uint4 vals;
vals[0] = data[gid * 4 + 0];
vals[1] = data[gid * 4 + 1];
vals[2] = data[gid * 4 + 2];
vals[3] = data[gid * 4 + 3];
```

Benefits:
- 4x throughput improvement
- Better memory coalescing
- Reduced thread count (better occupancy)

Expected: 11ms → 2.75ms

### 2. Precompute Partner Offsets - 1.5x Potential

Avoid repeated modulo/division operations:

```metal
uint partnerOffsets[22];
for (depth = 0; depth < k; depth++) {
    partnerOffsets[depth] = (local_idx >= halfSize) ? gid - halfSize : -1;
}
```

Benefits:
- Eliminates 22 modulo operations per element
- Eliminates 22 division operations per element
- Better register reuse

Expected: 2.75ms → 1.8ms

### 3. Optimized LUT Access - 2x Potential

Current LUT access pattern causes cache misses. Use:

- Threadgroup memory for LUT tiles
- Swizzled LUT layout for better cache lines
- Prefetch next LUT entries

Expected: 1.8ms → 0.9ms

### 4. Register Tiling - 1.5x Potential

Process multiple consecutive elements in registers:

```metal
for (int i = 0; i < 4; i++) {
    uint8_t val = vals[i];
    // Process all k levels for this element
    // Store result back
}
```

Benefits:
- Better temporal locality
- Reduced memory traffic
- Better instruction-level parallelism

Expected: 0.9ms → 0.6ms

### 5. Batched Processing - Amortized Overhead

Process multiple FFTs in single kernel:

```metal
kernel void additive_fft_batch(
    uint batchSize,
    uint gid
) {
    uint fftIdx = gid / n;
    uint elemIdx = gid % n;
    // Process FFT[fftIdx][elemIdx]
}
```

Benefits:
- Single kernel launch for multiple FFTs
- Better GPU utilization
- Reduced per-call overhead

## Implementation Priority

| Priority | Optimization | Expected Speedup | Complexity |
|----------|-------------|------------------|------------|
| 1 | SIMD (uchar4) | 4x | Medium |
| 2 | Register Tiling | 1.5x | Low |
| 3 | Optimized LUT | 2x | Medium |
| 4 | Precomputed Offsets | 1.5x | Low |
| 5 | Batched Processing | Amortized overhead | Medium |

Combined: 11ms → 0.5ms (22x speedup)

## Key Insights

1. **The LUT approach is not the bottleneck** - memory access patterns are
2. **SIMD vectorization is critical** - GPU is designed for SIMD
3. **Memory coalescing matters** - adjacent threads should access adjacent memory
4. **Threadgroup memory has limited benefit** - max 1024 elements, but we have 4M

## Next Steps

1. Implement SIMD kernel
2. Benchmark to verify improvement
3. Iterate on memory access patterns
4. Consider hybrid CPU/GPU approach for very large FFTs
