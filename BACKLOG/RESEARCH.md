# Research Notes

Detailed theoretical analysis and research findings.

## GPU Additive FFT over GF(2^8) (2026-04-12)

**Current**: 13ms for 2^22 elements. **Theoretical floor**: ~0.5ms. **Headroom**: ~26x.

**Bottleneck**: Each element goes through k=22 butterfly levels serially. Each level requires one GF(2^8) multiply via 8-iteration shift-XOR carry-less multiply + reduction (totaling ~176 primitive ops per element).

### Optimization 1: Precomputed GF(2^8) Multiplication LUT

Replace the shift-XOR multiply with a precomputed lookup table. GF(2^8) has only 256 elements, so a 256-entry LUT costs only 256 bytes. A full 256x256 table is 64KB (fits in Metal's 32KB constant address space per threadgroup).

**Estimated impact**: 5-10x faster multiply. **Overall FFT: 3-6x speedup** (13ms → 2-4ms).

### Optimization 2: Process 4 Elements Per Thread with SIMD Shuffle Cooperation

Use 4 threads per group of 4 elements and employ Metal SIMD shuffle to cooperatively compute the 8-bit products.

**Estimated impact**: 2-4x speedup in the multiply phase. Combined with LUT, could reach **6-10x total speedup**.

### Optimization 3: Batch Pointwise Multiply Kernel Fusion

Fuse the forward FFT butterfly chain with a following pointwise multiply pass, all in one dispatch.

**Estimated impact**: 10-20% additional speedup for the polynomial multiply use case.

---

## secp256k1 MSM (2026-04-12)

**Current**: GPU 770ms, CPU 260ms (CPU 3x faster). **Empirical floor**: ~260ms (CPU).

**Bottleneck Analysis**:
- GPU MSM at 2^18: 770ms (262K points, centered scalars, wb=16, 32769 buckets)
- CPU C Pippenger at 2^18: 260ms
- GLV was tested and made things **WORSE** because it doubled points -> doubled sort time

### Optimization 1: CPU GLV — REJECTED

GLV with CPU decomposition makes GPU MSM **worse**, not better. GLV doubles the effective point count, which hurts the memory-bound sort phase more than the bucket reduction phase can compensate.

### Optimization 2: Cooperative SIMD Bucket Reduction with Warp-Level Parallelism

Use a **wider cooperative reduction** where the entire warp (32 threads) cooperates on each bucket's reduction in a tree pattern.

**Estimated impact**: 15-25% speedup on the bucket reduction phase.

### Optimization 3: Eliminate Redundant GLV Endomorphism Kernel

Precompute the GLV endomorphism on CPU during SRS loading instead of running a separate GPU kernel.

**Estimated impact**: ~45ms saving per proof.

---

## GPU MSM Theoretical Transformations (2026-04-12)

**Problem**: GPU is 3x slower than CPU (781ms vs 276ms at 2^18). Root cause is **memory-bound sort phase** and **poor GPU utilization**.

### Why MSM is GPU-hostile
1. **Random bucket indices** — scalar bits cause non-coalesced memory access
2. **Irregular occupancy** — some buckets get many points, most get 0-1
3. **Sort phase dominates** — O(n log n) with terrible GPU cache behavior
4. **Dependent reductions** — can't parallelize across buckets until sort completes

### Transformation 1: Batch Many Small MSMs *(HIGHEST IMPACT)*

Instead of one 2^18 MSM, run 256 × 2^10 MSMs in parallel. Each small MSM:
- No sorting needed (small window fits in registers)
- Fully predictable memory access
- Trivially parallel across the 256 instances

**Estimated impact**: If each 2^10 MSM takes ~0.5ms on GPU, 256 × 0.5ms = 128ms theoretical. Could be **3-6x faster** than current 781ms.

### Transformation 2: Precomputed Window Tables — REJECTED

**Rejected** - For k=8, 18 × 256 = 4,608 entries × 64 bytes = 288KB per point. For 262K points: 72GB (too large).

### Transformation 3: NAF Representation — COMPLETE

NAF guarantees at most n/3 ones per scalar (vs n/2 for binary): ~33% fewer point additions.

**Status**: ✅ Complete (reference implementation added)

### Transformation 4: Higher Radix + Shared Memory Tree Reduction — COMPLETE

Use radix-2^r with r > wb and have each thread block aggregate one bucket using shared memory parallel reduction.

**Status**: ✅ Complete (256-thread kernel added)

### Transformation 5: Bucket-Interleaved Memory Layout

Pre-sort points into bucket-interleaved layout so adjacent threads access adjacent memory.

**Estimated impact**: 15-25% speedup.

### Transformation 6: Interleaved GLV + Batch Small MSMs

Combine GLV decomposition with batch-small approach.

**Status**: Uncertain — GLV alone failed but small-batch combination is unexplored.

### Transformation 7: MSM via Polynomial Evaluation — THEORETICAL ONLY

Requires computing discrete logs, which is intractable for arbitrary points.

### Transformation 8: Cubic Sums / Multilinear Extension — THEORETICAL ONLY

Still requires discrete log.
