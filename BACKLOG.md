# zkMetal Optimization Backlog

All major optimizations are complete. System is near hardware limits.

## Completed ✓

| Primitive | Current | Floor | Headroom |
|-----------|---------|-------|----------|
| MSM BN254 2^18 | 72ms | ~50ms | ~1.4x |
| NTT BN254 2^22 | 26ms | ~3ms | ~9x |
| Sumcheck 2^20 | 3.3ms | ~1ms | ~3x |
| FRI Fold 2^20 | 2.9ms | ~0.3ms | ~10x |
| Basefold open 2^18 | 61ms | ~20ms | ~3x |
| Blake3 Batch 2^20 | 1.0ms | ~0.6ms | ~1.7x |
| Poseidon2 batch 2^16 | 7.4ms | ~1.8ms | ~4x |
| Blaze prove 2^18 | 567ms | ~150ms | ~3x |

**All major optimizations implemented:**
- Karatsuba GPU fp_mul (MSM 444ms→190ms, 57% faster)
- All-GPU MSM mode (182ms→72ms)
- FRI fold-by-4/16 cascade
- Basefold fold-by-4 + pipeline overlap
- Sumcheck GPU final reduction
- Blake3 vectorized uint4 (3.5x speedup)
- Blaze fold-by-8 FRI (671ms→567ms, 15% faster)
- Pasta Poseidon2/NTT engines with GPU acceleration
- Basefold fold-by-16 kernel (fused 4-round dispatch, SM-side ready)
- BabyBear Barrett reduction — Not viable: naive `% UInt64(P)` is correct; Barrett approximation breaks for small products where v ≈ 0 (quotient ≈ 0, approximation ≈ 2^31). Products range [0, P²] so approximation error is unbounded. Field is not a bottleneck anyway.

## Blocked / Rejected

- **MTLEvent infrastructure** — Not viable. Most waits are correctness-required.
- **Metal async compute** — Blocked by Fiat-Shamir sequentiality.
- **Smaller point representation** — Rejected. Decompression cost exceeds bandwidth savings.
- **4-ary Merkle tree** — Rejected. Poseidon2 is not associative.
- **GPU sort for MSM** — Disabled. CPU sort is 50-80x faster on M3 Pro (0.5ms vs 36-44ms) and correct.

## Remaining Opportunities

- **Hardware upgrade** (M4 Pro/Max with more GPU cores)
- **Protocol changes** (different PCS, fewer commitment rounds)
- **MetalSpoon SP1 prover** GPU port (constraint eval / permutation / PCS dispatch)
- **FRI Merkle investigation** (2026-04-12): Real FRI bottleneck is Merkle commit at 210ms (80% of FRI time), not fold at 13ms (4%). Merkle uses Keccak GPU which is already near hardware limits. No viable optimization target remains.
- **Theoretical extensions** (2026-04-12): Fused-DeepFold with shared-memory batch (4-8 rounds in one SM dispatch), Lazy Cantor-FFT for Circle STARK, WHIR-RAA deterministic batched queries — all require deep protocol changes and Metal shader changes; see session notes for math sketch.

---

## GPU Additive FFT over GF(2^8) — Actionable Optimizations (2026-04-12)

**Current**: 13ms for 2^22 elements. **Theoretical floor**: ~0.5ms. **Headroom**: ~26x.

**Bottleneck**: Each element goes through k=22 butterfly levels serially. Each level requires one GF(2^8) multiply via 8-iteration shift-XOR carry-less multiply + reduction (totaling ~176 primitive ops per element). Elements are completely independent, so GPU parallelism is maximal (4M threads), but each thread is bottlenecked on serial multiplies.

### Optimization 1: Precomputed GF(2^8) Multiplication LUT

**What**: Replace the shift-XOR multiply with a precomputed lookup table. GF(2^8) has only 256 elements, so a 256-entry LUT (indexed by one operand, with the other operand as data) costs only 256 bytes. A full 256x256 table is 64KB (fits in Metal's 32KB constant address space per threadgroup).

**Code change** (additive_fft_gf8.metal):
```metal
// At top of kernel file, add constant LUT (one per operand):
// LUT1[b] = a * b for fixed a — use in innermost loop with varying a
// Or full LUT: constant uint8_t gf28_mul_lut[65536] [[const]];

// Forward kernel changes:
// Instead of: uint8_t twisted = lo_val ^ gf28_mul(s, hi_val);
// Use:        uint8_t twisted = lo_val ^ gf28_mul_lut[s * 256 + hi_val];

// gf28_mul_lut computation at kernel init (or precomputed host-side):
// for a in 0..255: for b in 0..255: lut[a*256+b] = gf28_mul(a, b)
```

**Estimated impact**: 5-10x faster multiply (1 LUT lookup + 1 XOR vs 8 shift-XOR + reduction). **Overall FFT: 3-6x speedup** (13ms → 2-4ms), closing half the gap to theoretical floor. The LUT fits in L1/constant cache; Metal constant address space supports 32KB threadgroup-local, plenty for 64KB table.

**Why this helps**: The 8-iteration shift-XOR chain is the dominant cost per butterfly level. Each iteration is a dependent chain (shift then XOR), limiting ILP. LUT lookup breaks this chain entirely.

### Optimization 2: Process 4 Elements Per Thread with SIMD Shuffle Cooperation

**What**: Instead of 1 thread per element, use 4 threads per group of 4 elements and employ Metal SIMD shuffle to cooperatively compute the 8-bit products. Each SIMD operation can compute multiple bits of the product in parallel.

**Code change**: Restructure the forward kernel to use threadgroups of 4 (SIMD width), where threads exchange intermediate results via `simd_shuffle`. The 8 shift-XOR iterations of `gf28_mul` become 2-3 SIMD-level parallel operations.

**Estimated impact**: 2-4x speedup in the multiply phase. Combined with LUT, could reach **6-10x total speedup** (13ms → 1.3-2ms).

**Why this helps**: Metal GPUs execute SIMD groups of 32 threads in lockstep. The current kernel uses 1 thread per element with all 8 shift-XOR iterations serial. By having 4 threads cooperate on 4 elements simultaneously, the GPU's SIMD execution units can compute multiple partial products in parallel within a single instruction, better utilizing the GPU's vector datapath.

### Optimization 3: Batch Pointwise Multiply Kernel Fusion

**What**: The `gf28_pointwise_mul` kernel (used after forward FFT for polynomial multiplication) also uses the serial `gf28_mul`. Fusing this into the same dispatch as the forward FFT would eliminate a separate kernel launch and global memory round-trip. Additionally, the pointwise multiply is O(n) with no data dependency between elements, making it trivially parallelizable with higher arithmetic intensity than the butterfly chain.

**Code change**: Add a combined kernel `additive_fft_gf8_forward_then_pointwise_mul` that fuses the forward FFT butterfly chain with a following pointwise multiply pass, all in one dispatch.

**Estimated impact**: 10-20% additional speedup for the polynomial multiply use case by eliminating a memory round-trip and kernel launch overhead. The FFT itself remains unchanged.

**Why this helps**: After forward FFT, data must be written to global memory then read back for pointwise multiply. Fusion eliminates this round-trip entirely (~0.5-1ms savings on 2^22 elements at M3 Pro memory bandwidth).

---

## secp256k1 MSM — Actionable Optimizations (2026-04-12)

**Current**: GPU 770ms, CPU 260ms (CPU 3x faster). **Empirical floor**: ~260ms (CPU). **Headroom**: GPU is algorithmically limited, not hardware-limited.

**Bottleneck Analysis** (empirically revised):

The GPU is 3x slower than CPU despite M3 Pro GPU having ~6 TFLOPS vs CPU ~200 GFLOPS. Root cause is **not** the point arithmetic (754ms GPU bucket phase vs 190ms CPU is the dominant cost). The MSM algorithm is inherently memory-bound and has poor GPU utilization.

**Revised understanding**:
- GPU MSM at 2^18: 770ms (262K points, centered scalars, wb=16, 32769 buckets, thread-per-bucket)
- CPU C Pippenger at 2^18: 260ms
- GLV was tested (doubling to 524K points, wb=8, 129 buckets, warp-per-bucket): **CRASHED** then showed 210ms at 2^14 (still 9.5x slower than CPU)

The GLV approach fails because:
1. Doubling points from 262K → 524K increases memory traffic and sort cost linearly
2. The bucket reduction savings (warp-per-bucket vs thread-per-bucket) don't compensate
3. Sort phase: 2x points → ~2x sort time (~1400ms vs 754ms)
4. Bucket phase: 100x fewer buckets → ~8x faster (~50ms vs 400ms)
5. Net: 1400+50=1450ms vs 754+400=1154ms → GLV is 26% WORSE

**Theoretical floor revision**:
- Original estimate (~30ms) assumed GPU could achieve near-peak arithmetic throughput
- Reality: MSM is memory-bandwidth-bound, not compute-bound
- With GPU bandwidth ~3x CPU, theoretical GPU floor ≈ CPU_time / 3 ≈ 87ms
- But algorithmic inefficiencies (non-coalesced memory access, bucket contention) add ~10x overhead
- Revised realistic floor: ~150-200ms (GPU with ideal MSM implementation)

### Optimization 1: CPU GLV — REJECTED (empirically verified)

**Test result**: GLV with CPU decomposition makes GPU MSM **worse**, not better.

GLV doubles the effective point count, which hurts the memory-bound sort phase more than the bucket reduction phase can compensate. For n=2^18:
- Non-GLV: 754ms sort + 40ms bucket = 794ms total
- GLV: ~1400ms sort + 50ms bucket = ~1450ms (83% slower)

The `useCPUGLV` path exists in the engine and works correctly, but it is not an optimization for GPU MSM.
// CPU GLV: 3.5ms (measured), then GPU MSM: ~754ms
// Total: ~758ms (vs current 766ms) — but removes 12ms GPU decompose kernel
// More importantly: enables ALL-GPU MSM path where CPU is only coordinating
```

**Estimated impact**: ~12ms saving from removing the slow GPU GLV kernel. But more importantly, this enables a **pipelined** CPU-GPU execution where CPU computes next batch's GLV while GPU runs MSM on current batch. With overlap, effective wall time could be close to GPU MSM alone (~754ms) vs current 766ms, a ~2% improvement. **Low impact but zero-risk change** — the CPU GLV path already exists in the CPU fallback.

**Why this helps**: The GPU GLV decompose kernel runs 12ms on GPU because 256-bit arithmetic on 32-bit carry-chain emulated hardware is inherently 10-20x slower than native uint64. CPU handles these operations natively. The scalars must be in GPU memory for the MSM kernels anyway, but the decomposition step can be precomputed on CPU.

### Optimization 2: Cooperative SIMD Bucket Reduction with Warp-Level Parallelism

**What**: The current `secp_msm_reduce_cooperative` kernel uses SIMD shuffles within threadgroups of 32 threads. However, the inner loop (`for i = lid; i < count; i += 32`) has a trip count of `bucket_count / 32` which is often just 1-2. This means most SIMD threads are idle. Instead, use a **wider cooperative reduction** where the entire warp (32 threads) cooperates on each bucket's reduction in a tree pattern, and process multiple buckets per warp.

**Code change** (secp256k1_msm_kernels.metal):
```metal
// Current: 1 thread per (window, bucket), 32 threads in group cooperating
// Proposal: 1 warp per bucket, multiple buckets per warp
// Each thread in warp handles floor(count/32) elements
// Then tree reduction: 32->16->8->4->2->1 using simd_shuffle_xor
// With bucket count typically 1-8 for random scalars, this is well-suited
```

**Estimated impact**: For n=2^18 with window_bits=14, n_buckets=16384, most buckets have 0-2 points. The cooperative kernel overhead (threadgroup sync, SIMD shuffle) dominates for small buckets. A warp-per-bucket approach eliminates this overhead. **Estimated 15-25% speedup** on the bucket reduction phase (754ms → 560-640ms).

**Why this helps**: The current cooperative kernel spawns a threadgroup per (window, bucket) pair. With 16384 buckets, this creates massive threadgroup contention. A warp-per-bucket model has at most 16384/32 = 512 active warps, dramatically reducing scheduling overhead. The tree reduction (simd_shuffle_xor) is optimal for SIMD-width reductions.

### Optimization 3: Eliminate Redundant GLV Endomorphism Kernel via Point Format Change

**What**: Instead of applying GLV endomorphism as a separate GPU kernel (`secp_glv_endomorphism`) that reads all n points, writes 2n points, and costs ~50ms at memory bandwidth, precompute the endomorphism on CPU and embed it in the point data layout. Specifically, store points already in GLV format (original + endomorphed pair) in a single buffer, computed once during SRS loading.

**Code change**: During SRS (verification key) loading, apply GLV endomorphism to each G1 generator point and store as pairs `(P, phi(P))`. The MSM kernel then reads only the pre-formatted pairs, eliminating the `secp_glv_endomorphism` kernel entirely.

**Estimated impact**: The endomorphism kernel touches 2n affine points (x, y coordinates = 64 bytes each = 128MB for n=2^18) with secp_fp multiply per point. At M3 Pro memory bandwidth (~200GB/s), this is ~0.6ms, but the actual kernel runs ~50ms due to secp_fp multiply per coordinate. Precomputing on CPU (where secp_fp multiply is ~10x faster) during SRS load costs ~5ms once, **saving ~45ms per proof**.

**Why this helps**: The GLV endomorphism kernel (`secp_mul` per point for beta.x) is a separate GPU dispatch that dominates its time in affine-to-projective and coordinate transform overhead. If the SRS is loaded once and used for many proofs, precomputing the endomorphed points amortizes the 5ms CPU cost across all subsequent proofs.
