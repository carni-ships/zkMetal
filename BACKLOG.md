# zkMetal Optimization Backlog

Sorted by headroom (most room for improvement first). Items marked with status.

## MSM BN254 (~11x headroom, 54ms at 2^18 vs ~5ms floor)

93.8% of time is GPU bucket reduce. Memory-latency-bound (random 64B reads across 8MB buffer), 217x above ALU floor.

- [ ] **Batch affine accumulation** — Montgomery's trick: 2 muls + amortized inversion per add vs 12 muls projective. 6x ALU reduction. (Agent running)
- [ ] **Point locality sorting** — Reorder points buffer so bucket-adjacent points are memory-contiguous. Eliminates cache thrashing in reduce kernel. O(n) reorder cost. (Agent running)
- [x] **Multi-window point processing** — REJECTED: Bottleneck is compute (14.7M field muls), not memory BW. 36MB SLC cache absorbs 16MB points buffer. No atomic EC point add on Metal.
- [x] **Shared-memory tiled buckets** — REJECTED: w=16 creates 32769 buckets, only 341 fit in 32KB threadgroup memory. Near-zero cache hit rate.
- [x] **Count-sort fix for w=14/15** — ALREADY IMPLEMENTED: Count-sorting is production code. Pathology is M3 Pro hardware register spilling, not software-fixable.
- [ ] **Karatsuba GPU fp_mul** — Split 8×8 schoolbook (64 mul32) into 3×(4×4) Karatsuba (48 mul32). 25% ALU reduction. Risk: register pressure. (Agent running)
- [ ] **Hybrid GPU-sort CPU-accumulate** — GPU phases 0-1, CPU phases 2-4 using all 12 cores with native 64-bit. (Agent running)
- [ ] **CPU Pippenger crossover benchmark** — Find where GPU actually beats CPU; threshold may be much higher than 2048. (Agent running)
- [ ] **Precomputed SRS tables** — For fixed-point KZG, precompute w-bit lookup tables to eliminate bucket accumulation entirely
- [ ] **Cooperative reduce for ALL buckets** — SIMD tree reduction via simd_shuffle_down instead of serial per-bucket loops
- [ ] **Redundant representation** — Larger-than-32-bit limbs to defer carry propagation
- [x] **Point locality sorting** — MARGINAL: ~1-3ms within noise. Apple Silicon hides latency via thread parallelism. Confirmed compute-bound.

## NTT BN254 (~9x headroom, 26ms at 2^22 vs ~2.9ms floor)

Bottleneck: 256-bit Montgomery multiply on 32-bit ALUs + strided memory access.

- [x] **4-step FFT with shared memory** — ALREADY IMPLEMENTED + transpose tested: No improvement. Apple Silicon unified memory handles strides well; transpose overhead cancels coalescing benefit.
- [x] **Fused butterfly kernels** — ALREADY IMPLEMENTED: 10-stage fused kernel exists, 3 dispatches + 1 blit for logN=22.
- [x] **RNS decomposition** — REJECTED: NTT is memory-bandwidth-bound, not compute-bound. RNS saves 3.6x ALU but memory traffic is identical (0.97x).
- [x] **Strided access optimization** — REJECTED: Transpose-first tested, no gain. Unified memory handles strides with ~1.5x penalty; 2 extra transpose passes cancel benefit.

## Basefold open (~7x headroom, 138ms at 2^18 vs ~20ms floor)

18 iterative fold+commit rounds. Merkle tree per round.

- [x] **Wire basefold_fold_fused2 kernel** — DONE: Merged with pipeline overlap. Paired rounds dispatch fused2 + single fold in parallel.
- [x] **Pipeline overlap fold+commit** — DONE: 139.2ms → 123.7ms (-11.2%). Dual command queue, Merkle overlaps fold. 143/143 tests pass.
- [ ] **Streaming Merkle** — Build Merkle tree incrementally as fold results stream out.
- [x] **Fold-by-4** — DONE: Rounds drop from n to ceil(n/2), halving Merkle count. Backward-compatible proof format with levelStrides. 143/143 tests pass.

## FRI Fold (~7x headroom, 1.96ms at 2^20 vs ~0.3ms floor)

- [x] **Fold-by-4 cascade wiring** — DONE: Wired fri_fold_fused2_kernel through all 3 FRI engine layers. 901/901 tests pass.
- [ ] **Coalesced memory access** — Ensure fold reads are contiguous (current interleaved layout may cause strided access).

## Sumcheck (~6x headroom, 4.9ms at 2^20 vs ~0.85ms floor)

- [x] **Tune CPU crossover + hybrid GPU→C** — ALREADY OPTIMAL: crossover at numVars<=14, hybrid handoff at 1024 elements. Fused2 + SIMD shuffle reduction already implemented.
- [x] **Threadgroup memory for round_poly reduction** — ALREADY IMPLEMENTED: SIMD shuffle + inter-SIMD shared memory.
- [x] **Multi-round batching** — ALREADY IMPLEMENTED: fused2_strided (2 rounds/dispatch), fusedMultiround (up to 8 rounds in threadgroup mem).

## Blake3 Batch (~6x headroom, 3.5ms at 2^20 vs ~0.6ms floor)

Bandwidth-limited: 96MB traffic at 2^20.

- [x] **Vectorized uint4 loads + cycle permute + specialized compress** — DONE: 3.5ms → 1.0ms (3.5x). uint4 loads, in-place cycle permute (-15 registers), fused IV compress. 67% of peak BW.
- [ ] **Fused Merkle tree** — Combine hash + tree building to avoid intermediate writes.
- [ ] **Coalesced memory access** — Verify AoS vs SoA layout.

## Poseidon2 (~4.5x headroom, 8.1ms at 2^16 vs ~1.8ms floor)

- [x] **SIMD group (warp-level) optimization** — REJECTED: BN254 needs 8 shuffles per Fr (multi-limb). BabyBear/M31 wastes 93.75% threads in partial rounds. S-box is bottleneck, not MDS.

## General / Cross-cutting (audit: 336 waits, ~18 eliminable, ~12ms savings)

- [ ] **MSM GPU sort pipeline chaining** — Replace 3-step endo-wait/histogram-wait/scatter-wait with single CB using GPUPrefixSumEngine. ~1.2ms/MSM, ~7.2ms across 6 calls. HIGHEST IMPACT.
- [ ] **Sumcheck reduce table wait removal** — reduceBN254Table returns MTLBuffer that goes straight to GPU. Metal guarantees coherence for GPU→GPU. ~0.4ms/round savings.
- [ ] **NTT encode API migration** — Switch ntt()/intt() callers to encodeNTT()/encodeINTT() when result feeds another GPU op. Infrastructure exists. ~0.8ms per NTT-mul-INTT pipeline.
- [ ] **Basefold fold+Merkle CB merge** — Combine fold and Merkle command buffers with memory barriers. ~0.5ms savings.
- [ ] **MTLEvent infrastructure** — GPUEventFence wrapper for cross-engine dependencies without CPU waits. Unlocks ~20-30 more wait eliminations long-term.
- [ ] **Metal async compute** — Multiple command queues for independent operations (partially done in Basefold pipeline overlap)
- [ ] **Smaller point representation** — Compressed affine (33B vs 64B) for 2x bandwidth reduction. Trade: sqrt decompression cost.
