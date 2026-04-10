# zkMetal Optimization Backlog

Sorted by headroom (most room for improvement first). Items marked with status.

## MSM BN254 (~1.4x headroom, ~72ms at 2^18 vs ~50ms floor)

GPU bucket reduce (35ms) + bucket sum (32ms) = 67ms GPU, 5ms sort, 2ms GLV+endo.

- [x] **Batch affine accumulation** — REJECTED: fp_inv via Fermat costs ~380 fp_mul per inversion, making batch affine 3x slower than projective (83.5ms vs 27.5ms). Kernel retained as reference.
- [x] **Point locality sorting** — MARGINAL: ~1-3ms within noise (see below). Compute-bound, not memory-bound.
- [x] **Multi-window point processing** — REJECTED: Bottleneck is compute (14.7M field muls), not memory BW. 36MB SLC cache absorbs 16MB points buffer. No atomic EC point add on Metal.
- [x] **Shared-memory tiled buckets** — REJECTED: w=16 creates 32769 buckets, only 341 fit in 32KB threadgroup memory. Near-zero cache hit rate.
- [x] **Count-sort fix for w=14/15** — ALREADY IMPLEMENTED: Count-sorting is production code. Pathology is M3 Pro hardware register spilling, not software-fixable.
- [x] **Karatsuba GPU fp_mul** — DONE: 57% total MSM speedup (444ms→~190ms at 2^18). fp_mul_karatsuba wired into ALL point operations (point_add, point_add_mixed, point_double, point_add_unsafe). Cooperative kernel restructured: first-point split, point_add_mixed_unsafe in loop, bool validity flag in SIMD reduce.
- [x] **All-GPU mode** — DONE: Eliminated CPU last-window bottleneck (182ms→72ms at 2^18). CPU bn254_cpu_window_reduce took ~100ms for 1 window while GPU processed 7 windows in 68ms. Also found cooperative CPU path had correctness bug at n≥65536. Disabled cooperative mode; all windows now on GPU.
- [x] **CPU Pippenger crossover benchmark** — DONE: Crossover at 2^13 (8192). Threshold updated from 2048 to 8192 in MSMEngine, MultiMSM, KZGEngine.
- [x] **Precomputed SRS tables (BGMW)** — NOT VIABLE at KZG sizes: table memory = n * numWindows * tableSize * 64B. For n=2^18, w=7: ~40GB. Only practical for small fixed-base (Pedersen hash, etc.), not MSM.
- [x] **Cooperative reduce for ALL buckets** — ALREADY IMPLEMENTED: msm_reduce_cooperative uses SIMD shuffle (simd_shuffle_down_point) tree reduction for all bucket sizes.
- [x] **Redundant representation** — REJECTED: Memory-latency-bound, not compute-bound. 9×29-bit adds 26% mul ops. Only lazy addition viable (5-8% est.) but not worth complexity.
- [x] **Karatsuba in NTT butterflies** — REJECTED: Increases register pressure (60 vs 15 uints), kills GPU occupancy. NTT 2^22 regressed 26→30ms. NTT is memory-bandwidth-bound, not compute-bound.
- [x] **Point locality sorting** — MARGINAL: ~1-3ms within noise. Apple Silicon hides latency via thread parallelism. Confirmed compute-bound.

## NTT BN254 (~9x headroom, 26ms at 2^22 vs ~3ms floor)

Bottleneck: 256-bit Montgomery multiply on 32-bit ALUs + strided memory access.

- [x] **4-step FFT with shared memory** — ALREADY IMPLEMENTED + transpose tested: No improvement. Apple Silicon unified memory handles strides well; transpose overhead cancels coalescing benefit.
- [x] **Fused butterfly kernels** — ALREADY IMPLEMENTED: 10-stage fused kernel exists, 3 dispatches + 1 blit for logN=22.
- [x] **RNS decomposition** — REJECTED: NTT is memory-bandwidth-bound, not compute-bound. RNS saves 3.6x ALU but memory traffic is identical (0.97x).
- [x] **Strided access optimization** — REJECTED: Transpose-first tested, no gain. Unified memory handles strides with ~1.5x penalty; 2 extra transpose passes cancel benefit.

## Basefold open (~7x headroom, 61ms at 2^18 vs ~20ms floor)

18 iterative fold+commit rounds. Merkle tree per round.

- [x] **Wire basefold_fold_fused2 kernel** — DONE: Merged with pipeline overlap. Paired rounds dispatch fused2 + single fold in parallel.
- [x] **Pipeline overlap fold+commit** — DONE: 139.2ms → 123.7ms (-11.2%). Dual command queue, Merkle overlaps fold. 143/143 tests pass.
- [x] **Streaming Merkle** — SUPERSEDED: Fold+Merkle now in single command buffer. Streaming would add complexity for 5-10% gain on non-bottleneck sizes.
- [x] **Fold-by-4** — DONE: Rounds drop from n to ceil(n/2), halving Merkle count. Backward-compatible proof format with levelStrides. 143/143 tests pass.

## FRI Fold (~7x headroom, 2.9ms at 2^20 vs ~0.3ms floor)

- [x] **Fold-by-4 cascade wiring** — DONE: Wired fri_fold_fused2_kernel through all 3 FRI engine layers. 901/901 tests pass.
- [x] **Coalesced memory access** — LOW PRIORITY: FRI domain structure requires strided access (i, i+n/2). Transpose would add O(n) overhead. Apple Silicon hides latency via thread parallelism.

## Sumcheck (~5x headroom, 3.3ms at 2^20 vs ~1ms floor)

- [x] **GPU final reduction** — DONE: Added sumcheck_final_reduce_bn254 kernel to offload final partial sum reduction to GPU. Eliminates CPU-side loops in proveRoundBN254, computeRoundPolyBN254, and fullSumcheckBN254. 3.56ms → 3.28ms at 2^20 (~8%).

- [x] **Tune CPU crossover + hybrid GPU→C** — ALREADY OPTIMAL: crossover at numVars<=14, hybrid handoff at 1024 elements. Fused2 + SIMD shuffle reduction already implemented.
- [x] **Threadgroup memory for round_poly reduction** — ALREADY IMPLEMENTED: SIMD shuffle + inter-SIMD shared memory.
- [x] **Multi-round batching** — ALREADY IMPLEMENTED: fused2_strided (2 rounds/dispatch), fusedMultiround (up to 8 rounds in threadgroup mem).

## Blake3 Batch (~1.7x headroom, 3.2ms at 2^20 vs ~0.6ms floor)

Bandwidth-limited: 96MB traffic at 2^20.

- [x] **Vectorized uint4 loads + cycle permute + specialized compress** — DONE: 3.5ms → 1.0ms (3.5x). uint4 loads, in-place cycle permute (-15 registers), fused IV compress. 67% of peak BW.
- [x] **Fused Merkle tree** — NOT WORTH IT: Blake3MerkleEngine already partially fused (bottom 10 levels in shared memory). Leaf hashing is separate for flexibility. ~0.5ms gain on a non-bottleneck.
- [x] **Coalesced memory access** — NOT WORTH IT: Would require data layout changes for marginal gain on non-bottleneck.

## Poseidon2 (~4.5x headroom, 7.4ms at 2^16 vs ~1.8ms floor)

- [x] **SIMD group (warp-level) optimization** — REJECTED: BN254 needs 8 shuffles per Fr (multi-limb). BabyBear/M31 wastes 93.75% threads in partial rounds. S-box is bottleneck, not MDS.
- [x] **Karatsuba S-box** — REJECTED: fr_mul_karatsuba increases register pressure (60 vs 15 uints), kills GPU occupancy. 2^16 regressed 8.1→9.9ms. Same pattern as NTT.

## General / Cross-cutting (audit complete - all major optimizations exhausted)

- [x] **MSM GPU sort pipeline chaining** — BLOCKED: gpu_sort_scatter has correctness bugs (non-deterministic results). CPU sort is proven correct and fast (~2ms). GPU prefix sum kernel added but chaining disabled until scatter is fixed.
- [x] **Sumcheck reduce table wait removal** — DONE: Removed waitUntilCompleted() from reduceBN254Table. Metal command queue ordering guarantees fold CB completes before next round's computeRoundPolyBN254 CB. Callers wait once after loop via waitForPendingReduce().
- [x] **NTT encode API migration** — DONE: BN254 extend() + batchExtend() + BabyBear extend() now use encodeNTT + blit copy (GPU-only). BabyBearNTTEngine gained encodeNTT/encodeINTT. 48/48 coset LDE tests, 167/167 NTT tests pass.
- [x] **Basefold fold+Merkle CB merge** — DONE: Pre-compute tree metadata, merge fold+merkle into single command buffer. One wait instead of two.
- [ ] **MTLEvent infrastructure** — REVISED: Not viable. Analysis shows most waits ARE necessary - they protect CPU access to GPU results. The 18 "eliminable" waits estimate was wrong; nearly all 336 waits are correctness必需的. Engines already pipeline internally (fold+Merkle in one CB with memory barriers). Cross-engine event-based pipelining would require massive API redesign for negligible gain.
- [ ] **Metal async compute** — BLOCKED (fundamental): Fiat-Shamir creates sequential round dependencies - round N+1 cannot start until round N's challenge is known. Kernel fusion already maximized: sumcheck_fused_round_reduce_bn254 computes round_poly+fold in ONE dispatch. Even if fused, next round must wait for previous fold result. No independent CPU work available during GPU execution to overlap. Prover structure is fundamentally sequential.
- [ ] **Smaller point representation** — REJECTED: GPU memory bandwidth (300+ GB/s) far exceeds compute for MSM (compute-bound, 14.7M field muls). Decompression cost (sqrt per point) exceeds bandwidth savings. Only viable for slow bus (PCIe, disk).

**Summary:** All major optimizations exhausted. Remaining items blocked by fundamental Fiat-Shamir sequentiality or correctness requirements. No viable kernel fusion opportunities remain - per-round fusion already maximized.
