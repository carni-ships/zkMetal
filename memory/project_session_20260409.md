---
name: Session 2026-04-09
description: Recovered crashed session, Karatsuba 26% MSM win, sumcheck wait removal, basefold CB merge, backlog resolved
type: project
---

Recovered from crashed session. Key changes:

1. **Karatsuba Montgomery mul** — fp_mul_karatsuba + fr_mul_karatsuba in Metal shaders. Wired into point_add_mixed_unsafe. 26% MSM speedup (601ms→444ms at 2^18). Tests pass.

2. **Batch affine bucket reduce** — Implemented but REJECTED (3x slower due to fp_inv cost). Kernel removed from msm_kernels.metal.

3. **CPU/GPU crossover threshold** — Bumped 2048→8192 in MSMEngine, MultiMSM, KZGEngine.

4. **GPU sort pipeline chaining** — Infrastructure built (gpu_prefix_sum_per_window kernel, single-CB chain). DISABLED: pre-existing GPU sort scatter has correctness bugs (atomic non-determinism → wrong results, not just reordering).

5. **Sumcheck reduce table wait removal** — Removed waitUntilCompleted() from reduceBN254Table. Added waitForPendingReduce(). Metal queue ordering handles synchronization.

6. **Basefold fold+Merkle CB merge** — Pre-compute tree metadata, merge fold+merkle into single command buffer. One wait instead of two.

7. **Backlog resolution** — Marked hybrid GPU-sort CPU-accumulate (already implemented), cooperative reduce ALL (already implemented), FRI coalesced (low priority), Blake3 fused merkle (not worth it), streaming merkle (superseded).
