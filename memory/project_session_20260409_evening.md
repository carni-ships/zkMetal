---
name: Session 2026-04-09 evening
description: NTT encode API migration completed, remaining backlog items assessed as blocked/rejected
type: project
---

## Session 2026-04-09 evening — NTT encode API migration

### What was done

**NTT encode API migration (GPUCosetLDEEngine.swift)**:
- BN254 `extend()`: replaced `try engine.ntt(shifted)` with blit copy + encodeNTT (GPU-only, no CPU round-trip)
- BN254 `batchExtend()`: per-column blit copy + encodeNTT loop (GPU-only per column)
- BabyBear `extend()`: left unchanged — BabyBearNTTEngine lacks encodeNTT method

**Verification**: 48/48 coset LDE tests, 167/167 NTT tests, 1408/1408 plonk tests all pass.

**BACKLOG updated**: NTT encode API marked done (partial). Remaining items assessed:
- MTLEvent infrastructure: BLOCKED — requires engine API redesign across 116 files with waitUntilCompleted
- Metal async compute: BLOCKED — prover ops are sequential data dependencies
- Smaller point representation: REJECTED — GPU memory bandwidth (300+ GB/s) >> compute bottleneck; decompression cost exceeds savings

**Two commits pushed**:
1. `6d8c5f0` — NTT encode API: BN254 extend + batchExtend GPU-only path
2. `248765a` — BACKLOG: NTT encode API done, remaining items blocked/rejected

### Why remaining items were skipped

1. **MTLEvent**: 350 occurrences of waitUntilCompleted across 116 files. The bottleneck is not the wait itself but that most callers need the result back on CPU (Swift arrays). MTLEvent only helps for GPU→GPU chaining without CPU involvement. Would require major engine API redesign.

2. **Metal async compute**: Metal command queues execute in parallel on GPU. But provers are sequential - each step depends on previous output. Async compute only helps for truly independent operations. Basefold already has dual CB overlap which is the practical limit.

3. **Smaller point repr**: MSM at 2^18 has 262K points. Current PointAffine = 64 bytes = 16 MB. Compressed = 33 bytes = 8.6 MB. Saves 8 MB of 300 GB/s GPU bandwidth. Decompression (sqrt + muls per point) costs more than bandwidth savings. Compute-bound, not mem-BW bound.
