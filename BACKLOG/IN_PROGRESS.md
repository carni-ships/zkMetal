# In Progress

## Priority Ranking (Updated 2026-04-14)

| Transformation | Impact | Cost | Risk | Status |
|---------------|--------|------|------|--------|
| 1. Batch Small MSMs | **Very High** | Medium | Low | ✅ Complete (impl + tests, blocked by shader compile) |
| 2. NAF Representation | Medium | Low | Very Low | ✅ Complete (reference impl) |
| 3. Higher Radix + Shared Mem | Medium-High | Medium | Medium | ✅ Complete (kernel added) |
| 4. Bucket-Interleaved Layout | Medium | Medium | Low | ✅ Complete |
| 5. GPU Fused Sumcheck Round | Medium | Medium | Low | ✅ Complete |
| 6. GPU CSR Sparse Matvec | **Highest** | Very High | Medium | In Progress (3-4 wks) |
| 7. GPU Additive FFT (GF2^8) | High | Medium | Low | In Progress |
| 8. Precomputed Window Tables | High | High | Medium | Rejected (memory) |
| 9. GLV + Batch Small | Unknown | Medium | Medium | Uncertain |

## Current Folding State

| Variant | GPU Status | Benchmark |
|---------|-----------|-----------|
| Nova | GPU folds via GPUNovaFoldEngine | ~0.66ms/fold (1-constraint), ~5.9ms/fold (256-constraint) |
| HyperNova | GPU via NEON batch ops | 0.09ms/fold (1000 steps) |
| Supernova | GPU via NEON batch ops | ~0.67ms/fold |

## Remaining Folding Opportunities

| Idea | Impact | Effort | Notes |
|------|--------|--------|-------|
| GPU CSR sparse matvec | **Highest** | Very High (3-4 wks) | Main bottleneck - 6 matvecs per fold |
| GPU fused sumcheck round | Medium | Medium (3-4 days) | Fuses eq-weighting with fold |
| Bucket-Interleaved Layout | Medium | Medium | 15-25% speedup for secp256k1 |
