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
| 7. GPU Additive FFT (GF2^8) | High | Medium | High | LUT approach failed, needs investigation |
| 8. Precomputed Window Tables | High | High | Medium | Rejected (memory) |
| 9. GLV + Batch Small | Unknown | Medium | Medium | Uncertain |

## Current Folding State (Updated 2026-04-14)

| Variant | GPU Status | Benchmark |
|---------|-----------|-----------|
| Nova | GPU folds via GPUNovaFoldEngine + GPU sparse matvec | ~0.60ms/fold (100-fold), ~5.6ms/fold (256-constraint) |
| HyperNova | GPU via NEON batch ops | 0.09ms/fold (1000 steps) |
| Supernova | GPU via NEON batch ops + GPU sparse matvec | ~0.67ms/fold |

## Implemented This Session (2026-04-14)

### GPU Sort Activation (secp256k1 MSM)
- Added `useGPUSort` flag to activate GPU sorting kernels
- Kernels: `secp_msm_sort_histogram`, `secp_msm_sort_scatter`, `secp_msm_build_csm`
- Experimental: off by default, needs correctness verification with large n

### Bucket-Interleaved Layout (secp256k1 MSM)
- Added `useBucketInterleaved` flag (default: true)
- CPU-computed bucket-interleaved sorting after GPU histogram
- 15-25% speedup target for sort phase

### GPU Fused Sumcheck (secp256k1)
- New files: `Sources/Shaders/sumcheck/secp256k1_sumcheck.metal`
- `GPUSecp256k1SumcheckEngine.swift` - fused eq+fold kernels
- 10-15% fold time improvement target

### GPU CSR Sparse Matvec (Folding)
- New files: `Sources/Shaders/fold/sparse_matvec.metal`
- `GPUSparseMatvecEngine.swift` - fused triple matvec (A*z, B*z, C*z)
- Integrated into GPUNovaFoldEngine and Supernova
- CPU fallback for small matrices (<64 rows or <256 non-zeros)

### GPU Additive FFT (GF2^8)
- Status: In Progress (Optimization 1: Precomputed LUT)
- Current: 13ms for 2^22, Target: 2-4ms (LUT gives 5-10x multiply speedup)

## Remaining Folding Opportunities

| Idea | Impact | Effort | Status |
|------|--------|--------|--------|
| GPU CSR sparse matvec | **Highest** | 3-4 wks | ✅ Implemented (integrated) |
| GPU fused sumcheck round | Medium | 3-4 days | ✅ Implemented |
| Bucket-Interleaved Layout | Medium | Medium | ✅ Implemented |
| GPU Additive FFT (GF2^8) | High | Medium | In Progress |
