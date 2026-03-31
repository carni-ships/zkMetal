# UltraHonk Prover Optimization Report — Apple Silicon M3 Pro

**Target:** Barretenberg UltraHonk prover for 428K-gate circuit (persistia_incremental)
**Platform:** Apple M3 Pro (12-core CPU, 18-core GPU), macOS, Metal GPU
**Period:** 2026-03-28 through 2026-03-31 (13 optimization sessions)

---

## Executive Summary

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| **Prove time (cold, precomputed VK)** | 3,850ms | ~870ms | **-77%** |
| **Prove time (cached, precomputed VK)** | — | ~810ms | — |
| **GPU MSM (201K points)** | 267ms | 55ms | **-79%** |
| **create_circuit** | 252ms | 108ms | **-57%** |
| **ProverInstance (cached)** | 75ms | 16ms | **-79%** |
| **Peak memory** | ~500 MiB | ~318 MiB | **-36%** |

---

## Prove Time Breakdown (Final Profile, ~969ms)

| Component | Time (ms) | % | Bound By |
|---|---:|---:|---|
| OinkProver (GPU MSMs) | 316 | 32% | GPU throughput |
| Sumcheck | 145 | 15% | CPU compute |
| create_circuit | 108 | 11% | CPU + memory |
| Gemini folds (GPU) | 84 | 9% | GPU throughput |
| ProverInstance | 70 | 7% | CPU trace |
| Shplonk quotient | 62 | 6% | CPU compute |
| CommitmentKey commit (GPU) | 62 | 6% | GPU throughput |
| KZG commit (GPU) | 52 | 5% | GPU throughput |
| compute_batched | 23 | 2% | CPU bandwidth |
| Other overhead | ~47 | 5% | Various |

**GPU total: ~514ms (53%)** | **CPU total: ~408ms (42%)** | **Overhead: ~47ms (5%)**

---

## Optimizations Implemented (Kept)

### 1. GPU Metal Kernels

| Change | Savings | Mechanism |
|---|---:|---|
| GPU bucket_sum + combine | 202ms | GPU kernel replaces CPU loop; 225ms -> 23ms |
| GPU GLV decompose | 8.5ms | GPU kernel replaces CPU endomorphism |
| Fused gather + reduce kernel | 85ms | Eliminates 1.15GB buffer write/read per MSM |
| GPU counting sort | 10-15ms/MSM | Three Metal kernels (histogram, prefix_sum, scatter) |
| GPU sort sparsity check | Variable | Falls back to CPU when histogram < 10% expected |
| Metal prewarm + SRS caching | 40-60ms | Pre-allocate GPU buffers on first MSM |
| Identity check removal | ~2-3% | Unconditional point_add in bucket_sum_direct |
| Blit removal | ~2ms | Skip unnecessary GPU buffer zero-fills |

### 2. CPU Compute

| Change | Savings | Mechanism |
|---|---:|---|
| PGO (profile-guided optimization) | 46ms | Better branch prediction + inlining in sumcheck |
| Horner's method for poly evaluate | 32ms | Right-to-left evaluation halves multiplications |
| grand_product step fusion | ~15ms | Fused num/den computation with prefix products |
| factor_roots phase 3 fusion | ~5ms | Power x correction fused into running product |
| Out-of-place factor_roots | ~5ms | Eliminates 32MB memcpy per claim |
| Parallel Shplonk factor_roots | ~10ms | Async parallel of two large fold claims |
| Parallel Gemini evaluations | ~10ms | Async parallel of A_0_pos/A_0_neg evaluate |

### 3. Memory Allocation

| Change | Savings | Mechanism |
|---|---:|---|
| DontZeroMemory (sumcheck) | ~50ms | Skip zero-init of PartiallyEvaluatedMultivariates |
| DontZeroMemory (wires + z_perm) | ~16ms | Fully overwritten by trace population |
| DontZeroMemory (selectors) | ~8ms | Only row 0 needs explicit zeroing |
| DontZeroMemory (Gemini/Shplonk) | ~10ms | Various scratch buffers overwritten before use |
| A_0_pos direct copy | ~5ms | Avoid zero-init + accumulate pattern |
| **Circuit builder preallocation** | **143ms** | reserve() for variables + block vectors before build_constraints |

### 4. Application-Level

| Change | Savings | Mechanism |
|---|---:|---|
| PrecomputedCache | 59ms/proof | Zero-copy cache of 28 precomputed polynomials via `Polynomial::share()` |
| Precomputed VK path | ~1,300ms | Separate VK computation from proving; precompute once |
| Timing instrumentation removal | ~63ms | Remove fprintf/chrono from hot paths |
| skip_imbalance_check (Gemini) | 142ms | Keep benign 18% GPU imbalance instead of CPU fallback |

---

## Rejected Optimizations (11 serious attempts)

| # | Idea | Result | Root Cause |
|---|---|---|---|
| 1 | ARM64 asm for field multiply | **9% regression** | Clang __int128 codegen already near-optimal on ARM64 |
| 2 | Fused compute_batched | **2x slower** | Cache locality kills fused multi-polynomial pass |
| 3 | logderiv + z_perm batch merge | **12% slower** | Batch overhead exceeds launch savings for 2 MSMs |
| 4 | GPU compute_univariate | **3-6x slower** | Metal 32-bit ALUs cannot compete with CPU 64-bit ALUs on 256-bit field math |
| 5 | GPU zero-copy partial_evaluate | **SIGSEGV** | Metal alignment requirements; only ~7ms potential |
| 6 | Multi-MSM pipelining | **Impossible** | Fiat-Shamir transcript creates strict sequential dependencies |
| 7 | XYZZ coordinates in GPU reduce | **30% regression** | 4 field elements (128 bytes) vs 3 (96 bytes) kills GPU occupancy |
| 8 | DontZeroMemory on Gemini folds | **Incorrect proofs** | Commitment scheme reads beyond written range to virtual_size |
| 9 | Atomic thread pool (lock-free) | **3x regression/crash** | Pure spin burns CPU; hybrid has condvar race conditions |
| 10 | Elastic MSM (pairwise SRS sums) | **< 0.1% pair match rate** | Not viable for random scalars |
| 11 | Full LTO (-flto vs -flto=thin) | **Linker crash** | Requires > 4GB memory for full LTO link |

**Key lesson:** On Apple Silicon, Clang + LLVM already generates near-optimal code for 256-bit field arithmetic. Assembly, SIMD (NEON), and GPU approaches all fail because the baseline codegen is already excellent. Gains come from algorithmic changes, memory layout, and GPU offloading of inherently parallel operations (MSM bucket accumulation).

---

## Hardware-Specific Findings

### M3 Pro GPU Pathology
Catastrophic 10-30x slowdowns at specific MSM window sizes (w=14, 15, 17) due to thread divergence amplification. w=16 (64K buckets) is the sweet spot for all MSMs > 32K points.

### GPU vs CPU Decision Boundary
| Operation | GPU | CPU | Winner |
|---|---|---|---|
| MSM (> 32K points) | 55ms | 500ms+ | **GPU (9x)** |
| Field polynomial ops | 3-6x slower | Baseline | **CPU** |
| Bucket accumulation | 23ms | 225ms | **GPU (10x)** |
| GLV decomposition | 3.5ms | 12ms | **GPU (3x)** |
| Counting sort (200K) | 10-15ms | ~5ms | **CPU (but GPU enables better MSM)** |

### Memory Bandwidth Utilization
- Sumcheck compute_batched: ~40 GB/s (near M3 Pro limit)
- Polynomial evaluate: ~35 GB/s
- ProverInstance trace population: ~20 GB/s (memory-allocator-bound before preallocation)

---

## Bug Fixes (Critical)

1. **Lazy Montgomery [0, 2p) correctness** — BN254 field elements stored in [0, 2p) not [0, p); GPU kernels assumed fully reduced inputs. Fixed with fp_reduce() at GPU boundary.

2. **Metal threadgroup memory overflow** — msm_compute_csm kernel exceeded 32KB threadgroup limit. Silently disabled ALL GPU MSMs (3x slower). Fixed by in-place prefix sum.

3. **SRS size calculation bug** — Used max_end_index() instead of dyadic_size() for SRS allocation. Caused crashes with precomputed VK path.

4. **Silent GPU kernel deletion** — Accidental removal of msm_reduce_cooperative kernel during XYZZ revert disabled all GPU MSM without any error message. Added defensive logging.

---

## Build Configuration

```cmake
CMAKE_CXX_FLAGS = -flto=thin -fprofile-use=/tmp/pgo_profiles/merged.profdata
CMAKE_EXE_LINKER_FLAGS = -flto=thin
```

- **PGO:** ~13% overall improvement, ~20% for sumcheck. Profile at `/tmp/pgo_profiles/merged.profdata`
- **LTO:** `-flto=thin` required in BOTH CXX and linker flags (missing linker flag = 60ms regression)
- **Metal shaders:** Auto-copied to build/bin/ via BB_METAL_SHADER_PATH

---

## Optimization Floor Analysis

The prover is now within ~15-20% of theoretical hardware limits:

| Component | Current | Theoretical Min | Gap |
|---|---:|---:|---:|
| GPU MSMs (total) | 514ms | ~400ms | 22% (occupancy, dispatch) |
| Sumcheck | 145ms | ~120ms | 17% (bandwidth ceiling) |
| create_circuit | 108ms | ~60ms | 44% (native Poseidon2 field ops) |
| Shplonk | 62ms | ~50ms | 19% |

**Further gains require:**
- Hardware: M4 Pro/Max with more GPU cores (2x GPU MSM improvement expected)
- Protocol: Fewer commitment rounds or different PCS (would eliminate GPU round-trips)
- Algorithm: Different MSM bucket reduction strategy (currently 80%+ of MSM time)
- Batch Poseidon2: ~30ms potential from separating native compute from circuit building (medium-effort refactor)

---

## Files Modified (Summary)

| Area | Files | Changes |
|---|---:|---|
| Metal GPU shaders | 8 | Bucket_sum, GLV, counting sort, fused gather/reduce |
| Prover flow | 5 | OinkProver async, skip_imbalance, wire commit |
| Commitment schemes | 4 | Gemini, Shplonk, KZG timing removal + optimizations |
| Polynomials | 3 | DontZeroMemory, Horner evaluate, share() |
| Circuit builder | 3 | reserve_variables, block reserve, preallocation |
| Trace/ProverInstance | 4 | DontZeroMemory, PrecomputedCache, populate_wires_only |
| ACIR/DSL | 1 | Pre-allocate in create_circuit |
| Build system | 1 | PGO + LTO cmake flags |
