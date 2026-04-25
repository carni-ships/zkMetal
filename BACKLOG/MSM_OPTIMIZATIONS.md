# MSM Optimization Backlog

## 1. GPU Horner Combine ✅ DONE
**Impact**: ~221ms savings (45% at 2^20 scale)
**Description**: Offload Horner polynomial evaluation from CPU to GPU
**Status**: Implemented (commit 64d0046e)

## 2. Multi-threaded GLV Scalar Decomposition ❌ N/A
**Impact**: ~49ms savings at 2^20 scale
**Description**: Parallelize the GLV endomorphism and signed digit decomposition across CPU threads
**Status**: GLV is already on GPU, not CPU. No CPU GLV to parallelize.

## 3. Larger GPU Threadgroups ❌ N/A
**Impact**: ~10-20% GPU improvement
**Description**: Experiment with larger threadgroup sizes for better occupancy at large scales (2^20)
**Status**: Already at 256 which is optimal for M3 Pro. Kernel has no shared memory requirements.

## 4. Batch GLV Processing ❌ N/A
**Impact**: Memory bandwidth improvement
**Description**: Process multiple windows concurrently to better utilize memory bandwidth
**Status**: Already batched. GLV kernel processes all scalars in one dispatch.

## 5. GPU Sort ✅ FIXED
**Impact**: ~65ms savings at 2^20 scale (when enabled)
**Description**: GPU sort using RadixSortEngine for deterministic parallel sorting
**Status**: Fixed (Apr 23, 2026)

**Root cause**: Race condition in atomic gpu_sort_scatter kernel.
- GPU scatter used `atomic_fetch_add` on positions array to claim output positions
- Multiple threads could read the same position value before any thread's atomic completed
- This caused indices to be written to wrong positions, producing non-deterministic results

**Fix**: Replaced atomic-based GPU scatter with RadixSortEngine's deterministic parallel sort.
- RadixSortEngine uses SIMD-level ranking + threadgroup barriers (no atomics for ordering)
- For each window: create (digit, index) pairs, sort by digit using GPU radix sort, extract indices
- This is equivalent to the old GPU sort path but without the race condition

**Changes**:
- MSMEngine.swift: Set `useGpuSort = true` to enable RadixSortEngine-based GPU sort
- The old atomic `gpu_sort_scatter` is now only used as fallback when `radixSortEngine == nil`
- Removed the `useGpuSortWithTest` dead code path

## 6. NEON Vectorized Horner ❌ N/A
**Impact**: ~100ms savings on CPU path
**Description**: Use NEON SIMD to vectorize the Horner combine on ARM
**Status**: CPU Horner only used when nWindows <= 1 (scales <= 2^16 with 16-bit windows). GPU Horner used for all practical scales.

---

# P1 FRI Fold-by-8 Optimization Backlog

## 1. Complete commitPhaseFused with all layers
**Impact**: Enable full proof generation with fused commit
**Description**: Current commitPhaseFused only returns 2 layers. Need to capture all intermediate layers
**Status**: Ideas

## 2. Fold-by-16 kernel
**Impact**: Further kernel launch overhead reduction
**Description**: Extend the fold-by-8 cascade to fold-by-16
**Status**: Ideas

## 3. inv2t buffer reuse optimization
**Impact**: Memory allocation savings
**Description**: Pre-allocate and reuse inv2t buffers across FRI rounds
**Status**: Ideas

---

# Other Optimizations

## 1. Rust bindings for C Pippenger ✅ DONE
**Impact**: Enable Rust to use C Pippenger MSM
**Description**: Added ark_fr_to_pippenger_scalar() and bn254_projectiveto_affine() with correctness tests
**Status**: Implemented (commit 4a418ae0)

## 2. Circle STARK prover cache warmup
**Impact**: Faster subsequent proof generation
**Description**: Pre-warm GPU buffers and caches before proof generation
**Status**: Ideas

## 3. Poseidon2 M31 optimization
**Impact**: Hash throughput improvement
**Description**: Research faster Poseidon2 permutation for Mersenne31 field
**Status**: Research

---

# GPU GLV Kernel Bugs (BLS12-377)

**Status:** Fixed (Apr 2026)

Two bugs were found and fixed in `Sources/Shaders/msm/bls12377_glv_kernels.metal`:

### Bug 1: Redundant write and double-negation (line 241-246)
```metal
// Before (broken):
if (neg1_flags[gid]) {
    p.y = fq377_neg(p.y);
    points[gid] = p;  // Redundant write to original location
    p.y = fq377_neg(p.y);  // Undo negation - breaks endomorphism!
}

// After (fixed):
if (neg1_flags[gid]) {
    p.y = fq377_neg(p.y);
}
```

### Bug 2: Incorrect β values (lines 248-273)
```metal
// Before (broken): Hardcoded R mod q = 1 in Montgomery form (NOT β!)
beta.v[0] = 0xffffff68;

// After (fixed): Correct β in Montgomery form from BLS12377GLV.betaMontgomery
Fq377 beta_mont;
beta_mont.v[0]  = 0x5a7b8727; beta_mont.v[1]  = 0x2c766f92;
beta_mont.v[2]  = 0x253d58b5; beta_mont.v[3]  = 0x03d7f6b0;
beta_mont.v[4]  = 0xec122131; beta_mont.v[5]  = 0x838ec0de;
beta_mont.v[6]  = 0xf658bb10; beta_mont.v[7]  = 0xbd5eb3e9;
beta_mont.v[8]  = 0x6ed3e52e; beta_mont.v[9]  = 0x6942bd12;
beta_mont.v[10] = 0xdd04ed6a; beta_mont.v[11] = 0x01673786;
```

**Verification:** All GLV tests pass:
- β³ = 1 in Fq: [pass]
- λ² + λ + 1 = 0 in Fr: [pass]
- φ(G) = λ·G: [pass]
- GLV decomposition roundtrip (10 trials): [pass]

**Performance Note:** GLV is still not recommended for BLS12-377 GPU MSM:
- GLV reduces 253-bit scalars to 128 bits (~50% reduction)
- But requires doubling points (each scalar → two endomorphized points)
- For 12-limb field, point additions are ~2× expensive vs BN254
- At w=15: GLV = 18n work vs Non-GLV = 17n work (**5% MORE work with GLV**)

See `Sources/zkMetal/MSM/BLS12377MSMEngine.swift:51-55` for rationale.

## On-the-fly Endomorphism (GLV Memory Optimization) ✅ DONE (Apr 2026)
**Impact:** Reduces GLV memory from 2n to n points (halves GPU memory for endomorphized points)
**Description:** Instead of precomputing and storing β·P at `points[n+gid]`, compute β·x on-the-fly when loading points in the reduction kernel
**Implementation:**
- Added `point377_get_glv()` helper that checks idx >= n to trigger β·x multiplication
- Uses constant GLV377_BETA_MONT_REDUCE (GPU-cached)
- Added GLV-aware signed digit extraction kernel `msm377_signed_digit_extract_glv`
- Removed precomputed endomorphism preprocessing from GLV pipeline

**Trade-offs:**
- Saves GPU memory: 2n → n points for points buffer
- Adds computation: ~12 fq377_mul per point load (multiply by β in Montgomery form)
- Net benefit: Memory savings > compute cost for large scales

**Files modified:**
- `Sources/Shaders/msm/bls12377_msm_kernels.metal`: Added on-the-fly endomorphism helper
- `Sources/zkMetal/MSM/BLS12377MSMEngine.swift`: Updated GLV pipeline to use on-the-fly computation

## Other GLV Optimization Ideas (Backlog)
**Option 2: Selective GLV based on scale**
- Enable GLV only for n < 2^16, disable for larger scales
- Non-GLV wins at large scales due to 2n point overhead

**Option 3: Batch endomorphism table**
- Precompute endomorphism for a subset of points in local memory
- May not help for random access patterns

**Option 4: Single-table GLV (G1 only)**
- Since G1 endomorphism is trivial (swap x,y for cube root of unity), could optimize further
- For β = exp(2πi/3), point377_get_glv just changes x sign or rotates based on cube root encoding
- BLS12-377's β is more complex (involves field arithmetic)

**Option 5: Window-by-window GLV processing**
- Process k1 and k2 for one window at a time, reducing peak memory
- May improve cache locality for very large scales

## Window Size Investigation (Apr 2026)
**Finding:** w=12,13,14 previously reported as causing regression on M3 Pro.
**Fact:** After investigation, all window sizes (w=12,13,14,15) perform similarly for large n:
- w=15: 182.7 ms at 2^18
- w=14: 175.8 ms at 2^18
- w=13: 177.4 ms at 2^18
- w=12: 178.9 ms at 2^18

**Conclusion:** The regression was likely due to other factors (small n, warmup issues, or measurement noise).
All window sizes are safe to use. w=15 remains the default for historical consistency.

## GPU Register Pressure Solutions (Apr 2026)

**Problem:** BLS12-377 GPU MSM hangs at n >= 4096 with GLV enabled. GPU kernels are scheduled (status=3) but never complete. Root cause is extreme register pressure from 12-limb field operations combined with SIMD shuffle tree reduction in cooperative kernels.

**Root Cause Analysis:**
- BLS12-377 uses 12-limb field representation (vs 8-limb for BN254)
- Each `fq377_add`, `fq377_mul` operates on 12×64-bit limbs
- Cooperative bucket sum kernel uses SIMD shuffle tree + threadgroup barriers
- Combined register pressure exceeds M3 Pro GPU limits
- When registers spill to memory, GPU scheduler can't complete wavefronts

**Solution Approaches Implemented:**

1. **Option 2: Direct bucket sum kernel** ✅ Implemented
   - Replace cooperative kernel with direct kernel for n >= 4096
   - Direct kernel is simpler (no threadgroup sync, no SIMD shuffle tree)
   - Slightly more compute per thread but lower register pressure
   - File: `Sources/zkMetal/MSM/BLS12377MSMEngine.swift:680-720`

2. **Option 5: Increase segments for large n** ✅ Implemented
   - n >= 16384: nSegments = min(1024, nBuckets) - many small segments
   - n >= 4096: nSegments = min(512, nBuckets/2) - medium segments
   - n < 4096: nSegments = min(256, nBuckets/2) - original logic
   - Fewer buckets per segment = simpler reduction per segment
   - File: `Sources/zkMetal/MSM/BLS12377MSMEngine.swift:401-414`

**Other Ideas (Backlog - Not Yet Tested):**

3. **Two-pass bucket reduction**
   - Pass 1: Each thread reduces its buckets to single accumulator
   - Pass 2: Combine accumulators from all threads
   - Simplifies Phase 2 - no complexity of SIMD shuffle tree mid-reduction

4. **Horizontal bucket sum**
   - Each thread processes contiguous block instead of strided buckets
   - Then reduce thread results at the end
   - Simple, lower register pressure

5. **CPU bucket sum + GPU final combine**
   - Move problematic Phase 2 bucket reduction to CPU
   - GPU just does final window combination (Horner)
   - CPU handles 12-limb ops with NEON + cache-friendly access

6. **Separate GLV k1/k2 processing**
   - Process k1 and k2 in separate passes instead of together
   - Simplifies kernel - only handles one scalar component at a time

7. **Pre-bucket partitioning**
   - Pre-partition points into buckets on CPU (or simple GPU kernel)
   - Phase 2 just sums pre-grouped points
   - Moves complexity to sorting phase where registers don't matter