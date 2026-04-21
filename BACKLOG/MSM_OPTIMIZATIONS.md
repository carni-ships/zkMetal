# MSM Optimization Backlog

## 1. GPU Horner Combine
**Impact**: ~221ms savings (45% at 2^20 scale)
**Description**: Offload Horner polynomial evaluation from CPU to GPU
**Status**: Ideas

## 2. Multi-threaded GLV Scalar Decomposition
**Impact**: ~49ms savings at 2^20 scale
**Description**: Parallelize the GLV endomorphism and signed digit decomposition across CPU threads
**Status**: Ideas

## 3. Larger GPU Threadgroups
**Impact**: ~10-20% GPU improvement
**Description**: Experiment with larger threadgroup sizes for better occupancy at large scales (2^20)
**Status**: Ideas

## 4. Batch GLV Processing
**Impact**: Memory bandwidth improvement
**Description**: Process multiple windows concurrently to better utilize memory bandwidth
**Status**: Ideas

## 5. GPU Sort Verification Bypass
**Impact**: ~65ms savings at 2^20 scale
**Description**: The GPU sort currently runs both GPU and CPU then compares. For production, skip verification
**Status**: Ideas

## 6. NEON Vectorized Horner
**Impact**: ~100ms savings on CPU path
**Description**: Use NEON SIMD to vectorize the Horner combine on ARM
**Status**: Ideas

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

## 1. Rust bindings for C Pippenger
**Impact**: Enable Rust to use C Pippenger MSM
**Description**: Currently Rust tests can't link to NeonFieldOps. Need proper static library
**Status**: Ideas

## 2. Circle STARK prover cache warmup
**Impact**: Faster subsequent proof generation
**Description**: Pre-warm GPU buffers and caches before proof generation
**Status**: Ideas

## 3. Poseidon2 M31 optimization
**Impact**: Hash throughput improvement
**Description**: Research faster Poseidon2 permutation for Mersenne31 field
**Status**: Research