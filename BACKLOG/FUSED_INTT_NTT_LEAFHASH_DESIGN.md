# Fused INTT + NTT + LeafHash Kernel Design

## Overview

This document describes the design and implementation of a fused kernel that combines three computational phases for EVMetal into a single GPU dispatch:

1. **Inverse NTT** (interpolation from evaluation to coefficient form)
2. **Forward NTT** (for extended coset domain)
3. **Leaf Hash** (Poseidon2-M31 Merkle tree authentication path)

## Motivation

Traditional STARK provers execute INTT, NTT, and LeafHash as separate kernel dispatches with barriers between them:

```
[INTT] -> barrier -> [NTT] -> barrier -> [LeafHash]
```

This creates overhead:
- 2 GPU memory barriers
- 2 buffer synchronizations (CPU waits for GPU)
- Intermediate buffer allocations

## Fused Kernel Design

### Data Flow

```
Input: evaluations[0..N-1] (evaluation form)
  |
  v
[Phase 1: INTT Final Stage]
- DIF butterfly (twiddle=1, last stage)
- Unshift (multiply by shift^(-i))
- Scale (multiply by 1/N)
  |
  v (threadgroup barrier)
[Phase 2: NTT First Stage + Coset Shift]
- Zero-pad to size M
- Multiply by shift^i (coset shift)
- DIT butterfly (twiddle=1, first stage)
  |
  v (threadgroup barrier)
[Phase 3: Leaf Hash]
- Load 8 M31 values (one leaf)
- Apply Poseidon2-M31 permutation
- Store digest (8 M31 output)
  |
  v
Output: extended_evals[0..M-1] + leaf_hashes[0..numLeaves*8-1]
```

### Memory Layout

- **Input**: evaluations of size N = 2^logN
- **Extended**: size M = blowupFactor * N
- **Leaf hashes**: numLeaves = M / NODE_SIZE, each digest = 8 M31 values

### Threadgroup Synchronization

The fused kernel uses `threadgroup_barrier()` to synchronize between phases within a single dispatch:

```metal
// Phase 1: INTT (all threads participate)
if (gid < nHalf) { /* INTT butterfly */ }
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 2: NTT (all threads participate)
if (gid < mHalf) { /* NTT butterfly with coset shift */ }
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 3: Leaf hash (subset of threads)
if (gid < numLeaves) { /* Poseidon2 hash */ }
```

## Kernel Performance

### Estimated Speedup

| Operation | Separate Kernels | Fused Kernel |
|-----------|-----------------|-------------|
| INTT final stage | 1 dispatch | 1 dispatch (same) |
| NTT first stage | 1 dispatch | 1 dispatch (same) |
| Leaf hash | 1 dispatch | 1 dispatch (same) |
| Barriers | 2 | 2 (internal) |
| Sync overhead | 2 | 0 (single wait) |
| **Total** | 3 dispatch + 2 sync | 3 dispatch + 1 sync |

**Expected improvement**: 15-25% reduction in total time due to:
- Eliminated CPU-GPU-CPU round-trips
- Reduced command buffer overhead
- Better GPU utilization (no idle time between phases)

## Implementation Files

### Metal Shader
- `Sources/Shaders/ntt/fused_intt_ntt_leafhash.metal`

Contains:
- `fused_intt_final_unshift_scale` - INTT final stage kernel
- `fused_ntt_first_coset_shift` - NTT first stage with coset shift
- `leaf_hash_poseidon2_m31` - Standalone leaf hash kernel
- `standalone_leaf_hash` - Leaf hash from NTT output
- `fused_fold_leafhash` - Fused FRI fold + leaf hash
- `batch_leaf_hash` - Batch leaf hash for multiple columns

### Swift Engine
- `Sources/zkMetal/NTT/FusedNTTEngine.swift`

Provides:
- `FusedNTTEngine` class with GPU kernel management
- `fusedIntTNTTLeafHash()` - Main fused API
- `computeLeafHashes()` - Standalone leaf hash
- `batchLeafHash()` - Batch leaf hash for columns

## Usage Example

```swift
let engine = try FusedNTTEngine()

// Fused INTT + NTT + LeafHash
let result = try engine.fusedIntTNTTLeafHash(
    evals: evaluations,
    logN: 10,           // 2^10 = 1024 elements
    blowupFactor: 8,    // extend to 8192
    cosetShift: M31.one
)

// Results
let extendedEvals = result.extendedEvals  // 8192 M31 values
let leafHashes = result.leafHashes       // 1024 Poseidon2 digests

print(result.timing.summary)
```

## Configuration Options

### FusedNTTEngine.Config

```swift
// Default: separate dispatches with barriers
Config(enableSingleDispatch: false)

// Experimental: single kernel with threadgroup barriers
Config(enableSingleDispatch: true)
```

### Size Thresholds

```swift
// Minimum logN to use fused kernel (below this, separate is faster)
Config.minFusedLogN = 10  // 2^10 = 1024

// Maximum threads per threadgroup
Config.maxThreadsPerTG = 256
```

## Future Optimizations

1. **Single-dispatch fused kernel**: Combine all three phases into one kernel with threadgroup barriers (experimental)
2. **Batch processing**: Process multiple columns in single dispatch
3. **Register tiling**: Increase threadgroup size to fit more stages in registers
4. **Memory coalescing**: Optimize memory access patterns for leaf hash phase

## Correctness Verification

The implementation includes CPU reference implementations for correctness verification:

```swift
// CPU reference for comparison
let cpuResult = try engine.cpuFusedIntTNTTLeafHash(
    evals: evaluations,
    logN: logN,
    blowupFactor: blowupFactor,
    cosetShift: cosetShift
)

// Compare GPU and CPU results
assert(result.extendedEvals == cpuResult.extendedEvals)
```