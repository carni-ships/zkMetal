# zkMetal Architecture

## Why Metal for Zero-Knowledge Proofs

Metal is Apple's GPU compute framework. Three architectural properties make it uniquely suited for ZK workloads:

### 1. Unified Memory Architecture (UMA)

Apple Silicon shares a single physical memory pool between CPU and GPU. This eliminates the PCIe transfer bottleneck that dominates discrete GPU (CUDA) ZK implementations:

- **No explicit data transfers.** CPU and GPU read/write the same buffers (`.storageModeShared`). A BN254 MSM with 2^18 points (16MB of point data + 8MB of scalars) would require a ~24MB PCIe transfer on CUDA. On Metal, it's zero-copy.
- **Fine-grained CPU/GPU interleaving.** Proof phases that mix CPU logic (Fiat-Shamir transcript, challenge derivation) with GPU compute (MSM, NTT) don't pay transfer penalties between phases. This is critical for multi-round interactive protocols (Plonk: 5 rounds, each mixing CPU transcript with GPU polynomial operations).
- **Buffer caching is cheap.** Grow-only buffer pools persist across calls without managing host/device copies. 29 engines use this pattern.

### 2. 32-bit Native ALU

Metal GPU cores have native 32-bit integer multiply but no 64-bit multiply instruction. This shapes our field arithmetic:

- **8x32-bit limb representation.** All 256-bit fields (BN254, BLS12-381, secp256k1) use 8 limbs of 32 bits. CIOS Montgomery multiplication chains 32-bit multiply-accumulate operations.
- **Small fields are extremely fast.** BabyBear (31-bit) and Mersenne31 use single 32-bit words. One field multiply = one hardware multiply. This gives Metal a structural advantage for modern STARK fields.
- **Goldilocks (64-bit) uses special reduction.** p = 2^64 - 2^32 + 1 allows reduction via shifts and subtracts, avoiding expensive multi-word multiply.

### 3. Threadgroup Shared Memory (32KB)

Metal exposes 32KB of fast shared memory per threadgroup, enabling data reuse patterns critical for ZK:

- **Fused Merkle subtrees.** A single threadgroup hashes a 1024-leaf Poseidon2 subtree entirely in shared memory (10 levels, 512 threads). Eliminates 10 round-trips to device memory. Benchmarked 1.7x faster than level-by-level dispatch.
- **NTT butterfly stages.** Small NTT stages (up to 2^10 elements) execute entirely in shared memory. The four-step FFT decomposes larger transforms into shared-memory row transforms + strided column transforms.
- **Cooperative reductions.** Sumcheck round polynomial computation uses threadgroup shuffle + shared memory reduction. FRI folding uses shared memory for multi-element fold accumulation.

## Dispatch Architecture

### Single Command Buffer Pattern

GPU work is organized into command buffers containing compute encoders. Minimizing command buffer submissions reduces driver overhead (~0.1ms per submit on M3 Pro):

```
Command Buffer
├── Compute Encoder
│   ├── Dispatch 1 (NTT butterfly stage 0)
│   ├── Memory Barrier
│   ├── Dispatch 2 (NTT butterfly stage 1)
│   ├── Memory Barrier
│   └── ...
└── Commit + Wait
```

Engines that need intermediate CPU reads (e.g., Fiat-Shamir challenge between rounds) must split into separate command buffers. The optimization is to batch all GPU-only work into single submits.

### Buffer Caching (Grow-Only)

Every engine maintains cached GPU buffers that grow but never shrink:

```swift
if n > cachedBufElements {
    cachedBuf = device.makeBuffer(length: n * stride, options: .storageModeShared)
    cachedBufElements = n
}
```

This eliminates per-call allocation overhead. On UMA, these buffers are simultaneously CPU- and GPU-accessible with no synchronization cost when accessed sequentially.

## Field Arithmetic on Metal

### CIOS Montgomery Multiplication

All 256-bit field multiplies use Coarsely Integrated Operand Scanning (CIOS), which interleaves multiplication and reduction:

```
For each limb i of multiplier b:
  1. Multiply: accumulate a[j] * b[i] into temp
  2. Reduce: compute Montgomery quotient m, accumulate m * p[j]
  3. Shift: propagate carries to next limb position
```

This keeps the intermediate accumulator small (avoids 512-bit intermediates) and maximizes register reuse. On Metal's 32-bit ALU, each limb operation is a native multiply-add.

### Specialized Squaring (SOS)

`fp_sqr` uses Sum-of-Squares optimization: exploiting a[i]*a[j] = a[j]*a[i] to halve the number of multiplications. This saves ~25% of multiply operations in point doubling (which dominates MSM bucket reduction).

## MSM Architecture

Multi-scalar multiplication uses Pippenger's bucket method with Metal-specific optimizations:

```
1. Signed-digit scalar recoding (CPU)
   └── Reduces bucket count by 2x, eliminates large-index buckets
2. Counting sort by window (CPU or GPU)
   └── GPU sort for N ≥ 262144 (3 kernels: histogram + prefix sum + scatter)
3. Bucket accumulation (GPU)
   └── One thread per bucket, Z=1 mixed affine addition
4. Bucket summation (GPU)
   └── Running sum: B[k] + B[k-1] + ... with weight multiplication
5. Window combination (CPU)
   └── Double-and-add across windows
```

GLV endomorphism is available for curves with efficient endomorphism (BN254) but disabled for curves where 2x point count exceeds the scalar width savings (BLS12-377, secp256k1).

## NTT Architecture

### Four-Step FFT for Large Transforms

For N > 2^16, the transform is decomposed:

```
N = N1 × N2 (e.g., 2^22 = 2^11 × 2^11)

1. N1 row NTTs of size N2 (fused bitrev + butterfly in shared memory)
2. Twiddle factor multiply (fused into row output)
3. N2 column NTTs of size N1 (strided access, cache-unfriendly for 32B elements)
```

For small fields (BabyBear 4B, Goldilocks 8B), cache-line utilization is 4-16x better than BN254 (32B), explaining the performance gap at large sizes.

### Fused Kernels

Multiple NTT optimizations are fused into single kernel dispatches:
- **Bitrev + butterfly:** First butterfly stage includes bit-reversal permutation (saves one pass)
- **Twiddle fusion:** Twiddle factor multiply merged into butterfly output (saves one pass, 4-5x improvement at 2^24)
- **NTT + constraint evaluation:** For STARK provers, NTT output feeds directly into constraint evaluation without memory round-trip

## Hash Architecture

### Algebraic Hashes (Poseidon2)

Poseidon2 with t=3 state over BN254 Fr. 64 rounds (8 full + 56 partial). Each round: S-box (x^5), external/internal linear layer, round constant addition. One GPU thread per hash — no cross-thread communication needed.

Merkle trees use fused subtree kernels: a threadgroup of 512 threads hashes a 1024-leaf subtree entirely in shared memory, writing only the root back to device memory. Upper levels use pair-hashing dispatches with memory barriers.

### Standard Hashes (Keccak, Blake3, SHA-256)

32-bit ARX operations map directly to Metal's ALU. Keccak's 25×64-bit state causes register spilling (200 bytes per thread), limiting occupancy. Blake3 and SHA-256 have smaller state and better occupancy.

## Proof System Composition

Higher-level proof systems compose the GPU primitives:

```
Groth16 prove:
  computeH    → NTT (GPU) + polynomial division (GPU)
  proofA      → MSM (GPU)
  proofBG2    → G2 Straus MSM (CPU, parallel with GPU)
  proofBG1    → MSM (GPU)
  proofC      → MSM (GPU)

Plonk prove:
  Round 1     → 3 iNTT + 3 MSM (GPU)
  Round 2     → batch inversion (CPU) + NTT (GPU)
  Round 3     → 12+ polyMulNTT (GPU) — quotient polynomial
  Round 4     → polynomial evaluation (CPU)
  Round 5     → 2 MSM (GPU)

Circle STARK prove:
  Trace LDE   → Circle NTT (GPU)
  Constraints → fused NTT+constraint (GPU)
  Commit      → Poseidon2 Merkle (GPU)
  FRI         → fold (GPU) + Merkle commit (GPU) per round
```

## Cross-Primitive Optimization Patterns

Techniques discovered in one primitive and transferred to others:

| Pattern | Origin | Applied To | Effect |
|---------|--------|-----------|--------|
| Buffer caching (grow-only) | Poseidon2 | 29 engines | Eliminates per-call alloc |
| Batch inversion (Montgomery) | PolyEngine | Plonk R2, LogUp, Lasso | 10x for inversion-heavy phases |
| Fused kernels | NTT bitrev+butterfly | Sumcheck, Merkle, FRI, STARK | 20-47% per fusion |
| Sorted array vs Dictionary | GKR sparse wiring | SparseSumcheck, LogUp | 2-5x for sparse iterations |
| Single command buffer | Merkle | MSM GLV, NTT stages | ~0.1ms per saved dispatch |
| Signed-digit recoding | MSM | All curve MSM engines | 2x bucket reduction |
