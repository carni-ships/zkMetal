# zkMetal

GPU-accelerated zero-knowledge cryptography library for Apple Silicon, written in Metal and Swift. 50+ primitives spanning field arithmetic, MSM, NTT, hash functions, polynomial commitments, proof protocols, post-quantum crypto, and homomorphic encryption across 15 fields and 7 elliptic curves.

## Contents

- [Primitives](#primitives)
- [Performance](#performance)
  - [MSM (BN254 G1)](#msm-bn254-g1)
  - [NTT](#ntt)
  - [Hashing](#hashing)
  - [Merkle Trees](#merkle-trees)
  - [FRI Folding](#fri-folding-bn254-fr)
  - [Sumcheck](#sumcheck-bn254-fr)
  - [Sparse Sumcheck](#sparse-sumcheck-bn254-fr)
  - [Univariate Sumcheck](#univariate-sumcheck-bn254-fr)
  - [Polynomial Ops](#polynomial-ops-bn254-fr)
  - [KZG Commitments](#kzg-commitments-bn254-g1)
  - [Batch KZG](#batch-kzg-bn254-g1)
  - [Blake3 Hashing](#blake3-hashing)
  - [IPA](#ipa-bulletproofs-style-inner-product-argument)
  - [Verkle Trees (CPU)](#verkle-trees-cpu-pedersen--ipa)
  - [ECDSA (CPU)](#ecdsa-cpu-secp256k1-batch-verification)
  - [Circle STARK](#circle-stark-mersenne31)
  - [Plonk](#plonk-bn254-kzg)
  - [Groth16](#groth16-bn254)
  - [GKR](#gkr-bn254-fr-layered-circuits)
  - [Data-Parallel GKR](#data-parallel-gkr)
  - [Circle NTT](#circle-ntt-mersenne31-gpu)
  - [Radix Sort](#gpu-radix-sort)
  - [Basefold PCS](#basefold-pcs-bn254-fr)
  - [Brakedown PCS](#brakedown-pcs)
  - [Zeromorph PCS](#zeromorph-pcs)
  - [HyperNova Folding](#hypernova-folding)
  - [BLS12-381](#bls12-381-field--curve--pairing)
  - [Pasta Curves](#pasta-curves-pallasvesta)
  - [Binius Binary Tower](#binius-binary-tower)
  - [Lasso Structured Lookups](#lasso-structured-lookups)
  - [LogUp Lookup Argument](#logup-lookup-argument)
  - [cq Cached Quotients](#cq-cached-quotients-lookup)
  - [Poseidon2 M31](#poseidon2-m31-mersenne31-t16)
  - [BLS12-377 NTT and MSM](#bls12-377-ntt-and-msm)
  - [secp256k1 MSM](#secp256k1-msm)
  - [Pasta MSM](#pasta-msm-pallasvesta)
  - [Tensor Proof Compression](#tensor-proof-compression)
  - [WHIR](#whir)
  - [Transcript (Fiat-Shamir)](#transcript-fiat-shamir)
  - [Proof Serialization](#proof-serialization)
  - [GPU Witness Generation](#gpu-witness-generation)
  - [Constraint IR Evaluation](#constraint-ir-evaluation)
  - [Reed-Solomon Erasure Coding](#reed-solomon-erasure-coding)
  - [Lattice Crypto (Kyber/Dilithium)](#lattice-crypto-kyberdilithium)
  - [HE NTT](#he-ntt-rns-based)
  - [NEON CPU Benchmarks](#neon-cpu-benchmarks)
  - [CPU Benchmarks (C Pippenger, C NTT)](#cpu-benchmarks-c-pippenger-c-ntt)
  - [Streaming Verification](#streaming-verification)
  - [Incremental Merkle](#incremental-merkle)
  - [Batch Field Operations](#batch-field-operations)
  - [IPA Accumulation (Pallas)](#ipa-accumulation-pallas)
  - [Theoretical Performance Analysis](#theoretical-performance-analysis)
- [Supported Fields](#supported-fields)
- [Architecture](#architecture)
- [Usage](#usage)
  - [As a library](#as-a-library)
  - [Benchmarks](#benchmarks)
  - [MSM CLI](#msm-cli)
- [Auto-Tuning](#auto-tuning)
- [Building](#building)
- [Design Decisions](#design-decisions)
- [Correctness & Testing](#correctness--testing)
- [Optimization](#optimization)

## Primitives

| Primitive | Description |
|-----------|-------------|
| **MSM** | Multi-scalar multiplication (Pippenger + signed-digit + GLV endomorphism) -- BN254, BLS12-377, secp256k1, Pallas, Vesta |
| **NTT** | Number theoretic transform (four-step FFT with fused bitrev+butterfly, twiddle fusion) -- BN254, BLS12-377, Goldilocks, BabyBear |
| **Poseidon2** | Algebraic hash function (t=3, BN254 Fr; t=16, Mersenne31) |
| **Keccak-256** | SHA-3 hash (fused subtree Merkle) |
| **Blake3** | BLAKE3 hash (batch hashing + Merkle trees) |
| **Merkle Trees** | Poseidon2, Keccak-256, and Blake3 backends |
| **FRI** | Fast Reed-Solomon IOP (fold + commit + query + verify; fold-by-2, fold-by-4, fold-by-8) |
| **Circle FRI** | FRI protocol adapted for circle domain over M31 |
| **Sumcheck** | Interactive sumcheck protocol (fused round+reduce with SIMD shuffles) |
| **Sparse Sumcheck** | O(nnz) sumcheck for sparse multilinear polynomials |
| **Univariate Sumcheck** | Sumcheck for univariate polynomials |
| **KZG** | Polynomial commitment scheme (commit + open via MSM) |
| **Batch KZG** | Batch polynomial openings via random linear combination |
| **IPA** | Inner product argument (Bulletproofs-style, GPU batch fold) |
| **Verkle Trees (CPU)** | Width-N tree with Pedersen commitments + IPA opening proofs |
| **LogUp** | Lookup argument via logarithmic derivatives + sumcheck |
| **Lasso** | Structured lookup via tensor decomposition of large tables |
| **cq** | Cached quotients lookup (prover time independent of table size) |
| **Range Proofs** | Proves values in [0, R) via LogUp with limb decomposition |
| **ECDSA (CPU)** | secp256k1 batch verification (probabilistic + individual) |
| **Radix Sort** | GPU 32-bit LSD radix sort (4-pass, 8-bit radix) |
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees |
| **Circle NTT** | Circle-group NTT over Mersenne31 (order 2^31, full 2-adicity) |
| **Circle STARK** | Full STARK prover/verifier over M31 circle domain |
| **Plonk** | Preprocessed polynomial IOP prover with KZG commitments |
| **Groth16** | zk-SNARK with BN254 pairings (R1CS, trusted setup, prove, verify) |
| **GKR** | Goldwasser-Kalai-Rothblum interactive proof for layered circuits |
| **Data-Parallel GKR** | Batched GKR for N circuit instances (experimental) |
| **Basefold** | NTT-free multilinear polynomial commitment via sumcheck folding |
| **Brakedown** | Linear-code polynomial commitment (NTT-free, expander-based) |
| **Zeromorph** | Multilinear-to-univariate polynomial commitment reduction |
| **HyperNova** | CCS folding scheme for incremental verifiable computation |
| **BLS12-381** | Full tower arithmetic (Fp/Fp2/Fp6/Fp12), G1/G2, Miller loop, pairings |
| **Pasta Curves** | Pallas/Vesta cycle (recursive proof composition ready) |
| **Binius** | Binary tower field arithmetic (GF(2^8)->GF(2^128)), additive FFT |
| **Tensor Proof** | Tensor-based multilinear proof compression (sqrt(N) proof size) |
| **WHIR** | Weighted Hashing of Indices for Reed-Solomon proximity testing |
| **Transcript** | Fiat-Shamir duplex sponge (Poseidon2 + Keccak backends) |
| **Serialization** | Proof serialization/deserialization (ProofWriter/ProofReader) |
| **Witness Gen** | GPU witness trace evaluation (instruction-stream architecture) |
| **Constraint IR** | Runtime constraint compilation (IR -> Metal source -> GPU pipeline) |
| **IPA Accumulation** | Halo-style accumulation (Pallas curve, batch decide) |
| **Kyber/Dilithium** | Post-quantum lattice crypto (GPU-accelerated NTT) |
| **HE NTT** | RNS-based NTT for homomorphic encryption (CKKS/BFV) |
| **Reed-Solomon** | GPU erasure coding for data availability sampling |
| **Streaming Verify** | Task-queue streaming proof verification |
| **Incremental Merkle** | Append/update Merkle tree without full rebuild |
| **Batch Field Ops** | C-optimized vectorized field arithmetic (add/sub/neg/mul) |
| **STIR** | *Planned/unimplemented* |
| **Marlin** | *Planned/unimplemented* |
| **Spartan** | *Planned/unimplemented* |
| **Jolt** | *Planned/unimplemented* |

## Performance

All benchmarks measured on Apple M3 Pro (6P+6E cores), comparing GPU (Metal), optimized C CPU (CIOS Montgomery with `__uint128_t`, multi-threaded Pippenger, NEON SIMD), and single-threaded CPU (vanilla Swift).
Run `swift run -c release zkbench all` to reproduce, or `swift run -c release zkbench cpu` for the 3-way comparison. For small inputs (MSM n<=2048), the engine automatically routes to C Pippenger instead of GPU to avoid dispatch overhead.

### MSM (BN254 G1)

| Points | Vanilla CPU | Swift Pippenger | C Pippenger | GPU (Metal) |
|--------|-------------|----------------|-------------|-------------|
| 2^8 | 450ms | 16ms | **1.3ms** | 1.1ms |
| 2^10 | 1.8s | 45ms | **2.9ms** | 3.0ms |
| 2^12 | 7.3s | 129ms | **8.1ms** | 14ms |
| 2^14 | 35s | 429ms | 29ms | **22ms** |
| 2^16 | -- | -- | 68ms | **27ms** |
| 2^18 | -- | -- | 240ms | **45ms** |
| 2^20 | -- | -- | 856ms | **119ms** |

C Pippenger uses multi-threaded bucket accumulation with `__uint128_t` CIOS Montgomery (8 pthreads). At n<=2048, C Pippenger is automatically used instead of GPU to avoid dispatch overhead. GPU wins at n>=2^14.

**Other curve MSM:**

| Points | BN254 GPU | BLS12-377 GPU | secp256k1 GPU | secp256k1 C Pip |
|--------|-----------|---------------|---------------|-----------------|
| 2^8 | 1.1ms | 9ms | 1.3ms | 1.4ms |
| 2^14 | 22ms | 36ms | 22ms | 31ms |
| 2^16 | 27ms | 176ms | 38ms | 92ms |
| 2^18 | 45ms | 205ms | 78ms | 339ms |

**Comparison to other implementations (BN254 MSM):**

| Points | zkMetal (M3 Pro)&#185; | ICICLE-Metal (M3 Pro)&#185; | ICICLE CPU (M3 Pro)&#185; | ICICLE-Metal (M3 Air)&#178; | MoPro v2 (M3 Air)&#178; | Arkworks CPU (M3 Air)&#178; | ICICLE CUDA&#179; |
|--------|---------|-------------|-----------|-------------|-----------|-----------|-----------|
| 2^16 | **27ms** | 1,083ms | 114ms | -- | 253ms | 69ms | ~9ms |
| 2^18 | **45ms** | 1,475ms | 556ms | 149ms | 678ms | 266ms | -- |
| 2^20 | **119ms** | 2,590ms | 2,349ms | 421ms | 1,702ms | 592ms | -- |

&#185; Measured locally. ICICLE-Metal v3.8.0 has ~600ms constant overhead per call (license server).
&#178; Reported by [MoPro blog](https://zkmopro.org/blog/metal-msm-v2/) -- different hardware and methodology, not directly comparable.
&#179; [Ingonyama](https://github.com/ingonyama-zk/icicle) on RTX 3090 Ti (native 64-bit integer multiply).

Metal GPU MSM is competitive with other Metal implementations and faster than ICICLE-Metal, but still slower than optimized multithreaded CPU (Arkworks). The fundamental bottleneck is that 256-bit field arithmetic requires 8x32-bit limbs on Metal (no native 64-bit integer multiply), while CPU implementations use 4x64-bit limbs with hand-tuned assembly, out-of-order execution, and deep pipelines. CUDA GPUs (like those targeted by [Ingonyama's ICICLE](https://github.com/ingonyama-zk/icicle)) have native 64-bit integer multiply. The GPU advantage is clear for smaller fields: BabyBear NTT achieves **8.5B elements/sec** (2ms at 2^24) and Goldilocks **5.7B elements/sec** (3ms at 2^24) -- both dramatically faster than BN254 on the same GPU (see NTT table below).

GPU scaling is strongly sublinear: 1024x more points (2^8 to 2^18) costs only ~40x more time, as fixed GPU overhead dominates at small sizes.

### NTT

**BN254 Fr (256-bit, 8x32-bit limbs):**

| Size | Vanilla CPU | Opt C | Opt C vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|------|-------------|-------|------------------|-------------|----------------|
| 2^14 | 79ms | 2.6ms | **30x** | 0.45ms | **176x** |
| 2^16 | 369ms | 12ms | **30x** | 0.76ms | **483x** |
| 2^18 | 1.6s | 55ms | **30x** | 2.2ms | **749x** |
| 2^20 | 7.3s | 246ms | **30x** | 6.1ms | **1198x** |

Optimized C uses fully unrolled 4-limb CIOS Montgomery multiplication with `__uint128_t` (compiled with `-O3`). Also available: parallel CPU (GCD, 12 cores) at 5x over vanilla.

**Multi-field NTT comparison (GPU):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.47ms | 1.4ms | 0.14ms | 0.18ms |
| 2^18 | 1.6ms | 2.1ms | 0.19ms | 0.26ms |
| 2^20 | 6.1ms | 5.8ms | 0.81ms | 0.95ms |
| 2^22 | 26ms | 25ms | 4.2ms | 2.8ms |
| 2^24 | 116ms | 110ms | 3.0ms | 2.0ms |

Smaller fields see dramatic throughput gains: BabyBear NTT at 2^24 (16M elements) runs in **2ms** -- one element per 0.12ns, or **8.5B elements/sec**. The GPU advantage for small fields comes from native 32-bit arithmetic (1 mul per element vs 64 muls for BN254 CIOS), 8x higher memory density, and better threadgroup utilization.

**CPU optimization results by field:**
- **BN254 Fr:** C with unrolled CIOS Montgomery gives **29-30x** over vanilla Swift (Swift's optimizer is very poor for 256-bit multi-limb carry chains).
- **BabyBear:** NEON SIMD (4-wide Montgomery via `vqdmulhq_s32`, Plonky3 technique) gives **5.6x** over vanilla.
- **Goldilocks:** Optimized C with `__uint128_t` gives **2.1-2.2x** over vanilla.
- GCD parallel dispatch **regresses** for BabyBear/Goldilocks (0.4-0.5x) -- field ops are too cheap for thread overhead.

**Comparison to ICICLE-Metal v3.8 NTT (measured locally, M3 Pro):**

| Size | zkMetal BN254&#185; | ICICLE BN254&#185; | zkMetal BabyBear&#185; | ICICLE BabyBear&#185; |
|------|------------|-------------|----------------|----------------|
| 2^16 | **0.76ms** | 89ms | **0.18ms** | 86ms |
| 2^18 | **1.6ms** | 108ms | **0.26ms** | 92ms |
| 2^20 | **6.1ms** | 194ms | **0.95ms** | 108ms |
| 2^22 | **26ms** | 915ms | **2.8ms** | 181ms |
| 2^24 | **116ms** | 3,892ms | **2.0ms** | 709ms |

&#185; Measured locally on M3 Pro. ICICLE-Metal has ~85ms per-call overhead.

zkMetal is **30-90x faster** on BN254 and **90-500x faster** on BabyBear. ICICLE does not ship Goldilocks in their Metal backend.

GPU scales sublinearly: 2^10 to 2^22 is 4096x more data for ~100x more time. CPU scales linearly with n log n. Speedup grows with input size.

### Hashing

| Primitive | Batch Size | Vanilla CPU | Parallel CPU (12 cores) | Parallel vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|-----------|-----------|-------------|------------------------|--------------------|--------------------|----------------|
| Poseidon2 | 2^12 | 523ms | 71ms | **7x** | 2.3ms | **227x** |
| Poseidon2 | 2^14 | 2.0s | 278ms | **7x** | 2.3ms | **871x** |
| Poseidon2 | 2^16 | 8.0s | 1.1s | **7x** | 8.1ms | **993x** |
| Keccak-256 | 2^14 | 100ms | 23ms | **4x** | 0.20ms | **500x** |
| Keccak-256 | 2^16 | 387ms | 89ms | **4x** | 0.45ms | **860x** |
| Keccak-256 | 2^18 | 1.6s | 360ms | **4x** | 1.4ms | **1143x** |

Parallel CPU achieves 4-7x over vanilla (embarrassingly parallel -- each hash independent). GPU achieves 227-1143x over vanilla. No other Metal implementations of Poseidon2 or Keccak-256 batch hashing are known.

### Merkle Trees

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 7.4ms | 272ms | **37x** |
| Poseidon2 | 2^12 | 8.8ms | 2.0s | **227x** |
| Poseidon2 | 2^14 | 10ms | 4.7s | **470x** |
| Poseidon2 | 2^16 | 22ms | 20s | **909x** |
| Poseidon2 | 2^18 | 46ms | 66s | **1435x** |
| Poseidon2 | 2^20 | 130ms | -- | -- |
| Keccak-256 | 2^12 | 0.37ms | 44ms | **119x** |
| Keccak-256 | 2^14 | 0.51ms | 155ms | **304x** |
| Keccak-256 | 2^16 | 1.4ms | 783ms | **559x** |
| Keccak-256 | 2^18 | 4.5ms | 3.0s | **667x** |
| Keccak-256 | 2^20 | 13ms | -- | -- |
| Blake3 | 2^12 | 0.72ms | 4ms | **6x** |
| Blake3 | 2^14 | 0.92ms | 16ms | **17x** |
| Blake3 | 2^16 | 1.3ms | 101ms | **78x** |
| Blake3 | 2^18 | 3.9ms | 345ms | **88x** |
| Blake3 | 2^20 | 12ms | -- | -- |

All three backends scale linearly (O(n) tree construction). GPU speedup grows with size as fixed dispatch overhead is amortized. Keccak and Blake3 are the fastest Merkle backends at large sizes (13ms and 12ms at 2^20) due to simpler arithmetic vs Poseidon2 (130ms).

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.22ms | 8.9ms | **41x** |
| 2^16 | 0.35ms | 35ms | **99x** |
| 2^18 | 0.92ms | 137ms | **149x** |
| 2^20 | 1.96ms | 542ms | **276x** |
| 2^22 | 7.52ms | 2.2s | **295x** |

Full fold-to-constant: 2^20 in 3.0ms (20 rounds, fused 4-round kernels).

**FRI fold-by-4 and fold-by-8 results:**

| Size | Fold-by-2 | Fold-by-4 | Fold-by-8 | 4/2 speedup | 8/2 speedup |
|------|-----------|-----------|-----------|-------------|-------------|
| 2^14 | 0.22ms | 0.34ms | 0.25ms | -- | -- |
| 2^16 | 0.35ms | 0.59ms | 0.31ms | -- | -- |
| 2^18 | 0.92ms | 1.64ms | 1.06ms | -- | -- |
| 2^20 | 1.96ms | 2.84ms | 3.37ms | -- | -- |
| 2^22 | 7.52ms | 10.5ms | 9.61ms | -- | -- |

**FRI commit phase (fold + Merkle tree, full protocol):**

| Size | Fold-by-2 | Fold-by-4 | Fold-by-8 | 4/2 speedup | 8/2 speedup |
|------|-----------|-----------|-----------|-------------|-------------|
| 2^15 | 66ms | 36ms | 19ms | **1.9x** | **3.4x** |
| 2^16 | 78ms | 36ms | 31ms | **2.2x** | **2.5x** |
| 2^18 | 135ms | 57ms | 32ms | **2.4x** | **4.2x** |
| 2^20 | 323ms | 126ms | 99ms | **2.6x** | **3.3x** |

Fold-by-4 and fold-by-8 reduce Merkle tree layers (8 and 6 vs 15-17 for fold-by-2), cutting the Merkle-dominated commit phase by 2-4x. Individual fold kernels are slightly slower per-element, but the protocol-level speedup from fewer layers is substantial.

GPU scales sublinearly: 2^14 to 2^22 is 256x more data for ~34x more time, as each folding round halves the domain. CPU scales linearly. Speedup grows from 41x to 295x. No other Metal FRI implementations are known.

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 0.89ms | 21ms | **24x** |
| 2^16 | 0.85ms | 83ms | **98x** |
| 2^18 | 2.2ms | 337ms | **153x** |
| 2^20 | 7.3ms | 1.3s | **178x** |
| 2^22 | 16ms | 5.2s | **325x** |

GPU scales sublinearly: 2^14 to 2^22 is 256x more variables for ~18x more time. CPU scales linearly. Each sumcheck round reduces the problem by half, and fused round+reduce kernels keep GPU utilization high. No other Metal sumcheck implementations are known; ICICLE (CUDA) offers GPU sumcheck but no published comparison numbers.

### Sparse Sumcheck (BN254 Fr)

| Variables | Density | Sparse | Dense | Speedup |
|-----------|---------|--------|-------|---------|
| 2^14 | 1% (nnz=162) | 1.9ms | 17ms | **9.0x** |
| 2^16 | 1% (nnz=654) | 7.5ms | 63ms | **8.4x** |
| 2^18 | 1% (nnz=2611) | 27ms | 250ms | **9.1x** |
| 2^16 | 10% (nnz=6243) | 37ms | 70ms | **1.9x** |

O(nnz) sparse multilinear sumcheck. At 1% density, sparse sumcheck is 8-9x faster than dense. At 10% density, still nearly 2x faster. Correctness verified: round polynomials and final evaluations match the dense implementation.

### Univariate Sumcheck (BN254 Fr)

Sumcheck protocol adapted for univariate polynomials. Currently crashes on some hardware configurations (Metal shader compilation issue). Benchmark command: `swift run -c release zkbench usc`.

### Polynomial Ops (BN254 Fr)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Multiply (NTT) | deg 2^10 | 57ms | 1.7ms | **34x** |
| Multiply (NTT) | deg 2^12 | 218ms | 2.0ms | **109x** |
| Multiply (NTT) | deg 2^14 | 1.1s | 3.3ms | **328x** |
| Multiply (NTT) | deg 2^16 | 2.4s | 7.7ms | **319x** |
| Multi-eval (Horner) | deg 2^10, 1024 pts | -- | 1.7ms | -- |
| Multi-eval (Horner) | deg 2^12, 4096 pts | -- | 8.2ms | -- |
| Multi-eval (Horner) | deg 2^14, 16384 pts | -- | 114ms | -- |

Polynomial multiplication uses NTT under the hood (CPU baseline = 2 forward NTTs + pointwise mul + inverse NTT). Multi-point evaluation uses GPU Horner's method (one thread per evaluation point). Subproduct-tree evaluation is available but currently slower than Horner for these sizes due to high constant factors.

### KZG Commitments (BN254 G1)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Commit | deg 2^8 | 293ms | 0.4ms | **652x** |
| Commit | deg 2^10 | 2.2s | 4.6ms | **490x** |
| Open (eval + witness) | deg 2^8 | 859ms | 3.9ms | **223x** |
| Open (eval + witness) | deg 2^10 | 2.1s | 4.6ms | **459x** |

KZG performance is MSM-dominated. Commit = MSM(SRS, coefficients). Open = Horner eval + C synthetic division + MSM for witness. CPU baseline uses sequential double-and-add scalar multiplication. SRS generation and quotient polynomial use C CIOS for fast field arithmetic.

### Batch KZG (BN254 G1)

| N Polys | Deg | N Individual Opens | 1 Batch Open | Speedup |
|---------|-----|-------------------|--------------|---------|
| 4 | 256 | 14.4ms | 9.8ms | **1.5x** |
| 8 | 256 | 24.5ms | 13.0ms | **1.9x** |
| 16 | 256 | 47.3ms | 21.5ms | **2.2x** |
| 32 | 256 | 75.2ms | 34.5ms | **2.2x** |

**Multi-point batch open:**

| N Polys | Deg | N Individual | 1 Multi-Point Batch | Speedup |
|---------|-----|-------------|---------------------|---------|
| 4 | 256 | 7.4ms | 5.8ms | **1.3x** |
| 8 | 256 | 15.4ms | 10.0ms | **1.5x** |
| 16 | 256 | 29.7ms | 18.2ms | **1.6x** |

Batch KZG amortizes witness polynomial computation across N openings via random linear combination. Speedup grows with batch size (up to 2.2x at N=32). All proofs verified via algebraic batch verification.

### Blake3 Hashing

| Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----|-------|---------|
| 2^14 | 0.012 us/hash | 0.6 us/hash | **50x** |
| 2^16 | 0.006 us/hash | 0.6 us/hash | **100x** |
| 2^18 | 0.007 us/hash | 0.6 us/hash | **86x** |
| 2^20 | 0.003 us/hash | 0.6 us/hash | **200x** |

Blake3 is much simpler than Keccak (7 rounds of 32-bit ARX ops vs 24 rounds of 64-bit Keccak-f). GPU speedup scales with batch size as fixed dispatch overhead amortizes. CPU Blake3 is very fast (0.6us) so GPU only wins at large batch sizes.

### IPA (Bulletproofs-style Inner Product Argument)

| Size | Prove | Verify |
|------|-------|--------|
| n=4 | 1.6ms | 1.7ms |
| n=16 | 3.1ms | 2.0ms |
| n=64 | 6.1ms | 2.9ms |
| n=256 | 12.8ms | 3.8ms |

Log(n) halving rounds with GPU batch generator folding (Metal kernel `batch_fold_generators`) and C CIOS scalar multiplication. Fiat-Shamir challenges via Blake3.

### Verkle Trees (CPU, Pedersen + IPA)

| Operation | Time |
|-----------|------|
| Build (width=16, 256 leaves) | 12ms |
| Path proof (2 openings) | 23ms |
| Verify path | 5ms |

Verkle tree performance is IPA-dominated. Previous version: 931ms path proof -- C CIOS gives **40x** improvement.

### ECDSA (CPU, secp256k1 Batch Verification)

| Operation | Time |
|-----------|------|
| Single verify | 0.32ms |
| Batch probabilistic 64 sigs | 8.0ms (0.13ms/sig) |

secp256k1 ECDSA using C CIOS Montgomery field arithmetic. Previous version (Swift scalar mul): 3.96ms/sig single verify -- C CIOS gives **12x** improvement.

### Circle STARK (Mersenne31)

| Trace Size | Prove | Verify | Proof Size |
|-----------|-------|--------|------------|
| 2^8 | 109ms | 15ms | 39 KB |
| 2^10 | 24ms | 14ms | 53 KB |
| 2^12 | 22ms | 16ms | 69 KB |
| 2^14 | 56ms | 28ms | 87 KB |

GPU-accelerated Circle STARK prover over Mersenne31 with Fibonacci AIR. Full GPU pipeline: Circle NTT for LDE, GPU constraint evaluation, GPU Keccak Merkle trees (hash_m31 + level-by-level tree), CPU FRI fold with GPU Merkle per round. Profile at 2^14: LDE 10ms, commit 12ms, constraint eval 2ms, FRI 27ms.

### Plonk (BN254, KZG)

| Gates | Setup | Prove | Verify |
|-------|-------|-------|--------|
| 16 | 18ms | 26ms | 3ms |
| 64 | 32ms | 44ms | 3ms |
| 256 | 17ms | 56ms | 3ms |
| 1024 | 48ms | 157ms | 3ms |

Preprocessed Plonk prover with KZG polynomial commitments over BN254. NTT-based polynomial multiplication for quotient computation. Previous version (naive O(n^2) poly mul): 7365ms at n=1024 -- GPU NTT gives **43x** improvement.

### Groth16 (BN254)

| Constraints | Setup | Prove | Verify |
|-------------|-------|-------|--------|
| 8 | 121ms | 68ms | 73ms |
| 64 | 625ms | 317ms | 76ms |
| 256 | 2.4s | 1.5s | 104ms |

zk-SNARK with BN254 pairings. R1CS constraint system, trusted setup (powers of tau), MSM-based proving, pairing-based verification. BN254 pairing: 35ms (bilinearity verified).

### GKR (BN254 Fr, Layered Circuits)

| Circuit | Prove | Verify |
|---------|-------|--------|
| 2^4 width, d=4 | 1.8ms | 2.7ms |
| 2^5 width, d=4 | 4.2ms | 3.6ms |
| 2^6 width, d=4 | 9.5ms | 4.7ms |
| 2^6 width, d=8 | 19ms | 9.3ms |
| 2^8 width, d=4 | 46ms | 8.8ms |
| 2^8 width, d=8 | 93ms | 17ms |
| 2^10 width, d=4 | 241ms | 23ms |

Goldwasser-Kalai-Rothblum interactive proof for layered arithmetic circuits via batched sumcheck. Sparse wiring predicate evaluation reduces prover work from O(2^(3*logW)) to O(numGates) per sumcheck round. Previous dense implementation: 3728ms at 2^6 d=8 -- sparse sumcheck gives **190x** improvement.

### Data-Parallel GKR

Batched GKR for proving N instances of the same circuit simultaneously. Currently experimental -- correctness tests for multi-instance squaring and repeated operations are failing. The tamper-detection path works correctly. Benchmark command: `swift run -c release zkbench dp`.

### Circle NTT (Mersenne31, GPU)

| Size | GPU Time | Throughput |
|------|----------|------------|
| 2^14 | 0.24ms | 68M elem/s |
| 2^16 | 0.44ms | 149M elem/s |
| 2^18 | 1.2ms | 222M elem/s |
| 2^20 | 4.0ms | 262M elem/s |

Circle-group NTT exploits the unique structure of the circle domain (x^2+y^2=1 over M31). First fold uses y-coordinates, subsequent folds use x-coordinate squaring map. All operations are single-word 32-bit arithmetic.

### GPU Radix Sort

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^16 | 0.6ms | 3.1ms | **5x** |
| 2^18 | 1.4ms | 14ms | **10x** |
| 2^20 | 4.1ms | 62ms | **15x** |
| 2^22 | 7.9ms | 278ms | **35x** |

4-pass LSD radix sort with 8-bit radix. Used internally by MSM for bucket assignment. Speedup grows with size as GPU parallelism amortizes dispatch overhead.

### Basefold PCS (BN254 Fr)

| Operation | Size | Time |
|-----------|------|------|
| Single fold | 2^10 | 0.39ms |
| Single fold | 2^14 | 0.60ms |
| Single fold | 2^18 | 7.2ms |
| Single fold | 2^22 | 48ms |
| Multi-fold to constant | 2^14 | 8.0ms |
| Multi-fold to constant | 2^18 | 0.93ms |
| Multi-fold to constant | 2^22 | 6.5ms |

**Full protocol (commit + open + verify):**

| Size | Commit | Open | Verify | Total |
|------|--------|------|--------|-------|
| 2^10 | 7.4ms | 34ms | 0.38ms | 41ms |
| 2^14 | 10ms | 67ms | 0.52ms | 78ms |
| 2^18 | 45ms | 144ms | 0.65ms | 190ms |

NTT-free multilinear polynomial commitment via recursive sumcheck-based folding. Verification is very fast (<1ms) since the verifier only checks sumcheck proofs. Open time is dominated by iterative fold+commit rounds.

### Brakedown PCS

Linear-code polynomial commitment (NTT-free, expander-based). Currently crashes on some hardware configurations (signal 139). Benchmark command: `swift run -c release zkbench brakedown`.

### Zeromorph PCS

Multilinear-to-univariate polynomial commitment reduction. Currently crashes on some hardware configurations (signal 139). Benchmark command: `swift run -c release zkbench zeromorph`.

### HyperNova Folding

| Steps | Total | Per-fold | Commit | Fold | Init |
|-------|-------|----------|--------|------|------|
| 10 | 55ms | 3.7ms | 19ms | 33ms | 2.5ms |
| 100 | 590ms | 3.6ms | 227ms | 360ms | 2.6ms |
| 1000 | 7.1s | 3.7ms | 3.3s | 3.7s | 2.5ms |

CCS folding scheme for incremental verifiable computation (IVC). Folds N computation steps into a single instance: the final instance size is O(1) regardless of N. A single SNARK proof on the final instance proves all N steps. Per-fold cost is ~3.7ms, dominated by commitment (MSM) and cross-term folding.

### BLS12-381 (Field + Curve + Pairing)

| Operation | Time |
|-----------|------|
| Fp mul | 339 ns |
| Fp2 mul | 1.8 us |
| Fp12 mul | 54 us |
| G1 add | 7.2 us |
| G1 scalar mul (128-bit) | 1.9ms |
| G2 add | 29 us |
| Fp inverse | 155 us |
| Fr mul | 220 ns |

Full tower arithmetic: Fp (381-bit, 12x32-bit limbs), Fp2/Fp6/Fp12 extensions, G1/G2 curve operations, Miller loop, and optimal ate pairing. All field and curve correctness tests pass (arithmetic properties, on-curve checks, bilinearity).

### Pasta Curves (Pallas/Vesta)

| Test | Result |
|------|--------|
| Pallas field arithmetic (add, mul, inv, distributivity) | PASS |
| Vesta field arithmetic (add, mul, inv, distributivity) | PASS |
| Cycle property (Pallas Fq = Vesta Fp) | PASS |
| Pallas curve ops (double, add, scalar mul, identity) | PASS |
| Vesta curve ops (double, add, scalar mul, identity) | PASS |

Pallas/Vesta cycle of curves (y^2 = x^3 + 5) for recursive proof composition. Each curve's scalar field equals the other's base field, enabling efficient recursion. CPU-side field arithmetic with Montgomery representation.

### Binius (Binary Tower)

**CPU field throughput:**

| Field | Mul throughput | ns/op |
|-------|---------------|-------|
| GF(2^8) | 130 Mop/s | 7.7 ns |
| GF(2^16) | 64 Mop/s | 16 ns |
| GF(2^32) | 23 Mop/s | 43 ns |
| GF(2^64) | 6.7 Mop/s | 149 ns |
| GF(2^128) | 1.9 Mop/s | 529 ns |

**GPU batch operations (GF(2^32)):**

| Size | GPU add | GPU mul |
|------|---------|---------|
| 2^14 | 0.60ms | 0.47ms |
| 2^16 | 0.86ms | 0.49ms |
| 2^18 | 0.56ms | 0.67ms |
| 2^20 | 3.5ms | 2.5ms |

**CPU additive FFT (binary):**

| Size | Time |
|------|------|
| 2^10 | 0.17ms |
| 2^12 | 0.81ms |
| 2^14 | 3.9ms |
| 2^16 | 21ms |

Binary tower field arithmetic with XOR addition (zero-cost) and table-based multiplication. Additive FFT for binary fields (distinct from standard NTT). All correctness tests pass: associativity, commutativity, distributivity, inverses, and FFT round-trip at all sizes through 2^12.

### Lasso Structured Lookups

| Lookups | Lasso Prove | Lasso Verify | LogUp Prove (comparison) |
|---------|-------------|--------------|--------------------------|
| 2^10 | 18ms | 29ms | 4.6ms |
| 2^14 | 44ms | 136ms | 16ms |
| 2^18 | 481ms | 1.6s | infeasible |

Structured lookup via tensor decomposition of large tables (e.g., range [0, 2^32) via 4 subtables of 256). Lasso's advantage over LogUp is for very large tables: at 2^18 lookups into a 2^32 table, LogUp is infeasible (would need 2^32 table entries) while Lasso handles it via subtable decomposition. Compression ratio: sqrt(T) instead of T.

### LogUp Lookup Argument

| Lookups (m=N) | Prove | Verify |
|---------------|-------|--------|
| 2^8 (256) | 8.5ms | 7.1ms |
| 2^10 (1024) | 12ms | 11ms |
| 2^12 (4096) | 15ms | 16ms |

Lookup argument via logarithmic derivatives. Optimal for small-to-medium tables where the full table fits in memory. Includes range proof support (limb decomposition for [0, 2^k) ranges). 7 correctness tests: simple, repeated, multiplicities, batch inverse, tamper rejection.

### cq (Cached Quotients Lookup)

Cached quotients lookup where prover time is independent of table size |T|. Correctness tests pass for all sizes (N=4 to N=32, various |T|). Currently crashes during performance benchmarks at larger sizes (signal 139). Benchmark command: `swift run -c release zkbench cq`.

### Poseidon2 M31 (Mersenne31, t=16)

**GPU hash pairs:**

| Pairs | GPU Time | Throughput |
|-------|----------|------------|
| 2^10 | 0.76ms | 1.4M hash/s |
| 2^14 | 0.90ms | 18M hash/s |
| 2^16 | 2.3ms | 29M hash/s |
| 2^18 | 3.3ms | 79M hash/s |
| 2^20 | 10.5ms | 100M hash/s |

**GPU Merkle tree (Poseidon2 M31):**

| Leaves | GPU Time |
|--------|----------|
| 2^10 | 0.66ms |
| 2^14 | 1.5ms |
| 2^16 | 2.5ms |
| 2^18 | 5.6ms |

CPU permutation: 2.0 us/perm. GPU achieves up to 100M hash/s at large batch sizes. Used by Circle STARK for Merkle commitments over M31 evaluations.

### BLS12-377 NTT and MSM

**BLS12-377 NTT (GPU):**

| Size | GPU Time | Throughput |
|------|----------|------------|
| 2^14 | 0.64ms | 26M elem/s |
| 2^16 | 1.5ms | 44M elem/s |
| 2^18 | 2.3ms | 114M elem/s |
| 2^20 | 6.2ms | 170M elem/s |
| 2^22 | 25ms | 167M elem/s |
| 2^24 | 110ms | 153M elem/s |

**BLS12-377 MSM (GPU):**

| Points | GPU Time |
|--------|----------|
| 2^8 | 10ms |
| 2^10 | 35ms |
| 2^14 | 40ms |
| 2^16 | 184ms |
| 2^18 | 218ms |

BLS12-377 uses 253-bit scalar field (TWO_ADICITY=47, ideal for NTT). MSM uses Pippenger with signed-digit recoding. NTT throughput is comparable to BN254 Fr (both 256-bit fields with 8x32-bit limbs on GPU).

### secp256k1 MSM

| Points | GPU (Metal) | C Pippenger | GPU/C speedup |
|--------|-------------|-------------|---------------|
| 2^8 | 1.5ms | 1.5ms | 1.0x |
| 2^10 | 4.3ms | 4.3ms | 1.0x |
| 2^12 | 16ms | 10ms | 0.6x |
| 2^14 | 21ms | 37ms | **1.7x** |
| 2^16 | 38ms | 98ms | **2.6x** |
| 2^18 | 77ms | 372ms | **4.8x** |

secp256k1 MSM with Pippenger + signed-digit recoding. GPU overtakes C Pippenger at n>=2^14. secp256k1 lacks the GLV endomorphism efficiency of BN254 (no known efficient endomorphism for short Weierstrass form over secp256k1 with these parameters).

### Pasta MSM (Pallas/Vesta)

| Points | Pallas GPU | Vesta GPU |
|--------|------------|-----------|
| 2^8 | 5.9ms | 4.4ms |
| 2^10 | 12ms | 11ms |
| 2^12 | 17ms | 17ms |
| 2^14 | 194ms | 204ms |
| 2^16 | 40ms | 51ms |

GPU MSM for Pasta curves. On-curve verification passes for both Pallas and Vesta. The 2^14 anomaly (slower than 2^16) is a window-size auto-tuning artifact that will be addressed in a future optimization pass.

### Tensor Proof Compression

| N (evaluations) | Compress | Verify | Proof Size | Compression Ratio |
|-----------------|----------|--------|------------|-------------------|
| 2^10 (1024) | 6.5ms | 6.2ms | 65 elems | **15.8x** |
| 2^14 (16384) | 25ms | 14ms | 173 elems | **94.7x** |
| 2^18 (262144) | 229ms | 39ms | 569 elems | **460.7x** |

**Proof size comparison (field elements):**

| numVars | Direct | Tensor | Basefold |
|---------|--------|--------|----------|
| 10 | 1024 | **65** | 4800 |
| 14 | 16384 | **173** | 8960 |
| 18 | 262144 | **569** | 14400 |
| 22 | 4194304 | **2117** | 21120 |

Tensor-based multilinear proof compression via sqrt(N) decomposition. At 2^22 evaluations, compresses from 4.2M field elements to 2117 -- a **1981x** compression ratio. Component breakdown at 2^14: tensor product 0.05ms, matrix-vector 5.8ms, sumcheck 0.3ms, eq+row eval 7.2ms.

### WHIR

| Size | Prove | Verify (full) | Verify (succinct) | Proof Size |
|------|-------|---------------|-------------------|------------|
| 2^10 | -- | -- | -- | 14.5 KB |
| 2^14 | 53ms | 16ms | 56ms | 28.2 KB |

Weighted Hashing of Indices for Reed-Solomon proximity testing. Prover time comparable to FRI (WHIR/FRI = 0.94x at 2^14), but proof size is ~3x larger (28 KB vs 9 KB). WHIR provides different security tradeoffs vs FRI: proximity to RS codes rather than low-degree testing.

### Transcript (Fiat-Shamir)

| Backend | 1000 absorb + 1000 squeeze | Throughput |
|---------|---------------------------|------------|
| Poseidon2 | 176ms | 11K ops/s |
| Keccak-256 | 0.89ms | 2.2M ops/s |

Duplex sponge transcript with domain separation, fork separation, and label-based absorption. Keccak backend is ~200x faster than Poseidon2 for transcript operations (Poseidon2's algebraic structure has higher per-operation cost). Both backends produce deterministic, distinct challenges.

### Proof Serialization

| Proof Type | Size |
|------------|------|
| KZG | 138 B |
| IPA (8 rounds) | 1586 B |
| Sumcheck (16 vars) | 1587 B |
| Lookup (N=64) | 3353 B |
| Lasso (4 chunks, m=32) | 7548 B |
| FRI commitment (2^8) | 16.5 KB |
| FRI commitment (2^14) | 1025 KB |

**Serialization throughput:**

| Format | 10KB x 1000 encode | 10KB x 1000 decode |
|--------|--------------------|--------------------|
| Base64 | 3.3ms | 3.6ms |
| Hex | 6.5s | 0.9s |

ProofWriter/ProofReader with type-safe field element, point, and label serialization. Supports Fr, PointProjective, UInt32/UInt64, raw bytes, hex, and base64 encoding. Truncation and label mismatch detection for tamper resistance.

### GPU Witness Generation

**BN254 Fr witness engine:**

| Trace | Rows | GPU | CPU | Speedup | Throughput |
|-------|------|-----|-----|---------|------------|
| Add chain (10 cols) | 2^16 | 0.71ms | 87ms | **121x** | 917M cells/s |
| Add chain (10 cols) | 2^18 | 3.0ms | 351ms | **117x** | 877M cells/s |
| Add chain (10 cols) | 2^20 | 11ms | -- | -- | 965M cells/s |
| Mul-heavy (8 cols) | 2^16 | 2.6ms | 87ms | **33x** | 201M cells/s |
| Mul-heavy (8 cols) | 2^20 | 7.5ms | -- | -- | 1.1B cells/s |
| Poseidon2-like (4 cols) | 2^16 | 2.7ms | 808ms | **295x** | 96M cells/s |

**M31 witness engine (Circle STARK):**

| Trace | Rows | GPU | Throughput |
|-------|------|-----|------------|
| Fibonacci AIR (2 cols) | 2^20 | 1.4ms | 1.5B cells/s |
| Fibonacci AIR (2 cols) | 2^22 | 5.4ms | 1.5B cells/s |
| Generic M31 (4 cols) | 2^20 | 1.3ms | 3.3B cells/s |

Instruction-stream GPU witness evaluation. The engine compiles a trace program (add/mul/const operations) into a Metal compute kernel that evaluates all rows in parallel. M31 witness engine achieves up to 3.3B cells/s due to single-word 32-bit arithmetic.

### Constraint IR Evaluation

| Rows | Constraints | GPU | CPU | Speedup | Throughput |
|------|-------------|-----|-----|---------|------------|
| 2^10 | 20 | 0.59ms | 12ms | **21x** | 35M/s |
| 2^14 | 20 | 1.4ms | 198ms | **140x** | 232M/s |
| 2^16 | 20 | 5.3ms | -- | -- | 248M/s |
| 2^14 | 48 (Fibonacci) | 2.1ms | -- | -- | 383M/s |
| 2^16 | 48 (Fibonacci) | 7.0ms | -- | -- | 449M/s |

**Compilation time:**

| Circuit | Constraints | Compile Time |
|---------|-------------|-------------|
| R1CS 10 gates | 10 | 74ms |
| R1CS 100 gates | 100 | 202ms |
| Fibonacci 100 steps | 98 | 130ms |
| RangeCheck 32-bit | 33 | 69ms |

Runtime constraint compilation: IR -> Metal shader source -> GPU pipeline. Constraint expressions are compiled to Metal compute kernels that evaluate all rows in parallel. Compilation is a one-time cost; subsequent evaluations reuse the compiled pipeline.

### Fused NTT+Constraint

Fused NTT and constraint evaluation in a single GPU pass. Currently limited by threadgroup memory constraints (requires >32KB shared memory on some configurations). Benchmark command: `swift run -c release zkbench fused`.

### Reed-Solomon Erasure Coding

NTT-based Reed-Solomon erasure coding over BabyBear (GF(2^31-2^27+1)) and GF(2^16). Supports encode (k data shards -> n coded shards) and decode (any k of n shards recovers original data). Correctness verified: encode then decode from first k or last k shards. Benchmark command: `swift run -c release zkbench erasure`.

### Lattice Crypto (Kyber/Dilithium)

**Kyber-768 KEM (post-quantum key encapsulation):**

| Operation | Time | Throughput |
|-----------|------|------------|
| KeyGen | 0.07ms | 13K ops/s |
| Encapsulate | 0.08ms | 12K ops/s |
| Decapsulate | 0.02ms | 67K ops/s |

**Dilithium2 (ML-DSA-44) signatures:**

| Operation | Time | Throughput |
|-----------|------|------------|
| KeyGen | 0.07ms | 15K ops/s |
| Sign | 0.07ms | 14K ops/s |
| Verify | 0.04ms | 25K ops/s |

**GPU NTT throughput (256-element polynomials):**

| Batch Size | Kyber NTTs/s | Dilithium NTTs/s |
|------------|-------------|-----------------|
| 100 | 278K | 158K |
| 1,000 | 2.8M | 757K |
| 10,000 | 5.6M | 1.6M |

CPU NTT baseline: Kyber 133K NTTs/s, Dilithium 415K NTTs/s. GPU batch NTT provides 42x (Kyber) and 3.9x (Dilithium) throughput improvement at batch size 10K. Correctness verified for both KEM shared-secret agreement and signature verification.

### HE NTT (RNS-based)

RNS-based Number Theoretic Transform for homomorphic encryption (CKKS/BFV). Processes L CRT limbs in parallel, each using an independent NTT modulus (verified primes with required roots of unity). Supports forward/inverse NTT, pointwise multiply, and batch NTT across multiple polynomials. BFV operations: keygen, encrypt, decrypt, homomorphic add/multiply with relinearization. Benchmark not yet wired into main harness; available via `runHEBench()` in source.

### NEON CPU Benchmarks

**BabyBear NTT (NEON SIMD 4-wide Montgomery):**

| Size | Vanilla CPU | NEON C | Speedup | GPU |
|------|-------------|--------|---------|-----|
| 2^16 | 2.2ms | 0.4ms | **5.9x** | 0.13ms |
| 2^18 | 9.9ms | 1.8ms | **5.6x** | 0.44ms |
| 2^20 | 45ms | 8.5ms | **5.3x** | 2.5ms |
| 2^22 | 202ms | 37ms | **5.4x** | 2.8ms |

**Goldilocks NTT (C with `__uint128_t`):**

| Size | Vanilla CPU | C Opt | Speedup | NEON NTT |
|------|-------------|-------|---------|----------|
| 2^16 | 2.6ms | 1.2ms | **2.2x** | 0.76ms (1.6x) |
| 2^18 | 12ms | 5.5ms | **2.1x** | 3.7ms (1.5x) |
| 2^20 | 53ms | 25ms | **2.1x** | 16ms (1.5x) |

**Keccak-256 NEON (single hash_pair):**

| Size | NEON | GPU | GPU/NEON |
|------|------|-----|----------|
| 2^10 | 0.55ms | 0.44ms | 1.2x |
| 2^12 | 2.4ms | 0.43ms | **5.5x** |
| 2^14 | 11ms | 0.51ms | **22x** |
| 2^16 | 38ms | 1.5ms | **25x** |

NEON single hash: 0.63 us/hash (11.3x over Swift CPU). GPU overtakes NEON at 2^12+.

**Blake3 NEON (parent hash pairs):**

| Size | NEON | GPU | GPU/NEON |
|------|------|-----|----------|
| 2^10 | 0.06ms | 0.50ms | NEON wins |
| 2^12 | 0.62ms | 0.52ms | ~tied |
| 2^14 | 1.1ms | 0.32ms | **3.4x** |
| 2^16 | 4.3ms | 0.84ms | **5.1x** |
| 2^18 | 17ms | 1.4ms | **12x** |

NEON single hash: 0.10 us/hash (6.1x over Swift CPU). GPU overtakes NEON at 2^14+.

### CPU Benchmarks (C Pippenger, C NTT)

**BN254 Fr NTT (C with unrolled CIOS Montgomery):**

| Size | Vanilla CPU | C Opt | Speedup | GPU |
|------|-------------|-------|---------|-----|
| 2^14 | 82ms | 2.7ms | **30x** | 0.45ms |
| 2^16 | 371ms | 12ms | **30x** | 1.5ms |
| 2^18 | 1.7s | 57ms | **29x** | 2.0ms |
| 2^20 | 7.5s | 247ms | **30x** | 6.1ms |

**BN254 MSM (4-way comparison):**

| Points | Vanilla | Swift Pip | C Pip | GPU |
|--------|---------|-----------|-------|-----|
| 2^8 | 440ms | 17ms | **1.4ms** | 1.5ms |
| 2^10 | 1.8s | 44ms | **3.2ms** | 3.7ms |
| 2^12 | 7.0s | 119ms | **7.8ms** | 14ms |
| 2^14 | 28s | 334ms | **22ms** | **22ms** |
| 2^16 | -- | -- | 58ms | **26ms** |
| 2^18 | -- | -- | 228ms | **44ms** |
| 2^20 | -- | -- | 774ms | **119ms** |

C Pippenger is 12-15x faster than Swift Pippenger via `__uint128_t` CIOS Montgomery. GPU overtakes C Pippenger at n>=2^14. ARM64 ASM Montgomery multiply matches C -O3 performance (within 5%) -- the compiler generates near-optimal `mul`/`umulh` sequences from `__uint128_t`.

### Streaming Verification

**FRI proof verification pipeline:**

| Evaluations | Sequential | Task-Queue | Speedup |
|-------------|------------|------------|---------|
| 2^10 | 11ms | 6.9ms | **1.7x** |
| 2^12 | 14ms | 8.1ms | **1.7x** |
| 2^14 | 16ms | 9.4ms | **1.7x** |
| 2^16 | 19ms | 11ms | **1.8x** |

**GPU EC on-curve batch verification:**

| Points | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| 100 | 0.15ms | 0.17ms | ~1x |
| 1,000 | 1.4ms | 0.41ms | **3.4x** |
| 10,000 | 14ms | 0.33ms | **42x** |
| 100,000 | 145ms | 1.8ms | **82x** |

Task-queue streaming verification uses Apple Silicon unified memory for zero-copy CPU/GPU handoff. GPU batch on-curve checking scales to 82x speedup at 100K points.

### Incremental Merkle

| Operation | Tree Size | Time |
|-----------|-----------|------|
| Append 1 leaf | 2^16 (32K filled) | 1.9ms |
| Append 1 leaf | 2^20 (512K filled) | 2.4ms |
| Batch 256 append | 2^20 | 13ms (vs 124ms full rebuild = **9.2x**) |
| Batch 1024 append | 2^20 | 14ms (vs 124ms full rebuild = **8.9x**) |
| Batch 4096 append | 2^20 | 17ms (vs 124ms full rebuild = **7.3x**) |
| Update 1 leaf | 2^16 | 2.0ms (vs 21ms full = **10.9x**) |

Poseidon2 BN254 Fr incremental Merkle tree with single-leaf append, batch append, and random update. Exploits Apple Silicon unified memory. Note: large sequential builds have a known correctness regression (root mismatch after many incremental appends). Batch append and single-leaf operations are correct.

### Batch Field Operations

| Operation | Size | C Batch | Swift | Speedup |
|-----------|------|---------|-------|---------|
| Add | 1K | 3 us (2.9 ns/op) | 196 us | **66x** |
| Add | 10K | 26 us (2.6 ns/op) | 2.1ms | **79x** |
| Add | 100K | 264 us (2.6 ns/op) | 16ms | **60x** |
| Mul | 100K | 1.3ms (13.4 ns/op) | -- | -- |
| Sub | 100K | 225 us (2.3 ns/op) | -- | -- |
| Neg | 100K | 140 us (1.4 ns/op) | -- | -- |

C-optimized vectorized BN254 Fr field arithmetic with `__uint128_t` CIOS Montgomery. 60-79x faster than Swift for addition. Parallel (GCD) provides additional 2.8x at 1M elements.

### IPA Accumulation (Pallas)

| Operation | n=4 | n=16 |
|-----------|-----|------|
| Prove | 32ms | 121ms |
| Verify | 17ms | 41ms |
| Accumulate | 7.3ms | 14ms |
| Decide | 9.0ms | 26ms |
| Batch decide (5x) | 16ms (2.7x vs individual) | -- |

Halo-style IPA accumulation over Pallas curve. Accumulates multiple IPA proofs into a single instance; batch decide verifies all at once. Supports recursive hash composition (3 steps: 140ms). Used with Vesta for recursive proof systems.

### Theoretical Performance Analysis

How close each primitive is to the hardware floor (M3 Pro: ~3.6 TFLOPS, ~150 GB/s bandwidth), ranked by optimization headroom:

| Rank | Primitive | Current | Theoretical Floor | Bottleneck | Headroom |
|------|-----------|---------|-------------------|------------|----------|
| 1 | P2 Merkle 2^16 | 22ms | ~0.6ms (compute) | Dispatch latency (16 levels) | ~37x |
| 2 | KZG commit 2^10 | 4.6ms | ~0.5ms | MSM-dominated (small N) | ~9x |
| 3 | MSM BN254 2^18 | 45ms | ~5ms (scatter BW) | Random-access BW | ~9x |
| 4 | FRI Fold 2^20 | 1.96ms | ~0.3ms (BW) | Bandwidth | ~7x |
| 5 | ECDSA batch 64 (CPU) | 8ms | ~1ms | C CIOS scalar mul | ~8x |
| 6 | Sumcheck 2^20 | 7.3ms | ~0.85ms (BW) | Bandwidth | ~9x |
| 7 | Radix Sort 2^20 | 4.1ms | ~1ms (BW) | Sequential passes + BW | ~4x |
| 8 | NTT BN254 2^22 | 26ms | ~2.9ms (compute) | Compute + strided BW | ~9x |
| 9 | Blake3 Batch 2^20 | 3.5ms | ~0.6ms (BW) | Bandwidth | ~6x |
| 10 | IPA prove n=256 | 13ms | ~10ms | C scalar mul + GPU batch fold | ~1.3x |
| 11 | Keccak Batch 2^18 | 1.4ms | ~0.5ms (compute) | Compute | ~3x |
| 12 | Verkle proof 256 (CPU) | 23ms | ~10ms | IPA-dominated (C scalar mul) | ~2.3x |
| 13 | NTT Goldilocks 2^24 | 3.0ms | ~1.8ms (compute) | Compute ~= BW | ~1.7x |
| 14 | NTT BabyBear 2^24 | 2.0ms | ~1.7ms (BW) | Bandwidth | ~1.2x |

Notes: IPA/Verkle near theoretical floor after C CIOS + GPU batch fold. GKR achieves 190x improvement via sparse wiring predicates. MSM's realistic floor accounts for scatter-gather inefficiency in bucket accumulation. Poseidon2 Merkle overhead comes from 16 sequential kernel dispatches (~0.5ms each). BabyBear and Goldilocks NTT are near-optimal -- within 1-2x of hardware bandwidth limits.

## Supported Fields

- **BN254 Fr** (scalar field) -- Montgomery CIOS, 8x32-bit limbs (GPU) / 4x64-bit (C CPU)
- **BN254 Fp** (base field) -- Montgomery CIOS, SOS squaring
- **BLS12-377 Fr** (scalar field) -- Montgomery CIOS, 8x32-bit limbs, TWO_ADICITY=47
- **BLS12-377 Fq** (base field) -- Montgomery CIOS, 6x64-bit (C CPU)
- **secp256k1 Fp** (base field) -- Montgomery CIOS, 4x64-bit (C CPU)
- **secp256k1 Fr** (scalar field) -- for ECDSA signature verification
- **Goldilocks** (p = 2^64 - 2^32 + 1) -- native 64-bit reduction
- **BabyBear** (p = 2^31 - 2^27 + 1) -- 32-bit arithmetic
- **Mersenne31** (p = 2^31 - 1) -- single-word arithmetic, circle group order 2^31
- **CM31** -- complex extension M31[i]/(i^2+1) for Circle STARK challenges
- **BLS12-381 Fr** -- 256-bit scalar field, Montgomery CIOS, TWO_ADICITY=32
- **BLS12-381 Fp** -- 381-bit base field (12x32-bit limbs), Fp2/Fp6/Fp12 tower
- **Pallas Fp** -- 255-bit base field for Pallas curve (cycle with Vesta)
- **Vesta Fp** -- 255-bit base field for Vesta curve (cycle with Pallas)
- **Binary Tower** -- GF(2^8)->GF(2^16)->GF(2^32)->GF(2^64)->GF(2^128) (XOR addition)

## Architecture

Metal GPU shaders handle compute-intensive operations. The Swift engine layer manages buffer caching, pipeline dispatch, and host-device coordination. C CIOS libraries (`NeonFieldOps`) provide optimized CPU paths for field arithmetic, NTT, and MSM using ARM64 `__uint128_t` and NEON SIMD.

```
Sources/
  Shaders/         # Metal GPU kernels
    fields/        # Field arithmetic (BN254, BLS12-377/381, secp256k1, Goldilocks, BabyBear, M31, Pallas, Vesta, binary tower)
    geometry/      # Elliptic curve operations (BN254 G1, BLS12-377 G1, secp256k1, Pallas, Vesta)
    msm/           # Multi-scalar multiplication kernels
    ntt/           # NTT butterfly + fused sub-block + Circle NTT kernels
    hash/          # Poseidon2 (BN254 + M31), Keccak-256, Blake3
    fri/           # FRI + Circle FRI folding kernels
    sumcheck/      # Sumcheck round kernels
    poly/          # Polynomial evaluation/interpolation
    sort/          # GPU radix sort kernels
    witness/       # GPU witness trace evaluation
    constraint/    # Fused NTT+constraint kernels
    basefold/      # Basefold fold kernels
    lattice/       # Kyber/Dilithium NTT kernels
    erasure/       # Reed-Solomon erasure coding
  NeonFieldOps/    # C/ARM64 optimized CPU primitives
    babybear_ntt.c       # NEON SIMD NTT (4-wide Montgomery)
    goldilocks_ntt.c     # __uint128_t optimized NTT
    mont256.c            # BN254 Fr CIOS NTT (29-38x over Swift)
    bn254_msm.c          # BN254 Pippenger MSM + synthetic division
    secp256k1_ops.c      # secp256k1 field/curve ops + Pippenger MSM
  zkMetal/         # Swift engine layer
    Fields/        # CPU-side field arithmetic (BN254, BLS12-377/381, secp256k1, M31, Pallas, Vesta, binary tower)
    MSM/           # MSM engines (BN254, BLS12-377, secp256k1, Pallas, Vesta)
    NTT/           # NTT engines (BN254, BLS12-377, Goldilocks, BabyBear, Circle NTT M31, RNS/HE)
    Hash/          # Poseidon2 (BN254 + M31), Keccak, Blake3, Merkle tree engines
    Polynomial/    # FRI, Circle FRI, Sumcheck, Sparse Sumcheck engines
    Poly/          # Polynomial operations engine
    PCS/           # Basefold, Brakedown polynomial commitment engines
    KZG/           # KZG polynomial commitment engine (single + batch)
    IPA/           # Inner product argument (Bulletproofs-style)
    Verkle/        # Verkle tree engine (Pedersen + IPA)
    ECDSA/         # secp256k1 batch ECDSA verification
    Lookup/        # LogUp, Lasso, cq lookup argument engines
    Sort/          # GPU radix sort engine
    Curve/         # BLS12-381 G1/G2 + pairings
    CircleSTARK/   # Full Circle STARK prover/verifier over M31
    Folding/       # HyperNova CCS folding scheme
    GKR/           # GKR interactive proof for layered circuits
    Plonk/         # Plonk preprocessed prover/verifier
    Groth16/       # Groth16 zk-SNARK with BN254 pairings
    Witness/       # GPU witness trace evaluation engine
    Constraint/    # Constraint IR compiler + fused evaluation
    Transcript/    # Fiat-Shamir transcript (Poseidon2 + Keccak)
    Serialization/ # Proof serialization/deserialization
    Accumulation/  # Halo-style accumulation schemes (IPA + Pasta)
    Lattice/       # Post-quantum crypto (Kyber KEM, Dilithium signatures)
    HE/            # Homomorphic encryption (RNS NTT for CKKS/BFV)
    ErasureCoding/ # Reed-Solomon erasure coding
    Verifier/      # Streaming/batch proof verification
    CPU/           # GCD parallel CPU implementations
  zkbench/         # Benchmark harness
  zkmsm-cli/       # Standalone MSM CLI tool
Tests/
  zkMetalTests/    # Correctness tests
```

## Usage

### As a library

```swift
import zkMetal

// MSM
let msm = try MetalMSM()
let result = try msm.msm(points: points, scalars: scalars)

// NTT (BN254)
let ntt = try NTTEngine()
let transformed = try ntt.ntt(values)
let inverse = try ntt.intt(transformed)

// Poseidon2 hashing
let p2 = try Poseidon2Engine()
let hashes = try p2.hashBatch(inputs)

// Keccak-256
let keccak = try Keccak256Engine()
let digests = try keccak.hashBatch(messages)

// Merkle tree (Poseidon2)
let merkle = try Poseidon2MerkleEngine()
let tree = try merkle.buildTree(leaves)

// FRI folding
let fri = try FRIEngine()
let folded = try fri.multiFold(evals: evaluations, betas: challenges)

// Sumcheck
let sc = try SumcheckEngine()
let (rounds, finalEval) = try sc.fullSumcheck(evals: evals, challenges: challenges)

// KZG Commitments
let kzg = try KZGEngine(srs: srs)
let commitment = try kzg.commit(polynomial)
let proof = try kzg.open(polynomial, at: z)

// Blake3
let blake3 = try Blake3Engine()
let hashes = try blake3.hash64(input)

// BLS12-377 NTT
let bls377ntt = try BLS12377NTTEngine()
let transformed = try bls377ntt.ntt(values377)

// IPA (Bulletproofs-style)
let ipa = try IPAEngine()
let proof = try ipa.prove(generators: G, Q: Q, a: values, b: basis)
let valid = try ipa.verify(generators: G, Q: Q, commitment: C, proof: proof)

// Verkle tree
let verkle = VerkleEngine(width: 16, ipa: ipa)
let tree = try verkle.buildTree(leaves: values)
let pathProof = try verkle.provePathProof(tree: tree, leafIndex: 0)

// ECDSA batch verification
let ecdsa = ECDSAEngine()
let valid = try ecdsa.batchVerifyProbabilistic(signatures: sigs, messages: msgs, publicKeys: keys)

// Circle NTT (Mersenne31)
let circleNTT = try CircleNTTEngine()
let transformed = try circleNTT.ntt(m31Values)

// Circle FRI
let circleFRI = try CircleFRIEngine()
let commitment = try circleFRI.commitPhase(evaluations: evals, challenges: challenges)

// Basefold PCS (NTT-free)
let basefold = try BasefoldEngine()
let bfCommit = try basefold.commit(evaluations: multilinearEvals)
let bfProof = try basefold.open(commitment: bfCommit, point: evalPoint)

// Fiat-Shamir transcript
let transcript = Transcript(backend: .keccak256)
transcript.absorb(commitment)
let challenge = transcript.squeeze()

// BLS12-381 pairing
let engine = BLS12381Engine()
let pairingResult = engine.pairing(g1Point, g2Point)

// Poseidon2 over M31
let p2m31 = try Poseidon2M31Engine()
let merkleRoot = try p2m31.merkleCommit(leaves: m31Leaves)
```

### Benchmarks

```bash
swift run -c release zkbench msm       # MSM (BN254 G1)
swift run -c release zkbench ntt       # NTT (BN254 Fr)
swift run -c release zkbench bls377    # NTT (BLS12-377 Fr)
swift run -c release zkbench p2        # Poseidon2
swift run -c release zkbench keccak    # Keccak-256
swift run -c release zkbench blake3    # Blake3
swift run -c release zkbench merkle    # Merkle trees
swift run -c release zkbench poly      # Polynomial ops
swift run -c release zkbench kzg       # KZG commitments
swift run -c release zkbench kzg-batch # Batch KZG openings
swift run -c release zkbench fri       # FRI folding (incl. fold-by-4/8)
swift run -c release zkbench sumcheck  # Sumcheck
swift run -c release zkbench sparse    # Sparse sumcheck
swift run -c release zkbench usc       # Univariate sumcheck
swift run -c release zkbench bls377msm # MSM (BLS12-377 G1)
swift run -c release zkbench secpmsm   # MSM (secp256k1 G1)
swift run -c release zkbench pastamsm  # MSM (Pallas/Vesta)
swift run -c release zkbench ecdsa     # ECDSA batch verification
swift run -c release zkbench ipa       # IPA (Bulletproofs-style)
swift run -c release zkbench verkle    # Verkle trees
swift run -c release zkbench lookup    # LogUp lookup argument
swift run -c release zkbench lasso     # Lasso structured lookups
swift run -c release zkbench cq        # cq cached quotients
swift run -c release zkbench circle    # Circle NTT + M31 benchmarks
swift run -c release zkbench circle-fri # Circle FRI over M31
swift run -c release zkbench circle-stark # Circle STARK prover/verifier
swift run -c release zkbench p2m31     # Poseidon2 over Mersenne31
swift run -c release zkbench basefold  # Basefold PCS
swift run -c release zkbench zeromorph # Zeromorph PCS
swift run -c release zkbench brakedown # Brakedown PCS
swift run -c release zkbench transcript # Fiat-Shamir transcript
swift run -c release zkbench serialize # Proof serialization
swift run -c release zkbench witness   # GPU witness generation
swift run -c release zkbench constraint # Constraint IR evaluation
swift run -c release zkbench fused     # Fused NTT+constraint
swift run -c release zkbench erasure   # Reed-Solomon erasure coding
swift run -c release zkbench lattice   # Kyber/Dilithium post-quantum
swift run -c release zkbench batch-field # Batch field operations
swift run -c release zkbench imerkle   # Incremental Merkle
swift run -c release zkbench stream-verify # Streaming verification
swift run -c release zkbench sort      # GPU radix sort
swift run -c release zkbench pasta     # Pasta curves (Pallas/Vesta)
swift run -c release zkbench binius    # Binary tower fields
swift run -c release zkbench fold      # HyperNova folding
swift run -c release zkbench tensor    # Tensor proof compression
swift run -c release zkbench whir      # WHIR
swift run -c release zkbench accum     # IPA accumulation (Pallas)
swift run -c release zkbench dp        # Data-parallel GKR
swift run -c release zkbench gkr       # GKR protocol
swift run -c release zkbench plonk     # Plonk prover
swift run -c release zkbench groth16   # Groth16 zk-SNARK
swift run -c release zkbench bls381    # BLS12-381 field + curve + pairing
swift run -c release zkbench versions  # Print primitive versions
swift run -c release zkbench test      # Correctness tests (all primitives)
swift run -c release zkbench cpu       # CPU-optimized vs GPU comparison
swift run -c release zkbench calibrate # Re-calibrate GPU parameters
swift run -c release zkbench all --no-cpu  # GPU-only (skip slow CPU baselines)
swift run -c release zkbench gl-neon   # Goldilocks NEON NTT
swift run -c release zkbench keccak-neon # Keccak-256 NEON
swift run -c release zkbench blake3-neon # Blake3 NEON
swift run -c release zkbench asm       # ARM64 ASM vs C Montgomery multiply
```

### MSM CLI

```bash
# Benchmark
swift run -c release zkmsm --bench 65536

# Compute from JSON
echo '{"points": [["0x1","0x2"]], "scalars": ["0x2a"]}' | swift run -c release zkmsm
```

## Auto-Tuning

zkMetal automatically calibrates GPU parameters (threadgroup sizes, FFT thresholds, MSM window sizes) on first use. Results are cached to `~/.zkmetal/tuning.json` and reused across runs. Calibration re-triggers automatically when the GPU changes.

To force re-calibration:

```bash
swift run -c release zkbench calibrate
```

This takes ~500ms and prints the discovered parameters. Different Apple Silicon chips (M1, M2, M3, M4) have different optimal settings -- auto-tuning ensures peak performance on any hardware.

## Building

Requires macOS 13+ and Xcode with Metal support.

```bash
swift build -c release
```

## Design Decisions

- **Montgomery form everywhere**: All field elements stay in Montgomery representation on GPU. Conversion happens only at host boundaries.
- **Buffer caching**: GPU Metal buffers are cached and reused across calls to avoid allocation overhead.
- **Four-step FFT**: Large NTTs (>2^16) split into sub-blocks processed in shared memory, reducing global memory traffic.
- **Fused kernels**: Multi-round FRI folding and Poseidon2 full permutations avoid intermediate buffer round-trips. NTT uses fused bitrev+butterfly and twiddle fusion for 30-47% improvement.
- **Signed-digit MSM**: Scalar recoding halves bucket count, reducing bucket accumulation work.
- **GLV endomorphism**: BN254's efficient endomorphism splits 256-bit scalar muls into two 128-bit half-width muls.
- **C CIOS field arithmetic**: Hot-path 256-bit Montgomery multiplication uses C `__uint128_t` compiled with `-O3`, which is 29-30x faster than Swift for BN254 Fr carry chains. Used in CPU NTT, MSM, IPA, KZG, and ECDSA.
- **Small-input fast path**: MSM automatically routes to multi-threaded C Pippenger for small inputs (BN254 n<=2048, secp256k1 n<=1024) to avoid GPU dispatch overhead.

## Correctness & Testing

Run the full correctness suite with `swift run -c release zkbench test`. All GPU kernels are verified against CPU reference implementations. The CPU references are vanilla single-threaded implementations preserved unchanged for correctness verification.

| Component | Source | Verification |
|-----------|--------|-------------|
| **BN254 curve** | Standard parameters (same as Ethereum/bn256) | Field arithmetic unit tests |
| **BN254 MSM** | Pippenger + signed-digit + GLV | GPU vs CPU cross-check (low + full scalars), on-curve check |
| **BLS12-377 MSM** | Pippenger + signed-digit | GPU vs CPU cross-check, determinism, on-curve check |
| **secp256k1 MSM** | Pippenger + signed-digit | Identity, 2G, 5G, 16-pt cross-check, 256-pt determinism, on-curve |
| **Pallas/Vesta MSM** | Pippenger + signed-digit | GPU on-curve check for both curves |
| **Poseidon2** | [HorizenLabs reference](https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs) | Known-answer test + GPU vs CPU batch cross-check |
| **Poseidon2 M31** | t=16 Mersenne31 permutation | GPU vs CPU cross-check + Merkle root match |
| **Keccak-256** | FIPS 202 (SHA-3) | NIST test vectors + GPU vs CPU batch cross-check |
| **Blake3** | [BLAKE3 spec](https://github.com/BLAKE3-team/BLAKE3-specs) | Reference vector + GPU vs CPU batch cross-check |
| **Merkle trees** | Poseidon2, Keccak, Blake3 backends | GPU vs CPU root comparison + parallel structure validation |
| **NTT** | Cooley-Tukey (DIT) / Gentleman-Sande (DIF) | Round-trip (2^10, 2^20, 2^22) + CPU cross-check (all 4 fields) |
| **FRI** | Standard FRI protocol | GPU vs CPU fold cross-check + full commit/query/verify round-trip |
| **FRI fold-by-4/8** | Multi-element folding | Cross-check vs fold-by-2 + full protocol verification |
| **Sumcheck** | Standard interactive protocol | S(0)+S(1)=sum verification + GPU vs CPU reduce/roundPoly |
| **Sparse Sumcheck** | O(nnz) sparse multilinear | Round-poly match, reduce match, full protocol, proof verify |
| **KZG** | Polynomial commitment (MSM-based) | Commit linearity, multi-point eval, constant/linear poly checks |
| **Batch KZG** | Random linear combination | Batch verify-by-reopen, algebraic verify, tamper rejection |
| **IPA** | Bulletproofs-style inner product argument | Prove + verify at n=4,16,64,256 + wrong-value rejection |
| **Verkle trees** | Width-N Pedersen + IPA openings | Single openings, tree build, path proofs, wrong-root rejection |
| **LogUp** | Lookup via logarithmic derivatives | 7 tests: simple, repeated, multiplicities, batch inverse, tamper rejection |
| **Lasso** | Tensor decomposition lookups | Range [0,2^32) via 4 subtables, tamper rejection, wrong-lookup rejection |
| **cq** | Cached quotients lookup | Simple, repeated, multiplicities, asymmetric, N>|T|, tamper rejection |
| **ECDSA** | secp256k1 batch verification | Single verify, wrong msg/key rejection, batch 64, bad-sig detection |
| **Radix sort** | LSD radix sort (4-pass, 8-bit) | 10 tests: sorted, reverse, duplicates, random, KV, edge cases |
| **Goldilocks** | p = 2^64 - 2^32 + 1 (standard) | NTT round-trip + CPU cross-check |
| **BabyBear** | p = 2^31 - 2^27 + 1 (standard) | NTT round-trip + CPU cross-check |
| **Mersenne31** | p = 2^31 - 1 (standard) | Field arithmetic + circle group tests |
| **Circle NTT** | Circle group over M31 | GPU vs CPU roundtrip at all sizes 2 through 4096 |
| **Circle STARK** | Fibonacci AIR over M31 | Prove + verify + tampered proof rejection |
| **Plonk** | Preprocessed BN254 + KZG | Prove + verify at n=16,64,256,1024 |
| **Groth16** | BN254 R1CS + pairings | Prove + verify, bilinearity checks |
| **GKR** | Layered circuit sumcheck | 1-layer, 2-layer, hash circuits, inner product circuits |
| **Basefold** | Sumcheck-based folding PCS | Single fold, multi-fold, eval correctness, proof verify, rejection |
| **HyperNova** | CCS folding | CCS satisfaction, fold verification, folded decide |
| **BLS12-381** | Full tower arithmetic | Fr/Fp/Fp2/Fp6/Fp12 arithmetic, G1/G2 ops, bilinearity |
| **Pasta curves** | Pallas/Vesta cycle | Field arithmetic, cycle property, curve ops for both curves |
| **Binius** | Binary tower GF(2^k) | Arithmetic properties, inverses (all 255 GF(2^8)), FFT roundtrip |
| **Tensor proof** | sqrt(N) compression | Decomposition correctness, compress+verify, wrong-value rejection |
| **WHIR** | RS proximity testing | Prove + verifyFull + verify (succinct) |
| **Transcript** | Fiat-Shamir duplex sponge | Determinism, domain separation, fork separation, sequential squeezes |
| **Serialization** | ProofWriter/ProofReader | Fr/Point/array roundtrip, truncation detection, label mismatch |
| **Witness gen** | GPU instruction-stream | GPU vs CPU cross-check for add-chain, mul-heavy, Poseidon2-like |
| **Constraint IR** | Runtime compilation | Fibonacci valid/invalid, R1CS, boolean, GPU vs CPU match |
| **Lattice crypto** | Kyber-768 + Dilithium2 | KEM shared secret agreement, signature verification |
| **Streaming verify** | Task-queue pipeline | Sequential/pipelined/double-buffered, batch Merkle, wrong root rejection |
| **Parallel CPU** | GCD multithreaded implementations | Cross-checked against vanilla CPU for NTT (Fr, Bb, Gl), MSM, batch hash, Merkle |
| **NEON BabyBear** | C/ARM NEON Montgomery NTT (4-wide SIMD, Plonky3 technique) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **NEON Keccak** | C/ARM NEON Keccak-256 | NIST test vectors + cross-check against Swift CPU and GPU |
| **NEON Blake3** | C/ARM NEON Blake3 parent hash | Cross-check against Swift CPU and GPU for 100 pairs |
| **C Goldilocks** | Optimized C NTT (`__uint128_t` mul pipelining) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **C BN254 Fr** | Fully unrolled 4-limb CIOS Montgomery NTT | Cross-checked against vanilla cpuNTT + round-trip verification |

Every benchmark run includes correctness checks (printed as PASS/FAIL). The test suite (`swift test`) covers field arithmetic, curve operations, and NTT correctness.

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization. To continue tuning for your hardware or workload, install the skill and run `/floptimizer` in a Claude Code session from this repo.
