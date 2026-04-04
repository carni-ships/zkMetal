# zkMetal

GPU-accelerated zero-knowledge cryptography library for Apple Silicon, written in Metal and Swift. 60+ primitives spanning field arithmetic, MSM, NTT, hash functions, polynomial commitments, proof protocols, signatures, post-quantum crypto, and homomorphic encryption across 18 fields and 10 elliptic curves.

## Contents

- [Primitives](#primitives)
- [Performance](#performance)
  - Core: [MSM](#msm-bn254-g1) | [NTT](#ntt) | [Hashing](#hashing) | [Merkle Trees](#merkle-trees) | [Radix Sort](#gpu-radix-sort)
  - Polynomial: [FRI](#fri-folding-bn254-fr) | [Sumcheck](#sumcheck-bn254-fr) | [Polynomial Ops](#polynomial-ops-bn254-fr)
  - Commitments: [KZG](#kzg-commitments-bn254-g1) | [Batch KZG](#batch-kzg-bn254-g1) | [Basefold](#basefold-pcs-bn254-fr)
  - Proof Systems: [Circle STARK](#circle-stark-mersenne31) | [Plonk](#plonk-bn254-kzg) | [Groth16](#groth16-bn254) | [GKR](#gkr-bn254-fr-layered-circuits)
  - Consolidated: [Other Curve MSM](#other-curve-msm) | [CPU Optimizations](#cpu-optimizations) | [Supporting Primitives](#supporting-primitives) | [Advanced Protocols](#advanced-protocols) | [Application Primitives](#application-primitives)
- [Theoretical Performance Analysis](#theoretical-performance-analysis)
- [Supported Fields](#supported-fields)
- [Architecture](#architecture)
- [Usage](#usage)
- [Auto-Tuning](#auto-tuning)
- [Building](#building)
- [Design Decisions](#design-decisions)
- [Correctness & Testing](#correctness--testing)

## Primitives

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **MSM** | GPU/CPU | Multi-scalar multiplication (Pippenger + signed-digit + GLV) -- BN254, BLS12-377, secp256k1, Pallas, Vesta, Ed25519, Grumpkin |
| **NTT** | GPU/CPU | Number theoretic transform (four-step FFT, fused bitrev+butterfly, twiddle fusion) -- BN254, BLS12-377, Goldilocks, BabyBear |
| **Poseidon2** | GPU | Algebraic hash (t=3 BN254 Fr; t=16 M31) |
| **Keccak-256** | GPU/CPU | SHA-3 hash (fused subtree Merkle) |
| **Blake3** | GPU/CPU | BLAKE3 hash (batch + Merkle trees) |
| **Merkle Trees** | GPU | Poseidon2, Keccak-256, Blake3 backends |
| **FRI** | GPU | Fast Reed-Solomon IOP (fold-by-2/4/8, commit, query, verify) |
| **Circle FRI** | GPU | FRI adapted for circle domain over M31 |
| **Sumcheck** | GPU | Interactive sumcheck (fused round+reduce, sparse O(nnz), univariate) |
| **KZG** | GPU | Polynomial commitment (commit + open + batch) |
| **IPA** | GPU/CPU | Inner product argument (Bulletproofs-style, GPU batch fold) |
| **Verkle Trees** | CPU | Width-N tree with Pedersen commitments + IPA opening proofs |
| **LogUp / Lasso / cq** | GPU | Lookup arguments (logarithmic derivatives, tensor decomposition, cached quotients) |
| **Range Proofs** | GPU | [0, R) via LogUp with limb decomposition |
| **ECDSA** | CPU | secp256k1 batch verification (probabilistic + individual) |
| **Circle STARK** | GPU | Full STARK prover/verifier over M31 circle domain |
| **Plonk** | GPU | Preprocessed polynomial IOP with KZG commitments |
| **Groth16** | GPU | zk-SNARK with BN254 pairings (R1CS, trusted setup, prove, verify) |
| **GKR** | GPU | Goldwasser-Kalai-Rothblum interactive proof for layered circuits |
| **Basefold / Brakedown / Zeromorph** | GPU | Polynomial commitment schemes (NTT-free, expander-based, multilinear-to-univariate) |
| **HyperNova** | GPU | CCS folding scheme for incremental verifiable computation |
| **BLS12-381** | CPU | Full tower (Fp/Fp2/Fp6/Fp12), G1/G2, Miller loop, pairings |
| **Pasta Curves** | GPU/CPU | Pallas/Vesta cycle (recursive proof composition ready) |
| **Binius** | GPU/CPU | Binary tower GF(2^8)->GF(2^128), additive FFT |
| **Tensor / WHIR** | GPU | Multilinear proof compression, RS proximity testing |
| **Transcript** | CPU | Fiat-Shamir duplex sponge (Poseidon2 + Keccak backends) |
| **Witness Gen** | GPU | GPU witness trace evaluation (instruction-stream architecture) |
| **Constraint IR** | GPU | Runtime constraint compilation (IR -> Metal source -> GPU pipeline) |
| **IPA Accumulation** | CPU | Halo-style accumulation (Pallas curve, batch decide) |
| **Kyber / Dilithium** | GPU | Post-quantum lattice crypto (GPU-accelerated NTT) |
| **HE NTT** | GPU | RNS-based NTT for homomorphic encryption (CKKS/BFV) |
| **Reed-Solomon** | GPU | Erasure coding for data availability sampling |
| **Serialization** | CPU | Proof serialization/deserialization (ProofWriter/ProofReader) |
| **Radix Sort** | GPU | 32-bit LSD radix sort (4-pass, 8-bit radix) |
| **Streaming Verify** | GPU/CPU | Task-queue streaming proof verification |
| **Incremental Merkle** | GPU | Append/update without full rebuild |
| **Batch Field Ops** | CPU | C-optimized vectorized field arithmetic |
| **STIR** | GPU | Shift-based proximity testing (FRI alternative) |
| **Marlin** | GPU | Preprocessed SNARK with algebraic holographic proof + KZG |
| **Spartan** | GPU | Transparent SNARK (no trusted setup) via multilinear extensions + sumcheck |
| **Jolt** | GPU | zkVM via Lasso structured lookups (10 RISC-like opcodes) |
| **SHA-256** | GPU | SHA-256 hash (batch + fused Merkle subtree) |
| **Ed25519** | GPU/CPU | Curve25519 field, twisted Edwards curve, EdDSA, GPU MSM |
| **BabyJubjub** | GPU/CPU | Twisted Edwards over BN254 Fr, Pedersen hash, EdDSA |
| **Grumpkin** | GPU | BN254 inner curve (y²=x³-17), GPU MSM with signed-digit |
| **Stark252** | GPU/CPU | StarkNet/Cairo native field (p=2^251+17·2^192+1), TWO_ADICITY=192, NTT |
| **Schnorr** | CPU | BIP 340 Bitcoin Taproot signatures (x-only pubkeys, tagged hashing) |
| **Poseidon2 BB** | GPU | Poseidon2 BabyBear width-16 (SP1/Plonky3 config, x^7 S-box) |

## Performance

All benchmarks on Apple M3 Pro (6P+6E cores). Run `swift run -c release zkbench all` to reproduce.

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

C Pippenger uses multi-threaded bucket accumulation with `__uint128_t` CIOS Montgomery (8 pthreads). At n<=2048, C Pippenger is automatically used instead of GPU. GPU wins at n>=2^14.

**Comparison to other implementations (BN254 MSM):**

| Points | zkMetal (M3 Pro)&#185; | ICICLE-Metal (M3 Pro)&#185; | ICICLE CPU (M3 Pro)&#185; | ICICLE-Metal (M3 Air)&#178; | MoPro v2 (M3 Air)&#178; | Arkworks CPU (M3 Air)&#178; | ICICLE CUDA&#179; |
|--------|---------|-------------|-----------|-------------|-----------|-----------|-----------|
| 2^16 | **27ms** | 1,083ms | 114ms | -- | 253ms | 69ms | ~9ms |
| 2^18 | **45ms** | 1,475ms | 556ms | 149ms | 678ms | 266ms | -- |
| 2^20 | **119ms** | 2,590ms | 2,349ms | 421ms | 1,702ms | 592ms | -- |

&#185; Measured locally. ICICLE-Metal v3.8.0 has ~600ms constant overhead per call (license server).
&#178; Reported by [MoPro blog](https://zkmopro.org/blog/metal-msm-v2/) -- different hardware/methodology.
&#179; [Ingonyama](https://github.com/ingonyama-zk/icicle) on RTX 3090 Ti (native 64-bit integer multiply).

### NTT

**Multi-field NTT comparison (GPU):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.47ms | 1.4ms | 0.14ms | 0.18ms |
| 2^18 | 1.6ms | 2.1ms | 0.19ms | 0.26ms |
| 2^20 | 6.1ms | 5.8ms | 0.81ms | 0.95ms |
| 2^22 | 26ms | 25ms | 4.2ms | 2.8ms |
| 2^24 | 116ms | 110ms | 3.0ms | 2.0ms |

BabyBear at 2^24: **8.5B elements/sec** (native 32-bit arithmetic). Goldilocks: **5.7B elements/sec**.

**BN254 Fr GPU vs CPU:**

| Size | Vanilla CPU | Opt C | C vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|------|-------------|-------|--------------|-------------|----------------|
| 2^14 | 79ms | 2.6ms | **30x** | 0.45ms | **176x** |
| 2^16 | 369ms | 12ms | **30x** | 0.76ms | **483x** |
| 2^18 | 1.6s | 55ms | **30x** | 2.2ms | **749x** |
| 2^20 | 7.3s | 246ms | **30x** | 6.1ms | **1198x** |

**Comparison to ICICLE-Metal v3.8 NTT (measured locally, M3 Pro):**

| Size | zkMetal BN254&#185; | ICICLE BN254&#185; | zkMetal BabyBear&#185; | ICICLE BabyBear&#185; |
|------|------------|-------------|----------------|----------------|
| 2^16 | **0.76ms** | 89ms | **0.18ms** | 86ms |
| 2^18 | **1.6ms** | 108ms | **0.26ms** | 92ms |
| 2^20 | **6.1ms** | 194ms | **0.95ms** | 108ms |
| 2^22 | **26ms** | 915ms | **2.8ms** | 181ms |
| 2^24 | **116ms** | 3,892ms | **2.0ms** | 709ms |

&#185; ICICLE-Metal has ~85ms per-call overhead. zkMetal is **30-90x faster** on BN254 and **90-500x faster** on BabyBear.

### Hashing

| Primitive | Batch Size | Vanilla CPU | Parallel CPU (12 cores) | GPU (Metal) | GPU vs Vanilla |
|-----------|-----------|-------------|------------------------|-------------|----------------|
| Poseidon2 | 2^12 | 523ms | 71ms | 2.3ms | **227x** |
| Poseidon2 | 2^14 | 2.0s | 278ms | 2.3ms | **871x** |
| Poseidon2 | 2^16 | 8.0s | 1.1s | 8.1ms | **993x** |
| Keccak-256 | 2^14 | 100ms | 23ms | 0.20ms | **500x** |
| Keccak-256 | 2^16 | 387ms | 89ms | 0.45ms | **860x** |
| Keccak-256 | 2^18 | 1.6s | 360ms | 1.4ms | **1143x** |

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

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.22ms | 8.9ms | **41x** |
| 2^16 | 0.35ms | 35ms | **99x** |
| 2^18 | 0.92ms | 137ms | **149x** |
| 2^20 | 1.96ms | 542ms | **276x** |
| 2^22 | 7.52ms | 2.2s | **295x** |

Full fold-to-constant: 2^20 in 3.0ms (20 rounds, fused 4-round kernels).

**FRI commit phase (fold + Merkle, full protocol):**

| Size | Fold-by-2 | Fold-by-4 | Fold-by-8 | 8/2 speedup |
|------|-----------|-----------|-----------|-------------|
| 2^15 | 66ms | 36ms | 19ms | **3.4x** |
| 2^16 | 78ms | 36ms | 31ms | **2.5x** |
| 2^18 | 135ms | 57ms | 32ms | **4.2x** |
| 2^20 | 323ms | 126ms | 99ms | **3.3x** |

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 0.89ms | 21ms | **24x** |
| 2^16 | 0.85ms | 83ms | **98x** |
| 2^18 | 2.2ms | 337ms | **153x** |
| 2^20 | 7.3ms | 1.3s | **178x** |
| 2^22 | 16ms | 5.2s | **325x** |

**Sparse sumcheck** at 1% density: 8-9x faster than dense. At 10%: ~2x faster.

### Polynomial Ops (BN254 Fr)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Multiply (NTT) | deg 2^10 | 57ms | 1.7ms | **34x** |
| Multiply (NTT) | deg 2^12 | 218ms | 2.0ms | **109x** |
| Multiply (NTT) | deg 2^14 | 1.1s | 3.3ms | **328x** |
| Multiply (NTT) | deg 2^16 | 2.4s | 7.7ms | **319x** |
| Multi-eval (Horner) | deg 2^10, 1024 pts | -- | 1.7ms | -- |
| Multi-eval (Horner) | deg 2^14, 16384 pts | -- | 114ms | -- |

### KZG Commitments (BN254 G1)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Commit | deg 2^8 | 293ms | 0.4ms | **652x** |
| Commit | deg 2^10 | 2.2s | 4.6ms | **490x** |
| Open (eval + witness) | deg 2^8 | 859ms | 3.9ms | **223x** |
| Open (eval + witness) | deg 2^10 | 2.1s | 4.6ms | **459x** |

### Batch KZG (BN254 G1)

| N Polys | Deg | N Individual Opens | 1 Batch Open | Speedup |
|---------|-----|-------------------|--------------|---------|
| 4 | 256 | 14.4ms | 9.8ms | **1.5x** |
| 8 | 256 | 24.5ms | 13.0ms | **1.9x** |
| 16 | 256 | 47.3ms | 21.5ms | **2.2x** |
| 32 | 256 | 75.2ms | 34.5ms | **2.2x** |

### Basefold PCS (BN254 Fr)

| Size | Commit | Open | Verify | Total |
|------|--------|------|--------|-------|
| 2^10 | 7.4ms | 34ms | 0.38ms | 41ms |
| 2^14 | 10ms | 67ms | 0.52ms | 78ms |
| 2^18 | 45ms | 144ms | 0.65ms | 190ms |

NTT-free multilinear polynomial commitment via recursive sumcheck-based folding.

### Circle STARK (Mersenne31)

| Trace Size | Prove | Verify | Proof Size |
|-----------|-------|--------|------------|
| 2^8 | 109ms | 15ms | 39 KB |
| 2^10 | 24ms | 14ms | 53 KB |
| 2^12 | 22ms | 16ms | 69 KB |
| 2^14 | 56ms | 28ms | 87 KB |

Full GPU pipeline: Circle NTT for LDE, GPU constraint evaluation, GPU Keccak Merkle trees, CPU FRI fold. Profile at 2^14: LDE 10ms, commit 12ms, constraint eval 2ms, FRI 27ms.

### Plonk (BN254, KZG)

| Gates | Setup | Prove | Verify |
|-------|-------|-------|--------|
| 16 | 18ms | 26ms | 3ms |
| 64 | 32ms | 44ms | 3ms |
| 256 | 17ms | 56ms | 3ms |
| 1024 | 48ms | 157ms | 3ms |

Previous version (naive O(n^2) poly mul): 7365ms at n=1024 -- GPU NTT gives **43x** improvement.

### Groth16 (BN254)

| Constraints | Setup | Prove | Verify |
|-------------|-------|-------|--------|
| 8 | 121ms | 68ms | 73ms |
| 64 | 625ms | 317ms | 76ms |
| 256 | 2.4s | 1.5s | 104ms |

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

Sparse wiring predicate evaluation: previous dense implementation 3728ms at 2^6 d=8 -- **190x** improvement.

### GPU Radix Sort

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^16 | 0.7ms | 2.9ms | **4x** |
| 2^18 | 1.3ms | 13ms | **10x** |
| 2^20 | 2.1ms | 59ms | **28x** |
| 2^22 | 6.4ms | 278ms | **43x** |

---

### Other Curve MSM

| Points | BN254 GPU | BLS12-377 GPU | secp256k1 GPU | secp256k1 C Pip | Pallas GPU | Vesta GPU |
|--------|-----------|---------------|---------------|-----------------|------------|-----------|
| 2^8 | 1.1ms | 9ms | 1.3ms | 1.4ms | 5.3ms | 4.9ms |
| 2^10 | 3.0ms | 35ms | 4.3ms | 4.3ms | 12ms | 10ms |
| 2^12 | -- | -- | -- | -- | 17ms | 17ms |
| 2^14 | 22ms | 36ms | 22ms | 31ms | 20ms | 20ms |
| 2^16 | 27ms | 176ms | 38ms | 92ms | 39ms | 39ms |
| 2^18 | 45ms | 205ms | 78ms | 339ms | 66ms | 65ms |

### CPU Optimizations

| Primitive | Size | Vanilla | Optimized | Speedup | Notes |
|-----------|------|---------|-----------|---------|-------|
| BN254 NTT (C CIOS) | 2^20 | 7.5s | 247ms | **30x** | `__uint128_t` unrolled Montgomery |
| BabyBear NTT (NEON) | 2^22 | 202ms | 37ms | **5.4x** | 4-wide SIMD Montgomery |
| Goldilocks NTT (C) | 2^20 | 53ms | 25ms | **2.1x** | `__uint128_t` pipelining |
| Keccak NEON | 2^16 | 38ms | 1.5ms (GPU) | **25x** | NEON 0.63 us/hash, GPU overtakes at 2^12 |
| Blake3 NEON | 2^18 | 17ms | 1.4ms (GPU) | **12x** | NEON 0.10 us/hash, GPU overtakes at 2^14 |
| BN254 batch add (C) | 100K | 16ms | 264 us | **60x** | 2.6 ns/op vectorized |
| BN254 batch mul (C) | 100K | -- | 1.3ms | -- | 13.4 ns/op CIOS |
| ECDSA batch 64 (CPU) | 64 sigs | -- | 8.0ms | **12x** | 0.13ms/sig, C CIOS scalar mul |

### Supporting Primitives

| Primitive | Metric | Value |
|-----------|--------|-------|
| Transcript (Keccak) | 1K absorb+squeeze | 0.89ms (2.2M ops/s) |
| Transcript (Poseidon2) | 1K absorb+squeeze | 176ms (11K ops/s) |
| Serialization (Base64) | 10KB x 1K encode | 3.3ms |
| Serialization (Base64) | 10KB x 1K decode | 3.6ms |
| KZG proof size | -- | 138 B |
| IPA proof size (8 rounds) | -- | 1586 B |
| FRI commitment (2^14) | -- | 1025 KB |
| Blake3 batch GPU | 2^20 | 0.003 us/hash (**200x** vs CPU) |

### Advanced Protocols

| Primitive | Key Benchmark | Notes |
|-----------|---------------|-------|
| HyperNova fold | 3.7ms/fold (10-1000 steps) | CCS folding, MSM+cross-term dominated |
| Basefold open 2^18 | 144ms | NTT-free multilinear PCS |
| Brakedown PCS | -- | Crashes on some hardware (signal 139) |
| Zeromorph PCS | -- | Crashes on some hardware (signal 139) |
| IPA prove n=256 | 12.8ms | Log(n) halving rounds, GPU batch fold |
| Verkle Trees (CPU) | 12ms build, 23ms proof, 5ms verify | Pedersen+IPA, C CIOS **40x** improvement |
| IPA Accumulation (Pallas) | 7.3ms accumulate (n=4) | Halo-style, batch decide 2.7x |
| Tensor compress 2^18 | 229ms compress, 39ms verify | **460.7x** compression ratio |
| WHIR 2^14 | 53ms prove, 16ms verify | 28.2 KB proof size |
| Lasso 2^18 | 59ms prove, 1.6s verify | C-accelerated decompose, fused GPU CB, **8.2x** from 481ms |
| LogUp 2^12 | 15ms prove, 16ms verify | Optimal for small-medium tables |
| cq | Correctness passes | Crashes at larger benchmark sizes |
| Binius FFT 2^16 | 21ms (CPU) | Binary tower GF(2^32) GPU batch: 0.67ms mul at 2^18 |
| BLS12-381 | Fp mul 339ns, G1 add 7.2us, pairing 27ms | Full tower Fp->Fp12 |
| BN254 GPU Pairing (n=16) | 51ms (vs 239ms CPU = **4.7x**) | Projective Miller loop, batched final exp |
| Schnorr BIP 340 | Sign 0.30ms, Batch verify 0.20ms/sig | x-only pubkeys, SHA-256 tagged hashing |
| Stark252 NTT 2^20 | 238M elem/s (GPU) | StarkNet/Cairo native field |
| Poseidon2 BabyBear (width-16) | 104M hash/s (GPU) | SP1/Plonky3 exact config |
| Pasta curves | Pallas/Vesta cycle | Field+curve ops, recursive composition ready |
| Data-Parallel GKR | Experimental | Multi-instance correctness tests failing |
| Incremental Merkle batch 256 | 13ms (vs 124ms full = **9.2x**) | Known regression on large sequential builds |

### Application Primitives

| Primitive | Key Benchmark | Notes |
|-----------|---------------|-------|
| Kyber-768 KEM | KeyGen 0.07ms, Encap 0.08ms, Decap 0.02ms | GPU NTT: 5.6M NTTs/s at batch 10K |
| Dilithium2 signatures | KeyGen 0.07ms, Sign 0.07ms, Verify 0.04ms | GPU NTT: 1.6M NTTs/s at batch 10K |
| HE NTT (RNS) | L CRT limbs in parallel | BFV keygen/encrypt/decrypt/add/mul |
| Reed-Solomon | NTT-based erasure coding | BabyBear + GF(2^16), encode+decode verified |
| Witness Gen BN254 2^18 | 3.0ms (877M cells/s) | GPU instruction-stream, **117x** vs CPU |
| Witness Gen M31 2^22 | 5.4ms (1.5B cells/s) | Circle STARK Fibonacci AIR |
| Constraint IR 2^16 | 5.3ms (248M constraints/s) | Runtime IR -> Metal -> GPU, **140x** at 2^14 |
| Streaming verify 2^16 | 11ms (task-queue) | **1.8x** vs sequential |
| GPU EC on-curve 100K | 1.8ms | **82x** vs CPU |
| Circle NTT 2^20 | 4.0ms (262M elem/s) | Circle-group over M31, 32-bit arithmetic |
| Poseidon2 M31 Merkle 2^18 | 5.6ms | 100M hash/s at 2^20 |

---

### Theoretical Performance Analysis

How close each primitive is to the hardware floor on M3 Pro (~3.6 TFLOPS GPU compute, ~150 GB/s memory bandwidth, ~0.5ms minimum kernel dispatch overhead). Sorted by headroom (most room for improvement first).

Methodology: Compute-bound = total_ops / 3.6T flops (BN254 mul = ~64 32-bit muls). Memory-bound = total_bytes / 150 GB/s. Dispatch-bound = N_dispatches x 0.5ms. Compound protocols = sum of component floors.

| Rank | Primitive | Current | Theoretical Floor | Bottleneck | Headroom |
|------|-----------|---------|-------------------|------------|----------|
| 1 | Groth16 prove 256 | 1.5s | ~60ms | MSM dominated (3 large MSMs + NTT) | ~25x |
| 2 | Lasso prove 2^18 | 59ms | ~30ms | C-accelerated decompose + fused GPU CB | ~2x |
| 3 | GKR 2^10 d=4 | 241ms | ~17ms | Sumcheck dominated (10 rounds x 2^10 vars) | ~14x |
| 4 | Plonk prove 1024 | 86ms | ~15ms | NTT + MSM (batch inversion 2.1x from 179ms) | ~6x |
| 5 | NTT BN254 2^22 | 26ms | ~2.9ms | Compute + strided BW (256-bit: 64 muls/elem) | ~9x |
| 6 | MSM BN254 2^18 | 45ms | ~5ms | Random-access BW (scatter bucket accumulation) | ~9x |
| 7 | KZG commit 2^10 | 4.6ms | ~0.5ms | MSM dominated (small N, dispatch overhead) | ~9x |
| 8 | Sumcheck 2^20 | 7.3ms | ~0.85ms | Bandwidth (2^20 x 32B per round) | ~9x |
| 9 | ECDSA batch 64 (CPU) | 8ms | ~1ms | C CIOS scalar mul (64 x ~300 doublings) | ~8x |
| 10 | Basefold open 2^18 | 144ms | ~20ms | Iterative fold+commit (18 rounds x Merkle) | ~7x |
| 11 | FRI Fold 2^20 | 1.96ms | ~0.3ms | Bandwidth (2^20 x 32B read+write) | ~7x |
| 12 | BLS12-377 MSM 2^18 | 218ms | ~35ms | Wider limbs (253-bit), less optimized window sizes | ~6x |
| 13 | Keccak Merkle 2^20 | 13ms | ~2.2ms | Compute (24 rounds x 64-bit) + 20 levels | ~6x |
| 14 | Blake3 Batch 2^20 | 3.5ms | ~0.6ms | Bandwidth (2^20 x 64B) | ~6x |
| 15 | Circle STARK prove 2^14 | 56ms | ~10ms | Multi-phase pipeline (LDE+commit+FRI, 5+ dispatches) | ~6x |
| 16 | HyperNova per-fold | 3.7ms | ~0.7ms | MSM dominated (commitment + cross-term) | ~5x |
| 17 | secp256k1 MSM 2^18 | 77ms | ~15ms | No GLV, wider scatter than BN254 | ~5x |
| 18 | Poseidon2 batch 2^16 | 8.1ms | ~1.8ms | Compute (390 ops/elem, 22 sequential rounds limit parallelism) | ~4.5x |
| 19 | Radix Sort 2^20 | 2.1ms | ~1ms | Vectorized histogram + flat clearing | ~2x |
| 20 | Binius FFT 2^16 (CPU) | 21ms | ~5ms | CPU only; XOR-add is free but table mul is serial | ~4x |
| 21 | Constraint IR 2^16 | 5.3ms | ~1.5ms | Compute (20 constraints x 65K rows, pipeline compile overhead) | ~3.5x |
| 22 | Witness Gen BN254 2^18 | 3.0ms | ~0.9ms | Memory bandwidth (10 cols x 262K x 32B = 84MB) | ~3.3x |
| 23 | Keccak Batch 2^18 | 1.4ms | ~0.5ms | Compute (24 rounds Keccak-f per hash) | ~3x |
| 24 | P2 Merkle 2^18 | 46ms | ~16ms | Compute (262K hashes, level-by-level at n>65K) | ~2.9x |
| 25 | P2 Merkle 2^16 | 22ms | ~8ms | Compute (65K hashes, fused subtree 1.7x over level-by-level) | ~2.8x |
| 26 | Streaming verify 2^16 | 11ms | ~4ms | Task-queue overhead + Merkle verification | ~2.8x |
| 27 | Verkle proof 256 (CPU) | 23ms | ~10ms | IPA dominated (C scalar mul) | ~2.3x |
| 28 | Incremental Merkle batch 256 | 13ms | ~6ms | Path updates (log(N) hashes per leaf) | ~2.2x |
| 29 | NTT Goldilocks 2^24 | 3.0ms | ~1.8ms | Compute ~= BW (64-bit) | ~1.7x |
| 30 | Lattice Kyber NTT 10K | 5.6M NTTs/s | ~8M NTTs/s | Compute (256-pt NTT, 32-bit) | ~1.4x |
| 31 | Circle NTT 2^20 | 4.0ms | ~3ms | Compute ~= BW (32-bit elements, single-word) | ~1.3x |
| 32 | IPA prove n=256 | 13ms | ~10ms | C scalar mul + GPU batch fold | ~1.3x |
| 33 | NTT BabyBear 2^24 | 2.0ms | ~1.7ms | Bandwidth (2^24 x 4B) | ~1.2x |

BabyBear/Goldilocks NTT and IPA are near-optimal (within 1-2x of hardware limits). Biggest opportunities: Groth16/Plonk/GKR (MSM/NTT bottleneck) and Lasso (algorithmic complexity). P2 Merkle is compute-bound (~2.8x headroom) — fused subtree kernel with shared memory is 1.7x faster than level-by-level despite 80% thread waste.

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
- **Stark252** (p = 2^251 + 17·2^192 + 1) -- StarkNet/Cairo native field, TWO_ADICITY=192
- **Ed25519 Fp** (p = 2^255 - 19) -- Curve25519 base field, 4x64-bit CIOS
- **Ed25519 Fq** -- Ed25519 scalar field for EdDSA
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
  zkMetal/         # Swift engine layer
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

// KZG Commitments
let kzg = try KZGEngine(srs: srs)
let commitment = try kzg.commit(polynomial)
let proof = try kzg.open(polynomial, at: z)

// Circle STARK
let circleNTT = try CircleNTTEngine()
let transformed = try circleNTT.ntt(m31Values)

// Verkle tree (CPU)
let verkle = VerkleEngine(width: 16, ipa: ipa)
let tree = try verkle.buildTree(leaves: values)

// ECDSA batch verification (CPU)
let ecdsa = ECDSAEngine()
let valid = try ecdsa.batchVerifyProbabilistic(signatures: sigs, messages: msgs, publicKeys: keys)
```

### Benchmarks

```bash
swift run -c release zkbench all          # Everything
swift run -c release zkbench test         # Correctness tests
swift run -c release zkbench cpu          # CPU vs GPU comparison
swift run -c release zkbench all --no-cpu # GPU-only (skip slow CPU baselines)
swift run -c release zkbench calibrate    # Re-calibrate GPU parameters

# Core:       msm, ntt, p2, keccak, blake3, merkle, sort, poly, batch-field
# Proofs:     plonk, groth16, gkr, dp, circle-stark, fold, accum
# PCS:        kzg, kzg-batch, basefold, zeromorph, brakedown, ipa, whir, tensor
# Polynomial: fri, sumcheck, sparse, usc, circle, circle-fri
# Lookups:    lookup, lasso, cq
# Curves:     bls377, bls377msm, secpmsm, pastamsm, bls381, pasta, binius
# Apps:       lattice, erasure, witness, constraint, fused, stream-verify, imerkle
# CPU:        ecdsa, verkle, gl-neon, keccak-neon, blake3-neon, asm, p2m31
# Meta:       versions, serialize
```

### MSM CLI

```bash
swift run -c release zkmsm --bench 65536
echo '{"points": [["0x1","0x2"]], "scalars": ["0x2a"]}' | swift run -c release zkmsm
```

## Auto-Tuning

zkMetal automatically calibrates GPU parameters (threadgroup sizes, FFT thresholds, MSM window sizes) on first use. Results are cached to `~/.zkmetal/tuning.json` and reused across runs. Calibration re-triggers automatically when the GPU changes.

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
- **C CIOS field arithmetic**: Hot-path 256-bit Montgomery multiplication uses C `__uint128_t` compiled with `-O3`, which is 29-30x faster than Swift for BN254 Fr carry chains.
- **Small-input fast path**: MSM automatically routes to multi-threaded C Pippenger for small inputs (BN254 n<=2048, secp256k1 n<=1024) to avoid GPU dispatch overhead.

## Correctness & Testing

Run `swift run -c release zkbench test`. All GPU kernels verified against CPU reference implementations.

| Category | Primitives | Verification |
|----------|------------|-------------|
| Field arithmetic | BN254, BLS12-377/381, secp256k1, Goldilocks, BabyBear, M31, Pallas/Vesta, Binary Tower | Unit tests + cross-checks (arithmetic properties, inverses, distributivity) |
| MSM | BN254, BLS12-377, secp256k1, Pallas/Vesta, Ed25519, Grumpkin | GPU vs CPU cross-check, on-curve, determinism |
| NTT | BN254, BLS12-377, Goldilocks, BabyBear, Stark252, Circle NTT | Round-trip + CPU cross-check (all fields, sizes 2^2 through 2^22) |
| Hashing | Poseidon2 (BN254+M31+BabyBear), Keccak-256, Blake3, SHA-256 | Known-answer tests (NIST, HorizenLabs, BLAKE3 spec) + GPU vs CPU batch |
| Merkle trees | Poseidon2, Keccak, Blake3 backends | GPU vs CPU root comparison + parallel structure validation |
| Polynomial protocols | FRI (fold-by-2/4/8), Sumcheck (dense+sparse), KZG, Batch KZG | S(0)+S(1)=sum, round-poly match, full protocol verify, tamper rejection |
| PCS | Basefold, Tensor, WHIR | Fold correctness, compress+verify, proof verify+rejection |
| Proof systems | Circle STARK, Plonk, Groth16, GKR | Prove+verify, tampered proof rejection, bilinearity |
| Lookups | LogUp, Lasso, cq | Simple/repeated/multiplicities, tamper rejection |
| CPU optimized | C BN254/Goldilocks NTT, NEON BabyBear/Keccak/Blake3 | Cross-checked against vanilla CPU + round-trip |
| Signatures | EdDSA (Ed25519+BabyJubjub), ECDSA (secp256k1), Schnorr (BIP 340) | RFC 8032, batch verification, tagged hashing |
| Protocols | IPA, Verkle (CPU), HyperNova, IPA Accumulation | Prove+verify, wrong-value rejection, batch verification |
| Infrastructure | Transcript, Serialization, Witness Gen, Constraint IR, Radix Sort | Determinism, roundtrip, domain/fork separation, tamper detection |
| Applications | Kyber KEM, Dilithium signatures, Reed-Solomon, Streaming Verify | Shared secret agreement, signature verification, encode+decode |

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization.
