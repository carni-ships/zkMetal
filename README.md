# zkMetal

GPU-accelerated zero-knowledge proof library for Apple Silicon. Metal compute shaders + C/NEON field arithmetic + Swift orchestration.

**~211 primitives** across 18 fields and 10 elliptic curves. 573 source files, 107 Metal shaders, 33 C/NEON files, 239 test files. 50+ engines converted to optimized C batch kernels with prefetch, branchless arithmetic, and auto-parallel dispatch (`dispatch_apply` for n >= 4096). All BN254 Fr serial field loops eliminated — `frBatchInverse` (9 call sites), Hadamard vector ops, axpy/inner product, in-place fold, Montgomery's trick patterns all batch-converted. BabyBear/Goldilocks STARK provers now use C kernels for FRI fold, vanishing polynomial, and batch inverse.

- **Core:** MSM (Pippenger+GLV), NTT (four-step FFT), Poseidon2/Keccak/Blake3/SHA-256, Merkle trees
- **Proof systems:** Plonk, HyperPlonk, Fflonk, Groth16, STARK (Circle/BabyBear/Goldilocks/Stark252), Spartan, Marlin, GKR
- **Commitments:** KZG (+ degree bounds, batch verify, multi-open), IPA, Basefold, Brakedown, Zeromorph, Verkle (+ multiproofs), Pedersen
- **Folding/IVC:** HyperNova, Protogalaxy (+ decider), Nova/SuperNova (+ decider circuit, relaxation, verifier)
- **Lookup arguments:** LogUp, Lasso, CQ (cached quotients), Plookup, grand product
- **zkVM:** Jolt (RV32I instruction decomposition, Lasso lookups, RISC-V decoder), Cairo (memory argument, trace)
- **STARK pipeline:** AIR constraint compiler, R1CS-to-AIR, Plonky3 AIR, FRI (commit/query/grind), STIR, trace LDE, deep composition
- **Tooling:** Circom R1CS parser, AIR constraint DSL, Solidity verifier gen, proof serialization, proof aggregation

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

### Core Arithmetic

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **MSM** | GPU/CPU | Multi-scalar multiplication (Pippenger + signed-digit + GLV) — BN254, BLS12-377, secp256k1, Pallas, Vesta, Ed25519, Grumpkin, BLS12-381, BN254 G2 |
| **NTT** | GPU/CPU | Number theoretic transform (four-step FFT, fused bitrev+butterfly, parallel CPU butterfly for n >= 4096) — BN254, BLS12-377, Goldilocks, BabyBear, Stark252, Circle M31 |
| **Batch Field Ops** | GPU/CPU | Vectorized add/mul/sub/inverse across all fields, C CIOS Montgomery, auto-parallel dispatch (n >= 4096) |
| **Radix Sort** | GPU | 32-bit LSD radix sort (4-pass, 8-bit radix) |
| **Prefix Scan** | GPU | GPU prefix product for grand product arguments |
| **Parallel Reduction** | GPU | SIMD shuffle + shared memory for field elements |

### Hashing & Merkle Trees

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Poseidon2** | GPU | Algebraic hash — BN254 (t=3), M31 (t=16), BabyBear (width-16, SP1 config). Duplex sponge mode |
| **Keccak-256** | GPU/CPU | SHA-3 hash with NEON acceleration, fused Merkle subtree |
| **Blake3** | GPU/CPU | BLAKE3 hash with NEON acceleration, batch + Merkle trees |
| **SHA-256** | GPU | Batch hash + fused Merkle subtree |
| **Merkle Trees** | GPU | Poseidon2/Keccak/Blake3 backends + incremental append/update |
| **Pedersen Hash** | CPU | Multi-curve Pedersen commitment + IPA vector commitments |

### Polynomial & IOP Protocols

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **FRI** | GPU | Fast Reed-Solomon IOP — fold-by-2/4/8, commit phase, query phase, grinding (GPU nonce search) |
| **STIR** | GPU | Shift-based proximity testing (FRI alternative) |
| **Sumcheck** | GPU | Interactive sumcheck — dense, sparse O(nnz), univariate, multilinear |
| **Polynomial Ops** | GPU | NTT multiply, Horner multi-eval, division, interpolation, vanishing polynomial |
| **Coset LDE** | GPU | Fused zero-pad + coset-shift for BabyBear/Goldilocks/BN254 |
| **Reed-Solomon** | GPU | Erasure coding for data availability sampling |

### Polynomial Commitment Schemes

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **KZG** | GPU | Commit, open, batch open, multi-open, degree bound proofs, batch verify, trusted setup |
| **IPA** | GPU/CPU | Bulletproofs-style inner product argument, BGMW precomputed tables |
| **Basefold** | GPU | NTT-free multilinear PCS via recursive sumcheck folding |
| **Brakedown** | GPU | Expander-based multilinear PCS |
| **Zeromorph** | GPU | Multilinear-to-univariate PCS reduction |
| **Verkle Trees** | CPU | Width-N tree with Pedersen + IPA, multiproof generation/verification |
| **Tensor / WHIR** | GPU | Multilinear proof compression, RS proximity testing |

### Proof Systems

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Plonk** | GPU | Preprocessed polynomial IOP with KZG — wire assignment, permutation, quotient, linearization, custom gates |
| **HyperPlonk** | GPU | Multilinear Plonk variant with sumcheck-based IOP, zero-check, LogUp lookups |
| **Fflonk** | GPU | Combined polynomial commitment Plonk variant |
| **Groth16** | GPU | zk-SNARK with BN254 pairings — R1CS, trusted setup, prove, verify, aggregate, recursive verifier, Solidity export |
| **STARK** | GPU | Full pipeline — Circle/BabyBear/Goldilocks/Stark252, Plonky3 AIR, trace LDE, deep composition, query phase |
| **GKR** | GPU | Goldwasser-Kalai-Rothblum interactive proof for layered circuits, data-parallel mode |
| **Spartan** | GPU | Transparent SNARK via multilinear extensions + sumcheck, linearization |
| **Marlin** | GPU | Preprocessed SNARK with algebraic holographic proof + KZG poly IOP |

### Lookup Arguments

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **LogUp** | GPU | Logarithmic derivative lookup with GKR-enhanced grand product |
| **Lasso** | GPU | Tensor decomposition structured lookups |
| **CQ** | GPU | Cached quotients lookup with KZG commitments |
| **Range Proofs** | GPU | Bulletproofs-style range proof protocol |
| **Unified Lookup** | GPU | LogUp/Lasso/CQ with auto-strategy selection |

### Folding & IVC

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Nova/SuperNova** | GPU/CPU | IVC with cross-term folding — relaxation, verifier, decider circuit |
| **HyperNova** | GPU | CCS folding scheme for incremental verifiable computation |
| **Protogalaxy** | GPU/CPU | Plonk-native folding with O(k log k) per step, decider |
| **Proof Aggregation** | GPU | SnarkPack-style multi-proof batch aggregation |

### Curves & Signatures

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **BN254** | GPU/CPU | Full pairing engine (Fp/Fp2/Fp6/Fp12 tower, Miller loop, final exp) |
| **BLS12-381** | CPU | Full tower (Fp->Fp12 in C), G1/G2, pairings (0.9ms), hash-to-curve G2 (RFC 9380), BLS signatures |
| **BLS12-377** | GPU/CPU | Scalar + base field, G1 MSM, NTT |
| **Pasta (Pallas/Vesta)** | GPU/CPU | Curve cycle for recursive proof composition |
| **secp256k1** | GPU/CPU | ECDSA batch verification (Shamir's trick) |
| **Ed25519** | GPU/CPU | Curve25519 field, twisted Edwards, EdDSA (C Fq CIOS + Shamir's trick), GPU MSM |
| **BabyJubjub** | GPU/CPU | Twisted Edwards over BN254 Fr, Pedersen hash, EdDSA |
| **Grumpkin** | GPU | BN254 inner curve, GPU MSM |
| **Jubjub** | CPU | Twisted Edwards over BLS12-381 Fr (Zcash Sapling) |
| **Schnorr** | CPU | BIP 340 Bitcoin Taproot signatures |
| **Binius** | GPU/CPU | Binary tower GF(2^8)→GF(2^128), additive FFT, multilinear polynomial ops |

### zkVM

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Jolt** | GPU | Instruction decomposition, Lasso subtable lookups, consistency checking |
| **RISC-V Decoder** | CPU | All 40 RV32I opcodes + M extension, execution trace |
| **Cairo** | GPU/CPU | Memory argument (permutation + continuity), trace generation |
| **Witness Gen** | GPU | GPU instruction-stream witness trace evaluation |
| **Memory Checking** | CPU | GKR-based grand product for read/write consistency |

### Constraint Systems & Compilation

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **R1CS** | GPU | R1CS witness solver, constraint evaluation, R1CS-to-QAP, R1CS-to-AIR compiler |
| **CCS** | CPU | Customizable constraint system (unified R1CS/Plonk/AIR for folding) |
| **AIR** | GPU/CPU | Constraint DSL, constraint compiler, trace validation, composition |
| **Plonk Gates** | GPU | Custom gate engine (range, EC, Poseidon, boolean, XOR), Halo2-compatible |
| **Constraint Optimizer** | CPU | Dead elimination, constant fold, CSE, R1CS reduce |
| **Circom Parser** | CPU | Binary .r1cs/.wtns format parser for Groth16 proving |

### Infrastructure

| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Transcript** | CPU | Fiat-Shamir duplex sponge — Poseidon2 + Keccak backends, Merlin STROBE-128 |
| **Proof Serialization** | CPU | BN254/BLS12-381 point compression, snarkjs JSON, EIP-4844, binary/JSON universal format |
| **GPU Buffer Pool** | GPU | Power-of-2 size-class recycling with scoped contexts |
| **Shader Cache** | GPU | MTLBinaryArchive persistent cache + background precompilation |
| **Constraint IR** | GPU | Runtime IR → Metal source → GPU pipeline |
| **Data Availability** | GPU | EIP-4844 blob extension + cell proofs + Reed-Solomon |
| **Kyber / Dilithium** | GPU | Post-quantum lattice crypto (GPU-accelerated NTT) |
| **HE NTT** | GPU | RNS-based NTT for homomorphic encryption (CKKS/BFV) |

## Performance

All benchmarks on Apple M3 Pro (6P+6E cores). Run `swift run -c release zkbench all` to reproduce.

### MSM (BN254 G1)

| Points | Vanilla CPU | Swift Pippenger | C Pippenger | GPU (Metal) |
|--------|-------------|----------------|-------------|-------------|
| 2^8 | 450ms | 16ms | **1.3ms** | 0.9ms |
| 2^10 | 1.8s | 45ms | **2.9ms** | 2.6ms |
| 2^12 | 7.3s | 129ms | **8.1ms** | 14ms |
| 2^14 | 35s | 429ms | 29ms | **22ms** |
| 2^16 | -- | -- | 68ms | **31ms** |
| 2^18 | -- | -- | 240ms | **53ms** |
| 2^20 | -- | -- | 856ms | **137ms** |

C Pippenger uses multi-threaded bucket accumulation with `__uint128_t` CIOS Montgomery (8 pthreads). At n<=2048, C Pippenger is automatically used instead of GPU. GPU wins at n>=2^14.

**Comparison to other implementations (BN254 MSM):**

| Points | zkMetal (M3 Pro)&#185; | ICICLE-Metal (M3 Pro)&#185; | ICICLE CPU (M3 Pro)&#185; | ICICLE-Metal (M3 Air)&#178; | MoPro v2 (M3 Air)&#178; | Arkworks CPU (M3 Air)&#178; | ICICLE CUDA&#179; |
|--------|---------|-------------|-----------|-------------|-----------|-----------|-----------|
| 2^16 | **31ms** | 1,083ms | 114ms | -- | 253ms | 69ms | ~9ms |
| 2^18 | **53ms** | 1,475ms | 556ms | 149ms | 678ms | 266ms | -- |
| 2^20 | **137ms** | 2,590ms | 2,349ms | 421ms | 1,702ms | 592ms | -- |

&#185; Measured locally. ICICLE-Metal v3.8.0 has ~600ms constant overhead per call (license server).
&#178; Reported by [MoPro blog](https://zkmopro.org/blog/metal-msm-v2/) -- different hardware/methodology.
&#179; [Ingonyama](https://github.com/ingonyama-zk/icicle) on RTX 3090 Ti (native 64-bit integer multiply).

### NTT

**Multi-field NTT comparison (GPU):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.69ms | 1.4ms | 0.15ms | 0.26ms |
| 2^18 | 2.9ms | 2.1ms | 0.21ms | 0.54ms |
| 2^20 | 10.8ms | 5.8ms | 0.70ms | 0.79ms |
| 2^22 | 28ms | 25ms | 4.3ms | 3.0ms |
| 2^24 | 113ms | 110ms | 3.1ms | 2.3ms |

BabyBear at 2^24: **7.3B elements/sec** (native 32-bit arithmetic). Goldilocks: **5.4B elements/sec**.

**BN254 Fr GPU vs CPU:**

| Size | Vanilla CPU | Opt C | C vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|------|-------------|-------|--------------|-------------|----------------|
| 2^14 | 97ms | 2.6ms | **37x** | 0.57ms | **169x** |
| 2^16 | 507ms | 12ms | **42x** | 0.69ms | **738x** |
| 2^18 | 2.2s | 55ms | **40x** | 2.9ms | **746x** |
| 2^20 | 13.0s | 246ms | **53x** | 10.8ms | **1206x** |

**Comparison to ICICLE-Metal v3.8 NTT (measured locally, M3 Pro):**

| Size | zkMetal BN254&#185; | ICICLE BN254&#185; | zkMetal BabyBear&#185; | ICICLE BabyBear&#185; |
|------|------------|-------------|----------------|----------------|
| 2^16 | **0.69ms** | 89ms | **0.26ms** | 86ms |
| 2^18 | **2.9ms** | 108ms | **0.54ms** | 92ms |
| 2^20 | **10.8ms** | 194ms | **0.79ms** | 108ms |
| 2^22 | **28ms** | 915ms | **3.0ms** | 181ms |
| 2^24 | **113ms** | 3,892ms | **2.3ms** | 709ms |

&#185; ICICLE-Metal has ~85ms per-call overhead. zkMetal is **18-33x faster** on BN254 and **60-330x faster** on BabyBear.

### Hashing

| Primitive | Batch Size | Vanilla CPU | Optimized CPU | GPU (Metal) | GPU vs Opt CPU |
|-----------|-----------|-------------|--------------|-------------|----------------|
| Poseidon2 | 2^12 | 523ms | 19ms (C CIOS) | 2.3ms | **8x** |
| Poseidon2 | 2^14 | 2.0s | 75ms (C CIOS) | 2.3ms | **33x** |
| Poseidon2 | 2^16 | 8.0s | 302ms (C CIOS) | 8.5ms | **36x** |
| Keccak-256 | 2^14 | 100ms | 23ms (parallel) | 0.20ms | **500x** |
| Keccak-256 | 2^16 | 387ms | 89ms (parallel) | 0.45ms | **860x** |
| Keccak-256 | 2^18 | 1.6s | 360ms (parallel) | 1.4ms | **1143x** |
| SHA-256 | 2^14 | -- | 0.81us/hash (CPU) | 0.58ms | **28x** |
| SHA-256 | 2^16 | -- | -- | 2.0ms | **32M hash/s** |
| SHA-256 | 2^18 | -- | -- | 3.9ms | **67M hash/s** |
| SHA-256 | 2^20 | -- | -- | 8.7ms | **121M hash/s** |

### Merkle Trees

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 7.3ms | 6ms | **1x** |
| Poseidon2 | 2^12 | 8.7ms | 23ms | **3x** |
| Poseidon2 | 2^14 | 10ms | 91ms | **9x** |
| Poseidon2 | 2^16 | 21ms | 364ms | **17x** |
| Poseidon2 | 2^18 | 45ms | 1.4s | **32x** |
| Poseidon2 | 2^20 | 129ms | -- | -- |
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
| SHA-256 | 2^12 | 0.83ms | -- | -- |
| SHA-256 | 2^14 | 0.87ms | -- | -- |
| SHA-256 | 2^16 | 1.3ms | -- | -- |
| SHA-256 | 2^18 | 3.3ms | -- | -- |
| SHA-256 | 2^20 | 10ms | -- | -- |

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
| Commit | deg 2^8 | 261ms | 0.3ms | **813x** |
| Commit | deg 2^10 | 1.0s | 0.7ms | **1396x** |
| Open (eval + witness) | deg 2^8 | 381ms | 0.9ms | **446x** |
| Open (eval + witness) | deg 2^10 | 1.6s | 2.5ms | **669x** |

C CIOS Horner evaluation + fused eval/division, cached SRS affine points, CPU MSM for small sizes. Commit **6.6x** faster (4.6→0.7ms at 2^10).

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
| 2^14 | 21ms | 28ms | 87 KB |

Full GPU pipeline: Circle NTT for LDE, GPU constraint evaluation, GPU Keccak Merkle trees, CPU FRI fold. Optimized: batched CBs, trace caching, LDE 0.9ms (was 5ms). Profile at 2^14: LDE 0.9ms, commit 6ms, constraint eval 2ms, FRI 12ms.

### Plonk (BN254, KZG)

| Gates | Setup | Prove | Verify |
|-------|-------|-------|--------|
| 16 | 9ms | 5ms | 2ms |
| 64 | 10ms | 7ms | 2ms |
| 256 | 12ms | 14ms | 2ms |
| 1024 | 25ms | 49ms | 2ms |

C CIOS constraint evaluation, Keccak transcript, CPU NTT for small sizes, batched polynomial ops. Prove **3.2x** at n=1024 (157→49ms).

### Groth16 (BN254)

| Constraints | Setup | Prove | Verify |
|-------------|-------|-------|--------|
| 8 | 114ms | 12ms | 69ms |
| 64 | 603ms | 13ms | 79ms |
| 256 | 2.5s | 14ms | 73ms |

Verification now **VALID** with C-accelerated pairing (30x faster than Swift path). Cached affine points, CPU NTT for small sizes, parallel BG2, C-accelerated polynomial ops. Prove **107x** improvement at n=256 (1.5s→14ms).

### GKR (BN254 Fr, Layered Circuits)

| Circuit | Prove | Verify |
|---------|-------|--------|
| 2^4 width, d=4 | 0.15ms | 0.30ms |
| 2^5 width, d=4 | 0.25ms | 0.46ms |
| 2^6 width, d=4 | 0.44ms | 0.73ms |
| 2^6 width, d=8 | 0.99ms | 1.51ms |
| 2^8 width, d=4 | 1.78ms | 2.39ms |
| 2^8 width, d=8 | 3.53ms | 4.61ms |
| 2^10 width, d=4 | 7.68ms | 8.39ms |

C CIOS Montgomery acceleration: pre-computed wiring topology, cached buffers, eq polynomial, sumcheck rounds, MLE fold all in C. Previous Swift-only: 241ms at 2^10 d=4 -- **31x** improvement.

### GPU Radix Sort

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^16 | 0.7ms | 2.9ms | **4x** |
| 2^18 | 1.3ms | 13ms | **10x** |
| 2^20 | 2.1ms | 59ms | **28x** |
| 2^22 | 6.4ms | 278ms | **43x** |

---

### Other Curve MSM

| Points | BN254 GPU | BLS12-377 GPU | secp256k1 GPU | secp256k1 C Pip | Pallas GPU | Vesta GPU | Grumpkin GPU | Ed25519 GPU | BN254 G2 GPU |
|--------|-----------|---------------|---------------|-----------------|------------|-----------|--------------|-------------|--------------|
| 2^8 | 1.1ms | 9ms | 1.3ms | 1.4ms | 5.3ms | 4.9ms | 3.3ms | 0.8ms | 13ms |
| 2^10 | 3.0ms | 35ms | 4.3ms | 4.3ms | 12ms | 10ms | -- | 22ms | 38ms |
| 2^12 | 8.1ms | 14ms | 6.5ms | 10ms | 17ms | 17ms | 20ms | -- | 63ms |
| 2^14 | 22ms | 36ms | 12ms | 31ms | 20ms | 20ms | 258ms | -- | 812ms |
| 2^16 | 27ms | 176ms | 24ms | 92ms | 39ms | 39ms | 48ms | -- | -- |
| 2^18 | 45ms | 205ms | 113ms | 339ms | 66ms | 65ms | -- | -- | -- |

### CPU Optimizations

| Primitive | Size | Vanilla | Optimized | Speedup | Notes |
|-----------|------|---------|-----------|---------|-------|
| BN254 NTT (C CIOS) | 2^20 | 7.5s | 247ms | **30x** | `__uint128_t` unrolled Montgomery |
| BabyBear NTT (NEON) | 2^22 | 202ms | 37ms | **5.4x** | 4-wide SIMD Montgomery |
| Goldilocks NTT (C) | 2^20 | 53ms | 25ms | **2.1x** | `__uint128_t` pipelining |
| Keccak NEON | 2^16 | 38ms | 1.5ms (GPU) | **25x** | NEON 0.63 us/hash, GPU overtakes at 2^12 |
| Blake3 NEON | 2^18 | 17ms | 1.4ms (GPU) | **12x** | NEON 0.10 us/hash, GPU overtakes at 2^14 |
| BN254 Fr mul (C CIOS) | single | 2500ns | 16ns | **156x** | Zero-copy Swift↔C bridge, `__uint128_t` |
| BN254 Fr add (C) | single | ~50ns | 4.5ns | **11x** | Branchless modular add |
| BN254 batch add (C) | 100K | 16ms | 264us | **60x** | 2.6 ns/op vectorized |
| BN254 batch mul (C) | 100K | 250ms | 1.3ms | **192x** | 13.4 ns/op CIOS |
| BN254 batch MAC (C) | 100K | 270ms | 1.5ms | **180x** | Fused scalar*vec + accumulate |
| BN254 batch inverse (C) | 100K | 1.0s | 1.6ms | **625x** | Montgomery's trick, zero-safe variant (9 call sites) |
| BN254 batch axpy (C) | 100K | 270ms | 1.5ms | **180x** | result[i] += scalar * x[i], in-place |
| BN254 inner product (C) | 100K | 250ms | 1.3ms | **192x** | Dot product + vector_sum |
| BN254 fold_interleaved (C) | 2^18 | 1.3s | 5.2ms | **250x** | In-place: data[i] = data[2i] + c * (data[2i+1] - data[2i]) |
| BN254 Horner eval (C) | deg 2^16 | 163ms | 1.0ms | **163x** | Prefetch + branchless, replaces Swift `evaluatePolynomial` |
| BN254 synthetic div (C) | deg 2^16 | 163ms | 1.0ms | **163x** | Replaces Swift `syntheticDivide` in 5 PCS engines |
| BN254 fold_halves (C) | 2^18 | 1.3s | 5.2ms | **250x** | Fused fold for non-interleaved layout, auto-parallel |
| Parallel NTT/INTT (C) | 2^18 | 55ms | 12ms | **4.6x** | Butterfly stages dispatch across cores for n >= 4096 |
| Sumcheck reduce (C) | 2^20 | 3.3s | 18ms | **183x** | Single C call per round |
| IPA s-vector | 2^20 | O(n·logN) | O(n) | **20x** | Butterfly construction |
| BabyBear batch inverse (C) | 100K | 500ms | 0.8ms | **625x** | Standard-form Montgomery's trick |
| BabyBear FRI fold (C) | 2^18 | 650ms | 2.6ms | **250x** | Fused even/odd + beta combine |
| BabyBear vanishing poly (C) | 2^18 | 650ms | 2.6ms | **250x** | Chain multiply base * gen^i - 1 |
| Goldilocks batch inverse (C) | 100K | 800ms | 1.2ms | **667x** | Standard-form Montgomery's trick |
| Goldilocks FRI fold (C) | 2^18 | 1.0s | 4.2ms | **238x** | Fused even/odd + beta combine |
| Goldilocks vanishing poly (C) | 2^18 | 1.0s | 4.2ms | **238x** | Chain multiply base * gen^i - 1 |
| ECDSA batch 64 (CPU) | 64 sigs | -- | 1.7ms | **57x** | 0.03ms/sig, C CIOS Fr + batch prepare |

### Supporting Primitives

| Primitive | Metric | Value |
|-----------|--------|-------|
| Transcript (Keccak) | 1K absorb+squeeze | 0.89ms (2.2M ops/s) |
| Transcript (Poseidon2) | 1K absorb+squeeze | 9.9ms (202K ops/s) |
| Serialization (Base64) | 10KB x 1K encode | 3.3ms |
| Serialization (Base64) | 10KB x 1K decode | 3.6ms |
| KZG proof size | -- | 138 B |
| IPA proof size (8 rounds) | -- | 1586 B |
| FRI commitment (2^14) | -- | 1025 KB |
| Blake3 batch GPU | 2^20 | 0.003 us/hash (**200x** vs CPU) |

### Advanced Protocols

| Primitive | Key Benchmark | Notes |
|-----------|---------------|-------|
| HyperNova fold | 0.09ms/fold (1000 steps), N=4: 0.21ms, N=8: 0.46ms | Keccak256 transcript + C CIOS + pre-computed affine: **40x** (3.6ms→0.09ms) |
| Basefold open 2^18 | 110ms | C CIOS fold + zero-copy Merkle paths: **1.3x** faster |
| Brakedown PCS | -- | Crashes on some hardware (signal 139) |
| Zeromorph PCS | -- | Crashes on some hardware (signal 139) |
| IPA prove n=256 | 11.8ms | Log(n) halving rounds, C CIOS batch fold + Blake3 NEON + BGMW fixed-base commit |
| Verkle Trees (CPU) | 14ms build, 5ms proof, 2.4ms verify | C CIOS Pedersen+IPA: build **24x**, proof **134x**, verify **38x** |
| IPA Accumulation (Pallas) | 7.3ms accumulate (n=4) | Halo-style, batch decide 2.7x |
| Tensor compress 2^18 | 8.9ms compress, 2.1ms verify | **460.7x** compression ratio, Keccak transcript: **26x** compress, **19x** verify |
| WHIR 2^14 | 30ms prove, 4.8ms verify | C CIOS fold + CPU P2 Merkle: prove **1.9x**, verify **3.6x** |
| Marlin prove+verify | 2.7ms verify | Batch KZG openings (19 MSMs→2) + C CIOS: **25x** faster verify |
| Spartan prove 2^14 | 121ms prove, 8ms verify | C CIOS sumcheck + sparse matvec: **8.6x** (1051→122ms) |
| Lasso 2^18 | 30ms prove, 31ms verify | C-accelerated: prove **16x** (481→30ms), verify **52x** (1.6s→31ms) |
| LogUp 2^12 | 15ms prove, 16ms verify | Optimal for small-medium tables |
| cq 2^16 | 8ms prove, 2ms verify | O(N log N) independent of table size |
| Binius FFT 2^16 | 21ms (CPU) | Binary tower GF(2^32) GPU batch: 0.67ms mul at 2^18 |
| BLS12-381 | Sign 26ms, Verify 82ms, **Pairing 1.0ms** | Projective G2 Miller loop + sparse line mul + dedicated fp_sqr: **78×** (78→1.0ms) |
| BN254 GPU Pairing (n=16) | 34ms (vs 5.6ms C = **0.16x**) | Projective Miller loop, fused line-line mul, batched final exp |
| BN254 C Pairing (n=1) | **0.5ms** (15× vs Swift) | CIOS __uint128_t + Granger-Scott cyc_sqr + sparse line + projective G2 |
| Schnorr BIP 340 | Sign 0.12ms, Verify 0.11ms, Batch 0.03ms/sig (256) | x-only pubkeys, SHA-256 tagged hashing |
| Ed25519 EdDSA | Sign 0.06ms, Verify 0.08ms, MSM 256pt 0.8ms | C Fq CIOS + Shamir's trick, RFC 8032 test vectors |
| BabyJubjub EdDSA | Sign 0.11ms, Verify 0.14ms, Batch 10 1.9ms | Pedersen hash (0.01ms/op), twisted Edwards over BN254 Fr |
| Plookup | Prove 3.2ms (2^8), 70ms (2^12), 204ms (2^14) | Split-half grand product, domain padding, verify 0.3-16ms |
| Stark252 NTT 2^20 | 238M elem/s (GPU) | StarkNet/Cairo native field |
| Poseidon2 BabyBear (width-16) | 104M hash/s (GPU) | SP1/Plonky3 exact config |
| Pasta MSM 2^18 | Pallas 125ms, Vesta 128ms | C CIOS field+curve ops, Jacobian projective, windowed scalar mul |
| Data-Parallel GKR (N=16) | 2.3ms prove, 0.5ms verify | O(|C| + N log N), repeated sub-circuit prover with cached wiring MLEs |
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
| 1 | Groth16 prove 256 | 14ms | ~10ms | MSM dominated (cached affine + CPU NTT) | ~1.4x |
| 2 | Lasso prove 2^18 | 56ms | ~30ms | Near floor — C-accelerated + fused GPU | ~2x |
| 3 | GKR 2^10 d=4 | 7.7ms | ~5ms | C CIOS + pre-computed wiring topology (near floor) | ~1.5x |
| 4 | Plonk prove 1024 | 49ms | ~15ms | C CIOS + Keccak transcript + batched poly ops | ~3x |
| 5 | NTT BN254 2^22 | 26ms | ~2.9ms | Compute + strided BW (256-bit: 64 muls/elem) | ~9x |
| 6 | MSM BN254 2^18 | 45ms | ~5ms | Random-access BW (scatter bucket accumulation) | ~9x |
| 7 | KZG commit 2^10 | 0.7ms | ~0.5ms | C Horner + fused eval/div, cached affine SRS | ~1.4x |
| 8 | Sumcheck 2^20 | 7.3ms | ~0.85ms | Bandwidth (2^20 x 32B per round) | ~9x |
| 9 | ECDSA batch 64 (CPU) | 1.7ms | ~0.5ms | C CIOS Fr + fused batch prepare (was 8ms) | ~3x |
| 10 | Basefold open 2^18 | 144ms | ~20ms | Iterative fold+commit (18 rounds x Merkle) | ~7x |
| 11 | FRI Fold 2^20 | 1.96ms | ~0.3ms | Bandwidth (2^20 x 32B read+write) | ~7x |
| 12 | BLS12-377 MSM 2^18 | 218ms | ~35ms | Wider limbs (253-bit), less optimized window sizes | ~6x |
| 13 | Keccak Merkle 2^20 | 13ms | ~2.2ms | Compute (24 rounds x 64-bit) + 20 levels | ~6x |
| 14 | Blake3 Batch 2^20 | 3.5ms | ~0.6ms | Bandwidth (2^20 x 64B) | ~6x |
| 15 | Circle STARK prove 2^14 | 21ms | ~10ms | Batched CBs, trace caching (56ms→21ms, **2.7x**) | ~2x |
| 16 | HyperNova per-fold | 0.09ms | ~0.07ms | Near floor: C CIOS + Keccak + pre-computed affine (40x from 3.6ms) | ~1.3x |
| 17 | secp256k1 MSM 2^18 | 113ms | ~30ms | No GLV, buffer caching + mixed-add unsafe (**10x** from 1133ms) | ~4x |
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

BabyBear/Goldilocks NTT and IPA are near-optimal (within 1-2x of hardware limits).

**CPU batch operations are now near-optimal.** 50+ engines converted to C kernels with prefetch hints, branchless arithmetic, and auto-parallel dispatch (`dispatch_apply` for n >= 4096). Key conversions: `batch_mul_scalar`, `batch_inverse` (Montgomery's trick, 9 call sites), `batch_axpy`, `inner_product`, `vector_sum`, `fold_interleaved_inplace`, `horner_eval`, `synthetic_div`, `fold_halves`, `linear_combine`, `prefix_product`. Hadamard vector ops (l0/r0/r1 patterns in 4 files) decomposed into 5-6 batch kernel calls each. Manual Montgomery's trick patterns replaced with `batch_scalar_sub` + `batch_inverse_safe`. In-place fold eliminates allocation in sumcheck loops. NTT/INTT butterfly stages dispatch across cores for large transforms. All BN254 Fr serial field loops are exhausted -- remaining loops are inherently serial (power sequences), complex multi-step, or small-n fallbacks (<4 elements).

**The system is GPU-bound.** At peak optimization on M3 Pro (BN254 UltraHonk 428K gates, ~969ms prove), the profile shows ~59% GPU time (MSM commits, Gemini, KZG), ~31% CPU, ~10% overhead. CPU micro-optimizations are exhausted -- all BN254 Fr batch patterns converted, all allocation patterns optimized (in-place fold, pointer offsets instead of Array copies, removeLast instead of new allocations).

**Remaining systemic opportunities**: Command buffer chaining (332 `waitUntilCompleted` sync points could batch into ~10 chained dispatches, saving 3-8ms). FRI fold-by-4 halves round count, reducing Merkle commit overhead. BabyBear/Goldilocks now have C batch kernels for FRI fold, vanishing polynomial, and batch inverse (standard-form arithmetic matching Swift `Bb`/`Gl` structs).

**Algorithmic (mostly realized)**: Granger-Scott cyclotomic squaring, projective G2 Miller loop (BLS12-381 78→0.9ms), sparse line multiplication, dedicated fp_sqr, fp_mul9 shift-add chains, precomputed G2 line coefficients, Frobenius precomputation. Binary tower PMULL intrinsics realized (GF(2^{16,32,64,128}) via ARM NEON carry-less multiply + Karatsuba).

**Near floor** (< 1.5x headroom): BabyBear NTT, Goldilocks NTT, Circle NTT, IPA prove, HyperNova fold, KZG commit, Groth16 prove (cached). These are within noise of hardware limits. All BN254 Fr CPU paths are at their theoretical floor -- batch inverse, Hadamard ops, fold operations, and inner products all converted to C NEON kernels with zero-copy bridging. Further gains require hardware upgrade (M4 Pro/Max with more GPU cores), protocol changes (fewer commitment rounds), or application-level caching (circuit/ProverInstance reuse).

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

## Language Bindings

| Language | Location | Description |
|----------|----------|-------------|
| **Rust** | `bindings/rust/` | `zkmetal-sys` crate — safe wrappers for BN254 Fr, MSM, NTT, multi-curve ops |
| **Go** | `bindings/go/` | cgo package — GPU MSM/NTT/hash/pairing/FRI for gnark integration |
| **C++ (Barretenberg)** | `bindings/barretenberg/` | CMake find module + bridge headers for `libzkmetal.a` linking |
| **WebGPU** | `Sources/zkMetal/WebGPU/` | WGSL shader codegen (u64 half-limb emulation) for browser-based proving |

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
- **C CIOS field arithmetic**: Hot-path 256-bit Montgomery multiplication uses C `__uint128_t` compiled with `-O3`, which is 156x faster than Swift for BN254 Fr (16ns vs 2500ns). All 10 field types (BN254/BLS12-381/BLS12-377/Ed25519/Secp256k1/Pallas/Vesta/Stark252 Fr/Fp) use zero-copy C bridge.
- **Zero-copy Swift↔C bridge**: Field elements (8×UInt32 or 4×UInt64 tuples) share memory layout with C `uint64_t[4]`. `UnsafeRawPointer` cast avoids all heap allocation. 50+ batch C kernels (mul_scalar, batch_inverse, batch_axpy, inner_product, vector_sum, fold_interleaved_inplace, horner_eval, synthetic_div, fold_halves, linear_combine, prefix_product, sumcheck_reduce, batch_scalar_sub, batch_add_scalar, batch_mac, bb_fri_fold, bb_batch_inverse, bb_vanishing_poly, gl_fri_fold, gl_batch_inverse, gl_vanishing_poly) eliminate per-element call overhead with prefetch hints and branchless arithmetic. Auto-parallel dispatch via `dispatch_apply` for n >= 4096. GPU buffer results read via `bindMemory(to: Fr.self)` instead of per-element reconstruction.
- **Small-input fast path**: MSM automatically routes to multi-threaded C Pippenger for small inputs (BN254 n<=2048, secp256k1 n<=1024) to avoid GPU dispatch overhead.

## Correctness & Testing

Run `swift build -c release && .build/release/zkMetalTests`. 234 test files, 233 test suites. All GPU kernels verified against CPU reference implementations.

Filter tests by keyword: `.build/release/zkMetalTests pairing groth16 gpu` runs only matching suites. Use `--list` to see all test names.

| Category | Primitives | Verification |
|----------|------------|-------------|
| Field arithmetic | BN254, BLS12-377/381, secp256k1, Goldilocks, BabyBear, M31, Pallas/Vesta, Binary Tower | Unit tests + cross-checks (arithmetic properties, inverses, distributivity) |
| MSM | BN254, BLS12-377, secp256k1, Pallas/Vesta, Ed25519, Grumpkin, Multi-MSM | GPU vs CPU cross-check, on-curve, determinism |
| NTT | BN254, BLS12-377, Goldilocks, BabyBear, Stark252, Circle NTT | Round-trip + CPU cross-check (all fields, sizes 2^2 through 2^22) |
| Hashing | Poseidon2 (BN254+M31+BabyBear), Keccak-256, Blake3, SHA-256, Poseidon2 Sponge | Known-answer tests (NIST, HorizenLabs, BLAKE3 spec) + GPU vs CPU batch + duplex sponge |
| Merkle trees | Poseidon2, Keccak, Blake3 backends | GPU vs CPU root comparison + parallel structure validation |
| Polynomial protocols | FRI (fold-by-2/4/8), Sumcheck (dense+sparse), KZG, Batch KZG, Coset LDE | S(0)+S(1)=sum, round-poly match, full protocol verify, tamper rejection |
| PCS | Basefold, Tensor, WHIR, Zeromorph, Pedersen, Hyrax | Fold correctness, compress+verify, proof verify+rejection |
| Proof systems | Circle STARK, Plonk, Groth16, GKR, Spartan, Protogalaxy, Nova/SuperNova IVC | Prove+verify, tampered proof rejection, bilinearity, fold+cross-term+error propagation |
| Lookups | LogUp, Lasso, cq, Unified Lookup | Simple/repeated/multiplicities, tamper rejection, auto-strategy |
| CPU optimized | C BN254/Goldilocks NTT, NEON BabyBear/Keccak/Blake3 | Cross-checked against vanilla CPU + round-trip |
| Signatures | EdDSA (Ed25519+BabyJubjub), ECDSA (secp256k1), Schnorr (BIP 340) | RFC 8032, batch verification, tagged hashing |
| Protocols | IPA, Verkle Proofs (CPU), HyperNova, IPA Accumulation | Prove+verify, wrong-value rejection, batch verification |
| Constraint systems | Circom R1CS parser, Plonk constraint compiler, AIR DSL, constraint optimizer, Plonk custom gates, Plonk permutation argument | Binary format parsing, gate compilation, dead elimination, CSE, grand product |
| zkVM | RISC-V decoder, Jolt VM, witness gen | All 40 RV32I opcodes, execution trace, instruction-stream architecture |
| Infrastructure | Transcript (Merlin STROBE), Serialization, Witness Gen, Constraint IR, Radix Sort, GPU Buffer Pool | Determinism, roundtrip, domain/fork separation, tamper detection, pool recycling |
| Interop | Groth16 Solidity verifier, Universal Proof Format, Plonky2 recursive verifier, SP1 bridge | Contract gen, binary/JSON roundtrip, snarkjs JSON, proof composition |
| Advanced | STARK trace compression, Data availability sampling, Multilinear polynomial engine, Proof aggregation | Algebraic/interleave/ZK, EIP-4844 blobs, MLE + PCS adapters, multi-proof batching |
| Applications | Kyber KEM, Dilithium signatures, Reed-Solomon, Streaming Verify, Batch Verify | Shared secret agreement, signature verification, encode+decode |

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization.
