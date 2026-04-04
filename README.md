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
  - [Polynomial Ops](#polynomial-ops-bn254-fr)
  - [KZG Commitments](#kzg-commitments-bn254-g1)
  - [Blake3 Hashing](#blake3-hashing)
  - [IPA](#ipa-bulletproofs-style-inner-product-argument)
  - [Verkle Trees (CPU)](#verkle-trees-cpu-pedersen--ipa)
  - [ECDSA (CPU)](#ecdsa-cpu-secp256k1-batch-verification)
  - [Circle STARK](#circle-stark-mersenne31)
  - [Plonk](#plonk-bn254-kzg)
  - [Groth16](#groth16-bn254)
  - [GKR](#gkr-bn254-fr-layered-circuits)
  - [Circle NTT](#circle-ntt-mersenne31-gpu)
  - [Radix Sort](#gpu-radix-sort)
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
| **MSM** | Multi-scalar multiplication (Pippenger + signed-digit + GLV endomorphism) -- BN254, BLS12-377, secp256k1 |
| **NTT** | Number theoretic transform (four-step FFT with fused bitrev+butterfly, twiddle fusion) -- BN254, BLS12-377, Goldilocks, BabyBear |
| **Poseidon2** | Algebraic hash function (t=3, BN254 Fr; t=16, Mersenne31) |
| **Keccak-256** | SHA-3 hash (fused subtree Merkle) |
| **Blake3** | BLAKE3 hash (batch hashing + Merkle trees) |
| **Merkle Trees** | Poseidon2, Keccak-256, and Blake3 backends |
| **FRI** | Fast Reed-Solomon IOP (fold + commit + query + verify) |
| **Circle FRI** | FRI protocol adapted for circle domain over M31 |
| **Sumcheck** | Interactive sumcheck protocol (fused round+reduce with SIMD shuffles) |
| **Sparse Sumcheck** | O(nnz) sumcheck for sparse multilinear polynomials |
| **KZG** | Polynomial commitment scheme (commit + open via MSM) |
| **Batch KZG** | Batch polynomial openings via random linear combination |
| **IPA** | Inner product argument (Bulletproofs-style, GPU batch fold) |
| **Verkle Trees (CPU)** | Width-N tree with Pedersen commitments + IPA opening proofs |
| **LogUp** | Lookup argument via logarithmic derivatives + sumcheck |
| **Lasso** | Structured lookup via tensor decomposition of large tables |
| **Range Proofs** | Proves values in [0, R) via LogUp with limb decomposition |
| **ECDSA (CPU)** | secp256k1 batch verification (probabilistic + individual) |
| **Radix Sort** | GPU 32-bit LSD radix sort (4-pass, 8-bit radix) |
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees |
| **Circle NTT** | Circle-group NTT over Mersenne31 (order 2^31, full 2-adicity) |
| **Circle STARK** | Full STARK prover/verifier over M31 circle domain |
| **Plonk** | Preprocessed polynomial IOP prover with KZG commitments |
| **Groth16** | zk-SNARK with BN254 pairings (R1CS, trusted setup, prove, verify) |
| **GKR** | Goldwasser-Kalai-Rothblum interactive proof for layered circuits |
| **Basefold** | NTT-free multilinear polynomial commitment via sumcheck folding |
| **Brakedown** | Linear-code polynomial commitment (NTT-free, expander-based) |
| **HyperNova** | CCS folding scheme for incremental verifiable computation |
| **BLS12-381** | Full tower arithmetic (Fp/Fp2/Fp6/Fp12), G1/G2, Miller loop, pairings |
| **Pasta Curves** | Pallas/Vesta cycle (recursive proof composition ready) |
| **Binius** | Binary tower field arithmetic (GF(2^8)->GF(2^128)), additive FFT |
| **Transcript** | Fiat-Shamir duplex sponge (Poseidon2 + Keccak backends) |
| **Serialization** | Proof serialization/deserialization (ProofWriter/ProofReader) |
| **Witness Gen** | GPU witness trace evaluation (instruction-stream architecture) |
| **Constraint IR** | Runtime constraint compilation (IR -> Metal source -> GPU pipeline) |
| **Kyber/Dilithium** | Post-quantum lattice crypto (GPU-accelerated NTT) |
| **HE NTT** | RNS-based NTT for homomorphic encryption (CKKS/BFV) |
| **Reed-Solomon** | GPU erasure coding for data availability sampling |

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
| 2^14 | 0.41ms | 15ms | **37x** |
| 2^16 | 0.59ms | 58ms | **98x** |
| 2^18 | 1.2ms | 225ms | **188x** |
| 2^20 | 4.4ms | 878ms | **200x** |
| 2^22 | 9.2ms | 2.7s | **293x** |

Full fold-to-constant: 2^20 in 4.5ms (20 rounds, fused 4-round kernels).

GPU scales sublinearly: 2^14 to 2^22 is 256x more data for ~22x more time, as each folding round halves the domain. CPU scales linearly. Speedup grows from 37x to 293x. No other Metal FRI implementations are known.

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 0.89ms | 21ms | **24x** |
| 2^16 | 0.85ms | 83ms | **98x** |
| 2^18 | 2.2ms | 337ms | **153x** |
| 2^20 | 7.3ms | 1.3s | **178x** |
| 2^22 | 16ms | 5.2s | **325x** |

GPU scales sublinearly: 2^14 to 2^22 is 256x more variables for ~18x more time. CPU scales linearly. Each sumcheck round reduces the problem by half, and fused round+reduce kernels keep GPU utilization high. No other Metal sumcheck implementations are known; ICICLE (CUDA) offers GPU sumcheck but no published comparison numbers.

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

### Blake3 Hashing

| Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----|-------|---------|
| 2^14 | 0.012 µs/hash | 0.6 µs/hash | **50x** |
| 2^16 | 0.006 µs/hash | 0.6 µs/hash | **100x** |
| 2^18 | 0.007 µs/hash | 0.6 µs/hash | **86x** |
| 2^20 | 0.003 µs/hash | 0.6 µs/hash | **200x** |

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

### Theoretical Performance Analysis

How close each primitive is to the hardware floor (M3 Pro: ~3.6 TFLOPS, ~150 GB/s bandwidth), ranked by optimization headroom:

| Rank | Primitive | Current | Theoretical Floor | Bottleneck | Headroom |
|------|-----------|---------|-------------------|------------|----------|
| 1 | P2 Merkle 2^16 | 22ms | ~0.6ms (compute) | Dispatch latency (16 levels) | ~37x |
| 2 | KZG commit 2^10 | 4.6ms | ~0.5ms | MSM-dominated (small N) | ~9x |
| 3 | MSM BN254 2^18 | 45ms | ~5ms (scatter BW) | Random-access BW | ~9x |
| 4 | FRI Fold 2^20 | 4.4ms | ~0.3ms (BW) | Bandwidth | ~15x |
| 5 | ECDSA batch 64 | 8ms | ~1ms | C CIOS scalar mul | ~8x |
| 6 | Sumcheck 2^20 | 7.3ms | ~0.85ms (BW) | Bandwidth | ~9x |
| 7 | Radix Sort 2^20 | 4.1ms | ~1ms (BW) | Sequential passes + BW | ~4x |
| 8 | NTT BN254 2^22 | 26ms | ~2.9ms (compute) | Compute + strided BW | ~9x |
| 9 | Blake3 Batch 2^20 | 3.5ms | ~0.6ms (BW) | Bandwidth | ~6x |
| 10 | IPA prove n=256 | 13ms | ~10ms | C scalar mul + GPU batch fold | ~1.3x |
| 11 | Keccak Batch 2^18 | 1.4ms | ~0.5ms (compute) | Compute | ~3x |
| 12 | Verkle proof 256 | 23ms | ~10ms | IPA-dominated (C scalar mul) | ~2.3x |
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
swift run -c release zkbench fri       # FRI folding
swift run -c release zkbench sumcheck  # Sumcheck
swift run -c release zkbench bls377msm  # MSM (BLS12-377 G1)
swift run -c release zkbench secpmsm    # MSM (secp256k1 G1)
swift run -c release zkbench ecdsa      # ECDSA batch verification
swift run -c release zkbench ipa        # IPA (Bulletproofs-style)
swift run -c release zkbench verkle     # Verkle trees
swift run -c release zkbench lookup     # LogUp lookup argument
swift run -c release zkbench sparse     # Sparse sumcheck
swift run -c release zkbench sort       # GPU radix sort
swift run -c release zkbench lasso      # Lasso structured lookups
swift run -c release zkbench circle     # Circle NTT + M31 benchmarks
swift run -c release zkbench circle-fri # Circle FRI over M31
swift run -c release zkbench circle-stark # Circle STARK prover/verifier
swift run -c release zkbench p2m31      # Poseidon2 over Mersenne31
swift run -c release zkbench basefold   # Basefold PCS
swift run -c release zkbench transcript # Fiat-Shamir transcript
swift run -c release zkbench witness    # GPU witness generation
swift run -c release zkbench constraint # Constraint IR evaluation
swift run -c release zkbench kzg-batch  # Batch KZG openings
swift run -c release zkbench bls381     # BLS12-381 field + curve + pairing
swift run -c release zkbench pasta      # Pasta curves (Pallas/Vesta)
swift run -c release zkbench pastamsm   # Pasta MSM
swift run -c release zkbench binius     # Binary tower fields
swift run -c release zkbench fold       # HyperNova folding
swift run -c release zkbench gkr        # GKR protocol
swift run -c release zkbench plonk      # Plonk prover
swift run -c release zkbench groth16    # Groth16 zk-SNARK
swift run -c release zkbench serialize  # Proof serialization
swift run -c release zkbench versions   # Print primitive versions
swift run -c release zkbench all        # Everything
swift run -c release zkbench test       # Correctness tests (all primitives)
swift run -c release zkbench cpu        # CPU-optimized vs GPU comparison
swift run -c release zkbench calibrate  # Re-calibrate GPU parameters
swift run -c release zkbench all --no-cpu  # GPU-only (skip slow CPU baselines)
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
| **Poseidon2** | [HorizenLabs reference](https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs) | Known-answer test + GPU vs CPU batch cross-check |
| **Keccak-256** | FIPS 202 (SHA-3) | NIST test vectors + GPU vs CPU batch cross-check |
| **Blake3** | [BLAKE3 spec](https://github.com/BLAKE3-team/BLAKE3-specs) | Reference vector + GPU vs CPU batch cross-check |
| **Merkle trees** | Poseidon2, Keccak, Blake3 backends | GPU vs CPU root comparison + parallel structure validation |
| **NTT** | Cooley-Tukey (DIT) / Gentleman-Sande (DIF) | Round-trip (2^10, 2^20, 2^22) + CPU cross-check (all 4 fields) |
| **FRI** | Standard FRI protocol | GPU vs CPU fold cross-check + full commit/query/verify round-trip |
| **Sumcheck** | Standard interactive protocol | S(0)+S(1)=sum verification + GPU vs CPU reduce/roundPoly |
| **Sparse Sumcheck** | O(nnz) sparse multilinear | Round-poly match, reduce match, full protocol, proof verify |
| **KZG** | Polynomial commitment (MSM-based) | Commit linearity, multi-point eval, constant/linear poly checks |
| **IPA** | Bulletproofs-style inner product argument | Prove + verify at n=4,16,64,256 + wrong-value rejection |
| **Verkle trees** | Width-N Pedersen + IPA openings | Single openings, tree build, path proofs, wrong-root rejection |
| **LogUp** | Lookup via logarithmic derivatives | 7 tests: simple, repeated, multiplicities, batch inverse, tamper rejection |
| **ECDSA** | secp256k1 batch verification | Single verify, wrong msg/key rejection, batch 64, bad-sig detection |
| **Radix sort** | LSD radix sort (4-pass, 8-bit) | 10 tests: sorted, reverse, duplicates, random, KV, edge cases |
| **Goldilocks** | p = 2^64 - 2^32 + 1 (standard) | NTT round-trip + CPU cross-check |
| **BabyBear** | p = 2^31 - 2^27 + 1 (standard) | NTT round-trip + CPU cross-check |
| **Circle NTT** | Circle group over M31 | GPU vs CPU roundtrip at all sizes 2 through 4096 |
| **Circle STARK** | Fibonacci AIR over M31 | Prove + verify + tampered proof rejection |
| **Plonk** | Preprocessed BN254 + KZG | Prove + verify at n=16,64,256,1024 |
| **Groth16** | BN254 R1CS + pairings | Prove + verify, bilinearity checks |
| **GKR** | Layered circuit sumcheck | 1-layer, 2-layer, hash circuits, inner product circuits |
| **Parallel CPU** | GCD multithreaded implementations | Cross-checked against vanilla CPU for NTT (Fr, Bb, Gl), MSM, batch hash, Merkle |
| **NEON BabyBear** | C/ARM NEON Montgomery NTT (4-wide SIMD, Plonky3 technique) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **C Goldilocks** | Optimized C NTT (`__uint128_t` mul pipelining) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **C BN254 Fr** | Fully unrolled 4-limb CIOS Montgomery NTT | Cross-checked against vanilla cpuNTT + round-trip verification |

Every benchmark run includes correctness checks (printed as PASS/FAIL). The test suite (`swift test`) covers field arithmetic, curve operations, and NTT correctness.

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization. To continue tuning for your hardware or workload, install the skill and run `/floptimizer` in a Claude Code session from this repo.
