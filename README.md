# zkMetal

GPU-accelerated zero-knowledge cryptography library for Apple Silicon, written in Metal and Swift. 50+ primitives spanning field arithmetic, MSM, NTT, hash functions, polynomial commitments, proof protocols, post-quantum crypto, and homomorphic encryption across 12+ fields and 7 elliptic curves.

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
  - [Verkle Trees](#verkle-trees-pedersen--ipa)
  - [ECDSA](#ecdsa-secp256k1-batch-verification)
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
| **MSM** | Multi-scalar multiplication (Pippenger + signed-digit + GLV endomorphism) — BN254, BLS12-377, secp256k1 |
| **NTT** | Number theoretic transform (four-step FFT with fused sub-blocks) — BN254, BLS12-377, Goldilocks, BabyBear |
| **Poseidon2** | Algebraic hash function (t=3, BN254 Fr) |
| **Keccak-256** | SHA-3 hash (fused subtree Merkle) |
| **Blake3** | BLAKE3 hash (batch hashing + Merkle trees) |
| **Merkle Trees** | Poseidon2, Keccak-256, and Blake3 backends |
| **FRI** | Fast Reed-Solomon IOP (fold + commit + query + verify) |
| **Sumcheck** | Interactive sumcheck protocol (fused round+reduce with SIMD shuffles) |
| **Sparse Sumcheck** | O(nnz) sumcheck for sparse multilinear polynomials |
| **KZG** | Polynomial commitment scheme (commit + open via MSM) |
| **IPA** | Inner product argument (Bulletproofs-style, GPU batch fold) |
| **Verkle Trees** | Width-N tree with Pedersen commitments + IPA opening proofs |
| **LogUp** | Lookup argument via logarithmic derivatives + sumcheck |
| **Range Proofs** | Proves values ∈ [0, R) via LogUp with limb decomposition |
| **ECDSA** | secp256k1 batch verification (probabilistic + individual) |
| **Radix Sort** | GPU 32-bit LSD radix sort (4-pass, 8-bit radix) |
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees |
| **Lasso** | Structured lookup via tensor decomposition of large tables |
| **Circle NTT** | Circle-group NTT over Mersenne31 (order 2^31, full 2-adicity) |
| **Circle FRI** | FRI protocol adapted for circle domain over M31 |
| **Poseidon2 M31** | Poseidon2 hash over Mersenne31 (t=16, rate=8) with GPU Merkle |
| **Circle STARK** | Full STARK prover/verifier over M31 circle domain |
| **Basefold** | NTT-free multilinear polynomial commitment via sumcheck folding |
| **Transcript** | Fiat-Shamir duplex sponge (Poseidon2 + Keccak backends) |
| **Serialization** | Proof serialization/deserialization (ProofWriter/ProofReader) |
| **Witness Gen** | GPU witness trace evaluation (instruction-stream architecture) |
| **Constraint IR** | Runtime constraint compilation (IR → Metal source → GPU pipeline) |
| **Batch KZG** | Batch polynomial openings via random linear combination |
| **BLS12-381** | Full tower arithmetic (Fp/Fp2/Fp6/Fp12), G1/G2, Miller loop, pairings |
| **Pasta Curves** | Pallas/Vesta cycle (recursive proof composition ready) |
| **Binius** | Binary tower field arithmetic (GF(2^8)→GF(2^128)), additive FFT |
| **HyperNova** | CCS folding scheme for incremental verifiable computation |
| **GKR** | Goldwasser-Kalai-Rothblum interactive proof for layered circuits |
| **Brakedown** | Linear-code polynomial commitment (NTT-free, expander-based) |
| **Plonk** | Preprocessed polynomial IOP prover with KZG commitments |
| **Kyber/Dilithium** | Post-quantum lattice crypto (GPU-accelerated NTT) |
| **HE NTT** | RNS-based NTT for homomorphic encryption (CKKS/BFV) |
| **Reed-Solomon** | GPU erasure coding for data availability sampling |

## Performance

All benchmarks measured on Apple M3 Pro (6P+6E cores), comparing GPU (Metal), optimized C CPU (CIOS Montgomery with `__uint128_t`, multi-threaded Pippenger, NEON SIMD), and single-threaded CPU (vanilla Swift).
Run `swift run -c release zkbench all` to reproduce, or `swift run -c release zkbench cpu` for the 3-way comparison. For small inputs (MSM n≤2048), the engine automatically routes to C Pippenger instead of GPU to avoid dispatch overhead.

### MSM (BN254 G1)

| Points | Vanilla CPU | Swift Pippenger | C Pippenger | GPU (Metal) |
|--------|-------------|----------------|-------------|-------------|
| 2^8 | 5.5s | 63ms | **11ms** | 8ms |
| 2^10 | 24s | 121ms | **6ms** | 4ms |
| 2^12 | 19s | 494ms | **9ms** | 16ms |
| 2^14 | 87s | 848ms | **28ms** | 29ms |
| 2^16 | — | — | **71ms** | 43ms |
| 2^18 | — | — | — | 102ms |
| 2^20 | — | — | — | 294ms |

C Pippenger uses multi-threaded bucket accumulation with `__uint128_t` CIOS Montgomery (8 pthreads). At n≤2048, C Pippenger is automatically used instead of GPU to avoid dispatch overhead. GPU wins at n≥2^16.

**Comparison to other implementations (BN254 MSM):**

| Points | zkMetal (M3 Pro)&#185; | ICICLE-Metal (M3 Pro)&#185; | ICICLE CPU (M3 Pro)&#185; | ICICLE-Metal (M3 Air)&#178; | MoPro v2 (M3 Air)&#178; | Arkworks CPU (M3 Air)&#178; | ICICLE CUDA&#179; |
|--------|---------|-------------|-----------|-------------|-----------|-----------|-----------|
| 2^16 | **37ms** | 1,083ms | 114ms | — | 253ms | 69ms | ~9ms |
| 2^18 | **102ms** | 1,475ms | 556ms | 149ms | 678ms | 266ms | — |
| 2^20 | **294ms** | 2,590ms | 2,349ms | 421ms | 1,702ms | 592ms | — |

&#185; Measured locally. ICICLE-Metal v3.8.0 has ~600ms constant overhead per call (license server).
&#178; Reported by [MoPro blog](https://zkmopro.org/blog/metal-msm-v2/) — different hardware and methodology, not directly comparable.
&#179; [Ingonyama](https://github.com/ingonyama-zk/icicle) on RTX 3090 Ti (native 64-bit integer multiply).

Metal GPU MSM is competitive with other Metal implementations and faster than ICICLE-Metal, but still slower than optimized multithreaded CPU (Arkworks). The fundamental bottleneck is that 256-bit field arithmetic requires 8x32-bit limbs on Metal (no native 64-bit integer multiply), while CPU implementations use 4x64-bit limbs with hand-tuned assembly, out-of-order execution, and deep pipelines. CUDA GPUs (like those targeted by [Ingonyama's ICICLE](https://github.com/ingonyama-zk/icicle)) have native 64-bit integer multiply. The GPU advantage is clear for smaller fields: BabyBear NTT achieves **2.4B elements/sec** (7ms at 2^24) and Goldilocks **2.5B elements/sec** (6.6ms at 2^24) — both dramatically faster than BN254 on the same GPU (see NTT table below).

GPU scaling is strongly sublinear: 1024x more points (2^8 to 2^18) costs only ~9x more time, as fixed GPU overhead dominates at small sizes.

### NTT

**BN254 Fr (256-bit, 8x32-bit limbs):**

| Size | Vanilla CPU | Opt C | Opt C vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|------|-------------|-------|------------------|-------------|----------------|
| 2^14 | 94ms | 2.7ms | **35x** | 0.49ms | **192x** |
| 2^16 | 481ms | 18ms | **27x** | 0.95ms | **506x** |
| 2^18 | 3.3s | 108ms | **30x** | 1.9ms | **1737x** |
| 2^20 | 9.0s | 503ms | **18x** | 6.1ms | **1475x** |

Optimized C uses fully unrolled 4-limb CIOS Montgomery multiplication with `__uint128_t` (compiled with `-O3`). Also available: parallel CPU (GCD, 12 cores) at 5.4x over vanilla.

**Multi-field NTT comparison (GPU):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.95ms | 1.8ms | 0.15ms | 0.17ms |
| 2^18 | 1.9ms | 2.1ms | 0.21ms | 0.26ms |
| 2^20 | 6.1ms | 6.3ms | 0.84ms | 1.2ms |
| 2^22 | 26ms | 26ms | 4.4ms | 2.9ms |
| 2^24 | 113ms | 116ms | 3.0ms | 2.0ms |

Smaller fields see dramatic throughput gains: BabyBear NTT at 2^24 (16M elements) runs in **2ms** — one element per 0.12ns, or **8.5B elements/sec**. The GPU advantage for small fields comes from native 32-bit arithmetic (1 mul per element vs 64 muls for BN254 CIOS), 8x higher memory density, and better threadgroup utilization.

**CPU optimization results by field:**
- **BN254 Fr:** C with unrolled CIOS Montgomery gives **29-38x** over vanilla Swift (Swift's optimizer is very poor for 256-bit multi-limb carry chains).
- **BabyBear:** NEON SIMD (4-wide Montgomery via `vqdmulhq_s32`, Plonky3 technique) gives **5.6x** over vanilla.
- **Goldilocks:** Optimized C with `__uint128_t` gives **1.8-3.1x** over vanilla.
- GCD parallel dispatch **regresses** for BabyBear/Goldilocks (0.4x) — field ops are too cheap for thread overhead.

**Comparison to ICICLE-Metal v3.8 NTT (measured locally, M3 Pro):**

| Size | zkMetal BN254&#185; | ICICLE BN254&#185; | zkMetal BabyBear&#185; | ICICLE BabyBear&#185; |
|------|------------|-------------|----------------|----------------|
| 2^16 | **0.95ms** | 89ms | **0.17ms** | 86ms |
| 2^18 | **1.9ms** | 108ms | **0.26ms** | 92ms |
| 2^20 | **6.1ms** | 194ms | **1.2ms** | 108ms |
| 2^22 | **26ms** | 915ms | **2.9ms** | 181ms |
| 2^24 | **113ms** | 3,892ms | **2.0ms** | 709ms |

&#185; Measured locally on M3 Pro. ICICLE-Metal has ~85ms per-call overhead.

zkMetal is **30-90x faster** on BN254 and **90-500x faster** on BabyBear. ICICLE does not ship Goldilocks in their Metal backend.

GPU scales sublinearly: 2^10 to 2^22 is 4096x more data for ~100x more time. CPU scales linearly with n log n. Speedup grows with input size.

### Hashing

| Primitive | Batch Size | Vanilla CPU | Parallel CPU (12 cores) | Parallel vs Vanilla | GPU (Metal) | GPU vs Vanilla |
|-----------|-----------|-------------|------------------------|--------------------|--------------------|----------------|
| Poseidon2 | 2^12 | 119 µs/hash | 21 µs/hash | **6x** | 0.61 µs/hash | **195x** |
| Poseidon2 | 2^14 | 128 µs/hash | 20 µs/hash | **6x** | 0.17 µs/hash | **753x** |
| Poseidon2 | 2^16 | 150 µs/hash | 19 µs/hash | **8x** | 0.14 µs/hash | **1071x** |
| Keccak-256 | 2^14 | 6 µs/hash | 1.5 µs/hash | **4x** | 0.035 µs/hash | **171x** |
| Keccak-256 | 2^16 | 6 µs/hash | 1.4 µs/hash | **4x** | 0.027 µs/hash | **222x** |
| Keccak-256 | 2^18 | 6.2 µs/hash | 1.5 µs/hash | **4x** | 0.012 µs/hash | **517x** |

Parallel CPU achieves 4-8x over vanilla (embarrassingly parallel — each hash independent). GPU achieves 195-1071x over vanilla. No other Metal implementations of Poseidon2 or Keccak-256 batch hashing are known.

### Merkle Trees

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 19ms | 272ms | **14x** |
| Poseidon2 | 2^12 | 19ms | 2.0s | **102x** |
| Poseidon2 | 2^14 | 16ms | 4.7s | **305x** |
| Poseidon2 | 2^16 | 23ms | 20s | **906x** |
| Poseidon2 | 2^18 | 47ms | 66s | **1418x** |
| Poseidon2 | 2^20 | 144ms | — | — |
| Keccak-256 | 2^12 | 1.1ms | 44ms | **39x** |
| Keccak-256 | 2^14 | 3.3ms | 155ms | **48x** |
| Keccak-256 | 2^16 | 12ms | 783ms | **67x** |
| Keccak-256 | 2^18 | 42ms | 3.0s | **72x** |
| Keccak-256 | 2^20 | 159ms | — | — |
| Blake3 | 2^12 | 1.1ms | 4ms | **4x** |
| Blake3 | 2^14 | 3.4ms | 16ms | **5x** |
| Blake3 | 2^16 | 10ms | 101ms | **10x** |
| Blake3 | 2^18 | 49ms | 345ms | **7x** |
| Blake3 | 2^20 | 158ms | — | — |

All three backends scale linearly (O(n) tree construction). GPU speedup grows with size as fixed dispatch overhead is amortized. Blake3 is the fastest Merkle backend at large sizes (96ms vs 136ms Keccak vs 1.5s Poseidon2 at 2^20) due to its simpler 32-bit ARX arithmetic.

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.41ms | 15ms | **37x** |
| 2^16 | 0.68ms | 58ms | **84x** |
| 2^18 | 1.4ms | 225ms | **164x** |
| 2^20 | 5.3ms | 878ms | **167x** |
| 2^22 | 15ms | 2.7s | **182x** |

Full fold-to-constant: 2^20 in 4.7ms (20 rounds, fused 4-round kernels).

GPU scales sublinearly: 2^14 to 2^22 is 256x more data for ~37x more time, as each folding round halves the domain. CPU scales linearly. Speedup grows from 37x to 182x. No other Metal FRI implementations are known.

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 0.68ms | 21ms | **31x** |
| 2^16 | 0.90ms | 83ms | **93x** |
| 2^18 | 2.3ms | 337ms | **144x** |
| 2^20 | 4.4ms | 1.3s | **301x** |
| 2^22 | 14ms | 5.2s | **361x** |

GPU scales sublinearly: 2^14 to 2^22 is 256x more variables for ~17x more time. CPU scales linearly. Each sumcheck round reduces the problem by half, and fused round+reduce kernels keep GPU utilization high. No other Metal sumcheck implementations are known; ICICLE (CUDA) offers GPU sumcheck but no published comparison numbers.

### Polynomial Ops (BN254 Fr)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Multiply (NTT) | deg 2^10 | 27ms | 1.3ms | **21x** |
| Multiply (NTT) | deg 2^12 | 122ms | 1.6ms | **79x** |
| Multiply (NTT) | deg 2^14 | 642ms | 2.9ms | **222x** |
| Multiply (NTT) | deg 2^16 | 2.4s | 4.5ms | **541x** |
| Multi-eval (Horner) | deg 2^10, 1024 pts | — | 1.7ms | — |
| Multi-eval (Horner) | deg 2^12, 4096 pts | — | 8.6ms | — |
| Multi-eval (Horner) | deg 2^14, 16384 pts | — | 115ms | — |

Polynomial multiplication uses NTT under the hood (CPU baseline = 2 forward NTTs + pointwise mul + inverse NTT). Multi-point evaluation uses GPU Horner's method (one thread per evaluation point). Subproduct-tree evaluation is available but currently slower than Horner for these sizes due to high constant factors.

### KZG Commitments (BN254 G1)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Commit | deg 2^8 | 294ms | 9ms | **31x** |
| Commit | deg 2^10 | 1.2s | 15ms | **80x** |
| Open (eval + witness) | deg 2^8 | 444ms | 5ms | **87x** |
| Open (eval + witness) | deg 2^10 | 1.8s | 10ms | **179x** |

KZG performance is MSM-dominated. Commit = MSM(SRS, coefficients). Open = Horner eval + C synthetic division + MSM for witness. CPU baseline uses sequential double-and-add scalar multiplication. SRS generation and quotient polynomial use C CIOS for fast field arithmetic.

### Blake3 Hashing

| Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----|-------|---------|
| 2^14 | 0.011 µs/hash | 0.6 µs/hash | **55x** |
| 2^16 | 0.004 µs/hash | 0.6 µs/hash | **150x** |
| 2^18 | 0.004 µs/hash | 0.6 µs/hash | **150x** |
| 2^20 | 0.004 µs/hash | 0.6 µs/hash | **150x** |

Blake3 is much simpler than Keccak (7 rounds of 32-bit ARX ops vs 24 rounds of 64-bit Keccak-f). GPU speedup scales with batch size as fixed dispatch overhead amortizes. CPU Blake3 is very fast (0.6µs) so GPU only wins at large batch sizes.

### IPA (Bulletproofs-style Inner Product Argument)

| Size | Prove | Verify |
|------|-------|--------|
| n=4 | 1.4ms | 1.5ms |
| n=16 | 3.0ms | 1.8ms |
| n=64 | 5.3ms | 2.5ms |
| n=256 | 12ms | 3.7ms |

Log(n) halving rounds with GPU batch generator folding (Metal kernel `batch_fold_generators`) and C CIOS scalar multiplication. Fiat-Shamir challenges via Blake3.

### Verkle Trees (Pedersen + IPA)

| Operation | Time |
|-----------|------|
| Build (width=16, 256 leaves) | 10ms |
| Path proof (2 openings) | 44ms |
| Verify path | 6ms |

Verkle tree performance is IPA-dominated. Previous version: 931ms path proof — C CIOS gives **21×** improvement.

### ECDSA (secp256k1 Batch Verification)

| Operation | Time |
|-----------|------|
| Single verify | 0.36ms |
| Batch probabilistic 64 sigs | 14ms (0.22ms/sig) |

secp256k1 ECDSA using C CIOS Montgomery field arithmetic. Previous version (Swift scalar mul): 3.96ms/sig single verify — C CIOS gives **11×** improvement.

### Circle STARK (Mersenne31)

| Trace Size | Prove | Verify | Proof Size |
|-----------|-------|--------|------------|
| 2^6 | 62ms | 3.9ms | 17 KB |
| 2^8 | 75ms | 8.8ms | 39 KB |
| 2^10 | 240ms | 12ms | 53 KB |
| 2^12 | 1.34s | 24ms | 69 KB |

GPU-accelerated Circle STARK prover over Mersenne31 with Fibonacci AIR. Merkle commitments via Keccak-256, Fiat-Shamir via custom transcript. M31 arithmetic uses single 32-bit multiply (vs 64 multiplies for BN254), giving native hardware efficiency.

### Circle NTT (Mersenne31, GPU)

| Size | GPU Time | Throughput |
|------|----------|------------|
| 2^14 | 0.40ms | 41M elem/s |
| 2^16 | 0.36ms | 182M elem/s |
| 2^18 | 0.95ms | 276M elem/s |
| 2^20 | 1.82ms | 576M elem/s |

Circle-group NTT exploits the unique structure of the circle domain (x^2+y^2=1 over M31). First fold uses y-coordinates, subsequent folds use x-coordinate squaring map. All operations are single-word 32-bit arithmetic.

### Theoretical Performance Analysis

How close each primitive is to the hardware floor (M3 Pro: ~3.6 TFLOPS, ~150 GB/s bandwidth), ranked by optimization headroom:

| Rank | Primitive | Current | Theoretical Floor | Bottleneck | Headroom |
|------|-----------|---------|-------------------|------------|----------|
| 1 | P2 Merkle 2^16 | 23ms | ~0.6ms (compute) | Dispatch latency (16 levels) | ~37x |
| 2 | KZG commit 2^10 | 15ms | ~0.5ms | MSM-dominated (small N) | ~30x |
| 3 | MSM BN254 2^18 | 102ms | ~5ms (scatter BW) | Random-access BW | ~20x |
| 4 | FRI Fold 2^20 | 5.3ms | ~0.3ms (BW) | Bandwidth | ~17x |
| 5 | ECDSA batch 64 | 14ms | ~1ms | C CIOS scalar mul | ~14x |
| 6 | P2 Batch 2^16 | 9.2ms | ~0.6ms (compute) | Compute | ~15x |
| 7 | Sumcheck 2^20 | 8.3ms | ~0.85ms (BW) | Bandwidth | ~10x |
| 8 | Radix Sort 2^20 | 10ms | ~1ms (BW) | Sequential passes + BW | ~10x |
| 9 | NTT BN254 2^22 | 26ms | ~2.9ms (compute) | Compute + strided BW | ~9x |
| 10 | Blake3 Batch 2^20 | 4.2ms | ~0.6ms (BW) | Bandwidth | ~7x |
| 11 | IPA prove n=256 | 59ms | ~10ms | C scalar mul + GPU batch fold | ~6x |
| 12 | Keccak Batch 2^18 | 3.1ms | ~0.5ms (compute) | Compute | ~6x |
| 13 | Verkle proof 256 | 44ms | ~10ms | IPA-dominated (C scalar mul) | ~4x |
| 14 | NTT Goldilocks 2^24 | 3.0ms | ~1.8ms (compute) | Compute ≈ BW | ~1.7x |
| 15 | NTT BabyBear 2^24 | 2.0ms | ~1.7ms (BW) | Bandwidth | ~1.2x |

Notes: IPA/Verkle dramatically improved (14-21×) by replacing Swift scalar multiplication with C CIOS `__uint128_t` and GPU batch generator folding. MSM's realistic floor accounts for scatter-gather inefficiency in bucket accumulation. Poseidon2 Merkle overhead comes from 16 sequential kernel dispatches (~0.5ms each). KZG at small sizes is dispatch-overhead dominated. BabyBear and Goldilocks NTT are near-optimal — within 1-2x of hardware bandwidth limits.

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
- **Binary Tower** -- GF(2^8)→GF(2^16)→GF(2^32)→GF(2^64)→GF(2^128) (XOR addition)

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
    mont256.c            # BN254 Fr CIOS NTT (29-38× over Swift)
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

This takes ~500ms and prints the discovered parameters. Different Apple Silicon chips (M1, M2, M3, M4) have different optimal settings — auto-tuning ensures peak performance on any hardware.

## Building

Requires macOS 13+ and Xcode with Metal support.

```bash
swift build -c release
```

## Design Decisions

- **Montgomery form everywhere**: All field elements stay in Montgomery representation on GPU. Conversion happens only at host boundaries.
- **Buffer caching**: GPU Metal buffers are cached and reused across calls to avoid allocation overhead.
- **Four-step FFT**: Large NTTs (>2^16) split into sub-blocks processed in shared memory, reducing global memory traffic.
- **Fused kernels**: Multi-round FRI folding and Poseidon2 full permutations avoid intermediate buffer round-trips.
- **Signed-digit MSM**: Scalar recoding halves bucket count, reducing bucket accumulation work.
- **GLV endomorphism**: BN254's efficient endomorphism splits 256-bit scalar muls into two 128-bit half-width muls.
- **C CIOS field arithmetic**: Hot-path 256-bit Montgomery multiplication uses C `__uint128_t` compiled with `-O3`, which is 29-38× faster than Swift for BN254 Fr carry chains. Used in CPU NTT, MSM, IPA, KZG, and ECDSA.
- **Small-input fast path**: MSM automatically routes to multi-threaded C Pippenger for small inputs (BN254 n≤2048, secp256k1 n≤1024) to avoid GPU dispatch overhead.

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
| **Parallel CPU** | GCD multithreaded implementations | Cross-checked against vanilla CPU for NTT (Fr, Bb, Gl), MSM, batch hash, Merkle |
| **NEON BabyBear** | C/ARM NEON Montgomery NTT (4-wide SIMD, Plonky3 technique) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **C Goldilocks** | Optimized C NTT (`__uint128_t` mul pipelining) | Cross-checked against vanilla cpuNTT + round-trip verification |
| **C BN254 Fr** | Fully unrolled 4-limb CIOS Montgomery NTT | Cross-checked against vanilla cpuNTT + round-trip verification |

Every benchmark run includes correctness checks (printed as PASS/FAIL). The test suite (`swift test`) covers field arithmetic, curve operations, and NTT correctness.

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization. To continue tuning for your hardware or workload, install the skill and run `/floptimizer` in a Claude Code session from this repo.
