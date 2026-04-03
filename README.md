# zkMetal

GPU-accelerated zero-knowledge cryptography primitives for Apple Silicon, written in Metal and Swift.

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

## Performance

All benchmarks measured on Apple M3 Pro, comparing GPU (Metal) vs single-threaded CPU (Swift).
Run `swift run -c release zkbench all` to reproduce.

### MSM (BN254 G1)

| Points | GPU (Metal) | CPU (single-threaded) | Speedup |
|--------|-------------|----------------------|---------|
| 2^8 | 8ms | 428ms | **51x** |
| 2^10 | 9ms | 1.7s | **194x** |
| 2^12 | 14ms | 6.9s | **484x** |
| 2^14 | 24ms | 32.6s | **1345x** |
| 2^16 | 37ms | ~2.2min* | ~3500x |
| 2^17 | 68ms | — | — |
| 2^18 | 102ms | — | — |
| 2^20 | 294ms | — | — |

\* Extrapolated from measured 2^14 = 32.6s (sequential double-and-add). CPU times for 2^17+ would exceed 10 minutes.

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

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^10 | 0.32ms | 4ms | **12x** |
| 2^12 | 0.43ms | 18ms | **41x** |
| 2^14 | 0.49ms | 100ms | **206x** |
| 2^16 | 0.95ms | 369ms | **387x** |
| 2^18 | 1.9ms | 1.6s | **856x** |
| 2^20 | 6.1ms | 7.2s | **1184x** |
| 2^22 | 26ms | 31s | **1194x** |
| 2^24 | 113ms | 140s | **1237x** |

**Multi-field NTT comparison (GPU, with CPU baselines):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.65ms (369ms CPU, **557x**) | 1.8ms | 0.15ms (2.6ms CPU, **17x**) | 0.17ms (2.1ms CPU, **12x**) |
| 2^18 | 1.7ms (1.6s CPU, **971x**) | 2.1ms | 0.21ms (12ms CPU, **55x**) | 0.26ms (10ms CPU, **38x**) |
| 2^20 | 6.1ms (7.2s CPU, **1184x**) | 6.3ms | 0.84ms (52ms CPU, **62x**) | 1.2ms (44ms CPU, **36x**) |
| 2^22 | 27ms (31s CPU, **1194x**) | 26ms | 4.4ms | 2.9ms |
| 2^24 | 116ms (140s CPU, **1237x**) | — | 3.0ms | 2.0ms |

Smaller fields see dramatic throughput gains: BabyBear NTT at 2^24 (16M elements) runs in **2ms** — one element per 0.12ns, or **8.5B elements/sec**. The GPU advantage for small fields comes from native 32-bit arithmetic (1 mul per element vs 64 muls for BN254 CIOS), 8x higher memory density, and better threadgroup utilization. Note: CPU baselines for BabyBear and Goldilocks are already fast (single-threaded), so GPU speedups are lower (12-62x) compared to BN254 (557-1237x) where CPU field arithmetic is much more expensive.

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

| Primitive | Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----------|-----|-------|---------|
| Poseidon2 | 2^14 | 0.17 µs/hash | 280 µs/hash | **1647x** |
| Poseidon2 | 2^16 | 0.14 µs/hash | 280 µs/hash | **2000x** |
| Poseidon2 | 2^18 | 0.13 µs/hash | 280 µs/hash | **2154x** |
| Poseidon2 | 2^20 | 0.11 µs/hash | 280 µs/hash | **2545x** |
| Keccak-256 | 2^14 | 0.035 µs/hash | 9 µs/hash | **257x** |
| Keccak-256 | 2^16 | 0.027 µs/hash | 9 µs/hash | **333x** |
| Keccak-256 | 2^18 | 0.012 µs/hash | 9 µs/hash | **750x** |
| Keccak-256 | 2^20 | 0.011 µs/hash | 9 µs/hash | **818x** |

GPU per-hash cost is roughly constant across batch sizes (linear scaling), while CPU per-hash cost is constant by definition. Speedup peaks at 2^16--2^18 where GPU occupancy is saturated without memory pressure. No other Metal implementations of Poseidon2 or Keccak-256 batch hashing are known.

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
| 2^14 | 0.94ms | 21ms | **22x** |
| 2^16 | 1.3ms | 83ms | **62x** |
| 2^18 | 1.9ms | 351ms | **186x** |
| 2^20 | 8.3ms | 1.3s | **161x** |
| 2^22 | 16ms | 5.3s | **333x** |

GPU scales sublinearly: 2^14 to 2^22 is 256x more variables for ~17x more time. CPU scales linearly. Each sumcheck round reduces the problem by half, and fused round+reduce kernels keep GPU utilization high. No other Metal sumcheck implementations are known; ICICLE (CUDA) offers GPU sumcheck but no published comparison numbers.

### Polynomial Ops (BN254 Fr)

| Operation | Size | GPU |
|-----------|------|-----|
| Multiply (NTT) | deg 2^10 | 1.5ms |
| Multiply (NTT) | deg 2^12 | 1.8ms |
| Multiply (NTT) | deg 2^14 | 3.5ms |
| Multiply (NTT) | deg 2^16 | 6.6ms |
| Multi-eval (Horner) | deg 2^10, 1024 pts | 1.7ms |
| Multi-eval (Horner) | deg 2^12, 4096 pts | 8.6ms |
| Multi-eval (Horner) | deg 2^14, 16384 pts | 115ms |

Polynomial multiplication uses NTT under the hood. Multi-point evaluation uses GPU Horner's method (one thread per evaluation point). Subproduct-tree evaluation is available but currently slower than Horner for these sizes due to high constant factors.

### KZG Commitments (BN254 G1)

| Operation | Size | GPU |
|-----------|------|-----|
| Commit | deg 2^8 | 9ms |
| Commit | deg 2^10 | 15ms |
| Open (commit + witness) | deg 2^8 | 5ms |
| Open (commit + witness) | deg 2^10 | 10ms |

KZG performance is MSM-dominated. Commit and open are thin wrappers around MSM + polynomial division.

### Blake3 Hashing

| Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----|-------|---------|
| 2^14 | 0.011 µs/hash | 0.6 µs/hash | **55x** |
| 2^16 | 0.004 µs/hash | 0.6 µs/hash | **150x** |
| 2^18 | 0.004 µs/hash | 0.6 µs/hash | **150x** |
| 2^20 | 0.004 µs/hash | 0.6 µs/hash | **150x** |

Blake3 is much simpler than Keccak (7 rounds of 32-bit ARX ops vs 24 rounds of 64-bit Keccak-f). GPU speedup scales with batch size as fixed dispatch overhead amortizes. CPU Blake3 is very fast (0.6µs) so GPU only wins at large batch sizes.

### Theoretical Performance Analysis

How close each primitive is to the hardware floor (M3 Pro: ~3.6 TFLOPS, ~150 GB/s bandwidth), ranked by optimization opportunity:

| Rank | Primitive | Current | Theoretical Floor | Bottleneck | Headroom |
|------|-----------|---------|-------------------|------------|----------|
| 1 | MSM BN254 2^18 | 173ms | ~5ms (scatter BW) | Random-access BW | ~35x |
| 2 | FRI Fold 2^20 | 16ms | 0.32ms (BW) | Bandwidth | ~50x |
| 3 | Blake3 Batch 2^20 | 21ms | 0.64ms (BW) | Bandwidth | ~33x |
| 4 | P2 Merkle 2^16 | 17ms | 0.62ms (compute) | Dispatch latency | ~27x |
| 5 | NTT BabyBear 2^24 | 37ms | 1.71ms (BW) | Bandwidth | ~22x |
| 6 | NTT Goldilocks 2^24 | 37ms | 1.79ms (compute) | Compute ≈ BW | ~21x |
| 7 | Keccak Batch 2^18 | 8ms | 0.52ms (compute) | Compute | ~15x |
| 8 | P2 Batch 2^16 | 9ms | 0.62ms (compute) | Compute | ~14.5x |
| 9 | Sumcheck 2^20 | 10ms | 0.85ms (BW) | Bandwidth | ~12x |
| 10 | NTT BN254 2^22 | 365ms | 2.87ms (compute) | Compute + strided BW | ~127x |

Notes: MSM's realistic floor accounts for scatter-gather inefficiency in bucket accumulation. Poseidon2 Merkle overhead is dominated by 16 sequential kernel dispatches (~0.5ms each). FRI fold headroom is inflated by single-fold API overhead (multiFold is ~4x from floor). BN254 NTT 2^22 uses extended four-step with strided column access (32KB per element) — the 127x gap is dominated by non-coalesced memory patterns for 32-byte Fr elements.

## Supported Fields

- **BN254 Fr** (scalar field) -- Montgomery CIOS, 8x32-bit limbs
- **BN254 Fp** (base field) -- Montgomery CIOS, SOS squaring
- **BLS12-377 Fr** (scalar field) -- Montgomery CIOS, 8x32-bit limbs, TWO_ADICITY=47
- **Goldilocks** (p = 2^64 - 2^32 + 1) -- native 64-bit reduction
- **BabyBear** (p = 2^31 - 2^27 + 1) -- 32-bit arithmetic

## Architecture

All compute runs on Metal GPU shaders. The Swift layer handles buffer management, pipeline dispatch, and host-device coordination.

```
Sources/
  Shaders/         # Metal GPU kernels
    fields/        # Field arithmetic (BN254 Fr/Fp, BLS12-377 Fr, Goldilocks, BabyBear)
    geometry/      # Elliptic curve operations (BN254 G1)
    msm/           # Multi-scalar multiplication kernels
    ntt/           # NTT butterfly + fused sub-block kernels
    hash/          # Poseidon2, Keccak-256, Blake3
    fri/           # FRI folding kernels
    sumcheck/      # Sumcheck round kernels
    poly/          # Polynomial evaluation/interpolation
  zkMetal/         # Swift engine layer
    Fields/        # CPU-side field arithmetic
    MSM/           # MSM engine (Pippenger, GLV, signed-digit)
    NTT/           # NTT engines (BN254, Goldilocks, BabyBear)
    Hash/          # Poseidon2, Keccak, Merkle tree engines
    Polynomial/    # FRI, Sumcheck engines
    Poly/          # Polynomial operations engine
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
swift run -c release zkbench all       # Everything
swift run -c release zkbench calibrate # Re-calibrate GPU parameters
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

## Correctness & Provenance

All GPU kernels are verified against CPU reference implementations. The CPU implementations use standard, well-known algorithms and externally-sourced parameters:

| Component | Source | Verification |
|-----------|--------|-------------|
| **BN254 curve** | Standard parameters (same as Ethereum/bn256) | Field arithmetic unit tests |
| **Poseidon2 constants** | [HorizenLabs reference](https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs) | Constants copied from official repo |
| **Keccak-256** | FIPS 202 (SHA-3) | Validated against NIST test vectors |
| **NTT** | Cooley-Tukey (DIT) / Gentleman-Sande (DIF) | Round-trip tests + root of unity verification |
| **FRI folding** | Standard FRI protocol | GPU vs CPU cross-check + multi-fold to constant |
| **Sumcheck** | Standard interactive protocol | Protocol-level verification (S(0)+S(1) = sum at each round) |
| **Goldilocks** | p = 2^64 - 2^32 + 1 (standard) | NTT round-trip + CPU cross-check |
| **BabyBear** | p = 2^31 - 2^27 + 1 (standard) | NTT round-trip + CPU cross-check |

Every benchmark run includes correctness checks (printed as PASS/FAIL). The test suite (`swift test`) covers field arithmetic, curve operations, and NTT correctness.

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization. To continue tuning for your hardware or workload, install the skill and run `/floptimizer` in a Claude Code session from this repo.
