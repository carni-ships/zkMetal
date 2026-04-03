# zkMetal

GPU-accelerated zero-knowledge cryptography primitives for Apple Silicon, written in Metal and Swift.

## Primitives

| Primitive | Description |
|-----------|-------------|
| **MSM** | Multi-scalar multiplication (Pippenger + signed-digit + GLV endomorphism) |
| **NTT** | Number theoretic transform (four-step FFT with fused sub-blocks) |
| **Poseidon2** | Algebraic hash function (t=3, BN254 Fr) |
| **Keccak-256** | SHA-3 hash (fused subtree Merkle) |
| **Merkle Trees** | Poseidon2 and Keccak-256 backends |
| **FRI** | Fast Reed-Solomon IOP folding (fused 2/4-round kernels) |
| **Sumcheck** | Interactive sumcheck protocol (fused round+reduce with SIMD shuffles) |
| **KZG** | Polynomial commitment scheme (commit + open via MSM) |
| **Blake3** | BLAKE3 hash (batch hashing, Merkle-ready) |
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees |

## Performance

All benchmarks measured on Apple M3 Pro, comparing GPU (Metal) vs single-threaded CPU (Swift).
Run `swift run -c release zkbench all` to reproduce.

### MSM (BN254 G1)

| Points | GPU (Metal) |
|--------|-------------|
| 2^8 | 46ms |
| 2^10 | 41ms |
| 2^12 | 76ms |
| 2^14 | 134ms |
| 2^16 | 155ms |
| 2^17 | 240ms |
| 2^18 | 412ms |

**Comparison to other Metal GPU implementations (BN254, Apple Silicon):**

| Points | zkMetal (M3 Pro) | [ICICLE-Metal](https://github.com/ingonyama-zk/icicle) (M3) | [MoPro v2](https://github.com/zkmopro/gpu-acceleration) (M3) |
|--------|-----------------|-------------|-----------|
| 2^16 | **155ms** | — | 253ms |
| 2^18 | **412ms** | 149ms | 678ms |
| 2^20 | — | 421ms | 1,702ms |

*ICICLE-Metal and MoPro numbers from [MoPro blog](https://zkmopro.org/blog/metal-msm-v2/) on M3 MacBook Air.*

**Comparison to CPU and CUDA (BN254, 2^16 points):**

| Implementation | Hardware | Time |
|----------------|----------|------|
| Ingonyama ICICLE (CUDA) | RTX 3090 Ti | ~9ms |
| Arkworks (Rust, multithreaded) | M3 CPU | 69ms |
| zkMetal (this) | M3 Pro Metal GPU | 155ms |

Metal GPU MSM is currently **slower than optimized multithreaded CPU** for BN254. The fundamental bottleneck is that 256-bit field arithmetic requires 8x32-bit limbs on Metal (no native 64-bit integer multiply), while CPU implementations use 4x64-bit limbs with hand-tuned assembly, out-of-order execution, and deep pipelines. CUDA GPUs (like those targeted by [Ingonyama's ICICLE](https://github.com/ingonyama-zk/icicle)) have native 64-bit integer multiply, giving them a ~20x advantage over Metal. GPU MSM on Metal would become competitive for smaller fields (Goldilocks, BabyBear) where the arithmetic fits native GPU word sizes.

GPU scaling is strongly sublinear: 1024x more points (2^8 to 2^18) costs only ~9x more time, as fixed GPU overhead dominates at small sizes.

### NTT (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^10 | 2.9ms | 3.7ms | 1.3x |
| 2^12 | 3.2ms | 19ms | **6x** |
| 2^14 | 6.8ms | 88ms | **13x** |
| 2^16 | 15ms | 679ms | **47x** |
| 2^18 | 22ms | 1.7s | **79x** |
| 2^20 | 74ms | 7.3s | **99x** |
| 2^22 | 285ms | 31s | **109x** |

NTT is also available for BLS12-377 (339ms at 2^22), Goldilocks (249ms at 2^24), and BabyBear (262ms at 2^24).

GPU scales sublinearly (O(n log n) algorithm, but GPU utilization improves with size): 2^10 to 2^22 is 4096x more data for ~100x more time. CPU scales linearly with n log n. Speedup grows with input size. No other Metal NTT implementations are known for comparison; CUDA NTT (ICICLE) reports 320x improvement over SnarkJS at 2^22, though that baseline is JavaScript.

### Hashing

| Primitive | Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----------|-----|-------|---------|
| Poseidon2 | 2^14 | 0.5 µs/hash | 113 µs/hash | **240x** |
| Poseidon2 | 2^16 | 0.3 µs/hash | 113 µs/hash | **340x** |
| Poseidon2 | 2^18 | 1.2 µs/hash | 113 µs/hash | **97x** |
| Poseidon2 | 2^20 | 1.1 µs/hash | 113 µs/hash | **103x** |
| Keccak-256 | 2^14 | 0.18 µs/hash | 6.4 µs/hash | **35x** |
| Keccak-256 | 2^16 | 0.05 µs/hash | 6.4 µs/hash | **140x** |
| Keccak-256 | 2^18 | 0.04 µs/hash | 6.4 µs/hash | **180x** |
| Keccak-256 | 2^20 | 0.04 µs/hash | 6.4 µs/hash | **160x** |

GPU per-hash cost is roughly constant across batch sizes (linear scaling), while CPU per-hash cost is constant by definition. Speedup peaks at 2^16--2^18 where GPU occupancy is saturated without memory pressure. No other Metal implementations of Poseidon2 or Keccak-256 batch hashing are known.

### Merkle Trees

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 30ms | 126ms | **4x** |
| Poseidon2 | 2^12 | 41ms | 487ms | **12x** |
| Poseidon2 | 2^14 | 75ms | 1.9s | **26x** |
| Poseidon2 | 2^16 | 122ms | 7.9s | **65x** |
| Poseidon2 | 2^18 | 394ms | 30s | **77x** |
| Poseidon2 | 2^20 | 1.5s | 2.3min | **90x** |
| Keccak-256 | 2^12 | 8ms | 25ms | **3x** |
| Keccak-256 | 2^14 | 16ms | 101ms | **6x** |
| Keccak-256 | 2^16 | 17ms | 390ms | **23x** |
| Keccak-256 | 2^18 | 39ms | 1.5s | **40x** |
| Keccak-256 | 2^20 | 155ms | 6.2s | **42x** |

Both GPU and CPU scale linearly (O(n) tree construction). GPU speedup grows with size as fixed dispatch overhead is amortized -- Poseidon2 reaches 90x at 2^20, while Keccak plateaus around 42x due to its simpler per-hash arithmetic offering less GPU parallelism advantage. No other Metal Merkle tree implementations are known.

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 3.8ms | 10ms | **3x** |
| 2^16 | 2.9ms | 37ms | **13x** |
| 2^18 | 7.4ms | 137ms | **18x** |
| 2^20 | 22ms | 578ms | **26x** |
| 2^22 | 36ms | 2.2s | **61x** |

Full fold-to-constant: 2^20 in 32ms (20 rounds, fused 4-round kernels).

GPU scales sublinearly: 2^14 to 2^22 is 256x more data for ~10x more time, as each folding round halves the domain. CPU scales linearly. Speedup grows from 3x to 61x. No other Metal FRI implementations are known.

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 5.2ms | 21ms | **4x** |
| 2^16 | 14ms | 85ms | **6x** |
| 2^18 | 27ms | 328ms | **12x** |
| 2^20 | 46ms | 1.3s | **29x** |
| 2^22 | 84ms | 5.1s | **61x** |

GPU scales sublinearly: 2^14 to 2^22 is 256x more variables for ~16x more time. CPU scales linearly. Each sumcheck round reduces the problem by half, and fused round+reduce kernels keep GPU utilization high. No other Metal sumcheck implementations are known; ICICLE (CUDA) offers GPU sumcheck but no published comparison numbers.

### Polynomial Ops (BN254 Fr)

| Operation | Size | GPU |
|-----------|------|-----|
| Multiply (NTT) | deg 2^10 | 32ms |
| Multiply (NTT) | deg 2^12 | 41ms |
| Multiply (NTT) | deg 2^14 | 50ms |
| Multiply (NTT) | deg 2^16 | 67ms |
| Multi-eval (Horner) | deg 2^10, 1024 pts | 19ms |
| Multi-eval (Horner) | deg 2^12, 4096 pts | 104ms |
| Multi-eval (Horner) | deg 2^14, 16384 pts | 1.5s |

Polynomial multiplication uses NTT under the hood. Multi-point evaluation uses GPU Horner's method (one thread per evaluation point). Subproduct-tree evaluation is available but currently slower than Horner for these sizes due to high constant factors.

### KZG Commitments (BN254 G1)

| Operation | Size | GPU |
|-----------|------|-----|
| Commit | deg 2^8 | 35ms |
| Commit | deg 2^10 | 65ms |
| Open (commit + witness) | deg 2^8 | 29ms |
| Open (commit + witness) | deg 2^10 | 70ms |

KZG performance is MSM-dominated. Commit and open are thin wrappers around MSM + polynomial division.

### Blake3 Hashing

| Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----|-------|---------|
| 2^14 | 0.87 µs/hash | 0.6 µs/hash | 0.7x |
| 2^16 | 0.14 µs/hash | 0.6 µs/hash | **4x** |
| 2^18 | 0.07 µs/hash | 0.6 µs/hash | **9x** |
| 2^20 | 0.02 µs/hash | 0.6 µs/hash | **30x** |

Blake3 is much simpler than Keccak (7 rounds of 32-bit ARX ops vs 24 rounds of 64-bit Keccak-f). GPU speedup scales with batch size as fixed dispatch overhead amortizes. CPU Blake3 is very fast (0.6µs) so GPU only wins at large batch sizes.

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
