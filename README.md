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
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees |

## Performance (Apple M3 Pro)

All benchmarks compare GPU (Metal) vs single-threaded CPU (Swift) on the same machine.
Run `swift run -c release zkbench all` to reproduce.

### MSM (BN254 G1)

| Points | GPU |
|--------|-----|
| 65,536 | ~12ms |

No single-threaded CPU comparison is provided -- a naive CPU MSM at 65K points takes minutes. For reference, Barretenberg's multithreaded Pippenger on the same hardware takes ~200ms for 200K points.

### NTT (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^10 | 2.9ms | 3.7ms | 1.3x |
| 2^12 | 3.2ms | 19ms | **6x** |
| 2^14 | 6.8ms | 88ms | **13x** |
| 2^16 | 15ms | 679ms | **47x** |
| 2^18 | 22ms | 1.7s | **79x** |
| 2^20 | 74ms | - | |
| 2^22 | 285ms | - | |

NTT is also available for Goldilocks (249ms at 2^24) and BabyBear (262ms at 2^24).

### Hashing

| Primitive | Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----------|-----|-------|---------|
| Poseidon2 | 2^14 | 0.5 µs/hash | 113 µs/hash | **240x** |
| Poseidon2 | 2^16 | 0.3 µs/hash | 113 µs/hash | **340x** |
| Keccak-256 | 2^14 | 0.18 µs/hash | 6.4 µs/hash | **35x** |
| Keccak-256 | 2^16 | 0.05 µs/hash | 6.4 µs/hash | **140x** |
| Keccak-256 | 2^18 | 0.04 µs/hash | 6.4 µs/hash | **180x** |

### Merkle Trees

| Backend | Leaves | GPU |
|---------|--------|-----|
| Poseidon2 | 2^14 | 44ms |
| Poseidon2 | 2^16 | 59ms |
| Keccak-256 | 2^14 | 4.3ms |
| Keccak-256 | 2^16 | 18ms |

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 3.8ms | 10ms | **3x** |
| 2^16 | 2.9ms | 37ms | **13x** |
| 2^18 | 7.4ms | 137ms | **18x** |
| 2^20 | 22ms | 578ms | **26x** |

Full fold-to-constant: 2^20 in 32ms (20 rounds, fused 4-round kernels).

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 5.2ms | 21ms | **4x** |
| 2^16 | 14ms | 85ms | **6x** |
| 2^18 | 27ms | 328ms | **12x** |
| 2^20 | 46ms | 1.3s | **29x** |

## Supported Fields

- **BN254 Fr** (scalar field) -- Montgomery CIOS, 8x32-bit limbs
- **BN254 Fp** (base field) -- Montgomery CIOS, SOS squaring
- **Goldilocks** (p = 2^64 - 2^32 + 1) -- native 64-bit reduction
- **BabyBear** (p = 2^31 - 2^27 + 1) -- 32-bit arithmetic

## Architecture

All compute runs on Metal GPU shaders. The Swift layer handles buffer management, pipeline dispatch, and host-device coordination.

```
Sources/
  Shaders/         # Metal GPU kernels
    fields/        # Field arithmetic (BN254 Fr/Fp, Goldilocks, BabyBear)
    geometry/      # Elliptic curve operations (BN254 G1)
    msm/           # Multi-scalar multiplication kernels
    ntt/           # NTT butterfly + fused sub-block kernels
    hash/          # Poseidon2, Keccak-256
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

```

### Benchmarks

```bash
swift run -c release zkbench msm       # MSM (65K points)
swift run -c release zkbench ntt       # NTT (all fields)
swift run -c release zkbench p2        # Poseidon2
swift run -c release zkbench keccak    # Keccak-256
swift run -c release zkbench merkle    # Merkle trees
swift run -c release zkbench fri       # FRI folding
swift run -c release zkbench sumcheck  # Sumcheck
swift run -c release zkbench all       # Everything
swift run -c release zkbench calibrate # Re-calibrate GPU parameters
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

## Optimization

These primitives were profiled and tuned using [floptimizer](https://github.com/carni-ships/floptimizer), a Claude Code skill for systematic GPU/CPU performance optimization. To continue tuning for your hardware or workload, install the skill and run `/floptimizer` in a Claude Code session from this repo.
