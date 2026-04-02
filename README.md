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
| 2^10 | 3.2ms | 4.6ms | 1.4x |
| 2^12 | 10ms | 24ms | 2x |
| 2^14 | 8.4ms | 99ms | **12x** |
| 2^16 | 6.2ms | 508ms | **82x** |
| 2^18 | 35ms | 2.2s | **63x** |
| 2^20 | 85ms | - | |
| 2^22 | 800ms | - | |

NTT is also available for Goldilocks (116ms at 2^24) and BabyBear (285ms at 2^24).

### Hashing

| Primitive | Batch Size | GPU | CPU (single-core) | Speedup |
|-----------|-----------|-----|-------|---------|
| Poseidon2 | 2^14 | 0.5 µs/hash | 140 µs/hash | **280x** |
| Poseidon2 | 2^16 | 0.2 µs/hash | 140 µs/hash | **730x** |
| Keccak-256 | 2^14 | 0.06 µs/hash | 9.7 µs/hash | **170x** |
| Keccak-256 | 2^16 | 0.09 µs/hash | 9.7 µs/hash | **108x** |
| Keccak-256 | 2^18 | 0.07 µs/hash | 9.7 µs/hash | **140x** |

### Merkle Trees

| Backend | Leaves | GPU |
|---------|--------|-----|
| Poseidon2 | 2^14 | 49ms |
| Poseidon2 | 2^16 | 68ms |
| Keccak-256 | 2^14 | 19ms |
| Keccak-256 | 2^16 | 21ms |

### FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 1.3ms | 15ms | **11x** |
| 2^16 | 2.2ms | 68ms | **31x** |
| 2^18 | 5.2ms | 205ms | **39x** |
| 2^20 | 39ms | 639ms | **17x** |

Full fold-to-constant: 2^20 in 17ms (20 rounds, fused 4-round kernels).

### Sumcheck (BN254 Fr)

| Variables | GPU | CPU | Speedup |
|-----------|-----|-----|---------|
| 2^14 | 6ms | 21ms | **3x** |
| 2^16 | 13ms | 82ms | **6x** |
| 2^18 | 23ms | 408ms | **18x** |
| 2^20 | 42ms | 2.4s | **57x** |

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
```

### MSM CLI

```bash
# Benchmark
swift run -c release zkmsm --bench 65536

# Compute from JSON
echo '{"points": [["0x1","0x2"]], "scalars": ["0x2a"]}' | swift run -c release zkmsm
```

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
