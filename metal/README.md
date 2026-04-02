# zkMetal

GPU-accelerated zero-knowledge cryptography primitives for Apple Silicon, written in Metal and Swift.

## Primitives

| Primitive | Description | Performance (M3 Pro) |
|-----------|-------------|---------------------|
| **MSM** | Multi-scalar multiplication (Pippenger + signed-digit + GLV endomorphism) | ~12ms / 65K points |
| **NTT** | Number theoretic transform (four-step FFT with fused sub-blocks) | ~28ms / 2^22 elements (BN254) |
| **Poseidon2** | Algebraic hash function (t=3, BN254 Fr) | ~9ms / 2^16 hashes |
| **Keccak-256** | SHA-3 hash (fused subtree Merkle optimization) | ~8ms / 2^18 hashes |
| **Merkle Trees** | Poseidon2 and Keccak-256 backends | ~17ms / 2^16 leaves (P2) |
| **FRI** | Fast Reed-Solomon IOP folding (fused 2/4-round kernels) | ~4ms / 2^20 fold-to-constant |
| **Sumcheck** | Interactive sumcheck protocol (fused round+reduce with SIMD shuffles) | ~10ms / 2^20 |
| **Polynomial Ops** | Evaluation, interpolation, subproduct trees | |
| **Radix Sort** | LSD radix-256 with segmented scatter | ~15ms / 4M elements |

## Supported Fields

- **BN254 Fr** (scalar field) — Montgomery CIOS, 8x32-bit limbs
- **BN254 Fp** (base field) — Montgomery CIOS, SOS squaring
- **Goldilocks** (p = 2^64 - 2^32 + 1) — native 64-bit reduction
- **BabyBear** (p = 2^31 - 2^27 + 1) — 32-bit arithmetic

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
    sort/          # GPU radix sort
  zkMetal/         # Swift engine layer
    Fields/        # CPU-side field arithmetic
    MSM/           # MSM engine (Pippenger, GLV, signed-digit)
    NTT/           # NTT engines (BN254, Goldilocks, BabyBear)
    Hash/          # Poseidon2, Keccak, Merkle tree engines
    Polynomial/    # FRI, Sumcheck engines
    Poly/          # Polynomial operations engine
    Sort/          # Radix sort engine
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

// Radix sort
let sort = try RadixSortEngine()
let sorted = try sort.sort(keys)
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
swift run -c release zkbench sort      # Radix sort
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
