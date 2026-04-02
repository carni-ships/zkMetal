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
| **Radix Sort** | LSD radix-256 GPU sort |

## Performance (Apple M3 Pro)

### MSM (BN254 G1)

| Points | Time |
|--------|------|
| 65,536 | ~12ms |

### NTT

| Size | BN254 Fr | Goldilocks | BabyBear |
|------|----------|------------|----------|
| 2^14 | 11ms | 7ms | 2ms |
| 2^16 | 26ms | 9ms | 1ms |
| 2^18 | 56ms | 16ms | 16ms |
| 2^20 | 134ms | 54ms | 101ms |
| 2^22 | 405ms | 221ms | 260ms |
| 2^24 | - | 116ms | 285ms |

### Hashing

| Primitive | Batch Size | Time | Throughput |
|-----------|-----------|------|------------|
| Poseidon2 | 2^14 | 5.1ms | 3.2M hash/s |
| Poseidon2 | 2^16 | 21ms | 3.1M hash/s |
| Keccak-256 | 2^14 | 2.3ms | 7.1M hash/s |
| Keccak-256 | 2^16 | 5.7ms | 11.5M hash/s |
| Keccak-256 | 2^18 | 13ms | 19.7M hash/s |

### Merkle Trees

| Backend | Leaves | Time |
|---------|--------|------|
| Poseidon2 | 2^14 | 49ms |
| Poseidon2 | 2^16 | 68ms |
| Keccak-256 | 2^14 | 19ms |
| Keccak-256 | 2^16 | 21ms |

### FRI Folding (BN254 Fr)

| Size | Single Fold | Full Protocol (fold to 1) |
|------|-------------|--------------------------|
| 2^16 | 2.0ms | 8.0ms |
| 2^18 | 9.7ms | 10.5ms |
| 2^20 | 24ms | 19ms |

### Sumcheck (BN254 Fr)

| Variables | Time |
|-----------|------|
| 2^14 | 4.7ms |
| 2^16 | 9.9ms |
| 2^18 | 17ms |
| 2^20 | 32ms |

### Radix Sort

| Elements | Time | Throughput |
|----------|------|------------|
| 65,536 | 12ms | 5M keys/s |
| 262,144 | 40ms | 7M keys/s |
| 1,048,576 | 116ms | 9M keys/s |

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
