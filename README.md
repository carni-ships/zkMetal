# zkMetal

> **⚠️ Experimental Software** — This library is under active development. Expect breaking changes, incomplete implementations, and undocumented behavior. API stability is not guaranteed. Use at your own risk.

GPU-accelerated zero-knowledge proof library for Apple Silicon. Metal compute shaders + C/NEON field arithmetic + Swift orchestration.

**~211 primitives** across 18 fields and 10 elliptic curves. 573 source files, 107 Metal shaders, 33 C/NEON files, 244 test files.

- **Core:** MSM (Pippenger+GLV), NTT (four-step FFT), Poseidon2/Keccak/Blake3/SHA-256, Merkle trees
- **Proof systems:** Plonk, HyperPlonk, Groth16, STARK (Circle/BabyBear/Goldilocks/Stark252), Spartan, Marlin, GKR
- **Commitments:** KZG, IPA, Basefold, Brakedown, Zeromorph, Verkle, Pedersen
- **Folding/IVC:** HyperNova, Protogalaxy, Nova/SuperNova
- **Lookup arguments:** LogUp, Lasso, CQ, Plookup, grand product
- **zkVM:** Jolt (RV32I), Cairo

## Contents

- [Primitives](#primitives) — all implemented primitives
- [Quick Start](#quick-start) — code examples
- [Architecture](#architecture) — system design
- [Building](#building) — compilation
- [Performance](PERFORMANCE.md) — detailed benchmarks
- [Documentation](docs/) — architecture, tuning, guides
- [GETTING_STARTED.md](GETTING_STARTED.md) — user guide

## Primitives

### Core Arithmetic
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **MSM** | GPU/CPU | Multi-scalar multiplication — BN254, BLS12-377, secp256k1, Pallas, Vesta, Ed25519, Grumpkin |
| **NTT** | GPU/CPU | Number theoretic transform — BN254, BLS12-377, Goldilocks, BabyBear, Stark252, Circle M31 |
| **Batch Field Ops** | GPU/CPU | Vectorized add/mul/sub/inverse, C CIOS Montgomery, auto-parallel |
| **Radix Sort** | GPU | 32-bit LSD radix sort (4-pass, 8-bit radix) |

### Hashing & Merkle Trees
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Poseidon2** | GPU | Algebraic hash — BN254, M31, BabyBear. Duplex sponge mode |
| **Keccak-256** | GPU/CPU | SHA-3 with NEON acceleration |
| **Blake3** | GPU/CPU | BLAKE3 with NEON acceleration |
| **Merkle Trees** | GPU | Poseidon2/Keccak/Blake3 backends |

### Polynomial & IOP
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **FRI** | GPU | Fast Reed-Solomon IOP — fold-by-2/4/8 |
| **STIR** | GPU | Shift-based proximity testing |
| **Sumcheck** | GPU | Interactive sumcheck — dense, sparse, univariate, multilinear |
| **Polynomial Ops** | GPU | NTT multiply, Horner multi-eval, division, interpolation |
| **Coset LDE** | GPU | Zero-pad + coset-shift |
| **GPU Additive FFT** | GPU | GF(2^8) additive FFT, all k levels in single dispatch |

### Polynomial Commitment Schemes
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **KZG** | GPU | Commit, open, batch open, multi-open, degree bounds |
| **IPA** | GPU/CPU | Bulletproofs-style inner product argument |
| **Basefold** | GPU | NTT-free multilinear PCS |
| **Brakedown** | GPU | Expander-based multilinear PCS |
| **Zeromorph** | GPU | Multilinear-to-univariate PCS reduction |

### Proof Systems
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Plonk** | GPU | Preprocessed polynomial IOP with KZG |
| **HyperPlonk** | GPU | Multilinear Plonk with sumcheck-based IOP |
| **Groth16** | GPU | zk-SNARK with BN254 pairings |
| **STARK** | GPU | Circle/BabyBear/Goldilocks/Stark252 |
| **GKR** | GPU | Goldwasser-Kalai-Rothblum interactive proof |
| **Spartan** | GPU | Transparent SNARK via multilinear extensions |
| **Marlin** | GPU | Preprocessed SNARK with KZG |

### Folding & IVC
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Nova/SuperNova** | GPU/CPU | IVC with cross-term folding |
| **HyperNova** | GPU | CCS folding scheme |
| **Protogalaxy** | GPU/CPU | Plonk-native folding |

### Curves & Signatures
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **BN254** | GPU/CPU | Full pairing engine (Fp/Fp2/Fp6/Fp12) |
| **BLS12-381** | CPU | Full tower, G1/G2, pairings, BLS signatures |
| **BLS12-377** | GPU/CPU | Scalar + base field, G1 MSM, NTT |
| **secp256k1** | GPU/CPU | ECDSA batch verification |
| **Ed25519** | GPU/CPU | EdDSA signatures |
| **Pasta** | GPU/CPU | Pallas/Vesta curve cycle |

### Infrastructure
| Primitive | Platform | Description |
|-----------|----------|-------------|
| **Transcript** | CPU | Fiat-Shamir — Poseidon2 + Keccak backends |
| **Proof Serialization** | CPU | BN254/BLS12-381 point compression, snarkjs JSON, EIP-4844 |

## Quick Start

```swift
import zkMetal

// MSM
let msm = try MetalMSM()
let result = try msm.msm(points: points, scalars: scalars)

// NTT
let ntt = try NTTEngine()
let transformed = try ntt.ntt(values)

// Poseidon2 hashing
let p2 = try Poseidon2Engine()
let hashes = try p2.hashBatch(inputs)

// FRI folding
let fri = try FRIEngine()
let folded = try fri.multiFold(evals: evaluations, betas: challenges)

// KZG
let kzg = try KZGEngine(srs: srs)
let commitment = try kzg.commit(polynomial)
```

### Benchmarks

```bash
swift run -c release zkbench all          # Everything
swift run -c release zkbench test         # Correctness tests
swift run -c release zkbench cpu          # CPU vs GPU comparison
swift run -c release zkbench all --no-cpu # GPU-only

# Core:       msm, ntt, p2, keccak, blake3, merkle, sort, poly
# Proofs:     plonk, groth16, gkr, circle-stark, fold
# PCS:        kzg, basefold, zeromorph, ipa
# Polynomial: fri, sumcheck, sparse
# Lookups:    lookup, lasso, cq
```

## Architecture

```
Sources/
  Shaders/         # Metal GPU kernels
    fields/        # Field arithmetic (BN254, BLS12-377/381, secp256k1, Goldilocks, BabyBear, M31)
    geometry/      # Elliptic curve operations
    msm/           # Multi-scalar multiplication kernels
    ntt/           # NTT butterfly + fused sub-block + Circle NTT
    hash/          # Poseidon2, Keccak-256, Blake3
    fri/           # FRI + Circle FRI folding kernels
    sumcheck/      # Sumcheck round kernels
    poly/          # Polynomial evaluation/interpolation
    sort/          # GPU radix sort kernels
    lattice/       # Kyber/Dilithium NTT kernels
  NeonFieldOps/    # C/ARM64 optimized CPU primitives
  zkMetal/         # Swift engine layer (66 subdirectories)
Tests/
  zkMetalTests/    # Correctness tests
```

See [docs/architecture.md](docs/architecture.md) for detailed Metal GPU dispatch patterns.

## Building

Requires macOS 13+ and Xcode with Metal support.

```bash
swift build -c release
swift test   # Run all tests
```

## Correctness & Testing

244 test files, 241 test suites. All GPU kernels verified against CPU reference implementations.

Filter tests: `.build/release/zkMetalTests pairing groth16 gpu`. Use `--list` to see all test names.

| Category | Verification |
|----------|-------------|
| Field arithmetic | Unit tests + cross-checks (arithmetic properties, inverses, distributivity) |
| MSM | GPU vs CPU cross-check, on-curve, determinism |
| NTT | Round-trip + CPU cross-check (all fields, sizes 2^2 through 2^22) |
| Hashing | Known-answer tests + GPU vs CPU batch |
| Polynomial protocols | S(0)+S(1)=sum, round-poly match, full protocol verify |
| Proof systems | Prove+verify, tampered proof rejection |
| Signatures | RFC 8032, batch verification |

## Design Decisions

- **Montgomery form everywhere**: All field elements stay in Montgomery representation on GPU.
- **Buffer caching**: GPU Metal buffers are cached and reused across calls.
- **Four-step FFT**: Large NTTs (>2^16) split into sub-blocks in shared memory.
- **C CIOS field arithmetic**: Hot-path 256-bit Montgomery multiplication using C `__uint128_t`.
- **Zero-copy Swift↔C bridge**: Field elements share memory layout with C `uint64_t[4]`.
- **Small-input fast path**: MSM routes to C Pippenger for small inputs (n<=2048) to avoid GPU dispatch overhead.

## Language Bindings

| Language | Location | Description |
|----------|----------|-------------|
| **Rust** | `bindings/rust/` | `zkmetal-sys` crate — safe wrappers |
| **Go** | `bindings/go/` | cgo package — GPU MSM/NTT/hash/pairing/FRI |
| **C++** | `bindings/barretenberg/` | CMake + bridge headers |
| **WebGPU** | `Sources/zkMetal/WebGPU/` | WGSL shader codegen |

---

Optimized with [floptimizer](https://github.com/carni-ships/floptimizer).
