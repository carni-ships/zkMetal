# Metal Shaders Guide

This directory contains 25 subdirectories organizing Metal GPU shaders.

## Directory Map

| Directory | Description |
|-----------|-------------|
| `fields/` | Field arithmetic (BN254, BLS12-377/381, secp256k1, Goldilocks, BabyBear, M31, Pallas, Vesta, binary tower) |
| `geometry/` | Elliptic curve operations (BN254 G1, BLS12-377 G1, secp256k1, Pallas, Vesta) |
| `msm/` | Multi-scalar multiplication kernels (bucket method, GLV) |
| `ntt/` | NTT butterfly + fused sub-block + Circle NTT kernels |
| `hash/` | Poseidon2 (BN254 + M31), Keccak-256, Blake3, SHA-256 |
| `fri/` | FRI + Circle FRI folding kernels |
| `sumcheck/` | Sumcheck round kernels |
| `poly/` | Polynomial evaluation/interpolation kernels |
| `sort/` | GPU radix sort kernels (4-pass, 8-bit radix) |
| `constraint/` | Fused NTT+constraint kernels |
| `basefold/` | Basefold fold kernels |
| `lattice/` | Kyber/Dilithium NTT kernels |
| `erasure/` | Reed-Solomon erasure coding |
| `witness/` | GPU witness trace evaluation |
| `additive/` | Additive FFT over GF(2^8) |
| `brakedown/` | Brakedown polynomial commitment |
| `pairing/` | BN254/BLS12-381 pairing operations |
| `reduction/` | Reduction operations |
| `scan/` | Prefix scan kernels |
| `verify/` | Verification kernels |
| `he/` | Homomorphic encryption NTT |
| `utility/` | Shared utility kernels |

## Naming Conventions

- **cooperative**: SIMD shuffle-based cooperative reduction
- **shared**: Shared memory parallel reduction
- **fused**: Multi-kernel operations fused into single dispatch
- **batch**: Batch processing of multiple items
- **neon**: NEON SIMD accelerated

## Key Patterns

### Cooperative Reduction
Threads within a SIMD group cooperate using `simd_shuffle_xor` for tree reduction.

### Shared Memory
Threadgroups use shared memory for intermediate results before writing to global memory.

### Fused Kernels
Multiple passes fused into a single dispatch to avoid memory round-trips.

## Folders with Most Activity

- `fields/` — largest directory, field arithmetic for all curves
- `ntt/` — butterfly and FFT kernels
- `hash/` — Poseidon2 and Keccak permutations
- `msm/` — bucket accumulation and reduction
