# zkMetal Source Code Guide

This directory contains 66 subdirectories organizing the zkMetal implementation.

## By Functional Area

### Mathematical Foundations
| Directory | Description |
|-----------|-------------|
| `Fields/` | Finite field implementations (BN254, BLS12-377/381, Goldilocks, BabyBear, Stark252, Ed25519, secp256k1, Pallas/Vesta, Mersenne31, Binary Tower) |
| `Curve/` | Elliptic curve definitions and pairings (BN254, BLS12-381, Ed25519, BabyJubjub, Grumpkin, Pallas, Vesta, Secp256k1) |
| `BLS12_381/` | BLS12-381 specific aggregation and signature engines |
| `BinaryTower/` | Binary tower field extension (merged from top-level) |

### Core Primitives
| Directory | Description |
|-----------|-------------|
| `MSM/` | Multi-scalar multiplication engines (BLS12-377, BN254, Ed25519, Grumpkin, Pallas, secp256k1, Vesta) |
| `NTT/` | Number theoretic transform implementations (BLS12-377, BabyBear, Goldilocks, Circle, Pallas, Vesta, Stark252) |
| `Hash/` | Hash functions: Poseidon, Poseidon2, Pedersen, Keccak, SHA256, Blake3, Groestl, XHash |
| `Sort/` | Radix sort engine |

### Polynomial & Commitment
| Directory | Description |
|-----------|-------------|
| `Polynomial/` | Polynomial operations: FRI, sumcheck, multilinear polynomials, circle FRI, STIR, WHIR, erasure coding |
| `Commitment/` | Generic vector commitments, Pedersen, inner product arguments, bulletproofs, Verkle proofs |
| `KZG/` | KZG polynomial commitments, batch openings, fflonk |
| `IPA/` | Inner Product Argument engine |
| `Zeromorph/` | Zeromorph polynomial commitment |
| `FRI/` | FRI protocol GPU implementations |
| `PCS/` | Polynomial Commitment Scheme factory |

### Proof Systems
| Directory | Description |
|-----------|-------------|
| `Plonk/` | Plonk proof system, custom gates, lookup gates |
| `Halo2/` | Halo2 proof system backend |
| `Groth16/` | Groth16 proof system, aggregate provers |
| `Marlin/` | Marlin proof system |
| `Spartan/` | Spartan SNARK implementations |
| `Basefold/` | Basefold proof system |
| `Binius/` | Binius STARK system |
| `Brakedown/` | Brakedown proof system |
| `STARK/` | STARK proving/verification (BabyBear, Goldilocks, Stark252) |
| `CircleSTARK/` | Circle STARK specific implementations |
| `STIR/` | STIR proximity testing protocol |
| `WHIR/` | WHIR proximity testing protocol |

### Folding & IVC
| Directory | Description |
|-----------|-------------|
| `Folding/` | Folding schemes: Nova, Supernova, HyperNova, Protogalaxy, CCS |
| `IVC/` | Incremental Verifiable Computation engines |
| `Recursion/` | Recursive verifier circuits, CycleFold, gadgets |
| `Recursive/` | Recursive composition, proof aggregation |

### Lookup Arguments
| Directory | Description |
|-----------|-------------|
| `Lookup/` | Lookup arguments: Lasso, LogUp, Plookup, CQ, range proofs |

### Constraint Systems & Witnesses
| Directory | Description |
|-----------|-------------|
| `Constraint/` | R1CS constraint compilation, GPU circuit compilation |
| `R1CS/` | R1CS solver and witness generation |
| `AIR/` | Algebraic Intermediate Representation compilation |
| `Witness/` | Witness generation engines |
| `WitnessGen/` | GPU witness generation |

### Cryptographic Schemes
| Directory | Description |
|-----------|-------------|
| `Signature/` | BLS signatures, Schnorr signatures |
| `ECDSA/` | ECDSA/EdDSA implementations |
| `Lattice/` | Dilithium, Kyber (CRYSTALS), lattice NTT |

### Virtual Machines
| Directory | Description |
|-----------|-------------|
| `Jolt/` | Jolt RISC-V executor |
| `VM/` | zkVM implementations (Cairo, RISC-V) |

### Infrastructure
| Directory | Description |
|-----------|-------------|
| `Serialization/` | Proof serialization (SSZ, snarkjs, Ethereum ABI) |
| `Transcript/` | Fiat-Shamir transcripts |
| `Verifier/` | General verifiers |
| `Engine/` | Generic GPU batch operations |
| `CPU/` | CPU-parallel operations |
| `DataParallel/` | Data-parallel proving infrastructure |

### Data Availability
| Directory | Description |
|-----------|-------------|
| `DAS/` | Data Availability Sampling |
| `ErasureCoding/` | Reed-Solomon erasure coding |

### Experimental/New
| Directory | Description |
|-----------|-------------|
| `HE/` | Homomorphic Encryption engine |
| `WebGPU/` | WebGPU engine and WGSL codegen |
| `TensorProof/` | Tensor proof system |
| `Sumcheck/` | Sumcheck protocol implementations |
| `GKR/` | Goldwasser-Kalai-Rothblum MPC protocol |
| `Plonky2/` | Plonky2 recursive verifier |
| `Circom/` | Circom proof parsing |

### Proof Aggregation
| Directory | Description |
|-----------|-------------|
| `Aggregation/` | Cross-scheme batch verification, Groth16 aggregation |
| `Proof/` | General proof aggregation |
| `PCS/` | Polynomial Commitment Scheme factory |

## Naming Conventions

- **GPU-prefix files**: GPU implementations mixed with CPU files in same directories
  - Example: `GPUFRIEngine.swift` alongside `FRIEngine.swift` in `Polynomial/`
- **Engine suffix**: Most implementations use `*Engine` naming
  - Example: `NTTEngine.swift`, `Poseidon2Engine.swift`
- **Curve-specific suffixes**: `BN254*`, `BLS12381*`, `Secp256k1*`, etc.

## Key Files

- `zkMetal.swift` — Main module entry point
- `Versions.swift` — Version constants for all engines
- `Tuning.swift` — Auto-tuning parameters
