# zkMetal Primitive Dependency Graph

```
                        ┌─────────────────────────────────┐
                        │        FIELD ARITHMETIC          │
                        │  (BN254, BLS12-381, BabyBear,   │
                        │   Goldilocks, M31, secp256k1…)  │
                        └──────┬──────────┬───────────┬────┘
                               │          │           │
                    ┌──────────▼──┐  ┌────▼─────┐  ┌──▼──────────┐
                    │    NTT      │  │  CURVE   │  │    HASH     │
                    │             │  │  OPS     │  │ (P2/Keccak/ │
                    │             │  │(add/dbl/ │  │  Blake3/SHA) │
                    │             │  │ scalar)  │  │             │
                    └──┬──────┬──┘  └──┬───┬───┘  └──────┬──────┘
                       │      │        │   │             │
                  ┌────▼──┐   │   ┌────▼─┐ │        ┌────▼──────┐
                  │ POLY  │   │   │ MSM  │ │        │  MERKLE   │
                  │ OPS   │   │   │      │ │        │  TREE     │
                  └──┬──┬─┘   │   └──┬─┬─┘ │        └────┬──────┘
                     │  │     │      │ │   │             │
        ┌────────────┘  │     │      │ │   │             │
        │               │     │      │ │   │             │
   ┌────▼────┐     ┌────▼─────▼──────▼─▼───▼──┐    ┌────▼───┐
   │ SUMCHECK│     │      PAIRING              │    │  FRI   │
   │         │     │ (Fp2/Fp6/Fp12 tower +     │    │        │
   │         │     │  Miller loop + final exp)  │    │        │
   └──┬──┬───┘     └──────────┬────────────────┘    └──┬──┬──┘
      │  │                    │                        │  │
      │  │         ┌──────────▼──────────┐             │  │
      │  │         │       KZG           │◄────────────┘  │
      │  │         │ (commit=MSM,        │                │
      │  │         │  open=poly+MSM,     │                │
      │  │         │  verify=pairing)    │                │
      │  │         └───┬─────┬──────┬────┘                │
      │  │             │     │      │                     │
┌─────▼──▼──┐   ┌──────▼──┐ │ ┌────▼─────┐         ┌─────▼──────┐
│  LOOKUPS  │   │  PLONK  │ │ │  MARLIN  │         │   STARK    │
│(LogUp/    │   │         │ │ │          │         │(Circle/FRI)│
│Lasso/cq)  │   └─────────┘ │ └──────────┘         └────────────┘
└─────┬─────┘               │
      │              ┌──────▼──────┐
      │              │  GROTH16    │
      │              └─────────────┘
      │
┌─────▼──────────┐   ┌─────────┐   ┌──────────┐
│   JOLT zkVM    │   │  IPA    │   │ SPARTAN  │
│(Lasso+sumcheck)│   │(=MSM)  │   │(=sumcheck│
└────────────────┘   └────┬────┘   │ +MSM)    │
                          │        └──────────┘
                     ┌────▼────┐
                     │ VERKLE  │
                     │(=IPA)   │
                     └─────────┘
```

## Hub Primitives (most depended on)

### MSM — #1 most depended on
Used by: KZG commit & open, Groth16 prover (3 MSMs: [A],[B],[C]), Plonk prover (commitment phase), Marlin prover, Spartan (commitment), IPA (every round), Verkle trees (via IPA), ECDSA batch verify, HyperNova folding, BLS signatures (sign=scalar mul, verify=MSM)

### NTT — #2
Used by: all polynomial ops (eval, interpolation, multiply), KZG (poly arithmetic), Plonk (constraint eval in coset domain), Groth16 (H(x) polynomial), Marlin (AHP polynomials), FRI (low-degree extension), STARK (trace LDE)

### Pairing — gatekeeper for SNARKs
Without it: no KZG verification, no Groth16 verification, no BLS signature verification, Plonk/Marlin verification blocked

### Sumcheck — gatekeeper for modern protocols
Used by: Spartan, GKR, Lasso/Jolt, HyperNova, Marlin (univariate variant)

### Hash + Merkle — gatekeeper for STARKs
FRI commitments, STARK prover/verifier, data availability, Fiat-Shamir transcript

## Dependency Table for Primitives

| Primitive | Depends On | Status |
|---|---|---|
| ~~Ed25519~~ | Fp (2^255-19), curve, SHA-512 | **DONE** — Solinas Fp, EdDSA, GPU MSM |
| ~~BabyJubjub~~ | BN254 Fr | **DONE** — twisted Edwards, Pedersen, EdDSA |
| ~~SHA-256~~ | Nothing | **DONE** — GPU batch 119M/s, fused Merkle |
| ~~BLS Signatures~~ | BLS12-381 pairing, MSM | **DONE** — sign/verify/aggregate + hash-to-curve G2 |
| ~~Grumpkin~~ | BN254 Fr + Fq | **DONE** — GPU MSM with signed-digit |
| ~~Schnorr~~ | secp256k1, SHA-256 | **DONE** — sign/verify/batch |
| ~~Jubjub~~ | BLS12-381 Fr | **DONE** — twisted Edwards, scalar mul |
| ~~Stark252~~ | Nothing (new field) | **DONE** — field + C NTT shipped |
| Binary Tower (Binius) | New GF(2) tower | **IN PROGRESS** — PMULL-based arithmetic |
| GKR Protocol | Sumcheck | **IN PROGRESS** — layered circuit prover |
| Fiat-Shamir Transcript | Hash (Keccak/Poseidon2) | **IN PROGRESS** — reusable duplex sponge |

## Production Stack Coverage

How zkMetal maps to the major ZK stacks deployed in production today.

### Coverage Matrix

"Core" = all field arithmetic, curve ops, MSM, NTT, hash, commitment, and proving primitives needed for the stack's main prover/verifier pipeline. "Gap" = what would be needed for a drop-in replacement of the reference implementation.

| Production Stack | Used By | Required Primitives | zkMetal Coverage | Remaining Gap |
|-----------------|---------|-------------------|-----------------|---------------|
| **Plonky3 / SP1** | Succinct SP1 zkVM, Valida | BabyBear NTT, Poseidon2 (width-16 BB), FRI, AIR constraints | **Core complete** | VM instruction tables, memory checking argument |
| **Halo2 (PSE)** | Scroll, Taiko, PSE circuits | Pasta (Pallas/Vesta), IPA, Plonk + lookups | **Core complete** | Circuit-specific custom gates, Halo2 API compat |
| **Cairo / Stwo** | StarkNet, StarkWare | Stark252 field, Circle STARK over M31, Poseidon | **Core complete** | Cairo VM trace generation, bytecode interpreter |
| **RISC Zero** | RISC Zero zkVM | BabyBear, FRI, STARK, Poseidon2 | **Core complete** | RISC-V instruction circuit, continuations |
| **Jolt / Lasso** | a16z Labs | BN254 pairing, Lasso lookups, sumcheck | **Core complete** | Instruction decomposition tables, GKR (in progress) |
| **Barretenberg** | Aztec Network | BN254, Grumpkin, Plonk (UltraHonk), KZG | **Core complete** | Goblin Plonk recursive verifier, Protogalaxy |
| **Gnark** | Linea, ConsenSys | BN254/BLS12-381, Groth16, Plonk+KZG | **Core complete** | Constraint system compiler, witness solver |
| **Circom / Snarkjs** | Polygon zkEVM, Semaphore, Tornado Cash | BN254 Groth16, BabyJubjub, Poseidon | **Core complete** | R1CS constraint file parser, WASM witness gen |
| **Boojum** | zkSync Era (Matter Labs) | Goldilocks, Poseidon2, FRI, custom gates | **Core complete** | Boojum-specific custom gate set, GPU witness gen |
| **Kimchi** | Mina Protocol | Pasta curves, IPA, Plonk | **Core complete** | Kimchi-specific lookup arguments, recursion |
| **Ethereum Consensus** | All validators | BLS12-381 pairing, BLS signatures | **Core complete** | SSZ serialization, networking layer (out of scope) |

### Gaps Blocking Full Coverage

All core cryptographic primitives are shipped for every stack. Remaining gaps are stack-specific integration work (VM instruction tables, constraint compilers, wire format parsers) — not missing crypto.

### Recently Closed Gaps

- **BLS Signatures** — shipped (BLS12-381 sign/verify/aggregate, Ethereum consensus ready)
- **BLS12-381 C Pairing** — shipped (30× speedup: 78ms → 2.6ms via C Miller loop + final exp)
- **BN254 C Pairing** — shipped (Fp2/Fp6/Fp12 tower + Miller loop + final exponentiation in C)
- **Hash-to-curve G2** — shipped (RFC 9380, SSWU + 3-isogeny + cofactor clearing)
- **BLS12-377 / Stark252 C NTT** — shipped (Cooley-Tukey DIT forward, Gentleman-Sande DIF inverse, twiddle caching)
- **BGMW Fixed-Base MSM** — shipped (precomputed generator tables, multi-threaded, IPA/Pedersen acceleration)
- **Ed25519 C Acceleration** — shipped (Fq CIOS + Shamir's trick for EdDSA verify)
- **Poseidon2 BabyBear width-16** — shipped (SP1/Plonky3 exact config, 104M hash/s)
- **Stark252 field + NTT** — shipped (StarkNet native field, TWO_ADICITY=192, 238M elem/s)
- **BabyJubjub** — shipped (twisted Edwards over BN254 Fr, Pedersen hash, EdDSA)
- **SHA-256** — shipped (GPU batch hash 119M/s, fused Merkle subtree)
- **Grumpkin curve** — shipped (BN254 inner curve, GPU MSM with signed-digit)
- **Ed25519** — shipped (Curve25519, EdDSA with RFC 8032 test vectors, GPU MSM)

### What Full Coverage Looks Like

zkMetal ships all core cryptographic primitives for every major production ZK stack. The remaining work to reach drop-in replacement status is stack-specific integration (VM circuits, constraint compilers, wire formats) rather than missing crypto.

## What Unlocks What

Completing a hub primitive unlocks downstream work:

- **New field** -> unlocks NTT, curve ops, MSM, hash over that field
- **New curve** -> unlocks MSM, signatures, commitments on that curve
- **Pairing on new curve** -> unlocks KZG, Groth16, BLS sigs for that curve
- **KZG on new curve** -> unlocks Plonk, Marlin, batch openings
- **FRI on new field** -> unlocks STARK prover for that field
- **Sumcheck on new field** -> unlocks Spartan, GKR, Lasso, HyperNova
