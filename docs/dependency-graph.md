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

## Dependency Table for New Primitives

| New Primitive | Depends On | Notes |
|---|---|---|
| Ed25519 | New Fp (2^255-19), new curve, SHA-512 | Fully new stack |
| BabyJubjub | BN254 Fr (EXISTS) | Cheap — just curve ops on existing field |
| SHA-256 | Nothing | Standalone like Keccak |
| ~~BLS Signatures~~ | BLS12-381 pairing, MSM | **DONE** — sign/verify/aggregate + hash-to-curve G2 |
| Grumpkin | BN254 Fr + Fq (BOTH EXIST) | Fp=BN254 Fr, Fq=BN254 Fq |
| Schnorr | secp256k1 (EXISTS), SHA-256 | Needs SHA-256 first |
| Jubjub | BLS12-381 Fr (EXISTS) | Same pattern as BabyJubjub |
| ~~Stark252~~ | Nothing (new field) | **DONE** — field + C NTT shipped |

## Production Stack Coverage

How zkMetal maps to the major ZK stacks deployed in production today.

### Coverage Matrix

| Production Stack | Used By | Required Primitives | zkMetal Coverage |
|-----------------|---------|-------------------|-----------------|
| **Plonky3 / SP1** | Succinct SP1 zkVM, Valida | BabyBear NTT, Poseidon2 (width-16 BB), FRI, AIR constraints | **95%** — BabyBear NTT, Poseidon2 BB width-16, FRI, Circle STARK all shipped |
| **Halo2 (PSE)** | Scroll, Taiko, PSE circuits | Pasta (Pallas/Vesta), IPA, Plonk + lookups | **95%** — Pallas/Vesta, IPA, Plonk, LogUp all shipped |
| **Cairo / Stwo** | StarkNet, StarkWare | Stark252 field, Circle STARK over M31, Poseidon | **95%** — Circle STARK + M31 + Stark252 field + NTT all shipped |
| **RISC Zero** | RISC Zero zkVM | BabyBear, FRI, STARK, Poseidon2 | **95%** — all core primitives present |
| **Jolt / Lasso** | a16z Labs | BN254 pairing, Lasso lookups, sumcheck | **95%** — pairing, Lasso, sumcheck, batch FFI all shipped |
| **Barretenberg** | Aztec Network | BN254, Grumpkin, Plonk (UltraHonk), KZG | **95%** — BN254+Plonk+KZG+Grumpkin all shipped |
| **Gnark** | Linea, ConsenSys | BN254/BLS12-381, Groth16, Plonk+KZG | **95%** — all present |
| **Circom / Snarkjs** | Polygon zkEVM, Semaphore, Tornado Cash | BN254 Groth16, BabyJubjub, Poseidon | **95%** — Groth16+Poseidon+BabyJubjub+Pedersen+EdDSA all shipped |
| **Boojum** | zkSync Era (Matter Labs) | Goldilocks, Poseidon2, FRI, custom gates | **90%** — Goldilocks NTT, P2, FRI all shipped |
| **Kimchi** | Mina Protocol | Pasta curves, IPA, Plonk | **95%** — Pallas/Vesta, IPA, Plonk all shipped |
| **Ethereum Consensus** | All validators | BLS12-381 pairing, BLS signatures | **95%** — BLS12-381 C pairing (30×), BLS signatures, hash-to-curve G2 all shipped |

### Gaps Blocking Full Coverage

No gaps remaining — all 11 production stacks at 95% coverage.

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

zkMetal now has **95% coverage of all 11 major production ZK stacks**. No critical gaps remain.

## What Unlocks What

Completing a hub primitive unlocks downstream work:

- **New field** -> unlocks NTT, curve ops, MSM, hash over that field
- **New curve** -> unlocks MSM, signatures, commitments on that curve
- **Pairing on new curve** -> unlocks KZG, Groth16, BLS sigs for that curve
- **KZG on new curve** -> unlocks Plonk, Marlin, batch openings
- **FRI on new field** -> unlocks STARK prover for that field
- **Sumcheck on new field** -> unlocks Spartan, GKR, Lasso, HyperNova
