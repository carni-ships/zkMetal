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
| BLS Signatures | BLS12-381 pairing (EXISTS), MSM (EXISTS) | Mostly glue code |
| Grumpkin | BN254 Fr + Fq (BOTH EXIST) | Fp=BN254 Fr, Fq=BN254 Fq |
| Schnorr | secp256k1 (EXISTS), SHA-256 | Needs SHA-256 first |
| Jubjub | BLS12-381 Fr (EXISTS) | Same pattern as BabyJubjub |
| Stark252 | Nothing (new field) | New field + NTT integration |

## What Unlocks What

Completing a hub primitive unlocks downstream work:

- **New field** -> unlocks NTT, curve ops, MSM, hash over that field
- **New curve** -> unlocks MSM, signatures, commitments on that curve
- **Pairing on new curve** -> unlocks KZG, Groth16, BLS sigs for that curve
- **KZG on new curve** -> unlocks Plonk, Marlin, batch openings
- **FRI on new field** -> unlocks STARK prover for that field
- **Sumcheck on new field** -> unlocks Spartan, GKR, Lasso, HyperNova
