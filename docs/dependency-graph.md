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

## Production Stack Coverage

How zkMetal maps to the major ZK stacks deployed in production today.

### Coverage Matrix

| Production Stack | Used By | Required Primitives | zkMetal Coverage |
|-----------------|---------|-------------------|-----------------|
| **Plonky3 / SP1** | Succinct SP1 zkVM, Valida | BabyBear NTT, Poseidon2 (width-16 BB), FRI, AIR constraints | **90%** — BabyBear NTT, P2 (BN254/M31), FRI, Circle STARK. Gap: P2 width-16 over BabyBear |
| **Halo2 (PSE)** | Scroll, Taiko, PSE circuits | Pasta (Pallas/Vesta), IPA, Plonk + lookups | **95%** — Pallas/Vesta, IPA, Plonk, LogUp all shipped |
| **Cairo / Stwo** | StarkNet, StarkWare | Stark252 field, Circle STARK over M31, Poseidon | **70%** — Circle STARK + M31 yes. **Gap: Stark252 field** |
| **RISC Zero** | RISC Zero zkVM | BabyBear, FRI, STARK, Poseidon2 | **95%** — all core primitives present |
| **Jolt / Lasso** | a16z Labs | BN254 pairing, Lasso lookups, sumcheck | **95%** — pairing, Lasso, sumcheck, batch FFI all shipped |
| **Barretenberg** | Aztec Network | BN254, Grumpkin, Plonk (UltraHonk), KZG | **95%** — BN254+Plonk+KZG+Grumpkin all shipped |
| **Gnark** | Linea, ConsenSys | BN254/BLS12-381, Groth16, Plonk+KZG | **95%** — all present |
| **Circom / Snarkjs** | Polygon zkEVM, Semaphore, Tornado Cash | BN254 Groth16, BabyJubjub, Poseidon | **80%** — Groth16+Poseidon yes. **Gap: BabyJubjub** (in progress) |
| **Boojum** | zkSync Era (Matter Labs) | Goldilocks, Poseidon2, FRI, custom gates | **90%** — Goldilocks NTT, P2, FRI all shipped |
| **Kimchi** | Mina Protocol | Pasta curves, IPA, Plonk | **95%** — Pallas/Vesta, IPA, Plonk all shipped |
| **Ethereum Consensus** | All validators | BLS12-381 pairing, BLS signatures | **70%** — pairing exists. **Gap: BLS signature scheme** |

### Gaps Blocking Full Coverage

| Gap | Blocks | Status |
|-----|--------|--------|
| **BabyJubjub curve** | Circom, Semaphore, Tornado Cash, Polygon | In progress (agent running) |
| **Stark252 field** | StarkNet/Cairo | Backlog (GAP7) |
| **BLS Signatures** | Ethereum consensus | Backlog (GAP5) |
| **Poseidon2 width-16 BabyBear** | SP1/Plonky3 exact config | Backlog (config change) |

### Recently Closed Gaps

- **SHA-256** — shipped (GPU batch hash 119M/s, fused Merkle subtree)
- **Grumpkin curve** — shipped (BN254 inner curve, GPU MSM with signed-digit)

### What Full Coverage Looks Like

With BabyJubjub (in progress) plus Stark252 and BLS signatures, zkMetal would have **90%+ coverage of every major production ZK stack**.

## What Unlocks What

Completing a hub primitive unlocks downstream work:

- **New field** -> unlocks NTT, curve ops, MSM, hash over that field
- **New curve** -> unlocks MSM, signatures, commitments on that curve
- **Pairing on new curve** -> unlocks KZG, Groth16, BLS sigs for that curve
- **KZG on new curve** -> unlocks Plonk, Marlin, batch openings
- **FRI on new field** -> unlocks STARK prover for that field
- **Sumcheck on new field** -> unlocks Spartan, GKR, Lasso, HyperNova
