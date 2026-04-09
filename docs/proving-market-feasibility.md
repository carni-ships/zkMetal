# Proving Market Feasibility (M3 Pro)

Real-time local proving feasibility ranked by likelihood of being a competitive prover.

| Rank | System | Proof Time Needed | Our Current | Gap | Feasibility |
|------|--------|-------------------|-------------|-----|-------------|
| 1 | **Mina SNARK work** | ~1-5s/proof | IPA 13ms, Pasta MSM 125ms | Already fast enough | **Very high** — 10-40x faster than needed |
| 2 | **HyperNova folding** | <100ms/fold | 0.09ms/fold | Already there | **Very high** — 1000x margin |
| 3 | **SP1 segment proving** | ~1-5s/segment | BabyBear NTT 2.0ms, FRI fold fast | Needs full prover integration | **High** — primitives are fast, need pipeline |
| 4 | **Aligned Layer (any proof)** | Varies by proof type | Groth16 14ms, STARK 17ms | Need wrapping/submission | **High** — our proofs are already fast |
| 5 | **Taiko (SP1/RISC0)** | ~10-30s/block proof | SP1 primitives fast | Need full block proving pipeline | **Medium-high** — block-level integration needed |
| 6 | **Aztec/Noir circuits** | <2s/prove (UltraHonk) | ~969ms (428K gates) | Already sub-second | **Medium** — need Aztec client integration |
| 7 | **Scroll (Halo2)** | ~minutes/batch | Halo2 primitives available | Massive circuit, needs aggregation | **Medium-low** — circuit too large for single machine |
| 8 | **Aleo (Marlin)** | ~1-5s/tx | Marlin verify 1.4ms, KZG fast | Need full Aleo prover | **Medium-low** — proprietary circuit |

## Best Bets for Immediate Local Proving Income

1. **Mina** — We can already prove 10-40x faster than needed. Just need the SNARK worker client integration.
2. **Aligned Layer** — Submit any proof type we can generate. Our Groth16/STARK provers are ready.
3. **SP1 Network** — BabyBear primitives are near-optimal. Need SP1 prover binary + our Metal acceleration.

## Mina Gap Analysis

| Kimchi needs | zkMetal status |
|---|---|
| Pallas/Vesta field arithmetic | GPU + C/NEON |
| Pallas/Vesta MSM | GPU — 125ms/128ms at 2^18 |
| IPA commitment/opening | GPU + CPU — 11.8ms prove, 7.3ms accumulate |
| Plonk gates, permutations, quotient, grand product | GPU engines exist |
| Poseidon hash (Pasta fields) | Partial — Poseidon2 exists for BN254/M31/BabyBear, needs Pasta instantiation |
| NTT over Pasta Fp/Fq | Missing — NTT exists for BN254/BLS12-377/Goldilocks/BabyBear/Stark252, not Pasta |
| Kimchi custom gates (range check, foreign field, endo, xor, rot) | Not yet — Plonk gate infra exists but Kimchi-specific gates need implementing |

## MetalSpoon Integration Gaps

MetalSpoon is our SP1 prover repo that consumes zkMetal primitives. It currently has GPU MSM/NTT/hash but is missing four pipeline-level capabilities that need to be ported from zkMetal:

| Capability | zkMetal | MetalSpoon | Why It Matters |
|---|---|---|---|
| **GPU constraint evaluation (IR interpreter)** | Yes | Not yet | Runs arbitrary gate expressions on GPU. Without this, constraint evaluation is CPU-bottlenecked — typically 20-40% of prove time. |
| **GPU permutation trace finalization** | Yes | Not yet | Grand product accumulation on GPU. Permutation argument is O(n) field muls over the full trace. |
| **GPU PCS interpolation/reduction** | Yes | Not yet | Polynomial commitment operations stay on GPU, avoiding GPU->CPU->GPU round-trips that kill throughput. |
| **Smart CPU/GPU dispatch with swap pressure detection** | Yes | Not yet | Adaptive threshold routing (e.g., small ops to CPU, large to GPU) with memory pressure detection. |

**Next step:** Port these four capabilities from zkMetal into MetalSpoon to complete the SP1 GPU-native proving pipeline.
