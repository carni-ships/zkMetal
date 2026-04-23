# Roadmap

High-level priorities and future directions.

## Remaining Opportunities

### High Priority
- [OUT-OF-SCOPE] **Hardware upgrade** (M4 Pro/Max with more GPU cores) - depends on external decision
- **Protocol changes** (different PCS, fewer commitment rounds)

### Medium Priority
- **GPU CSR sparse matvec** — Main bottleneck for folding (6 matvecs per fold). Very high effort (3-4 weeks).
- **GPU fused sumcheck round** — Fuses eq-weighting with fold. Medium effort (3-4 days).

### Lower Priority
- **Bucket-Interleaved Layout** — 15-25% speedup for secp256k1 MSM. Medium effort.
- [OUT-OF-SCOPE] **MetalSpoon SP1 prover** GPU port — constraint eval / permutation / PCS dispatch.

### Theoretical Extensions (Require Deep Protocol Changes)
- Fused-DeepFold with shared-memory batch (4-8 rounds in one SM dispatch)
- Lazy Cantor-FFT for Circle STARK
- WHIR-RAA deterministic batched queries

## FRI Merkle Investigation (2026-04-12)

Real FRI bottleneck is Merkle commit at 210ms (80% of FRI time), not fold at 13ms (4%). Merkle uses Keccak GPU which is already near hardware limits. **No viable optimization target remains.**

## System Status

The system is **GPU-bound**. At peak optimization on M3 Pro (BN254 UltraHonk 428K gates, ~969ms prove):
- ~59% GPU time (MSM commits, Gemini, KZG)
- ~31% CPU
- ~10% overhead

**CPU micro-optimizations are exhausted** — all BN254 Fr batch patterns converted, all allocation patterns optimized.

**Remaining systemic opportunities**:
- Command buffer chaining (332 `waitUntilCompleted` sync points could batch into ~10 chained dispatches, saving 3-8ms)
- FRI fold-by-4 halves round count, reducing Merkle commit overhead

**Near floor** (< 1.5x headroom):
- BabyBear NTT, Goldilocks NTT, Circle NTT, IPA prove, HyperNova fold, KZG commit, Groth16 prove

All BN254 Fr CPU paths are at their theoretical floor. Further gains require:
1. Hardware upgrade (M4 Pro/Max with more GPU cores)
2. Protocol changes (fewer commitment rounds)
3. Application-level caching (circuit/ProverInstance reuse)
