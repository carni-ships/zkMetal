# Roadmap

High-level priorities and future directions.

## Remaining Opportunities

### High Priority
- [OUT-OF-SCOPE] **Hardware upgrade** (M4 Pro/Max with more GPU cores) - depends on external decision
- **Protocol changes** (different PCS, fewer commitment rounds)

### Medium Priority
- ~~GPU CSR sparse matvec~~ — ✅ Implemented in `GPUSparseMatvecEngine.swift`, integrated into `GPUNovaFoldEngine`
- ~~GPU fused sumcheck round~~ — ✅ Implemented in `GPUSumcheckProtocolEngine.swift` with fused kernels

### Lower Priority
- ~~Bucket-Interleaved Layout~~ — ✅ Implemented (flag: `useBucketInterleaved`)
- [OUT-OF-SCOPE] **MetalSpoon SP1 prover** GPU port — constraint eval / permutation / PCS dispatch.

### Theoretical Extensions (Require Deep Protocol Changes)
- ~~Fused-DeepFold with shared-memory batch~~ — ✅ Complete (GPU correctness fixed 2026-04-28)
- Lazy NTT for Circle STARK — Requires "lazy twiddle computation" for standard NTT; "lazy Cantor-FFT" was misnamed (Cantor FFT is for binary fields, not Mersenne31); not yet started
- ~~WHIR-RAA~~ — ✅ Implemented (RAA weight derivation pattern)

## FRI Merkle Investigation (2026-04-12)

Real FRI bottleneck is Merkle commit at 210ms (80% of FRI time), not fold at 13ms (4%). Merkle uses Keccak GPU which is already near hardware limits. **No viable optimization target remains.**

## Command Buffer Chaining Investigation (2026-04-28)

**Analyzed**: 439 `waitUntilCompleted` calls across 131 files

**Finding**: Command buffer chaining is **already well-optimized** in existing code.

**Circle FRI multiFold** (CircleFRIEngine.swift):
- All fold rounds already batched in single CB with one `waitUntilCompleted()`
- Single CB for all rounds: 0.41ms for 19 rounds

**Circle STARK Prover FRI loop** (CircleSTARKProver.swift):
- Each round: squeeze(alpha) → GPU fold+merkle → wait → read root → absorb commitment
- **Sequential Fiat-Shamir dependency chain** is the bottleneck, not CB overhead

**Performance breakdown** (Circle STARK 2^14, 15 rounds):
| Component | Time | % |
|-----------|------|---|
| GPU fold | 0.3ms | 5% |
| GPU merkle | 3.2ms | 56% |
| CPU overhead (alphas, readback) | 2.1ms | 37% |
| query phase | 0.1ms | 2% |

**Root cause of 37% CPU overhead**: Each FRI round's alpha depends on the previous round's commitment being absorbed (Fiat-Shamir sequentiality). This is a **fundamental protocol constraint**, not a code structure issue.

**Potential optimizations explored**:

1. **Pre-compute all alphas with PRNG**
   - Idea: Seed PRNG with initial commitment, pre-compute all alphas, batch all GPU work
   - Problem: Each alpha depends on previous commitments being absorbed (sequential dependency chain)
   - Two-pass approach (placeholder → derive alphas → real) requires 2x GPU fold time
   - Savings: ~2ms Fiat-Shamir overhead vs 2x GPU fold time (~0.6ms) → net negative
   - **Verdict: Not viable** - overhead exceeds savings

2. ~~GPU-based Fiat-Shamir transcript~~ — **IMPLEMENTED: Poseidon2-M31 transcript**
   - Implemented `CircleSTARKPoseidon2Transcript` using Poseidon2-M31 permutation
   - Benchmarks show **3.34x speedup** (265ms vs 886ms for 1000 absorb+squeeze)
   - Field-native: `squeezeM31()` returns `M31` directly without byte conversion
   - Both prover and verifier now use Poseidon2-based transcript
   - See `Sources/zkMetal/Transcript/Poseidon2Transcript.swift`

3. **Batch all FRI commitments, absorb later**
   - Idea: Fold all rounds with placeholder alphas, get all commitments, derive all alphas, redo folds
   - Problem: Same as #1 - requires 2x GPU work, changes commitment structure
   - **Verdict: Not viable** - same fundamental issue as #1

**Conclusion**: The Fiat-Shamir sequential dependency is inherent to the protocol. However, **Poseidon2-M31 transcript provides 3.34x speedup** over Keccak-based transcript, reducing the CPU overhead impact.

## Fiat-Shamir Optimization: Poseidon2 Transcript (2026-04-28)

**Implemented**: `CircleSTARKPoseidon2Transcript` replaces `CircleSTARKTranscript` for Circle STARK.

**Benchmark Results** (1000 absorb + 1000 squeeze):
| Backend | Time | Throughput | Speedup |
|---------|------|------------|---------|
| Poseidon2 | 265ms | 7,547 ops/s | **3.34x** |
| Keccak | 886ms | 2,257 ops/s | 1x |

**Correctness Verified**:
- Determinism: same inputs → same challenges ✅
- Domain separation: different labels → different challenges ✅
- Round-trip: prover/verifier alpha and fold-alpha match ✅

**Breaking Change**: Proofs generated with Poseidon2 transcript are NOT compatible with Keccak-based verifier.

## System Status

The system is **GPU-bound**. At peak optimization on M3 Pro (BN254 UltraHonk 428K gates, ~969ms prove):
- ~59% GPU time (MSM commits, Gemini, KZG)
- ~31% CPU
- ~10% overhead

**CPU micro-optimizations are exhausted** — all BN254 Fr batch patterns converted, all allocation patterns optimized.

**Remaining systemic opportunities**:
- Command buffer chaining — Already optimized; Fiat-Shamir sequentiality is the bottleneck (see analysis above)
- FRI fold-by-4 halves round count, reducing Merkle commit overhead (already done in Circle FRI)

**Near floor** (< 1.5x headroom):
- BabyBear NTT, Goldilocks NTT, Circle NTT, IPA prove, HyperNova fold, KZG commit, Groth16 prove

All BN254 Fr CPU paths are at their theoretical floor. Further gains require:
1. Hardware upgrade (M4 Pro/Max with more GPU cores)
2. Protocol changes (fewer commitment rounds)
3. Application-level caching (circuit/ProverInstance reuse)
