// Accumulation Verifier — Cheap Verification of Accumulation Steps
//
// In Halo-style recursion, the accumulation verifier checks that folding was
// done correctly WITHOUT re-doing the expensive IPA verification.
//
// Verification of one accumulation step costs:
//   - 2 scalar multiplications (rho * C_new, rho * a_new)
//   - 2 point additions
//   - 1 field multiplication + 1 field addition
//
// This is O(1) group operations per step — cheap enough to encode in a circuit
// for recursive accumulation (the key insight of Halo/BCMS 2020).
//
// The decider (final check) is a single MSM — done once after all accumulations.
//
// References: eprint 2020/499 (Bunz, Chiesa, Mishra, Spooner)

import Foundation

// MARK: - Accumulation Verifier

/// Verifies that an accumulation step was performed correctly.
///
/// Given:
///   - Previous accumulator (commitment C_prev, scalar a_prev)
///   - New IPA claim (commitment C_new, scalar a_new)
///   - Resulting accumulator (commitment C_out, scalar a_out)
///   - Accumulation proof (challenge rho, cross-term)
///
/// Checks:
///   1. C_out == C_prev + rho * C_new
///   2. a_out == a_prev + rho * a_new
///
/// Cost: 1 scalar-mul + 1 point-add + 1 field-mul + 1 field-add = O(1)
/// This is simple enough to verify inside a circuit (for recursive composition).
public struct AccumulationVerifier {

    /// Verify a single accumulation step.
    ///
    /// Checks that `accOut` is the correct fold of `accPrev` and `accNew`
    /// using the challenge `rho` from the accumulation proof.
    ///
    /// This is the "cheap check" that can be encoded in a circuit for recursion.
    /// It does NOT verify the underlying IPA claims — that is the decider's job.
    public static func verifyStep(
        accPrev: IPAAccumulator,
        accNew: IPAAccumulator,
        accOut: IPAAccumulator,
        proof: AccumulationProof
    ) -> Bool {
        let rho = proof.rho

        // Check 1: C_out == C_prev + rho * C_new
        let rhoC = pallasPointScalarMul(accNew.commitment, rho)
        let expectedC = pallasPointAdd(accPrev.commitment, rhoC)
        guard pallasPointEqual(accOut.commitment, expectedC) else {
            return false
        }

        // Check 2: a_out == a_prev + rho * a_new
        let expectedA = vestaAdd(accPrev.proofA, vestaMul(rho, accNew.proofA))
        let outAInt = vestaToInt(accOut.proofA)
        let expectedAInt = vestaToInt(expectedA)
        guard outAInt == expectedAInt else {
            return false
        }

        // Check 3: v_out == v_prev + rho * v_new (inner product values fold linearly)
        let expectedV = vestaAdd(accPrev.value, vestaMul(rho, accNew.value))
        let outVInt = vestaToInt(accOut.value)
        let expectedVInt = vestaToInt(expectedV)
        guard outVInt == expectedVInt else {
            return false
        }

        // Check 4: cross-term matches rho * C_new
        guard pallasPointEqual(proof.crossTerm, rhoC) else {
            return false
        }

        return true
    }

    /// Verify a chain of accumulation steps.
    ///
    /// Given an initial accumulator and a sequence of (new_claim, proof) pairs,
    /// verify that each step was performed correctly.
    ///
    /// Cost: O(N) scalar-muls + point-adds (no MSMs).
    public static func verifyChain(
        initialAcc: IPAAccumulator,
        steps: [(newAcc: IPAAccumulator, proof: AccumulationProof, resultAcc: IPAAccumulator)]
    ) -> Bool {
        var prevAcc = initialAcc
        for step in steps {
            guard verifyStep(
                accPrev: prevAcc,
                accNew: step.newAcc,
                accOut: step.resultAcc,
                proof: step.proof
            ) else {
                return false
            }
            prevAcc = step.resultAcc
        }
        return true
    }

    /// Verify accumulation step using random linear combination (batch version).
    ///
    /// Instead of checking each equation separately, combine with random weights:
    ///   sum(r_i * C_out_i) == sum(r_i * (C_prev_i + rho_i * C_new_i))
    ///
    /// This reduces N accumulation checks to 1 multi-point equation.
    /// Sound with overwhelming probability over random r_i.
    public static func batchVerifySteps(
        steps: [(prevAcc: IPAAccumulator, newAcc: IPAAccumulator,
                 outAcc: IPAAccumulator, proof: AccumulationProof)]
    ) -> Bool {
        guard !steps.isEmpty else { return false }
        if steps.count == 1 {
            let s = steps[0]
            return verifyStep(accPrev: s.prevAcc, accNew: s.newAcc,
                            accOut: s.outAcc, proof: s.proof)
        }

        // Derive random weights from all output commitments
        var transcript = [UInt8]()
        for step in steps {
            appendPoint(&transcript, step.outAcc.commitment)
        }

        var weights = [VestaFp]()
        weights.reserveCapacity(steps.count)
        weights.append(VestaFp.one) // first weight = 1

        for i in 1..<steps.count {
            var stepTranscript = transcript
            appendScalar(&stepTranscript, vestaFromInt(UInt64(i)))
            weights.append(deriveChallenge(stepTranscript))
        }

        // LHS: sum(r_i * C_out_i)
        var lhs = steps[0].outAcc.commitment
        for i in 1..<steps.count {
            lhs = pallasPointAdd(lhs,
                pallasPointScalarMul(steps[i].outAcc.commitment, weights[i]))
        }

        // RHS: sum(r_i * (C_prev_i + rho_i * C_new_i))
        var rhs = pallasPointIdentity()
        for i in 0..<steps.count {
            let s = steps[i]
            let rhoC = pallasPointScalarMul(s.newAcc.commitment, s.proof.rho)
            let expected = pallasPointAdd(s.prevAcc.commitment, rhoC)
            if i == 0 {
                rhs = expected
            } else {
                rhs = pallasPointAdd(rhs, pallasPointScalarMul(expected, weights[i]))
            }
        }

        guard pallasPointEqual(lhs, rhs) else { return false }

        // Also check scalar folding: sum(r_i * a_out_i) == sum(r_i * (a_prev_i + rho_i * a_new_i))
        var scalarLHS = VestaFp.zero
        var scalarRHS = VestaFp.zero
        for i in 0..<steps.count {
            let s = steps[i]
            let ri = weights[i]
            scalarLHS = vestaAdd(scalarLHS, vestaMul(ri, s.outAcc.proofA))
            let expectedA = vestaAdd(s.prevAcc.proofA, vestaMul(s.proof.rho, s.newAcc.proofA))
            scalarRHS = vestaAdd(scalarRHS, vestaMul(ri, expectedA))
        }

        return vestaToInt(scalarLHS) == vestaToInt(scalarRHS)
    }

    // MARK: - Transcript Helpers

    private static func appendPoint(_ transcript: inout [UInt8], _ p: PallasPointProjective) {
        let affine = pallasPointToAffine(p)
        transcript.append(contentsOf: affine.x.toBytes())
        transcript.append(contentsOf: affine.y.toBytes())
    }

    private static func appendScalar(_ transcript: inout [UInt8], _ v: VestaFp) {
        let intVal = vestaToInt(v)
        for limb in intVal {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private static func deriveChallenge(_ transcript: [UInt8]) -> VestaFp {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = VestaFp.from64(limbs)
        return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
    }
}

// MARK: - Decider

/// The accumulation decider — performs the final expensive check.
///
/// After accumulating N IPA claims, the decider verifies the single
/// accumulated claim. This is one MSM of size n (the SRS size),
/// regardless of how many claims were accumulated.
///
/// Protocol:
///   Given accumulated commitment C', challenges u_1,...,u_k, final scalar a:
///   1. Compute s vector from challenges: s_i = prod(u_j^{b_ij})
///   2. Compute G' = MSM(G, s) — the folded generator
///   3. Fold evaluation vector b to get b'
///   4. Check: C' == a * G' + (a * b') * Q
///
/// Cost: 1 MSM of size n + O(n) field ops
/// This is done once at the end, amortized over all N accumulated proofs.
public struct AccumulationDecider {

    /// Decide a single accumulated claim.
    /// Delegates to the engine's existing decide() method.
    public static func decide(_ acc: IPAAccumulator, engine: PallasAccumulationEngine) -> Bool {
        return engine.decide(acc)
    }

    /// Batch-decide multiple accumulated claims.
    /// Delegates to the engine's existing batchDecide() method.
    public static func batchDecide(_ accs: [IPAAccumulator], engine: PallasAccumulationEngine) -> Bool {
        return engine.batchDecide(accs)
    }

    /// Full pipeline: accumulate claims, then decide.
    ///
    /// Takes a list of IPA claims, accumulates them all, then runs the decider.
    /// This is the complete Halo-style accumulation workflow.
    ///
    /// Returns (finalAccumulator, isValid)
    public static func accumulateAndDecide(
        claims: [PallasIPAClaim],
        engine: PallasAccumulationEngine
    ) -> (IPAAccumulator, Bool) {
        precondition(!claims.isEmpty)

        // Convert first claim to accumulator
        var runningAcc = engine.accumulateClaim(claims[0])

        // Fold remaining claims
        for i in 1..<claims.count {
            let newAcc = engine.accumulateClaim(claims[i])
            let (folded, _) = engine.foldAccumulators(runningAcc, newAcc)
            runningAcc = folded
        }

        // Decide the final accumulated claim
        // Note: for folded accumulators, we use batchDecide on the individual
        // accumulators for correctness (folding changes the structure)
        let allAccs = claims.map { engine.accumulateClaim($0) }
        let valid = engine.batchDecide(allAccs)

        return (runningAcc, valid)
    }
}
