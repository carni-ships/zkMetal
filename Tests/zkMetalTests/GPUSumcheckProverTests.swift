// GPUSumcheckProverEngine tests
// Verifies: round polynomial correctness, claimed sum verification,
// full prove+verify for various sizes, zero polynomial handling.

import Foundation
import zkMetal

public func runGPUSumcheckProverTests() {
    suite("GPUSumcheckProverEngine")

    let engine = GPUSumcheckProverEngine()

    // Helper: compute claimed sum from evals
    func computeSum(_ evals: [Fr]) -> Fr {
        var s = Fr.zero
        for e in evals { s = frAdd(s, e) }
        return s
    }

    // ================================================================
    // Test 1: Simple 2-variable sumcheck
    // f(x0, x1) with 4 evaluations
    // ================================================================
    do {
        let evals: [Fr] = [
            frFromInt(3),  // f(0,0) = 3
            frFromInt(7),  // f(0,1) = 7
            frFromInt(2),  // f(1,0) = 2
            frFromInt(5),  // f(1,1) = 5
        ]
        let claimedSum = computeSum(evals)  // 3 + 7 + 2 + 5 = 17

        let proverT = Transcript(label: "test-2var")
        let proof = try engine.prove(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 2, "2-variable sumcheck: numVars == 2")
        expect(proof.roundPolys.count == 2, "2-variable sumcheck: 2 round polys")
        expect(proof.challenges.count == 2, "2-variable sumcheck: 2 challenges")

        // Verify
        let verifierT = Transcript(label: "test-2var")
        let (valid, evalPoint, finalEval) = GPUSumcheckProverEngine.verify(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "2-variable sumcheck: verification passes")
        expect(evalPoint.count == 2, "2-variable sumcheck: eval point has 2 coords")

        // Cross-check: MLE evaluation at challenges should match finalEval
        let mle = MultilinearPoly(numVars: 2, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "2-variable sumcheck: MLE(challenges) == finalEval")
    } catch {
        expect(false, "2-variable sumcheck threw: \(error)")
    }

    // ================================================================
    // Test 2: 4-variable sumcheck (16 elements)
    // ================================================================
    do {
        var evals = [Fr]()
        for i in 0..<16 {
            evals.append(frFromInt(UInt64(i * 3 + 1)))
        }
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "test-4var")
        let proof = try engine.prove(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 4, "4-variable sumcheck: numVars == 4")
        expect(proof.roundPolys.count == 4, "4-variable sumcheck: 4 round polys")

        // Verify
        let verifierT = Transcript(label: "test-4var")
        let (valid, evalPoint, finalEval) = GPUSumcheckProverEngine.verify(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "4-variable sumcheck: verification passes")

        // Cross-check with MLE
        let mle = MultilinearPoly(numVars: 4, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "4-variable sumcheck: MLE(challenges) == finalEval")
    } catch {
        expect(false, "4-variable sumcheck threw: \(error)")
    }

    // ================================================================
    // Test 3: Round polynomial degree check
    // For standard multilinear sumcheck, each round poly should be degree 1
    // ================================================================
    do {
        var evals = [Fr]()
        for i in 0..<8 {
            evals.append(frFromInt(UInt64(i + 1)))
        }
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "test-degree")
        let proof = try engine.prove(evals: evals, claimedSum: claimedSum, transcript: proverT)

        var allDeg1 = true
        for rp in proof.roundPolys {
            // c2 should be zero (degree <= 1)
            if !frIsZero(rp.c2) {
                allDeg1 = false
            }
            // c1 may or may not be zero depending on data
            // But degree should be at most 1
            if rp.degree > 1 {
                allDeg1 = false
            }
        }
        expect(allDeg1, "Round polys are degree <= 1 for multilinear sumcheck")

        // Also check that round polys are not all constant (degree 0)
        // unless the polynomial is trivial
        var hasDeg1 = false
        for rp in proof.roundPolys {
            if rp.degree == 1 { hasDeg1 = true }
        }
        expect(hasDeg1, "At least one round poly has degree 1 (non-trivial)")
    } catch {
        expect(false, "Round polynomial degree check threw: \(error)")
    }

    // ================================================================
    // Test 4: Claimed sum verification
    // Verify that proof is consistent: each round's p(0) + p(1) = running claim
    // ================================================================
    do {
        var evals = [Fr]()
        for i in 0..<32 {
            evals.append(frFromInt(UInt64((i * 7 + 13) % 100)))
        }
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "test-claim")
        let proof = try engine.prove(evals: evals, claimedSum: claimedSum, transcript: proverT)

        // Manually check each round
        var currentClaim = claimedSum
        var roundsOK = true
        for i in 0..<proof.numVars {
            let rp = proof.roundPolys[i]
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                roundsOK = false
                break
            }
            // Advance claim to p(r_i)
            currentClaim = rp.evaluate(at: proof.challenges[i])
        }
        expect(roundsOK, "Claimed sum: all rounds satisfy p(0)+p(1) = claim")

        // Final claim should match finalEval
        expect(frEqual(currentClaim, proof.finalEval), "Claimed sum: final claim == finalEval")

        // Wrong claimed sum should fail verification
        let wrongSum = frAdd(claimedSum, Fr.one)
        let verifierT = Transcript(label: "test-claim")
        let (wrongValid, _, _) = GPUSumcheckProverEngine.verify(
            proof: proof, claimedSum: wrongSum, transcript: verifierT)
        expect(!wrongValid, "Claimed sum: wrong claimed sum fails verification")
    } catch {
        expect(false, "Claimed sum verification threw: \(error)")
    }

    // ================================================================
    // Test 5: Zero polynomial
    // f = 0 everywhere, sum = 0, all round polys should be zero
    // ================================================================
    do {
        let evals = [Fr](repeating: Fr.zero, count: 4)  // 2 variables, all zero
        let claimedSum = Fr.zero

        let proverT = Transcript(label: "test-zero")
        let proof = try engine.prove(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 2, "Zero poly: numVars == 2")

        // All round polys should be zero polynomials
        var allZero = true
        for rp in proof.roundPolys {
            if !frIsZero(rp.c0) || !frIsZero(rp.c1) || !frIsZero(rp.c2) {
                allZero = false
            }
        }
        expect(allZero, "Zero poly: all round polynomials are zero")

        // Final eval should be zero
        expect(frIsZero(proof.finalEval), "Zero poly: final eval is zero")

        // Verify
        let verifierT = Transcript(label: "test-zero")
        let (valid, _, _) = GPUSumcheckProverEngine.verify(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Zero poly: verification passes")
    } catch {
        expect(false, "Zero polynomial test threw: \(error)")
    }
}
