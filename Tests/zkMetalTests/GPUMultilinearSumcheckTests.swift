// GPUMultilinearSumcheckEngine tests
//
// Verifies:
//   - Single-polynomial sumcheck (prove + verify, various sizes)
//   - Product sumcheck (degree-2 round polys, correctness)
//   - Batch sumcheck (multiple instances combined)
//   - Round polynomial interpolation and evaluation
//   - GPU-accelerated path (when available)
//   - Edge cases: zero polynomial, single variable, constant polynomial

import Foundation
@testable import zkMetal

public func runGPUMultilinearSumcheckTests() {
    suite("GPUMultilinearSumcheckEngine")

    let engine = GPUMultilinearSumcheckEngine()

    // Helpers

    func computeSum(_ evals: [Fr]) -> Fr {
        var s = Fr.zero
        for e in evals { s = frAdd(s, e) }
        return s
    }

    func computeProductSum(_ f: [Fr], _ g: [Fr]) -> Fr {
        precondition(f.count == g.count)
        var s = Fr.zero
        for i in 0..<f.count {
            s = frAdd(s, frMul(f[i], g[i]))
        }
        return s
    }

    func pseudoRandomFr(seed: inout UInt64) -> Fr {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(seed >> 32)
    }

    func randomEvals(_ logSize: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [Fr] {
        var rng = seed
        let n = 1 << logSize
        return (0..<n).map { _ in pseudoRandomFr(seed: &rng) }
    }

    // ================================================================
    // SECTION 1: MultilinearRoundPoly interpolation
    // ================================================================

    suite("MultilinearRoundPoly — interpolation")

    // Test: degree-1 interpolation
    do {
        let s0 = frFromInt(5)
        let s1 = frFromInt(13)
        let rp = MultilinearRoundPoly(evals: [s0, s1])

        expect(frEqual(rp.atZero, s0), "Degree-1 round poly: p(0) = 5")
        expect(frEqual(rp.atOne, s1), "Degree-1 round poly: p(1) = 13")
        expect(rp.degree == 1, "Degree-1 round poly: degree == 1")

        // p(2) = 5 + 2*(13-5) = 21
        let two = frAdd(Fr.one, Fr.one)
        let at2 = rp.evaluate(at: two)
        expect(frEqual(at2, frFromInt(21)), "Degree-1 round poly: p(2) = 21")
    }

    // Test: degree-2 interpolation
    do {
        // p(x) = 1 + 3x + 2x^2 => p(0)=1, p(1)=6, p(2)=15
        let e0 = frFromInt(1)
        let e1 = frFromInt(6)
        let e2 = frFromInt(15)
        let rp = MultilinearRoundPoly(evals: [e0, e1, e2])

        expect(rp.degree == 2, "Degree-2 round poly: degree == 2")
        expect(frEqual(rp.atZero, e0), "Degree-2 round poly: p(0) = 1")
        expect(frEqual(rp.atOne, e1), "Degree-2 round poly: p(1) = 6")

        // p(3) = 1 + 9 + 18 = 28
        let three = frFromInt(3)
        let at3 = rp.evaluate(at: three)
        expect(frEqual(at3, frFromInt(28)), "Degree-2 round poly: p(3) = 28")

        // p(0) should still match
        let at0 = rp.evaluate(at: Fr.zero)
        expect(frEqual(at0, e0), "Degree-2 round poly: evaluate(0) matches p(0)")
    }

    // ================================================================
    // SECTION 2: Single-polynomial sumcheck — small instances
    // ================================================================

    suite("Single-polynomial sumcheck — small instances")

    // Test: 1-variable sumcheck (2 elements)
    do {
        let evals: [Fr] = [frFromInt(3), frFromInt(7)]
        let claimedSum = computeSum(evals)  // 10

        let proverT = Transcript(label: "single-1var")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 1, "1-var sumcheck: numVars == 1")
        expect(proof.roundPolys.count == 1, "1-var sumcheck: 1 round poly")

        // Verify
        let verifierT = Transcript(label: "single-1var")
        let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "1-var sumcheck: verification passes")
        expect(evalPoint.count == 1, "1-var sumcheck: eval point has 1 coord")

        // Cross-check with MLE
        let mle = MultilinearPoly(numVars: 1, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "1-var sumcheck: MLE(r) == finalEval")
    }

    // Test: 2-variable sumcheck (4 elements)
    do {
        let evals: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]
        let claimedSum = computeSum(evals)  // 17

        let proverT = Transcript(label: "single-2var")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 2, "2-var sumcheck: numVars == 2")
        expect(proof.roundPolys.count == 2, "2-var sumcheck: 2 round polys")

        let verifierT = Transcript(label: "single-2var")
        let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "2-var sumcheck: verification passes")

        let mle = MultilinearPoly(numVars: 2, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "2-var sumcheck: MLE(r) == finalEval")
    }

    // Test: 4-variable sumcheck (16 elements)
    do {
        var evals = [Fr]()
        for i in 0..<16 { evals.append(frFromInt(UInt64(i * 3 + 1))) }
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "single-4var")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 4, "4-var sumcheck: numVars == 4")

        let verifierT = Transcript(label: "single-4var")
        let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "4-var sumcheck: verification passes")

        let mle = MultilinearPoly(numVars: 4, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "4-var sumcheck: MLE(r) == finalEval")
    }

    // ================================================================
    // SECTION 3: Single-polynomial sumcheck — round poly consistency
    // ================================================================

    suite("Single-polynomial sumcheck — round consistency")

    do {
        var evals = [Fr]()
        for i in 0..<32 { evals.append(frFromInt(UInt64((i * 7 + 13) % 100))) }
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "consistency-5var")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        // Manually check each round: p(0) + p(1) = running claim
        var currentClaim = claimedSum
        var roundsOK = true
        for i in 0..<proof.numVars {
            let rp = proof.roundPolys[i]
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                roundsOK = false
                break
            }
            currentClaim = rp.evaluate(at: proof.challenges[i])
        }
        expect(roundsOK, "5-var: all rounds satisfy p(0)+p(1) = claim")
        expect(frEqual(currentClaim, proof.finalEval), "5-var: final claim == finalEval")
    }

    // Test: all round polys are degree <= 1 for single-poly sumcheck
    do {
        let evals = randomEvals(6)
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "degree-check")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        var allDeg1 = true
        for rp in proof.roundPolys {
            if rp.degree > 1 { allDeg1 = false }
        }
        expect(allDeg1, "Single-poly round polys are all degree <= 1")
    }

    // ================================================================
    // SECTION 4: Single-polynomial — wrong claimed sum rejected
    // ================================================================

    suite("Single-polynomial sumcheck — wrong sum rejection")

    do {
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let correctSum = computeSum(evals)  // 10
        let wrongSum = frAdd(correctSum, Fr.one)  // 11

        let proverT = Transcript(label: "wrong-sum")
        let proof = engine.proveSingle(evals: evals, claimedSum: correctSum, transcript: proverT)

        // Verify with wrong claimed sum
        let verifierT = Transcript(label: "wrong-sum")
        let (valid, _, _) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: wrongSum, transcript: verifierT)
        expect(!valid, "Wrong claimed sum is rejected by verifier")
    }

    // ================================================================
    // SECTION 5: Zero polynomial
    // ================================================================

    suite("Single-polynomial sumcheck — zero polynomial")

    do {
        let evals = [Fr](repeating: Fr.zero, count: 8)
        let claimedSum = Fr.zero

        let proverT = Transcript(label: "zero-poly")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 3, "Zero poly: numVars == 3")
        expect(frIsZero(proof.finalEval), "Zero poly: finalEval == 0")

        // All round polys should have zero evaluations
        var allZero = true
        for rp in proof.roundPolys {
            for e in rp.evals {
                if !frIsZero(e) { allZero = false }
            }
        }
        expect(allZero, "Zero poly: all round poly evals are zero")

        let verifierT = Transcript(label: "zero-poly")
        let (valid, _, _) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Zero poly: verification passes")
    }

    // ================================================================
    // SECTION 6: Product sumcheck — basic correctness
    // ================================================================

    suite("Product sumcheck — basic correctness")

    // Test: 2-variable product sumcheck
    do {
        let evalsF: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let evalsG: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let claimedSum = computeProductSum(evalsF, evalsG)
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

        let proverT = Transcript(label: "product-2var")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsG,
            claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 2, "Product 2-var: numVars == 2")
        expect(proof.roundPolys.count == 2, "Product 2-var: 2 round polys")
        expect(proof.finalEvalG != nil, "Product 2-var: has finalEvalG")

        // Round polys should be degree 2
        for rp in proof.roundPolys {
            expect(rp.degree == 2, "Product 2-var: round poly degree == 2")
        }

        // Verify
        let verifierT = Transcript(label: "product-2var")
        let (valid, evalPoint, finalF, finalG) = GPUMultilinearSumcheckEngine.verifyProduct(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Product 2-var: verification passes")

        // Cross-check: MLE evals at challenges
        let mleF = MultilinearPoly(numVars: 2, evals: evalsF)
        let mleG = MultilinearPoly(numVars: 2, evals: evalsG)
        let mleFVal = mleF.evaluate(at: evalPoint)
        let mleGVal = mleG.evaluate(at: evalPoint)
        expect(frEqual(mleFVal, finalF), "Product 2-var: MLE_f(r) == finalEvalF")
        expect(frEqual(mleGVal, finalG), "Product 2-var: MLE_g(r) == finalEvalG")
    }

    // Test: 4-variable product sumcheck with random data
    do {
        let evalsF = randomEvals(4, seed: 0x1111_2222_3333_4444)
        let evalsG = randomEvals(4, seed: 0x5555_6666_7777_8888)
        let claimedSum = computeProductSum(evalsF, evalsG)

        let proverT = Transcript(label: "product-4var")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsG,
            claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == 4, "Product 4-var: numVars == 4")

        let verifierT = Transcript(label: "product-4var")
        let (valid, evalPoint, finalF, finalG) = GPUMultilinearSumcheckEngine.verifyProduct(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Product 4-var: verification passes")

        let mleF = MultilinearPoly(numVars: 4, evals: evalsF)
        let mleG = MultilinearPoly(numVars: 4, evals: evalsG)
        expect(frEqual(mleF.evaluate(at: evalPoint), finalF), "Product 4-var: MLE_f matches")
        expect(frEqual(mleG.evaluate(at: evalPoint), finalG), "Product 4-var: MLE_g matches")
    }

    // ================================================================
    // SECTION 7: Product sumcheck — round poly consistency
    // ================================================================

    suite("Product sumcheck — round consistency")

    do {
        let evalsF = randomEvals(3, seed: 0xAAAA_BBBB_CCCC_DDDD)
        let evalsG = randomEvals(3, seed: 0xEEEE_FFFF_0000_1111)
        let claimedSum = computeProductSum(evalsF, evalsG)

        let proverT = Transcript(label: "product-rounds")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsG,
            claimedSum: claimedSum, transcript: proverT)

        // Check each round: p(0) + p(1) = currentClaim
        var currentClaim = claimedSum
        var roundsOK = true
        for i in 0..<proof.numVars {
            let rp = proof.roundPolys[i]
            let roundSum = frAdd(rp.atZero, rp.atOne)
            if !frEqual(roundSum, currentClaim) {
                roundsOK = false
                break
            }
            currentClaim = rp.evaluate(at: proof.challenges[i])
        }
        expect(roundsOK, "Product rounds: p(0)+p(1) = claim at every round")

        // Final claim should be f(r)*g(r)
        let expectedFinal = frMul(proof.finalEval, proof.finalEvalG!)
        expect(frEqual(currentClaim, expectedFinal), "Product: final claim == f(r)*g(r)")
    }

    // ================================================================
    // SECTION 8: Product sumcheck — wrong sum rejected
    // ================================================================

    suite("Product sumcheck — wrong sum rejection")

    do {
        let evalsF: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let evalsG: [Fr] = [frFromInt(11), frFromInt(13), frFromInt(17), frFromInt(19)]
        let correctSum = computeProductSum(evalsF, evalsG)
        let wrongSum = frAdd(correctSum, frFromInt(42))

        let proverT = Transcript(label: "product-wrong")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsG,
            claimedSum: correctSum, transcript: proverT)

        let verifierT = Transcript(label: "product-wrong")
        let (valid, _, _, _) = GPUMultilinearSumcheckEngine.verifyProduct(
            proof: proof, claimedSum: wrongSum, transcript: verifierT)
        expect(!valid, "Product: wrong claimed sum is rejected")
    }

    // ================================================================
    // SECTION 9: Batch sumcheck — basic correctness
    // ================================================================

    suite("Batch sumcheck — basic correctness")

    // Batch of 2 polynomials, 2 variables each
    do {
        let poly0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let poly1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let sum0 = computeSum(poly0)  // 10
        let sum1 = computeSum(poly1)  // 100

        let proverT = Transcript(label: "batch-2poly")
        let proof = engine.proveBatch(
            polys: [poly0, poly1], claimedSums: [sum0, sum1], transcript: proverT)

        expect(proof.batchWeights.count == 2, "Batch: 2 weights generated")
        expect(proof.finalEvals.count == 2, "Batch: 2 final evaluations")
        expect(proof.combinedProof.numVars == 2, "Batch: combined proof has 2 vars")

        // Verify
        let verifierT = Transcript(label: "batch-2poly")
        let (valid, evalPoint, finalEvals) = GPUMultilinearSumcheckEngine.verifyBatch(
            proof: proof, claimedSums: [sum0, sum1], transcript: verifierT)
        expect(valid, "Batch 2-poly: verification passes")

        // Cross-check: each final eval matches MLE evaluation
        let mle0 = MultilinearPoly(numVars: 2, evals: poly0)
        let mle1 = MultilinearPoly(numVars: 2, evals: poly1)
        expect(frEqual(mle0.evaluate(at: evalPoint), finalEvals[0]),
               "Batch: MLE_0(r) matches finalEvals[0]")
        expect(frEqual(mle1.evaluate(at: evalPoint), finalEvals[1]),
               "Batch: MLE_1(r) matches finalEvals[1]")
    }

    // Batch of 3 polynomials, 3 variables each
    do {
        let poly0 = randomEvals(3, seed: 0x1000)
        let poly1 = randomEvals(3, seed: 0x2000)
        let poly2 = randomEvals(3, seed: 0x3000)
        let sum0 = computeSum(poly0)
        let sum1 = computeSum(poly1)
        let sum2 = computeSum(poly2)

        let proverT = Transcript(label: "batch-3poly")
        let proof = engine.proveBatch(
            polys: [poly0, poly1, poly2],
            claimedSums: [sum0, sum1, sum2],
            transcript: proverT)

        expect(proof.batchWeights.count == 3, "Batch 3-poly: 3 weights")
        expect(proof.finalEvals.count == 3, "Batch 3-poly: 3 final evals")

        let verifierT = Transcript(label: "batch-3poly")
        let (valid, evalPoint, finalEvals) = GPUMultilinearSumcheckEngine.verifyBatch(
            proof: proof,
            claimedSums: [sum0, sum1, sum2],
            transcript: verifierT)
        expect(valid, "Batch 3-poly: verification passes")

        // Cross-check all three
        for (j, poly) in [poly0, poly1, poly2].enumerated() {
            let mle = MultilinearPoly(numVars: 3, evals: poly)
            let mleVal = mle.evaluate(at: evalPoint)
            expect(frEqual(mleVal, finalEvals[j]),
                   "Batch 3-poly: MLE_\(j)(r) matches finalEvals[\(j)]")
        }
    }

    // ================================================================
    // SECTION 10: Batch sumcheck — wrong sums rejected
    // ================================================================

    suite("Batch sumcheck — wrong sum rejection")

    do {
        let poly0: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]
        let poly1: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(1), frFromInt(1)]
        let sum0 = computeSum(poly0)
        let sum1 = computeSum(poly1)
        let wrongSum1 = frAdd(sum1, Fr.one)

        let proverT = Transcript(label: "batch-wrong")
        let proof = engine.proveBatch(
            polys: [poly0, poly1], claimedSums: [sum0, sum1], transcript: proverT)

        // Verify with wrong sum for second polynomial
        let verifierT = Transcript(label: "batch-wrong")
        let (valid, _, _) = GPUMultilinearSumcheckEngine.verifyBatch(
            proof: proof, claimedSums: [sum0, wrongSum1], transcript: verifierT)
        expect(!valid, "Batch: wrong claimed sum for one poly is rejected")
    }

    // ================================================================
    // SECTION 11: Constant polynomial
    // ================================================================

    suite("Sumcheck — constant polynomial")

    do {
        let c = frFromInt(42)
        let evals = [Fr](repeating: c, count: 8)  // 3 variables, constant 42
        let claimedSum = frMul(c, frFromInt(8))     // 42 * 8 = 336

        let proverT = Transcript(label: "constant")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        let verifierT = Transcript(label: "constant")
        let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Constant poly: verification passes")

        // Final eval should be 42 (constant poly evaluates to 42 everywhere)
        let mle = MultilinearPoly(numVars: 3, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "Constant poly: MLE(r) == finalEval")
        expect(frEqual(finalEval, c), "Constant poly: finalEval == 42")
    }

    // ================================================================
    // SECTION 12: Larger single-poly sumcheck (8-10 variables)
    // ================================================================

    suite("Single-polynomial sumcheck — larger instances")

    for logSize in [8, 10] {
        let evals = randomEvals(logSize, seed: UInt64(logSize) &* 0xABCDABCD)
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "single-\(logSize)var")
        let proof = engine.proveSingle(evals: evals, claimedSum: claimedSum, transcript: proverT)

        expect(proof.numVars == logSize, "\(logSize)-var: numVars == \(logSize)")

        let verifierT = Transcript(label: "single-\(logSize)var")
        let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "\(logSize)-var: verification passes")

        let mle = MultilinearPoly(numVars: logSize, evals: evals)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalEval), "\(logSize)-var: MLE(r) == finalEval")
    }

    // ================================================================
    // SECTION 13: GPU-accelerated single-poly sumcheck
    // ================================================================

    suite("GPU-accelerated single-poly sumcheck")

    do {
        // Test GPU path with a table large enough to trigger GPU threshold
        let logSize = 12
        let evals = randomEvals(logSize)
        let claimedSum = computeSum(evals)

        let proverT = Transcript(label: "gpu-single-12var")
        do {
            let proof = try engine.proveSingleGPU(
                evals: evals, claimedSum: claimedSum, transcript: proverT)

            expect(proof.numVars == logSize, "GPU single \(logSize)-var: numVars correct")

            let verifierT = Transcript(label: "gpu-single-12var")
            let (valid, evalPoint, finalEval) = GPUMultilinearSumcheckEngine.verifySingle(
                proof: proof, claimedSum: claimedSum, transcript: verifierT)
            expect(valid, "GPU single \(logSize)-var: verification passes")

            let mle = MultilinearPoly(numVars: logSize, evals: evals)
            let mleVal = mle.evaluate(at: evalPoint)
            expect(frEqual(mleVal, finalEval), "GPU single \(logSize)-var: MLE(r) == finalEval")

            // Also verify GPU result matches CPU result
            let cpuProverT = Transcript(label: "cpu-single-12var")
            let cpuProof = engine.proveSingle(
                evals: evals, claimedSum: claimedSum, transcript: cpuProverT)

            // Final evals should match
            expect(frEqual(proof.finalEval, cpuProof.finalEval),
                   "GPU single: GPU finalEval == CPU finalEval")
        } catch {
            // GPU not available — expected on some machines
            print("  [SKIP] GPU path not available: \(error)")
        }
    }

    // ================================================================
    // SECTION 14: Product sumcheck — identity (f*1 = f)
    // ================================================================

    suite("Product sumcheck — f * 1 = sum(f)")

    do {
        let evalsF: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let evalsG: [Fr] = [Fr.one, Fr.one, Fr.one, Fr.one]
        let claimedSum = computeProductSum(evalsF, evalsG)
        // f*1 = f, so sum = 10+20+30+40 = 100
        expect(frEqual(claimedSum, frFromInt(100)), "Product f*1: claimed sum = 100")

        let proverT = Transcript(label: "product-identity")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsG,
            claimedSum: claimedSum, transcript: proverT)

        let verifierT = Transcript(label: "product-identity")
        let (valid, _, _, _) = GPUMultilinearSumcheckEngine.verifyProduct(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Product f*1: verification passes")
    }

    // ================================================================
    // SECTION 15: Product sumcheck — f*f (squaring)
    // ================================================================

    suite("Product sumcheck — f * f (squaring)")

    do {
        let evalsF: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let claimedSum = computeProductSum(evalsF, evalsF)
        // 4 + 9 + 25 + 49 = 87

        let proverT = Transcript(label: "product-square")
        let proof = engine.proveProduct(
            evalsF: evalsF, evalsG: evalsF,
            claimedSum: claimedSum, transcript: proverT)

        let verifierT = Transcript(label: "product-square")
        let (valid, evalPoint, finalF, finalG) = GPUMultilinearSumcheckEngine.verifyProduct(
            proof: proof, claimedSum: claimedSum, transcript: verifierT)
        expect(valid, "Product f*f: verification passes")

        // f and g are the same poly, so finalF == finalG
        expect(frEqual(finalF, finalG), "Product f*f: finalEvalF == finalEvalG")

        let mle = MultilinearPoly(numVars: 2, evals: evalsF)
        let mleVal = mle.evaluate(at: evalPoint)
        expect(frEqual(mleVal, finalF), "Product f*f: MLE(r) == finalEvalF")
    }
}
