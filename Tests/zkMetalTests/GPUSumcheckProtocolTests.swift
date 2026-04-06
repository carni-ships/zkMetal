// GPUSumcheckProtocolEngine tests
//
// Verifies:
//   - Standard sumcheck prove + verify (various sizes)
//   - Product sumcheck (GKR-style, degree-2 and degree-3 round polys)
//   - Combined/batched sumcheck (multiple claims merged)
//   - GPU-accelerated path (when available)
//   - RoundUnivariate interpolation at arbitrary points
//   - Configurable degree bound per round
//   - SumcheckClaim construction and sum computation
//   - Equality polynomial (eq poly) generation
//   - Multilinear evaluation via sequential folding
//   - Edge cases: single variable, zero polynomial, constant polynomial, identity

import zkMetal
import Foundation

public func runGPUSumcheckProtocolTests() {
    suite("GPU Sumcheck Protocol Engine")

    testRoundUnivariateLinear()
    testRoundUnivariateQuadratic()
    testRoundUnivariateCubic()
    testRoundUnivariateEquality()
    testSumcheckClaimConstruction()
    testSumcheckClaimComputedSum()
    testStandardSumcheck2Vars()
    testStandardSumcheck3Vars()
    testStandardSumcheck4Vars()
    testStandardSumcheckSingleVar()
    testStandardSumcheckZeroPoly()
    testStandardSumcheckConstantPoly()
    testStandardSumcheckLargeRandom()
    testProductSumcheck2Factors()
    testProductSumcheck3Factors()
    testProductSumcheckIdentityFactor()
    testProductSumcheckVerifierRejects()
    testCombinedSumcheck2Claims()
    testCombinedSumcheck3Claims()
    testCombinedSumcheckSingleClaim()
    testGPUAcceleratedStandard()
    testEqPolyGeneration()
    testEqPolyProductCheck()
    testMultilinearEvaluation()
    testMultilinearEvalConsistency()
    testVerifierRejectsBadSum()
    testVerifierRejectsBadRoundPoly()
    testDegreeBoundConfig()
    testCPUOnlyConfig()
    testStandardSumcheckDeterministic()
}

// MARK: - Helpers

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func pseudoRandom(seed: inout UInt64) -> Fr {
    seed = seed &* 6364136223846793005 &+ 1442695040888963407
    return frFromInt(seed >> 32)
}

private func randomEvals(_ logSize: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [Fr] {
    var rng = seed
    let n = 1 << logSize
    return (0..<n).map { _ in pseudoRandom(seed: &rng) }
}

private func computeSum(_ evals: [Fr]) -> Fr {
    var s = Fr.zero
    for e in evals { s = frAdd(s, e) }
    return s
}

private func computeProductSum(_ f: [Fr], _ g: [Fr]) -> Fr {
    var s = Fr.zero
    for i in 0..<f.count { s = frAdd(s, frMul(f[i], g[i])) }
    return s
}

private func computeTripleProductSum(_ f: [Fr], _ g: [Fr], _ h: [Fr]) -> Fr {
    var s = Fr.zero
    for i in 0..<f.count { s = frAdd(s, frMul(frMul(f[i], g[i]), h[i])) }
    return s
}

// MARK: - RoundUnivariate Tests

private func testRoundUnivariateLinear() {
    // p(X) = 5 + 8*X => p(0) = 5, p(1) = 13
    let s0 = frFromInt(5)
    let s1 = frFromInt(13)
    let rp = RoundUnivariate(evals: [s0, s1], degreeBound: 1)

    expect(frEq(rp.atZero, s0), "Linear round poly: p(0) = 5")
    expect(frEq(rp.atOne, s1), "Linear round poly: p(1) = 13")
    expect(rp.degree == 1, "Linear round poly: degree == 1")

    // p(2) = 5 + 2*(13-5) = 21
    let two = frAdd(Fr.one, Fr.one)
    let at2 = rp.evaluate(at: two)
    expect(frEq(at2, frFromInt(21)), "Linear round poly: p(2) = 21")

    // p(0) check via evaluate
    let at0 = rp.evaluate(at: Fr.zero)
    expect(frEq(at0, s0), "Linear round poly: evaluate(0) == atZero")
}

private func testRoundUnivariateQuadratic() {
    // p(x) = 1 + 3x + 2x^2 => p(0)=1, p(1)=6, p(2)=15
    let e0 = frFromInt(1)
    let e1 = frFromInt(6)
    let e2 = frFromInt(15)
    let rp = RoundUnivariate(evals: [e0, e1, e2], degreeBound: 2)

    expect(rp.degree == 2, "Quadratic round poly: degree == 2")
    expect(frEq(rp.atZero, e0), "Quadratic round poly: p(0) = 1")
    expect(frEq(rp.atOne, e1), "Quadratic round poly: p(1) = 6")

    // p(3) = 1 + 9 + 18 = 28
    let three = frFromInt(3)
    let at3 = rp.evaluate(at: three)
    expect(frEq(at3, frFromInt(28)), "Quadratic round poly: p(3) = 28")

    // p(0) + p(1) = 7
    let sum01 = frAdd(rp.atZero, rp.atOne)
    expect(frEq(sum01, frFromInt(7)), "Quadratic round poly: p(0)+p(1) = 7")
}

private func testRoundUnivariateCubic() {
    // p(x) = x^3 => p(0)=0, p(1)=1, p(2)=8, p(3)=27
    let e0 = frFromInt(0)
    let e1 = frFromInt(1)
    let e2 = frFromInt(8)
    let e3 = frFromInt(27)
    let rp = RoundUnivariate(evals: [e0, e1, e2, e3], degreeBound: 3)

    expect(rp.degree == 3, "Cubic round poly: degree == 3")

    // p(4) = 64
    let four = frFromInt(4)
    let at4 = rp.evaluate(at: four)
    expect(frEq(at4, frFromInt(64)), "Cubic round poly: p(4) = 64")

    // p(5) = 125
    let five = frFromInt(5)
    let at5 = rp.evaluate(at: five)
    expect(frEq(at5, frFromInt(125)), "Cubic round poly: p(5) = 125")
}

private func testRoundUnivariateEquality() {
    let e0 = frFromInt(42)
    let e1 = frFromInt(99)
    let rp1 = RoundUnivariate(evals: [e0, e1], degreeBound: 1)
    let rp2 = RoundUnivariate(evals: [e0, e1], degreeBound: 1)
    let rp3 = RoundUnivariate(evals: [e0, frFromInt(100)], degreeBound: 1)

    expect(rp1 == rp2, "RoundUnivariate equality: same evals")
    expect(!(rp1 == rp3), "RoundUnivariate inequality: different evals")
}

// MARK: - SumcheckClaim Tests

private func testSumcheckClaimConstruction() {
    let evals: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]
    let claim = SumcheckClaim(evals: evals, claimedSum: frFromInt(17))

    expectEqual(claim.numVars, 2, "SumcheckClaim: numVars == 2")
    expect(frEq(claim.claimedSum, frFromInt(17)), "SumcheckClaim: claimedSum == 17")
}

private func testSumcheckClaimComputedSum() {
    let evals: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    expect(frEq(claim.claimedSum, frFromInt(100)), "SumcheckClaim computed sum: 10+20+30+40=100")
    expectEqual(claim.numVars, 2, "SumcheckClaim computed sum: numVars == 2")
}

// MARK: - Standard Sumcheck Tests

private func testStandardSumcheck2Vars() {
    let engine = GPUSumcheckProtocolEngine()

    // f(x0, x1): f(0,0)=3, f(0,1)=7, f(1,0)=2, f(1,1)=5
    let evals: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]
    let claim = SumcheckClaim.withComputedSum(evals: evals) // sum = 17

    let proverT = Transcript(label: "proto-2var")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expectEqual(proof.numVars, 2, "Standard 2-var: numVars == 2")
    expectEqual(proof.roundPolys.count, 2, "Standard 2-var: 2 round polys")
    expectEqual(proof.challenges.count, 2, "Standard 2-var: 2 challenges")

    // Verify
    let verifierT = Transcript(label: "proto-2var")
    let (valid, evalPoint, finalEval) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "Standard 2-var: verification passes")
    expectEqual(evalPoint.count, 2, "Standard 2-var: evalPoint has 2 coords")

    // Cross-check with MLE
    let mle = MultilinearPoly(numVars: 2, evals: evals)
    let mleVal = mle.evaluate(at: evalPoint)
    expect(frEq(mleVal, finalEval), "Standard 2-var: MLE(challenges) == finalEval")
}

private func testStandardSumcheck3Vars() {
    let engine = GPUSumcheckProtocolEngine()
    let evals = randomEvals(3, seed: 0xABCD_1234)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let proverT = Transcript(label: "proto-3var")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expectEqual(proof.numVars, 3, "Standard 3-var: numVars == 3")
    expectEqual(proof.roundPolys.count, 3, "Standard 3-var: 3 round polys")

    // First round check: p(0) + p(1) = claimed sum
    let r0 = proof.roundPolys[0]
    let sum01 = frAdd(r0.atZero, r0.atOne)
    expect(frEq(sum01, claim.claimedSum), "Standard 3-var: round 0 sum check")

    let verifierT = Transcript(label: "proto-3var")
    let (valid, evalPoint, finalEval) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "Standard 3-var: verification passes")

    let mle = MultilinearPoly(numVars: 3, evals: evals)
    expect(frEq(mle.evaluate(at: evalPoint), finalEval), "Standard 3-var: MLE consistency")
}

private func testStandardSumcheck4Vars() {
    let engine = GPUSumcheckProtocolEngine()
    let evals = randomEvals(4, seed: 0x1111_2222)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let proverT = Transcript(label: "proto-4var")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expectEqual(proof.numVars, 4, "Standard 4-var: numVars == 4")

    let verifierT = Transcript(label: "proto-4var")
    let (valid, _, finalEval) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "Standard 4-var: verification passes")

    let mle = MultilinearPoly(numVars: 4, evals: evals)
    expect(frEq(mle.evaluate(at: proof.challenges), finalEval), "Standard 4-var: MLE consistency")
}

private func testStandardSumcheckSingleVar() {
    let engine = GPUSumcheckProtocolEngine()

    // f(x0) with 2 evaluations: f(0)=10, f(1)=20
    let evals: [Fr] = [frFromInt(10), frFromInt(20)]
    let claim = SumcheckClaim.withComputedSum(evals: evals) // sum = 30

    let proverT = Transcript(label: "proto-1var")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expectEqual(proof.numVars, 1, "Single var: numVars == 1")
    expectEqual(proof.roundPolys.count, 1, "Single var: 1 round poly")

    // The round poly p(0)=10, p(1)=20, p(0)+p(1)=30
    let rp = proof.roundPolys[0]
    expect(frEq(rp.atZero, frFromInt(10)), "Single var: p(0) = 10")
    expect(frEq(rp.atOne, frFromInt(20)), "Single var: p(1) = 20")

    let verifierT = Transcript(label: "proto-1var")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "Single var: verification passes")
}

private func testStandardSumcheckZeroPoly() {
    let engine = GPUSumcheckProtocolEngine()

    // All zeros: sum = 0
    let evals: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]
    let claim = SumcheckClaim(evals: evals, claimedSum: Fr.zero)

    let proverT = Transcript(label: "proto-zero")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expect(frEq(proof.finalEval, Fr.zero), "Zero poly: finalEval == 0")

    let verifierT = Transcript(label: "proto-zero")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: Fr.zero, transcript: verifierT)
    expect(valid, "Zero poly: verification passes")
}

private func testStandardSumcheckConstantPoly() {
    let engine = GPUSumcheckProtocolEngine()

    // Constant polynomial f(x) = 7 for all x in {0,1}^3
    let n = 8
    let seven = frFromInt(7)
    let evals = [Fr](repeating: seven, count: n)
    let expectedSum = frFromInt(UInt64(n) * 7) // 56
    let claim = SumcheckClaim(evals: evals, claimedSum: expectedSum)

    let proverT = Transcript(label: "proto-const")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    // Final eval should be 7 (constant everywhere)
    expect(frEq(proof.finalEval, seven), "Constant poly: finalEval == 7")

    let verifierT = Transcript(label: "proto-const")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: expectedSum, transcript: verifierT)
    expect(valid, "Constant poly: verification passes")
}

private func testStandardSumcheckLargeRandom() {
    let engine = GPUSumcheckProtocolEngine()
    let logSize = 10 // 1024 elements
    let evals = randomEvals(logSize, seed: 0xCAFE_BABE)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let proverT = Transcript(label: "proto-large")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    expectEqual(proof.numVars, logSize, "Large random: numVars == \(logSize)")

    let verifierT = Transcript(label: "proto-large")
    let (valid, evalPoint, finalEval) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "Large random: verification passes")

    let mle = MultilinearPoly(numVars: logSize, evals: evals)
    expect(frEq(mle.evaluate(at: evalPoint), finalEval), "Large random: MLE consistency")
}

// MARK: - Product Sumcheck Tests

private func testProductSumcheck2Factors() {
    let engine = GPUSumcheckProtocolEngine(config: .product2)

    let f = randomEvals(3, seed: 0x1111)
    let g = randomEvals(3, seed: 0x2222)
    let claimedSum = computeProductSum(f, g)

    let proverT = Transcript(label: "proto-prod2")
    let proof = engine.proveProduct(factors: [f, g], claimedSum: claimedSum, transcript: proverT)

    expectEqual(proof.numVars, 3, "Product 2-factor: numVars == 3")
    expectEqual(proof.degreeBound, 2, "Product 2-factor: degreeBound == 2")

    // Each round poly should have degree <= 2 (3 evaluations)
    for (i, rp) in proof.roundPolys.enumerated() {
        expect(rp.degree <= 2, "Product 2-factor: round \(i) degree <= 2")
        expectEqual(rp.evals.count, 3, "Product 2-factor: round \(i) has 3 evals")
    }

    // Verify
    let verifierT = Transcript(label: "proto-prod2")
    let (valid, evalPoint, factorEvals) = GPUSumcheckProtocolEngine.verifyProduct(
        proof: proof, claimedSum: claimedSum, transcript: verifierT)
    expect(valid, "Product 2-factor: verification passes")
    expectEqual(factorEvals.count, 2, "Product 2-factor: 2 factor final evals")

    // Cross-check factor evaluations
    let mleF = MultilinearPoly(numVars: 3, evals: f)
    let mleG = MultilinearPoly(numVars: 3, evals: g)
    expect(frEq(mleF.evaluate(at: evalPoint), factorEvals[0]),
           "Product 2-factor: f(challenges) matches")
    expect(frEq(mleG.evaluate(at: evalPoint), factorEvals[1]),
           "Product 2-factor: g(challenges) matches")

    // Product of factor evals should equal the final eval
    let prod = frMul(factorEvals[0], factorEvals[1])
    expect(frEq(prod, proof.finalEval), "Product 2-factor: product == finalEval")
}

private func testProductSumcheck3Factors() {
    let engine = GPUSumcheckProtocolEngine(config: .product3)

    let f = randomEvals(2, seed: 0xAAAA)
    let g = randomEvals(2, seed: 0xBBBB)
    let h = randomEvals(2, seed: 0xCCCC)
    let claimedSum = computeTripleProductSum(f, g, h)

    let proverT = Transcript(label: "proto-prod3")
    let proof = engine.proveProduct(factors: [f, g, h], claimedSum: claimedSum, transcript: proverT)

    expectEqual(proof.numVars, 2, "Product 3-factor: numVars == 2")
    expectEqual(proof.degreeBound, 3, "Product 3-factor: degreeBound == 3")

    // Round polys should have degree <= 3 (4 evaluations)
    for (i, rp) in proof.roundPolys.enumerated() {
        expect(rp.degree <= 3, "Product 3-factor: round \(i) degree <= 3")
        expectEqual(rp.evals.count, 4, "Product 3-factor: round \(i) has 4 evals")
    }

    let verifierT = Transcript(label: "proto-prod3")
    let (valid, evalPoint, factorEvals) = GPUSumcheckProtocolEngine.verifyProduct(
        proof: proof, claimedSum: claimedSum, transcript: verifierT)
    expect(valid, "Product 3-factor: verification passes")
    expectEqual(factorEvals.count, 3, "Product 3-factor: 3 factor final evals")

    // Check factor evaluations individually
    let mleF = MultilinearPoly(numVars: 2, evals: f)
    let mleG = MultilinearPoly(numVars: 2, evals: g)
    let mleH = MultilinearPoly(numVars: 2, evals: h)
    expect(frEq(mleF.evaluate(at: evalPoint), factorEvals[0]),
           "Product 3-factor: f(challenges) matches")
    expect(frEq(mleG.evaluate(at: evalPoint), factorEvals[1]),
           "Product 3-factor: g(challenges) matches")
    expect(frEq(mleH.evaluate(at: evalPoint), factorEvals[2]),
           "Product 3-factor: h(challenges) matches")
}

private func testProductSumcheckIdentityFactor() {
    // Product sumcheck where one factor is the all-ones polynomial
    // sum f(x) * 1 = sum f(x)
    let engine = GPUSumcheckProtocolEngine(config: .product2)

    let f: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]
    let ones: [Fr] = [Fr.one, Fr.one, Fr.one, Fr.one]
    let claimedSum = computeSum(f) // f * 1 = f

    let proverT = Transcript(label: "proto-prod-id")
    let proof = engine.proveProduct(factors: [f, ones], claimedSum: claimedSum, transcript: proverT)

    let verifierT = Transcript(label: "proto-prod-id")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyProduct(
        proof: proof, claimedSum: claimedSum, transcript: verifierT)
    expect(valid, "Product identity factor: verification passes")
}

private func testProductSumcheckVerifierRejects() {
    let engine = GPUSumcheckProtocolEngine(config: .product2)

    let f = randomEvals(2, seed: 0x5555)
    let g = randomEvals(2, seed: 0x6666)
    let correctSum = computeProductSum(f, g)
    let wrongSum = frAdd(correctSum, Fr.one)

    let proverT = Transcript(label: "proto-prod-bad")
    let proof = engine.proveProduct(factors: [f, g], claimedSum: correctSum, transcript: proverT)

    // Verify with wrong claimed sum should fail
    let verifierT = Transcript(label: "proto-prod-bad")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyProduct(
        proof: proof, claimedSum: wrongSum, transcript: verifierT)
    expect(!valid, "Product verifier rejects wrong claimed sum")
}

// MARK: - Combined Sumcheck Tests

private func testCombinedSumcheck2Claims() {
    let engine = GPUSumcheckProtocolEngine()

    let f0 = randomEvals(3, seed: 0x1000)
    let f1 = randomEvals(3, seed: 0x2000)
    let claim0 = SumcheckClaim.withComputedSum(evals: f0)
    let claim1 = SumcheckClaim.withComputedSum(evals: f1)

    let proverT = Transcript(label: "proto-combined2")
    let proof = engine.proveCombined(claims: [claim0, claim1], transcript: proverT)

    expectEqual(proof.combinationWeights.count, 2, "Combined 2: 2 weights")
    expectEqual(proof.individualFinalEvals.count, 2, "Combined 2: 2 individual final evals")
    expectEqual(proof.innerProof.numVars, 3, "Combined 2: numVars == 3")

    // Verify
    let verifierT = Transcript(label: "proto-combined2")
    let (valid, evalPoint, indivEvals) = GPUSumcheckProtocolEngine.verifyCombined(
        proof: proof,
        claimedSums: [claim0.claimedSum, claim1.claimedSum],
        transcript: verifierT)
    expect(valid, "Combined 2: verification passes")
    expectEqual(indivEvals.count, 2, "Combined 2: 2 individual evals returned")

    // Cross-check individual final evals
    let mle0 = MultilinearPoly(numVars: 3, evals: f0)
    let mle1 = MultilinearPoly(numVars: 3, evals: f1)
    expect(frEq(mle0.evaluate(at: evalPoint), indivEvals[0]),
           "Combined 2: f0 final eval matches MLE")
    expect(frEq(mle1.evaluate(at: evalPoint), indivEvals[1]),
           "Combined 2: f1 final eval matches MLE")
}

private func testCombinedSumcheck3Claims() {
    let engine = GPUSumcheckProtocolEngine()

    let f0 = randomEvals(2, seed: 0xA000)
    let f1 = randomEvals(2, seed: 0xB000)
    let f2 = randomEvals(2, seed: 0xC000)
    let claims = [
        SumcheckClaim.withComputedSum(evals: f0),
        SumcheckClaim.withComputedSum(evals: f1),
        SumcheckClaim.withComputedSum(evals: f2),
    ]

    let proverT = Transcript(label: "proto-combined3")
    let proof = engine.proveCombined(claims: claims, transcript: proverT)

    expectEqual(proof.combinationWeights.count, 3, "Combined 3: 3 weights")

    let verifierT = Transcript(label: "proto-combined3")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyCombined(
        proof: proof,
        claimedSums: claims.map { $0.claimedSum },
        transcript: verifierT)
    expect(valid, "Combined 3: verification passes")
}

private func testCombinedSumcheckSingleClaim() {
    // Combined with a single claim should still work
    let engine = GPUSumcheckProtocolEngine()

    let f = randomEvals(3, seed: 0xDDDD)
    let claim = SumcheckClaim.withComputedSum(evals: f)

    let proverT = Transcript(label: "proto-combined1")
    let proof = engine.proveCombined(claims: [claim], transcript: proverT)

    expectEqual(proof.combinationWeights.count, 1, "Combined single: 1 weight")

    let verifierT = Transcript(label: "proto-combined1")
    let (valid, evalPoint, indivEvals) = GPUSumcheckProtocolEngine.verifyCombined(
        proof: proof,
        claimedSums: [claim.claimedSum],
        transcript: verifierT)
    expect(valid, "Combined single: verification passes")

    let mle = MultilinearPoly(numVars: 3, evals: f)
    expect(frEq(mle.evaluate(at: evalPoint), indivEvals[0]),
           "Combined single: MLE consistency")
}

// MARK: - GPU Accelerated Tests

private func testGPUAcceleratedStandard() {
    // Test GPU-accelerated path (falls back to CPU if no GPU)
    let engine = GPUSumcheckProtocolEngine()
    let logSize = 5 // 32 elements (below threshold, will use CPU fallback)
    let evals = randomEvals(logSize, seed: 0xFEED)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    do {
        let proverT = Transcript(label: "proto-gpu")
        let gpuProof = try engine.proveStandardGPU(claim: claim, transcript: proverT)

        let verifierT = Transcript(label: "proto-gpu")
        let (valid, evalPoint, finalEval) = GPUSumcheckProtocolEngine.verifyStandard(
            proof: gpuProof, claimedSum: claim.claimedSum, transcript: verifierT)
        expect(valid, "GPU standard: verification passes")

        let mle = MultilinearPoly(numVars: logSize, evals: evals)
        expect(frEq(mle.evaluate(at: evalPoint), finalEval),
               "GPU standard: MLE consistency")

        // Compare with CPU proof
        let cpuT = Transcript(label: "proto-gpu")
        let cpuProof = engine.proveStandard(claim: claim, transcript: cpuT)

        // Final evals should match since both start from same transcript state
        expect(frEq(gpuProof.finalEval, cpuProof.finalEval),
               "GPU standard: finalEval matches CPU")
        expectEqual(gpuProof.numVars, cpuProof.numVars,
                    "GPU standard: numVars matches CPU")
    } catch {
        expect(false, "GPU standard: unexpected error \(error)")
    }
}

// MARK: - Eq Poly Tests

private func testEqPolyGeneration() {
    // eq(x, r) at a specific point r = (1, 0)
    // eq((0,0), (1,0)) = (1-1)*(1-0) = 0
    // eq((0,1), (1,0)) = (1-1)*(0) = 0
    // eq((1,0), (1,0)) = 1*1 = 1
    // eq((1,1), (1,0)) = 1*0 = 0
    let point: [Fr] = [Fr.one, Fr.zero]
    let eqEvals = GPUSumcheckProtocolEngine.eqPoly(point: point)

    expectEqual(eqEvals.count, 4, "Eq poly (1,0): 4 evaluations")
    expect(frEq(eqEvals[0], Fr.zero), "Eq poly (1,0): eq(0,0) = 0")
    expect(frEq(eqEvals[1], Fr.zero), "Eq poly (1,0): eq(0,1) = 0")
    expect(frEq(eqEvals[2], Fr.one), "Eq poly (1,0): eq(1,0) = 1")
    expect(frEq(eqEvals[3], Fr.zero), "Eq poly (1,0): eq(1,1) = 0")
}

private func testEqPolyProductCheck() {
    // For any point r, sum_{x in {0,1}^n} eq(x, r) = 1
    let two = frAdd(Fr.one, Fr.one)
    let three = frFromInt(3)
    let point: [Fr] = [two, three, frFromInt(5)]
    let eqEvals = GPUSumcheckProtocolEngine.eqPoly(point: point)

    expectEqual(eqEvals.count, 8, "Eq poly product: 8 evaluations")

    let total = computeSum(eqEvals)
    expect(frEq(total, Fr.one), "Eq poly product: sum over hypercube == 1")
}

// MARK: - Multilinear Evaluation Tests

private func testMultilinearEvaluation() {
    // f(x0, x1) = 3*(1-x0)*(1-x1) + 7*(1-x0)*x1 + 2*x0*(1-x1) + 5*x0*x1
    // f(0,0)=3, f(0,1)=7, f(1,0)=2, f(1,1)=5
    let evals: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]

    // Evaluate at (0, 0) should give 3
    let v00 = GPUSumcheckProtocolEngine.evaluateMultilinear(evals: evals, at: [Fr.zero, Fr.zero])
    expect(frEq(v00, frFromInt(3)), "MLE eval at (0,0) = 3")

    // Evaluate at (1, 1) should give 5
    let v11 = GPUSumcheckProtocolEngine.evaluateMultilinear(evals: evals, at: [Fr.one, Fr.one])
    expect(frEq(v11, frFromInt(5)), "MLE eval at (1,1) = 5")

    // Evaluate at (0, 1) should give 7
    let v01 = GPUSumcheckProtocolEngine.evaluateMultilinear(evals: evals, at: [Fr.zero, Fr.one])
    expect(frEq(v01, frFromInt(7)), "MLE eval at (0,1) = 7")

    // Evaluate at (1, 0) should give 2
    let v10 = GPUSumcheckProtocolEngine.evaluateMultilinear(evals: evals, at: [Fr.one, Fr.zero])
    expect(frEq(v10, frFromInt(2)), "MLE eval at (1,0) = 2")
}

private func testMultilinearEvalConsistency() {
    // Our evaluateMultilinear should match MultilinearPoly.evaluate
    let evals = randomEvals(4, seed: 0x9999)
    var rng: UInt64 = 0x7777
    let point = (0..<4).map { _ in pseudoRandom(seed: &rng) }

    let engineEval = GPUSumcheckProtocolEngine.evaluateMultilinear(evals: evals, at: point)
    let mle = MultilinearPoly(numVars: 4, evals: evals)
    let mleEval = mle.evaluate(at: point)

    expect(frEq(engineEval, mleEval), "evaluateMultilinear matches MultilinearPoly.evaluate")
}

// MARK: - Verifier Rejection Tests

private func testVerifierRejectsBadSum() {
    let engine = GPUSumcheckProtocolEngine()
    let evals = randomEvals(3, seed: 0xBAAD)
    let correctSum = computeSum(evals)
    let claim = SumcheckClaim(evals: evals, claimedSum: correctSum)

    let proverT = Transcript(label: "proto-bad-sum")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    // Try to verify with wrong sum
    let wrongSum = frAdd(correctSum, Fr.one)
    let verifierT = Transcript(label: "proto-bad-sum")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: wrongSum, transcript: verifierT)
    expect(!valid, "Verifier rejects bad claimed sum")
}

private func testVerifierRejectsBadRoundPoly() {
    let engine = GPUSumcheckProtocolEngine()
    let evals: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(2), frFromInt(5)]
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let proverT = Transcript(label: "proto-bad-rp")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    // Tamper with round poly: change first round poly eval
    let tamperedS0 = frAdd(proof.roundPolys[0].atZero, Fr.one)
    let tamperedRp = RoundUnivariate(
        evals: [tamperedS0, proof.roundPolys[0].atOne],
        degreeBound: proof.degreeBound)
    var tamperedPolys = proof.roundPolys
    tamperedPolys[0] = tamperedRp

    let tamperedProof = SumcheckProtocolProof(
        roundPolys: tamperedPolys,
        challenges: proof.challenges,
        finalEval: proof.finalEval,
        numVars: proof.numVars,
        factorFinalEvals: nil,
        degreeBound: proof.degreeBound)

    let verifierT = Transcript(label: "proto-bad-rp")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: tamperedProof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(!valid, "Verifier rejects tampered round poly")
}

// MARK: - Config Tests

private func testDegreeBoundConfig() {
    let cfg1 = SumcheckProtocolConfig.standard
    expectEqual(cfg1.degreeBound, 1, "Standard config: degreeBound == 1")
    expect(cfg1.useGPU, "Standard config: useGPU == true")

    let cfg2 = SumcheckProtocolConfig.product2
    expectEqual(cfg2.degreeBound, 2, "Product2 config: degreeBound == 2")

    let cfg3 = SumcheckProtocolConfig.product3
    expectEqual(cfg3.degreeBound, 3, "Product3 config: degreeBound == 3")

    let custom = SumcheckProtocolConfig(degreeBound: 5, gpuThreshold: 2048, useGPU: false)
    expectEqual(custom.degreeBound, 5, "Custom config: degreeBound == 5")
    expectEqual(custom.gpuThreshold, 2048, "Custom config: gpuThreshold == 2048")
    expect(!custom.useGPU, "Custom config: useGPU == false")
}

private func testCPUOnlyConfig() {
    let engine = GPUSumcheckProtocolEngine(config: .cpuOnly)
    // CPU-only engine should not have GPU
    // (it will still have GPU if Metal is available, but it skips GPU path)
    // Just verify it produces correct results
    let evals = randomEvals(3, seed: 0xC0B0)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let proverT = Transcript(label: "proto-cpuonly")
    let proof = engine.proveStandard(claim: claim, transcript: proverT)

    let verifierT = Transcript(label: "proto-cpuonly")
    let (valid, _, _) = GPUSumcheckProtocolEngine.verifyStandard(
        proof: proof, claimedSum: claim.claimedSum, transcript: verifierT)
    expect(valid, "CPU-only config: verification passes")
}

private func testStandardSumcheckDeterministic() {
    // Running the same sumcheck twice should produce identical proofs
    let engine = GPUSumcheckProtocolEngine()
    let evals = randomEvals(3, seed: 0xDE7E)
    let claim = SumcheckClaim.withComputedSum(evals: evals)

    let t1 = Transcript(label: "proto-det")
    let proof1 = engine.proveStandard(claim: claim, transcript: t1)

    let t2 = Transcript(label: "proto-det")
    let proof2 = engine.proveStandard(claim: claim, transcript: t2)

    expect(frEq(proof1.finalEval, proof2.finalEval),
           "Deterministic: finalEval matches across runs")
    expectEqual(proof1.challenges.count, proof2.challenges.count,
                "Deterministic: same number of challenges")

    for i in 0..<proof1.challenges.count {
        expect(frEq(proof1.challenges[i], proof2.challenges[i]),
               "Deterministic: challenge \(i) matches")
    }

    for i in 0..<proof1.roundPolys.count {
        expect(proof1.roundPolys[i] == proof2.roundPolys[i],
               "Deterministic: round poly \(i) matches")
    }
}
