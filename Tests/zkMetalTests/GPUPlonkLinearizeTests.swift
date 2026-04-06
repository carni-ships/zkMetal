// GPUPlonkLinearizeTests — Comprehensive tests for GPU-accelerated Plonk linearization engine
//
// Tests cover:
//   1.  Linearization scalar computation (gate scalars)
//   2.  Linearization scalar computation (permutation scalars)
//   3.  Basic linearization polynomial construction
//   4.  Gate-only linearization (addition gate)
//   5.  Gate-only linearization (multiplication gate)
//   6.  Permutation linearization contribution
//   7.  Boundary constraint in linearization (L_1 computation)
//   8.  Combined linearization (gate + perm + boundary)
//   9.  Linearization evaluation at zeta (scalar-only path)
//  10.  Linearization identity verification
//  11.  Quotient consistency check
//  12.  Batch evaluation of linearization polynomial
//  13.  Range gate linearization contribution
//  14.  Lookup gate linearization contribution
//  15.  Poseidon gate linearization contribution
//  16.  All custom gates combined
//  17.  Zero polynomial linearization
//  18.  Constant polynomial linearization
//  19.  Large domain linearization scaling
//  20.  CPU scalar-poly sum correctness

import zkMetal
import Foundation

// MARK: - Test entry point

public func runGPUPlonkLinearizeTests() {
    suite("GPU Plonk Linearize Engine")
    testGateScalars()
    testPermutationScalars()
    testBasicLinearization()
    testGateOnlyLinearizationAddition()
    testGateOnlyLinearizationMultiplication()
    testPermutationLinearization()
    testL1Computation()
    testCombinedLinearization()
    testLinearizationEvalAtZeta()
    testLinearizationIdentityVerification()
    testQuotientConsistency()
    testBatchEvaluation()
    testRangeGateLinearization()
    testLookupGateLinearization()
    testPoseidonGateLinearization()
    testAllCustomGatesCombined()
    testZeroPolynomialLinearization()
    testConstantPolynomialLinearization()
    testLargerDomainLinearization()
    testCpuScalarPolySumCorrectness()
}

// MARK: - Helpers

private let engine = GPUPlonkLinearizeEngine()

/// Simple deterministic pseudo-random Fr generator for tests.
private struct LinearizeTestRNG {
    var state: UInt64

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state >> 33
    }

    mutating func nextFr() -> Fr {
        let v = next() & 0xFFFFFFF
        return frFromInt(v)
    }
}

/// Build a standard eval input with given wire values and default challenges
private func makeEvalInput(a: Fr, b: Fr, c: Fr,
                           sigma1: Fr, sigma2: Fr, zOmega: Fr,
                           alpha: Fr, beta: Fr, gamma: Fr, zeta: Fr,
                           k1: Fr, k2: Fr, n: Int) -> LinearizationEvalInput {
    return LinearizationEvalInput(
        aEval: a, bEval: b, cEval: c,
        sigma1Eval: sigma1, sigma2Eval: sigma2, zOmegaEval: zOmega,
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )
}

/// Build a simple poly input where all polynomials are given constant coefficients
private func makeConstantPolyInput(n: Int, qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr,
                                   sigma3: Fr = Fr.zero, z: Fr = Fr.zero) -> LinearizationPolyInput {
    let makeConst: (Fr) -> [Fr] = { val in
        var c = [Fr](repeating: Fr.zero, count: n)
        c[0] = val
        return c
    }
    return LinearizationPolyInput(
        qLCoeffs: makeConst(qL), qRCoeffs: makeConst(qR),
        qOCoeffs: makeConst(qO), qMCoeffs: makeConst(qM), qCCoeffs: makeConst(qC),
        qRangeCoeffs: makeConst(Fr.zero), qLookupCoeffs: makeConst(Fr.zero),
        qPoseidonCoeffs: makeConst(Fr.zero),
        sigma3Coeffs: makeConst(sigma3), zCoeffs: makeConst(z)
    )
}

/// Build a poly input with full coefficient arrays
private func makePolyInput(qL: [Fr], qR: [Fr], qO: [Fr], qM: [Fr], qC: [Fr],
                           qRange: [Fr], qLookup: [Fr], qPoseidon: [Fr],
                           sigma3: [Fr], z: [Fr]) -> LinearizationPolyInput {
    return LinearizationPolyInput(
        qLCoeffs: qL, qRCoeffs: qR, qOCoeffs: qO,
        qMCoeffs: qM, qCCoeffs: qC,
        qRangeCoeffs: qRange, qLookupCoeffs: qLookup, qPoseidonCoeffs: qPoseidon,
        sigma3Coeffs: sigma3, zCoeffs: z
    )
}

/// Evaluate polynomial at a point (Horner)
private func evalPoly(_ coeffs: [Fr], at x: Fr) -> Fr {
    return polyEval(coeffs, at: x)
}

// MARK: - Test 1: Gate Scalars

private func testGateScalars() {
    // Gate scalars should be: qL -> a, qR -> b, qO -> c, qM -> a*b, qC -> 1
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.one, beta: Fr.zero, gamma: Fr.zero,
        zeta: frFromInt(7), k1: frFromInt(2), k2: frFromInt(3), n: 4
    )
    let config = LinearizationConfig(domainSize: 4,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)
    let scalars = engine.computeScalars(input: input, config: config)

    expect(frEqual(scalars.qLScalar, a), "qL scalar = a_eval")
    expect(frEqual(scalars.qRScalar, b), "qR scalar = b_eval")
    expect(frEqual(scalars.qOScalar, c), "qO scalar = c_eval")
    expect(frEqual(scalars.qMScalar, frMul(a, b)), "qM scalar = a_eval * b_eval")
    expect(frEqual(scalars.qCScalar, Fr.one), "qC scalar = 1")
}

// MARK: - Test 2: Permutation Scalars

private func testPermutationScalars() {
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let alpha = frFromInt(11)
    let beta = frFromInt(2)
    let gamma = frFromInt(7)
    let zeta = frFromInt(13)
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let sigma1 = frFromInt(4)
    let sigma2 = frFromInt(6)
    let zOmega = frFromInt(9)
    let n = 4

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: sigma1, sigma2: sigma2, zOmega: zOmega,
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)
    let scalars = engine.computeScalars(input: input, config: config)

    // Verify permutation numerator: (a + beta*zeta + gamma)(b + beta*k1*zeta + gamma)(c + beta*k2*zeta + gamma)
    let permA = frAdd(frAdd(a, frMul(beta, zeta)), gamma)
    let permB = frAdd(frAdd(b, frMul(beta, frMul(k1, zeta))), gamma)
    let permC = frAdd(frAdd(c, frMul(beta, frMul(k2, zeta))), gamma)
    let permNum = frMul(frMul(permA, permB), permC)

    // L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
    let l1Zeta = engine.computeL1(zeta: zeta, n: n)

    // z_scalar = alpha * permNum + alpha^2 * L_1(zeta)
    let alpha2 = frSqr(alpha)
    let expectedZ = frAdd(frMul(alpha, permNum), frMul(alpha2, l1Zeta))
    expect(frEqual(scalars.zScalar, expectedZ), "Z scalar matches expected")

    // sigma3_scalar = -alpha * (a+beta*s1+gamma)(b+beta*s2+gamma) * beta * z_omega
    let sigA = frAdd(frAdd(a, frMul(beta, sigma1)), gamma)
    let sigB = frAdd(frAdd(b, frMul(beta, sigma2)), gamma)
    let expectedSig3 = frNeg(frMul(alpha, frMul(frMul(sigA, sigB), frMul(beta, zOmega))))
    expect(frEqual(scalars.sigma3Scalar, expectedSig3), "sigma3 scalar matches expected")
}

// MARK: - Test 3: Basic Linearization

private func testBasicLinearization() {
    // Simple case: single addition gate with constant selectors
    // qL=1, qR=1, qO=-1, qM=0, qC=0
    // Wire evals: a=3, b=5, c=8
    let n = 4
    let negOne = frNeg(Fr.one)
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let zeta = frFromInt(7)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let polyInput = makeConstantPolyInput(n: n, qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let result = engine.computeLinearization(evalInput: input, polyInput: polyInput, config: config)

    // r(X) = a*qL(X) + b*qR(X) + c*qO(X) + ab*qM(X) + qC(X)
    // With constant selectors: r(X) = 3*1 + 5*1 + 8*(-1) + 0 + 0 = 0 (constant poly)
    // So r(zeta) should be 0 (the gate is satisfied)
    let gateContrib = frAdd(frAdd(frMul(a, Fr.one), frMul(b, Fr.one)), frMul(c, negOne))
    expect(frEqual(result.rCoeffs[0], gateContrib), "r(X) constant term for addition gate")

    // r(zeta) should match polyEval of rCoeffs
    let rAtZeta = evalPoly(result.rCoeffs, at: zeta)
    expect(frEqual(result.rEval, rAtZeta), "r(zeta) matches polynomial evaluation")
}

// MARK: - Test 4: Gate-Only Linearization (Addition)

private func testGateOnlyLinearizationAddition() {
    // Gate: a + b - c = 0 with a=3, b=5, c=8 (satisfied)
    let n = 4
    let negOne = frNeg(Fr.one)
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let zeta = frFromInt(11)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let polyInput = makeConstantPolyInput(n: n, qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    let gateCoeffs = engine.computeGateLinearization(evalInput: input, polyInput: polyInput, size: n)

    // Gate linearization with constant selectors: r_gate(X) = a*1 + b*1 + c*(-1) = 0
    expect(frEqual(gateCoeffs[0], Fr.zero), "Addition gate linearization = 0 when satisfied")
    for i in 1..<n {
        expect(gateCoeffs[i].isZero, "Higher coeffs zero for constant selectors [\(i)]")
    }
}

// MARK: - Test 5: Gate-Only Linearization (Multiplication)

private func testGateOnlyLinearizationMultiplication() {
    // Gate: a*b - c = 0 with a=4, b=7, c=28
    let n = 4
    let negOne = frNeg(Fr.one)
    let a = frFromInt(4)
    let b = frFromInt(7)
    let c = frFromInt(28)
    let zeta = frFromInt(13)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let polyInput = makeConstantPolyInput(n: n, qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)

    let gateCoeffs = engine.computeGateLinearization(evalInput: input, polyInput: polyInput, size: n)

    // r_gate(X) = a*b * qM(X) + c*(-1)*qO(X) = 28 * 1 + 28 * (-1) = 0
    expect(frEqual(gateCoeffs[0], Fr.zero), "Multiplication gate linearization = 0 when satisfied")
}

// MARK: - Test 6: Permutation Linearization

private func testPermutationLinearization() {
    let n = 4
    let alpha = frFromInt(11)
    let beta = frFromInt(2)
    let gamma = frFromInt(7)
    let zeta = frFromInt(13)
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let sigma1 = frFromInt(4)
    let sigma2 = frFromInt(6)
    let zOmega = frFromInt(9)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: sigma1, sigma2: sigma2, zOmega: zOmega,
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )

    // Z(X) = constant polynomial = 1 (simplified for test)
    // sigma3(X) = constant polynomial = some value
    let sigma3Val = frFromInt(10)
    var zCoeffs = [Fr](repeating: Fr.zero, count: n)
    zCoeffs[0] = Fr.one
    var sig3Coeffs = [Fr](repeating: Fr.zero, count: n)
    sig3Coeffs[0] = sigma3Val

    let polyInput = makePolyInput(
        qL: [Fr](repeating: Fr.zero, count: n),
        qR: [Fr](repeating: Fr.zero, count: n),
        qO: [Fr](repeating: Fr.zero, count: n),
        qM: [Fr](repeating: Fr.zero, count: n),
        qC: [Fr](repeating: Fr.zero, count: n),
        qRange: [Fr](repeating: Fr.zero, count: n),
        qLookup: [Fr](repeating: Fr.zero, count: n),
        qPoseidon: [Fr](repeating: Fr.zero, count: n),
        sigma3: sig3Coeffs, z: zCoeffs
    )

    let permCoeffs = engine.computePermutationLinearization(evalInput: input, polyInput: polyInput, size: n)

    // Verify: permCoeffs should be zScalar * Z + sigma3Scalar * sigma3
    let config = LinearizationConfig(domainSize: n, enableRangeGates: false,
                                      enableLookupGates: false, enablePoseidonGates: false)
    let scalars = engine.computeScalars(input: input, config: config)

    let expectedConst = frAdd(frMul(scalars.zScalar, Fr.one), frMul(scalars.sigma3Scalar, sigma3Val))
    expect(frEqual(permCoeffs[0], expectedConst), "Permutation linearization constant term")
}

// MARK: - Test 7: L_1 Computation

private func testL1Computation() {
    // L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
    let zeta = frFromInt(7)
    let n = 4

    let l1 = engine.computeL1(zeta: zeta, n: n)

    // Manual computation: zeta^4 = 7^4 = 2401
    // (2401 - 1) / (4 * (7 - 1)) = 2400 / 24 = 100
    let zetaN = frPow(zeta, 4)
    let num = frSub(zetaN, Fr.one)
    let den = frMul(frFromInt(4), frSub(zeta, Fr.one))
    let expected = frMul(num, frInverse(den))

    expect(frEqual(l1, expected), "L_1(zeta) = (zeta^n-1)/(n*(zeta-1))")
    expect(frEqual(l1, frFromInt(100)), "L_1(7) with n=4 = 100")
}

// MARK: - Test 8: Combined Linearization

private func testCombinedLinearization() {
    let n = 4
    let negOne = frNeg(Fr.one)
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let alpha = frFromInt(11)
    let beta = frFromInt(2)
    let gamma = frFromInt(7)
    let zeta = frFromInt(13)
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: frFromInt(4), sigma2: frFromInt(6), zOmega: frFromInt(9),
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )

    // Use non-trivial polynomials: qL(X) = 1 + X, qR(X) = 1, etc.
    var qLCoeffs = [Fr](repeating: Fr.zero, count: n)
    qLCoeffs[0] = Fr.one; qLCoeffs[1] = Fr.one
    var qRCoeffs = [Fr](repeating: Fr.zero, count: n)
    qRCoeffs[0] = Fr.one
    var qOCoeffs = [Fr](repeating: Fr.zero, count: n)
    qOCoeffs[0] = negOne
    var qMCoeffs = [Fr](repeating: Fr.zero, count: n)
    var qCCoeffs = [Fr](repeating: Fr.zero, count: n)
    var zCoeffs = [Fr](repeating: Fr.zero, count: n)
    zCoeffs[0] = Fr.one
    var sig3Coeffs = [Fr](repeating: Fr.zero, count: n)
    sig3Coeffs[0] = frFromInt(10)

    let polyInput = makePolyInput(
        qL: qLCoeffs, qR: qRCoeffs, qO: qOCoeffs, qM: qMCoeffs, qC: qCCoeffs,
        qRange: [Fr](repeating: Fr.zero, count: n),
        qLookup: [Fr](repeating: Fr.zero, count: n),
        qPoseidon: [Fr](repeating: Fr.zero, count: n),
        sigma3: sig3Coeffs, z: zCoeffs
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let result = engine.computeLinearization(evalInput: input, polyInput: polyInput, config: config)

    // Verify r(zeta) by computing manually
    let rManual = evalPoly(result.rCoeffs, at: zeta)
    expect(frEqual(result.rEval, rManual), "Combined linearization: r(zeta) matches poly eval")
    expect(result.rCoeffs.count == n, "Result has correct size")
}

// MARK: - Test 9: Linearization Evaluation at Zeta (Scalar-Only)

private func testLinearizationEvalAtZeta() {
    let n = 4
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let alpha = frFromInt(11)
    let beta = frFromInt(2)
    let gamma = frFromInt(7)
    let zeta = frFromInt(13)
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: frFromInt(4), sigma2: frFromInt(6), zOmega: frFromInt(9),
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )

    // Selector evals at zeta: qL(zeta)=1, qR(zeta)=1, qO(zeta)=-1, qM(zeta)=0, qC(zeta)=0
    let negOne = frNeg(Fr.one)
    let selectorEvals = [Fr.one, Fr.one, negOne, Fr.zero, Fr.zero]
    let customSelectorEvals: [Fr] = []
    let sigma3Eval = frFromInt(10)
    let zEval = Fr.one

    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let rEval = engine.evaluateLinearizationAtZeta(
        input: input, selectorEvals: selectorEvals,
        customSelectorEvals: customSelectorEvals,
        sigma3Eval: sigma3Eval, zEval: zEval, config: config
    )

    // Manual: gate part = a*1 + b*1 + c*(-1) + ab*0 + 0 = 3+5-8 = 0
    let gatePart = frAdd(frAdd(frMul(a, Fr.one), frMul(b, Fr.one)), frMul(c, negOne))
    let scalars = engine.computeScalars(input: input, config: config)
    let permPart = frAdd(frMul(scalars.zScalar, zEval), frMul(scalars.sigma3Scalar, sigma3Eval))
    let expected = frAdd(gatePart, permPart)

    expect(frEqual(rEval, expected), "Scalar-only r(zeta) matches manual computation")
}

// MARK: - Test 10: Linearization Identity Verification

private func testLinearizationIdentityVerification() {
    let n = 4
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let zeta = frFromInt(13)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: frFromInt(4), sigma2: frFromInt(6), zOmega: frFromInt(9),
        alpha: frFromInt(11), beta: frFromInt(2), gamma: frFromInt(7), zeta: zeta,
        k1: frFromInt(2), k2: frFromInt(3), n: n
    )

    let negOne = frNeg(Fr.one)
    let selectorEvals = [Fr.one, Fr.one, negOne, Fr.zero, Fr.zero]
    let sigma3Eval = frFromInt(10)
    let zEval = Fr.one
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let rEval = engine.evaluateLinearizationAtZeta(
        input: input, selectorEvals: selectorEvals,
        customSelectorEvals: [], sigma3Eval: sigma3Eval, zEval: zEval, config: config
    )

    // Should verify against itself
    let verified = engine.verifyLinearizationIdentity(
        rEval: rEval, input: input,
        selectorEvals: selectorEvals, customSelectorEvals: [],
        sigma3Eval: sigma3Eval, zEval: zEval, config: config
    )
    expect(verified, "Linearization identity verifies with correct r(zeta)")

    // Should fail with wrong r(zeta)
    let wrongReval = frAdd(rEval, Fr.one)
    let failedVerify = engine.verifyLinearizationIdentity(
        rEval: wrongReval, input: input,
        selectorEvals: selectorEvals, customSelectorEvals: [],
        sigma3Eval: sigma3Eval, zEval: zEval, config: config
    )
    expect(!failedVerify, "Linearization identity fails with wrong r(zeta)")
}

// MARK: - Test 11: Quotient Consistency

private func testQuotientConsistency() {
    // Construct a known r(zeta) and quotient chunks such that r(zeta) = Z_H(zeta) * t(zeta)
    let n = 4
    let zeta = frFromInt(7)

    // Z_H(zeta) = zeta^n - 1 = 7^4 - 1 = 2400
    let zetaN = frPow(zeta, UInt64(n))
    let zhZeta = frSub(zetaN, Fr.one)

    // Build a simple quotient: t(X) = 5 (constant)
    let tConst = frFromInt(5)
    var chunk0 = [Fr](repeating: Fr.zero, count: n)
    chunk0[0] = tConst

    // r(zeta) should be Z_H(zeta) * t(zeta) = 2400 * 5 = 12000
    let rEval = frMul(zhZeta, tConst)

    let verified = engine.verifyQuotientConsistency(
        rEval: rEval, quotientChunks: [chunk0], zeta: zeta, n: n
    )
    expect(verified, "Quotient consistency holds for known r(zeta)")

    // Wrong r(zeta) should fail
    let wrongR = frAdd(rEval, Fr.one)
    let failed = engine.verifyQuotientConsistency(
        rEval: wrongR, quotientChunks: [chunk0], zeta: zeta, n: n
    )
    expect(!failed, "Quotient consistency fails with wrong r(zeta)")
}

// MARK: - Test 12: Batch Evaluation

private func testBatchEvaluation() {
    // Build a polynomial r(X) = 3X^2 + 5X + 1
    let n = 4
    var rCoeffs = [Fr](repeating: Fr.zero, count: n)
    rCoeffs[0] = frFromInt(1)
    rCoeffs[1] = frFromInt(5)
    rCoeffs[2] = frFromInt(3)

    let points = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(7)]
    let result = engine.batchEvaluate(rCoeffs: rCoeffs, points: points)

    // r(0) = 1, r(1) = 1+5+3 = 9, r(2) = 1+10+12 = 23, r(7) = 1+35+147 = 183
    expect(frEqual(result.evaluations[0], frFromInt(1)), "Batch eval: r(0) = 1")
    expect(frEqual(result.evaluations[1], frFromInt(9)), "Batch eval: r(1) = 9")
    expect(frEqual(result.evaluations[2], frFromInt(23)), "Batch eval: r(2) = 23")
    expect(frEqual(result.evaluations[3], frFromInt(183)), "Batch eval: r(7) = 183")
    expectEqual(result.points.count, 4, "Batch eval: 4 points")
}

// MARK: - Test 13: Range Gate Linearization

private func testRangeGateLinearization() {
    let n = 4
    let a = frFromInt(0)  // Boolean: 0
    let b = frFromInt(5)
    let c = frFromInt(8)
    let zeta = frFromInt(7)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: true,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)
    let scalars = engine.computeScalars(input: input, config: config)

    // Range scalar: a*(1-a) = 0*(1-0) = 0
    expect(frEqual(scalars.qRangeScalar, Fr.zero), "Range scalar = 0 when a=0")

    // Test with a=1: range scalar should also be 0
    let input1 = makeEvalInput(
        a: Fr.one, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let scalars1 = engine.computeScalars(input: input1, config: config)
    expect(frEqual(scalars1.qRangeScalar, Fr.zero), "Range scalar = 0 when a=1")

    // Test with a=2: range scalar should be 2*(1-2) = -2
    let input2 = makeEvalInput(
        a: frFromInt(2), b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let scalars2 = engine.computeScalars(input: input2, config: config)
    let expected2 = frMul(frFromInt(2), frSub(Fr.one, frFromInt(2)))
    expect(frEqual(scalars2.qRangeScalar, expected2), "Range scalar = 2*(1-2) = -2 when a=2")
}

// MARK: - Test 14: Lookup Gate Linearization

private func testLookupGateLinearization() {
    let n = 4
    let a = frFromInt(3)
    let zeta = frFromInt(7)

    // Table values: [1, 2, 3]
    let tableValues = [frFromInt(1), frFromInt(2), frFromInt(3)]

    let input = makeEvalInput(
        a: a, b: frFromInt(5), c: frFromInt(8),
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: true,
                                      enablePoseidonGates: false,
                                      lookupTableValues: tableValues)
    let scalars = engine.computeScalars(input: input, config: config)

    // Lookup scalar: prod(a - t_i) = (3-1)*(3-2)*(3-3) = 2*1*0 = 0
    // When a is in the table, the scalar is 0 (gate satisfied)
    expect(frEqual(scalars.qLookupScalar, Fr.zero), "Lookup scalar = 0 when a is in table")

    // Test with a=5 (not in table): prod(5-t_i) = (5-1)*(5-2)*(5-3) = 4*3*2 = 24
    let input2 = makeEvalInput(
        a: frFromInt(5), b: frFromInt(5), c: frFromInt(8),
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let scalars2 = engine.computeScalars(input: input2, config: config)
    expect(frEqual(scalars2.qLookupScalar, frFromInt(24)), "Lookup scalar = 24 when a=5 not in table")
}

// MARK: - Test 15: Poseidon Gate Linearization

private func testPoseidonGateLinearization() {
    let n = 4
    // c = a^5: with a=2, c=32
    let a = frFromInt(2)
    let c = frFromInt(32)
    let zeta = frFromInt(7)

    let input = makeEvalInput(
        a: a, b: frFromInt(5), c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: true)
    let scalars = engine.computeScalars(input: input, config: config)

    // Poseidon scalar: c - a^5 = 32 - 32 = 0
    expect(frEqual(scalars.qPoseidonScalar, Fr.zero), "Poseidon scalar = 0 when c = a^5")

    // Test with wrong c: c=33, a^5=32 => scalar = 33-32 = 1
    let inputWrong = makeEvalInput(
        a: a, b: frFromInt(5), c: frFromInt(33),
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let scalarsWrong = engine.computeScalars(input: inputWrong, config: config)
    expect(frEqual(scalarsWrong.qPoseidonScalar, Fr.one), "Poseidon scalar = 1 when c = a^5 + 1")
}

// MARK: - Test 16: All Custom Gates Combined

private func testAllCustomGatesCombined() {
    let n = 4
    let a = frFromInt(1)  // boolean, in table {1,2,3}, and a^5=1 so c should be 1
    let b = frFromInt(5)
    let c = Fr.one
    let zeta = frFromInt(7)

    let tableValues = [frFromInt(1), frFromInt(2), frFromInt(3)]
    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: true,
                                      enableLookupGates: true,
                                      enablePoseidonGates: true,
                                      lookupTableValues: tableValues)
    let scalars = engine.computeScalars(input: input, config: config)

    // a=1: range scalar = 1*(1-1) = 0, lookup = (1-1)*(1-2)*(1-3) = 0, poseidon = 1 - 1^5 = 0
    expect(frEqual(scalars.qRangeScalar, Fr.zero), "All custom gates: range scalar = 0")
    expect(frEqual(scalars.qLookupScalar, Fr.zero), "All custom gates: lookup scalar = 0")
    expect(frEqual(scalars.qPoseidonScalar, Fr.zero), "All custom gates: poseidon scalar = 0")
}

// MARK: - Test 17: Zero Polynomial Linearization

private func testZeroPolynomialLinearization() {
    let n = 4
    let zeta = frFromInt(7)

    let input = makeEvalInput(
        a: Fr.zero, b: Fr.zero, c: Fr.zero,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )
    let polyInput = makeConstantPolyInput(n: n, qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                                           qM: Fr.zero, qC: Fr.zero)
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let result = engine.computeLinearization(evalInput: input, polyInput: polyInput, config: config)

    // All zero inputs with alpha=0 => r(X) should be zero
    for i in 0..<n {
        expect(result.rCoeffs[i].isZero, "Zero linearization: coeff[\(i)] = 0")
    }
    expect(result.rEval.isZero, "Zero linearization: r(zeta) = 0")
}

// MARK: - Test 18: Constant Polynomial Linearization

private func testConstantPolynomialLinearization() {
    // If all selectors are constant and alpha=0, r(X) should be a constant
    let n = 4
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(15) // a*b = 15
    let zeta = frFromInt(7)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: Fr.zero, sigma2: Fr.zero, zOmega: Fr.zero,
        alpha: Fr.zero, beta: Fr.zero, gamma: Fr.zero,
        zeta: zeta, k1: frFromInt(2), k2: frFromInt(3), n: n
    )

    // qM=1, qO=-1: constraint is a*b - c = 15 - 15 = 0
    let negOne = frNeg(Fr.one)
    let polyInput = makeConstantPolyInput(n: n, qL: Fr.zero, qR: Fr.zero, qO: negOne,
                                           qM: Fr.one, qC: Fr.zero)
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: false,
                                      enableLookupGates: false,
                                      enablePoseidonGates: false)

    let result = engine.computeLinearization(evalInput: input, polyInput: polyInput, config: config)

    // r(X) = ab * qM(X) + c * qO(X) = 15 * [1,0,0,0] + 15 * [-1,0,0,0] = [0,0,0,0]
    expect(result.rCoeffs[0].isZero, "Constant poly linearization: satisfied gate => r=0")
    expect(result.rEval.isZero, "Constant poly linearization: r(zeta) = 0")
}

// MARK: - Test 19: Larger Domain Linearization

private func testLargerDomainLinearization() {
    let n = 16
    let a = frFromInt(3)
    let b = frFromInt(5)
    let c = frFromInt(8)
    let alpha = frFromInt(11)
    let beta = frFromInt(2)
    let gamma = frFromInt(7)
    let zeta = frFromInt(17)
    let k1 = frFromInt(2)
    let k2 = frFromInt(3)

    let input = makeEvalInput(
        a: a, b: b, c: c,
        sigma1: frFromInt(4), sigma2: frFromInt(6), zOmega: frFromInt(9),
        alpha: alpha, beta: beta, gamma: gamma, zeta: zeta,
        k1: k1, k2: k2, n: n
    )

    // Generate random-ish polynomials for the selectors
    var rng = LinearizeTestRNG(state: 42)
    let genPoly: () -> [Fr] = {
        var p = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { p[i] = rng.nextFr() }
        return p
    }

    let polyInput = makePolyInput(
        qL: genPoly(), qR: genPoly(), qO: genPoly(),
        qM: genPoly(), qC: genPoly(),
        qRange: genPoly(), qLookup: genPoly(), qPoseidon: genPoly(),
        sigma3: genPoly(), z: genPoly()
    )
    let config = LinearizationConfig(domainSize: n,
                                      enableRangeGates: true,
                                      enableLookupGates: false,
                                      enablePoseidonGates: true)

    let result = engine.computeLinearization(evalInput: input, polyInput: polyInput, config: config)

    // Verify r(zeta) matches polynomial evaluation
    let rCheck = evalPoly(result.rCoeffs, at: zeta)
    expect(frEqual(result.rEval, rCheck), "Large domain: r(zeta) matches poly eval")
    expectEqual(result.rCoeffs.count, n, "Large domain: correct coefficient count")
}

// MARK: - Test 20: CPU Scalar-Poly Sum Correctness

private func testCpuScalarPolySumCorrectness() {
    let n = 4

    // Test: 3 * [1, 2, 3, 0] + 5 * [0, 1, 0, 4] = [3, 11, 9, 20]
    let scalar1 = frFromInt(3)
    let poly1 = [frFromInt(1), frFromInt(2), frFromInt(3), Fr.zero]
    let scalar2 = frFromInt(5)
    let poly2 = [Fr.zero, frFromInt(1), Fr.zero, frFromInt(4)]

    let terms: [(Fr, [Fr])] = [(scalar1, poly1), (scalar2, poly2)]
    let result = engine.cpuScalarPolySum(terms: terms, size: n)

    expect(frEqual(result[0], frFromInt(3)), "ScalarPolySum: coeff[0] = 3*1 + 5*0 = 3")
    expect(frEqual(result[1], frFromInt(11)), "ScalarPolySum: coeff[1] = 3*2 + 5*1 = 11")
    expect(frEqual(result[2], frFromInt(9)), "ScalarPolySum: coeff[2] = 3*3 + 5*0 = 9")
    expect(frEqual(result[3], frFromInt(20)), "ScalarPolySum: coeff[3] = 3*0 + 5*4 = 20")

    // Test with scalar = 1 optimization
    let termsOne: [(Fr, [Fr])] = [(Fr.one, poly1)]
    let resultOne = engine.cpuScalarPolySum(terms: termsOne, size: n)
    expect(frEqual(resultOne[0], frFromInt(1)), "ScalarPolySum with scalar=1: coeff[0]")
    expect(frEqual(resultOne[1], frFromInt(2)), "ScalarPolySum with scalar=1: coeff[1]")

    // Test with scalar = 0 (should skip)
    let termsZero: [(Fr, [Fr])] = [(Fr.zero, poly1), (scalar2, poly2)]
    let resultZero = engine.cpuScalarPolySum(terms: termsZero, size: n)
    expect(frEqual(resultZero[0], Fr.zero), "ScalarPolySum with scalar=0: coeff[0] = 0")
    expect(frEqual(resultZero[1], frFromInt(5)), "ScalarPolySum with scalar=0: coeff[1] = 5")
    expect(frEqual(resultZero[3], frFromInt(20)), "ScalarPolySum with scalar=0: coeff[3] = 20")
}
