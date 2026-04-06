import zkMetal
import Foundation

// MARK: - Public test entry point

public func runGPUPolyDivisionTests() {
    testDivideByLinearFactorSimple()
    testDivideByLinearFactorExact()
    testVanishingPolynomialDivision()
    testVanishingPolynomialWithRemainder()
    testExactDivisionVerification()
    testRemainderCorrectness()
    testBatchDivisionSameRoot()
    testLongDivisionBasic()
    testLongDivisionHigherDegree()
    testSyntheticDivisionMatchesLongDivision()
}

// MARK: - Division by linear factor

/// (x^2 + 2x + 1) / (x - 1) = (x + 3) remainder 4
private func testDivideByLinearFactorSimple() {
    suite("GPUPolyDivision linear factor simple")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 1 + 2x + x^2 (ascending order)
    let poly = [frFromInt(1), frFromInt(2), frFromInt(1)]
    let root = frFromInt(1)

    let (quotient, remainder) = engine.divideByLinearFactor(poly, root: root)

    // q(x) = 3 + x
    expect(quotient.count == 2, "quotient degree 1 => 2 coeffs")
    expect(frEqual(quotient[0], frFromInt(3)), "q[0] = 3")
    expect(frEqual(quotient[1], frFromInt(1)), "q[1] = 1")

    // remainder = 4
    expect(remainder.count == 1, "remainder is scalar")
    expect(frEqual(remainder[0], frFromInt(4)), "remainder = 4")
}

/// (x^2 - 1) / (x - 1) = (x + 1) remainder 0 (exact division)
private func testDivideByLinearFactorExact() {
    suite("GPUPolyDivision linear factor exact")
    let engine = GPUPolyDivisionEngine()

    // p(x) = -1 + 0*x + x^2
    let negOne = frSub(Fr.zero, Fr.one)
    let poly = [negOne, Fr.zero, Fr.one]
    let root = frFromInt(1)

    let (quotient, remainder) = engine.divideByLinearFactor(poly, root: root)

    // q(x) = 1 + x
    expect(frEqual(quotient[0], frFromInt(1)), "q[0] = 1")
    expect(frEqual(quotient[1], frFromInt(1)), "q[1] = 1")

    // remainder = 0
    expect(frEqual(remainder[0], Fr.zero), "remainder = 0 for exact division")
}

// MARK: - Vanishing polynomial division

/// Divide p(x) = x^4 - 1 by Z_H(x) = x^2 - 1, expect quotient = x^2 + 1
private func testVanishingPolynomialDivision() {
    suite("GPUPolyDivision vanishing poly")
    let engine = GPUPolyDivisionEngine()

    // p(x) = -1 + 0*x + 0*x^2 + 0*x^3 + x^4
    let negOne = frSub(Fr.zero, Fr.one)
    let poly = [negOne, Fr.zero, Fr.zero, Fr.zero, Fr.one]
    let subgroupSize = 2

    let (quotient, remainder) = engine.divideByVanishing(poly, subgroupSize: subgroupSize)

    // x^4 - 1 = (x^2 - 1)(x^2 + 1)
    // quotient should be x^2 + 1 = [1, 0, 1]
    expect(quotient.count >= 3, "quotient has at least 3 coeffs")
    expect(frEqual(quotient[0], frFromInt(1)), "q[0] = 1")
    if quotient.count > 1 {
        expect(frEqual(quotient[1], Fr.zero), "q[1] = 0")
    }
    if quotient.count > 2 {
        expect(frEqual(quotient[2], frFromInt(1)), "q[2] = 1")
    }

    // Remainder should be zero (all coefficients)
    let remIsZero = remainder.allSatisfy { frEqual($0, Fr.zero) }
    expect(remIsZero, "remainder = 0 for exact vanishing division")
}

/// Non-exact division: (x^3 + x + 1) / (x^2 - 1) has nonzero remainder
private func testVanishingPolynomialWithRemainder() {
    suite("GPUPolyDivision vanishing with remainder")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 1 + x + 0*x^2 + x^3
    let poly = [frFromInt(1), frFromInt(1), Fr.zero, frFromInt(1)]
    let subgroupSize = 2

    let (quotient, remainder) = engine.divideByVanishing(poly, subgroupSize: subgroupSize)

    // Verify reconstruction: q * (X^n - 1) + r == p
    // Multiply quotient by (X^2 - 1) and add remainder
    let negOne = frSub(Fr.zero, Fr.one)
    let vanishing = [negOne, Fr.zero, Fr.one]  // X^2 - 1

    // Manual polynomial multiplication q * vanishing
    let qLen = quotient.count
    let vLen = vanishing.count
    let prodLen = qLen + vLen - 1
    var product = [Fr](repeating: Fr.zero, count: prodLen)
    for i in 0..<qLen {
        for j in 0..<vLen {
            product[i + j] = frAdd(product[i + j], frMul(quotient[i], vanishing[j]))
        }
    }

    // Add remainder
    let maxLen = max(prodLen, max(remainder.count, poly.count))
    var reconstructed = [Fr](repeating: Fr.zero, count: maxLen)
    for i in 0..<prodLen { reconstructed[i] = frAdd(reconstructed[i], product[i]) }
    for i in 0..<remainder.count { reconstructed[i] = frAdd(reconstructed[i], remainder[i]) }

    var match = true
    for i in 0..<poly.count {
        if !frEqual(reconstructed[i], poly[i]) { match = false; break }
    }
    // Higher coefficients should be zero
    for i in poly.count..<maxLen {
        if !frEqual(reconstructed[i], Fr.zero) { match = false; break }
    }
    expect(match, "q*(X^n - 1) + r == original polynomial")
}

// MARK: - Exact division verification

/// Construct p(x) = q(x) * (x - r), divide, verify exact
private func testExactDivisionVerification() {
    suite("GPUPolyDivision exact division verification")
    let engine = GPUPolyDivisionEngine()

    // q(x) = 2 + 3x + x^2
    let q = [frFromInt(2), frFromInt(3), frFromInt(1)]
    let r = frFromInt(5)

    // Compute p(x) = q(x) * (x - 5) manually
    // (x - 5) = [-5, 1]
    let negR = frSub(Fr.zero, r)
    let linear = [negR, Fr.one]

    var poly = [Fr](repeating: Fr.zero, count: q.count + 1)
    for i in 0..<q.count {
        for j in 0..<linear.count {
            poly[i + j] = frAdd(poly[i + j], frMul(q[i], linear[j]))
        }
    }

    let (gotQ, gotRem) = engine.divideByLinearFactor(poly, root: r)

    // Quotient should match q
    expect(gotQ.count == q.count, "quotient degree matches")
    var qMatch = true
    for i in 0..<min(gotQ.count, q.count) {
        if !frEqual(gotQ[i], q[i]) { qMatch = false; break }
    }
    expect(qMatch, "quotient matches constructed q(x)")

    // Remainder should be zero
    expect(frEqual(gotRem[0], Fr.zero), "remainder = 0 for exact division")
}

// MARK: - Remainder correctness (Remainder Theorem)

/// p(r) should equal the remainder of p(x) / (x - r)
private func testRemainderCorrectness() {
    suite("GPUPolyDivision remainder theorem")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 1 + 2x + 3x^2 + 4x^3
    let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let r = frFromInt(7)

    // Evaluate p(7) via Horner: 4*343 + 3*49 + 2*7 + 1 = 1372 + 147 + 14 + 1 = 1534
    let expected = frFromInt(1534)

    let (_, remainder) = engine.divideByLinearFactor(poly, root: r)
    expect(frEqual(remainder[0], expected), "remainder = p(7) = 1534")

    // Also verify via the evaluate helper
    let evalResult = engine.evaluate(poly, at: r)
    expect(frEqual(evalResult, expected), "evaluate(p, 7) = 1534")
}

// MARK: - Batch division

/// Divide multiple polynomials by the same linear factor
private func testBatchDivisionSameRoot() {
    suite("GPUPolyDivision batch division")
    let engine = GPUPolyDivisionEngine()

    let root = frFromInt(3)

    // p1(x) = x^2 - 9 = (x-3)(x+3), should have remainder 0
    let negNine = frSub(Fr.zero, frFromInt(9))
    let p1 = [negNine, Fr.zero, Fr.one]

    // p2(x) = x^2 + x + 1, p2(3) = 9 + 3 + 1 = 13
    let p2 = [frFromInt(1), frFromInt(1), frFromInt(1)]

    // p3(x) = 2x + 6 = 2(x + 3), p3(3) = 12
    let p3 = [frFromInt(6), frFromInt(2)]

    let results = engine.batchDivideByLinear([p1, p2, p3], root: root)

    expect(results.count == 3, "3 results")

    // p1 / (x-3): remainder = 0
    expect(frEqual(results[0].1[0], Fr.zero), "p1 remainder = 0")

    // p2 / (x-3): remainder = 13
    expect(frEqual(results[1].1[0], frFromInt(13)), "p2 remainder = 13")

    // p3 / (x-3): remainder = 12
    expect(frEqual(results[2].1[0], frFromInt(12)), "p3 remainder = 12")

    // Verify p1 quotient: (x+3) = [3, 1]
    let q1 = results[0].0
    expect(frEqual(q1[0], frFromInt(3)), "p1 quotient c0 = 3")
    expect(frEqual(q1[1], frFromInt(1)), "p1 quotient c1 = 1")
}

// MARK: - Long division

/// (2x^3 + 3x^2 + x + 5) / (x + 1) = (2x^2 + x) remainder 5
private func testLongDivisionBasic() {
    suite("GPUPolyDivision long division basic")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 5 + x + 3x^2 + 2x^3
    let poly = [frFromInt(5), frFromInt(1), frFromInt(3), frFromInt(2)]
    // d(x) = 1 + x
    let divisor = [frFromInt(1), frFromInt(1)]

    let (quotient, remainder) = engine.longDivide(poly, by: divisor)

    // Expected: q(x) = 2x^2 + x + 0 => wait, let me recalculate:
    // 2x^3 + 3x^2 + x + 5 divided by (x + 1):
    //   2x^3 / x = 2x^2. 2x^2 * (x+1) = 2x^3 + 2x^2
    //   remainder: x^2 + x + 5
    //   x^2 / x = x. x * (x+1) = x^2 + x
    //   remainder: 5
    //   quotient = 2x^2 + x, remainder = 5

    expect(quotient.count == 3, "quotient has 3 coeffs")
    expect(frEqual(quotient[0], Fr.zero), "q[0] = 0")
    expect(frEqual(quotient[1], frFromInt(1)), "q[1] = 1")
    expect(frEqual(quotient[2], frFromInt(2)), "q[2] = 2")

    expect(frEqual(remainder[0], frFromInt(5)), "remainder = 5")
}

/// (x^4 + 2x^3 + 3x^2 + 4x + 5) / (x^2 + 1) = (x^2 + 2x + 2) remainder (2x + 3)
private func testLongDivisionHigherDegree() {
    suite("GPUPolyDivision long division higher degree")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 5 + 4x + 3x^2 + 2x^3 + x^4
    let poly = [frFromInt(5), frFromInt(4), frFromInt(3), frFromInt(2), frFromInt(1)]
    // d(x) = 1 + 0*x + x^2
    let divisor = [frFromInt(1), Fr.zero, frFromInt(1)]

    let (quotient, remainder) = engine.longDivide(poly, by: divisor)

    // x^4 + 2x^3 + 3x^2 + 4x + 5 / (x^2 + 1):
    //   x^4 / x^2 = x^2. x^2*(x^2+1) = x^4 + x^2
    //   rem: 2x^3 + 2x^2 + 4x + 5
    //   2x^3 / x^2 = 2x. 2x*(x^2+1) = 2x^3 + 2x
    //   rem: 2x^2 + 2x + 5
    //   2x^2 / x^2 = 2. 2*(x^2+1) = 2x^2 + 2
    //   rem: 2x + 3
    //   quotient = x^2 + 2x + 2, remainder = 2x + 3

    expect(quotient.count == 3, "quotient has 3 coeffs")
    expect(frEqual(quotient[0], frFromInt(2)), "q[0] = 2")
    expect(frEqual(quotient[1], frFromInt(2)), "q[1] = 2")
    expect(frEqual(quotient[2], frFromInt(1)), "q[2] = 1")

    expect(remainder.count == 2, "remainder has 2 coeffs")
    expect(frEqual(remainder[0], frFromInt(3)), "r[0] = 3")
    expect(frEqual(remainder[1], frFromInt(2)), "r[1] = 2")
}

// MARK: - Synthetic vs long division consistency

/// Verify synthetic division matches long division for linear divisors
private func testSyntheticDivisionMatchesLongDivision() {
    suite("GPUPolyDivision synthetic vs long division")
    let engine = GPUPolyDivisionEngine()

    // p(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4
    let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4), frFromInt(5)]
    let root = frFromInt(7)

    // Synthetic division by (X - 7)
    let (synQ, synRem) = engine.divideByLinearFactor(poly, root: root)

    // Long division by (X - 7) = [-7, 1]
    let negRoot = frSub(Fr.zero, root)
    let divisor = [negRoot, Fr.one]
    let (longQ, longRem) = engine.longDivide(poly, by: divisor)

    // Quotients should match
    expect(synQ.count == longQ.count, "quotient lengths match")
    var qMatch = true
    for i in 0..<min(synQ.count, longQ.count) {
        if !frEqual(synQ[i], longQ[i]) { qMatch = false; break }
    }
    expect(qMatch, "synthetic quotient == long division quotient")

    // Remainders should match
    expect(frEqual(synRem[0], longRem[0]), "synthetic remainder == long division remainder")
}
