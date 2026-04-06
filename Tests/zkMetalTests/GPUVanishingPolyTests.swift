import zkMetal
import Foundation

// MARK: - Test helpers

/// Simple LCG for reproducible tests
private struct VanishTestRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextBb() -> UInt32 {
        return next32() % Bb.P
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

private func frToWords(_ f: Fr) -> [UInt32] {
    [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
}

private func wordsToFr(_ w: [UInt32], offset: Int = 0) -> Fr {
    Fr(v: (w[offset], w[offset+1], w[offset+2], w[offset+3],
           w[offset+4], w[offset+5], w[offset+6], w[offset+7]))
}

private func isZeroWords(_ w: [UInt32]) -> Bool {
    w.allSatisfy { $0 == 0 }
}

private func frNeg(_ a: Fr) -> Fr {
    if a.isZero { return a }
    return frSub(Fr.zero, a)
}

// MARK: - Public test entry point

public func runGPUVanishingPolyTests() {
    // Z_H evaluation tests
    testEvaluateZHBn254AtOne()
    testEvaluateZHBn254AtRootOfUnity()
    testEvaluateZHBn254AtNonRoot()
    testEvaluateZHBabyBear()

    // Batch Z_H evaluation
    testBatchEvaluateZHBn254Small()
    testBatchEvaluateZHBn254GPUPath()
    testBatchEvaluateZHBabyBearSmall()

    // Division by vanishing (evaluation domain)
    testDivideByVanishingEvalBn254()
    testDivideByVanishingEvalBabyBear()
    testDivideByVanishingEvalGPUPath()

    // Division by vanishing (coefficient form)
    testDivideByVanishingCoeffBn254Simple()
    testDivideByVanishingCoeffBn254Exact()
    testDivideByVanishingCoeffBn254WithRemainder()
    testDivideByVanishingCoeffBabyBear()
    testDivideByVanishingCoeffSmallPoly()

    // Sparse vanishing polynomial
    testSparseVanishingBn254SingleRoot()
    testSparseVanishingBn254MultipleRoots()
    testSparseVanishingBabyBear()
    testSparseVanishingEvalAtRoot()
    testSparseVanishingEvalAtNonRoot()

    // Batch evaluation at arbitrary points
    testBatchEvaluateZHAtPointsBn254()
    testBatchEvaluateZHAtPointsBabyBear()
    testBatchEvaluateZHAtPointsGPUPath()

    // Batch division
    testBatchDivideByZHBn254()
    testBatchDivideByZHBabyBear()

    // Vanishing coefficients
    testVanishingCoefficientsBn254()
    testVanishingCoefficientsBabyBear()

    // Root membership test
    testIsRootOfVanishingBn254()
    testIsRootOfVanishingBabyBear()

    // Coset generator
    testDefaultCosetGeneratorBn254()
    testDefaultCosetGeneratorBabyBear()

    // Cross-consistency tests
    testEvalCoeffConsistencyBn254()
    testCPUGPUConsistencyBn254()
    testSparseVanishingMatchesDirectEval()

    // Performance
    testLargeBatchPerformance()
}

// MARK: - Z_H Evaluation Tests

/// Z_H(1) = 1^n - 1 = 0 for any subgroup size
private func testEvaluateZHBn254AtOne() {
    suite("VanishPoly Z_H(1) = 0 BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        let one = frToWords(Fr.one)

        for logN in [1, 4, 8, 16] {
            let zh = engine.evaluateZH(point: one, logSubgroup: logN, field: .bn254)
            expect(isZeroWords(zh), "Z_H(1) = 0 for logN=\(logN)")
        }
    } catch {
        expect(false, "Z_H(1) test failed: \(error)")
    }
}

/// omega^n = 1 so Z_H(omega) = omega^n - 1 = 0
private func testEvaluateZHBn254AtRootOfUnity() {
    suite("VanishPoly Z_H(omega) = 0 BN254")
    do {
        let engine = try GPUVanishingPolyEngine()

        for logN in [2, 4, 8, 12] {
            let omega = frRootOfUnity(logN: logN)
            let pt = frToWords(omega)
            let zh = engine.evaluateZH(point: pt, logSubgroup: logN, field: .bn254)
            expect(isZeroWords(zh), "Z_H(omega_\(logN)) = 0")
        }

        // Also test omega^k for k < n (should also be zero since omega^k is in the subgroup)
        let logN = 4
        let omega = frRootOfUnity(logN: logN)
        var pt = omega
        for k in 1..<(1 << logN) {
            let zh = engine.evaluateZH(point: frToWords(pt), logSubgroup: logN, field: .bn254)
            expect(isZeroWords(zh), "Z_H(omega^\(k)) = 0")
            pt = frMul(pt, omega)
        }
    } catch {
        expect(false, "Z_H(omega) test failed: \(error)")
    }
}

/// Z_H at a random non-root point should be non-zero
private func testEvaluateZHBn254AtNonRoot() {
    suite("VanishPoly Z_H(random) != 0 BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xCAFEBABE)

        for _ in 0..<10 {
            let pt = frToWords(rng.nextFr())
            let zh = engine.evaluateZH(point: pt, logSubgroup: 4, field: .bn254)
            // Overwhelmingly likely to be non-zero
            expect(!isZeroWords(zh), "Z_H(random) != 0")
        }
    } catch {
        expect(false, "Z_H(random) test failed: \(error)")
    }
}

/// BabyBear Z_H evaluation
private func testEvaluateZHBabyBear() {
    suite("VanishPoly Z_H BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()

        // Z_H(1) = 1^n - 1 = 0
        let zh1 = engine.evaluateZH(point: [1], logSubgroup: 4, field: .babybear)
        expectEqual(zh1[0], UInt32(0), "Z_H(1) = 0 BabyBear")

        // Z_H at root of unity = 0
        let omega = bbRootOfUnity(logN: 4)
        let zhOmega = engine.evaluateZH(point: [omega.v], logSubgroup: 4, field: .babybear)
        expectEqual(zhOmega[0], UInt32(0), "Z_H(omega) = 0 BabyBear")

        // Z_H at 2 for logN=1 (n=2): 2^2 - 1 = 3
        let zh2 = engine.evaluateZH(point: [2], logSubgroup: 1, field: .babybear)
        expectEqual(zh2[0], UInt32(3), "Z_H(2) = 3 for n=2 BabyBear")

        // Z_H at 3 for logN=1 (n=2): 3^2 - 1 = 8
        let zh3 = engine.evaluateZH(point: [3], logSubgroup: 1, field: .babybear)
        expectEqual(zh3[0], UInt32(8), "Z_H(3) = 8 for n=2 BabyBear")
    } catch {
        expect(false, "Z_H BabyBear test failed: \(error)")
    }
}

// MARK: - Batch Z_H Evaluation Tests

/// Small batch (CPU path) on BN254 coset domain
private func testBatchEvaluateZHBn254Small() {
    suite("VanishPoly batch Z_H BN254 small")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logDomain = 4  // 16 points
        let logSubgroup = 2  // n = 4
        let cosetGen = frToWords(frFromInt(Fr.GENERATOR))

        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .bn254)

        let domainSize = 1 << logDomain
        expectEqual(zhVals.count, domainSize * 8, "correct output size")

        // Verify each value matches single-point evaluation
        let omega = frRootOfUnity(logN: logDomain)
        let g = frFromInt(Fr.GENERATOR)
        var pt = g
        for i in 0..<domainSize {
            let expected = engine.evaluateZH(point: frToWords(pt), logSubgroup: logSubgroup, field: .bn254)
            let actual = Array(zhVals[(i*8)..<((i+1)*8)])
            expectEqual(actual, expected, "batch Z_H matches single at index \(i)")
            pt = frMul(pt, omega)
        }
    } catch {
        expect(false, "batch Z_H BN254 small failed: \(error)")
    }
}

/// Larger batch (GPU path) on BN254
private func testBatchEvaluateZHBn254GPUPath() {
    suite("VanishPoly batch Z_H BN254 GPU path")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logDomain = 10  // 1024 points (above GPU threshold)
        let logSubgroup = 4
        let cosetGen = frToWords(frFromInt(Fr.GENERATOR))

        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .bn254)

        let domainSize = 1 << logDomain
        expectEqual(zhVals.count, domainSize * 8, "correct output size for GPU path")

        // Spot-check first and last entries
        let omega = frRootOfUnity(logN: logDomain)
        let g = frFromInt(Fr.GENERATOR)

        // First point: g
        let expected0 = engine.evaluateZH(point: frToWords(g), logSubgroup: logSubgroup, field: .bn254)
        let actual0 = Array(zhVals[0..<8])
        expectEqual(actual0, expected0, "GPU batch first element matches")

        // Last point: g * omega^(N-1)
        let lastPt = frMul(g, frPow(omega, UInt64(domainSize - 1)))
        let expectedLast = engine.evaluateZH(point: frToWords(lastPt), logSubgroup: logSubgroup, field: .bn254)
        let lastBase = (domainSize - 1) * 8
        let actualLast = Array(zhVals[lastBase..<(lastBase + 8)])
        expectEqual(actualLast, expectedLast, "GPU batch last element matches")

        // All values should be non-zero (coset not in subgroup)
        for i in 0..<domainSize {
            let val = Array(zhVals[(i*8)..<((i+1)*8)])
            expect(!isZeroWords(val), "Z_H non-zero on coset at index \(i)")
        }
    } catch {
        expect(false, "batch Z_H BN254 GPU path failed: \(error)")
    }
}

/// Small batch BabyBear
private func testBatchEvaluateZHBabyBearSmall() {
    suite("VanishPoly batch Z_H BabyBear small")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logDomain = 4
        let logSubgroup = 2
        let cosetGen: [UInt32] = [31]  // primitive root

        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .babybear)

        let domainSize = 1 << logDomain
        expectEqual(zhVals.count, domainSize, "correct output size BabyBear")

        // Verify against single-point evaluation
        let omega = bbRootOfUnity(logN: logDomain)
        var pt = Bb(v: 31)
        for i in 0..<domainSize {
            let expected = engine.evaluateZH(point: [pt.v], logSubgroup: logSubgroup, field: .babybear)
            expectEqual(zhVals[i], expected[0], "batch Z_H BabyBear matches at index \(i)")
            pt = bbMul(pt, omega)
        }
    } catch {
        expect(false, "batch Z_H BabyBear small failed: \(error)")
    }
}

// MARK: - Division by Vanishing (Evaluation Domain) Tests

/// Divide polynomial evals by Z_H on BN254 coset, verify f = Z_H * q
private func testDivideByVanishingEvalBn254() {
    suite("VanishPoly div eval BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xBEEFCAFE)

        let logDomain = 4  // 16 points
        let logSubgroup = 2  // n = 4
        let cosetGen = frToWords(frFromInt(Fr.GENERATOR))

        // Generate random evaluations that are multiples of Z_H
        // First compute Z_H values
        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .bn254)

        // Create f = Z_H * q for a random q
        let domainSize = 1 << logDomain
        var evals = [UInt32]()
        evals.reserveCapacity(domainSize * 8)
        var expectedQuot = [Fr]()
        for i in 0..<domainSize {
            let q = rng.nextFr()
            expectedQuot.append(q)
            let zhVal = wordsToFr(zhVals, offset: i * 8)
            let fVal = frMul(q, zhVal)
            evals.append(contentsOf: frToWords(fVal))
        }

        let result = try engine.divideByVanishingEval(evals: evals, logDomain: logDomain,
                                                       logSubgroup: logSubgroup,
                                                       cosetGen: cosetGen, field: .bn254)

        // Verify quotient matches expected
        for i in 0..<domainSize {
            let actual = wordsToFr(result, offset: i * 8)
            expectEqual(actual, expectedQuot[i], "quotient matches at index \(i)")
        }
    } catch {
        expect(false, "div eval BN254 failed: \(error)")
    }
}

/// BabyBear evaluation domain division
private func testDivideByVanishingEvalBabyBear() {
    suite("VanishPoly div eval BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0x12345678)

        let logDomain = 4
        let logSubgroup = 2
        let cosetGen: [UInt32] = [31]

        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .babybear)

        let domainSize = 1 << logDomain
        var evals = [UInt32]()
        var expectedQ = [UInt32]()
        for i in 0..<domainSize {
            let q = rng.nextBb()
            expectedQ.append(q)
            let fVal = bbMul(Bb(v: q), Bb(v: zhVals[i]))
            evals.append(fVal.v)
        }

        let result = try engine.divideByVanishingEval(evals: evals, logDomain: logDomain,
                                                       logSubgroup: logSubgroup,
                                                       cosetGen: cosetGen, field: .babybear)

        for i in 0..<domainSize {
            expectEqual(result[i], expectedQ[i], "BabyBear quotient at index \(i)")
        }
    } catch {
        expect(false, "div eval BabyBear failed: \(error)")
    }
}

/// Force the GPU code path with a larger domain
private func testDivideByVanishingEvalGPUPath() {
    suite("VanishPoly div eval BN254 GPU path")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xDEADC0DE)

        let logDomain = 10  // 1024 points (above threshold)
        let logSubgroup = 4
        let cosetGen = frToWords(frFromInt(Fr.GENERATOR))

        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .bn254)

        let domainSize = 1 << logDomain
        var evals = [UInt32]()
        evals.reserveCapacity(domainSize * 8)
        var spotCheckQ = [Fr]()  // store a few for verification
        for i in 0..<domainSize {
            let q = rng.nextFr()
            if i < 4 || i == domainSize - 1 { spotCheckQ.append(q) }
            let zhVal = wordsToFr(zhVals, offset: i * 8)
            let fVal = frMul(q, zhVal)
            evals.append(contentsOf: frToWords(fVal))
        }

        let result = try engine.divideByVanishingEval(evals: evals, logDomain: logDomain,
                                                       logSubgroup: logSubgroup,
                                                       cosetGen: cosetGen, field: .bn254)

        expectEqual(result.count, domainSize * 8, "GPU path output size correct")

        // Spot check first 4 and last
        for j in 0..<4 {
            let actual = wordsToFr(result, offset: j * 8)
            expectEqual(actual, spotCheckQ[j], "GPU div spot check index \(j)")
        }
        let actual = wordsToFr(result, offset: (domainSize - 1) * 8)
        expectEqual(actual, spotCheckQ[4], "GPU div spot check last")
    } catch {
        expect(false, "div eval BN254 GPU path failed: \(error)")
    }
}

// MARK: - Division by Vanishing (Coefficient Form) Tests

/// Simple case: x^2 - 1 divided by x^2 - 1 = 1 with remainder 0
private func testDivideByVanishingCoeffBn254Simple() {
    suite("VanishPoly coeff div BN254 simple")
    do {
        let engine = try GPUVanishingPolyEngine()

        // f(x) = x^2 - 1 = Z_H for logSubgroup=1 (n=2)
        let negOne = frNeg(Fr.one)
        let coeffs = frToWords(negOne) + frToWords(Fr.zero) + frToWords(Fr.one)

        let (quot, rem) = try engine.divideByVanishingCoeff(coeffs: coeffs, logSubgroup: 1, field: .bn254)

        // Quotient should be 1 (degree 0)
        expectEqual(quot.count, 8, "quotient has 1 coefficient")
        let q0 = wordsToFr(quot)
        expectEqual(q0, Fr.one, "quotient = 1")

        // Remainder should be 0
        let n = 2
        expectEqual(rem.count, n * 8, "remainder has n coefficients")
        for i in 0..<n {
            let r = wordsToFr(rem, offset: i * 8)
            expect(r.isZero, "remainder[\(i)] = 0")
        }
    } catch {
        expect(false, "coeff div BN254 simple failed: \(error)")
    }
}

/// Exact division: f(x) = x^4 - 1 = (x^2 - 1)(x^2 + 1)
private func testDivideByVanishingCoeffBn254Exact() {
    suite("VanishPoly coeff div BN254 exact")
    do {
        let engine = try GPUVanishingPolyEngine()

        // f(x) = x^4 - 1 = [-1, 0, 0, 0, 1]
        let negOne = frNeg(Fr.one)
        let coeffs = frToWords(negOne) + frToWords(Fr.zero) + frToWords(Fr.zero) +
                     frToWords(Fr.zero) + frToWords(Fr.one)

        // Divide by x^2 - 1 (logSubgroup = 1, n = 2)
        let (quot, rem) = try engine.divideByVanishingCoeff(coeffs: coeffs, logSubgroup: 1, field: .bn254)

        // Quotient = x^2 + 1 = [1, 0, 1]
        expectEqual(quot.count, 3 * 8, "quotient degree 2")
        let q0 = wordsToFr(quot, offset: 0)
        let q1 = wordsToFr(quot, offset: 8)
        let q2 = wordsToFr(quot, offset: 16)
        expectEqual(q0, Fr.one, "q[0] = 1")
        expect(q1.isZero, "q[1] = 0")
        expectEqual(q2, Fr.one, "q[2] = 1")

        // Remainder = 0
        for i in 0..<2 {
            let r = wordsToFr(rem, offset: i * 8)
            expect(r.isZero, "remainder[\(i)] = 0 for exact division")
        }
    } catch {
        expect(false, "coeff div BN254 exact failed: \(error)")
    }
}

/// Division with non-zero remainder
private func testDivideByVanishingCoeffBn254WithRemainder() {
    suite("VanishPoly coeff div BN254 with remainder")
    do {
        let engine = try GPUVanishingPolyEngine()

        // f(x) = x^4 + 2x^3 + 3x^2 + 4x + 5
        // Divide by x^2 - 1 (logSubgroup = 1, n = 2)
        // x^4 + 2x^3 + 3x^2 + 4x + 5 = (x^2 - 1)(x^2 + 2x + 4) + (6x + 9)
        let c0 = frFromInt(5)
        let c1 = frFromInt(4)
        let c2 = frFromInt(3)
        let c3 = frFromInt(2)
        let c4 = frFromInt(1)
        let coeffs = frToWords(c0) + frToWords(c1) + frToWords(c2) + frToWords(c3) + frToWords(c4)

        let (quot, rem) = try engine.divideByVanishingCoeff(coeffs: coeffs, logSubgroup: 1, field: .bn254)

        // Quotient = x^2 + 2x + 4 = [4, 2, 1]
        expectEqual(quot.count, 3 * 8, "quotient size")
        let q0 = wordsToFr(quot, offset: 0)
        let q1 = wordsToFr(quot, offset: 8)
        let q2 = wordsToFr(quot, offset: 16)
        expectEqual(q0, frFromInt(4), "q[0] = 4")
        expectEqual(q1, frFromInt(2), "q[1] = 2")
        expectEqual(q2, frFromInt(1), "q[2] = 1")

        // Remainder = 6x + 9 = [9, 6]
        expectEqual(rem.count, 2 * 8, "remainder size")
        let r0 = wordsToFr(rem, offset: 0)
        let r1 = wordsToFr(rem, offset: 8)
        expectEqual(r0, frFromInt(9), "rem[0] = 9")
        expectEqual(r1, frFromInt(6), "rem[1] = 6")

        // Verify: q * Z_H + r = f
        // q * (x^2 - 1) = [4, 2, 1] * [-1, 0, 1]
        // = [-4, -2, -1+4, 2, 1] = [-4, -2, 3, 2, 1]
        // + remainder [9, 6] at low terms: [5, 4, 3, 2, 1] = f
        let negQ0 = frNeg(q0)
        let negQ1 = frNeg(q1)
        let reconst0 = frAdd(negQ0, r0)
        let reconst1 = frAdd(negQ1, r1)
        let reconst2 = frAdd(frNeg(q2), q0)
        expectEqual(reconst0, c0, "reconstruction c0")
        expectEqual(reconst1, c1, "reconstruction c1")
        expectEqual(reconst2, c2, "reconstruction c2")
    } catch {
        expect(false, "coeff div BN254 with remainder failed: \(error)")
    }
}

/// BabyBear coefficient division
private func testDivideByVanishingCoeffBabyBear() {
    suite("VanishPoly coeff div BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        let p = Bb.P

        // f(x) = x^4 - 1 = [p-1, 0, 0, 0, 1]
        let coeffs: [UInt32] = [p - 1, 0, 0, 0, 1]

        // Divide by x^2 - 1 (logSubgroup = 1, n = 2)
        let (quot, rem) = try engine.divideByVanishingCoeff(coeffs: coeffs, logSubgroup: 1, field: .babybear)

        // Quotient = x^2 + 1 = [1, 0, 1]
        expectEqual(quot.count, 3, "BabyBear quotient size")
        expectEqual(quot[0], UInt32(1), "q[0] = 1")
        expectEqual(quot[1], UInt32(0), "q[1] = 0")
        expectEqual(quot[2], UInt32(1), "q[2] = 1")

        // Remainder = [0, 0]
        expectEqual(rem.count, 2, "BabyBear remainder size")
        expectEqual(rem[0], UInt32(0), "rem[0] = 0")
        expectEqual(rem[1], UInt32(0), "rem[1] = 0")
    } catch {
        expect(false, "coeff div BabyBear failed: \(error)")
    }
}

/// Polynomial degree less than subgroup size returns empty quotient and full poly as remainder
private func testDivideByVanishingCoeffSmallPoly() {
    suite("VanishPoly coeff div small poly")
    do {
        let engine = try GPUVanishingPolyEngine()

        // f(x) = 3x + 2, divide by x^4 - 1 (logSubgroup = 2, n = 4)
        let coeffs = frToWords(frFromInt(2)) + frToWords(frFromInt(3))

        let (quot, rem) = try engine.divideByVanishingCoeff(coeffs: coeffs, logSubgroup: 2, field: .bn254)

        expectEqual(quot.count, 0, "quotient empty for deg < n")
        expectEqual(rem.count, coeffs.count, "remainder = original polynomial")
        expectEqual(rem, coeffs, "remainder equals input")
    } catch {
        expect(false, "coeff div small poly failed: \(error)")
    }
}

// MARK: - Sparse Vanishing Polynomial Tests

/// Sparse vanishing for a single root: V(x) = (x - a)
private func testSparseVanishingBn254SingleRoot() {
    suite("VanishPoly sparse BN254 single root")
    do {
        let engine = try GPUVanishingPolyEngine()

        let a = frFromInt(7)
        let domain = frToWords(a)

        let coeffs = engine.sparseVanishing(domain: domain, field: .bn254)

        // V(x) = x - 7 = [-7, 1]
        expectEqual(coeffs.count, 2 * 8, "2 coefficients")
        let c0 = wordsToFr(coeffs, offset: 0)
        let c1 = wordsToFr(coeffs, offset: 8)
        expectEqual(c0, frNeg(frFromInt(7)), "c0 = -7")
        expectEqual(c1, Fr.one, "c1 = 1")
    } catch {
        expect(false, "sparse BN254 single root failed: \(error)")
    }
}

/// Sparse vanishing for {1, 2, 3}: V(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
private func testSparseVanishingBn254MultipleRoots() {
    suite("VanishPoly sparse BN254 multiple roots")
    do {
        let engine = try GPUVanishingPolyEngine()

        let domain = frToWords(frFromInt(1)) + frToWords(frFromInt(2)) + frToWords(frFromInt(3))
        let coeffs = engine.sparseVanishing(domain: domain, field: .bn254)

        // V(x) = x^3 - 6x^2 + 11x - 6 = [-6, 11, -6, 1]
        expectEqual(coeffs.count, 4 * 8, "degree 3 polynomial has 4 coefficients")
        let c0 = wordsToFr(coeffs, offset: 0)
        let c1 = wordsToFr(coeffs, offset: 8)
        let c2 = wordsToFr(coeffs, offset: 16)
        let c3 = wordsToFr(coeffs, offset: 24)

        expectEqual(c0, frNeg(frFromInt(6)), "c0 = -6")
        expectEqual(c1, frFromInt(11), "c1 = 11")
        expectEqual(c2, frNeg(frFromInt(6)), "c2 = -6")
        expectEqual(c3, Fr.one, "c3 = 1")
    } catch {
        expect(false, "sparse BN254 multiple roots failed: \(error)")
    }
}

/// BabyBear sparse vanishing
private func testSparseVanishingBabyBear() {
    suite("VanishPoly sparse BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        let p = Bb.P

        // V(x) = (x - 1)(x - 2) = x^2 - 3x + 2
        let domain: [UInt32] = [1, 2]
        let coeffs = engine.sparseVanishing(domain: domain, field: .babybear)

        expectEqual(coeffs.count, 3, "degree 2 has 3 coefficients")
        expectEqual(coeffs[0], UInt32(2), "c0 = 2")
        expectEqual(coeffs[1], p - 3, "c1 = -3 mod P")
        expectEqual(coeffs[2], UInt32(1), "c2 = 1")
    } catch {
        expect(false, "sparse BabyBear failed: \(error)")
    }
}

/// Evaluate sparse vanishing at a root should give 0
private func testSparseVanishingEvalAtRoot() {
    suite("VanishPoly sparse eval at root")
    do {
        let engine = try GPUVanishingPolyEngine()

        let roots = [frFromInt(5), frFromInt(10), frFromInt(15)]
        var domain = [UInt32]()
        for r in roots {
            domain.append(contentsOf: frToWords(r))
        }

        for r in roots {
            let val = engine.evaluateSparseVanishing(point: frToWords(r), domain: domain, field: .bn254)
            expect(isZeroWords(val), "V(root) = 0")
        }
    } catch {
        expect(false, "sparse eval at root failed: \(error)")
    }
}

/// Evaluate sparse vanishing at a non-root should give non-zero
private func testSparseVanishingEvalAtNonRoot() {
    suite("VanishPoly sparse eval at non-root")
    do {
        let engine = try GPUVanishingPolyEngine()

        let domain = frToWords(frFromInt(1)) + frToWords(frFromInt(2))

        // V(3) = (3-1)(3-2) = 2
        let val = engine.evaluateSparseVanishing(point: frToWords(frFromInt(3)), domain: domain, field: .bn254)
        let result = wordsToFr(val)
        expectEqual(result, frFromInt(2), "V(3) = 2 for domain {1, 2}")

        // V(0) = (0-1)(0-2) = 2
        let val0 = engine.evaluateSparseVanishing(point: frToWords(frFromInt(0)), domain: domain, field: .bn254)
        let result0 = wordsToFr(val0)
        expectEqual(result0, frFromInt(2), "V(0) = 2 for domain {1, 2}")
    } catch {
        expect(false, "sparse eval at non-root failed: \(error)")
    }
}

// MARK: - Batch Z_H at Arbitrary Points Tests

/// Batch evaluate at known BN254 points
private func testBatchEvaluateZHAtPointsBn254() {
    suite("VanishPoly batch Z_H at points BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xABCDEF01)

        let logSubgroup = 4
        var points = [UInt32]()
        let numPts = 32
        for _ in 0..<numPts {
            points.append(contentsOf: frToWords(rng.nextFr()))
        }

        let results = try engine.batchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                                          field: .bn254)

        expectEqual(results.count, numPts * 8, "output size matches")

        // Verify each against single evaluation
        for i in 0..<numPts {
            let pt = Array(points[(i*8)..<((i+1)*8)])
            let expected = engine.evaluateZH(point: pt, logSubgroup: logSubgroup, field: .bn254)
            let actual = Array(results[(i*8)..<((i+1)*8)])
            expectEqual(actual, expected, "batch vs single at index \(i)")
        }
    } catch {
        expect(false, "batch Z_H at points BN254 failed: \(error)")
    }
}

/// Batch evaluate BabyBear
private func testBatchEvaluateZHAtPointsBabyBear() {
    suite("VanishPoly batch Z_H at points BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0x98765432)

        let logSubgroup = 3
        var points = [UInt32]()
        let numPts = 20
        for _ in 0..<numPts {
            points.append(rng.nextBb())
        }

        let results = try engine.batchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                                          field: .babybear)

        expectEqual(results.count, numPts, "output size BabyBear")

        for i in 0..<numPts {
            let expected = engine.evaluateZH(point: [points[i]], logSubgroup: logSubgroup, field: .babybear)
            expectEqual(results[i], expected[0], "BabyBear batch vs single at index \(i)")
        }
    } catch {
        expect(false, "batch Z_H at points BabyBear failed: \(error)")
    }
}

/// Force GPU path for batch evaluation at arbitrary points
private func testBatchEvaluateZHAtPointsGPUPath() {
    suite("VanishPoly batch Z_H at points GPU path")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0x11223344)

        let logSubgroup = 8
        let numPts = 1024  // above threshold
        var points = [UInt32]()
        points.reserveCapacity(numPts * 8)
        for _ in 0..<numPts {
            points.append(contentsOf: frToWords(rng.nextFr()))
        }

        let results = try engine.batchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                                          field: .bn254)

        expectEqual(results.count, numPts * 8, "GPU batch output size")

        // Spot check a few
        for i in [0, 1, numPts / 2, numPts - 1] {
            let pt = Array(points[(i*8)..<((i+1)*8)])
            let expected = engine.evaluateZH(point: pt, logSubgroup: logSubgroup, field: .bn254)
            let actual = Array(results[(i*8)..<((i+1)*8)])
            expectEqual(actual, expected, "GPU batch spot check \(i)")
        }
    } catch {
        expect(false, "batch Z_H at points GPU path failed: \(error)")
    }
}

// MARK: - Batch Division Tests

/// Batch divide evals by Z_H at given points (BN254)
private func testBatchDivideByZHBn254() {
    suite("VanishPoly batch div BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xFACEFEED)

        let logSubgroup = 4
        let numPts = 32
        var points = [UInt32]()
        var evals = [UInt32]()
        var expectedQ = [Fr]()

        for _ in 0..<numPts {
            let pt = rng.nextFr()
            points.append(contentsOf: frToWords(pt))

            // Compute Z_H(pt) and create f = q * Z_H
            let zhWords = engine.evaluateZH(point: frToWords(pt), logSubgroup: logSubgroup, field: .bn254)
            let zh = wordsToFr(zhWords)
            let q = rng.nextFr()
            expectedQ.append(q)
            let f = frMul(q, zh)
            evals.append(contentsOf: frToWords(f))
        }

        let result = try engine.batchDivideByZH(evals: evals, points: points,
                                                  logSubgroup: logSubgroup, field: .bn254)

        for i in 0..<numPts {
            let actual = wordsToFr(result, offset: i * 8)
            expectEqual(actual, expectedQ[i], "batch div quotient at \(i)")
        }
    } catch {
        expect(false, "batch div BN254 failed: \(error)")
    }
}

/// Batch divide BabyBear
private func testBatchDivideByZHBabyBear() {
    suite("VanishPoly batch div BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xAAAABBBB)

        let logSubgroup = 3
        let numPts = 20
        var points = [UInt32]()
        var evals = [UInt32]()
        var expectedQ = [UInt32]()

        for _ in 0..<numPts {
            let pt = rng.nextBb()
            points.append(pt)
            let zh = engine.evaluateZH(point: [pt], logSubgroup: logSubgroup, field: .babybear)
            let q = rng.nextBb()
            expectedQ.append(q)
            let f = bbMul(Bb(v: q), Bb(v: zh[0]))
            evals.append(f.v)
        }

        let result = try engine.batchDivideByZH(evals: evals, points: points,
                                                  logSubgroup: logSubgroup, field: .babybear)

        for i in 0..<numPts {
            expectEqual(result[i], expectedQ[i], "BabyBear batch div at \(i)")
        }
    } catch {
        expect(false, "batch div BabyBear failed: \(error)")
    }
}

// MARK: - Vanishing Coefficients Tests

/// Check Z_H coefficients for BN254: [-1, 0, ..., 0, 1]
private func testVanishingCoefficientsBn254() {
    suite("VanishPoly coefficients BN254")
    do {
        let engine = try GPUVanishingPolyEngine()

        for logN in [1, 2, 4] {
            let n = 1 << logN
            let coeffs = engine.vanishingCoefficients(logSubgroup: logN, field: .bn254)
            expectEqual(coeffs.count, (n + 1) * 8, "n+1 coefficients for logN=\(logN)")

            // c0 = -1
            let c0 = wordsToFr(coeffs, offset: 0)
            expectEqual(c0, frNeg(Fr.one), "c0 = -1 for logN=\(logN)")

            // Middle coefficients = 0
            for i in 1..<n {
                let ci = wordsToFr(coeffs, offset: i * 8)
                expect(ci.isZero, "c[\(i)] = 0 for logN=\(logN)")
            }

            // c_n = 1
            let cn = wordsToFr(coeffs, offset: n * 8)
            expectEqual(cn, Fr.one, "c[\(n)] = 1 for logN=\(logN)")
        }
    } catch {
        expect(false, "coefficients BN254 failed: \(error)")
    }
}

/// Check Z_H coefficients for BabyBear
private func testVanishingCoefficientsBabyBear() {
    suite("VanishPoly coefficients BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()
        let p = Bb.P

        for logN in [1, 2, 4] {
            let n = 1 << logN
            let coeffs = engine.vanishingCoefficients(logSubgroup: logN, field: .babybear)
            expectEqual(coeffs.count, n + 1, "n+1 coefficients BabyBear logN=\(logN)")

            expectEqual(coeffs[0], p - 1, "c0 = -1 mod P for logN=\(logN)")
            for i in 1..<n {
                expectEqual(coeffs[i], UInt32(0), "c[\(i)] = 0 for logN=\(logN)")
            }
            expectEqual(coeffs[n], UInt32(1), "c[\(n)] = 1 for logN=\(logN)")
        }
    } catch {
        expect(false, "coefficients BabyBear failed: \(error)")
    }
}

// MARK: - Root Membership Tests

/// Roots of unity should be roots of Z_H
private func testIsRootOfVanishingBn254() {
    suite("VanishPoly isRoot BN254")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logN = 4
        let omega = frRootOfUnity(logN: logN)

        // All powers of omega from 0 to n-1 should be roots
        var pt = Fr.one
        for k in 0..<(1 << logN) {
            let isRoot = engine.isRootOfVanishing(point: frToWords(pt), logSubgroup: logN, field: .bn254)
            expect(isRoot, "omega^\(k) is root of Z_H")
            pt = frMul(pt, omega)
        }

        // Random point should not be a root (overwhelming probability)
        var rng = VanishTestRNG(state: 0x55667788)
        for _ in 0..<5 {
            let isRoot = engine.isRootOfVanishing(point: frToWords(rng.nextFr()),
                                                   logSubgroup: logN, field: .bn254)
            expect(!isRoot, "random point is not root of Z_H")
        }
    } catch {
        expect(false, "isRoot BN254 failed: \(error)")
    }
}

/// BabyBear root membership
private func testIsRootOfVanishingBabyBear() {
    suite("VanishPoly isRoot BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()

        // 1 is always a root
        let isRoot1 = engine.isRootOfVanishing(point: [1], logSubgroup: 4, field: .babybear)
        expect(isRoot1, "1 is root of Z_H BabyBear")

        // Root of unity should be a root
        let omega = bbRootOfUnity(logN: 4)
        let isRootOmega = engine.isRootOfVanishing(point: [omega.v], logSubgroup: 4, field: .babybear)
        expect(isRootOmega, "omega is root of Z_H BabyBear")

        // 2 should not be a root for logN=4 (n=16)
        let isRoot2 = engine.isRootOfVanishing(point: [2], logSubgroup: 4, field: .babybear)
        expect(!isRoot2, "2 is not root of Z_H (n=16) BabyBear")
    } catch {
        expect(false, "isRoot BabyBear failed: \(error)")
    }
}

// MARK: - Coset Generator Tests

/// Default coset generator should produce non-zero Z_H values
private func testDefaultCosetGeneratorBn254() {
    suite("VanishPoly coset gen BN254")
    do {
        let engine = try GPUVanishingPolyEngine()

        for logN in [2, 4, 8, 16] {
            let gen = engine.defaultCosetGenerator(logSubgroup: logN, field: .bn254)
            let zh = engine.evaluateZH(point: gen, logSubgroup: logN, field: .bn254)
            expect(!isZeroWords(zh), "coset gen gives non-zero Z_H for logN=\(logN)")
        }
    } catch {
        expect(false, "coset gen BN254 failed: \(error)")
    }
}

/// BabyBear coset generator
private func testDefaultCosetGeneratorBabyBear() {
    suite("VanishPoly coset gen BabyBear")
    do {
        let engine = try GPUVanishingPolyEngine()

        for logN in [2, 4, 8] {
            let gen = engine.defaultCosetGenerator(logSubgroup: logN, field: .babybear)
            let zh = engine.evaluateZH(point: gen, logSubgroup: logN, field: .babybear)
            expect(zh[0] != 0, "BabyBear coset gen gives non-zero Z_H for logN=\(logN)")
        }
    } catch {
        expect(false, "coset gen BabyBear failed: \(error)")
    }
}

// MARK: - Cross-consistency Tests

/// Verify that coefficient form and evaluation form give consistent results
private func testEvalCoeffConsistencyBn254() {
    suite("VanishPoly eval-coeff consistency BN254")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logSubgroup = 2
        let n = 1 << logSubgroup  // n = 4

        // Get Z_H coefficients
        let zhCoeffs = engine.vanishingCoefficients(logSubgroup: logSubgroup, field: .bn254)

        // Evaluate Z_H(point) using Horner's method on coefficients
        var rng = VanishTestRNG(state: 0xDEADBEEF)
        for _ in 0..<10 {
            let pt = rng.nextFr()

            // Horner evaluation of zhCoeffs at pt
            var horner = Fr.zero
            for i in stride(from: n, through: 0, by: -1) {
                horner = frMul(horner, pt)
                horner = frAdd(horner, wordsToFr(zhCoeffs, offset: i * 8))
            }

            // Direct evaluation
            let direct = engine.evaluateZH(point: frToWords(pt), logSubgroup: logSubgroup, field: .bn254)

            expectEqual(frToWords(horner), direct, "Horner matches direct Z_H eval")
        }
    } catch {
        expect(false, "eval-coeff consistency failed: \(error)")
    }
}

/// Verify CPU and GPU paths produce identical results
private func testCPUGPUConsistencyBn254() {
    suite("VanishPoly CPU-GPU consistency BN254")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0x99887766)

        let logSubgroup = 4
        let numPts = 16  // Small enough for CPU path

        var points = [UInt32]()
        for _ in 0..<numPts {
            points.append(contentsOf: frToWords(rng.nextFr()))
        }

        // CPU path (small count)
        let cpuResults = try engine.batchEvaluateZHAtPoints(points: points, logSubgroup: logSubgroup,
                                                             field: .bn254)

        // Verify against individual evaluations
        for i in 0..<numPts {
            let pt = Array(points[(i*8)..<((i+1)*8)])
            let single = engine.evaluateZH(point: pt, logSubgroup: logSubgroup, field: .bn254)
            let batch = Array(cpuResults[(i*8)..<((i+1)*8)])
            expectEqual(batch, single, "CPU batch consistency at \(i)")
        }
    } catch {
        expect(false, "CPU-GPU consistency failed: \(error)")
    }
}

/// Sparse vanishing evaluated at roots should match direct product computation
private func testSparseVanishingMatchesDirectEval() {
    suite("VanishPoly sparse matches direct eval")
    do {
        let engine = try GPUVanishingPolyEngine()
        var rng = VanishTestRNG(state: 0xFEDCBA98)

        // Build domain of 5 random elements
        let k = 5
        var domain = [UInt32]()
        var domainFr = [Fr]()
        for _ in 0..<k {
            let d = rng.nextFr()
            domainFr.append(d)
            domain.append(contentsOf: frToWords(d))
        }

        // Build sparse vanishing polynomial coefficients
        let coeffs = engine.sparseVanishing(domain: domain, field: .bn254)

        // Evaluate at 10 random points using both methods
        for _ in 0..<10 {
            let pt = rng.nextFr()

            // Method 1: evaluate sparse vanishing directly as product
            let directWords = engine.evaluateSparseVanishing(point: frToWords(pt), domain: domain, field: .bn254)
            let direct = wordsToFr(directWords)

            // Method 2: Horner evaluation of polynomial coefficients
            let numCoeffs = coeffs.count / 8
            var horner = Fr.zero
            for i in stride(from: numCoeffs - 1, through: 0, by: -1) {
                horner = frMul(horner, pt)
                horner = frAdd(horner, wordsToFr(coeffs, offset: i * 8))
            }

            expectEqual(horner, direct, "sparse poly vs direct product eval")
        }
    } catch {
        expect(false, "sparse matches direct eval failed: \(error)")
    }
}

// MARK: - Performance Test

/// Benchmark GPU batch evaluation at scale
private func testLargeBatchPerformance() {
    suite("VanishPoly large batch perf")
    do {
        let engine = try GPUVanishingPolyEngine()

        let logDomain = 14  // 16384 points
        let logSubgroup = 8
        let cosetGen = frToWords(frFromInt(Fr.GENERATOR))

        let start = CFAbsoluteTimeGetCurrent()
        let zhVals = try engine.batchEvaluateZH(logDomain: logDomain, logSubgroup: logSubgroup,
                                                 cosetGen: cosetGen, field: .bn254)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let domainSize = 1 << logDomain
        expectEqual(zhVals.count, domainSize * 8, "large batch output size correct")

        // All values on coset should be non-zero
        var allNonZero = true
        for i in 0..<min(100, domainSize) {
            let val = Array(zhVals[(i*8)..<((i+1)*8)])
            if isZeroWords(val) { allNonZero = false; break }
        }
        expect(allNonZero, "coset Z_H values non-zero (sampled)")

        print("  batch Z_H eval \(domainSize) points: \(String(format: "%.2fms", elapsed * 1000))")

        // Benchmark division
        var rng = VanishTestRNG(state: 0xFEEDFACE)
        var evals = [UInt32]()
        evals.reserveCapacity(domainSize * 8)
        for i in 0..<domainSize {
            let q = rng.nextFr()
            let zhVal = wordsToFr(zhVals, offset: i * 8)
            let f = frMul(q, zhVal)
            evals.append(contentsOf: frToWords(f))
        }

        let start2 = CFAbsoluteTimeGetCurrent()
        let divResult = try engine.divideByVanishingEval(evals: evals, logDomain: logDomain,
                                                          logSubgroup: logSubgroup,
                                                          cosetGen: cosetGen, field: .bn254)
        let elapsed2 = CFAbsoluteTimeGetCurrent() - start2

        expectEqual(divResult.count, domainSize * 8, "large div output size correct")
        print("  div by vanishing \(domainSize) evals: \(String(format: "%.2fms", elapsed2 * 1000))")

    } catch {
        expect(false, "large batch perf failed: \(error)")
    }
}
