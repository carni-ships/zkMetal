import zkMetal
import Foundation

// MARK: - Test helpers

/// Simple LCG for reproducible tests
private struct DivTestRNG {
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

private func isZeroFr(_ w: [UInt32]) -> Bool {
    w.allSatisfy { $0 == 0 }
}

// MARK: - Public test entry point

public func runGPUPolyDivTests() {
    testDivByLinearBn254Simple()
    testDivByLinearBn254XSquaredMinus1()
    testDivByLinearBabyBear()
    testDivByLinearVerifyReconstruction()
    testBatchDivBn254()
    testBatchDivConsistency()
    testDivByVanishingBn254()
    testDivByVanishingBabyBear()
    testLargePolyDivPerformance()
}

// MARK: - Division by linear factor tests

/// Basic test: divide (x^2 + 2x + 1) by (x - 1)
/// (x^2 + 2x + 1) = (x - 1)(x + 3) + 4
private func testDivByLinearBn254Simple() {
    suite("GPUPolyDiv BN254 linear simple")
    do {
        let engine = try GPUPolyDivEngine()

        // p(x) = 1 + 2x + x^2 (ascending coefficients, Montgomery form)
        let c0 = frFromInt(1)
        let c1 = frFromInt(2)
        let c2 = frFromInt(1)
        let coeffs = frToWords(c0) + frToWords(c1) + frToWords(c2)

        // Divide by (x - 1), root = 1
        let root = frToWords(frFromInt(1))

        let (quotient, remainder) = try engine.divideByLinear(coeffs: coeffs, root: root, field: .bn254)

        // Quotient should be (x + 3) = [3, 1] in ascending order
        let q0 = wordsToFr(quotient, offset: 0)
        let q1 = wordsToFr(quotient, offset: 8)
        let expectedQ0 = frFromInt(3)
        let expectedQ1 = frFromInt(1)

        expectEqual(q0, expectedQ0, "quotient c0 = 3")
        expectEqual(q1, expectedQ1, "quotient c1 = 1")

        // Remainder should be 4
        let rem = wordsToFr(remainder)
        let expectedRem = frFromInt(4)
        expectEqual(rem, expectedRem, "remainder = 4")

    } catch {
        expect(false, "GPUPolyDiv BN254 linear simple failed: \(error)")
    }
}

/// (x^2 - 1) / (x - 1) = (x + 1) with remainder 0
private func testDivByLinearBn254XSquaredMinus1() {
    suite("GPUPolyDiv BN254 (x^2-1)/(x-1)")
    do {
        let engine = try GPUPolyDivEngine()

        // p(x) = -1 + 0*x + x^2
        let c0 = frNeg(frFromInt(1))  // -1
        let c1 = Fr.zero               // 0
        let c2 = frFromInt(1)          // 1
        let coeffs = frToWords(c0) + frToWords(c1) + frToWords(c2)

        let root = frToWords(frFromInt(1))
        let (quotient, remainder) = try engine.divideByLinear(coeffs: coeffs, root: root, field: .bn254)

        // Quotient = (x + 1) = [1, 1]
        let q0 = wordsToFr(quotient, offset: 0)
        let q1 = wordsToFr(quotient, offset: 8)
        expectEqual(q0, frFromInt(1), "quotient c0 = 1")
        expectEqual(q1, frFromInt(1), "quotient c1 = 1")

        // Remainder = 0
        expect(isZeroFr(remainder), "remainder = 0 for exact division")

    } catch {
        expect(false, "GPUPolyDiv BN254 x^2-1 test failed: \(error)")
    }
}

/// BabyBear linear division: (x^3 - 1) / (x - 1) = x^2 + x + 1
private func testDivByLinearBabyBear() {
    suite("GPUPolyDiv BabyBear linear")
    do {
        let engine = try GPUPolyDivEngine()

        // p(x) = -1 + 0*x + 0*x^2 + x^3
        let p = Bb.P
        let coeffs: [UInt32] = [p - 1, 0, 0, 1]  // -1, 0, 0, 1

        let root: [UInt32] = [1]  // divide by (x - 1)
        let (quotient, remainder) = try engine.divideByLinear(coeffs: coeffs, root: root, field: .babybear)

        // Quotient = x^2 + x + 1 = [1, 1, 1]
        expectEqual(quotient.count, 3, "quotient has degree 2")
        expectEqual(quotient[0], UInt32(1), "q[0] = 1")
        expectEqual(quotient[1], UInt32(1), "q[1] = 1")
        expectEqual(quotient[2], UInt32(1), "q[2] = 1")

        // Remainder = 0
        expectEqual(remainder[0], UInt32(0), "remainder = 0")

    } catch {
        expect(false, "GPUPolyDiv BabyBear linear failed: \(error)")
    }
}

/// Verify quotient * (x - root) + remainder = original polynomial
private func testDivByLinearVerifyReconstruction() {
    suite("GPUPolyDiv BN254 reconstruction")
    do {
        let engine = try GPUPolyDivEngine()
        var rng = DivTestRNG(state: 0xDEADBEEF)

        let degree = 16
        var coeffs = [UInt32]()
        for _ in 0..<degree {
            coeffs.append(contentsOf: frToWords(rng.nextFr()))
        }

        let root = frToWords(rng.nextFr())
        let (quotient, remainder) = try engine.divideByLinear(coeffs: coeffs, root: root, field: .bn254)

        // Reconstruct: q(x) * (x - root) + remainder should equal original
        // In coefficient form: if q = [q0, q1, ..., q_{n-2}]
        // then q(x)*(x - root) = [-root*q0, q0-root*q1, q1-root*q2, ..., q_{n-2}]
        let r = wordsToFr(root)
        let qCount = degree - 1
        var reconstructed = [Fr](repeating: Fr.zero, count: degree)

        for i in 0..<qCount {
            let qi = wordsToFr(quotient, offset: i * 8)
            // Add qi * x^{i+1}
            reconstructed[i + 1] = frAdd(reconstructed[i + 1], qi)
            // Subtract root * qi * x^i
            reconstructed[i] = frSub(reconstructed[i], frMul(r, qi))
        }
        // Add remainder to constant term
        reconstructed[0] = frAdd(reconstructed[0], wordsToFr(remainder))

        for i in 0..<degree {
            let orig = wordsToFr(coeffs, offset: i * 8)
            expectEqual(reconstructed[i], orig, "reconstruction matches at index \(i)")
        }
    } catch {
        expect(false, "GPUPolyDiv BN254 reconstruction failed: \(error)")
    }
}

// MARK: - Batch division tests

/// Batch divide one polynomial by multiple roots and verify each
private func testBatchDivBn254() {
    suite("GPUPolyDiv BN254 batch div")
    do {
        let engine = try GPUPolyDivEngine()

        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let c0 = frNeg(frFromInt(6))    // -6
        let c1 = frFromInt(11)           // 11
        let c2 = frNeg(frFromInt(6))     // -6
        let c3 = frFromInt(1)            // 1
        let coeffs = frToWords(c0) + frToWords(c1) + frToWords(c2) + frToWords(c3)

        let roots: [[UInt32]] = [
            frToWords(frFromInt(1)),
            frToWords(frFromInt(2)),
            frToWords(frFromInt(3)),
        ]

        let results = try engine.batchDivideByLinear(coeffs: coeffs, roots: roots, field: .bn254)

        expectEqual(results.count, 3, "3 results for 3 roots")

        // All divisions should have zero remainder since 1, 2, 3 are roots
        for (idx, (_, rem)) in results.enumerated() {
            expect(isZeroFr(rem), "remainder \(idx) = 0 for root of polynomial")
        }

        // Division by (x-1) should give x^2 - 5x + 6
        let q0_0 = wordsToFr(results[0].0, offset: 0)
        let q0_1 = wordsToFr(results[0].0, offset: 8)
        let q0_2 = wordsToFr(results[0].0, offset: 16)
        expectEqual(q0_0, frFromInt(6), "q/(x-1): c0 = 6")
        expectEqual(q0_1, frNeg(frFromInt(5)), "q/(x-1): c1 = -5")
        expectEqual(q0_2, frFromInt(1), "q/(x-1): c2 = 1")

    } catch {
        expect(false, "GPUPolyDiv BN254 batch div failed: \(error)")
    }
}

/// Verify batch division gives same results as individual division
private func testBatchDivConsistency() {
    suite("GPUPolyDiv batch consistency")
    do {
        let engine = try GPUPolyDivEngine()
        var rng = DivTestRNG(state: 0xCAFEF00D)

        let degree = 32
        var coeffs = [UInt32]()
        for _ in 0..<degree {
            coeffs.append(contentsOf: frToWords(rng.nextFr()))
        }

        let numRoots = 8
        var roots = [[UInt32]]()
        for _ in 0..<numRoots {
            roots.append(frToWords(rng.nextFr()))
        }

        // Batch divide
        let batchResults = try engine.batchDivideByLinear(coeffs: coeffs, roots: roots, field: .bn254)

        // Individual divide
        for i in 0..<numRoots {
            let (singleQ, singleR) = try engine.divideByLinear(coeffs: coeffs, root: roots[i], field: .bn254)
            let (batchQ, batchR) = batchResults[i]

            expectEqual(singleQ, batchQ, "batch vs single quotient match for root \(i)")
            expectEqual(singleR, batchR, "batch vs single remainder match for root \(i)")
        }

    } catch {
        expect(false, "GPUPolyDiv batch consistency failed: \(error)")
    }
}

// MARK: - Vanishing polynomial division tests

/// Test vanishing polynomial division for BN254
/// Construct a polynomial that is divisible by Z_H and verify the quotient
private func testDivByVanishingBn254() {
    suite("GPUPolyDiv BN254 vanishing")
    do {
        let engine = try GPUPolyDivEngine()

        // Use a small domain: N = 2^4 = 16 evaluation domain, n = 2^2 = 4 subgroup
        let logDomain = 4
        let logSubgroup = 2
        let domainSize = 1 << logDomain
        let subgroupSize = 1 << logSubgroup

        // Coset generator: use a small value in Montgomery form
        // g = frFromInt(7) works as a coset shift
        let g = frFromInt(7)
        let cosetGen = frToWords(g)

        // Compute omega (primitive domainSize-th root of unity)
        let omega = frRootOfUnity(logN: logDomain)

        // Construct evaluations: for each coset point, compute a polynomial value
        // that is divisible by Z_H. Use t(x) * Z_H(x) where t(x) = x + 1.
        // Z_H(x) = x^n - 1 where n = subgroupSize
        var evals = [UInt32]()
        evals.reserveCapacity(domainSize * 8)

        var expectedQuotient = [Fr]()
        expectedQuotient.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            // t(cosetPt) = cosetPt + 1
            let t_val = frAdd(cosetPt, frFromInt(1))

            // Z_H(cosetPt) = cosetPt^n - 1
            var zh = cosetPt
            for _ in 0..<logSubgroup {
                zh = frSqr(zh)
            }
            zh = frSub(zh, Fr.one)

            // eval = t(cosetPt) * Z_H(cosetPt)
            let eval = frMul(t_val, zh)
            evals.append(contentsOf: frToWords(eval))
            expectedQuotient.append(t_val)

            cosetPt = frMul(cosetPt, omega)
        }

        let evalsBuf = engine.device.makeBuffer(bytes: evals, length: evals.count * 4,
                                                 options: .storageModeShared)!

        let resultBuf = try engine.divideByVanishing(evals: evalsBuf, logDomain: logDomain,
                                                      logSubgroup: logSubgroup, cosetGen: cosetGen, field: .bn254)

        // Verify quotient evaluations match t(cosetPt)
        let outPtr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
        var allMatch = true
        for i in 0..<domainSize {
            let result = wordsToFr(Array(UnsafeBufferPointer(start: outPtr + i * 8, count: 8)))
            if result != expectedQuotient[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "vanishing division recovers t(x) evaluations")

    } catch {
        expect(false, "GPUPolyDiv BN254 vanishing failed: \(error)")
    }
}

/// Test vanishing polynomial division for BabyBear
private func testDivByVanishingBabyBear() {
    suite("GPUPolyDiv BabyBear vanishing")
    do {
        let engine = try GPUPolyDivEngine()

        let logDomain = 4
        let logSubgroup = 2
        let domainSize = 1 << logDomain

        // Coset generator
        let g = Bb(v: 7)
        let cosetGen: [UInt32] = [g.v]

        let omega = bbRootOfUnity(logN: logDomain)

        var evals = [UInt32]()
        evals.reserveCapacity(domainSize)
        var expectedQuotient = [UInt32]()
        expectedQuotient.reserveCapacity(domainSize)

        var cosetPt = g
        for _ in 0..<domainSize {
            // t(x) = x + 1
            let t_val = bbAdd(cosetPt, Bb.one)

            // Z_H(x) = x^n - 1
            var zh = cosetPt
            for _ in 0..<logSubgroup {
                zh = bbSqr(zh)
            }
            zh = bbSub(zh, Bb.one)

            let eval = bbMul(t_val, zh)
            evals.append(eval.v)
            expectedQuotient.append(t_val.v)

            cosetPt = bbMul(cosetPt, omega)
        }

        let evalsBuf = engine.device.makeBuffer(bytes: evals, length: evals.count * 4,
                                                 options: .storageModeShared)!

        let resultBuf = try engine.divideByVanishing(evals: evalsBuf, logDomain: logDomain,
                                                      logSubgroup: logSubgroup, cosetGen: cosetGen, field: .babybear)

        let outPtr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize)
        var allMatch = true
        for i in 0..<domainSize {
            if outPtr[i] != expectedQuotient[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "BabyBear vanishing division recovers t(x) evaluations")

    } catch {
        expect(false, "GPUPolyDiv BabyBear vanishing failed: \(error)")
    }
}

// MARK: - Performance test

/// Large polynomial division performance (degree 2^16)
private func testLargePolyDivPerformance() {
    suite("GPUPolyDiv performance (degree 2^16)")
    do {
        let engine = try GPUPolyDivEngine()
        var rng = DivTestRNG(state: 0x12345678)

        let degree = 1 << 16  // 65536 coefficients
        var coeffs = [UInt32]()
        coeffs.reserveCapacity(degree)
        for _ in 0..<degree {
            coeffs.append(rng.nextBb())
        }

        let numRoots = 16
        var roots = [[UInt32]]()
        for _ in 0..<numRoots {
            roots.append([rng.nextBb()])
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let results = try engine.batchDivideByLinear(coeffs: coeffs, roots: roots, field: .babybear)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        expectEqual(results.count, numRoots, "batch div returned \(numRoots) results")

        // Verify one result by checking f(root) = remainder
        let root0 = Bb(v: roots[0][0])
        var eval = Bb(v: coeffs[degree - 1])
        for i in stride(from: degree - 2, through: 0, by: -1) {
            eval = bbAdd(bbMul(eval, root0), Bb(v: coeffs[i]))
        }
        expectEqual(results[0].1[0], eval.v, "f(root) = remainder (Horner check)")

        print(String(format: "  Batch div %d roots, degree 2^16: %.2fms", numRoots, elapsed * 1000))

    } catch {
        expect(false, "GPUPolyDiv performance test failed: \(error)")
    }
}
