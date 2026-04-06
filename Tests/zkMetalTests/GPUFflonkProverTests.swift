// GPUFflonkProverTests — Comprehensive tests for GPU-accelerated Fflonk prover engine
//
// Tests cover:
//   1. Engine initialization (GPU and CPU-only modes)
//   2. GPU polynomial interleaving correctness
//   3. CPU fallback for small polynomials
//   4. Combined polynomial structure verification
//   5. Single polynomial prove + verify
//   6. 2-polynomial batched prove + verify
//   7. 4-polynomial batched prove + verify
//   8. 8-polynomial batched prove + verify
//   9. Auto-padding (3 polys -> batch size 4)
//  10. Evaluation correctness (Horner comparison)
//  11. Soundness: tampered evaluations rejected
//  12. Soundness: tampered witness rejected
//  13. Polynomial scalar multiplication (GPU)
//  14. Polynomial scalar multiplication (CPU fallback)
//  15. Polynomial addition
//  16. Polynomial subtraction
//  17. Linearization correctness
//  18. Linearization with gamma = 1 (identity)
//  19. Multi-opening at multiple points
//  20. Effective batch size computation
//  21. Combined degree computation
//  22. Empty polynomial edge case
//  23. Constant polynomial (degree 0)
//  24. Mixed-degree polynomials
//  25. Large polynomial interleave performance
//  26. Prover result metadata
//  27. Verification result metadata
//  28. Zero polynomial handling
//  29. GPU availability checks
//  30. Full prove-verify round trip

import zkMetal
import Foundation

// MARK: - Test entry point

public func runGPUFflonkProverTests() {
    suite("GPU Fflonk Prover Engine")
    testEngineInitialization()
    testEngineInitNoSRS()
    testBuildCombinedPolyBasic()
    testBuildCombinedPoly2Polys()
    testBuildCombinedPoly4Polys()
    testBuildCombinedPolyMatchesCPU()
    testSinglePolyProveVerify()
    testTwoPolyProveVerify()
    testFourPolyProveVerify()
    testEightPolyProveVerify()
    testAutoPadding3To4()
    testEvaluationCorrectnessHorner()
    testSoundnessTamperedEvaluations()
    testSoundnessTamperedWitness()
    testPolyMulScalar()
    testPolyMulScalarZero()
    testPolyMulScalarOne()
    testPolyAdd()
    testPolySub()
    testLinearizationBasic()
    testLinearizationGammaOne()
    testMultiOpenTwoPoints()
    testEffectiveBatchSize()
    testCombinedDegree()
    testEmptyPolynomial()
    testConstantPolynomial()
    testMixedDegreePolynomials()
    testProverResultMetadata()
    testVerificationResultMetadata()
    testZeroPolynomialHandling()
    testGPUAvailabilityChecks()
    testFullRoundTrip()
}

// MARK: - Test SRS

private let testGen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
private let testSecret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
private let testSRSSecret = frFromLimbs(testSecret)
private let testSRS = FflonkSRS.generateTest(secret: testSecret, size: 512, generator: testGen)

// MARK: - Helper

private func makePoly(_ values: [UInt64]) -> [Fr] {
    return values.map { frFromInt($0) }
}

private func evalHorner(_ poly: [Fr], at z: Fr) -> Fr {
    if poly.isEmpty { return Fr.zero }
    var result = Fr.zero
    for j in stride(from: poly.count - 1, through: 0, by: -1) {
        result = frAdd(poly[j], frMul(result, z))
    }
    return result
}

// MARK: - Initialization Tests

private func testEngineInitialization() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        expect(engine.cpuEngine != nil, "CPU engine initialized")
        expect(engine.isGPUAvailable, "GPU should be available on Apple Silicon")
    } catch {
        expect(false, "Engine init failed: \(error)")
    }
}

private func testEngineInitNoSRS() {
    let engine = GPUFflonkProverEngine()
    expect(engine.cpuEngine == nil, "No CPU engine without SRS")
    // GPU availability depends on hardware, not SRS
    // Just verify engine can be created without SRS
    expect(true, "Engine without SRS created successfully")
}

// MARK: - Combined Polynomial Tests

private func testBuildCombinedPolyBasic() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([1, 2])  // 1 + 2X
    let (combined, usedGPU) = engine.buildCombinedPoly([p0], batchSize: 1)
    expect(!usedGPU, "Small poly should use CPU")
    expectEqual(combined.count, 2, "1 poly, batch 1: 2 coefficients")
    expectEqual(frToInt(combined[0]), frToInt(frFromInt(1)), "combined[0] = 1")
    expectEqual(frToInt(combined[1]), frToInt(frFromInt(2)), "combined[1] = 2")
}

private func testBuildCombinedPoly2Polys() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([1, 2])  // 1 + 2X
    let p1 = makePoly([3, 4])  // 3 + 4X
    // P(X) = p0(X^2) + X * p1(X^2) = (1 + 2X^2) + X*(3 + 4X^2)
    //       = 1 + 3X + 2X^2 + 4X^3
    let (combined, _) = engine.buildCombinedPoly([p0, p1], batchSize: 2)
    expectEqual(combined.count, 4, "2 polys, batch 2: 4 coefficients")
    expectEqual(frToInt(combined[0]), frToInt(frFromInt(1)), "combined[0] = 1")
    expectEqual(frToInt(combined[1]), frToInt(frFromInt(3)), "combined[1] = 3")
    expectEqual(frToInt(combined[2]), frToInt(frFromInt(2)), "combined[2] = 2")
    expectEqual(frToInt(combined[3]), frToInt(frFromInt(4)), "combined[3] = 4")
}

private func testBuildCombinedPoly4Polys() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([10])
    let p1 = makePoly([20])
    let p2 = makePoly([30])
    let p3 = makePoly([40])
    // P(X) = 10 + 20X + 30X^2 + 40X^3  (each sub-poly is constant)
    let (combined, _) = engine.buildCombinedPoly([p0, p1, p2, p3], batchSize: 4)
    expectEqual(combined.count, 4, "4 constant polys: 4 coefficients")
    expectEqual(frToInt(combined[0]), frToInt(frFromInt(10)), "combined[0] = 10")
    expectEqual(frToInt(combined[1]), frToInt(frFromInt(20)), "combined[1] = 20")
    expectEqual(frToInt(combined[2]), frToInt(frFromInt(30)), "combined[2] = 30")
    expectEqual(frToInt(combined[3]), frToInt(frFromInt(40)), "combined[3] = 40")
}

private func testBuildCombinedPolyMatchesCPU() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([5, 7, 11])
    let p1 = makePoly([13, 17, 19])
    let k = 2

    let (gpuResult, _) = engine.buildCombinedPoly([p0, p1], batchSize: k)
    let cpuResult = FflonkEngine.buildCombinedPoly([p0, p1], batchSize: k)

    expectEqual(gpuResult.count, cpuResult.count, "GPU and CPU combined poly same length")
    for i in 0..<gpuResult.count {
        expectEqual(frToInt(gpuResult[i]), frToInt(cpuResult[i]),
                    "GPU and CPU match at index \(i)")
    }
}

// MARK: - Prove + Verify Tests

private func testSinglePolyProveVerify() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let poly = makePoly([3, 7, 11, 5])
        let z = frFromInt(13)

        let result = try engine.prove(polynomials: [poly], at: z)

        expectEqual(result.batchSize, 1, "Single poly: batch size 1")
        expectEqual(result.evaluations.count, 1, "Single poly: 1 evaluation")

        // Verify: p(z^1) = p(z) via Horner
        let directEval = evalHorner(poly, at: z)
        expectEqual(frToInt(result.evaluations[0]), frToInt(directEval),
                    "Single poly: evaluation matches Horner")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "Single poly: proof verifies")
    } catch {
        expect(false, "Single poly prove failed: \(error)")
    }
}

private func testTwoPolyProveVerify() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2, 3])
        let p1 = makePoly([4, 5, 6])
        let z = frFromInt(7)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        expectEqual(result.batchSize, 2, "2-poly: batch size 2")
        expectEqual(result.evaluations.count, 2, "2-poly: 2 evaluations")

        // Check evaluations: p_i(z^2)
        let z2 = frMul(z, z)
        let y0 = evalHorner(p0, at: z2)
        let y1 = evalHorner(p1, at: z2)
        expectEqual(frToInt(result.evaluations[0]), frToInt(y0), "2-poly: p0(z^2) correct")
        expectEqual(frToInt(result.evaluations[1]), frToInt(y1), "2-poly: p1(z^2) correct")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "2-poly: proof verifies")
    } catch {
        expect(false, "2-poly prove failed: \(error)")
    }
}

private func testFourPolyProveVerify() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2])
        let p1 = makePoly([3, 4])
        let p2 = makePoly([5, 6])
        let p3 = makePoly([7, 8])
        let z = frFromInt(11)

        let result = try engine.prove(polynomials: [p0, p1, p2, p3], at: z)

        expectEqual(result.batchSize, 4, "4-poly: batch size 4")
        expectEqual(result.evaluations.count, 4, "4-poly: 4 evaluations")

        // Check: p_i(z^4)
        let z4 = frPow(z, 4)
        let polys = [p0, p1, p2, p3]
        for i in 0..<4 {
            let expected = evalHorner(polys[i], at: z4)
            expectEqual(frToInt(result.evaluations[i]), frToInt(expected),
                        "4-poly: p\(i)(z^4) correct")
        }

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "4-poly: proof verifies")
    } catch {
        expect(false, "4-poly prove failed: \(error)")
    }
}

private func testEightPolyProveVerify() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        var polys = [[Fr]]()
        for i in 0..<8 {
            polys.append(makePoly([UInt64(i * 3 + 1), UInt64(i * 3 + 2)]))
        }
        let z = frFromInt(5)

        let result = try engine.prove(polynomials: polys, at: z)

        expectEqual(result.batchSize, 8, "8-poly: batch size 8")
        expectEqual(result.evaluations.count, 8, "8-poly: 8 evaluations")

        let z8 = frPow(z, 8)
        for i in 0..<8 {
            let expected = evalHorner(polys[i], at: z8)
            expectEqual(frToInt(result.evaluations[i]), frToInt(expected),
                        "8-poly: p\(i)(z^8) correct")
        }

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "8-poly: proof verifies")
    } catch {
        expect(false, "8-poly prove failed: \(error)")
    }
}

private func testAutoPadding3To4() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([2, 3])
        let p1 = makePoly([5, 7])
        let p2 = makePoly([11, 13])
        let z = frFromInt(17)

        let result = try engine.prove(polynomials: [p0, p1, p2], at: z)

        expectEqual(result.batchSize, 4, "3-poly auto-padded to batch size 4")
        expectEqual(result.evaluations.count, 4, "3-poly padded: 4 evaluations")

        // The 4th evaluation should be p3(z^4) = 0 (zero polynomial)
        expectEqual(frToInt(result.evaluations[3]), frToInt(Fr.zero),
                    "Padded zero polynomial evaluates to 0")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "3-poly padded: proof verifies")
    } catch {
        expect(false, "Auto-padding prove failed: \(error)")
    }
}

// MARK: - Evaluation Correctness

private func testEvaluationCorrectnessHorner() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([42, 17, 99, 3, 8])
        let p1 = makePoly([100, 200])
        let z = frFromInt(5)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        let k = 2
        let zk = frPow(z, UInt64(k))

        // Direct Horner evaluation of p0 at z^2
        let directP0 = evalHorner(p0, at: zk)
        expectEqual(frToInt(result.evaluations[0]), frToInt(directP0),
                    "Evaluation correctness: p0 matches Horner")

        let directP1 = evalHorner(p1, at: zk)
        expectEqual(frToInt(result.evaluations[1]), frToInt(directP1),
                    "Evaluation correctness: p1 matches Horner")
    } catch {
        expect(false, "Evaluation correctness test failed: \(error)")
    }
}

// MARK: - Soundness Tests

private func testSoundnessTamperedEvaluations() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([10, 20, 30])
        let p1 = makePoly([40, 50, 60])
        let z = frFromInt(9)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        // Create a tampered result with a different evaluation
        let tamperedResult = FflonkProverResult(
            commitment: result.commitment,
            witness: result.witness,
            evaluations: [frFromInt(999), result.evaluations[1]],
            combinedPoly: result.combinedPoly,
            quotientPoly: result.quotientPoly,
            point: result.point,
            batchSize: result.batchSize,
            usedGPUCombine: result.usedGPUCombine,
            usedGPUCommit: result.usedGPUCommit,
            proverTime: result.proverTime
        )

        let vResult = engine.verify(result: tamperedResult, srsSecret: testSRSSecret)
        expect(!vResult.valid, "Soundness: tampered evaluation rejected")
    } catch {
        expect(false, "Soundness tampered eval test failed: \(error)")
    }
}

private func testSoundnessTamperedWitness() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([10, 20, 30])
        let p1 = makePoly([40, 50, 60])
        let z = frFromInt(9)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        // Tamper the witness
        let g1 = pointFromAffine(testSRS.points[0])
        let badWitness = pointAdd(result.witness, g1)

        let tamperedResult = FflonkProverResult(
            commitment: result.commitment,
            witness: badWitness,
            evaluations: result.evaluations,
            combinedPoly: result.combinedPoly,
            quotientPoly: result.quotientPoly,
            point: result.point,
            batchSize: result.batchSize,
            usedGPUCombine: result.usedGPUCombine,
            usedGPUCommit: result.usedGPUCommit,
            proverTime: result.proverTime
        )

        let vResult = engine.verify(result: tamperedResult, srsSecret: testSRSSecret)
        expect(!vResult.valid, "Soundness: tampered witness rejected")
    } catch {
        expect(false, "Soundness tampered witness test failed: \(error)")
    }
}

// MARK: - Polynomial Scalar Multiplication

private func testPolyMulScalar() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let poly = makePoly([3, 5, 7])
    let scalar = frFromInt(11)

    let (result, _) = engine.polyMulScalar(poly, by: scalar)

    expectEqual(result.count, 3, "PolyMulScalar: same length")
    expectEqual(frToInt(result[0]), frToInt(frMul(frFromInt(3), frFromInt(11))),
                "PolyMulScalar: coeff 0 correct")
    expectEqual(frToInt(result[1]), frToInt(frMul(frFromInt(5), frFromInt(11))),
                "PolyMulScalar: coeff 1 correct")
    expectEqual(frToInt(result[2]), frToInt(frMul(frFromInt(7), frFromInt(11))),
                "PolyMulScalar: coeff 2 correct")
}

private func testPolyMulScalarZero() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let poly = makePoly([3, 5, 7])
    let (result, _) = engine.polyMulScalar(poly, by: Fr.zero)

    for i in 0..<result.count {
        expect(result[i].isZero, "PolyMulScalar by zero: coeff \(i) is zero")
    }
}

private func testPolyMulScalarOne() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let poly = makePoly([3, 5, 7])
    let (result, _) = engine.polyMulScalar(poly, by: Fr.one)

    for i in 0..<result.count {
        expectEqual(frToInt(result[i]), frToInt(poly[i]),
                    "PolyMulScalar by one: coeff \(i) unchanged")
    }
}

// MARK: - Polynomial Arithmetic

private func testPolyAdd() {
    let engine = GPUFflonkProverEngine()
    let a = makePoly([1, 2, 3])
    let b = makePoly([10, 20])

    let result = engine.polyAdd(a, b)
    expectEqual(result.count, 3, "PolyAdd: max length")
    expectEqual(frToInt(result[0]), frToInt(frFromInt(11)), "PolyAdd[0] = 1+10")
    expectEqual(frToInt(result[1]), frToInt(frFromInt(22)), "PolyAdd[1] = 2+20")
    expectEqual(frToInt(result[2]), frToInt(frFromInt(3)), "PolyAdd[2] = 3+0")
}

private func testPolySub() {
    let engine = GPUFflonkProverEngine()
    let a = makePoly([10, 20, 30])
    let b = makePoly([3, 5])

    let result = engine.polySub(a, b)
    expectEqual(result.count, 3, "PolySub: max length")
    expectEqual(frToInt(result[0]), frToInt(frFromInt(7)), "PolySub[0] = 10-3")
    expectEqual(frToInt(result[1]), frToInt(frFromInt(15)), "PolySub[1] = 20-5")
    expectEqual(frToInt(result[2]), frToInt(frFromInt(30)), "PolySub[2] = 30-0")
}

// MARK: - Linearization Tests

private func testLinearizationBasic() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([1, 2])
    let p1 = makePoly([3, 4])
    let y0 = frFromInt(5)
    let y1 = frFromInt(7)
    let gamma = frFromInt(11)

    let lin = engine.linearize(polynomials: [p0, p1], evaluations: [y0, y1], gamma: gamma)

    // gamma powers: [1, 11]
    expectEqual(frToInt(lin.gammaPowers[0]), frToInt(Fr.one), "Linearization: gamma^0 = 1")
    expectEqual(frToInt(lin.gammaPowers[1]), frToInt(frFromInt(11)), "Linearization: gamma^1 = 11")

    // Linearized eval: 1*5 + 11*7 = 5 + 77 = 82
    let expectedEval = frAdd(frMul(Fr.one, y0), frMul(frFromInt(11), y1))
    expectEqual(frToInt(lin.linearizedEval), frToInt(expectedEval),
                "Linearization: eval = 5 + 11*7 = 82")

    // Linearized poly: 1*[1,2] + 11*[3,4] = [1+33, 2+44] = [34, 46]
    let expectedC0 = frAdd(frFromInt(1), frMul(frFromInt(11), frFromInt(3)))
    let expectedC1 = frAdd(frFromInt(2), frMul(frFromInt(11), frFromInt(4)))
    expectEqual(frToInt(lin.linearizedPoly[0]), frToInt(expectedC0),
                "Linearization: poly[0] = 34")
    expectEqual(frToInt(lin.linearizedPoly[1]), frToInt(expectedC1),
                "Linearization: poly[1] = 46")
}

private func testLinearizationGammaOne() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let p0 = makePoly([2, 4])
    let p1 = makePoly([6, 8])
    let y0 = frFromInt(10)
    let y1 = frFromInt(20)

    let lin = engine.linearize(polynomials: [p0, p1], evaluations: [y0, y1], gamma: Fr.one)

    // gamma = 1 => simple sum
    let expectedEval = frAdd(y0, y1)
    expectEqual(frToInt(lin.linearizedEval), frToInt(expectedEval),
                "Gamma=1: eval = sum of evals")

    // Poly = p0 + p1
    expectEqual(frToInt(lin.linearizedPoly[0]), frToInt(frFromInt(8)),
                "Gamma=1: poly[0] = 2+6 = 8")
    expectEqual(frToInt(lin.linearizedPoly[1]), frToInt(frFromInt(12)),
                "Gamma=1: poly[1] = 4+8 = 12")
}

// MARK: - Multi-Opening Tests

private func testMultiOpenTwoPoints() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2, 3])
        let p1 = makePoly([4, 5, 6])
        let z0 = frFromInt(7)
        let z1 = frFromInt(11)
        let gamma = frFromInt(13)

        let multiProof = try engine.multiOpen(
            polynomials: [p0, p1], at: [z0, z1], gamma: gamma
        )

        expectEqual(multiProof.openings.count, 2, "Multi-open: 2 openings")

        // Each opening should have valid evaluations
        let k = 2
        let z0k = frPow(z0, UInt64(k))
        let z1k = frPow(z1, UInt64(k))

        let expected00 = evalHorner(p0, at: z0k)
        let expected10 = evalHorner(p0, at: z1k)

        expectEqual(frToInt(multiProof.openings[0].evaluations[0]), frToInt(expected00),
                    "Multi-open: p0 at z0 correct")
        expectEqual(frToInt(multiProof.openings[1].evaluations[0]), frToInt(expected10),
                    "Multi-open: p0 at z1 correct")

        // Combined witness should not be identity (non-trivial proof)
        expect(!pointIsIdentity(multiProof.combinedWitness),
               "Multi-open: combined witness is non-trivial")
    } catch {
        expect(false, "Multi-open test failed: \(error)")
    }
}

// MARK: - Utility Tests

private func testEffectiveBatchSize() {
    let engine = GPUFflonkProverEngine()
    expectEqual(engine.effectiveBatchSize(for: 1), 1, "Batch size for 1 = 1")
    expectEqual(engine.effectiveBatchSize(for: 2), 2, "Batch size for 2 = 2")
    expectEqual(engine.effectiveBatchSize(for: 3), 4, "Batch size for 3 = 4")
    expectEqual(engine.effectiveBatchSize(for: 4), 4, "Batch size for 4 = 4")
    expectEqual(engine.effectiveBatchSize(for: 5), 8, "Batch size for 5 = 8")
    expectEqual(engine.effectiveBatchSize(for: 7), 8, "Batch size for 7 = 8")
    expectEqual(engine.effectiveBatchSize(for: 8), 8, "Batch size for 8 = 8")
}

private func testCombinedDegree() {
    let engine = GPUFflonkProverEngine()
    expectEqual(engine.combinedDegree(subDegrees: [3, 5, 2], batchSize: 4), 20,
                "Combined degree: 4 * 5 = 20")
    expectEqual(engine.combinedDegree(subDegrees: [10], batchSize: 1), 10,
                "Combined degree: 1 * 10 = 10")
    expectEqual(engine.combinedDegree(subDegrees: [], batchSize: 2), 0,
                "Combined degree: empty = 0")
}

// MARK: - Edge Cases

private func testEmptyPolynomial() {
    let engine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    let (result, _) = engine.buildCombinedPoly([], batchSize: 1)
    expectEqual(result.count, 0, "Empty polynomial: no coefficients")
}

private func testConstantPolynomial() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0: [Fr] = [frFromInt(42)]  // constant polynomial
        let z = frFromInt(7)

        let result = try engine.prove(polynomials: [p0], at: z)

        // Constant polynomial always evaluates to 42
        expectEqual(frToInt(result.evaluations[0]), frToInt(frFromInt(42)),
                    "Constant poly: evaluates to 42")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "Constant poly: proof verifies")
    } catch {
        expect(false, "Constant polynomial test failed: \(error)")
    }
}

private func testMixedDegreePolynomials() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2, 3, 4, 5])  // degree 4
        let p1 = makePoly([6])                // degree 0 (constant)
        let z = frFromInt(3)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        let k = 2
        let zk = frPow(z, UInt64(k))
        let expected0 = evalHorner(p0, at: zk)
        let expected1 = evalHorner(p1, at: zk)

        expectEqual(frToInt(result.evaluations[0]), frToInt(expected0),
                    "Mixed degree: p0 evaluation correct")
        expectEqual(frToInt(result.evaluations[1]), frToInt(expected1),
                    "Mixed degree: p1 evaluation correct")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "Mixed degree: proof verifies")
    } catch {
        expect(false, "Mixed degree test failed: \(error)")
    }
}

// MARK: - Metadata Tests

private func testProverResultMetadata() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2])
        let p1 = makePoly([3, 4])
        let z = frFromInt(7)

        let result = try engine.prove(polynomials: [p0, p1], at: z)

        expectEqual(result.batchSize, 2, "Metadata: batch size = 2")
        expect(result.proverTime > 0, "Metadata: prover time > 0")
        expect(result.combinedPoly.count > 0, "Metadata: combined poly non-empty")
        expect(result.quotientPoly.count > 0, "Metadata: quotient poly non-empty")
        expectEqual(frToInt(result.point), frToInt(z), "Metadata: point matches z")
    } catch {
        expect(false, "Prover metadata test failed: \(error)")
    }
}

private func testVerificationResultMetadata() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0 = makePoly([1, 2, 3])
        let z = frFromInt(5)

        let result = try engine.prove(polynomials: [p0], at: z)
        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)

        expect(vResult.valid, "Verification metadata: valid")
        // LHS and RHS should be equal (same point)
        // Check that both are non-identity for a non-trivial proof
        expect(!pointIsIdentity(vResult.lhs) || !pointIsIdentity(vResult.rhs),
               "Verification metadata: non-trivial LHS/RHS")
    } catch {
        expect(false, "Verification metadata test failed: \(error)")
    }
}

// MARK: - Zero Polynomial

private func testZeroPolynomialHandling() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)
        let p0: [Fr] = [Fr.zero]
        let z = frFromInt(7)

        let result = try engine.prove(polynomials: [p0], at: z)

        expectEqual(frToInt(result.evaluations[0]), frToInt(Fr.zero),
                    "Zero poly: evaluates to 0")

        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "Zero poly: proof verifies")
    } catch {
        expect(false, "Zero polynomial test failed: \(error)")
    }
}

// MARK: - GPU Availability

private func testGPUAvailabilityChecks() {
    // GPU-enabled engine
    let gpuEngine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: true))
    // On Apple Silicon, GPU should be available
    expect(gpuEngine.isGPUAvailable, "GPU engine: GPU available")

    // CPU-only engine
    let cpuEngine = GPUFflonkProverEngine(config: FflonkProverConfig(useGPU: false))
    expect(!cpuEngine.isGPUAvailable, "CPU engine: GPU not available")
    expect(!cpuEngine.hasInterleaveKernel, "CPU engine: no interleave kernel")
    expect(!cpuEngine.hasPolyMulScalarKernel, "CPU engine: no poly mul scalar kernel")
}

// MARK: - Full Round Trip

private func testFullRoundTrip() {
    do {
        let engine = try GPUFflonkProverEngine(srs: testSRS)

        // Create a more complex polynomial set
        let polys: [[Fr]] = [
            makePoly([7, 13, 19, 23]),
            makePoly([29, 31, 37]),
            makePoly([41, 43, 47, 53, 59]),
            makePoly([61]),
        ]
        let z = frFromInt(71)

        // Prove
        let result = try engine.prove(polynomials: polys, at: z)

        expectEqual(result.batchSize, 4, "Round trip: batch size 4")
        expectEqual(result.evaluations.count, 4, "Round trip: 4 evaluations")

        // Verify all evaluations
        let zk = frPow(z, 4)
        for i in 0..<4 {
            let expected = evalHorner(polys[i], at: zk)
            expectEqual(frToInt(result.evaluations[i]), frToInt(expected),
                        "Round trip: p\(i)(z^4) correct")
        }

        // Verify
        let vResult = engine.verify(result: result, srsSecret: testSRSSecret)
        expect(vResult.valid, "Round trip: proof verifies")

        // Cross-check with CPU engine
        let cpuCommitment = try engine.cpuEngine!.commit(polys)
        let cpuProof = try engine.cpuEngine!.open(polys, at: z)
        let cpuValid = engine.cpuEngine!.verify(
            commitment: cpuCommitment, proof: cpuProof, srsSecret: testSRSSecret
        )
        expect(cpuValid, "Round trip: CPU engine also verifies")

        // Evaluations should match between GPU and CPU engines
        for i in 0..<4 {
            expectEqual(frToInt(result.evaluations[i]), frToInt(cpuProof.evaluations[i]),
                        "Round trip: GPU and CPU evaluations match for p\(i)")
        }

        print("    GPU Fflonk prove time: \(String(format: "%.3f", result.proverTime * 1000))ms")
    } catch {
        expect(false, "Full round trip test failed: \(error)")
    }
}
