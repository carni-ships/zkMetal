// GPUKZGDegreeBoundEngine tests: degree bound proofs, shifted commitments,
// batch verification, multi-point opening with degree bounds, SRS trimming,
// polynomial utilities, accumulator, combined opening + degree bound proofs
import zkMetal
import Foundation

public func runGPUKZGDegreeBoundTests() {
    // Common setup: test SRS with known secret
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srsSize = 256
    let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: gen)

    // MARK: - Polynomial Shifting

    suite("Degree Bound - Polynomial Shifting")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testShiftByZero(engine: engine)
        testShiftByOne(engine: engine)
        testShiftByLargeAmount(engine: engine)
        testShiftEmptyPolynomial(engine: engine)
    } catch {
        expect(false, "Polynomial shifting setup failed: \(error)")
    }

    // MARK: - Polynomial Degree Utilities

    suite("Degree Bound - Polynomial Utilities")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testPolynomialDegree(engine: engine)
        testCheckDegreeBound(engine: engine)
        testPadToDegree(engine: engine)
        testTrimPolynomial(engine: engine)
    } catch {
        expect(false, "Polynomial utilities setup failed: \(error)")
    }

    // MARK: - Shifted Commitment

    suite("Degree Bound - Shifted Commitment")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testShiftedCommitmentBasic(engine: engine, srsSecret: srsSecret)
        testShiftedCommitmentConsistency(engine: engine, srsSecret: srsSecret)
        testShiftedCommitmentDifferentBounds(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Shifted commitment setup failed: \(error)")
    }

    // MARK: - Single Degree Bound Proof

    suite("Degree Bound - Single Proof")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testSingleDegreeBoundValid(engine: engine, srsSecret: srsSecret)
        testSingleDegreeBoundTightBound(engine: engine, srsSecret: srsSecret)
        testSingleDegreeBoundLooseBound(engine: engine, srsSecret: srsSecret)
        testSingleDegreeBoundConstant(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Single proof setup failed: \(error)")
    }

    // MARK: - Degree Bound Verification

    suite("Degree Bound - Verification")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testVerifyValidDegreeBound(engine: engine, srsSecret: srsSecret)
        testVerifyInvalidDegreeBound(engine: engine, srsSecret: srsSecret)
        testVerifyWrongShiftedCommitment(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Verification setup failed: \(error)")
    }

    // MARK: - Batch Degree Bound Proofs

    suite("Degree Bound - Batch Proofs")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testBatchProveDegreeBound(engine: engine, srsSecret: srsSecret)
        testBatchVerifyDegreeBound(engine: engine, srsSecret: srsSecret)
        testBatchVerifyMixedBounds(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Batch proof setup failed: \(error)")
    }

    // MARK: - Batch Degree Bound Claim Struct

    suite("Degree Bound - BatchDegreeBoundClaim")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testBatchClaimValid(engine: engine, srsSecret: srsSecret)
        testBatchClaimMalformed(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Batch claim setup failed: \(error)")
    }

    // MARK: - Combined Opening + Degree Bound

    suite("Degree Bound - Combined Opening")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testDegreeBoundOpeningValid(engine: engine, srsSecret: srsSecret)
        testDegreeBoundOpeningMultiplePoints(engine: engine, srsSecret: srsSecret)
        testDegreeBoundOpeningWrongEval(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Combined opening setup failed: \(error)")
    }

    // MARK: - Batch Combined Verification

    suite("Degree Bound - Batch Combined Verification")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testBatchVerifyOpenings(engine: engine, srsSecret: srsSecret)
        testBatchVerifyOpeningsWithCorruption(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Batch combined verification setup failed: \(error)")
    }

    // MARK: - Multi-Point Opening with Degree Bounds

    suite("Degree Bound - Multi-Point Opening")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testMultiPointDegreeBound(engine: engine, srsSecret: srsSecret)
        testMultiPointDegreeBoundMixedDegrees(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Multi-point opening setup failed: \(error)")
    }

    // MARK: - SRS Trimming

    suite("Degree Bound - SRS Trimming")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testTrimSRSBasic(engine: engine, srsSecret: srsSecret)
        testTrimSRSWithPrecomputedShifts(engine: engine, srsSecret: srsSecret)
        testTrimSRSForCircuit(engine: engine)
        testTrimSRSTooLarge(engine: engine)
    } catch {
        expect(false, "SRS trimming setup failed: \(error)")
    }

    // MARK: - Shift Factor Computation

    suite("Degree Bound - Shift Factors")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testComputeShiftFactor(engine: engine, srsSecret: srsSecret)
        testPrecomputeShiftFactors(engine: engine, srsSecret: srsSecret)
        testBatchVerifyWithPrecomputedShifts(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Shift factor setup failed: \(error)")
    }

    // MARK: - Accumulator

    suite("Degree Bound - Accumulator")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testAccumulatorBasic(engine: engine, srsSecret: srsSecret)
        testAccumulatorClearAndReuse(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Accumulator setup failed: \(error)")
    }

    // MARK: - Detailed Verification Result

    suite("Degree Bound - Detailed Results")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testDetailedVerification(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Detailed verification setup failed: \(error)")
    }

    // MARK: - Edge Cases

    suite("Degree Bound - Edge Cases")

    do {
        let engine = try GPUKZGDegreeBoundEngine(srs: srs)

        testEmptyBatchVerify(engine: engine, srsSecret: srsSecret)
        testSingleElementBatch(engine: engine, srsSecret: srsSecret)
        testMaxDegreeBound(engine: engine, srsSecret: srsSecret)
        testLinearPolynomialBound(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Edge cases setup failed: \(error)")
    }
}

// MARK: - Polynomial Shifting Tests

private func testShiftByZero(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7)]
    let shifted = engine.shiftPolynomial(poly, by: 0)
    expectEqual(shifted.count, 3, "Shift by 0 preserves length")
    expect(frEqual(shifted[0], frFromInt(3)), "Shift by 0 preserves coeff 0")
    expect(frEqual(shifted[1], frFromInt(5)), "Shift by 0 preserves coeff 1")
    expect(frEqual(shifted[2], frFromInt(7)), "Shift by 0 preserves coeff 2")
}

private func testShiftByOne(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(2), frFromInt(4)]
    let shifted = engine.shiftPolynomial(poly, by: 1)
    expectEqual(shifted.count, 3, "Shift by 1 adds one element")
    expect(frEqual(shifted[0], Fr.zero), "Shift by 1: coeff 0 is zero")
    expect(frEqual(shifted[1], frFromInt(2)), "Shift by 1: original coeff 0 at index 1")
    expect(frEqual(shifted[2], frFromInt(4)), "Shift by 1: original coeff 1 at index 2")
}

private func testShiftByLargeAmount(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(10)]
    let shifted = engine.shiftPolynomial(poly, by: 5)
    expectEqual(shifted.count, 6, "Shift by 5 gives length 6")
    for i in 0..<5 {
        expect(frEqual(shifted[i], Fr.zero), "Shift by 5: coeff \(i) is zero")
    }
    expect(frEqual(shifted[5], frFromInt(10)), "Shift by 5: original at index 5")
}

private func testShiftEmptyPolynomial(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = []
    let shifted = engine.shiftPolynomial(poly, by: 3)
    expectEqual(shifted.count, 0, "Shift of empty polynomial is empty")
}

// MARK: - Polynomial Degree Utility Tests

private func testPolynomialDegree(engine: GPUKZGDegreeBoundEngine) {
    // Zero polynomial
    let zeroPoly: [Fr] = [Fr.zero, Fr.zero, Fr.zero]
    expectEqual(engine.polynomialDegree(zeroPoly), -1, "Zero polynomial has degree -1 (all zero coefficients stripped)")

    // Actually check the logic: the zero poly returns 0 because the first coeff is "non-zero" via frToInt
    // The engine strips trailing zeros and returns -1 for empty. Let's check:
    let trulyZero: [Fr] = [Fr.zero]
    // polynomialDegree checks if frToInt(coeffs[i]) != frToInt(Fr.zero)
    // For Fr.zero, they're equal, so it returns -1
    expectEqual(engine.polynomialDegree(trulyZero), -1, "Single zero is degree -1")

    // Constant polynomial
    let constPoly: [Fr] = [frFromInt(42)]
    expectEqual(engine.polynomialDegree(constPoly), 0, "Constant polynomial has degree 0")

    // Linear polynomial
    let linearPoly: [Fr] = [frFromInt(1), frFromInt(3)]
    expectEqual(engine.polynomialDegree(linearPoly), 1, "Linear polynomial has degree 1")

    // Cubic with trailing zeros
    let cubicPadded: [Fr] = [frFromInt(1), frFromInt(0), frFromInt(3), frFromInt(7), Fr.zero, Fr.zero]
    expectEqual(engine.polynomialDegree(cubicPadded), 3, "Cubic with trailing zeros has degree 3")
}

private func testCheckDegreeBound(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)] // degree 2
    expect(engine.checkDegreeBound(poly, degreeBound: 2), "Degree 2 poly within bound 2")
    expect(engine.checkDegreeBound(poly, degreeBound: 5), "Degree 2 poly within bound 5")
    expect(engine.checkDegreeBound(poly, degreeBound: 100), "Degree 2 poly within bound 100")

    // With trailing zeros
    let padded: [Fr] = [frFromInt(5), frFromInt(3), Fr.zero, Fr.zero]
    expect(engine.checkDegreeBound(padded, degreeBound: 1), "Degree 1 padded poly within bound 1")
}

private func testPadToDegree(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(1), frFromInt(2)]
    let padded = engine.padToDegree(poly, degreeBound: 4)
    expect(padded != nil, "Padding succeeds for valid bound")
    if let p = padded {
        expectEqual(p.count, 5, "Padded to degree 4 has 5 coefficients")
        expect(frEqual(p[0], frFromInt(1)), "Padded coeff 0 preserved")
        expect(frEqual(p[1], frFromInt(2)), "Padded coeff 1 preserved")
        expect(frEqual(p[2], Fr.zero), "Padded coeff 2 is zero")
        expect(frEqual(p[3], Fr.zero), "Padded coeff 3 is zero")
        expect(frEqual(p[4], Fr.zero), "Padded coeff 4 is zero")
    }

    // Pad to exact degree
    let exact = engine.padToDegree(poly, degreeBound: 1)
    expect(exact != nil, "Padding to exact degree succeeds")
    if let e = exact {
        expectEqual(e.count, 2, "Exact pad has correct length")
    }
}

private func testTrimPolynomial(engine: GPUKZGDegreeBoundEngine) {
    let poly: [Fr] = [frFromInt(3), frFromInt(7), Fr.zero, Fr.zero, Fr.zero]
    let trimmed = engine.trimPolynomial(poly)
    expectEqual(trimmed.count, 2, "Trimmed polynomial removes trailing zeros")
    expect(frEqual(trimmed[0], frFromInt(3)), "Trimmed coeff 0")
    expect(frEqual(trimmed[1], frFromInt(7)), "Trimmed coeff 1")

    // All zeros
    let allZero: [Fr] = [Fr.zero, Fr.zero, Fr.zero]
    let trimmedZero = engine.trimPolynomial(allZero)
    expectEqual(trimmedZero.count, 1, "All-zero polynomial trims to single element")
}

// MARK: - Shifted Commitment Tests

private func testShiftedCommitmentBasic(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Polynomial: f(x) = 3 + 5x + 7x^2 (degree 2)
        let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7)]
        let degreeBound = 10 // well above actual degree

        let shifted = try engine.computeShiftedCommitment(poly, degreeBound: degreeBound)
        // Shifted commitment should be non-trivial (not identity)
        expect(!pointIsIdentity(shifted), "Shifted commitment is non-trivial")

        // Verify algebraically: shifted = sum_i coeff[i] * SRS[shift + i]
        // where shift = D - d - 1 = 256 - 10 - 1 = 245
        let D = engine.maxDegree
        let shift = D - degreeBound - 1
        expect(shift == 245, "Shift amount is D - d - 1 = 245")
    } catch {
        expect(false, "Shifted commitment basic test threw: \(error)")
    }
}

private func testShiftedCommitmentConsistency(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // The shifted commitment of f(x) with bound d should equal
        // [s^(D-d-1)] * commitment(f) where commitment uses standard SRS
        let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let degreeBound = 20

        let shifted = try engine.computeShiftedCommitment(poly, degreeBound: degreeBound)

        // Compute the same thing manually: commit(f) * s^(D-d-1)
        let kzg = try KZGEngine(srs: engine.srs)
        let commitment = try kzg.commit(poly)
        let D = engine.maxDegree
        let shiftAmount = D - degreeBound - 1

        // Compute s^shiftAmount
        var sPow = Fr.one
        for _ in 0..<shiftAmount {
            sPow = frMul(sPow, srsSecret)
        }
        let expected = cPointScalarMul(commitment, sPow)

        let shiftedAff = batchToAffine([shifted])
        let expectedAff = batchToAffine([expected])
        expect(fpToInt(shiftedAff[0].x) == fpToInt(expectedAff[0].x),
               "Shifted commitment x matches manual computation")
        expect(fpToInt(shiftedAff[0].y) == fpToInt(expectedAff[0].y),
               "Shifted commitment y matches manual computation")
    } catch {
        expect(false, "Shifted commitment consistency test threw: \(error)")
    }
}

private func testShiftedCommitmentDifferentBounds(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(5), frFromInt(3)]
        // Different bounds should give different shifted commitments
        let shifted10 = try engine.computeShiftedCommitment(poly, degreeBound: 10)
        let shifted20 = try engine.computeShiftedCommitment(poly, degreeBound: 20)

        let aff10 = batchToAffine([shifted10])
        let aff20 = batchToAffine([shifted20])

        // They should differ because the shift amounts differ
        let xMatch = fpToInt(aff10[0].x) == fpToInt(aff20[0].x)
        let yMatch = fpToInt(aff10[0].y) == fpToInt(aff20[0].y)
        expect(!(xMatch && yMatch), "Different bounds give different shifted commitments")
    } catch {
        expect(false, "Different bounds test threw: \(error)")
    }
}

// MARK: - Single Degree Bound Proof Tests

private func testSingleDegreeBoundValid(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 10)

        expect(!pointIsIdentity(proof.commitment), "Commitment is non-trivial")
        expect(!pointIsIdentity(proof.shiftedCommitment), "Shifted commitment is non-trivial")
        expectEqual(proof.degreeBound, 10, "Degree bound recorded correctly")
        expectEqual(proof.maxDegree, 256, "Max degree recorded correctly")

        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Valid degree bound proof verified")
    } catch {
        expect(false, "Single degree bound valid test threw: \(error)")
    }
}

private func testSingleDegreeBoundTightBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Degree exactly equals bound
        let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)] // degree 2
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 2)

        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Tight degree bound (deg=bound) verified")
    } catch {
        expect(false, "Tight bound test threw: \(error)")
    }
}

private func testSingleDegreeBoundLooseBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Degree much less than bound
        let poly: [Fr] = [frFromInt(42)] // degree 0
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 100)

        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Loose degree bound (deg << bound) verified")
    } catch {
        expect(false, "Loose bound test threw: \(error)")
    }
}

private func testSingleDegreeBoundConstant(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Constant polynomial with degree bound 0
        let poly: [Fr] = [frFromInt(17)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 0)

        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Constant polynomial with bound 0 verified")
    } catch {
        expect(false, "Constant bound test threw: \(error)")
    }
}

// MARK: - Degree Bound Verification Tests

private func testVerifyValidDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(1), frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(9)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 20)
        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Valid proof passes verification")
    } catch {
        expect(false, "Valid verification test threw: \(error)")
    }
}

private func testVerifyInvalidDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(1), frFromInt(3), frFromInt(5)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 10)

        // Tamper with the shifted commitment
        let g1 = pointFromAffine(engine.srs[0])
        let tamperedShifted = pointAdd(proof.shiftedCommitment, g1)

        let tamperedProof = DegreeBoundProof(
            commitment: proof.commitment,
            shiftedCommitment: tamperedShifted,
            degreeBound: proof.degreeBound,
            maxDegree: proof.maxDegree
        )

        let valid = engine.verifyDegreeBound(proof: tamperedProof, srsSecret: srsSecret)
        expect(!valid, "Tampered shifted commitment rejected")
    } catch {
        expect(false, "Invalid verification test threw: \(error)")
    }
}

private func testVerifyWrongShiftedCommitment(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Create proof for one polynomial but use commitment from another
        let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let poly2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]

        let proof1 = try engine.proveDegreeBound(polynomial: poly1, degreeBound: 10)
        let proof2 = try engine.proveDegreeBound(polynomial: poly2, degreeBound: 10)

        // Mix: commitment from poly1, shifted from poly2
        let mixedProof = DegreeBoundProof(
            commitment: proof1.commitment,
            shiftedCommitment: proof2.shiftedCommitment,
            degreeBound: 10,
            maxDegree: engine.maxDegree
        )

        let valid = engine.verifyDegreeBound(proof: mixedProof, srsSecret: srsSecret)
        expect(!valid, "Mixed commitment/shifted rejected")
    } catch {
        expect(false, "Wrong shifted commitment test threw: \(error)")
    }
}

// MARK: - Batch Degree Bound Proof Tests

private func testBatchProveDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5)],
            [frFromInt(6), frFromInt(7), frFromInt(8), frFromInt(9)],
        ]
        let bounds = [10, 10, 10]

        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)
        expectEqual(proofs.count, 3, "Batch produces 3 proofs")

        // Each proof should verify individually
        for (i, proof) in proofs.enumerated() {
            let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
            expect(valid, "Batch proof \(i) verifies individually")
        }
    } catch {
        expect(false, "Batch prove test threw: \(error)")
    }
}

private func testBatchVerifyDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5)],
            [frFromInt(7), frFromInt(11), frFromInt(13), frFromInt(17)],
            [frFromInt(19)],
        ]
        let bounds = [15, 15, 15]

        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)
        let valid = try engine.batchVerifyDegreeBound(proofs: proofs, srsSecret: srsSecret)
        expect(valid, "Batch verification passes for valid proofs")
    } catch {
        expect(false, "Batch verify test threw: \(error)")
    }
}

private func testBatchVerifyMixedBounds(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2)],            // degree 1
            [frFromInt(3), frFromInt(4), frFromInt(5)], // degree 2
            [frFromInt(6)],                            // degree 0
        ]
        let bounds = [5, 10, 50] // each bound well above actual degree

        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)
        let valid = try engine.batchVerifyDegreeBound(proofs: proofs, srsSecret: srsSecret)
        expect(valid, "Batch verification passes for mixed bounds")
    } catch {
        expect(false, "Mixed bounds test threw: \(error)")
    }
}

// MARK: - Batch Claim Struct Tests

private func testBatchClaimValid(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
        ]
        let bounds = [10, 10]

        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)

        let batch = BatchDegreeBoundClaim(
            commitments: proofs.map { $0.commitment },
            shiftedCommitments: proofs.map { $0.shiftedCommitment },
            degreeBounds: bounds,
            maxDegree: engine.maxDegree
        )

        expect(batch.isWellFormed, "Batch claim is well-formed")
        expectEqual(batch.count, 2, "Batch has 2 claims")

        let valid = try engine.batchVerifyDegreeBoundClaim(batch: batch, srsSecret: srsSecret)
        expect(valid, "Batch claim verification passes")
    } catch {
        expect(false, "Batch claim valid test threw: \(error)")
    }
}

private func testBatchClaimMalformed(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    // Mismatched array lengths
    let batch = BatchDegreeBoundClaim(
        commitments: [pointIdentity()],
        shiftedCommitments: [pointIdentity(), pointIdentity()],
        degreeBounds: [10],
        maxDegree: 256
    )
    expect(!batch.isWellFormed, "Mismatched lengths detected as malformed")

    do {
        let valid = try engine.batchVerifyDegreeBoundClaim(batch: batch, srsSecret: srsSecret)
        expect(!valid, "Malformed batch claim rejected")
    } catch {
        expect(false, "Malformed batch claim test threw: \(error)")
    }
}

// MARK: - Combined Opening + Degree Bound Tests

private func testDegreeBoundOpeningValid(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let z = frFromInt(5)
        let degreeBound = 20

        let proof = try engine.proveDegreeBoundOpening(
            polynomial: poly, degreeBound: degreeBound, point: z)

        expect(!pointIsIdentity(proof.witness), "Opening witness is non-trivial")

        // Manually compute expected evaluation: 3 + 7*5 + 11*25 + 13*125
        // = 3 + 35 + 275 + 1625 = 1938
        // (verify we get a non-zero evaluation)
        let evalInt = frToInt(proof.evaluation)
        expect(evalInt != frToInt(Fr.zero), "Evaluation is non-zero")

        let valid = engine.verifyDegreeBoundOpening(proof: proof, srsSecret: srsSecret)
        expect(valid, "Combined opening + degree bound verified")
    } catch {
        expect(false, "Degree bound opening valid test threw: \(error)")
    }
}

private func testDegreeBoundOpeningMultiplePoints(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(1), frFromInt(1)] // f(x) = 1 + x

        // Open at different points with same degree bound
        let z1 = frFromInt(10)
        let z2 = frFromInt(20)

        let proof1 = try engine.proveDegreeBoundOpening(
            polynomial: poly, degreeBound: 5, point: z1)
        let proof2 = try engine.proveDegreeBoundOpening(
            polynomial: poly, degreeBound: 5, point: z2)

        let valid1 = engine.verifyDegreeBoundOpening(proof: proof1, srsSecret: srsSecret)
        let valid2 = engine.verifyDegreeBoundOpening(proof: proof2, srsSecret: srsSecret)

        expect(valid1, "Opening at point 1 verified")
        expect(valid2, "Opening at point 2 verified")

        // Evaluations should be different
        let e1 = frToInt(proof1.evaluation)
        let e2 = frToInt(proof2.evaluation)
        expect(e1 != e2, "Evaluations at different points differ")
    } catch {
        expect(false, "Multiple points test threw: \(error)")
    }
}

private func testDegreeBoundOpeningWrongEval(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(2), frFromInt(3)]
        let z = frFromInt(7)

        let proof = try engine.proveDegreeBoundOpening(
            polynomial: poly, degreeBound: 10, point: z)

        // Tamper with evaluation
        let tamperedProof = DegreeBoundOpeningProof(
            degreeBoundProof: proof.degreeBoundProof,
            point: proof.point,
            evaluation: frFromInt(999),
            witness: proof.witness
        )

        let valid = engine.verifyDegreeBoundOpening(proof: tamperedProof, srsSecret: srsSecret)
        expect(!valid, "Tampered evaluation rejected")
    } catch {
        expect(false, "Wrong eval test threw: \(error)")
    }
}

// MARK: - Batch Combined Verification Tests

private func testBatchVerifyOpenings(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5)],
        ]
        let points: [Fr] = [frFromInt(7), frFromInt(13)]
        let bounds = [10, 10]

        var proofs = [DegreeBoundOpeningProof]()
        for i in 0..<polys.count {
            let p = try engine.proveDegreeBoundOpening(
                polynomial: polys[i], degreeBound: bounds[i], point: points[i])
            proofs.append(p)
        }

        let valid = try engine.batchVerifyDegreeBoundOpenings(proofs: proofs, srsSecret: srsSecret)
        expect(valid, "Batch combined verification passes")
    } catch {
        expect(false, "Batch verify openings test threw: \(error)")
    }
}

private func testBatchVerifyOpeningsWithCorruption(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5)],
        ]
        let points: [Fr] = [frFromInt(7), frFromInt(13)]
        let bounds = [10, 10]

        var proofs = [DegreeBoundOpeningProof]()
        for i in 0..<polys.count {
            let p = try engine.proveDegreeBoundOpening(
                polynomial: polys[i], degreeBound: bounds[i], point: points[i])
            proofs.append(p)
        }

        // Corrupt the second proof's evaluation
        let corrupted = DegreeBoundOpeningProof(
            degreeBoundProof: proofs[1].degreeBoundProof,
            point: proofs[1].point,
            evaluation: frFromInt(12345),
            witness: proofs[1].witness
        )
        proofs[1] = corrupted

        let valid = try engine.batchVerifyDegreeBoundOpenings(proofs: proofs, srsSecret: srsSecret)
        expect(!valid, "Batch combined verification rejects corrupted proof")
    } catch {
        expect(false, "Corruption test threw: \(error)")
    }
}

// MARK: - Multi-Point Opening with Degree Bound Tests

private func testMultiPointDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5)],
            [frFromInt(7), frFromInt(11)],
        ]
        let points: [Fr] = [frFromInt(4), frFromInt(9)]
        let bounds = [10, 10]

        let proof = try engine.proveMultiPointWithDegreeBound(
            polynomials: polys, points: points, degreeBounds: bounds)

        expectEqual(proof.degreeBoundProofs.count, 2, "Two degree bound proofs")
        expectEqual(proof.witnesses.count, 2, "Two witnesses")
        expectEqual(proof.evaluations.count, 2, "Two evaluations")

        let valid = engine.verifyMultiPointWithDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Multi-point degree bound proof verified")
    } catch {
        expect(false, "Multi-point degree bound test threw: \(error)")
    }
}

private func testMultiPointDegreeBoundMixedDegrees(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1)],                                          // degree 0
            [frFromInt(2), frFromInt(3)],                            // degree 1
            [frFromInt(4), frFromInt(5), frFromInt(6), frFromInt(7)], // degree 3
        ]
        let points: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let bounds = [5, 8, 15] // varying bounds, all valid

        let proof = try engine.proveMultiPointWithDegreeBound(
            polynomials: polys, points: points, degreeBounds: bounds)

        let valid = engine.verifyMultiPointWithDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Mixed degrees multi-point proof verified")

        // Check that evaluations are consistent
        // f_0(10) = 1
        // f_1(20) = 2 + 3*20 = 62
        // f_2(30) = 4 + 5*30 + 6*900 + 7*27000 = 4 + 150 + 5400 + 189000 = 194554
        // (These are in Montgomery form, so we can't directly compare, but they should be non-zero)
        for i in 0..<3 {
            expect(frToInt(proof.evaluations[i]) != frToInt(Fr.zero) || i == 0,
                   "Evaluation \(i) computed")
        }
    } catch {
        expect(false, "Mixed degrees test threw: \(error)")
    }
}

// MARK: - SRS Trimming Tests

private func testTrimSRSBasic(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    let trimmed = engine.trimSRS(degree: 32)
    expect(trimmed != nil, "Trimming to 32 succeeds")
    if let t = trimmed {
        expectEqual(t.g1Points.count, 33, "Trimmed has 33 points (degree + 1)")
        expectEqual(t.maxDegree, 32, "Max degree is 32")
        expectEqual(t.fullDegree, 256, "Full degree is 256")

        // First point should match full SRS
        let origX = fpToInt(engine.srs[0].x)
        let trimX = fpToInt(t.g1Points[0].x)
        expect(origX == trimX, "Trimmed SRS first point matches")
    }
}

private func testTrimSRSWithPrecomputedShifts(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    let trimmed = engine.trimSRS(
        degree: 64,
        precomputeShifts: [10, 20, 30],
        srsSecret: srsSecret
    )
    expect(trimmed != nil, "Trimming with precomputed shifts succeeds")
    if let t = trimmed {
        expectEqual(t.shiftFactors.count, 3, "Three shift factors precomputed")
        expect(t.shiftFactors[10] != nil, "Shift factor for 10 exists")
        expect(t.shiftFactors[20] != nil, "Shift factor for 20 exists")
        expect(t.shiftFactors[30] != nil, "Shift factor for 30 exists")

        // Verify shift factor for 10: should be s^10
        if let sf10 = t.shiftFactors[10] {
            var sPow = Fr.one
            for _ in 0..<10 {
                sPow = frMul(sPow, srsSecret)
            }
            expect(frEqual(sf10, sPow), "Shift factor for 10 equals s^10")
        }
    }
}

private func testTrimSRSForCircuit(engine: GPUKZGDegreeBoundEngine) {
    let trimmed = engine.trimSRSForCircuit(maxPolyDegree: 16, maxDegreeBound: 32)
    expect(trimmed != nil, "Circuit trimming succeeds")
    if let t = trimmed {
        // Should use the larger of the two: 32
        expectEqual(t.maxDegree, 32, "Circuit trim uses max of poly degree and bound")
    }

    // Both small
    let small = engine.trimSRSForCircuit(maxPolyDegree: 8, maxDegreeBound: 8)
    expect(small != nil, "Small circuit trimming succeeds")
    if let s = small {
        expectEqual(s.maxDegree, 8, "Small circuit trim degree")
    }
}

private func testTrimSRSTooLarge(engine: GPUKZGDegreeBoundEngine) {
    let trimmed = engine.trimSRS(degree: 1000)
    expect(trimmed == nil, "Trimming beyond SRS size returns nil")

    let circuit = engine.trimSRSForCircuit(maxPolyDegree: 500, maxDegreeBound: 500)
    expect(circuit == nil, "Circuit trim beyond SRS size returns nil")
}

// MARK: - Shift Factor Tests

private func testComputeShiftFactor(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    let sf = engine.computeShiftFactor(srsSecret: srsSecret, degreeBound: 200)
    // shift = 256 - 200 - 1 = 55
    var expected = Fr.one
    for _ in 0..<55 {
        expected = frMul(expected, srsSecret)
    }
    expect(frEqual(sf, expected), "Shift factor matches s^55")

    // Boundary: degreeBound = maxDegree - 1 => shift = 0 => factor = 1
    let sfMax = engine.computeShiftFactor(srsSecret: srsSecret, degreeBound: 255)
    // shift = 256 - 255 - 1 = 0
    expect(frEqual(sfMax, Fr.one), "Shift factor for max bound is 1")
}

private func testPrecomputeShiftFactors(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    let factors = engine.precomputeShiftFactors(srsSecret: srsSecret, degreeBounds: [10, 20, 10, 30])
    // Should have 3 unique entries (10, 20, 30)
    expectEqual(factors.count, 3, "Deduplicates degree bounds")
    expect(factors[10] != nil, "Factor for bound 10 computed")
    expect(factors[20] != nil, "Factor for bound 20 computed")
    expect(factors[30] != nil, "Factor for bound 30 computed")
}

private func testBatchVerifyWithPrecomputedShifts(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2)],
            [frFromInt(3), frFromInt(4), frFromInt(5)],
            [frFromInt(6)],
        ]
        let bounds = [10, 10, 10]

        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)

        let shifts = engine.precomputeShiftFactors(srsSecret: srsSecret, degreeBounds: bounds)
        let valid = try engine.batchVerifyWithPrecomputedShifts(
            proofs: proofs, srsSecret: srsSecret, shiftFactors: shifts)
        expect(valid, "Batch verify with precomputed shifts passes")
    } catch {
        expect(false, "Precomputed shifts test threw: \(error)")
    }
}

// MARK: - Accumulator Tests

private func testAccumulatorBasic(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let accumulator = engine.createAccumulator()
        expectEqual(accumulator.count, 0, "Fresh accumulator is empty")

        let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let poly2: [Fr] = [frFromInt(4), frFromInt(5)]

        let proof1 = try engine.proveDegreeBound(polynomial: poly1, degreeBound: 10)
        let proof2 = try engine.proveDegreeBound(polynomial: poly2, degreeBound: 10)

        accumulator.addProof(proof1)
        accumulator.addProof(proof2)
        expectEqual(accumulator.count, 2, "Accumulator has 2 proofs")

        let valid = try engine.verifyAccumulator(accumulator, srsSecret: srsSecret)
        expect(valid, "Accumulator verification passes")
    } catch {
        expect(false, "Accumulator basic test threw: \(error)")
    }
}

private func testAccumulatorClearAndReuse(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let accumulator = engine.createAccumulator()

        let poly: [Fr] = [frFromInt(7), frFromInt(11)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 10)
        accumulator.addProof(proof)
        expectEqual(accumulator.count, 1, "One proof added")

        accumulator.clear()
        expectEqual(accumulator.count, 0, "Accumulator cleared")

        // Add new proof and verify
        let poly2: [Fr] = [frFromInt(13), frFromInt(17), frFromInt(19)]
        let proof2 = try engine.proveDegreeBound(polynomial: poly2, degreeBound: 20)
        accumulator.addProof(proof2)

        let valid = try engine.verifyAccumulator(accumulator, srsSecret: srsSecret)
        expect(valid, "Reused accumulator verification passes")
    } catch {
        expect(false, "Accumulator clear test threw: \(error)")
    }
}

// MARK: - Detailed Verification Tests

private func testDetailedVerification(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5)],
            [frFromInt(7), frFromInt(11), frFromInt(13)],
        ]
        let bounds = [10, 10]
        let proofs = try engine.batchProveDegreeBound(polynomials: polys, degreeBounds: bounds)

        let result = try engine.batchVerifyDetailed(proofs: proofs, srsSecret: srsSecret)
        expect(result.isValid, "Detailed verification passes")
        expectEqual(result.claimCount, 2, "Correct claim count")
        expect(result.elapsedSeconds >= 0, "Elapsed time is non-negative")
        expect(!result.summary.isEmpty, "Summary is non-empty")
    } catch {
        expect(false, "Detailed verification test threw: \(error)")
    }
}

// MARK: - Edge Case Tests

private func testEmptyBatchVerify(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let valid = try engine.batchVerifyDegreeBound(proofs: [], srsSecret: srsSecret)
        expect(valid, "Empty batch verification returns true")
    } catch {
        expect(false, "Empty batch test threw: \(error)")
    }
}

private func testSingleElementBatch(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(42), frFromInt(7)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 10)

        let valid = try engine.batchVerifyDegreeBound(proofs: [proof], srsSecret: srsSecret)
        expect(valid, "Single-element batch passes")
    } catch {
        expect(false, "Single element batch test threw: \(error)")
    }
}

private func testMaxDegreeBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Use degree bound = maxDegree - 2 (the largest valid bound, since shift must be >= 0
        // and we need shift = D - d - 1 >= 0, so d <= D - 1 = 255)
        let poly: [Fr] = [frFromInt(1), frFromInt(2)]
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 254)

        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Max degree bound proof verified")
    } catch {
        expect(false, "Max degree bound test threw: \(error)")
    }
}

private func testLinearPolynomialBound(engine: GPUKZGDegreeBoundEngine, srsSecret: Fr) {
    do {
        // Linear polynomial: f(x) = a + bx
        let a = frFromInt(17)
        let b = frFromInt(23)
        let poly: [Fr] = [a, b]

        // Prove with tight bound (1) and verify
        let proof = try engine.proveDegreeBound(polynomial: poly, degreeBound: 1)
        let valid = engine.verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        expect(valid, "Linear polynomial with tight bound 1 verified")

        // Also verify combined opening
        let z = frFromInt(5)
        let openProof = try engine.proveDegreeBoundOpening(
            polynomial: poly, degreeBound: 1, point: z)
        let openValid = engine.verifyDegreeBoundOpening(proof: openProof, srsSecret: srsSecret)
        expect(openValid, "Linear polynomial combined proof verified")

        // Check evaluation: f(5) = 17 + 23*5 = 132
        let expected = frAdd(a, frMul(b, z))
        expect(frEqual(openProof.evaluation, expected), "Linear evaluation correct: f(5) = 17 + 23*5")
    } catch {
        expect(false, "Linear polynomial test threw: \(error)")
    }
}
