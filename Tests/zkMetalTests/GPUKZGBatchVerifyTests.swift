// GPUKZGBatchVerifyEngine tests: single verify, batch verify, multi-point,
// same-point, cross-polynomial, accumulator, pairing, transcript-based
import zkMetal
import Foundation

public func runGPUKZGBatchVerifyTests() {
    // Common setup: test SRS with known secret
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)

    // MARK: - Single Claim Verification

    suite("GPU KZG Batch Verify - Single Claim")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testSingleClaimValid(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testSingleClaimWrongValue(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testSingleClaimWrongPoint(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Single claim setup failed: \(error)")
    }

    // MARK: - Multi-Point Batch Verification

    suite("GPU KZG Batch Verify - Multi-Point Batch")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testMultiPointBatchValid(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testMultiPointBatchCorrupted(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testMultiPointBatchMismatchedLengths(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Multi-point batch setup failed: \(error)")
    }

    // MARK: - Same-Point Batch Verification

    suite("GPU KZG Batch Verify - Same-Point Batch")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testSamePointBatchValid(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testSamePointBatchPlonkScale(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testSamePointBatchCorrupted(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Same-point batch setup failed: \(error)")
    }

    // MARK: - Struct-Based APIs

    suite("GPU KZG Batch Verify - Struct APIs")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testBatchVerifyClaimStruct(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testMultiOpenBatchClaimStruct(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testSamePointBatchClaimStruct(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Struct API setup failed: \(error)")
    }

    // MARK: - Empty and Edge Cases

    suite("GPU KZG Batch Verify - Edge Cases")

    do {
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testEmptyClaims(engine: engine, srsSecret: srsSecret)
        testMalformedMultiOpenBatch(engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Edge cases setup failed: \(error)")
    }

    // MARK: - Detailed Batch Verification

    suite("GPU KZG Batch Verify - Detailed Results")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testDetailedVerification(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Detailed verification setup failed: \(error)")
    }

    // MARK: - Batch Accumulator

    suite("GPU KZG Batch Verify - Accumulator")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testAccumulatorBasic(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testAccumulatorClearAndReuse(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Accumulator setup failed: \(error)")
    }

    // MARK: - Cross-Polynomial Batch Verification

    suite("GPU KZG Batch Verify - Cross-Polynomial")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testCrossPolynomialBatch(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testCrossPolynomialMultiplePointsPerPoly(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Cross-polynomial setup failed: \(error)")
    }

    // MARK: - Transcript-Based Verification

    suite("GPU KZG Batch Verify - Transcript-Based")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testTranscriptBasedVerification(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testTranscriptConsistency(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Transcript-based setup failed: \(error)")
    }

    // MARK: - Pairing-Based Verification

    suite("GPU KZG Batch Verify - Pairing Verification")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testPairingVerificationSingle(kzg: kzg, engine: engine, srs: srs, srsSecret: srsSecret)
    } catch {
        expect(false, "Pairing verification setup failed: \(error)")
    }

    // MARK: - Batch Size Scaling

    suite("GPU KZG Batch Verify - Batch Size Scaling")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testBatchSizeScaling(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Batch size scaling setup failed: \(error)")
    }

    // MARK: - Array-Based Convenience APIs

    suite("GPU KZG Batch Verify - Array APIs")

    do {
        let kzg = try KZGEngine(srs: srs)
        let engine = try GPUKZGBatchVerifyEngine(srs: srs)

        testArrayBasedBatchVerify(kzg: kzg, engine: engine, srsSecret: srsSecret)
        testArrayBasedSamePointVerify(kzg: kzg, engine: engine, srsSecret: srsSecret)
    } catch {
        expect(false, "Array API setup failed: \(error)")
    }
}

// MARK: - Single Claim Tests

private func testSingleClaimValid(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let z = frFromInt(13)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        let claim = BatchVerifyClaim(
            commitment: commitment, point: z,
            value: proof.evaluation, proof: proof.witness)

        let valid = try engine.batchVerify(claims: [claim], srsSecret: srsSecret)
        expect(valid, "Single valid claim accepted")
    } catch {
        expect(false, "Single claim valid test threw: \(error)")
    }
}

private func testSingleClaimWrongValue(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7)]
        let z = frFromInt(17)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        let claim = BatchVerifyClaim(
            commitment: commitment, point: z,
            value: frFromInt(999), proof: proof.witness)

        let valid = try engine.batchVerify(claims: [claim], srsSecret: srsSecret)
        expect(!valid, "Single claim with wrong value rejected")
    } catch {
        expect(false, "Single claim wrong value test threw: \(error)")
    }
}

private func testSingleClaimWrongPoint(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(2), frFromInt(4), frFromInt(6)]
        let z = frFromInt(10)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        // Use the correct evaluation but claim it was at a different point
        let claim = BatchVerifyClaim(
            commitment: commitment, point: frFromInt(20),
            value: proof.evaluation, proof: proof.witness)

        let valid = try engine.batchVerify(claims: [claim], srsSecret: srsSecret)
        expect(!valid, "Single claim with wrong point rejected")
    } catch {
        expect(false, "Single claim wrong point test threw: \(error)")
    }
}

// MARK: - Multi-Point Batch Tests

private func testMultiPointBatchValid(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
            [frFromInt(5), frFromInt(6), frFromInt(7)],
            [frFromInt(8), frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)],
        ]
        let points: [Fr] = [frFromInt(7), frFromInt(13), frFromInt(42)]

        var claims = [BatchVerifyClaim]()
        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            claims.append(BatchVerifyClaim(
                commitment: c, point: points[i],
                value: p.evaluation, proof: p.witness))
        }

        let valid = try engine.batchVerify(claims: claims, srsSecret: srsSecret)
        expect(valid, "Multi-point batch (3 polys, different points): valid")
    } catch {
        expect(false, "Multi-point batch valid test threw: \(error)")
    }
}

private func testMultiPointBatchCorrupted(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
            [frFromInt(7), frFromInt(8), frFromInt(9)],
        ]
        let points: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]

        var claims = [BatchVerifyClaim]()
        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            claims.append(BatchVerifyClaim(
                commitment: c, point: points[i],
                value: p.evaluation, proof: p.witness))
        }

        // Corrupt the second claim's value
        claims[1] = BatchVerifyClaim(
            commitment: claims[1].commitment,
            point: claims[1].point,
            value: frFromInt(12345),
            proof: claims[1].proof)

        let valid = try engine.batchVerify(claims: claims, srsSecret: srsSecret)
        expect(!valid, "Multi-point batch with corrupted value: rejected")
    } catch {
        expect(false, "Multi-point batch corrupted test threw: \(error)")
    }
}

private func testMultiPointBatchMismatchedLengths(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(1), frFromInt(2)]
        let z = frFromInt(3)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        // Create a MultiOpenBatchClaim with mismatched lengths
        let batch = MultiOpenBatchClaim(
            commitments: [commitment, commitment],
            points: [z],  // only 1 point for 2 commitments
            values: [proof.evaluation],
            proofs: [proof.witness])

        expect(!batch.isWellFormed, "Mismatched lengths detected by isWellFormed")

        let valid = try engine.batchVerifyMultiOpen(batch: batch, srsSecret: srsSecret)
        expect(!valid, "Mismatched array lengths: rejected")
    } catch {
        expect(false, "Mismatched lengths test threw: \(error)")
    }
}

// MARK: - Same-Point Batch Tests

private func testSamePointBatchValid(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5)],
            [frFromInt(7), frFromInt(11), frFromInt(13)],
            [frFromInt(17), frFromInt(19), frFromInt(23)],
            [frFromInt(29), frFromInt(31), frFromInt(37)],
        ]
        let z = frFromInt(42)

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofWitnesses = [PointProjective]()

        for poly in polys {
            let c = try kzg.commit(poly)
            let p = try kzg.open(poly, at: z)
            commitments.append(c)
            values.append(p.evaluation)
            proofWitnesses.append(p.witness)
        }

        let batch = SamePointBatchClaim(
            commitments: commitments, point: z,
            values: values, proofs: proofWitnesses)

        let valid = try engine.batchVerifySamePoint(batch: batch, srsSecret: srsSecret)
        expect(valid, "Same-point batch (4 polys): valid")
    } catch {
        expect(false, "Same-point batch valid test threw: \(error)")
    }
}

private func testSamePointBatchPlonkScale(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let numPolys = 20
        let z = frFromInt(77)

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofWitnesses = [PointProjective]()

        for i in 0..<numPolys {
            let poly: [Fr] = [
                frFromInt(UInt64(i * 4 + 1)),
                frFromInt(UInt64(i * 4 + 2)),
                frFromInt(UInt64(i * 4 + 3)),
                frFromInt(UInt64(i * 4 + 4)),
            ]
            let c = try kzg.commit(poly)
            let p = try kzg.open(poly, at: z)
            commitments.append(c)
            values.append(p.evaluation)
            proofWitnesses.append(p.witness)
        }

        let batch = SamePointBatchClaim(
            commitments: commitments, point: z,
            values: values, proofs: proofWitnesses)

        let valid = try engine.batchVerifySamePoint(batch: batch, srsSecret: srsSecret)
        expect(valid, "Plonk-scale same-point batch (20 polys): valid")
    } catch {
        expect(false, "Plonk-scale test threw: \(error)")
    }
}

private func testSamePointBatchCorrupted(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
        ]
        let z = frFromInt(50)

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofWitnesses = [PointProjective]()

        for poly in polys {
            let c = try kzg.commit(poly)
            let p = try kzg.open(poly, at: z)
            commitments.append(c)
            values.append(p.evaluation)
            proofWitnesses.append(p.witness)
        }

        // Corrupt the first value
        values[0] = frFromInt(77777)

        let batch = SamePointBatchClaim(
            commitments: commitments, point: z,
            values: values, proofs: proofWitnesses)

        let valid = try engine.batchVerifySamePoint(batch: batch, srsSecret: srsSecret)
        expect(!valid, "Same-point batch with corrupted value: rejected")
    } catch {
        expect(false, "Same-point corrupted test threw: \(error)")
    }
}

// MARK: - Struct API Tests

private func testBatchVerifyClaimStruct(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(11), frFromInt(22), frFromInt(33)]
        let z = frFromInt(5)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        let claim = BatchVerifyClaim(
            commitment: commitment, point: z,
            value: proof.evaluation, proof: proof.witness)

        // Verify using single-claim path
        let singleValid = engine.verifySingle(claim: claim, srsSecret: srsSecret)
        expect(singleValid, "BatchVerifyClaim struct: single verify valid")

        // Also verify through batch path
        let batchValid = try engine.batchVerify(claims: [claim], srsSecret: srsSecret)
        expect(batchValid, "BatchVerifyClaim struct: batch verify valid")
    } catch {
        expect(false, "BatchVerifyClaim struct test threw: \(error)")
    }
}

private func testMultiOpenBatchClaimStruct(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(3), frFromInt(7), frFromInt(11)],
            [frFromInt(13), frFromInt(17), frFromInt(19)],
        ]
        let points: [Fr] = [frFromInt(5), frFromInt(8)]

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofs = [PointProjective]()

        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            commitments.append(c)
            values.append(p.evaluation)
            proofs.append(p.witness)
        }

        let batch = MultiOpenBatchClaim(
            commitments: commitments, points: points,
            values: values, proofs: proofs)

        expect(batch.isWellFormed, "MultiOpenBatchClaim well-formed")
        expectEqual(batch.count, 2, "MultiOpenBatchClaim count")

        let valid = try engine.batchVerifyMultiOpen(batch: batch, srsSecret: srsSecret)
        expect(valid, "MultiOpenBatchClaim: batch verify valid")
    } catch {
        expect(false, "MultiOpenBatchClaim test threw: \(error)")
    }
}

private func testSamePointBatchClaimStruct(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300)]
        let z = frFromInt(9)
        let c = try kzg.commit(poly)
        let p = try kzg.open(poly, at: z)

        let batch = SamePointBatchClaim(
            commitments: [c], point: z,
            values: [p.evaluation], proofs: [p.witness])

        expectEqual(batch.count, 1, "SamePointBatchClaim count")

        let valid = try engine.batchVerifySamePoint(batch: batch, srsSecret: srsSecret)
        expect(valid, "SamePointBatchClaim: single-entry batch valid")
    } catch {
        expect(false, "SamePointBatchClaim test threw: \(error)")
    }
}

// MARK: - Edge Case Tests

private func testEmptyClaims(engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let valid = try engine.batchVerify(claims: [], srsSecret: srsSecret)
        expect(valid, "Empty claims: vacuously true")

        let emptyBatch = MultiOpenBatchClaim(
            commitments: [], points: [], values: [], proofs: [])
        let batchValid = try engine.batchVerifyMultiOpen(batch: emptyBatch, srsSecret: srsSecret)
        expect(batchValid, "Empty MultiOpenBatchClaim: vacuously true")
    } catch {
        expect(false, "Empty claims test threw: \(error)")
    }
}

private func testMalformedMultiOpenBatch(engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        // commitments.count != points.count
        let batch = MultiOpenBatchClaim(
            commitments: [pointIdentity()],
            points: [frFromInt(1), frFromInt(2)],
            values: [frFromInt(3)],
            proofs: [pointIdentity()])

        expect(!batch.isWellFormed, "Malformed batch detected")

        let valid = try engine.batchVerifyMultiOpen(batch: batch, srsSecret: srsSecret)
        expect(!valid, "Malformed batch: rejected")
    } catch {
        expect(false, "Malformed batch test threw: \(error)")
    }
}

// MARK: - Detailed Verification Test

private func testDetailedVerification(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
            [frFromInt(7), frFromInt(8), frFromInt(9)],
        ]
        let points: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]

        var claims = [BatchVerifyClaim]()
        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            claims.append(BatchVerifyClaim(
                commitment: c, point: points[i],
                value: p.evaluation, proof: p.witness))
        }

        let result = try engine.batchVerifyDetailed(claims: claims, srsSecret: srsSecret)
        expect(result.isValid, "Detailed result: valid")
        expectEqual(result.claimCount, 3, "Detailed result: claim count")
        expect(result.elapsedSeconds > 0, "Detailed result: positive elapsed time")
        expect(result.summary.contains("3 claim"), "Detailed result: summary mentions count")
        expect(result.summary.contains("valid=true"), "Detailed result: summary says valid")
    } catch {
        expect(false, "Detailed verification test threw: \(error)")
    }
}

// MARK: - Accumulator Tests

private func testAccumulatorBasic(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let accumulator = engine.createAccumulator()
        expectEqual(accumulator.count, 0, "Accumulator: initially empty")

        // Add 3 claims
        let polys: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5)],
            [frFromInt(7), frFromInt(11)],
            [frFromInt(13), frFromInt(17), frFromInt(19), frFromInt(23)],
        ]
        let points: [Fr] = [frFromInt(4), frFromInt(6), frFromInt(8)]

        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            accumulator.addClaim(
                commitment: c, point: points[i],
                value: p.evaluation, proof: p.witness)
        }

        expectEqual(accumulator.count, 3, "Accumulator: 3 claims added")

        let valid = try engine.verifyAccumulator(accumulator, srsSecret: srsSecret)
        expect(valid, "Accumulator: batch verify valid")
    } catch {
        expect(false, "Accumulator basic test threw: \(error)")
    }
}

private func testAccumulatorClearAndReuse(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let accumulator = engine.createAccumulator()

        // First batch: valid
        let poly1: [Fr] = [frFromInt(1), frFromInt(2)]
        let z1 = frFromInt(3)
        let c1 = try kzg.commit(poly1)
        let p1 = try kzg.open(poly1, at: z1)
        accumulator.addClaim(BatchVerifyClaim(
            commitment: c1, point: z1,
            value: p1.evaluation, proof: p1.witness))

        let valid1 = try engine.verifyAccumulator(accumulator, srsSecret: srsSecret)
        expect(valid1, "Accumulator round 1: valid")

        // Clear and add a new claim
        accumulator.clear()
        expectEqual(accumulator.count, 0, "Accumulator: cleared")

        let poly2: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let z2 = frFromInt(7)
        let c2 = try kzg.commit(poly2)
        let p2 = try kzg.open(poly2, at: z2)
        accumulator.addClaim(BatchVerifyClaim(
            commitment: c2, point: z2,
            value: p2.evaluation, proof: p2.witness))

        let valid2 = try engine.verifyAccumulator(accumulator, srsSecret: srsSecret)
        expect(valid2, "Accumulator round 2 (after clear): valid")
    } catch {
        expect(false, "Accumulator clear/reuse test threw: \(error)")
    }
}

// MARK: - Cross-Polynomial Batch Tests

private func testCrossPolynomialBatch(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        // Two polynomials, each opened at one point
        let poly0: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15)]
        let poly1: [Fr] = [frFromInt(20), frFromInt(25), frFromInt(30), frFromInt(35)]

        let z0 = frFromInt(3)
        let z1 = frFromInt(9)

        let c0 = try kzg.commit(poly0)
        let p0 = try kzg.open(poly0, at: z0)

        let c1 = try kzg.commit(poly1)
        let p1 = try kzg.open(poly1, at: z1)

        let valid = try engine.batchVerifyCrossPolynomial(
            polyOpenings: [
                (commitment: c0, openings: [(point: z0, value: p0.evaluation, proof: p0.witness)]),
                (commitment: c1, openings: [(point: z1, value: p1.evaluation, proof: p1.witness)]),
            ],
            srsSecret: srsSecret)

        expect(valid, "Cross-polynomial batch (2 polys, 1 point each): valid")
    } catch {
        expect(false, "Cross-polynomial batch test threw: \(error)")
    }
}

private func testCrossPolynomialMultiplePointsPerPoly(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        // One polynomial opened at two different points
        let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let z0 = frFromInt(5)
        let z1 = frFromInt(11)

        let c = try kzg.commit(poly)
        let p0 = try kzg.open(poly, at: z0)
        let p1 = try kzg.open(poly, at: z1)

        let valid = try engine.batchVerifyCrossPolynomial(
            polyOpenings: [
                (commitment: c, openings: [
                    (point: z0, value: p0.evaluation, proof: p0.witness),
                    (point: z1, value: p1.evaluation, proof: p1.witness),
                ]),
            ],
            srsSecret: srsSecret)

        expect(valid, "Cross-polynomial (1 poly, 2 points): valid")

        // Now corrupt one evaluation
        let invalid = try engine.batchVerifyCrossPolynomial(
            polyOpenings: [
                (commitment: c, openings: [
                    (point: z0, value: p0.evaluation, proof: p0.witness),
                    (point: z1, value: frFromInt(99999), proof: p1.witness),
                ]),
            ],
            srsSecret: srsSecret)

        expect(!invalid, "Cross-polynomial with corrupted eval: rejected")
    } catch {
        expect(false, "Cross-polynomial multi-point test threw: \(error)")
    }
}

// MARK: - Transcript-Based Tests

private func testTranscriptBasedVerification(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(3), frFromInt(5), frFromInt(7)],
            [frFromInt(11), frFromInt(13), frFromInt(17)],
        ]
        let points: [Fr] = [frFromInt(4), frFromInt(6)]

        var claims = [BatchVerifyClaim]()
        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            claims.append(BatchVerifyClaim(
                commitment: c, point: points[i],
                value: p.evaluation, proof: p.witness))
        }

        let transcript = Transcript(label: "batch-verify-test", backend: .poseidon2)
        let valid = try engine.batchVerifyWithTranscript(
            claims: claims, transcript: transcript, srsSecret: srsSecret)
        expect(valid, "Transcript-based batch verify: valid")
    } catch {
        expect(false, "Transcript-based verification test threw: \(error)")
    }
}

private func testTranscriptConsistency(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        // Verify that two runs with the same transcript label produce the same result
        let poly: [Fr] = [frFromInt(42), frFromInt(43), frFromInt(44)]
        let z = frFromInt(10)
        let c = try kzg.commit(poly)
        let p = try kzg.open(poly, at: z)

        let claim = BatchVerifyClaim(
            commitment: c, point: z,
            value: p.evaluation, proof: p.witness)

        let t1 = Transcript(label: "consistency-test", backend: .poseidon2)
        let valid1 = try engine.batchVerifyWithTranscript(
            claims: [claim, claim], transcript: t1, srsSecret: srsSecret)

        let t2 = Transcript(label: "consistency-test", backend: .poseidon2)
        let valid2 = try engine.batchVerifyWithTranscript(
            claims: [claim, claim], transcript: t2, srsSecret: srsSecret)

        expectEqual(valid1, valid2, "Transcript consistency: same result for same inputs")
        expect(valid1, "Transcript consistency: both valid")
    } catch {
        expect(false, "Transcript consistency test threw: \(error)")
    }
}

// MARK: - Pairing Verification Test

private func testPairingVerificationSingle(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine,
                                            srs: [PointAffine], srsSecret: Fr) {
    do {
        let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7)]
        let z = frFromInt(11)
        let commitment = try kzg.commit(poly)
        let proof = try kzg.open(poly, at: z)

        let claim = BatchVerifyClaim(
            commitment: commitment, point: z,
            value: proof.evaluation, proof: proof.witness)

        // First verify with SRS secret (baseline)
        let secretValid = try engine.batchVerify(claims: [claim], srsSecret: srsSecret)
        expect(secretValid, "Pairing test: SRS-secret verification valid")

        // Then verify via pairing
        let g2Gen = bn254G2Generator()
        // Compute [s]_2 = s * G2
        let g2Proj = g2FromAffine(g2Gen)
        let sLimbs = frToInt(srsSecret)
        let g2Tau = g2ScalarMul(g2Proj, sLimbs)
        let g2TauAff = g2ToAffine(g2Tau)!

        let pairingValid = try engine.batchVerifyPairing(
            claims: [claim], g2Gen: g2Gen, g2Tau: g2TauAff)
        expect(pairingValid, "Pairing test: pairing-based verification valid")
    } catch {
        expect(false, "Pairing verification single test threw: \(error)")
    }
}

// MARK: - Batch Size Scaling Test

private func testBatchSizeScaling(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        // Test batch sizes: 1, 2, 3, 5, 8 to exercise both CPU and GPU paths
        let batchSizes = [1, 2, 3, 5, 8]

        for batchSize in batchSizes {
            let z = frFromInt(UInt64(batchSize * 7 + 3))

            var claims = [BatchVerifyClaim]()
            for j in 0..<batchSize {
                let poly: [Fr] = [
                    frFromInt(UInt64(j * 3 + 1)),
                    frFromInt(UInt64(j * 3 + 2)),
                    frFromInt(UInt64(j * 3 + 3)),
                ]
                let c = try kzg.commit(poly)
                let p = try kzg.open(poly, at: z)
                claims.append(BatchVerifyClaim(
                    commitment: c, point: z,
                    value: p.evaluation, proof: p.witness))
            }

            let valid = try engine.batchVerify(claims: claims, srsSecret: srsSecret)
            expect(valid, "Batch size \(batchSize): valid")
        }
    } catch {
        expect(false, "Batch size scaling test threw: \(error)")
    }
}

// MARK: - Array-Based API Tests

private func testArrayBasedBatchVerify(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(10), frFromInt(20), frFromInt(30)],
            [frFromInt(40), frFromInt(50), frFromInt(60)],
        ]
        let points: [Fr] = [frFromInt(7), frFromInt(13)]

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofs = [PointProjective]()

        for i in 0..<polys.count {
            let c = try kzg.commit(polys[i])
            let p = try kzg.open(polys[i], at: points[i])
            commitments.append(c)
            values.append(p.evaluation)
            proofs.append(p.witness)
        }

        let valid = try engine.batchVerifyArrays(
            commitments: commitments, points: points,
            values: values, proofs: proofs,
            srsSecret: srsSecret)
        expect(valid, "Array-based batch verify: valid")

        // Mismatched lengths should fail
        let invalid = try engine.batchVerifyArrays(
            commitments: commitments, points: [points[0]],
            values: values, proofs: proofs,
            srsSecret: srsSecret)
        expect(!invalid, "Array-based batch verify: mismatched lengths rejected")
    } catch {
        expect(false, "Array-based batch verify test threw: \(error)")
    }
}

private func testArrayBasedSamePointVerify(kzg: KZGEngine, engine: GPUKZGBatchVerifyEngine, srsSecret: Fr) {
    do {
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
            [frFromInt(7), frFromInt(8), frFromInt(9)],
        ]
        let z = frFromInt(15)

        var commitments = [PointProjective]()
        var values = [Fr]()
        var proofs = [PointProjective]()

        for poly in polys {
            let c = try kzg.commit(poly)
            let p = try kzg.open(poly, at: z)
            commitments.append(c)
            values.append(p.evaluation)
            proofs.append(p.witness)
        }

        let valid = try engine.batchVerifySamePointArrays(
            commitments: commitments, point: z,
            values: values, proofs: proofs,
            srsSecret: srsSecret)
        expect(valid, "Array-based same-point batch verify: valid")
    } catch {
        expect(false, "Array-based same-point verify test threw: \(error)")
    }
}
