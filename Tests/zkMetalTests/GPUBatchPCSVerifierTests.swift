import zkMetal

public func runGPUBatchPCSVerifierTests() {
    // Common setup: test SRS
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)

    suite("GPU Batch PCS Verifier")

    do {
        let kzg = try KZGEngine(srs: srs)
        let verifier = try GPUBatchPCSVerifier()

        // Test 1: Single opening (degenerate case)
        do {
            let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
            let z = frFromInt(13)
            let commitment = try kzg.commit(poly)
            let proof = try kzg.open(poly, at: z)

            let valid = try verifier.batchVerifyKZG(
                commitments: [commitment],
                points: [z],
                values: [proof.evaluation],
                proofs: [proof.witness],
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "Single opening: valid proof accepted")
        }

        // Test 2: Single opening with wrong value should fail
        do {
            let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7)]
            let z = frFromInt(17)
            let commitment = try kzg.commit(poly)
            let proof = try kzg.open(poly, at: z)

            let invalid = try verifier.batchVerifyKZG(
                commitments: [commitment],
                points: [z],
                values: [frFromInt(999)],  // wrong value
                proofs: [proof.witness],
                srs: srs,
                srsSecret: srsSecret
            )
            expect(!invalid, "Single opening: wrong value rejected")
        }

        // Test 3: Multiple openings at different points
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
                [frFromInt(5), frFromInt(6), frFromInt(7)],
                [frFromInt(8), frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)],
            ]
            let points: [Fr] = [frFromInt(7), frFromInt(13), frFromInt(42)]

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

            let valid = try verifier.batchVerifyKZG(
                commitments: commitments,
                points: points,
                values: values,
                proofs: proofs,
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "Multi-point batch (3 polys): valid")
        }

        // Test 4: Same-point batch verification (Plonk-like)
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

            let valid = try verifier.batchVerifyKZGSamePoint(
                commitments: commitments,
                point: z,
                values: values,
                proofs: proofWitnesses,
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "Same-point batch (4 polys): valid")
        }

        // Test 5: Batch with one corrupted proof should fail
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3)],
                [frFromInt(4), frFromInt(5), frFromInt(6)],
                [frFromInt(7), frFromInt(8), frFromInt(9)],
            ]
            let points: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]

            var commitments = [PointProjective]()
            var values = [Fr]()
            var proofWitnesses = [PointProjective]()

            for i in 0..<polys.count {
                let c = try kzg.commit(polys[i])
                let p = try kzg.open(polys[i], at: points[i])
                commitments.append(c)
                values.append(p.evaluation)
                proofWitnesses.append(p.witness)
            }

            // Corrupt the second value
            values[1] = frFromInt(12345)

            let invalid = try verifier.batchVerifyKZG(
                commitments: commitments,
                points: points,
                values: values,
                proofs: proofWitnesses,
                srs: srs,
                srsSecret: srsSecret
            )
            expect(!invalid, "Batch with corrupted value: rejected")
        }

        // Test 6: Larger batch (Plonk-scale, 20 polynomials)
        do {
            let numPolys = 20
            let z = frFromInt(77)

            var commitments = [PointProjective]()
            var values = [Fr]()
            var proofWitnesses = [PointProjective]()

            for i in 0..<numPolys {
                // Each polynomial: [i+1, i+2, i+3, i+4]
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

            let valid = try verifier.batchVerifyKZGSamePoint(
                commitments: commitments,
                point: z,
                values: values,
                proofs: proofWitnesses,
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "Plonk-scale batch (20 polys same point): valid")
        }

        // Test 7: KZGOpeningClaim struct API
        do {
            let poly: [Fr] = [frFromInt(11), frFromInt(22), frFromInt(33)]
            let z = frFromInt(5)
            let commitment = try kzg.commit(poly)
            let proof = try kzg.open(poly, at: z)

            let claim = KZGOpeningClaim(
                commitment: commitment,
                point: z,
                value: proof.evaluation,
                proof: proof.witness
            )

            let valid = try verifier.batchVerifyKZG(
                claims: [claim],
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "KZGOpeningClaim API: valid")
        }

        // Test 8: Empty claims should return true
        do {
            let valid = try verifier.batchVerifyKZG(
                claims: [],
                srs: srs,
                srsSecret: srsSecret
            )
            expect(valid, "Empty claims: vacuously true")
        }

        // Test 9: Mismatched array lengths should return false
        do {
            let poly: [Fr] = [frFromInt(1), frFromInt(2)]
            let z = frFromInt(3)
            let commitment = try kzg.commit(poly)
            let proof = try kzg.open(poly, at: z)

            let valid = try verifier.batchVerifyKZG(
                commitments: [commitment, commitment],
                points: [z],  // mismatched length
                values: [proof.evaluation],
                proofs: [proof.witness],
                srs: srs,
                srsSecret: srsSecret
            )
            expect(!valid, "Mismatched array lengths: rejected")
        }

    } catch {
        print("  [ERROR] GPUBatchPCSVerifier tests threw: \(error)")
    }
}
