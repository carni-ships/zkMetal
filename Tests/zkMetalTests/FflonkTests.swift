import zkMetal
import Foundation

public func runFflonkTests() {
    // Common setup: test SRS (large enough for combined polynomials)
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let fflonkSRS = FflonkSRS.generateTest(secret: secret, size: 512, generator: gen)

    suite("Fflonk Polynomial Commitment")
    do {
        let engine = try FflonkEngine(srs: fflonkSRS)

        // Test 1: Single polynomial commit/open/verify
        do {
            let poly: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(5)]
            let z = frFromInt(13)

            let commitment = try engine.commit([poly])
            let proof = try engine.open([poly], at: z)

            expect(proof.evaluations.count == 1, "Single poly: 1 evaluation")
            expect(proof.batchSize == 1, "Single poly: batch size 1")

            // Direct evaluation check: p(z^1) = p(z)
            let directEval = frAdd(frFromInt(3),
                              frAdd(frMul(frFromInt(7), z),
                                frAdd(frMul(frFromInt(11), frMul(z, z)),
                                      frMul(frFromInt(5), frMul(z, frMul(z, z))))))
            expect(frToInt(proof.evaluations[0]) == frToInt(directEval),
                   "Single poly: evaluation matches direct computation")

            let valid = engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
            expect(valid, "Single poly: verify passes")
        }

        // Test 2: 2-polynomial batched commitment
        do {
            let p0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]  // 1 + 2x + 3x^2
            let p1: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]  // 4 + 5x + 6x^2
            let z = frFromInt(7)

            let commitment = try engine.commit([p0, p1])
            expect(commitment.batchSize == 2, "2-poly: batch size 2")

            let proof = try engine.open([p0, p1], at: z)
            expect(proof.evaluations.count == 2, "2-poly: 2 evaluations")
            expect(proof.batchSize == 2, "2-poly: batch size 2 in proof")

            // Check evaluations: p_i(z^2)
            let z2 = frMul(z, z)  // z^2 = 49
            let y0 = frAdd(frFromInt(1), frAdd(frMul(frFromInt(2), z2), frMul(frFromInt(3), frMul(z2, z2))))
            let y1 = frAdd(frFromInt(4), frAdd(frMul(frFromInt(5), z2), frMul(frFromInt(6), frMul(z2, z2))))

            expect(frToInt(proof.evaluations[0]) == frToInt(y0),
                   "2-poly: p0(z^2) correct")
            expect(frToInt(proof.evaluations[1]) == frToInt(y1),
                   "2-poly: p1(z^2) correct")

            let valid = engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
            expect(valid, "2-poly batched: verify passes")
        }

        // Test 3: 4-polynomial batched commitment
        do {
            let p0: [Fr] = [frFromInt(1), frFromInt(2)]
            let p1: [Fr] = [frFromInt(3), frFromInt(4)]
            let p2: [Fr] = [frFromInt(5), frFromInt(6)]
            let p3: [Fr] = [frFromInt(7), frFromInt(8)]
            let z = frFromInt(11)

            let commitment = try engine.commit([p0, p1, p2, p3])
            expect(commitment.batchSize == 4, "4-poly: batch size 4")

            let proof = try engine.open([p0, p1, p2, p3], at: z)
            expect(proof.evaluations.count == 4, "4-poly: 4 evaluations")

            // Check evaluations: p_i(z^4)
            let z4 = frPow(z, 4)  // z^4 = 11^4 = 14641
            for i in 0..<4 {
                let polys = [[frFromInt(1), frFromInt(2)],
                             [frFromInt(3), frFromInt(4)],
                             [frFromInt(5), frFromInt(6)],
                             [frFromInt(7), frFromInt(8)]]
                let expected = frAdd(polys[i][0], frMul(polys[i][1], z4))
                expect(frToInt(proof.evaluations[i]) == frToInt(expected),
                       "4-poly: p\(i)(z^4) correct")
            }

            let valid = engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
            expect(valid, "4-poly batched: verify passes")
        }

        // Test 4: Correctness — opened values match direct evaluation
        do {
            // Random-ish polynomials of varying degrees
            let p0: [Fr] = [frFromInt(42), frFromInt(17), frFromInt(99), frFromInt(3), frFromInt(8)]
            let p1: [Fr] = [frFromInt(100), frFromInt(200)]
            let z = frFromInt(5)

            let proof = try engine.open([p0, p1], at: z)
            let k = 2
            let zk = frPow(z, UInt64(k))  // z^2 = 25

            // Direct Horner evaluation of p0 at z^2
            var directP0 = Fr.zero
            for j in stride(from: p0.count - 1, through: 0, by: -1) {
                directP0 = frAdd(p0[j], frMul(directP0, zk))
            }
            expect(frToInt(proof.evaluations[0]) == frToInt(directP0),
                   "Correctness: p0 evaluation matches Horner")

            // Direct Horner evaluation of p1 at z^2
            var directP1 = Fr.zero
            for j in stride(from: p1.count - 1, through: 0, by: -1) {
                directP1 = frAdd(p1[j], frMul(directP1, zk))
            }
            expect(frToInt(proof.evaluations[1]) == frToInt(directP1),
                   "Correctness: p1 evaluation matches Horner")
        }

        // Test 5: Soundness — tampered evaluation rejected
        do {
            let p0: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
            let p1: [Fr] = [frFromInt(40), frFromInt(50), frFromInt(60)]
            let z = frFromInt(9)

            let commitment = try engine.commit([p0, p1])
            let proof = try engine.open([p0, p1], at: z)

            // Tamper: change one evaluation
            let tamperedProof = FflonkOpeningProof(
                witness: proof.witness,
                evaluations: [frFromInt(999), proof.evaluations[1]],
                point: proof.point,
                batchSize: proof.batchSize
            )
            let valid = engine.verify(commitment: commitment, proof: tamperedProof, srsSecret: srsSecret)
            expect(!valid, "Soundness: tampered evaluation rejected")

            // Also tamper the witness
            let g1 = pointFromAffine(fflonkSRS.points[0])
            let tamperedWitness = pointAdd(proof.witness, g1)
            let tamperedProof2 = FflonkOpeningProof(
                witness: tamperedWitness,
                evaluations: proof.evaluations,
                point: proof.point,
                batchSize: proof.batchSize
            )
            let valid2 = engine.verify(commitment: commitment, proof: tamperedProof2, srsSecret: srsSecret)
            expect(!valid2, "Soundness: tampered witness rejected")
        }

        // Test 6: Combined polynomial structure
        do {
            // Verify that buildCombinedPoly correctly interleaves
            let p0: [Fr] = [frFromInt(1), frFromInt(2)]  // 1 + 2X
            let p1: [Fr] = [frFromInt(3), frFromInt(4)]  // 3 + 4X
            // P(X) = p0(X^2) + X * p1(X^2) = (1 + 2X^2) + X*(3 + 4X^2)
            //       = 1 + 3X + 2X^2 + 4X^3
            let combined = FflonkEngine.buildCombinedPoly([p0, p1], batchSize: 2)
            expect(combined.count == 4, "Combined poly degree: 4 coefficients")
            expect(frToInt(combined[0]) == frToInt(frFromInt(1)), "Combined[0] = 1")
            expect(frToInt(combined[1]) == frToInt(frFromInt(3)), "Combined[1] = 3")
            expect(frToInt(combined[2]) == frToInt(frFromInt(2)), "Combined[2] = 2")
            expect(frToInt(combined[3]) == frToInt(frFromInt(4)), "Combined[3] = 4")
        }

        // Test 7: Performance comparison vs naive multi-open
        do {
            let numPolys = 4
            let polyDeg = 32
            var polys = [[Fr]]()
            for i in 0..<numPolys {
                polys.append((0..<polyDeg).map { j in frFromInt(UInt64(i * polyDeg + j + 1)) })
            }
            let z = frFromInt(42)

            // Fflonk: single combined opening
            let t0 = CFAbsoluteTimeGetCurrent()
            let commitment = try engine.commit(polys)
            let proof = try engine.open(polys, at: z)
            let valid = engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
            let fflonkTime = CFAbsoluteTimeGetCurrent() - t0
            expect(valid, "Performance: fflonk verify passes")

            // Naive: k separate KZG openings
            let kzg = engine.kzg
            let t1 = CFAbsoluteTimeGetCurrent()
            var naiveValid = true
            for poly in polys {
                let kzgProof = try kzg.open(poly, at: z)
                let kzgCommit = try kzg.commit(poly)
                // Verify using SRS secret
                let g1 = pointFromAffine(fflonkSRS.points[0])
                let yG = cPointScalarMul(g1, kzgProof.evaluation)
                let sMz = frSub(srsSecret, z)
                let szProof = cPointScalarMul(kzgProof.witness, sMz)
                let expected = pointAdd(yG, szProof)
                let cAff = batchToAffine([kzgCommit])
                let eAff = batchToAffine([expected])
                if fpToInt(cAff[0].x) != fpToInt(eAff[0].x) ||
                   fpToInt(cAff[0].y) != fpToInt(eAff[0].y) {
                    naiveValid = false
                }
            }
            let naiveTime = CFAbsoluteTimeGetCurrent() - t1
            expect(naiveValid, "Performance: naive multi-open verify passes")

            print("    fflonk \(numPolys) polys (deg \(polyDeg)): \(String(format: "%.3f", fflonkTime * 1000))ms")
            print("    naive  \(numPolys) opens:                  \(String(format: "%.3f", naiveTime * 1000))ms")
            // Fflonk produces 1 commitment + 1 witness vs 4 commitments + 4 witnesses
            print("    fflonk proof: 1 commitment + 1 witness")
            print("    naive  proof: \(numPolys) commitments + \(numPolys) witnesses")
        }

        // Test 8: 3-polynomial (auto-padded to 4)
        do {
            let p0: [Fr] = [frFromInt(2), frFromInt(3)]
            let p1: [Fr] = [frFromInt(5), frFromInt(7)]
            let p2: [Fr] = [frFromInt(11), frFromInt(13)]
            let z = frFromInt(17)

            let commitment = try engine.commit([p0, p1, p2])
            expect(commitment.batchSize == 4, "3-poly auto-padded to batch size 4")

            let proof = try engine.open([p0, p1, p2], at: z)
            expect(proof.evaluations.count == 4, "3-poly padded: 4 evaluations")

            // The 4th evaluation should be p3(z^4) = 0 (zero polynomial)
            expect(frToInt(proof.evaluations[3]) == frToInt(Fr.zero),
                   "Padded zero polynomial evaluates to 0")

            let valid = engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
            expect(valid, "3-poly padded: verify passes")
        }

    } catch { expect(false, "Fflonk error: \(error)") }
}
