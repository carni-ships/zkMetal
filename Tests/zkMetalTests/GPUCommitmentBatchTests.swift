import zkMetal
import Foundation

// MARK: - GPU Commitment Batch Engine Tests

public func runGPUCommitmentBatchTests() {
    // Common setup: test SRS
    let gen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)

    suite("GPU Commitment Batch Engine — Batch Opening")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Test 1: Single polynomial batch open (degenerate case)
        do {
            let poly: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
            let z = frFromInt(13)

            let proof = engine.batchOpen(polynomials: [poly], point: z)
            expectEqual(proof.commitments.count, 1, "Single poly: 1 commitment")
            expectEqual(proof.evaluations.count, 1, "Single poly: 1 evaluation")

            let valid = engine.verifyBatchOpening(proof, srsSecret: srsSecret)
            expect(valid, "Single poly batch open: valid proof accepted")
        }

        // Test 2: Multiple polynomials at one point
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
                [frFromInt(5), frFromInt(6), frFromInt(7)],
                [frFromInt(8), frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)],
            ]
            let z = frFromInt(42)

            let proof = engine.batchOpen(polynomials: polys, point: z)
            expectEqual(proof.commitments.count, 3, "3 polys: 3 commitments")
            expectEqual(proof.evaluations.count, 3, "3 polys: 3 evaluations")

            let valid = engine.verifyBatchOpening(proof, srsSecret: srsSecret)
            expect(valid, "Multi-poly batch open (3 polys): valid")
        }

        // Test 3: Batch open with corrupted evaluation should fail
        do {
            let polys: [[Fr]] = [
                [frFromInt(2), frFromInt(3), frFromInt(5)],
                [frFromInt(7), frFromInt(11), frFromInt(13)],
            ]
            let z = frFromInt(17)

            var proof = engine.batchOpen(polynomials: polys, point: z)
            // Corrupt the first evaluation
            let badEvals = [frFromInt(999), proof.evaluations[1]]
            proof = GPUBatchOpeningProof(
                commitments: proof.commitments, evaluations: badEvals,
                witness: proof.witness, point: proof.point, gamma: proof.gamma)

            let valid = engine.verifyBatchOpening(proof, srsSecret: srsSecret)
            expect(!valid, "Corrupted evaluation: rejected")
        }

        // Test 4: Batch open evaluations match individual evaluations
        do {
            let polys: [[Fr]] = [
                [frFromInt(10), frFromInt(20), frFromInt(30)],
                [frFromInt(40), frFromInt(50), frFromInt(60)],
            ]
            let z = frFromInt(7)

            let proof = engine.batchOpen(polynomials: polys, point: z)

            // Verify evaluations match Horner eval
            for i in 0..<polys.count {
                var expected = Fr.zero
                var zPow = Fr.one
                for coeff in polys[i] {
                    expected = frAdd(expected, frMul(coeff, zPow))
                    zPow = frMul(zPow, z)
                }
                let match = frToInt(proof.evaluations[i]) == frToInt(expected)
                expect(match, "Batch open eval[\(i)] matches Horner")
            }
        }

        // Test 5: Empty polynomial list
        do {
            let proof = engine.batchOpen(polynomials: [], point: frFromInt(1))
            expectEqual(proof.commitments.count, 0, "Empty batch: no commitments")
            expectEqual(proof.evaluations.count, 0, "Empty batch: no evaluations")
        }

        // Test 6: Larger batch (10 polynomials)
        do {
            var polys = [[Fr]]()
            for i in 0..<10 {
                let poly: [Fr] = [
                    frFromInt(UInt64(i * 3 + 1)),
                    frFromInt(UInt64(i * 3 + 2)),
                    frFromInt(UInt64(i * 3 + 3)),
                    frFromInt(UInt64(i * 3 + 4)),
                ]
                polys.append(poly)
            }
            let z = frFromInt(99)

            let proof = engine.batchOpen(polynomials: polys, point: z)
            let valid = engine.verifyBatchOpening(proof, srsSecret: srsSecret)
            expect(valid, "10-poly batch open: valid")
        }

    } catch {
        print("  [ERROR] Batch Opening tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Multi-Point Opening")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Test 7: Single point multi-point open (degenerate)
        do {
            let poly: [Fr] = [frFromInt(5), frFromInt(3), frFromInt(7)]
            let points: [Fr] = [frFromInt(2)]

            let proof = engine.multiPointOpen(polynomial: poly, points: points)
            expectEqual(proof.points.count, 1, "1-point multi-open: 1 point")
            expectEqual(proof.evaluations.count, 1, "1-point multi-open: 1 eval")

            let valid = engine.verifyMultiPointOpening(proof, srsSecret: srsSecret)
            expect(valid, "Single-point multi-open: valid")
        }

        // Test 8: Two-point multi-point open
        do {
            let poly: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let points: [Fr] = [frFromInt(5), frFromInt(10)]

            let proof = engine.multiPointOpen(polynomial: poly, points: points)
            expectEqual(proof.evaluations.count, 2, "2-point multi-open: 2 evals")

            let valid = engine.verifyMultiPointOpening(proof, srsSecret: srsSecret)
            expect(valid, "Two-point multi-open: valid")
        }

        // Test 9: Three-point multi-point open
        do {
            let poly: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5),
                              frFromInt(7), frFromInt(11)]
            let points: [Fr] = [frFromInt(1), frFromInt(3), frFromInt(7)]

            let proof = engine.multiPointOpen(polynomial: poly, points: points)

            let valid = engine.verifyMultiPointOpening(proof, srsSecret: srsSecret)
            expect(valid, "Three-point multi-open: valid")
        }

        // Test 10: Multi-point open with corrupted evaluation should fail
        do {
            let poly: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
            let points: [Fr] = [frFromInt(2), frFromInt(5)]

            var proof = engine.multiPointOpen(polynomial: poly, points: points)
            let badEvals = [frFromInt(12345), proof.evaluations[1]]
            proof = MultiPointOpeningProof(
                commitment: proof.commitment, points: proof.points,
                evaluations: badEvals, witness: proof.witness)

            let valid = engine.verifyMultiPointOpening(proof, srsSecret: srsSecret)
            expect(!valid, "Multi-point corrupted eval: rejected")
        }

    } catch {
        print("  [ERROR] Multi-Point Opening tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Random Linear Combination")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Test 11: Single commitment RLC
        do {
            let poly: [Fr] = [frFromInt(7), frFromInt(11), frFromInt(13)]
            let c = engine.commit(poly)

            let agg = engine.randomLinearCombination(commitments: [c])
            let valid = engine.verifyAggregation(agg)
            expect(valid, "Single commitment RLC: valid")

            // alpha^0 = 1, so aggregated == c
            let match = pointEqual(agg.aggregated, c)
            expect(match, "Single RLC: aggregated equals original")
        }

        // Test 12: Multiple commitments RLC
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3)],
                [frFromInt(4), frFromInt(5), frFromInt(6)],
                [frFromInt(7), frFromInt(8), frFromInt(9)],
            ]
            let commitments = polys.map { engine.commit($0) }

            let agg = engine.randomLinearCombination(commitments: commitments)
            let valid = engine.verifyAggregation(agg)
            expect(valid, "3-commitment RLC: aggregation verifies")
            expectEqual(agg.originals.count, 3, "3-commitment RLC: 3 originals stored")
        }

        // Test 13: Empty RLC
        do {
            let agg = engine.randomLinearCombination(commitments: [])
            expectEqual(agg.originals.count, 0, "Empty RLC: no originals")
        }

        // Test 14: RLC determinism (same inputs produce same result)
        do {
            let polys: [[Fr]] = [
                [frFromInt(10), frFromInt(20)],
                [frFromInt(30), frFromInt(40)],
            ]
            let commitments = polys.map { engine.commit($0) }

            let agg1 = engine.randomLinearCombination(commitments: commitments)
            let agg2 = engine.randomLinearCombination(commitments: commitments)
            let match = pointEqual(agg1.aggregated, agg2.aggregated)
            expect(match, "RLC deterministic: same inputs => same output")
        }

    } catch {
        print("  [ERROR] RLC tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Proof Aggregation")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)
        let kzg = try KZGEngine(srs: srs)

        // Test 15: Aggregate 2 proofs at same point
        do {
            let polys: [[Fr]] = [
                [frFromInt(2), frFromInt(3), frFromInt(5)],
                [frFromInt(7), frFromInt(11), frFromInt(13)],
            ]
            let z = frFromInt(42)

            var commitments = [PointProjective]()
            var evaluations = [Fr]()
            var witnesses = [PointProjective]()
            var pts = [Fr]()

            for poly in polys {
                let c = try kzg.commit(poly)
                let p = try kzg.open(poly, at: z)
                commitments.append(c)
                evaluations.append(p.evaluation)
                witnesses.append(p.witness)
                pts.append(z)
            }

            let (aggW, alpha) = engine.aggregateProofs(
                commitments: commitments, points: pts,
                evaluations: evaluations, witnesses: witnesses)

            let valid = engine.verifyAggregatedProof(
                commitments: commitments, points: pts,
                evaluations: evaluations, aggregatedWitness: aggW,
                challenge: alpha, srsSecret: srsSecret)
            expect(valid, "2-proof aggregation at same point: valid")
        }

        // Test 16: Aggregate 5 proofs at same point
        do {
            let z = frFromInt(77)
            var commitments = [PointProjective]()
            var evaluations = [Fr]()
            var witnesses = [PointProjective]()
            var pts = [Fr]()

            for i in 0..<5 {
                let poly: [Fr] = [
                    frFromInt(UInt64(i * 3 + 1)),
                    frFromInt(UInt64(i * 3 + 2)),
                    frFromInt(UInt64(i * 3 + 3)),
                ]
                let c = try kzg.commit(poly)
                let p = try kzg.open(poly, at: z)
                commitments.append(c)
                evaluations.append(p.evaluation)
                witnesses.append(p.witness)
                pts.append(z)
            }

            let (aggW, alpha) = engine.aggregateProofs(
                commitments: commitments, points: pts,
                evaluations: evaluations, witnesses: witnesses)

            let valid = engine.verifyAggregatedProof(
                commitments: commitments, points: pts,
                evaluations: evaluations, aggregatedWitness: aggW,
                challenge: alpha, srsSecret: srsSecret)
            expect(valid, "5-proof aggregation: valid")
        }

    } catch {
        print("  [ERROR] Proof Aggregation tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Cross-Commitment Consistency")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Test 17: Linear relation check — same polynomial
        do {
            let poly: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15)]
            let c = engine.commit(poly)

            // C with coefficient 1 should equal C with coefficient 1
            let valid = engine.checkLinearRelation(
                commitmentsA: [c], coefficientsA: [Fr.one],
                commitmentsB: [c], coefficientsB: [Fr.one])
            expect(valid, "Same commitment linear relation: valid")
        }

        // Test 18: Linear relation check — sum of commitments
        do {
            let poly1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let poly2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
            let polySum: [Fr] = [frFromInt(5), frFromInt(7), frFromInt(9)]

            let c1 = engine.commit(poly1)
            let c2 = engine.commit(poly2)
            let cSum = engine.commit(polySum)

            // C1 * 1 + C2 * 1 should equal CSum * 1
            let valid = engine.checkLinearRelation(
                commitmentsA: [c1, c2], coefficientsA: [Fr.one, Fr.one],
                commitmentsB: [cSum], coefficientsB: [Fr.one])
            expect(valid, "Sum of commitments linear relation: valid")
        }

        // Test 19: Linear relation check — scalar multiple
        do {
            let poly: [Fr] = [frFromInt(3), frFromInt(7)]
            let scalar = frFromInt(5)
            let scaledPoly: [Fr] = [frFromInt(15), frFromInt(35)]

            let c = engine.commit(poly)
            let cScaled = engine.commit(scaledPoly)

            // 5 * C should equal CScaled
            let valid = engine.checkLinearRelation(
                commitmentsA: [c], coefficientsA: [scalar],
                commitmentsB: [cScaled], coefficientsB: [Fr.one])
            expect(valid, "Scalar multiple linear relation: valid")
        }

        // Test 20: Linear relation check — should fail for wrong relation
        do {
            let poly1: [Fr] = [frFromInt(1), frFromInt(2)]
            let poly2: [Fr] = [frFromInt(3), frFromInt(4)]

            let c1 = engine.commit(poly1)
            let c2 = engine.commit(poly2)

            // C1 * 1 should NOT equal C2 * 1
            let valid = engine.checkLinearRelation(
                commitmentsA: [c1], coefficientsA: [Fr.one],
                commitmentsB: [c2], coefficientsB: [Fr.one])
            expect(!valid, "Wrong linear relation: rejected")
        }

    } catch {
        print("  [ERROR] Cross-Commitment tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Pedersen Batching")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Generate generators for Pedersen commitments
        let gProj = pointFromAffine(PointAffine(x: fpFromInt(1), y: fpFromInt(2)))
        var projPoints = [PointProjective]()
        var acc = gProj
        for _ in 0..<16 {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, gProj))
        }
        let generators = batchToAffine(projPoints)
        let blindingGen = generators[generators.count - 1]
        let pedGens = Array(generators.prefix(8))

        // Test 21: Pedersen batch commit — single vector
        do {
            let vec: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let r = frFromInt(42)

            let result = engine.batchPedersenCommit(
                vectors: [vec], randomness: [r],
                generators: pedGens, blindingGenerator: blindingGen)

            expectEqual(result.commitments.count, 1, "Pedersen batch: 1 commitment")
            let valid = engine.verifyPedersenBatch(result)
            expect(valid, "Pedersen single batch: aggregation verifies")
        }

        // Test 22: Pedersen batch commit — multiple vectors
        do {
            let vecs: [[Fr]] = [
                [frFromInt(1), frFromInt(2), frFromInt(3)],
                [frFromInt(4), frFromInt(5), frFromInt(6)],
                [frFromInt(7), frFromInt(8), frFromInt(9)],
            ]
            let rs: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]

            let result = engine.batchPedersenCommit(
                vectors: vecs, randomness: rs,
                generators: pedGens, blindingGenerator: blindingGen)

            expectEqual(result.commitments.count, 3, "Pedersen batch: 3 commitments")
            let valid = engine.verifyPedersenBatch(result)
            expect(valid, "Pedersen 3-vector batch: aggregation verifies")
        }

        // Test 23: Pedersen homomorphism check
        do {
            let vecA: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
            let vecB: [Fr] = [frFromInt(5), frFromInt(15), frFromInt(25)]
            let vecSum: [Fr] = [frFromInt(15), frFromInt(35), frFromInt(55)]
            let rA = frFromInt(7)
            let rB = frFromInt(11)
            let rSum = frAdd(rA, rB)

            let resultA = engine.batchPedersenCommit(
                vectors: [vecA], randomness: [rA],
                generators: pedGens, blindingGenerator: blindingGen)
            let resultB = engine.batchPedersenCommit(
                vectors: [vecB], randomness: [rB],
                generators: pedGens, blindingGenerator: blindingGen)
            let resultSum = engine.batchPedersenCommit(
                vectors: [vecSum], randomness: [rSum],
                generators: pedGens, blindingGenerator: blindingGen)

            let valid = engine.checkPedersenHomomorphism(
                commitmentA: resultA.commitments[0],
                commitmentB: resultB.commitments[0],
                commitmentSum: resultSum.commitments[0])
            expect(valid, "Pedersen homomorphism: C(a) + C(b) == C(a+b)")
        }

        // Test 24: Pedersen homomorphism should fail for wrong sum
        do {
            let vecA: [Fr] = [frFromInt(10), frFromInt(20)]
            let vecWrong: [Fr] = [frFromInt(999), frFromInt(999)]
            let rA = frFromInt(1)
            let rB = frFromInt(2)
            let rWrong = frFromInt(3)

            let resultA = engine.batchPedersenCommit(
                vectors: [vecA], randomness: [rA],
                generators: pedGens, blindingGenerator: blindingGen)
            let resultB = engine.batchPedersenCommit(
                vectors: [vecA], randomness: [rB],
                generators: pedGens, blindingGenerator: blindingGen)
            let resultWrong = engine.batchPedersenCommit(
                vectors: [vecWrong], randomness: [rWrong],
                generators: pedGens, blindingGenerator: blindingGen)

            let valid = engine.checkPedersenHomomorphism(
                commitmentA: resultA.commitments[0],
                commitmentB: resultB.commitments[0],
                commitmentSum: resultWrong.commitments[0])
            expect(!valid, "Pedersen homomorphism: wrong sum rejected")
        }

    } catch {
        print("  [ERROR] Pedersen Batching tests threw: \(error)")
    }

    suite("GPU Commitment Batch Engine — Edge Cases")

    do {
        let engine = GPUCommitmentBatchEngine(srs: srs)

        // Test 25: Constant polynomial (degree 0)
        do {
            let poly: [Fr] = [frFromInt(42)]
            let z = frFromInt(7)

            let proof = engine.batchOpen(polynomials: [poly], point: z)
            let evalMatch = frToInt(proof.evaluations[0]) == frToInt(frFromInt(42))
            expect(evalMatch, "Constant poly: eval is constant")
        }

        // Test 26: Linear polynomial batch open
        do {
            let polys: [[Fr]] = [
                [frFromInt(1), frFromInt(2)],
                [frFromInt(3), frFromInt(4)],
            ]
            let z = frFromInt(5)

            let proof = engine.batchOpen(polynomials: polys, point: z)
            let valid = engine.verifyBatchOpening(proof, srsSecret: srsSecret)
            expect(valid, "Linear polynomials batch open: valid")
        }

        // Test 27: Commitments to zero polynomial
        do {
            let zeroPoly: [Fr] = [Fr.zero, Fr.zero, Fr.zero]
            let c = engine.commit(zeroPoly)
            let isId = pointIsIdentity(c)
            expect(isId, "Zero polynomial commitment is identity")
        }

        // Test 28: Cross-commitment with empty arrays
        do {
            let valid = engine.checkLinearRelation(
                commitmentsA: [], coefficientsA: [],
                commitmentsB: [], coefficientsB: [])
            expect(valid, "Empty linear relation: vacuously true")
        }

        // Test 29: Batch commit consistency with single commit
        do {
            let polys: [[Fr]] = [
                [frFromInt(11), frFromInt(22), frFromInt(33)],
                [frFromInt(44), frFromInt(55), frFromInt(66)],
            ]

            let batchResults = engine.batchCommit(polys)
            let individual0 = engine.commit(polys[0])
            let individual1 = engine.commit(polys[1])

            expect(pointEqual(batchResults[0], individual0),
                   "Batch commit[0] matches individual")
            expect(pointEqual(batchResults[1], individual1),
                   "Batch commit[1] matches individual")
        }

        // Test 30: Multi-point open evaluation correctness
        do {
            let poly: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(1)]  // 1 + x + x^2
            let z = frFromInt(3)
            // p(3) = 1 + 3 + 9 = 13
            let proof = engine.multiPointOpen(polynomial: poly, points: [z])
            let expected = frFromInt(13)
            let match = frToInt(proof.evaluations[0]) == frToInt(expected)
            expect(match, "Multi-point eval: 1 + x + x^2 at 3 = 13")
        }

    } catch {
        print("  [ERROR] Edge Case tests threw: \(error)")
    }
}
