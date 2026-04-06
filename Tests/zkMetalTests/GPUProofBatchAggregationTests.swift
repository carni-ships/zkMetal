import zkMetal

// MARK: - GPU Proof Batch Aggregation Tests

public func runGPUProofBatchAggregationTests() {
    suite("GPU Proof Batch Aggregation Engine")

    let engine = GPUProofBatchAggregationEngine()
    let g1Aff = bn254G1Generator()
    let g1 = pointFromAffine(g1Aff)

    // Helper: create a test curve point from a scalar
    func makeTestPoint(_ scalar: Fr) -> PointProjective {
        cPointScalarMul(g1, scalar)
    }

    // Helper: check two projective points are equal
    func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    // --- KZG Batch Aggregation ---

    // Test 1: Aggregate 2 KZG proofs
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let c1 = makeTestPoint(frFromInt(7))
        let w0 = makeTestPoint(frFromInt(5))
        let w1 = makeTestPoint(frFromInt(11))
        let v0 = frFromInt(42)
        let v1 = frFromInt(99)

        let agg = engine.aggregateKZGBatch(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [v0, v1]
        )

        expectEqual(agg.count, 2, "KZG batch 2: count == 2")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "KZG batch 2: commitment non-identity")
        expect(!pointIsIdentity(agg.aggregatedWitness),
               "KZG batch 2: witness non-identity")
        expect(!frEqual(agg.aggregatedEvaluation, Fr.zero),
               "KZG batch 2: evaluation non-zero")
    }

    // Test 2: Aggregate 4 KZG proofs
    do {
        var commitments = [PointProjective]()
        var witnesses = [PointProjective]()
        var evaluations = [Fr]()

        for i in 1...4 {
            commitments.append(makeTestPoint(frFromInt(UInt64(i * 3))))
            witnesses.append(makeTestPoint(frFromInt(UInt64(i * 5))))
            evaluations.append(frFromInt(UInt64(i * 10)))
        }

        let agg = engine.aggregateKZGBatch(
            commitments: commitments,
            witnesses: witnesses,
            evaluations: evaluations
        )

        expectEqual(agg.count, 4, "KZG batch 4: count == 4")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "KZG batch 4: commitment non-identity")
        expect(!pointIsIdentity(agg.aggregatedWitness),
               "KZG batch 4: witness non-identity")
    }

    // Test 3: Single proof aggregation preserves original
    do {
        let c0 = makeTestPoint(frFromInt(42))
        let w0 = makeTestPoint(frFromInt(7))
        let v0 = frFromInt(100)

        let agg = engine.aggregateKZGBatch(
            commitments: [c0],
            witnesses: [w0],
            evaluations: [v0]
        )

        expectEqual(agg.count, 1, "KZG single: count == 1")
        // r^0 = 1, so aggregated == original
        expect(pointsEqual(agg.aggregatedCommitment, c0),
               "KZG single: commitment preserved")
        expect(pointsEqual(agg.aggregatedWitness, w0),
               "KZG single: witness preserved")
        let evalMatch = frEqual(agg.aggregatedEvaluation, v0)
        expect(evalMatch, "KZG single: evaluation preserved")
    }

    // Test 4: Verify aggregated KZG batch with SRS secret
    do {
        let secret = frFromInt(17)
        let sG1 = makeTestPoint(secret)
        let z = frFromInt(5)

        // Construct valid KZG proofs: p(x) = a0 + a1*x, C = [a0]*G + [a1]*[s]*G
        let a0_0 = frFromInt(20)
        let a1_0 = frFromInt(3)
        let pz0 = frAdd(a0_0, frMul(a1_0, z))  // 20 + 3*5 = 35
        let c0 = pointAdd(cPointScalarMul(g1, a0_0), cPointScalarMul(sG1, a1_0))
        let w0 = cPointScalarMul(g1, a1_0)

        let a0_1 = frFromInt(10)
        let a1_1 = frFromInt(7)
        let pz1 = frAdd(a0_1, frMul(a1_1, z))  // 10 + 7*5 = 45
        let c1 = pointAdd(cPointScalarMul(g1, a0_1), cPointScalarMul(sG1, a1_1))
        let w1 = cPointScalarMul(g1, a1_1)

        let agg = engine.aggregateKZGBatch(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [pz0, pz1]
        )

        let valid = engine.verifyKZGBatch(
            proof: agg,
            point: z,
            srsG1: g1,
            srsSecret: secret
        )
        expect(valid, "KZG batch verify: valid proofs pass")
    }

    // Test 5: Tampered KZG proof rejected
    do {
        let secret = frFromInt(17)
        let sG1 = makeTestPoint(secret)
        let z = frFromInt(5)

        let a0 = frFromInt(20)
        let a1 = frFromInt(3)
        let pz = frAdd(a0, frMul(a1, z))
        let c0 = pointAdd(cPointScalarMul(g1, a0), cPointScalarMul(sG1, a1))
        let w0 = cPointScalarMul(g1, a1)

        let a0b = frFromInt(10)
        let a1b = frFromInt(7)
        let c1 = pointAdd(cPointScalarMul(g1, a0b), cPointScalarMul(sG1, a1b))
        let w1 = cPointScalarMul(g1, a1b)

        // Tamper: use wrong evaluation
        let wrongEval = frFromInt(999)

        let agg = engine.aggregateKZGBatch(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [pz, wrongEval]
        )

        let valid = engine.verifyKZGBatch(
            proof: agg,
            point: z,
            srsG1: g1,
            srsSecret: secret
        )
        expect(!valid, "KZG batch verify: tampered proof rejected")
    }

    // Test 6: Deterministic transcript — same inputs produce same challenge
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let w0 = makeTestPoint(frFromInt(5))
        let v0 = frFromInt(42)

        let agg1 = engine.aggregateKZGBatch(
            commitments: [c0], witnesses: [w0], evaluations: [v0]
        )
        let agg2 = engine.aggregateKZGBatch(
            commitments: [c0], witnesses: [w0], evaluations: [v0]
        )

        expect(frEqual(agg1.challenge, agg2.challenge),
               "KZG deterministic transcript: same challenge")
        expect(frEqual(agg1.aggregatedEvaluation, agg2.aggregatedEvaluation),
               "KZG deterministic transcript: same evaluation")
    }

    // Test 7: Different inputs produce different challenges
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let w0 = makeTestPoint(frFromInt(5))
        let c1 = makeTestPoint(frFromInt(9))
        let w1 = makeTestPoint(frFromInt(13))

        let agg1 = engine.aggregateKZGBatch(
            commitments: [c0], witnesses: [w0], evaluations: [frFromInt(42)]
        )
        let agg2 = engine.aggregateKZGBatch(
            commitments: [c1], witnesses: [w1], evaluations: [frFromInt(42)]
        )

        expect(!frEqual(agg1.challenge, agg2.challenge),
               "KZG different inputs: different challenges")
    }

    // --- Multi-Point KZG Aggregation ---

    // Test 8: Multi-point KZG with 2 different evaluation points
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let c1 = makeTestPoint(frFromInt(7))
        let w0 = makeTestPoint(frFromInt(5))
        let w1 = makeTestPoint(frFromInt(11))
        let v0 = frFromInt(42)
        let v1 = frFromInt(99)
        let z0 = frFromInt(10)
        let z1 = frFromInt(20)

        let result = engine.aggregateMultiPointKZG(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [v0, v1],
            points: [z0, z1]
        )

        expectEqual(result.pointCount, 2, "Multi-point KZG 2: pointCount == 2")
        expect(!pointIsIdentity(result.combinedQuotient),
               "Multi-point KZG 2: combined quotient non-identity")
        expect(!frEqual(result.combinedEvaluation, Fr.zero),
               "Multi-point KZG 2: combined evaluation non-zero")
        expectEqual(result.gammaFactors.count, 2,
                    "Multi-point KZG 2: gamma factors count == 2")
        // First gamma factor is always 1 (gamma^0)
        expect(frEqual(result.gammaFactors[0], Fr.one),
               "Multi-point KZG 2: gamma^0 == 1")
    }

    // Test 9: Multi-point KZG with 4 points
    do {
        var comms = [PointProjective]()
        var wits = [PointProjective]()
        var evals = [Fr]()
        var pts = [Fr]()
        for i in 1...4 {
            comms.append(makeTestPoint(frFromInt(UInt64(i * 2))))
            wits.append(makeTestPoint(frFromInt(UInt64(i * 3))))
            evals.append(frFromInt(UInt64(i * 10)))
            pts.append(frFromInt(UInt64(i * 5)))
        }

        let result = engine.aggregateMultiPointKZG(
            commitments: comms, witnesses: wits,
            evaluations: evals, points: pts
        )

        expectEqual(result.pointCount, 4, "Multi-point KZG 4: pointCount == 4")
        expectEqual(result.gammaFactors.count, 4,
                    "Multi-point KZG 4: gamma factors count == 4")
        // gamma^0 = 1
        expect(frEqual(result.gammaFactors[0], Fr.one),
               "Multi-point KZG 4: gamma^0 == 1")
        // gamma^1 = gamma
        expect(frEqual(result.gammaFactors[1], result.gamma),
               "Multi-point KZG 4: gamma^1 == gamma")
        // gamma^2 = gamma * gamma^1
        let expectedGamma2 = frMul(result.gamma, result.gammaFactors[1])
        expect(frEqual(result.gammaFactors[2], expectedGamma2),
               "Multi-point KZG 4: gamma^2 consistent")
    }

    // --- Batch Pairing Equation Verification ---

    // Test 10: Single satisfied pairing equation
    do {
        // e(3, 5) = e(15, 1) in scalar model: 3*5 == 15*1
        let eq = PairingEquation(
            lhsG1Scalar: frFromInt(3),
            lhsG2Scalar: frFromInt(5),
            rhsG1Scalar: frFromInt(15),
            rhsG2Scalar: Fr.one
        )
        expect(eq.isSatisfied(), "Pairing eq: 3*5 == 15*1 satisfied")

        let valid = engine.batchVerifyPairings(equations: [eq])
        expect(valid, "Batch pairing verify single: passes")
    }

    // Test 11: Multiple satisfied pairing equations
    do {
        let eq1 = PairingEquation(
            lhsG1Scalar: frFromInt(2),
            lhsG2Scalar: frFromInt(3),
            rhsG1Scalar: frFromInt(6),
            rhsG2Scalar: Fr.one
        )
        let eq2 = PairingEquation(
            lhsG1Scalar: frFromInt(4),
            lhsG2Scalar: frFromInt(5),
            rhsG1Scalar: frFromInt(20),
            rhsG2Scalar: Fr.one
        )
        let eq3 = PairingEquation(
            lhsG1Scalar: frFromInt(7),
            lhsG2Scalar: frFromInt(11),
            rhsG1Scalar: frFromInt(77),
            rhsG2Scalar: Fr.one
        )

        let valid = engine.batchVerifyPairings(equations: [eq1, eq2, eq3])
        expect(valid, "Batch pairing verify 3 equations: all pass")
    }

    // Test 12: Unsatisfied pairing equation detected
    do {
        let good = PairingEquation(
            lhsG1Scalar: frFromInt(2),
            lhsG2Scalar: frFromInt(3),
            rhsG1Scalar: frFromInt(6),
            rhsG2Scalar: Fr.one
        )
        let bad = PairingEquation(
            lhsG1Scalar: frFromInt(2),
            lhsG2Scalar: frFromInt(3),
            rhsG1Scalar: frFromInt(7),  // wrong: 2*3 != 7*1
            rhsG2Scalar: Fr.one
        )

        let valid = engine.batchVerifyPairings(equations: [good, bad])
        expect(!valid, "Batch pairing verify: detects bad equation")
    }

    // Test 13: Empty batch trivially passes
    do {
        let valid = engine.batchVerifyPairings(equations: [])
        expect(valid, "Batch pairing verify empty: trivially true")
    }

    // Test 14: KZG to pairing equations conversion
    do {
        let secret = frFromInt(17)
        let z = frFromInt(5)
        let smz = frSub(secret, z)  // 12

        // Proof: commit_scalar = v + witness_scalar * (s - z)
        let v0 = frFromInt(35)
        let ws0 = frFromInt(3)
        let cs0 = frAdd(v0, frMul(ws0, smz))  // 35 + 3*12 = 71

        let v1 = frFromInt(45)
        let ws1 = frFromInt(7)
        let cs1 = frAdd(v1, frMul(ws1, smz))  // 45 + 7*12 = 129

        let equations = engine.kzgToPairingEquations(
            commitScalars: [cs0, cs1],
            witnessScalars: [ws0, ws1],
            evaluations: [v0, v1],
            point: z,
            srsSecret: secret
        )

        expectEqual(equations.count, 2, "KZG->pairing: 2 equations")
        // Each equation: (cs - v) * 1 == ws * (s - z)
        // cs0 - v0 = 71 - 35 = 36, ws0 * smz = 3 * 12 = 36
        expect(equations[0].isSatisfied(), "KZG->pairing: eq 0 satisfied")
        expect(equations[1].isSatisfied(), "KZG->pairing: eq 1 satisfied")

        let valid = engine.batchVerifyPairings(equations: equations)
        expect(valid, "KZG->pairing: batch verify passes")
    }

    // --- Groth16 SnarkPack-Style Aggregation ---

    // Test 15: Groth16 SnarkPack with 2 proofs
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        let proof0 = Groth16Proof(
            a: makeTestPoint(frFromInt(3)),
            b: g2,
            c: makeTestPoint(frFromInt(5))
        )
        let proof1 = Groth16Proof(
            a: makeTestPoint(frFromInt(7)),
            b: g2,
            c: makeTestPoint(frFromInt(11))
        )

        let agg = engine.aggregateGroth16SnarkPack(
            proofs: [proof0, proof1],
            publicInputs: [[frFromInt(42)], [frFromInt(99)]]
        )

        expectEqual(agg.count, 2, "Groth16 SP 2: count == 2")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "Groth16 SP 2: aggA non-identity")
        expect(!pointIsIdentity(agg.aggregatedWitness),
               "Groth16 SP 2: aggC non-identity")
    }

    // Test 16: Groth16 SnarkPack with 4 proofs — IPP structure
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        var proofs = [Groth16Proof]()
        var pubInputs = [[Fr]]()
        for i in 1...4 {
            proofs.append(Groth16Proof(
                a: makeTestPoint(frFromInt(UInt64(i * 3))),
                b: g2,
                c: makeTestPoint(frFromInt(UInt64(i * 7)))
            ))
            pubInputs.append([frFromInt(UInt64(i))])
        }

        let agg = engine.aggregateGroth16SnarkPack(
            proofs: proofs, publicInputs: pubInputs
        )

        expectEqual(agg.count, 4, "Groth16 SP 4: count == 4")
        expect(agg.ippLCommitments.count > 0,
               "Groth16 SP 4: IPP L present")
        expect(agg.ippRCommitments.count > 0,
               "Groth16 SP 4: IPP R present")
        expectEqual(agg.ippLCommitments.count, agg.ippRCommitments.count,
                    "Groth16 SP 4: IPP L/R counts match")
    }

    // Test 17: Groth16 SnarkPack verification roundtrip
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        let proof0 = Groth16Proof(
            a: makeTestPoint(frFromInt(13)),
            b: g2,
            c: makeTestPoint(frFromInt(17))
        )
        let proof1 = Groth16Proof(
            a: makeTestPoint(frFromInt(23)),
            b: g2,
            c: makeTestPoint(frFromInt(29))
        )
        let proofs = [proof0, proof1]
        let pubInputs: [[Fr]] = [[frFromInt(100)], [frFromInt(200)]]

        let agg = engine.aggregateGroth16SnarkPack(
            proofs: proofs, publicInputs: pubInputs
        )

        let valid = engine.verifyGroth16SnarkPack(
            aggregated: agg, proofs: proofs, publicInputs: pubInputs
        )
        expect(valid, "Groth16 SP verify: roundtrip passes")
    }

    // Test 18: Groth16 SnarkPack verification fails with wrong proofs
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        let proof0 = Groth16Proof(
            a: makeTestPoint(frFromInt(13)),
            b: g2,
            c: makeTestPoint(frFromInt(17))
        )
        let proof1 = Groth16Proof(
            a: makeTestPoint(frFromInt(23)),
            b: g2,
            c: makeTestPoint(frFromInt(29))
        )
        let proofs = [proof0, proof1]
        let pubInputs: [[Fr]] = [[frFromInt(100)], [frFromInt(200)]]

        let agg = engine.aggregateGroth16SnarkPack(
            proofs: proofs, publicInputs: pubInputs
        )

        // Swap proofs — verification should fail
        let wrongProofs = [proof1, proof0]
        let invalid = engine.verifyGroth16SnarkPack(
            aggregated: agg, proofs: wrongProofs, publicInputs: pubInputs
        )
        expect(!invalid, "Groth16 SP verify: wrong proofs rejected")
    }

    // Test 19: Groth16 SnarkPack with 8 proofs — deeper IPP
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        var proofs = [Groth16Proof]()
        var pubInputs = [[Fr]]()
        for i in 1...8 {
            proofs.append(Groth16Proof(
                a: makeTestPoint(frFromInt(UInt64(i * 5 + 1))),
                b: g2,
                c: makeTestPoint(frFromInt(UInt64(i * 3 + 2)))
            ))
            pubInputs.append([frFromInt(UInt64(i * 10))])
        }

        let agg = engine.aggregateGroth16SnarkPack(
            proofs: proofs, publicInputs: pubInputs
        )

        expectEqual(agg.count, 8, "Groth16 SP 8: count == 8")
        // 8 points -> 3 rounds of halving (8->4->2->1)
        expectEqual(agg.ippLCommitments.count, 3,
                    "Groth16 SP 8: IPP has 3 rounds (log2(8))")

        let valid = engine.verifyGroth16SnarkPack(
            aggregated: agg, proofs: proofs, publicInputs: pubInputs
        )
        expect(valid, "Groth16 SP 8: verification passes")
    }

    // --- Recursive Proof Composition ---

    // Test 20: Basic recursive composition of 2 polynomials
    do {
        let inner: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let outer: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let commitment = makeTestPoint(frFromInt(42))

        let composed = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        expectEqual(composed.depth, 1, "Recursive compose: depth == 1")
        expectEqual(composed.accPoly.count, 4, "Recursive compose: poly length == 4")
        expectEqual(composed.challenges.count, 2, "Recursive compose: 2 challenges")
        expect(!frEqual(composed.innerDigest, Fr.zero),
               "Recursive compose: inner digest non-zero")
        expect(!frEqual(composed.error, Fr.zero),
               "Recursive compose: error non-zero (non-orthogonal polys)")
    }

    // Test 21: Recursive composition verification roundtrip
    do {
        let inner: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let outer: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let commitment = makeTestPoint(frFromInt(42))

        let composed = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        let valid = engine.verifyRecursiveComposition(
            proof: composed, innerPoly: inner, outerPoly: outer
        )
        expect(valid, "Recursive compose verify: roundtrip passes")
    }

    // Test 22: Recursive composition rejects tampered inner poly
    do {
        let inner: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let outer: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let commitment = makeTestPoint(frFromInt(42))

        let composed = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        // Tamper with inner
        let tamperedInner: [Fr] = [frFromInt(99), frFromInt(2), frFromInt(3), frFromInt(4)]
        let invalid = engine.verifyRecursiveComposition(
            proof: composed, innerPoly: tamperedInner, outerPoly: outer
        )
        expect(!invalid, "Recursive compose verify: tampered inner rejected")
    }

    // Test 23: Recursive composition with orthogonal polys gives zero error
    do {
        // inner = [1, 0, 0, 0], outer = [0, 0, 0, 1]
        // Cross term = sum(inner[i] * outer[i]) = 0
        let inner: [Fr] = [Fr.one, Fr.zero, Fr.zero, Fr.zero]
        let outer: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.one]
        let commitment = makeTestPoint(frFromInt(7))

        let composed = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        expect(frEqual(composed.error, Fr.zero),
               "Recursive compose orthogonal: zero error")

        let valid = engine.verifyRecursiveComposition(
            proof: composed, innerPoly: inner, outerPoly: outer
        )
        expect(valid, "Recursive compose orthogonal: verifies")
    }

    // Test 24: Recursive composition inner digest is deterministic
    do {
        let inner: [Fr] = [frFromInt(10), frFromInt(20)]
        let outer: [Fr] = [frFromInt(30), frFromInt(40)]
        let commitment = makeTestPoint(frFromInt(5))

        let c1 = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )
        let c2 = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        expect(frEqual(c1.innerDigest, c2.innerDigest),
               "Recursive compose: deterministic inner digest")
        expect(frEqual(c1.error, c2.error),
               "Recursive compose: deterministic error")
    }

    // Test 25: Chain composition of 3 polynomials
    do {
        let p0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let p1: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let p2: [Fr] = [frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)]
        let commitment = makeTestPoint(frFromInt(100))

        let chain = engine.composeChain(
            polynomials: [p0, p1, p2], commitment: commitment
        )

        expectEqual(chain.depth, 2, "Chain compose 3: depth == 2")
        expectEqual(chain.accPoly.count, 4, "Chain compose 3: poly length == 4")
        expectEqual(chain.challenges.count, 2, "Chain compose 3: 2 challenges")
        expect(!frEqual(chain.error, Fr.zero),
               "Chain compose 3: non-zero error")
    }

    // Test 26: Chain composition of 4 polynomials
    do {
        let polys: [[Fr]] = (1...4).map { k in
            (1...4).map { i in frFromInt(UInt64(k * 4 + i)) }
        }
        let commitment = makeTestPoint(frFromInt(50))

        let chain = engine.composeChain(polynomials: polys, commitment: commitment)

        expectEqual(chain.depth, 3, "Chain compose 4: depth == 3")
        expectEqual(chain.challenges.count, 3, "Chain compose 4: 3 challenges")
    }

    // Test 27: Chain composition determinism
    do {
        let polys: [[Fr]] = [[frFromInt(1), frFromInt(2)],
                             [frFromInt(3), frFromInt(4)]]
        let commitment = makeTestPoint(frFromInt(7))

        let c1 = engine.composeChain(polynomials: polys, commitment: commitment)
        let c2 = engine.composeChain(polynomials: polys, commitment: commitment)

        expect(frEqual(c1.error, c2.error), "Chain compose determinism: same error")
        for i in 0..<c1.accPoly.count {
            expect(frEqual(c1.accPoly[i], c2.accPoly[i]),
                   "Chain compose determinism: accPoly[\(i)] matches")
        }
    }

    // --- GPU-Parallel MSM ---

    // Test 28: Parallel MSM with 2 points
    do {
        let p0 = makeTestPoint(frFromInt(3))
        let p1 = makeTestPoint(frFromInt(7))
        let s0 = frFromInt(5)
        let s1 = frFromInt(11)

        let result = engine.parallelMSM(points: [p0, p1], scalars: [s0, s1])

        // Expected: 5 * [3]G + 11 * [7]G = [15]G + [77]G = [92]G
        let expected = makeTestPoint(frFromInt(92))
        expect(pointsEqual(result, expected), "Parallel MSM 2: correct result")
    }

    // Test 29: Parallel MSM with zero scalar skips point
    do {
        let p0 = makeTestPoint(frFromInt(3))
        let p1 = makeTestPoint(frFromInt(7))
        let s0 = frFromInt(5)
        let s1 = Fr.zero

        let result = engine.parallelMSM(points: [p0, p1], scalars: [s0, s1])

        // Expected: 5 * [3]G + 0 * [7]G = [15]G
        let expected = makeTestPoint(frFromInt(15))
        expect(pointsEqual(result, expected), "Parallel MSM zero skip: correct")
    }

    // Test 30: Parallel MSM with unit scalar
    do {
        let p0 = makeTestPoint(frFromInt(3))
        let p1 = makeTestPoint(frFromInt(7))
        let s0 = Fr.one
        let s1 = Fr.one

        let result = engine.parallelMSM(points: [p0, p1], scalars: [s0, s1])

        // Expected: 1 * [3]G + 1 * [7]G = [10]G
        let expected = makeTestPoint(frFromInt(10))
        expect(pointsEqual(result, expected), "Parallel MSM unit scalars: correct")
    }

    // Test 31: Dual parallel MSM
    do {
        let pts1 = [makeTestPoint(frFromInt(2)), makeTestPoint(frFromInt(3))]
        let scs1 = [frFromInt(4), frFromInt(5)]
        let pts2 = [makeTestPoint(frFromInt(6)), makeTestPoint(frFromInt(7))]
        let scs2 = [frFromInt(8), frFromInt(9)]

        let (r1, r2) = engine.dualParallelMSM(
            points1: pts1, scalars1: scs1,
            points2: pts2, scalars2: scs2
        )

        // r1 = 4*[2]G + 5*[3]G = [8]G + [15]G = [23]G
        let exp1 = makeTestPoint(frFromInt(23))
        // r2 = 8*[6]G + 9*[7]G = [48]G + [63]G = [111]G
        let exp2 = makeTestPoint(frFromInt(111))

        expect(pointsEqual(r1, exp1), "Dual MSM: result1 correct")
        expect(pointsEqual(r2, exp2), "Dual MSM: result2 correct")
    }

    // Test 32: Parallel MSM with 8 points
    do {
        var pts = [PointProjective]()
        var scs = [Fr]()
        var expectedScalar: UInt64 = 0
        for i in 1...8 {
            pts.append(makeTestPoint(frFromInt(UInt64(i))))
            scs.append(frFromInt(UInt64(i)))
            expectedScalar += UInt64(i * i)  // i * [i]G = [i^2]G
        }

        let result = engine.parallelMSM(points: pts, scalars: scs)
        let expected = makeTestPoint(frFromInt(expectedScalar))
        expect(pointsEqual(result, expected), "Parallel MSM 8: sum(i^2) correct")
    }

    // --- Heterogeneous Batch Aggregation ---

    // Test 33: Heterogeneous batch with 2 KZG proofs
    do {
        let p0 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(3)),
            witness: makeTestPoint(frFromInt(5)),
            evaluation: frFromInt(42)
        )
        let p1 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(7)),
            witness: makeTestPoint(frFromInt(11)),
            evaluation: frFromInt(99)
        )

        let agg = engine.aggregateHeterogeneous(proofs: [p0, p1])

        expectEqual(agg.count, 2, "Hetero 2 KZG: count == 2")
        expectEqual(agg.typeCounts[.kzg], 2, "Hetero 2 KZG: type count == 2")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "Hetero 2 KZG: commitment non-identity")
    }

    // Test 34: Heterogeneous batch with mixed KZG + Groth16
    do {
        let kzgProof = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(3)),
            witness: makeTestPoint(frFromInt(5)),
            evaluation: frFromInt(42)
        )
        let groth16Proof = HeterogeneousProof.groth16(
            a: makeTestPoint(frFromInt(7)),
            c: makeTestPoint(frFromInt(11)),
            publicInputs: [frFromInt(100)]
        )

        let agg = engine.aggregateHeterogeneous(proofs: [kzgProof, groth16Proof])

        expectEqual(agg.count, 2, "Hetero mixed: count == 2")
        expectEqual(agg.typeCounts[.kzg], 1, "Hetero mixed: 1 KZG")
        expectEqual(agg.typeCounts[.groth16], 1, "Hetero mixed: 1 Groth16")
    }

    // Test 35: Heterogeneous with explicit strategy override
    do {
        let p0 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(3)),
            witness: makeTestPoint(frFromInt(5)),
            evaluation: frFromInt(42)
        )
        let p1 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(7)),
            witness: makeTestPoint(frFromInt(11)),
            evaluation: frFromInt(99)
        )

        let aggLinear = engine.aggregateHeterogeneous(
            proofs: [p0, p1], strategy: .linearCombination
        )
        let aggIPP = engine.aggregateHeterogeneous(
            proofs: [p0, p1], strategy: .innerProductArgument
        )

        // Both should produce valid results, but IPP has L/R commitments
        expectEqual(aggLinear.ippLCommitments.count, 0,
                    "Hetero linear: no IPP L commitments")
        expect(aggIPP.ippLCommitments.count > 0,
               "Hetero IPP: has IPP L commitments")
    }

    // Test 36: Heterogeneous with recursive folding strategy
    do {
        var proofs = [HeterogeneousProof]()
        for i in 1...3 {
            let pt: BatchProofType = i % 2 == 0 ? .kzg : .groth16
            let proof = HeterogeneousProof(
                proofType: pt,
                commitments: [makeTestPoint(frFromInt(UInt64(i * 3)))],
                evaluations: [frFromInt(UInt64(i * 10)), frFromInt(UInt64(i * 20))],
                witnesses: [makeTestPoint(frFromInt(UInt64(i * 5)))]
            )
            proofs.append(proof)
        }

        let agg = engine.aggregateHeterogeneous(
            proofs: proofs, strategy: .recursiveFolding
        )

        expectEqual(agg.count, 3, "Hetero recursive: count == 3")
        expect(agg.accumulatorPoly.count > 0,
               "Hetero recursive: accumulator poly present")
    }

    // --- Tree Aggregation ---

    // Test 38: Tree aggregation with 2 commitments
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let c1 = makeTestPoint(frFromInt(7))
        let v0 = frFromInt(42)
        let v1 = frFromInt(99)

        let (root, rootEval, depth) = engine.treeAggregate(
            commitments: [c0, c1], evaluations: [v0, v1]
        )

        expectEqual(depth, 1, "Tree agg 2: depth == 1")
        expect(!pointIsIdentity(root), "Tree agg 2: root non-identity")
        expect(!frEqual(rootEval, Fr.zero), "Tree agg 2: root eval non-zero")
    }

    // Test 39: Tree aggregation with 4 commitments (perfect binary tree)
    do {
        var comms = [PointProjective]()
        var evals = [Fr]()
        for i in 1...4 {
            comms.append(makeTestPoint(frFromInt(UInt64(i * 5))))
            evals.append(frFromInt(UInt64(i * 10)))
        }

        let (root, rootEval, depth) = engine.treeAggregate(
            commitments: comms, evaluations: evals
        )

        expectEqual(depth, 2, "Tree agg 4: depth == 2 (log2(4))")
        expect(!pointIsIdentity(root), "Tree agg 4: root non-identity")
        expect(!frEqual(rootEval, Fr.zero), "Tree agg 4: root eval non-zero")
    }

    // Test 40: Tree aggregation with 5 commitments (odd, carried forward)
    do {
        var comms = [PointProjective]()
        var evals = [Fr]()
        for i in 1...5 {
            comms.append(makeTestPoint(frFromInt(UInt64(i * 3))))
            evals.append(frFromInt(UInt64(i * 7)))
        }

        let (root, rootEval, depth) = engine.treeAggregate(
            commitments: comms, evaluations: evals
        )

        expect(depth >= 2, "Tree agg 5: depth >= 2")
        expect(!pointIsIdentity(root), "Tree agg 5: root non-identity")
    }

    // Test 41: Tree aggregation with 8 commitments
    do {
        var comms = [PointProjective]()
        var evals = [Fr]()
        for i in 1...8 {
            comms.append(makeTestPoint(frFromInt(UInt64(i * 2))))
            evals.append(frFromInt(UInt64(i)))
        }

        let (root, rootEval, depth) = engine.treeAggregate(
            commitments: comms, evaluations: evals
        )

        expectEqual(depth, 3, "Tree agg 8: depth == 3 (log2(8))")
        expect(!pointIsIdentity(root), "Tree agg 8: root non-identity")
    }

    // Test 42: Single element tree aggregation
    do {
        let c0 = makeTestPoint(frFromInt(42))
        let v0 = frFromInt(100)

        let (root, rootEval, depth) = engine.treeAggregate(
            commitments: [c0], evaluations: [v0]
        )

        expectEqual(depth, 0, "Tree agg 1: depth == 0")
        expect(pointsEqual(root, c0), "Tree agg 1: root == input")
        expect(frEqual(rootEval, v0), "Tree agg 1: eval preserved")
    }

    // --- Polynomial Operations ---

    // Test 44: Polynomial evaluation at a point
    do {
        // p(x) = 3 + 2x + x^2
        let poly: [Fr] = [frFromInt(3), frFromInt(2), Fr.one]
        let point = frFromInt(5)

        let result = engine.evaluatePolynomial(poly, at: point)

        // p(5) = 3 + 2*5 + 25 = 38
        let expected = frFromInt(38)
        expect(frEqual(result, expected), "Poly eval: p(5) = 38")
    }

    // Test 45: Polynomial evaluation at zero
    do {
        let poly: [Fr] = [frFromInt(7), frFromInt(2), frFromInt(3)]
        let result = engine.evaluatePolynomial(poly, at: Fr.zero)
        expect(frEqual(result, frFromInt(7)), "Poly eval at zero: constant term")
    }

    // Test 46: Batch polynomial evaluation
    do {
        let p0: [Fr] = [frFromInt(1), frFromInt(2)]  // 1 + 2x
        let p1: [Fr] = [frFromInt(3), frFromInt(4)]  // 3 + 4x
        let point = frFromInt(5)

        let results = engine.evaluatePolynomialsBatch([p0, p1], at: point)

        expectEqual(results.count, 2, "Batch eval: 2 results")
        expect(frEqual(results[0], frFromInt(11)), "Batch eval: p0(5) = 11")
        expect(frEqual(results[1], frFromInt(23)), "Batch eval: p1(5) = 23")
    }

    // Test 47: Synthetic division
    do {
        // p(x) = x^2 - 1 = (x-1)(x+1)
        // Divide by (x - 1) should give (x + 1) = [1, 1]
        let negOne = frSub(Fr.zero, Fr.one)
        let poly: [Fr] = [negOne, Fr.zero, Fr.one]  // -1 + 0*x + x^2
        let point = Fr.one

        let quotient = engine.syntheticDivision(poly: poly, point: point)

        expectEqual(quotient.count, 2, "Synthetic div: quotient degree 1")
        // The quotient should evaluate correctly: q(x) = x + 1
        let qAt2 = engine.evaluatePolynomial(quotient, at: frFromInt(2))
        // (x^2-1)/(x-1) at x=2 should be 3
        expect(frEqual(qAt2, frFromInt(3)), "Synthetic div: q(2) = 3")
    }

    // Test 48: Synthetic division of linear polynomial
    do {
        // p(x) = 2x + 6 = 2(x + 3), divide by (x - (-3)) = (x + 3)
        // quotient = [2]
        let poly: [Fr] = [frFromInt(6), frFromInt(2)]
        let negThree = frSub(Fr.zero, frFromInt(3))

        let quotient = engine.syntheticDivision(poly: poly, point: negThree)

        expectEqual(quotient.count, 1, "Synthetic div linear: quotient degree 0")
        expect(frEqual(quotient[0], frFromInt(2)), "Synthetic div linear: quotient = 2")
    }

    // Test 49: Empty polynomial evaluation
    do {
        let result = engine.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(result, Fr.zero), "Empty poly eval: returns zero")
    }

    // --- BatchAggregationTranscript ---

    // Test 50: Transcript determinism
    do {
        var t1 = BatchAggregationTranscript(label: "test")
        t1.appendScalar(frFromInt(42))
        let c1 = t1.squeeze()

        var t2 = BatchAggregationTranscript(label: "test")
        t2.appendScalar(frFromInt(42))
        let c2 = t2.squeeze()

        expect(frEqual(c1, c2), "Transcript determinism: same challenge")
    }

    // Test 51: Transcript with different labels produces different challenges
    do {
        var t1 = BatchAggregationTranscript(label: "alpha")
        t1.appendScalar(frFromInt(42))
        let c1 = t1.squeeze()

        var t2 = BatchAggregationTranscript(label: "beta")
        t2.appendScalar(frFromInt(42))
        let c2 = t2.squeeze()

        expect(!frEqual(c1, c2), "Transcript different labels: different challenges")
    }

    // Test 52: Transcript squeeze changes state (chaining)
    do {
        var t = BatchAggregationTranscript(label: "chain")
        t.appendScalar(frFromInt(1))
        let c1 = t.squeeze()
        let c2 = t.squeeze()

        expect(!frEqual(c1, c2), "Transcript chaining: consecutive squeezes differ")
    }

    // Test 53: Transcript with point appended
    do {
        var t1 = BatchAggregationTranscript(label: "point-test")
        let p = makeTestPoint(frFromInt(7))
        t1.appendPoint(p)
        let c1 = t1.squeeze()

        var t2 = BatchAggregationTranscript(label: "point-test")
        t2.appendPoint(p)
        let c2 = t2.squeeze()

        expect(frEqual(c1, c2), "Transcript point append: deterministic")

        // Different point should give different challenge
        var t3 = BatchAggregationTranscript(label: "point-test")
        t3.appendPoint(makeTestPoint(frFromInt(13)))
        let c3 = t3.squeeze()

        expect(!frEqual(c1, c3), "Transcript different point: different challenge")
    }

    // --- Integration / Edge Cases ---

    // Test 55: Large KZG batch (16 proofs)
    do {
        var comms = [PointProjective]()
        var wits = [PointProjective]()
        var evals = [Fr]()
        for i in 1...16 {
            comms.append(makeTestPoint(frFromInt(UInt64(i * 3 + 1))))
            wits.append(makeTestPoint(frFromInt(UInt64(i * 5 + 2))))
            evals.append(frFromInt(UInt64(i * 7)))
        }

        let agg = engine.aggregateKZGBatch(
            commitments: comms, witnesses: wits, evaluations: evals
        )

        expectEqual(agg.count, 16, "KZG batch 16: count == 16")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "KZG batch 16: commitment non-identity")
        expect(!frEqual(agg.challenge, Fr.zero),
               "KZG batch 16: challenge non-zero")
    }

    // Test 56: HeterogeneousProof.kzg factory
    do {
        let c = makeTestPoint(frFromInt(3))
        let w = makeTestPoint(frFromInt(5))
        let v = frFromInt(42)

        let proof = HeterogeneousProof.kzg(commitment: c, witness: w, evaluation: v)

        expectEqual(proof.commitments.count, 1, "KZG factory: 1 commitment")
        expectEqual(proof.witnesses.count, 1, "KZG factory: 1 witness")
        expectEqual(proof.evaluations.count, 1, "KZG factory: 1 evaluation")
        expectEqual(proof.auxScalars.count, 0, "KZG factory: 0 aux scalars")
    }

    // Test 57: HeterogeneousProof.groth16 factory
    do {
        let a = makeTestPoint(frFromInt(3))
        let c = makeTestPoint(frFromInt(5))
        let pubIn: [Fr] = [frFromInt(42), frFromInt(99)]

        let proof = HeterogeneousProof.groth16(a: a, c: c, publicInputs: pubIn)

        expectEqual(proof.commitments.count, 2, "Groth16 factory: 2 commitments (A,C)")
        expectEqual(proof.evaluations.count, 2, "Groth16 factory: 2 evaluations (pub inputs)")
        expectEqual(proof.witnesses.count, 0, "Groth16 factory: 0 witnesses")
    }

    // Test 58: PairingEquation satisfied and unsatisfied
    do {
        // 6 * 7 = 42 * 1
        let sat = PairingEquation(
            lhsG1Scalar: frFromInt(6),
            lhsG2Scalar: frFromInt(7),
            rhsG1Scalar: frFromInt(42),
            rhsG2Scalar: Fr.one
        )
        expect(sat.isSatisfied(), "PairingEquation: 6*7 == 42*1 satisfied")

        // 6 * 7 != 43 * 1
        let unsat = PairingEquation(
            lhsG1Scalar: frFromInt(6),
            lhsG2Scalar: frFromInt(7),
            rhsG1Scalar: frFromInt(43),
            rhsG2Scalar: Fr.one
        )
        expect(!unsat.isSatisfied(), "PairingEquation: 6*7 != 43*1 unsatisfied")
    }

    // Test 59: BatchProofType raw values
    do {
        let kzg = BatchProofType.kzg
        let groth16 = BatchProofType.groth16
        let ipa = BatchProofType.ipa
        let fri = BatchProofType.fri
        let plonk = BatchProofType.plonk

        expectEqual(kzg.rawValue, 0, "BatchProofType: kzg == 0")
        expectEqual(groth16.rawValue, 1, "BatchProofType: groth16 == 1")
        expectEqual(ipa.rawValue, 2, "BatchProofType: ipa == 2")
        expectEqual(fri.rawValue, 3, "BatchProofType: fri == 3")
        expectEqual(plonk.rawValue, 4, "BatchProofType: plonk == 4")
    }

    // Test 60: End-to-end: KZG batch aggregate + verify + multi-point
    do {
        let secret = frFromInt(23)
        let sG1 = makeTestPoint(secret)

        // 3 valid KZG proofs at same point
        let z = frFromInt(7)
        let smz = frSub(secret, z)

        var comms = [PointProjective]()
        var wits = [PointProjective]()
        var evals = [Fr]()

        // p0(x) = 10 + 2x, p0(7) = 24
        let a0 = frFromInt(10)
        let a1_0 = frFromInt(2)
        let pz0 = frAdd(a0, frMul(a1_0, z))
        comms.append(pointAdd(cPointScalarMul(g1, a0), cPointScalarMul(sG1, a1_0)))
        wits.append(cPointScalarMul(g1, a1_0))
        evals.append(pz0)

        // p1(x) = 5 + 3x, p1(7) = 26
        let b0 = frFromInt(5)
        let b1 = frFromInt(3)
        let pz1 = frAdd(b0, frMul(b1, z))
        comms.append(pointAdd(cPointScalarMul(g1, b0), cPointScalarMul(sG1, b1)))
        wits.append(cPointScalarMul(g1, b1))
        evals.append(pz1)

        // p2(x) = 1 + 4x, p2(7) = 29
        let d0 = frFromInt(1)
        let d1 = frFromInt(4)
        let pz2 = frAdd(d0, frMul(d1, z))
        comms.append(pointAdd(cPointScalarMul(g1, d0), cPointScalarMul(sG1, d1)))
        wits.append(cPointScalarMul(g1, d1))
        evals.append(pz2)

        // Aggregate and verify
        let agg = engine.aggregateKZGBatch(
            commitments: comms, witnesses: wits, evaluations: evals
        )
        let valid = engine.verifyKZGBatch(
            proof: agg, point: z, srsG1: g1, srsSecret: secret
        )
        expect(valid, "E2E KZG 3 proofs: aggregated batch verifies")

        // Also test multi-point with same point (should still work)
        let mp = engine.aggregateMultiPointKZG(
            commitments: comms, witnesses: wits,
            evaluations: evals, points: [z, z, z]
        )
        expectEqual(mp.pointCount, 3, "E2E multi-point same z: pointCount == 3")
        expect(!pointIsIdentity(mp.combinedQuotient),
               "E2E multi-point same z: non-identity quotient")
    }

    // Test 61: Recursive composition followed by chain composition
    do {
        let inner: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let outer: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let commitment = makeTestPoint(frFromInt(42))

        let composed = engine.composeRecursive(
            innerPoly: inner, outerPoly: outer, outerCommitment: commitment
        )

        // Verify it
        let valid = engine.verifyRecursiveComposition(
            proof: composed, innerPoly: inner, outerPoly: outer
        )
        expect(valid, "Recursive + chain: initial composition verifies")

        // Now chain the accumulated poly with a third polynomial
        let third: [Fr] = [frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)]
        let chain = engine.composeChain(
            polynomials: [composed.accPoly, third], commitment: commitment
        )

        expectEqual(chain.depth, 1, "Recursive + chain: chain depth == 1")
        expect(!frEqual(chain.error, Fr.zero),
               "Recursive + chain: chain has non-zero error")
    }

    // Test 62: BatchAggregatedProof type count tracking
    do {
        let kzg1 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(1)),
            witness: makeTestPoint(frFromInt(2)),
            evaluation: frFromInt(3)
        )
        let kzg2 = HeterogeneousProof.kzg(
            commitment: makeTestPoint(frFromInt(4)),
            witness: makeTestPoint(frFromInt(5)),
            evaluation: frFromInt(6)
        )
        let g16 = HeterogeneousProof.groth16(
            a: makeTestPoint(frFromInt(7)),
            c: makeTestPoint(frFromInt(8)),
            publicInputs: [frFromInt(9)]
        )

        let agg = engine.aggregateHeterogeneous(proofs: [kzg1, kzg2, g16])

        expectEqual(agg.count, 3, "Type tracking: total == 3")
        expectEqual(agg.typeCounts[.kzg], 2, "Type tracking: 2 KZG")
        expectEqual(agg.typeCounts[.groth16], 1, "Type tracking: 1 Groth16")
        expect(agg.typeCounts[.ipa] == nil, "Type tracking: no IPA")
    }

    // Test 63: GPU MSM threshold constant
    do {
        expectEqual(GPUProofBatchAggregationEngine.gpuMSMThreshold, 32,
                    "GPU MSM threshold == 32")
    }
}
