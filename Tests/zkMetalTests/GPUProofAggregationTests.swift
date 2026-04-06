import zkMetal

// MARK: - GPU Proof Aggregation Tests

public func runGPUProofAggregationTests() {
    suite("GPU Proof Aggregation Engine")

    let engine = GPUProofAggregationEngine()
    let g1Aff = bn254G1Generator()
    let g1 = pointFromAffine(g1Aff)

    // Helper: create a test KZG-style commitment/witness from a scalar
    func makeTestPoint(_ scalar: Fr) -> PointProjective {
        cPointScalarMul(g1, scalar)
    }

    // Test 1: Aggregate 2 KZG proofs
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let c1 = makeTestPoint(frFromInt(7))
        let w0 = makeTestPoint(frFromInt(5))
        let w1 = makeTestPoint(frFromInt(11))
        let v0 = frFromInt(42)
        let v1 = frFromInt(99)

        let agg = engine.aggregateKZGProofs(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [v0, v1]
        )

        expect(agg.count == 2, "Aggregate 2 KZG proofs: count == 2")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "Aggregate 2 KZG proofs: commitment non-identity")
        expect(!pointIsIdentity(agg.aggregatedWitness),
               "Aggregate 2 KZG proofs: witness non-identity")

        // Aggregated evaluation should be v0 + r * v1
        // Just check it's non-zero
        let aggVInt = frToInt(agg.aggregatedEvaluation)
        expect(aggVInt != frToInt(Fr.zero), "Aggregate 2 KZG proofs: evaluation non-zero")
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

        let agg = engine.aggregateKZGProofs(
            commitments: commitments,
            witnesses: witnesses,
            evaluations: evaluations
        )

        expect(agg.count == 4, "Aggregate 4 KZG proofs: count == 4")
        expect(!pointIsIdentity(agg.aggregatedCommitment),
               "Aggregate 4 KZG proofs: aggregated commitment non-identity")
        expect(!pointIsIdentity(agg.aggregatedWitness),
               "Aggregate 4 KZG proofs: aggregated witness non-identity")
    }

    // Test 3: Verify aggregated KZG proof (using SRS secret)
    do {
        let secret = frFromInt(17)   // toxic waste
        let srsG1 = g1               // generator
        let sG1 = makeTestPoint(secret) // [s]*G

        // Create two "commitments" as [coeff]*G and "witnesses" as [(coeff-v)/(s-z)]*G
        // For simplicity, create consistent proofs: C = [a]*G, eval at z, W = [(a-v)/(s-z)]*G
        let z = frFromInt(5)
        let a0 = frFromInt(20)
        let v0 = frFromInt(20)  // p(z) = a0 (constant poly)
        // quotient = (a0 - v0) / (s - z) = 0 for constant poly
        // For a non-trivial case: p(x) = a0 + a1*x, p(z) = a0 + a1*z
        let a1_0 = frFromInt(3)
        let pz0 = frAdd(a0, frMul(a1_0, z))  // 20 + 3*5 = 35
        // C0 = a0*G + a1*[s]*G = [a0 + a1*s]*G
        let c0 = pointAdd(cPointScalarMul(srsG1, a0), cPointScalarMul(sG1, a1_0))
        // W0 = [a1]*G (quotient of linear poly by (x - z) is just a1)
        let w0 = cPointScalarMul(srsG1, a1_0)

        let a0_1 = frFromInt(10)
        let a1_1 = frFromInt(7)
        let pz1 = frAdd(a0_1, frMul(a1_1, z))  // 10 + 7*5 = 45
        let c1 = pointAdd(cPointScalarMul(srsG1, a0_1), cPointScalarMul(sG1, a1_1))
        let w1 = cPointScalarMul(srsG1, a1_1)

        // Verify individual proofs first: C == [v]*G + [s-z]*W
        let smz = frSub(secret, z)
        let exp0 = pointAdd(cPointScalarMul(srsG1, pz0), cPointScalarMul(w0, smz))
        let c0Aff = batchToAffine([c0])
        let e0Aff = batchToAffine([exp0])
        let indiv0OK = fpToInt(c0Aff[0].x) == fpToInt(e0Aff[0].x) &&
                       fpToInt(c0Aff[0].y) == fpToInt(e0Aff[0].y)
        expect(indiv0OK, "Verify aggregated KZG: individual proof 0 valid")

        let agg = engine.aggregateKZGProofs(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [pz0, pz1]
        )

        let valid = engine.verifyAggregatedKZG(
            proof: agg,
            point: z,
            srsG1: srsG1,
            srsSecret: secret
        )
        expect(valid, "Verify aggregated KZG: valid proof passes")
    }

    // Test 4: Tampered proof rejection
    do {
        let secret = frFromInt(17)
        let z = frFromInt(5)
        let sG1 = makeTestPoint(secret)
        let smz = frSub(secret, z)

        let a0 = frFromInt(20)
        let a1 = frFromInt(3)
        let pz = frAdd(a0, frMul(a1, z))
        let c0 = pointAdd(cPointScalarMul(g1, a0), cPointScalarMul(sG1, a1))
        let w0 = cPointScalarMul(g1, a1)

        let a0b = frFromInt(10)
        let a1b = frFromInt(7)
        let pzb = frAdd(a0b, frMul(a1b, z))
        let c1 = pointAdd(cPointScalarMul(g1, a0b), cPointScalarMul(sG1, a1b))
        let w1 = cPointScalarMul(g1, a1b)

        // Tamper: use wrong evaluation for second proof
        let wrongEval = frFromInt(999)

        let agg = engine.aggregateKZGProofs(
            commitments: [c0, c1],
            witnesses: [w0, w1],
            evaluations: [pz, wrongEval]  // tampered!
        )

        let valid = engine.verifyAggregatedKZG(
            proof: agg,
            point: z,
            srsG1: g1,
            srsSecret: secret
        )
        expect(!valid, "Tampered KZG proof: rejected")
    }

    // Test 5: Empty aggregation guard (single proof aggregation)
    do {
        let c0 = makeTestPoint(frFromInt(42))
        let w0 = makeTestPoint(frFromInt(7))
        let v0 = frFromInt(100)

        let agg = engine.aggregateKZGProofs(
            commitments: [c0],
            witnesses: [w0],
            evaluations: [v0]
        )

        expect(agg.count == 1, "Single proof aggregation: count == 1")
        // For a single proof, r^0 = 1, so aggregated == original
        let cAff = batchToAffine([c0])
        let aggCAff = batchToAffine([agg.aggregatedCommitment])
        let commitMatch = fpToInt(cAff[0].x) == fpToInt(aggCAff[0].x) &&
                          fpToInt(cAff[0].y) == fpToInt(aggCAff[0].y)
        expect(commitMatch, "Single proof aggregation: commitment preserved")

        let wAff = batchToAffine([w0])
        let aggWAff = batchToAffine([agg.aggregatedWitness])
        let witnessMatch = fpToInt(wAff[0].x) == fpToInt(aggWAff[0].x) &&
                           fpToInt(wAff[0].y) == fpToInt(aggWAff[0].y)
        expect(witnessMatch, "Single proof aggregation: witness preserved")

        let evalMatch = frToInt(agg.aggregatedEvaluation) == frToInt(v0)
        expect(evalMatch, "Single proof aggregation: evaluation preserved")
    }

    // Test 6: Groth16 aggregation with 2 proofs
    do {
        let g2Aff = bn254G2Generator()
        let g2 = g2FromAffine(g2Aff)

        // Create dummy Groth16 proofs (not valid pairing-wise, but tests aggregation mechanics)
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
        let pubInputs0: [Fr] = [frFromInt(42)]
        let pubInputs1: [Fr] = [frFromInt(99)]

        let agg = engine.aggregateGroth16Proofs(
            proofs: [proof0, proof1],
            publicInputs: [pubInputs0, pubInputs1]
        )

        expect(agg.count == 2, "Groth16 aggregation 2 proofs: count == 2")
        expect(!pointIsIdentity(agg.aggA), "Groth16 aggregation: aggA non-identity")
        expect(!pointIsIdentity(agg.aggC), "Groth16 aggregation: aggC non-identity")

        // Verify aggregation consistency
        let valid = engine.verifyGroth16Aggregation(
            aggregatedProof: agg,
            originalProofs: [proof0, proof1]
        )
        expect(valid, "Groth16 aggregation 2 proofs: verification passes")
    }

    // Test 7: Groth16 aggregation with 4 proofs
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

        let agg = engine.aggregateGroth16Proofs(
            proofs: proofs,
            publicInputs: pubInputs
        )

        expect(agg.count == 4, "Groth16 aggregation 4 proofs: count == 4")
        expect(agg.ippLCommitments.count > 0, "Groth16 aggregation 4 proofs: IPP L commitments present")
        expect(agg.ippRCommitments.count > 0, "Groth16 aggregation 4 proofs: IPP R commitments present")
        expect(agg.ippLCommitments.count == agg.ippRCommitments.count,
               "Groth16 aggregation 4 proofs: IPP L/R counts match")

        let valid = engine.verifyGroth16Aggregation(
            aggregatedProof: agg,
            originalProofs: proofs
        )
        expect(valid, "Groth16 aggregation 4 proofs: verification passes")
    }

    // Test 8: Deterministic transcript (same inputs produce same challenge)
    do {
        let c0 = makeTestPoint(frFromInt(3))
        let w0 = makeTestPoint(frFromInt(5))
        let v0 = frFromInt(42)

        let agg1 = engine.aggregateKZGProofs(
            commitments: [c0], witnesses: [w0], evaluations: [v0]
        )
        let agg2 = engine.aggregateKZGProofs(
            commitments: [c0], witnesses: [w0], evaluations: [v0]
        )

        let match = frToInt(agg1.challenge) == frToInt(agg2.challenge)
        expect(match, "Deterministic transcript: same inputs produce same challenge")
    }
}
