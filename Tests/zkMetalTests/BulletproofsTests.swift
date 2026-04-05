import zkMetal

// MARK: - Bulletproofs Tests

public func runBulletproofsTests() {
    suite("Bulletproofs Range Proofs")

    // --- Inner Product Argument Sub-Protocol ---

    // Test 1: Inner product argument correctness (n=4)
    do {
        let n = 4
        let (gens, Q) = IPAEngine.generateTestGenerators(count: 2 * n + 1)
        let G = Array(gens[0..<n])
        let H = Array(gens[n..<(2 * n)])
        let U = pointFromAffine(gens[2 * n])

        let a: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let b: [Fr] = [frFromInt(2), frFromInt(4), frFromInt(6), frFromInt(8)]

        // Compute commitment P = <a, G> + <b, H> + <a,b> * U
        let P = InnerProductArgument.computeCommitment(G: G, H: H, U: U, a: a, b: b)

        let proof = InnerProductArgument.prove(G: G, H: H, U: U, a: a, b: b)
        let valid = InnerProductArgument.verify(G: G, H: H, U: U, P: P, proof: proof)
        expect(valid, "Inner product argument n=4")
    }

    // Test 2: Inner product argument (n=8)
    do {
        let n = 8
        let (gens, _) = IPAEngine.generateTestGenerators(count: 2 * n + 1)
        let G = Array(gens[0..<n])
        let H = Array(gens[n..<(2 * n)])
        let U = pointFromAffine(gens[2 * n])

        var a = [Fr]()
        var b = [Fr]()
        for i in 0..<n {
            a.append(frFromInt(UInt64(i + 1)))
            b.append(frFromInt(UInt64(n - i)))
        }

        let P = InnerProductArgument.computeCommitment(G: G, H: H, U: U, a: a, b: b)
        let proof = InnerProductArgument.prove(G: G, H: H, U: U, a: a, b: b)
        let valid = InnerProductArgument.verify(G: G, H: H, U: U, P: P, proof: proof)
        expect(valid, "Inner product argument n=8")
    }

    // Test 3: Inner product proof should fail with wrong commitment
    do {
        let n = 4
        let (gens, _) = IPAEngine.generateTestGenerators(count: 2 * n + 1)
        let G = Array(gens[0..<n])
        let H = Array(gens[n..<(2 * n)])
        let U = pointFromAffine(gens[2 * n])

        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let b: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

        // Compute correct commitment then tamper with it
        var P = InnerProductArgument.computeCommitment(G: G, H: H, U: U, a: a, b: b)
        // Add an extra point to corrupt the commitment
        P = pointAdd(P, pointFromAffine(G[0]))

        let proof = InnerProductArgument.prove(G: G, H: H, U: U, a: a, b: b)
        let valid = InnerProductArgument.verify(G: G, H: H, U: U, P: P, proof: proof)
        expect(!valid, "Inner product rejects wrong commitment")
    }

    // --- Single Range Proofs ---

    // Test 4: 8-bit range proof, value = 42
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(12345)

        let (V, proof) = BulletproofsProver.prove(value: 42, gamma: gamma, params: params)
        let valid = BulletproofsVerifier.verify(V: V, proof: proof, params: params)
        expect(valid, "8-bit range proof v=42")
    }

    // Test 5: 8-bit range proof, value = 0 (lower boundary)
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(99)

        let (V, proof) = BulletproofsProver.prove(value: 0, gamma: gamma, params: params)
        let valid = BulletproofsVerifier.verify(V: V, proof: proof, params: params)
        expect(valid, "8-bit range proof v=0 (boundary)")
    }

    // Test 6: 8-bit range proof, value = 255 (upper boundary)
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(777)

        let (V, proof) = BulletproofsProver.prove(value: 255, gamma: gamma, params: params)
        let valid = BulletproofsVerifier.verify(V: V, proof: proof, params: params)
        expect(valid, "8-bit range proof v=255 (boundary)")
    }

    // Test 7: 16-bit range proof
    do {
        let params = BulletproofsParams.generate(n: 16)
        let gamma = frFromInt(54321)

        let (V, proof) = BulletproofsProver.prove(value: 50000, gamma: gamma, params: params)
        let valid = BulletproofsVerifier.verify(V: V, proof: proof, params: params)
        expect(valid, "16-bit range proof v=50000")
    }

    // Test 8: 32-bit range proof
    do {
        let params = BulletproofsParams.generate(n: 32)
        let gamma = frFromInt(11111)

        let (V, proof) = BulletproofsProver.prove(value: 3_000_000_000, gamma: gamma, params: params)
        let valid = BulletproofsVerifier.verify(V: V, proof: proof, params: params)
        expect(valid, "32-bit range proof v=3B")
    }

    // Test 9: Verifier rejects tampered proof (modified tHat)
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(42)

        let (V, proof) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params)

        // Tamper with tHat
        let badProof = BulletproofsRangeProof(
            A: proof.A, S: proof.S, T1: proof.T1, T2: proof.T2,
            taux: proof.taux, mu: proof.mu,
            tHat: frAdd(proof.tHat, Fr.one),  // corrupted
            innerProductProof: proof.innerProductProof
        )
        let valid = BulletproofsVerifier.verify(V: V, proof: badProof, params: params)
        expect(!valid, "Verifier rejects tampered tHat")
    }

    // Test 10: Verifier rejects tampered proof (modified taux)
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(42)

        let (V, proof) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params)

        // Tamper with taux
        let badProof = BulletproofsRangeProof(
            A: proof.A, S: proof.S, T1: proof.T1, T2: proof.T2,
            taux: frAdd(proof.taux, Fr.one),  // corrupted
            mu: proof.mu, tHat: proof.tHat,
            innerProductProof: proof.innerProductProof
        )
        let valid = BulletproofsVerifier.verify(V: V, proof: badProof, params: params)
        expect(!valid, "Verifier rejects tampered taux")
    }

    // Test 11: Verifier rejects wrong value commitment
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(42)

        let (V, proof) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params)

        // Create a different commitment (for value 200)
        let gProj = pointFromAffine(params.g)
        let hProj = pointFromAffine(params.h)
        let wrongV = pointAdd(cPointScalarMul(gProj, frFromInt(200)),
                             cPointScalarMul(hProj, gamma))

        let valid = BulletproofsVerifier.verify(V: wrongV, proof: proof, params: params)
        expect(!valid, "Verifier rejects wrong V commitment")
    }

    // --- Aggregated Range Proofs ---

    // Test 12: Aggregated proof for 2 values
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gammas = [frFromInt(111), frFromInt(222)]

        let (Vs, proof) = BulletproofsAggregatedProver.prove(
            values: [42, 200], gammas: gammas, params: params
        )
        let valid = BulletproofsAggregatedVerifier.verify(
            Vs: Vs, proof: proof, params: params
        )
        expect(valid, "Aggregated proof m=2")
    }

    // Test 13: Aggregated proof for 4 values
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gammas = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

        let (Vs, proof) = BulletproofsAggregatedProver.prove(
            values: [0, 128, 200, 255], gammas: gammas, params: params
        )
        let valid = BulletproofsAggregatedVerifier.verify(
            Vs: Vs, proof: proof, params: params
        )
        expect(valid, "Aggregated proof m=4")
    }

    // Test 14: Aggregated verifier rejects tampered proof
    do {
        let params = BulletproofsParams.generate(n: 8)
        let gammas = [frFromInt(111), frFromInt(222)]

        let (Vs, proof) = BulletproofsAggregatedProver.prove(
            values: [42, 200], gammas: gammas, params: params
        )

        // Tamper with tHat
        let badProof = BulletproofsAggregatedProof(
            A: proof.A, S: proof.S, T1: proof.T1, T2: proof.T2,
            taux: proof.taux, mu: proof.mu,
            tHat: frAdd(proof.tHat, Fr.one),
            innerProductProof: proof.innerProductProof,
            m: proof.m
        )
        let valid = BulletproofsAggregatedVerifier.verify(
            Vs: Vs, proof: badProof, params: params
        )
        expect(!valid, "Aggregated verifier rejects tampered tHat")
    }

    // Test 15: Proof size is logarithmic (inner product proof has log(n) rounds)
    do {
        let params8 = BulletproofsParams.generate(n: 8)
        let gamma = frFromInt(42)
        let (_, proof8) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params8)
        expectEqual(proof8.innerProductProof.L.count, 3, "8-bit proof has log2(8)=3 rounds")

        let params16 = BulletproofsParams.generate(n: 16)
        let (_, proof16) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params16)
        expectEqual(proof16.innerProductProof.L.count, 4, "16-bit proof has log2(16)=4 rounds")

        let params32 = BulletproofsParams.generate(n: 32)
        let (_, proof32) = BulletproofsProver.prove(value: 100, gamma: gamma, params: params32)
        expectEqual(proof32.innerProductProof.L.count, 5, "32-bit proof has log2(32)=5 rounds")
    }
}
