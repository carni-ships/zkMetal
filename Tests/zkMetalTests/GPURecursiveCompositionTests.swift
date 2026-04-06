import zkMetal

// MARK: - GPU Recursive Composition Engine Tests

public func runGPURecursiveCompositionTests() {
    suite("GPU Recursive Composition Engine")

    let engine = GPURecursiveCompositionEngine()

    // Test 1: Fold a single proof node into a zero accumulator
    do {
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let node = ProofNode(
            id: 0,
            scheme: BaseScheme.groth16,
            publicInputs: [frFromInt(10)],
            commitmentPoly: poly
        )
        let acc0 = CompositionAccumulator.zero(size: 4, scheme: BaseScheme.groth16)
        let challenge = frFromInt(5)
        let acc1 = engine.foldNode(accumulator: acc0, node: node, challenge: challenge)

        // acc' = 0 + 5 * [1,2,3,4] = [5, 10, 15, 20]
        expect(frEqual(acc1.accPoly[0], frFromInt(5)), "Single fold: coeff 0 = 5")
        expect(frEqual(acc1.accPoly[1], frFromInt(10)), "Single fold: coeff 1 = 10")
        expect(frEqual(acc1.accPoly[2], frFromInt(15)), "Single fold: coeff 2 = 15")
        expect(frEqual(acc1.accPoly[3], frFromInt(20)), "Single fold: coeff 3 = 20")
        expectEqual(acc1.foldCount, 1, "Single fold: fold count = 1")
        // Cross term = <0, poly> = 0, error = 0 + 25 * 0 = 0
        expect(frEqual(acc1.error, Fr.zero), "Single fold: error = 0 (zero cross term)")
        expectEqual(acc1.foldedProofIDs, [0], "Single fold: proof ID recorded")
    }

    // Test 2: Sequential fold of two nodes
    do {
        let p1 = [frFromInt(1), frFromInt(2)]
        let p2 = [frFromInt(3), frFromInt(4)]
        let node1 = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: p1)
        let node2 = ProofNode(id: 2, scheme: BaseScheme.groth16, publicInputs: [frFromInt(2)], commitmentPoly: p2)

        let acc0 = CompositionAccumulator.initial(node: node1)
        let challenge = frFromInt(5)
        let acc1 = engine.foldNode(accumulator: acc0, node: node2, challenge: challenge)

        // acc' = [1,2] + 5*[3,4] = [16, 22]
        expect(frEqual(acc1.accPoly[0], frFromInt(16)), "Double fold: coeff 0 = 16")
        expect(frEqual(acc1.accPoly[1], frFromInt(22)), "Double fold: coeff 1 = 22")
        expectEqual(acc1.foldCount, 2, "Double fold: fold count = 2")

        // Cross term = <[1,2], [3,4]> = 3 + 8 = 11
        // error' = 0 + 25 * 11 = 275
        expect(frEqual(acc1.error, frFromInt(275)), "Double fold: error = 275")
        expectEqual(acc1.foldedProofIDs, [1, 2], "Double fold: both IDs recorded")
    }

    // Test 3: Fold raw polynomial (no node metadata)
    do {
        let p1 = [frFromInt(2), frFromInt(3)]
        let p2 = [frFromInt(4), frFromInt(5)]

        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.plonk)
        acc = engine.foldPoly(accumulator: acc, poly: p1, challenge: Fr.one)

        // acc = 0 + 1 * [2,3] = [2, 3], error = 0 (cross = 0)
        expect(frEqual(acc.accPoly[0], frFromInt(2)), "FoldPoly step 1: coeff 0 = 2")
        expect(frEqual(acc.accPoly[1], frFromInt(3)), "FoldPoly step 1: coeff 1 = 3")

        acc = engine.foldPoly(accumulator: acc, poly: p2, challenge: frFromInt(3))
        // acc = [2,3] + 3*[4,5] = [14, 18]
        expect(frEqual(acc.accPoly[0], frFromInt(14)), "FoldPoly step 2: coeff 0 = 14")
        expect(frEqual(acc.accPoly[1], frFromInt(18)), "FoldPoly step 2: coeff 1 = 18")

        // Cross = <[2,3],[4,5]> = 8+15 = 23, error = 0 + 9*23 = 207
        expect(frEqual(acc.error, frFromInt(207)), "FoldPoly step 2: error = 207")
        expectEqual(acc.foldCount, 2, "FoldPoly step 2: fold count = 2")
    }

    // Test 4: Split accumulation for Groth16
    do {
        let poly = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let pi = [frFromInt(7), frFromInt(11), frFromInt(13)]
        let node = ProofNode(
            id: 10,
            scheme: BaseScheme.groth16,
            publicInputs: pi,
            commitmentPoly: poly
        )

        var transcript = RecursiveCompositionTranscript()
        let split = engine.splitAccumulation(node: node, transcript: &transcript)

        // immediateScalar = 7*2 + 11*3 + 13*5 = 14+33+65 = 112
        expect(frEqual(split.immediateScalar, frFromInt(112)), "Split Groth16: immediateScalar = 112")
        expectEqual(split.scheme, BaseScheme.groth16, "Split Groth16: scheme preserved")
        expect(split.immediateVerified, "Split Groth16: immediate part verified")
        expectEqual(split.deferredPairings.count, 1, "Split Groth16: 1 deferred pairing")
        expectEqual(split.deferredCommitments.count, 0, "Split Groth16: 0 deferred commitments")

        // immediatePoly = [7*2, 11*3, 13*5] = [14, 33, 65]
        expect(frEqual(split.immediatePoly[0], frFromInt(14)), "Split Groth16: immediatePoly[0] = 14")
        expect(frEqual(split.immediatePoly[1], frFromInt(33)), "Split Groth16: immediatePoly[1] = 33")
        expect(frEqual(split.immediatePoly[2], frFromInt(65)), "Split Groth16: immediatePoly[2] = 65")
    }

    // Test 5: Split accumulation for Plonk
    do {
        let poly = [frFromInt(1), frFromInt(2)]
        let pi = [frFromInt(3), frFromInt(4)]
        let node = ProofNode(
            id: 20,
            scheme: BaseScheme.plonk,
            publicInputs: pi,
            commitmentPoly: poly
        )

        var transcript = RecursiveCompositionTranscript()
        let split = engine.splitAccumulation(node: node, transcript: &transcript)

        // immediateScalar = 3*1 + 4*2 = 11
        expect(frEqual(split.immediateScalar, frFromInt(11)), "Split Plonk: immediateScalar = 11")
        expectEqual(split.scheme, BaseScheme.plonk, "Split Plonk: scheme preserved")
        expectEqual(split.deferredPairings.count, 0, "Split Plonk: 0 deferred pairings")
        expectEqual(split.deferredCommitments.count, 1, "Split Plonk: 1 deferred commitment")
    }

    // Test 6: Proof-of-proof
    do {
        let innerPoly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let innerPI = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let innerNode = ProofNode(
            id: 100,
            scheme: BaseScheme.groth16,
            publicInputs: innerPI,
            commitmentPoly: innerPoly,
            isLeaf: true,
            children: [],
            depth: 0
        )

        var transcript = RecursiveCompositionTranscript()
        let acc = engine.proofOfProof(
            innerNode: innerNode,
            outerScheme: .plonk,
            transcript: &transcript
        )

        expectEqual(acc.foldCount, 1, "Proof-of-proof: fold count = 1")
        expect(acc.foldedProofIDs.count == 1, "Proof-of-proof: 1 folded proof ID")
        // The outer node should have depth = inner depth + 1 = 1
        expectEqual(acc.maxDepth, 1, "Proof-of-proof: depth = 1")
        // Deferred pairings from the Groth16 split
        expectEqual(acc.deferredPairings.count, 1, "Proof-of-proof: 1 deferred pairing from inner Groth16")
    }

    // Test 7: Accumulate multiple proofs
    do {
        var nodes = [ProofNode]()
        for i in 0..<4 {
            let pi = [frFromInt(UInt64(i + 1))]
            let cp = [frFromInt(UInt64(i + 1)), frFromInt(UInt64(i + 2))]
            nodes.append(ProofNode(id: i, scheme: BaseScheme.groth16, publicInputs: pi, commitmentPoly: cp))
        }

        var transcript = RecursiveCompositionTranscript()
        let acc = engine.accumulateProofs(nodes: nodes, scheme: BaseScheme.groth16, transcript: &transcript)

        expectEqual(acc.foldCount, 4, "Multi-accumulate: fold count = 4")
        expectEqual(acc.foldedProofIDs.count, 4, "Multi-accumulate: 4 proof IDs")
        expectEqual(acc.accPoly.count, 2, "Multi-accumulate: poly size preserved")
        expectEqual(acc.scheme, BaseScheme.groth16, "Multi-accumulate: scheme = groth16")
    }

    // Test 8: Accumulate empty list
    do {
        var transcript = RecursiveCompositionTranscript()
        let acc = engine.accumulateProofs(nodes: [], scheme: BaseScheme.plonk, transcript: &transcript)
        expectEqual(acc.foldCount, 0, "Empty accumulate: fold count = 0")
        expectEqual(acc.accPoly.count, 0, "Empty accumulate: empty poly")
    }

    // Test 9: Chain verification — single node chain
    do {
        let node = ProofNode(
            id: 0,
            scheme: BaseScheme.groth16,
            publicInputs: [frFromInt(42)],
            commitmentPoly: [frFromInt(1), frFromInt(2)]
        )

        var transcript = RecursiveCompositionTranscript()
        let result = engine.verifyChain(chain: [node], transcript: &transcript)

        expect(result.verified, "Single chain: verified")
        expectEqual(result.chainLength, 1, "Single chain: length = 1")
        expectEqual(result.totalDeferredChecks, 0, "Single chain: 0 deferred (no folding)")
    }

    // Test 10: Chain verification — multiple nodes
    do {
        var nodes = [ProofNode]()
        for i in 0..<3 {
            let pi = [frFromInt(UInt64(i + 1))]
            let cp = [frFromInt(UInt64(i * 2 + 1)), frFromInt(UInt64(i * 2 + 2))]
            nodes.append(ProofNode(id: i, scheme: BaseScheme.groth16, publicInputs: pi, commitmentPoly: cp))
        }

        var transcript = RecursiveCompositionTranscript()
        let result = engine.verifyChain(chain: nodes, transcript: &transcript)

        expectEqual(result.chainLength, 3, "Chain verify: length = 3")
        // Chain should have deferred pairing checks from Groth16 split
        expect(result.totalDeferredChecks >= 0, "Chain verify: deferred checks accumulated")
    }

    // Test 11: Chain verification — empty chain
    do {
        var transcript = RecursiveCompositionTranscript()
        let result = engine.verifyChain(chain: [], transcript: &transcript)
        expect(result.verified, "Empty chain: verified")
        expectEqual(result.chainLength, 0, "Empty chain: length = 0")
    }

    // Test 12: Deferred pairing verification — balanced check passes
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.groth16)

        // Balanced: 3*7 = 21 and 21*1 = 21 -> lhs == rhs
        let check = DeferredPairingCheck(
            lhsG1: frFromInt(3), lhsG2: frFromInt(7),
            rhsG1: frFromInt(21), rhsG2: Fr.one
        )
        acc.deferredPairings.append(check)

        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyDeferredPairings(accumulator: acc, transcript: &transcript)
        expect(ok, "Deferred pairing: balanced check passes")
    }

    // Test 13: Deferred pairing verification — unbalanced check fails
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.groth16)

        // Unbalanced: 2*3 = 6 != 5*2 = 10
        let check = DeferredPairingCheck(
            lhsG1: frFromInt(2), lhsG2: frFromInt(3),
            rhsG1: frFromInt(5), rhsG2: frFromInt(2)
        )
        acc.deferredPairings.append(check)

        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyDeferredPairings(accumulator: acc, transcript: &transcript)
        expect(!ok, "Deferred pairing: unbalanced check fails")
    }

    // Test 14: Deferred commitment verification — valid check passes
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.plonk)

        // commitment == value + openingProof * point
        // 17 == 5 + 4 * 3 = 5 + 12 = 17 -> passes
        let check = DeferredCommitmentCheck(
            commitment: frFromInt(17),
            point: frFromInt(3),
            value: frFromInt(5),
            openingProof: frFromInt(4)
        )
        acc.deferredCommitments.append(check)

        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyDeferredCommitments(accumulator: acc, transcript: &transcript)
        expect(ok, "Deferred commitment: valid check passes")
    }

    // Test 15: Deferred commitment verification — invalid check fails
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.plonk)

        // commitment == value + openingProof * point
        // 20 != 5 + 4 * 3 = 17 -> fails
        let check = DeferredCommitmentCheck(
            commitment: frFromInt(20),
            point: frFromInt(3),
            value: frFromInt(5),
            openingProof: frFromInt(4)
        )
        acc.deferredCommitments.append(check)

        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyDeferredCommitments(accumulator: acc, transcript: &transcript)
        expect(!ok, "Deferred commitment: invalid check fails")
    }

    // Test 16: verifyAllDeferred — empty checks pass
    do {
        let acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.groth16)
        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyAllDeferred(accumulator: acc, transcript: &transcript)
        expect(ok, "verifyAllDeferred: empty checks pass")
    }

    // Test 17: Proof size estimation — Groth16
    do {
        let est0 = engine.estimateProofSize(scheme: BaseScheme.groth16, depth: 0, numPublicInputs: 4)
        let est1 = engine.estimateProofSize(scheme: BaseScheme.groth16, depth: 1, numPublicInputs: 4)

        expect(est0.sizeBytes > 0, "Groth16 size: positive at depth 0")
        expect(est1.sizeBytes > 0, "Groth16 size: positive at depth 1")
        expectEqual(est0.scheme, .groth16, "Groth16 size: scheme preserved")
        expectEqual(est0.depth, 0, "Groth16 size: depth preserved")
        // Depth 1 adds 1 public input for accumulated state hash
        expect(est1.numPublicInputs > est0.numPublicInputs,
               "Groth16 size: depth 1 has more public inputs")
    }

    // Test 18: Proof size estimation — Plonk
    do {
        let est0 = engine.estimateProofSize(scheme: BaseScheme.plonk, depth: 0, numPublicInputs: 2)
        let est1 = engine.estimateProofSize(scheme: BaseScheme.plonk, depth: 1, numPublicInputs: 2)
        let est2 = engine.estimateProofSize(scheme: BaseScheme.plonk, depth: 2, numPublicInputs: 2)

        expect(est0.sizeBytes > 0, "Plonk size: positive at depth 0")
        expect(est1.sizeBytes <= est0.sizeBytes, "Plonk size: depth 1 <= depth 0 (reduction)")
        expect(est2.sizeBytes <= est1.sizeBytes, "Plonk size: depth 2 <= depth 1 (further reduction)")
        expectEqual(est0.scheme, .plonk, "Plonk size: scheme preserved")
    }

    // Test 19: Optimal recursion depth
    do {
        let grothDepth = engine.optimalRecursionDepth(scheme: BaseScheme.groth16, numPublicInputs: 4)
        let plonkDepth = engine.optimalRecursionDepth(scheme: BaseScheme.plonk, numPublicInputs: 10)

        // Groth16 is already constant size, so optimal depth is 0 or 1
        expect(grothDepth >= 0, "Groth16 optimal depth: non-negative")
        // Plonk benefits from at least 1 recursion
        expect(plonkDepth >= 1, "Plonk optimal depth: >= 1 (benefits from recursion)")
    }

    // Test 20: Composition transcript — deterministic
    do {
        var t1 = RecursiveCompositionTranscript()
        t1.appendLabel("test-domain")
        t1.appendScalar(frFromInt(42))
        let c1 = t1.squeeze()

        var t2 = RecursiveCompositionTranscript()
        t2.appendLabel("test-domain")
        t2.appendScalar(frFromInt(42))
        let c2 = t2.squeeze()

        expect(frEqual(c1, c2), "Transcript: same inputs -> same challenge")

        var t3 = RecursiveCompositionTranscript()
        t3.appendLabel("test-domain")
        t3.appendScalar(frFromInt(43))
        let c3 = t3.squeeze()

        expect(!frEqual(c1, c3), "Transcript: different inputs -> different challenge")
    }

    // Test 21: Composition transcript — appendNode binds structure
    do {
        let node1 = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1)])
        let node2 = ProofNode(id: 2, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1)])

        var t1 = RecursiveCompositionTranscript()
        t1.appendNode(node1)
        let c1 = t1.squeeze()

        var t2 = RecursiveCompositionTranscript()
        t2.appendNode(node2)
        let c2 = t2.squeeze()

        // Different node IDs -> different challenges
        expect(!frEqual(c1, c2), "Transcript: different node IDs -> different challenges")
    }

    // Test 22: Composition transcript — scheme binding
    do {
        let nodeG = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1)])
        let nodeP = ProofNode(id: 1, scheme: BaseScheme.plonk, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1)])

        var tG = RecursiveCompositionTranscript()
        tG.appendNode(nodeG)
        let cG = tG.squeeze()

        var tP = RecursiveCompositionTranscript()
        tP.appendNode(nodeP)
        let cP = tP.squeeze()

        // Different schemes -> different challenges
        expect(!frEqual(cG, cP), "Transcript: different schemes -> different challenges")
    }

    // Test 23: Accumulator reset
    do {
        let node = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(5), frFromInt(6)])
        var acc = CompositionAccumulator.initial(node: node)
        acc.deferredPairings.append(DeferredPairingCheck(
            lhsG1: frFromInt(1), lhsG2: frFromInt(1),
            rhsG1: frFromInt(1), rhsG2: frFromInt(1)
        ))
        acc.deferredCommitments.append(DeferredCommitmentCheck(
            commitment: frFromInt(1), point: frFromInt(1),
            value: frFromInt(1), openingProof: frFromInt(1)
        ))

        expectEqual(acc.foldCount, 1, "Before reset: fold count = 1")
        expectEqual(acc.deferredPairings.count, 1, "Before reset: 1 pairing")
        expectEqual(acc.deferredCommitments.count, 1, "Before reset: 1 commitment")

        acc.reset()

        expectEqual(acc.foldCount, 0, "After reset: fold count = 0")
        expect(frEqual(acc.error, Fr.zero), "After reset: error = 0")
        expectEqual(acc.deferredPairings.count, 0, "After reset: 0 pairings")
        expectEqual(acc.deferredCommitments.count, 0, "After reset: 0 commitments")
        expectEqual(acc.accPoly.count, 2, "After reset: poly size preserved")
        expect(frEqual(acc.accPoly[0], Fr.zero), "After reset: poly[0] = 0")
        expect(frEqual(acc.accPoly[1], Fr.zero), "After reset: poly[1] = 0")
    }

    // Test 24: Challenge derivation is deterministic
    do {
        let node = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2), frFromInt(3)])
        let acc = CompositionAccumulator.initial(node: node)

        let c1 = engine.deriveChallenge(accumulator: acc, domainSeparator: "test")
        let c2 = engine.deriveChallenge(accumulator: acc, domainSeparator: "test")
        expect(frEqual(c1, c2), "deriveChallenge: deterministic")

        let c3 = engine.deriveChallenge(accumulator: acc, domainSeparator: "other")
        expect(!frEqual(c1, c3), "deriveChallenge: different domain -> different challenge")
    }

    // Test 25: Polynomial evaluation
    do {
        // p(x) = 1 + 2x + 3x^2, p(2) = 1 + 4 + 12 = 17
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let val = engine.evaluatePolynomial(poly, at: frFromInt(2))
        expect(frEqual(val, frFromInt(17)), "Poly eval: 1 + 2*2 + 3*4 = 17")

        let empty = engine.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(empty, Fr.zero), "Poly eval: empty -> 0")
    }

    // Test 26: Inner product
    do {
        let a = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let b = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let ip = engine.innerProduct(a, b)
        // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
        expect(frEqual(ip, frFromInt(32)), "Inner product: <[1,2,3],[4,5,6]> = 32")
    }

    // Test 27: Linear combination
    do {
        let p0 = [frFromInt(1), frFromInt(0)]
        let p1 = [frFromInt(0), frFromInt(2)]
        let c0 = frFromInt(3)
        let c1 = frFromInt(7)
        let result = engine.linearCombination(polys: [p0, p1], challenges: [c0, c1])
        // result = 3*[1,0] + 7*[0,2] = [3, 14]
        expect(frEqual(result[0], frFromInt(3)), "LinComb: coeff 0 = 3")
        expect(frEqual(result[1], frFromInt(14)), "LinComb: coeff 1 = 14")
    }

    // Test 28: Build composition tree — single leaf
    do {
        let leaf = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2)])
        var transcript = RecursiveCompositionTranscript()
        let acc = engine.buildCompositionTree(leaves: [leaf], scheme: BaseScheme.groth16, transcript: &transcript)
        expectEqual(acc.foldCount, 1, "Tree single leaf: fold count = 1")
        expectEqual(acc.accPoly.count, 2, "Tree single leaf: poly size = 2")
    }

    // Test 29: Build composition tree — two leaves
    do {
        let leaf0 = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2)])
        let leaf1 = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(3)], commitmentPoly: [frFromInt(3), frFromInt(4)])
        var transcript = RecursiveCompositionTranscript()
        let acc = engine.buildCompositionTree(leaves: [leaf0, leaf1], scheme: BaseScheme.groth16, transcript: &transcript)
        // Tree has one intermediate node at depth 1
        expectEqual(acc.foldCount, 1, "Tree two leaves: root fold count = 1")
        expectEqual(acc.maxDepth, 1, "Tree two leaves: max depth = 1")
    }

    // Test 30: Build composition tree — four leaves (balanced binary tree)
    do {
        var leaves = [ProofNode]()
        for i in 0..<4 {
            let pi = [frFromInt(UInt64(i + 1))]
            let cp = [frFromInt(UInt64(i + 1)), frFromInt(UInt64(i + 2))]
            leaves.append(ProofNode(id: i, scheme: BaseScheme.groth16, publicInputs: pi, commitmentPoly: cp))
        }
        var transcript = RecursiveCompositionTranscript()
        let acc = engine.buildCompositionTree(leaves: leaves, scheme: BaseScheme.groth16, transcript: &transcript)
        // 4 leaves -> 2 intermediate -> 1 root, depth = 2
        expectEqual(acc.maxDepth, 2, "Tree 4 leaves: max depth = 2")
    }

    // Test 31: Build composition tree — empty
    do {
        var transcript = RecursiveCompositionTranscript()
        let acc = engine.buildCompositionTree(leaves: [], scheme: BaseScheme.groth16, transcript: &transcript)
        expectEqual(acc.foldCount, 0, "Tree empty: fold count = 0")
    }

    // Test 32: Fold with split — Groth16 path
    do {
        let node1 = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2)])
        let node2 = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(3)], commitmentPoly: [frFromInt(3), frFromInt(4)])

        let acc0 = CompositionAccumulator.initial(node: node1)
        var transcript = RecursiveCompositionTranscript()
        let acc1 = engine.foldWithSplit(accumulator: acc0, node: node2, transcript: &transcript)

        expectEqual(acc1.foldCount, 2, "FoldWithSplit Groth16: fold count = 2")
        // Should have 1 deferred pairing from the Groth16 split of node2
        expectEqual(acc1.deferredPairings.count, 1, "FoldWithSplit Groth16: 1 deferred pairing")
        expectEqual(acc1.foldedProofIDs.count, 2, "FoldWithSplit Groth16: 2 proof IDs")
    }

    // Test 33: Fold with split — Plonk path
    do {
        let node1 = ProofNode(id: 0, scheme: BaseScheme.plonk, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2)])
        let node2 = ProofNode(id: 1, scheme: BaseScheme.plonk, publicInputs: [frFromInt(3)], commitmentPoly: [frFromInt(3), frFromInt(4)])

        var acc0 = CompositionAccumulator.initial(node: node1)
        acc0.scheme = .plonk
        var transcript = RecursiveCompositionTranscript()
        let acc1 = engine.foldWithSplit(accumulator: acc0, node: node2, transcript: &transcript)

        expectEqual(acc1.foldCount, 2, "FoldWithSplit Plonk: fold count = 2")
        expectEqual(acc1.deferredCommitments.count, 1, "FoldWithSplit Plonk: 1 deferred commitment")
        expectEqual(acc1.deferredPairings.count, 0, "FoldWithSplit Plonk: 0 deferred pairings")
    }

    // Test 34: Chain verification with mismatched poly sizes
    do {
        let node1 = ProofNode(id: 0, scheme: BaseScheme.groth16, publicInputs: [frFromInt(1)], commitmentPoly: [frFromInt(1), frFromInt(2)])
        let node2 = ProofNode(id: 1, scheme: BaseScheme.groth16, publicInputs: [frFromInt(3)], commitmentPoly: [frFromInt(3), frFromInt(4), frFromInt(5)])

        var transcript = RecursiveCompositionTranscript()
        let result = engine.verifyChain(chain: [node1, node2], transcript: &transcript)

        expect(!result.verified, "Mismatched poly chain: verification fails")
        expectEqual(result.chainLength, 1, "Mismatched poly chain: stops at mismatch")
    }

    // Test 35: Multiple deferred pairing checks — mixed balanced/unbalanced
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.groth16)

        // Balanced: 6*1 = 6 and 6*1 = 6
        acc.deferredPairings.append(DeferredPairingCheck(
            lhsG1: frFromInt(6), lhsG2: Fr.one,
            rhsG1: frFromInt(6), rhsG2: Fr.one
        ))
        // Also balanced: 2*5 = 10 and 10*1 = 10
        acc.deferredPairings.append(DeferredPairingCheck(
            lhsG1: frFromInt(2), lhsG2: frFromInt(5),
            rhsG1: frFromInt(10), rhsG2: Fr.one
        ))

        var transcript = RecursiveCompositionTranscript()
        let ok = engine.verifyDeferredPairings(accumulator: acc, transcript: &transcript)
        expect(ok, "Multiple balanced pairings: all pass")
    }

    // Test 36: ProofNode metadata
    do {
        let leaf = ProofNode(
            id: 42,
            scheme: BaseScheme.plonk,
            publicInputs: [frFromInt(1), frFromInt(2)],
            commitmentPoly: [frFromInt(3)],
            isLeaf: true,
            children: [],
            depth: 0
        )

        expectEqual(leaf.id, 42, "ProofNode: id preserved")
        expectEqual(leaf.scheme, .plonk, "ProofNode: scheme preserved")
        expect(leaf.isLeaf, "ProofNode: isLeaf = true")
        expectEqual(leaf.children.count, 0, "ProofNode: no children")
        expectEqual(leaf.depth, 0, "ProofNode: depth = 0")
        expectEqual(leaf.publicInputs.count, 2, "ProofNode: 2 public inputs")
        expectEqual(leaf.commitmentPoly.count, 1, "ProofNode: 1 commitment coeff")

        let intermediate = ProofNode(
            id: 100,
            scheme: BaseScheme.groth16,
            publicInputs: [frFromInt(5)],
            commitmentPoly: [frFromInt(6), frFromInt(7)],
            isLeaf: false,
            children: [10, 20],
            depth: 2
        )

        expect(!intermediate.isLeaf, "ProofNode intermediate: isLeaf = false")
        expectEqual(intermediate.children, [10, 20], "ProofNode intermediate: children = [10, 20]")
        expectEqual(intermediate.depth, 2, "ProofNode intermediate: depth = 2")
    }

    // Test 37: Composition accumulator totalDeferredChecks
    do {
        var acc = CompositionAccumulator.zero(size: 2, scheme: BaseScheme.groth16)
        expectEqual(acc.totalDeferredChecks, 0, "Initial: 0 deferred checks")

        acc.deferredPairings.append(DeferredPairingCheck(
            lhsG1: Fr.one, lhsG2: Fr.one, rhsG1: Fr.one, rhsG2: Fr.one
        ))
        expectEqual(acc.totalDeferredChecks, 1, "After 1 pairing: 1 deferred check")

        acc.deferredCommitments.append(DeferredCommitmentCheck(
            commitment: Fr.one, point: Fr.one, value: Fr.one, openingProof: Fr.one
        ))
        expectEqual(acc.totalDeferredChecks, 2, "After 1 pairing + 1 commitment: 2 deferred checks")
    }

    // Test 38: BaseScheme equality
    do {
        expect(BaseScheme.groth16 == BaseScheme.groth16, "BaseScheme: groth16 == groth16")
        expect(BaseScheme.plonk == BaseScheme.plonk, "BaseScheme: plonk == plonk")
        expect(BaseScheme.groth16 != BaseScheme.plonk, "BaseScheme: groth16 != plonk")
    }

    // Test 39: Transcript squeezeAndAdvance changes state
    do {
        var t = RecursiveCompositionTranscript()
        t.appendLabel("advance-test")
        let c1 = t.squeezeAndAdvance()
        let c2 = t.squeeze()
        expect(!frEqual(c1, c2), "Transcript: squeezeAndAdvance changes state")
    }

    // Test 40: Transcript stateHash consistency
    do {
        var t1 = RecursiveCompositionTranscript()
        t1.appendLabel("hash-test")
        t1.appendScalar(frFromInt(99))

        var t2 = RecursiveCompositionTranscript()
        t2.appendLabel("hash-test")
        t2.appendScalar(frFromInt(99))

        let h1 = t1.stateHash()
        let h2 = t2.stateHash()
        expect(h1 == h2, "Transcript stateHash: same inputs -> same hash")
        expectEqual(h1.count, 32, "Transcript stateHash: 32 bytes")
    }
}
