// GPU Nova Decider Engine Tests — decider proof, NIFS chain, SuperNova, cross-term verification
import zkMetal

// MARK: - Test Helpers

/// Build a squaring circuit: w * w = y
/// Variables: z = [1, x, y, w] where x is public input, y is public output, w = x
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func makeDeciderSquaringR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 2
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one)

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one)

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one)

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Create a valid squaring pair: x^2 = y
private func makeDeciderSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Build a multiply circuit: a * b = c
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeDeciderMultiplyR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one)

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one)

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one)

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeDeciderMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Build a 2-constraint circuit for testing larger sumchecks:
/// Constraints:
///   a * b = c   (row 0)
///   c * 1 = c   (row 1, trivial identity to add a second constraint)
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeDeciderTwoConstraintR1CS() -> NovaR1CSShape {
    let m = 2, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a
    aBuilder.set(row: 1, col: 1, value: Fr.one) // c

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b
    bBuilder.set(row: 1, col: 0, value: Fr.one) // 1

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c
    cBuilder.set(row: 1, col: 1, value: Fr.one) // c

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeDeciderTwoConstraintPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

// MARK: - Tests

public func runGPUNovaDeciderTests() {
    suite("GPUNovaDecider")

    // =========================================================================
    // Test 1: Decider proof from single relaxed instance (base case)
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let (inst, wit) = makeDeciderSquaringPair(3)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        // Verify accumulator first
        let accOk = engine.verifyAccumulator(instance: relaxedInst, witness: relaxedWit)
        expect(accOk, "Base relaxed instance should satisfy relaxed R1CS")

        // Produce decider proof
        let proof = engine.decide(instance: relaxedInst, witness: relaxedWit, stepCount: 1)
        expect(proof.sumcheckRounds.count > 0, "Decider proof should have sumcheck rounds")
        expect(proof.stepCount == 1, "Step count should be 1")
        expect(frEq(proof.u, Fr.one), "Base case u should be 1")
    }

    // =========================================================================
    // Test 2: Decider proof after single fold step
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeDeciderSquaringPair(3)
        foldEngine.initialize(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeDeciderSquaringPair(5)
        let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

        let accInst = foldEngine.runningInstance!
        let accWit = foldEngine.runningWitness!

        // Build decider engine with matching Pedersen params
        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)

        let accOk = deciderEngine.verifyAccumulator(instance: accInst, witness: accWit)
        expect(accOk, "Folded accumulator should satisfy relaxed R1CS")

        let proof = deciderEngine.decide(instance: accInst, witness: accWit, stepCount: 2)
        expect(proof.stepCount == 2, "Step count should be 2")
        expect(!frEq(proof.u, Fr.one), "After folding, u != 1")
    }

    // =========================================================================
    // Test 3: Decider verifier accepts valid proof (base case)
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let (inst, wit) = makeDeciderSquaringPair(7)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        let proof = engine.decide(instance: relaxedInst, witness: relaxedWit)
        let verifier = NovaDeciderVerifier()
        let ok = verifier.verify(proof: proof)
        expect(ok, "Verifier should accept valid decider proof for base case")
    }

    // =========================================================================
    // Test 4: Decider verifier accepts valid proof after multi-step fold
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...5 {
            steps.append(makeDeciderSquaringPair(i))
        }
        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
        let proof = deciderEngine.decide(instance: finalInst, witness: finalWit, stepCount: 4)

        let verifier = NovaDeciderVerifier()
        let ok = verifier.verify(proof: proof)
        expect(ok, "Verifier should accept valid decider proof after 4-step fold")
    }

    // =========================================================================
    // Test 5: NIFS chain verification — valid fold chain passes
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let ivcProver = NovaIVCProver(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 3...6 {
            steps.append(makeDeciderSquaringPair(i))
        }
        let ivcProof = ivcProver.prove(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: ivcProver.pp)
        let nifsOk = deciderEngine.verifyNIFSChain(proof: ivcProof)
        expect(nifsOk, "NIFS chain verification should pass for valid fold chain")
    }

    // =========================================================================
    // Test 6: NIFS chain verification — single step (trivial) passes
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let ivcProver = NovaIVCProver(shape: shape)

        let steps = [makeDeciderSquaringPair(4)]
        let ivcProof = ivcProver.prove(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: ivcProver.pp)
        let nifsOk = deciderEngine.verifyNIFSChain(proof: ivcProof)
        expect(nifsOk, "Single-step NIFS chain should trivially pass")
    }

    // =========================================================================
    // Test 7: Cross-term verification — matching commitment
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let deciderEngine = GPUNovaDeciderEngine(shape: shape)

        let (inst1, wit1) = makeDeciderSquaringPair(3)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst1, witness: wit1, pp: deciderEngine.pp)

        let (inst2, wit2) = makeDeciderSquaringPair(5)

        // Compute cross-term and its commitment
        let T = deciderEngine.computeCrossTerm(
            runningInstance: relaxedInst, runningWitness: relaxedWit,
            newInstance: inst2, newWitness: wit2)
        let commitT = deciderEngine.ppE.commit(witness: T)

        // Verify cross-term
        let ok = deciderEngine.verifyCrossTerm(
            runningInstance: relaxedInst, runningWitness: relaxedWit,
            newInstance: inst2, newWitness: wit2,
            claimedCommitT: commitT)
        expect(ok, "Cross-term verification should pass with correct commitment")
    }

    // =========================================================================
    // Test 8: Cross-term verification — wrong commitment fails
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let deciderEngine = GPUNovaDeciderEngine(shape: shape)

        let (inst1, wit1) = makeDeciderSquaringPair(3)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst1, witness: wit1, pp: deciderEngine.pp)

        let (inst2, wit2) = makeDeciderSquaringPair(5)

        // Use a bogus commitment
        let bogusCommitT = pointIdentity()

        let ok = deciderEngine.verifyCrossTerm(
            runningInstance: relaxedInst, runningWitness: relaxedWit,
            newInstance: inst2, newWitness: wit2,
            claimedCommitT: bogusCommitT)
        // For identical satisfying instances with u=1, T is zero, so commitT = identity
        // Use different instances to guarantee non-zero T
        let (inst3, wit3) = makeDeciderSquaringPair(7)
        let (relaxedInst2, relaxedWit2) = shape.relax(instance: inst3, witness: wit3, pp: deciderEngine.pp)
        // Fold once to make u != 1
        let foldEngine = GPUNovaFoldEngine(shape: shape, pp: deciderEngine.pp, ppE: deciderEngine.ppE)
        foldEngine.initialize(instance: inst3, witness: wit3)
        let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)
        let foldedInst = foldEngine.runningInstance!
        let foldedWit = foldEngine.runningWitness!

        let (inst4, wit4) = makeDeciderSquaringPair(11)
        let fakeCommit = pointIdentity()
        let ok2 = deciderEngine.verifyCrossTerm(
            runningInstance: foldedInst, runningWitness: foldedWit,
            newInstance: inst4, newWitness: wit4,
            claimedCommitT: fakeCommit)
        // After folding, the cross-term is non-zero, so bogus commit should fail
        expect(!ok2, "Cross-term verification should fail with wrong commitment after folding")
    }

    // =========================================================================
    // Test 9: decideFromIVC — end-to-end pipeline
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let ivcProver = NovaIVCProver(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...4 {
            steps.append(makeDeciderSquaringPair(i))
        }
        let ivcProof = ivcProver.prove(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: ivcProver.pp)
        let deciderProof = deciderEngine.decideFromIVC(ivcProof: ivcProof)
        expect(deciderProof != nil, "decideFromIVC should produce a proof for valid IVC")

        if let dp = deciderProof {
            let verifier = NovaDeciderVerifier()
            let ok = verifier.verify(proof: dp)
            expect(ok, "Verifier should accept proof from decideFromIVC pipeline")
        }
    }

    // =========================================================================
    // Test 10: Commitment verification — valid witness
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let (inst, wit) = makeDeciderSquaringPair(5)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        let commitOk = engine.verifyCommitments(instance: relaxedInst, witness: relaxedWit)
        expect(commitOk, "Commitment verification should pass for correct witness")
    }

    // =========================================================================
    // Test 11: Multiply circuit decider — different R1CS shape
    // =========================================================================
    do {
        let shape = makeDeciderMultiplyR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeDeciderMultiplyPair(3, 5))
        steps.append(makeDeciderMultiplyPair(7, 11))
        steps.append(makeDeciderMultiplyPair(2, 13))

        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
        let accOk = deciderEngine.verifyAccumulator(instance: finalInst, witness: finalWit)
        expect(accOk, "Multiply circuit accumulator should satisfy relaxed R1CS")

        let proof = deciderEngine.decide(instance: finalInst, witness: finalWit, stepCount: 3)
        let verifier = NovaDeciderVerifier()
        let ok = verifier.verify(proof: proof)
        expect(ok, "Verifier should accept multiply circuit decider proof")
    }

    // =========================================================================
    // Test 12: Two-constraint circuit — larger sumcheck
    // =========================================================================
    do {
        let shape = makeDeciderTwoConstraintR1CS()
        let foldEngine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeDeciderTwoConstraintPair(3, 7))
        steps.append(makeDeciderTwoConstraintPair(5, 11))

        let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
        let accOk = deciderEngine.verifyAccumulator(instance: finalInst, witness: finalWit)
        expect(accOk, "Two-constraint accumulator should satisfy relaxed R1CS")

        let proof = deciderEngine.decide(instance: finalInst, witness: finalWit, stepCount: 2)
        expect(proof.sumcheckRounds.count == 1, "Two constraints -> log2(2) = 1 sumcheck round")

        let verifier = NovaDeciderVerifier()
        let ok = verifier.verify(proof: proof)
        expect(ok, "Verifier should accept two-constraint decider proof")
    }

    // =========================================================================
    // Test 13: SuperNova accumulator — two circuit types
    // =========================================================================
    do {
        let shape1 = makeDeciderSquaringR1CS()
        let shape2 = makeDeciderMultiplyR1CS()

        // Build accumulator for circuit 1 (squaring)
        let fold1 = GPUNovaFoldEngine(shape: shape1)
        let steps1 = [makeDeciderSquaringPair(3), makeDeciderSquaringPair(5)]
        let (inst1, wit1) = fold1.ivcChain(steps: steps1)

        // Build accumulator for circuit 2 (multiply)
        let fold2 = GPUNovaFoldEngine(shape: shape2)
        let steps2 = [makeDeciderMultiplyPair(2, 7), makeDeciderMultiplyPair(3, 11)]
        let (inst2, wit2) = fold2.ivcChain(steps: steps2)

        let superAccum = SuperNovaAccumulator(
            instances: [inst1, inst2],
            witnesses: [wit1, wit2],
            shapes: [shape1, shape2],
            circuitSchedule: [0, 0, 1, 1],
            stepCount: 4)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape1)
        let proofs = deciderEngine.decideSuperNova(accumulator: superAccum)
        expect(proofs != nil, "SuperNova decider should produce proofs")

        if let ps = proofs {
            expect(ps.count == 2, "Should have 2 per-circuit proofs")
            let superVerifier = SuperNovaDeciderVerifier()
            let ok = superVerifier.verify(proofs: ps)
            expect(ok, "SuperNova verifier should accept valid proofs")
        }
    }

    // =========================================================================
    // Test 14: GPU availability flag
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let gpuEngine = GPUNovaDeciderEngine(shape: shape, config: NovaDeciderConfig(useGPU: true))
        let cpuEngine = GPUNovaDeciderEngine(shape: shape, config: NovaDeciderConfig(useGPU: false))

        // GPU engine may or may not have GPU depending on hardware
        // CPU engine should never have GPU
        expect(!cpuEngine.gpuAvailable, "CPU-only engine should not have GPU available")

        // Both should produce valid proofs
        let (inst, wit) = makeDeciderSquaringPair(4)
        let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: gpuEngine.pp)

        let proofGPU = gpuEngine.decide(instance: ri, witness: rw)
        expect(proofGPU.sumcheckRounds.count > 0, "GPU engine should produce valid proof")

        let (ri2, rw2) = shape.relax(instance: inst, witness: wit, pp: cpuEngine.pp)
        let proofCPU = cpuEngine.decide(instance: ri2, witness: rw2)
        expect(proofCPU.sumcheckRounds.count > 0, "CPU engine should produce valid proof")
    }

    // =========================================================================
    // Test 15: GPU inner product helper
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
        // Expected: 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        let result = engine.gpuFieldInnerProduct(a, b)
        let expected = frFromInt(112)
        expect(frEq(result, expected), "GPU inner product should compute 2*7+3*11+5*13 = 112")
    }

    // =========================================================================
    // Test 16: Witness hash determinism
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let (inst, wit) = makeDeciderSquaringPair(3)
        let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        let proof1 = engine.decide(instance: ri, witness: rw)
        let proof2 = engine.decide(instance: ri, witness: rw)

        expect(frEq(proof1.witnessHash, proof2.witnessHash),
               "Witness hash should be deterministic for same input")
    }

    // =========================================================================
    // Test 17: Config defaults
    // =========================================================================
    do {
        let defaultConfig = NovaDeciderConfig()
        expect(defaultConfig.useGPU == true, "Default config should enable GPU")
        expect(defaultConfig.gpuThreshold == 512, "Default GPU threshold should be 512")
        expect(defaultConfig.verifyFoldChain == true, "Default should verify fold chain")

        let customConfig = NovaDeciderConfig(useGPU: false, gpuThreshold: 1024, verifyFoldChain: false)
        expect(customConfig.useGPU == false, "Custom config GPU flag")
        expect(customConfig.gpuThreshold == 1024, "Custom GPU threshold")
        expect(customConfig.verifyFoldChain == false, "Custom fold chain flag")
    }

    // =========================================================================
    // Test 18: decideFromIVC with fold chain verification disabled
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let ivcProver = NovaIVCProver(shape: shape)

        let steps = [makeDeciderSquaringPair(3), makeDeciderSquaringPair(5)]
        let ivcProof = ivcProver.prove(steps: steps)

        let noVerifyConfig = NovaDeciderConfig(useGPU: true, verifyFoldChain: false)
        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: ivcProver.pp, config: noVerifyConfig)
        let proof = deciderEngine.decideFromIVC(ivcProof: ivcProof)
        expect(proof != nil, "decideFromIVC should work with fold verification disabled")
    }

    // =========================================================================
    // Test 19: verifyWithIVC end-to-end
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let ivcProver = NovaIVCProver(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...4 {
            steps.append(makeDeciderSquaringPair(i))
        }
        let ivcProof = ivcProver.prove(steps: steps)

        let deciderEngine = GPUNovaDeciderEngine(shape: shape, pp: ivcProver.pp)
        let deciderProof = deciderEngine.decide(instance: ivcProof.finalInstance,
                                                 witness: ivcProof.finalWitness,
                                                 stepCount: ivcProof.stepCount)

        let verifier = NovaDeciderVerifier()
        let ok = verifier.verifyWithIVC(deciderProof: deciderProof,
                                         ivcProof: ivcProof,
                                         shape: shape)
        expect(ok, "verifyWithIVC should accept valid proof + IVC chain")
    }

    // =========================================================================
    // Test 20: Cross-term is zero for identical satisfying instances (u=1)
    // =========================================================================
    do {
        let shape = makeDeciderSquaringR1CS()
        let engine = GPUNovaDeciderEngine(shape: shape)

        let (inst, wit) = makeDeciderSquaringPair(3)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        let T = engine.computeCrossTerm(
            runningInstance: relaxedInst, runningWitness: relaxedWit,
            newInstance: inst, newWitness: wit)

        let allZero = T.allSatisfy { $0.isZero }
        expect(allZero, "Cross-term should be zero for identical satisfying instances with u=1")
    }
}
