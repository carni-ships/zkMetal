// GPU Nova Verifier Engine Tests -- relaxed R1CS, fold chain, cross-term, state hash,
// batch verification, SuperNova multi-circuit verification
import zkMetal

// MARK: - Test Helpers

/// Build a squaring circuit: w * w = y
/// Variables: z = [1, x, y, w] where x is public input, y is public output, w = x
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func makeVerifierSquaringR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 2
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one) // y

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Create a valid squaring pair: x^2 = y
private func makeVerifierSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Build a multiply circuit: a * b = c
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeVerifierMultiplyR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeVerifierMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Build a 2-constraint circuit for testing:
///   a * b = c   (row 0)
///   c * 1 = c   (row 1, identity)
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeVerifierTwoConstraintR1CS() -> NovaR1CSShape {
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

private func makeVerifierTwoConstraintPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

// MARK: - Test Implementations

private func testRelaxedR1CSCheck() {
    // Base case: relaxed instance with u=1, E=0 should satisfy relaxed R1CS
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst, wit) = makeVerifierSquaringPair(3)
    let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    let ok = engine.verifyRelaxedR1CS(instance: relaxedInst, witness: relaxedWit)
    expect(ok, "Relaxed instance with u=1, E=0 should satisfy relaxed R1CS")
}

private func testStrictR1CSCheck() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    // Valid: 5^2 = 25
    let (inst, wit) = makeVerifierSquaringPair(5)
    let ok = engine.verifyStrictR1CS(instance: inst, witness: wit)
    expect(ok, "5^2=25 should satisfy strict R1CS")

    // Invalid: wrong witness
    let badWit = NovaR1CSWitness(W: [frFromInt(7)]) // 7^2=49 != 25
    let badInst = NovaR1CSInput(x: [frFromInt(5), frFromInt(25)])
    let notOk = engine.verifyStrictR1CS(instance: badInst, witness: badWit)
    expect(!notOk, "Wrong witness should fail strict R1CS")
}

private func testRelaxedR1CSAfterFolding() {
    // After folding, the accumulated instance should still satisfy relaxed R1CS
    let shape = makeVerifierSquaringR1CS()
    let foldEngine = GPUNovaFoldEngine(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    foldEngine.initialize(instance: inst1, witness: wit1)

    let (inst2, wit2) = makeVerifierSquaringPair(7)
    let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
    let ok = verifier.verifyRelaxedR1CS(instance: foldEngine.runningInstance!,
                                         witness: foldEngine.runningWitness!)
    expect(ok, "Folded accumulator should satisfy relaxed R1CS")
    expect(!frEq(foldEngine.runningInstance!.u, Fr.one), "After folding, u != 1")
}

private func testCommitmentVerification() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst, wit) = makeVerifierSquaringPair(4)
    let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    let ok = engine.verifyCommitments(instance: relaxedInst, witness: relaxedWit)
    expect(ok, "Commitment verification should pass for valid witness")
}

private func testCrossTermZeroForIdentical() {
    // Cross-term should be zero when both instances are identical satisfying instances with u=1
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst, wit) = makeVerifierSquaringPair(3)
    let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    let T = engine.recomputeCrossTerm(
        runningInstance: relaxedInst, runningWitness: relaxedWit,
        newInstance: inst, newWitness: wit)

    let allZero = T.allSatisfy { $0.isZero }
    expect(allZero, "Cross-term should be zero for identical satisfying instances with u=1")
}

private func testCrossTermCommitmentValid() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    let (relaxedInst, relaxedWit) = shape.relax(instance: inst1, witness: wit1, pp: engine.pp)

    let (inst2, wit2) = makeVerifierSquaringPair(5)

    let T = engine.recomputeCrossTerm(
        runningInstance: relaxedInst, runningWitness: relaxedWit,
        newInstance: inst2, newWitness: wit2)
    let commitT = engine.ppE.commit(witness: T)

    let ok = engine.verifyCrossTermCommitment(
        runningInstance: relaxedInst, runningWitness: relaxedWit,
        newInstance: inst2, newWitness: wit2,
        claimedCommitT: commitT)
    expect(ok, "Cross-term commitment should verify with correct commitment")
}

private func testCrossTermCommitmentInvalid() {
    // After folding (u != 1), a bogus commitment should fail
    let shape = makeVerifierSquaringR1CS()
    let foldEngine = GPUNovaFoldEngine(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    foldEngine.initialize(instance: inst1, witness: wit1)

    let (inst2, wit2) = makeVerifierSquaringPair(5)
    let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)

    let (inst3, wit3) = makeVerifierSquaringPair(11)
    let bogusCommitT = pointIdentity()
    let ok = verifier.verifyCrossTermCommitment(
        runningInstance: foldEngine.runningInstance!,
        runningWitness: foldEngine.runningWitness!,
        newInstance: inst3, newWitness: wit3,
        claimedCommitT: bogusCommitT)
    expect(!ok, "Cross-term commitment should fail with bogus commitment after folding")
}

private func testFoldStepVerification() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    let steps = [makeVerifierSquaringPair(3), makeVerifierSquaringPair(5)]
    let ivcProof = ivcProver.prove(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)

    // Verify the single fold step
    let running = ivcProof.intermediateInstances[0]
    let fresh = ivcProof.freshInstances[0]
    let proof = ivcProof.foldProofs[0]
    let target = ivcProof.finalInstance

    let ok = verifier.verifyFoldStep(running: running, fresh: fresh,
                                      proof: proof, target: target)
    expect(ok, "Single fold step verification should pass")
}

private func testFoldChainValid() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
    for i: UInt64 in 2...5 {
        steps.append(makeVerifierSquaringPair(i))
    }
    let ivcProof = ivcProver.prove(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)
    let result = verifier.verifyFoldChain(proof: ivcProof)

    expect(result.valid, "Fold chain should be valid for 4-step IVC")
    expectEqual(result.stepsVerified, 3, "Should verify 3 fold steps (4 steps - 1 base)")
    expectEqual(result.failingStep, -1, "No failing step")
    expect(!frEq(result.finalStateHash, Fr.zero), "Final state hash should be non-zero")
    expectEqual(result.challenges.count, 3, "Should have 3 challenges")
}

private func testFoldChainSingleStep() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    let steps = [makeVerifierSquaringPair(4)]
    let ivcProof = ivcProver.prove(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)
    let result = verifier.verifyFoldChain(proof: ivcProof)

    expect(result.valid, "Single-step fold chain should trivially pass")
    expectEqual(result.stepsVerified, 1, "Should verify 1 step")
    expectEqual(result.challenges.count, 0, "No challenges for single step")
}

private func testAccumulationVerification() {
    // Use NovaFoldProver which shares the "nova-r1cs-fold" transcript label with the verifier
    let shape = makeVerifierSquaringR1CS()
    let foldProver = NovaFoldProver(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    let (runningBefore, witnessBefore) = shape.relax(instance: inst1, witness: wit1, pp: foldProver.pp)

    let (inst2, wit2) = makeVerifierSquaringPair(7)
    let (foldedInst, foldedWit, _) = foldProver.fold(
        runningInstance: runningBefore, runningWitness: witnessBefore,
        newInstance: inst2, newWitness: wit2)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldProver.pp)
    let ok = verifier.verifyAccumulation(
        runningInstance: runningBefore,
        runningWitness: witnessBefore,
        newInstance: inst2,
        newWitness: wit2,
        claimedResult: foldedInst,
        claimedWitness: foldedWit)
    expect(ok, "Accumulation verification should pass for valid fold")
}

private func testStateHash() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst, wit) = makeVerifierSquaringPair(5)
    let (relaxedInst, _) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    // Hash should be deterministic
    let h1 = engine.hashRelaxedInstance(relaxedInst)
    let h2 = engine.hashRelaxedInstance(relaxedInst)
    expect(frEq(h1, h2), "State hash should be deterministic")
    expect(!frEq(h1, Fr.zero), "State hash should be non-zero")
}

private func testStateHashDiffers() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    let (relaxed1, _) = shape.relax(instance: inst1, witness: wit1, pp: engine.pp)

    let (inst2, wit2) = makeVerifierSquaringPair(7)
    let (relaxed2, _) = shape.relax(instance: inst2, witness: wit2, pp: engine.pp)

    let h1 = engine.hashRelaxedInstance(relaxed1)
    let h2 = engine.hashRelaxedInstance(relaxed2)
    expect(!frEq(h1, h2), "Different instances should have different state hashes")
}

private func testStateHashVerification() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let (inst, wit) = makeVerifierSquaringPair(4)
    let (relaxedInst, _) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    let correctHash = engine.hashRelaxedInstance(relaxedInst)
    expect(engine.verifyStateHash(instance: relaxedInst, claimedHash: correctHash),
           "State hash verification should pass with correct hash")

    let wrongHash = frFromInt(999)
    expect(!engine.verifyStateHash(instance: relaxedInst, claimedHash: wrongHash),
           "State hash verification should fail with wrong hash")
}

private func testChainedStateHash() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    var instances = [NovaRelaxedInstance]()
    for i: UInt64 in 2...4 {
        let (inst, wit) = makeVerifierSquaringPair(i)
        let (relaxed, _) = shape.relax(instance: inst, witness: wit, pp: engine.pp)
        instances.append(relaxed)
    }

    let chainHash = engine.chainedStateHash(instances)
    expect(!frEq(chainHash, Fr.zero), "Chained state hash should be non-zero")

    // Same input should give same hash
    let chainHash2 = engine.chainedStateHash(instances)
    expect(frEq(chainHash, chainHash2), "Chained state hash should be deterministic")

    // Different order should give different hash
    let reversed = Array(instances.reversed())
    let reversedHash = engine.chainedStateHash(reversed)
    expect(!frEq(chainHash, reversedHash), "Reversed chain should have different hash")
}

private func testBatchVerification() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    // Create two valid IVC proofs
    let proof1 = ivcProver.prove(steps: [makeVerifierSquaringPair(3), makeVerifierSquaringPair(5)])
    let ivcProver2 = NovaIVCProver(shape: shape, pp: ivcProver.pp)
    let proof2 = ivcProver2.prove(steps: [makeVerifierSquaringPair(7), makeVerifierSquaringPair(11)])

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)
    let result = verifier.batchVerify(proofs: [proof1, proof2])

    expect(result.allValid, "Batch verification should pass for valid proofs")
    expectEqual(result.passCount, 2, "Both proofs should pass")
    expectEqual(result.failCount, 0, "No proofs should fail")
    expectEqual(result.perProofValid.count, 2, "Should have 2 per-proof results")
    expect(result.perProofValid[0], "Proof 1 should be valid")
    expect(result.perProofValid[1], "Proof 2 should be valid")
}

private func testBatchVerifyFoldSteps() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
    for i: UInt64 in 2...4 {
        steps.append(makeVerifierSquaringPair(i))
    }
    let ivcProof = ivcProver.prove(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)

    // Collect fold steps
    var foldSteps = [(running: NovaRelaxedInstance, fresh: NovaR1CSInput,
                      proof: NovaFoldProof, target: NovaRelaxedInstance)]()
    for i in 0..<ivcProof.foldProofs.count {
        let target: NovaRelaxedInstance
        if i < ivcProof.foldProofs.count - 1 {
            target = ivcProof.intermediateInstances[i + 1]
        } else {
            target = ivcProof.finalInstance
        }
        foldSteps.append((
            running: ivcProof.intermediateInstances[i],
            fresh: ivcProof.freshInstances[i],
            proof: ivcProof.foldProofs[i],
            target: target))
    }

    let results = verifier.batchVerifyFoldSteps(steps: foldSteps)
    expect(results.allSatisfy { $0 }, "All fold steps should verify")
    expectEqual(results.count, 2, "Should have 2 fold steps")
}

private func testSuperNovaVerification() {
    let shape1 = makeVerifierSquaringR1CS()
    let shape2 = makeVerifierMultiplyR1CS()

    // Build accumulator for circuit 1 (squaring)
    let fold1 = GPUNovaFoldEngine(shape: shape1)
    let steps1 = [makeVerifierSquaringPair(3), makeVerifierSquaringPair(5)]
    let (inst1, wit1) = fold1.ivcChain(steps: steps1)

    // Build accumulator for circuit 2 (multiply)
    let fold2 = GPUNovaFoldEngine(shape: shape2)
    let steps2 = [makeVerifierMultiplyPair(2, 7), makeVerifierMultiplyPair(3, 11)]
    let (inst2, wit2) = fold2.ivcChain(steps: steps2)

    let superAccum = SuperNovaAccumulator(
        instances: [inst1, inst2],
        witnesses: [wit1, wit2],
        shapes: [shape1, shape2],
        circuitSchedule: [0, 0, 1, 1],
        stepCount: 4)

    let verifier = GPUNovaVerifierEngine(shape: shape1)
    let result = verifier.verifySuperNova(accumulator: superAccum)

    expect(result.allValid, "SuperNova verification should pass for valid accumulators")
    expectEqual(result.perCircuitValid.count, 2, "Should have 2 per-circuit results")
    expect(result.perCircuitValid[0], "Circuit 1 should be valid")
    expect(result.perCircuitValid[1], "Circuit 2 should be valid")
    expect(!frEq(result.combinedStateHash, Fr.zero), "Combined state hash should be non-zero")
    expectEqual(result.perCircuitStateHashes.count, 2, "Should have 2 per-circuit hashes")
}

private func testSuperNovaWithCommitments() {
    let shape1 = makeVerifierSquaringR1CS()
    let shape2 = makeVerifierMultiplyR1CS()

    let fold1 = GPUNovaFoldEngine(shape: shape1)
    let (inst1, wit1) = fold1.ivcChain(steps: [makeVerifierSquaringPair(3), makeVerifierSquaringPair(5)])

    let fold2 = GPUNovaFoldEngine(shape: shape2)
    let (inst2, wit2) = fold2.ivcChain(steps: [makeVerifierMultiplyPair(2, 7)])

    let superAccum = SuperNovaAccumulator(
        instances: [inst1, inst2],
        witnesses: [wit1, wit2],
        shapes: [shape1, shape2],
        circuitSchedule: [0, 0, 1],
        stepCount: 3)

    let verifier = GPUNovaVerifierEngine(shape: shape1)
    let ok = verifier.verifySuperNovaWithCommitments(
        accumulator: superAccum,
        ppPerCircuit: [fold1.pp, fold2.pp],
        ppEPerCircuit: [fold1.ppE, fold2.ppE])
    expect(ok, "SuperNova with commitment checks should pass")
}

private func testFullIVCVerification() {
    let shape = makeVerifierSquaringR1CS()
    let ivcProver = NovaIVCProver(shape: shape)

    var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
    for i: UInt64 in 2...4 {
        steps.append(makeVerifierSquaringPair(i))
    }
    let ivcProof = ivcProver.prove(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: ivcProver.pp)

    // Verify without expected state hash
    let ok1 = verifier.verifyIVC(proof: ivcProof)
    expect(ok1, "Full IVC verification should pass")

    // Verify with correct expected state hash
    let expectedHash = verifier.hashRelaxedInstance(ivcProof.finalInstance)
    let ok2 = verifier.verifyIVC(proof: ivcProof, expectedStateHash: expectedHash)
    expect(ok2, "Full IVC verification with correct state hash should pass")

    // Verify with wrong expected state hash should fail
    let wrongHash = frFromInt(12345)
    let ok3 = verifier.verifyIVC(proof: ivcProof, expectedStateHash: wrongHash)
    expect(!ok3, "Full IVC verification with wrong state hash should fail")
}

private func testConfigDefaults() {
    let defaultConfig = NovaVerifierConfig()
    expect(defaultConfig.useGPU == true, "Default config should enable GPU")
    expectEqual(defaultConfig.gpuThreshold, 512, "Default GPU threshold should be 512")
    expect(defaultConfig.verifyCommitments == true, "Default should verify commitments")
    expect(defaultConfig.verifyStateHashes == true, "Default should verify state hashes")
    expect(defaultConfig.verifyCrossTerms == true, "Default should verify cross-terms")

    let customConfig = NovaVerifierConfig(useGPU: false, gpuThreshold: 1024,
                                           verifyCommitments: false, verifyStateHashes: false,
                                           verifyCrossTerms: false)
    expect(customConfig.useGPU == false, "Custom GPU flag")
    expectEqual(customConfig.gpuThreshold, 1024, "Custom GPU threshold")
    expect(customConfig.verifyCommitments == false, "Custom commitment flag")
    expect(customConfig.verifyStateHashes == false, "Custom state hash flag")
    expect(customConfig.verifyCrossTerms == false, "Custom cross-term flag")
}

private func testGPUAvailability() {
    let shape = makeVerifierSquaringR1CS()
    let gpuEngine = GPUNovaVerifierEngine(shape: shape, config: NovaVerifierConfig(useGPU: true))
    let cpuEngine = GPUNovaVerifierEngine(shape: shape, config: NovaVerifierConfig(useGPU: false))

    // CPU engine should never have GPU
    expect(!cpuEngine.gpuAvailable, "CPU-only engine should not have GPU available")

    // Both should produce valid verification results
    let (inst, wit) = makeVerifierSquaringPair(4)
    let (ri, rw) = shape.relax(instance: inst, witness: wit, pp: gpuEngine.pp)

    let ok1 = gpuEngine.verifyRelaxedR1CS(instance: ri, witness: rw)
    expect(ok1, "GPU engine should verify valid instance")

    let (ri2, rw2) = shape.relax(instance: inst, witness: wit, pp: cpuEngine.pp)
    let ok2 = cpuEngine.verifyRelaxedR1CS(instance: ri2, witness: rw2)
    expect(ok2, "CPU engine should verify valid instance")
}

private func testGPUInnerProduct() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
    let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
    // Expected: 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
    let result = engine.gpuFieldInnerProduct(a, b)
    let expected = frFromInt(112)
    expect(frEq(result, expected), "GPU inner product should compute 2*7+3*11+5*13 = 112")
}

private func testErrorVectorAnalysis() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    // Base case: E = 0
    let (inst, wit) = makeVerifierSquaringPair(3)
    let (_, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

    expect(engine.isBaseCase(witness: relaxedWit), "Freshly relaxed witness should be base case")

    let norm = engine.errorNormSquared(witness: relaxedWit)
    expect(frEq(norm, Fr.zero), "Error norm should be zero for base case")

    // After folding: E should be non-zero
    let foldEngine = GPUNovaFoldEngine(shape: shape, pp: engine.pp, ppE: engine.ppE)
    foldEngine.initialize(instance: inst, witness: wit)

    let (inst2, wit2) = makeVerifierSquaringPair(5)
    let _ = foldEngine.foldStep(newInstance: inst2, newWitness: wit2)

    let foldedWit = foldEngine.runningWitness!
    let hasNonZeroError = foldedWit.E.contains { !$0.isZero }
    expect(hasNonZeroError, "Folded error should be non-zero")
    expect(!engine.isBaseCase(witness: foldedWit), "Folded witness should not be base case")
}

private func testMultiplyCircuitVerification() {
    let shape = makeVerifierMultiplyR1CS()
    let foldEngine = GPUNovaFoldEngine(shape: shape)

    var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
    steps.append(makeVerifierMultiplyPair(3, 5))
    steps.append(makeVerifierMultiplyPair(7, 11))
    steps.append(makeVerifierMultiplyPair(2, 13))

    let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
    let ok = verifier.verifyRelaxedR1CS(instance: finalInst, witness: finalWit)
    expect(ok, "Multiply circuit accumulator should satisfy relaxed R1CS")
}

private func testTwoConstraintCircuit() {
    let shape = makeVerifierTwoConstraintR1CS()
    let foldEngine = GPUNovaFoldEngine(shape: shape)

    var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
    steps.append(makeVerifierTwoConstraintPair(3, 7))
    steps.append(makeVerifierTwoConstraintPair(5, 11))

    let (finalInst, finalWit) = foldEngine.ivcChain(steps: steps)

    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldEngine.pp, ppE: foldEngine.ppE)
    let ok = verifier.verifyRelaxedR1CS(instance: finalInst, witness: finalWit)
    expect(ok, "Two-constraint accumulator should satisfy relaxed R1CS")
}

private func testDeriveVerifierChallenge() {
    // The verifier's challenge derivation should match the NovaFoldProver's
    // (both use "nova-r1cs-fold" transcript label)
    let shape = makeVerifierSquaringR1CS()
    let foldProver = NovaFoldProver(shape: shape)

    let (inst1, wit1) = makeVerifierSquaringPair(3)
    let (relaxedInst, relaxedWit) = shape.relax(instance: inst1, witness: wit1, pp: foldProver.pp)

    let (inst2, _) = makeVerifierSquaringPair(5)
    let T = foldProver.computeCrossTerm(
        runningInstance: relaxedInst, runningWitness: relaxedWit,
        newInstance: inst2, newWitness: makeVerifierSquaringPair(5).1)
    let commitT = foldProver.pp.commit(witness: T)

    // Prover's challenge (uses "nova-r1cs-fold" transcript)
    let proverR = foldProver.deriveChallenge(
        runningInstance: relaxedInst, newInstance: inst2, commitT: commitT)

    // Verifier's challenge (also uses "nova-r1cs-fold" transcript)
    let verifier = GPUNovaVerifierEngine(shape: shape, pp: foldProver.pp)
    let verifierR = verifier.deriveVerifierChallenge(
        running: relaxedInst, fresh: inst2, commitT: commitT)

    expect(frEq(proverR, verifierR),
           "Prover and verifier should derive the same Fiat-Shamir challenge")
}

private func testEmptyChainedHash() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)

    let emptyHash = engine.chainedStateHash([])
    expect(frEq(emptyHash, Fr.zero), "Empty chained hash should be zero")
}

private func testVersionExists() {
    let shape = makeVerifierSquaringR1CS()
    let engine = GPUNovaVerifierEngine(shape: shape)
    // Just verify the version constant is accessible and non-nil
    let v = GPUNovaVerifierEngine.version
    expect(v.version == "1.0.0", "Version should be 1.0.0")
}

private func testFoldChainResultTypes() {
    // Test the result types are correctly constructed
    let result = NovaFoldChainResult(valid: true, stepsVerified: 5, failingStep: -1,
                                      finalStateHash: Fr.one, challenges: [Fr.one, Fr.zero])
    expect(result.valid, "Result should be valid")
    expectEqual(result.stepsVerified, 5, "Steps verified")
    expectEqual(result.failingStep, -1, "No failing step")
    expectEqual(result.challenges.count, 2, "Two challenges")

    let batchResult = NovaBatchVerifyResult(allValid: false, perProofValid: [true, false],
                                             passCount: 1, failCount: 1)
    expect(!batchResult.allValid, "Batch should not be all valid")
    expectEqual(batchResult.passCount, 1, "One pass")
    expectEqual(batchResult.failCount, 1, "One fail")
}

// MARK: - Test Runner

public func runGPUNovaVerifierTests() {
    suite("GPU Nova Verifier Engine")

    testRelaxedR1CSCheck()
    testStrictR1CSCheck()
    testRelaxedR1CSAfterFolding()
    testCommitmentVerification()
    testCrossTermZeroForIdentical()
    testCrossTermCommitmentValid()
    testCrossTermCommitmentInvalid()
    testFoldStepVerification()
    testFoldChainValid()
    testFoldChainSingleStep()
    testAccumulationVerification()
    testStateHash()
    testStateHashDiffers()
    testStateHashVerification()
    testChainedStateHash()
    testBatchVerification()
    testBatchVerifyFoldSteps()
    testSuperNovaVerification()
    testSuperNovaWithCommitments()
    testFullIVCVerification()
    testConfigDefaults()
    testGPUAvailability()
    testGPUInnerProduct()
    testErrorVectorAnalysis()
    testMultiplyCircuitVerification()
    testTwoConstraintCircuit()
    testDeriveVerifierChallenge()
    testEmptyChainedHash()
    testVersionExists()
    testFoldChainResultTypes()
}
