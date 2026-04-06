// GPU Protogalaxy Verifier Engine Tests -- fold verification, Lagrange basis,
// vanishing polynomial, Fiat-Shamir challenge, batch verification, error terms,
// IVC chain replay, state hashing, GPU inner product
import zkMetal
import Foundation

// MARK: - Test Helpers

/// Create a fresh Plonk instance with dummy commitments for verifier testing.
private func makeVerifierInstance(publicInput: [Fr], beta: Fr, gamma: Fr,
                                   scaleFactor: Fr = Fr.one) -> ProtogalaxyInstance {
    let g1 = pointFromAffine(bn254G1Generator())
    let c0 = cPointScalarMul(g1, scaleFactor)
    let c1 = cPointScalarMul(g1, frMul(scaleFactor, frFromInt(2)))
    let c2 = cPointScalarMul(g1, frMul(scaleFactor, frFromInt(3)))
    return ProtogalaxyInstance(
        witnessCommitments: [c0, c1, c2],
        publicInput: publicInput,
        beta: beta,
        gamma: gamma
    )
}

/// Create a satisfying witness: a + b = c
private func makeVerifierWitness(size: Int, seed: UInt64) -> [[Fr]] {
    var a = [Fr](repeating: Fr.zero, count: size)
    var b = [Fr](repeating: Fr.zero, count: size)
    var c = [Fr](repeating: Fr.zero, count: size)
    for i in 0..<size {
        a[i] = frFromInt(seed &+ UInt64(i) &+ 1)
        b[i] = frFromInt(seed &+ UInt64(i) &+ 100)
        c[i] = frAdd(a[i], b[i])
    }
    return [a, b, c]
}

/// Fold two instances and return (folded, proof) for verifier testing.
private func foldPair(inst0: ProtogalaxyInstance, inst1: ProtogalaxyInstance,
                       wit0: [[Fr]], wit1: [[Fr]],
                       circuitSize: Int) -> (ProtogalaxyInstance, ProtogalaxyFoldingProof) {
    let prover = ProtogalaxyProver(circuitSize: circuitSize)
    let (folded, _, proof) = prover.fold(instances: [inst0, inst1],
                                          witnesses: [wit0, wit1])
    return (folded, proof)
}

// MARK: - Test Implementations

private func testBasicFoldVerification() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3),
                                      scaleFactor: frFromInt(1))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(7))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let valid = engine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    expect(valid, "Basic fold verification should pass for valid fold")
}

private func testDetailedFoldVerification() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(10)],
                                      beta: frFromInt(2), gamma: frFromInt(3),
                                      scaleFactor: frFromInt(1))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(20)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(4))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 1)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 50)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: folded, proof: proof)

    expect(result.valid, "Detailed: overall valid")
    expect(result.fPolyConsistent, "Detailed: F(X) consistent")
    expect(result.commitmentsValid, "Detailed: commitments valid")
    expect(result.publicInputValid, "Detailed: public input valid")
    expect(result.challengesValid, "Detailed: challenges valid")
    expect(result.errorTermValid, "Detailed: error term valid")
    expect(result.relaxationValid, "Detailed: relaxation valid")
    expect(!frEq(result.challenge, Fr.zero), "Detailed: challenge is non-zero")
}

private func testFoldVerificationRejectsTampered() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3),
                                      scaleFactor: frFromInt(1))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(7))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    // Tamper with folded instance: wrong error term
    let tamperedFolded = ProtogalaxyInstance(
        witnessCommitments: folded.witnessCommitments,
        publicInput: folded.publicInput,
        beta: folded.beta,
        gamma: folded.gamma,
        errorTerm: frAdd(folded.errorTerm, Fr.one),
        u: folded.u)

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: tamperedFolded, proof: proof)
    expect(!result.valid, "Tampered error term should fail")
    expect(!result.errorTermValid, "Error term check should fail specifically")
    expect(result.fPolyConsistent, "F(X) should still be consistent")
}

private func testFoldVerificationRejectsBadPublicInput() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    // Tamper: wrong public input
    let tamperedFolded = ProtogalaxyInstance(
        witnessCommitments: folded.witnessCommitments,
        publicInput: [frFromInt(999)],
        beta: folded.beta,
        gamma: folded.gamma,
        errorTerm: folded.errorTerm,
        u: folded.u)

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: tamperedFolded, proof: proof)
    expect(!result.valid, "Tampered public input should fail")
    expect(!result.publicInputValid, "Public input check should fail specifically")
}

private func testFoldVerificationRejectsBadChallenges() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    // Tamper: wrong beta
    let tamperedFolded = ProtogalaxyInstance(
        witnessCommitments: folded.witnessCommitments,
        publicInput: folded.publicInput,
        beta: frAdd(folded.beta, Fr.one),
        gamma: folded.gamma,
        errorTerm: folded.errorTerm,
        u: folded.u)

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: tamperedFolded, proof: proof)
    expect(!result.valid, "Tampered beta should fail")
    expect(!result.challengesValid, "Challenges check should fail specifically")
}

private func testVanishingPolyEvaluation() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (_, proof) = foldPair(inst0: inst0, inst1: inst1,
                               wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()

    // F(0) should equal inst0's error term (zero for fresh instance)
    let f0 = engine.evaluateVanishingPoly(proof: proof, at: Fr.zero)
    expect(frEq(f0, inst0.errorTerm), "F(0) should equal instance 0 error term")

    // F(1) should equal inst1's error term
    let f1 = engine.evaluateVanishingPoly(proof: proof, at: Fr.one)
    expect(frEq(f1, inst1.errorTerm), "F(1) should equal instance 1 error term")
}

private func testVerifyVanishingPoly() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (_, proof) = foldPair(inst0: inst0, inst1: inst1,
                               wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyVanishingPoly(instances: [inst0, inst1], proof: proof)
    expect(ok, "Vanishing polynomial should verify for valid fold")

    // Tamper with F(X) coefficients
    let badProof = ProtogalaxyFoldingProof(
        fCoefficients: proof.fCoefficients.map { frAdd($0, Fr.one) },
        instanceCount: proof.instanceCount)
    let notOk = engine.verifyVanishingPoly(instances: [inst0, inst1], proof: badProof)
    expect(!notOk, "Tampered F(X) should fail vanishing poly check")
}

private func testLagrangePartitionOfUnity() {
    let engine = GPUProtogalaxyVerifierEngine()

    // For various domain sizes and random-ish points
    for k in 2...6 {
        let alpha = frFromInt(UInt64(k * 7 + 13))
        let ok = engine.verifyLagrangePartitionOfUnity(domainSize: k, point: alpha)
        expect(ok, "Lagrange partition of unity for k=\(k)")
    }
}

private func testLagrangeAtDomainPoints() {
    let engine = GPUProtogalaxyVerifierEngine()

    // L_i(j) = delta_{ij}
    for k in 2...5 {
        for j in 0..<k {
            let ok = engine.verifyLagrangeAtDomainPoint(domainSize: k, index: j)
            expect(ok, "Lagrange at domain point k=\(k), j=\(j)")
        }
    }
}

private func testLagrangeBasisComputation() {
    let engine = GPUProtogalaxyVerifierEngine()

    let basis = engine.computeLagrangeBasis(domainSize: 3, point: frFromInt(10))
    expectEqual(basis.count, 3, "Lagrange basis should have 3 elements for k=3")

    // Verify partition of unity
    var sum = Fr.zero
    for li in basis {
        sum = frAdd(sum, li)
    }
    expect(frEq(sum, Fr.one), "Lagrange basis should sum to 1")
}

private func testChallengeRederivation() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (_, proof) = foldPair(inst0: inst0, inst1: inst1,
                               wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()

    // Re-derive challenge twice, should be deterministic
    let alpha1 = engine.rederiveChallenge(instances: [inst0, inst1], proof: proof)
    let alpha2 = engine.rederiveChallenge(instances: [inst0, inst1], proof: proof)
    expect(frEq(alpha1, alpha2), "Challenge re-derivation should be deterministic")
    expect(!frEq(alpha1, Fr.zero), "Challenge should be non-zero")
}

private func testChallengeConsistencyVerification() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (_, proof) = foldPair(inst0: inst0, inst1: inst1,
                               wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let alpha = engine.rederiveChallenge(instances: [inst0, inst1], proof: proof)

    let ok = engine.verifyChallengeConsistency(instances: [inst0, inst1],
                                                proof: proof, claimedChallenge: alpha)
    expect(ok, "Correct challenge should verify")

    let wrongAlpha = frAdd(alpha, Fr.one)
    let notOk = engine.verifyChallengeConsistency(instances: [inst0, inst1],
                                                    proof: proof, claimedChallenge: wrongAlpha)
    expect(!notOk, "Wrong challenge should fail")
}

private func testChallengeDiffersForDifferentInstances() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let inst2 = makeVerifierInstance(publicInput: [frFromInt(7)],
                                      beta: frFromInt(8), gamma: frFromInt(9),
                                      scaleFactor: frFromInt(3))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)
    let wit2 = makeVerifierWitness(size: circuitSize, seed: 30)

    let (_, proof1) = foldPair(inst0: inst0, inst1: inst1,
                                wit0: wit0, wit1: wit1, circuitSize: circuitSize)
    let (_, proof2) = foldPair(inst0: inst0, inst1: inst2,
                                wit0: wit0, wit1: wit2, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let alpha1 = engine.rederiveChallenge(instances: [inst0, inst1], proof: proof1)
    let alpha2 = engine.rederiveChallenge(instances: [inst0, inst2], proof: proof2)

    expect(!frEq(alpha1, alpha2), "Different instances should produce different challenges")
}

private func testBatchVerification() {
    let circuitSize = 4
    let engine = GPUProtogalaxyVerifierEngine()

    // Create 3 valid folds
    var folds = [(instances: [ProtogalaxyInstance],
                  folded: ProtogalaxyInstance,
                  proof: ProtogalaxyFoldingProof)]()

    for i in 0..<3 {
        let inst0 = makeVerifierInstance(publicInput: [frFromInt(UInt64(i * 10 + 1))],
                                          beta: frFromInt(UInt64(i * 3 + 1)),
                                          gamma: frFromInt(UInt64(i * 3 + 2)),
                                          scaleFactor: frFromInt(UInt64(i + 1)))
        let inst1 = makeVerifierInstance(publicInput: [frFromInt(UInt64(i * 10 + 50))],
                                          beta: frFromInt(UInt64(i * 3 + 10)),
                                          gamma: frFromInt(UInt64(i * 3 + 11)),
                                          scaleFactor: frFromInt(UInt64(i + 5)))
        let wit0 = makeVerifierWitness(size: circuitSize, seed: UInt64(i * 100))
        let wit1 = makeVerifierWitness(size: circuitSize, seed: UInt64(i * 100 + 50))

        let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                        wit0: wit0, wit1: wit1, circuitSize: circuitSize)
        folds.append((instances: [inst0, inst1], folded: folded, proof: proof))
    }

    let result = engine.batchVerify(folds: folds)
    expect(result.allValid, "Batch verification should pass for all valid folds")
    expectEqual(result.passCount, 3, "All 3 folds should pass")
    expectEqual(result.failCount, 0, "No folds should fail")
    expectEqual(result.perFoldValid.count, 3, "Should have 3 per-fold results")
    expectEqual(result.challenges.count, 3, "Should have 3 challenges")
    for i in 0..<3 {
        expect(result.perFoldValid[i], "Fold \(i) should be valid")
        expect(!frEq(result.challenges[i], Fr.zero), "Challenge \(i) should be non-zero")
    }
}

private func testBatchVerificationWithInvalid() {
    let circuitSize = 4
    let engine = GPUProtogalaxyVerifierEngine()

    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    // Create a tampered fold
    let tamperedFolded = ProtogalaxyInstance(
        witnessCommitments: folded.witnessCommitments,
        publicInput: folded.publicInput,
        beta: folded.beta,
        gamma: folded.gamma,
        errorTerm: frAdd(folded.errorTerm, Fr.one),
        u: folded.u)

    let folds: [(instances: [ProtogalaxyInstance],
                  folded: ProtogalaxyInstance,
                  proof: ProtogalaxyFoldingProof)] = [
        (instances: [inst0, inst1], folded: folded, proof: proof),
        (instances: [inst0, inst1], folded: tamperedFolded, proof: proof)
    ]

    let result = engine.batchVerify(folds: folds)
    expect(!result.allValid, "Batch should not be all valid with tampered fold")
    expectEqual(result.passCount, 1, "1 valid fold")
    expectEqual(result.failCount, 1, "1 invalid fold")
    expect(result.perFoldValid[0], "First fold should be valid")
    expect(!result.perFoldValid[1], "Second (tampered) fold should be invalid")
}

private func testIVCChainVerification() {
    let circuitSize = 4
    let prover = ProtogalaxyProver(circuitSize: circuitSize)

    var instances = [ProtogalaxyInstance]()
    var witnesses = [[[Fr]]]()
    for i in 0..<4 {
        instances.append(makeVerifierInstance(
            publicInput: [frFromInt(UInt64(i * 10 + 1))],
            beta: frFromInt(UInt64(i * 3 + 1)),
            gamma: frFromInt(UInt64(i * 3 + 2)),
            scaleFactor: frFromInt(UInt64(i + 1))))
        witnesses.append(makeVerifierWitness(size: circuitSize, seed: UInt64(i * 50)))
    }

    // Build the fold chain manually
    var running = instances[0]
    var runningWit = witnesses[0]
    var foldingProofs = [ProtogalaxyFoldingProof]()

    for i in 1..<instances.count {
        let (folded, foldedWit, proof) = prover.fold(
            instances: [running, instances[i]],
            witnesses: [runningWit, witnesses[i]])
        running = folded
        runningWit = foldedWit
        foldingProofs.append(proof)
    }

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyChain(originalInstances: instances,
                                     foldingProofs: foldingProofs,
                                     finalInstance: running)

    expect(result.valid, "IVC chain verification should pass")
    expectEqual(result.stepsVerified, 3, "Should verify 3 fold steps")
    expectEqual(result.failingStep, -1, "No failing step")
    expect(!frEq(result.finalStateHash, Fr.zero), "Final state hash should be non-zero")
    expectEqual(result.challenges.count, 3, "Should have 3 challenges")
    expectEqual(result.perStepResults.count, 3, "Should have 3 per-step results")
    for i in 0..<3 {
        expect(result.perStepResults[i].valid, "Step \(i) should be valid")
    }
}

private func testIVCChainRejectsTamperedFinal() {
    let circuitSize = 4
    let prover = ProtogalaxyProver(circuitSize: circuitSize)

    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let inst2 = makeVerifierInstance(publicInput: [frFromInt(7)],
                                      beta: frFromInt(8), gamma: frFromInt(9),
                                      scaleFactor: frFromInt(3))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)
    let wit2 = makeVerifierWitness(size: circuitSize, seed: 30)

    var running = inst0
    var runningWit = wit0
    var foldingProofs = [ProtogalaxyFoldingProof]()

    let (folded1, foldedWit1, proof1) = prover.fold(
        instances: [running, inst1], witnesses: [runningWit, wit1])
    running = folded1; runningWit = foldedWit1; foldingProofs.append(proof1)

    let (folded2, _, proof2) = prover.fold(
        instances: [running, inst2], witnesses: [runningWit, wit2])
    foldingProofs.append(proof2)

    // Tamper with final instance
    let tampered = ProtogalaxyInstance(
        witnessCommitments: folded2.witnessCommitments,
        publicInput: folded2.publicInput,
        beta: folded2.beta,
        gamma: folded2.gamma,
        errorTerm: frAdd(folded2.errorTerm, Fr.one),
        u: folded2.u)

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyChain(originalInstances: [inst0, inst1, inst2],
                                     foldingProofs: foldingProofs,
                                     finalInstance: tampered)
    expect(!result.valid, "Chain with tampered final instance should fail")
}

private func testSingleInstanceChain() {
    let inst = makeVerifierInstance(publicInput: [frFromInt(1)],
                                     beta: frFromInt(2), gamma: frFromInt(3))

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyChain(originalInstances: [inst],
                                     foldingProofs: [],
                                     finalInstance: inst)
    expect(result.valid, "Single-instance chain should trivially pass")
    expectEqual(result.stepsVerified, 1, "Should verify 1 step")
    expectEqual(result.challenges.count, 0, "No challenges for single instance")
}

private func testErrorTermComputation() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let alpha = engine.rederiveChallenge(instances: [inst0, inst1], proof: proof)
    let computedError = engine.computeFoldedErrorTerm(proof: proof, alpha: alpha)

    expect(frEq(computedError, folded.errorTerm),
           "Computed error term should match folded instance")
}

private func testErrorPropagationVerification() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyErrorPropagation(instances: [inst0, inst1],
                                            foldedError: folded.errorTerm, proof: proof)
    expect(ok, "Error propagation should verify for valid fold")

    let notOk = engine.verifyErrorPropagation(instances: [inst0, inst1],
                                                foldedError: frFromInt(999), proof: proof)
    expect(!notOk, "Error propagation should fail with wrong error term")
}

private func testStateHashing() {
    let engine = GPUProtogalaxyVerifierEngine()
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst2 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    // Deterministic
    let h1 = engine.hashInstance(inst1)
    let h2 = engine.hashInstance(inst1)
    expect(frEq(h1, h2), "State hash should be deterministic")
    expect(!frEq(h1, Fr.zero), "State hash should be non-zero")
    // Different instances differ
    let h3 = engine.hashInstance(inst2)
    expect(!frEq(h1, h3), "Different instances should have different hashes")
    // Verify correct/incorrect
    expect(engine.verifyStateHash(instance: inst1, claimedHash: h1),
           "State hash verification should pass with correct hash")
    expect(!engine.verifyStateHash(instance: inst1, claimedHash: frFromInt(999)),
           "State hash verification should fail with wrong hash")
}

private func testChainedStateHash() {
    let engine = GPUProtogalaxyVerifierEngine()

    var instances = [ProtogalaxyInstance]()
    for i: UInt64 in 1...3 {
        instances.append(makeVerifierInstance(
            publicInput: [frFromInt(i)],
            beta: frFromInt(i * 2),
            gamma: frFromInt(i * 3),
            scaleFactor: frFromInt(i)))
    }

    let h = engine.chainedStateHash(instances)
    expect(!frEq(h, Fr.zero), "Chained state hash should be non-zero")

    // Deterministic
    let h2 = engine.chainedStateHash(instances)
    expect(frEq(h, h2), "Chained state hash should be deterministic")

    // Different order should give different hash
    let reversed = Array(instances.reversed())
    let hReversed = engine.chainedStateHash(reversed)
    expect(!frEq(h, hReversed), "Reversed chain should have different hash")

    // Empty chain
    let hEmpty = engine.chainedStateHash([])
    expect(frEq(hEmpty, Fr.zero), "Empty chained hash should be zero")
}

private func testGPUAvailability() {
    let gpuEngine = GPUProtogalaxyVerifierEngine(
        config: ProtogalaxyVerifierConfig(useGPU: true))
    let cpuEngine = GPUProtogalaxyVerifierEngine(
        config: ProtogalaxyVerifierConfig(useGPU: false))

    expect(!cpuEngine.gpuAvailable, "CPU-only engine should not have GPU")

    // Both should produce the same verification result
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let ok1 = gpuEngine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    let ok2 = cpuEngine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    expect(ok1, "GPU engine should verify valid fold")
    expect(ok2, "CPU engine should verify valid fold")
}

private func testGPUInnerProduct() {
    let engine = GPUProtogalaxyVerifierEngine()
    let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
    let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
    let result = engine.gpuFieldInnerProduct(a, b)
    expect(frEq(result, frFromInt(112)), "Inner product: 2*7+3*11+5*13 = 112")
    let empty = engine.gpuFieldInnerProduct([], [])
    expect(frEq(empty, Fr.zero), "Inner product of empty vectors should be zero")
}

private func testDeciderProofVerification() {
    let circuitSize = 4
    let prover = ProtogalaxyProver(circuitSize: circuitSize)

    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3),
                                      scaleFactor: frFromInt(1))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(7))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (acc, accWit, _) = prover.fold(instances: [inst0, inst1],
                                        witnesses: [wit0, wit1])

    let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
    let decider = ProtogalaxyDeciderProver(config: config)
    let proof = decider.decide(instance: acc, witnesses: accWit)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyDeciderProof(proof: proof)
    expect(ok, "Decider proof should verify via GPU engine")
}

private func testMultiplePublicInputs() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(
        publicInput: [frFromInt(1), frFromInt(2), frFromInt(3)],
        beta: frFromInt(10), gamma: frFromInt(11))
    let inst1 = makeVerifierInstance(
        publicInput: [frFromInt(4), frFromInt(5), frFromInt(6)],
        beta: frFromInt(12), gamma: frFromInt(13),
        scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 1)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 2)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    expect(ok, "Multiple public inputs fold should verify")
    expectEqual(folded.publicInput.count, 3, "Should preserve 3 public inputs")
}

private func testLargerCircuitSize() {
    let circuitSize = 8
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3),
                                      scaleFactor: frFromInt(1))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(7))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    expect(ok, "Larger circuit size (8) fold should verify")
}

private func testConfigDefaults() {
    let defaultConfig = ProtogalaxyVerifierConfig()
    expect(defaultConfig.useGPU == true, "Default should enable GPU")
    expectEqual(defaultConfig.gpuThreshold, 512, "Default threshold should be 512")
    expect(defaultConfig.verifyCommitments == true, "Default should verify commitments")
    expect(defaultConfig.verifyStateHashes == true, "Default should verify state hashes")
    expect(defaultConfig.verifyLagrangeBasis == true, "Default should verify Lagrange basis")

    let custom = ProtogalaxyVerifierConfig(useGPU: false, gpuThreshold: 1024,
                                             verifyCommitments: false, verifyStateHashes: false,
                                             verifyLagrangeBasis: false)
    expect(custom.useGPU == false, "Custom GPU flag")
    expectEqual(custom.gpuThreshold, 1024, "Custom threshold")
    expect(custom.verifyCommitments == false, "Custom commitment flag")
    expect(custom.verifyStateHashes == false, "Custom state hash flag")
    expect(custom.verifyLagrangeBasis == false, "Custom Lagrange flag")
}

private func testConfigSkipCommitments() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let config = ProtogalaxyVerifierConfig(verifyCommitments: false)
    let engine = GPUProtogalaxyVerifierEngine(config: config)
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: folded, proof: proof)
    expect(result.valid, "Should pass with commitments skipped")
    expect(result.commitmentsValid, "Commitments should show valid when skipped")
}

private func testResultTypes() {
    let foldResult = ProtogalaxyFoldVerifyResult(
        valid: true, fPolyConsistent: true, commitmentsValid: true,
        publicInputValid: true, challengesValid: true, errorTermValid: true,
        relaxationValid: true, challenge: Fr.one)
    expect(foldResult.valid, "Result should be valid")
    expect(foldResult.fPolyConsistent, "F poly should be consistent")

    let batchResult = ProtogalaxyBatchVerifyResult(
        allValid: false, perFoldValid: [true, false],
        passCount: 1, failCount: 1, challenges: [Fr.one, Fr.zero])
    expect(!batchResult.allValid, "Batch should not be all valid")
    expectEqual(batchResult.passCount, 1, "One pass")
    expectEqual(batchResult.failCount, 1, "One fail")
    expectEqual(batchResult.challenges.count, 2, "Two challenges")

    let chainResult = ProtogalaxyChainVerifyResult(
        valid: true, stepsVerified: 3, failingStep: -1,
        finalStateHash: Fr.one, challenges: [Fr.one],
        perStepResults: [])
    expect(chainResult.valid, "Chain should be valid")
    expectEqual(chainResult.stepsVerified, 3, "3 steps verified")
    expectEqual(chainResult.failingStep, -1, "No failing step")
}

private func testVersionExists() {
    let v = GPUProtogalaxyVerifierEngine.version
    expect(v.version == "1.0.0", "Version should be 1.0.0")
}

private func testEdgeCaseTwoInstances() {
    // Minimum valid fold: exactly 2 instances
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(42)],
                                      beta: frFromInt(7), gamma: frFromInt(13))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(99)],
                                      beta: frFromInt(17), gamma: frFromInt(23),
                                      scaleFactor: frFromInt(5))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 77)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 88)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    let engine = GPUProtogalaxyVerifierEngine()
    let ok = engine.verifyFold(instances: [inst0, inst1], folded: folded, proof: proof)
    expect(ok, "2-instance fold (minimum) should verify")

    // Verify the folded instance is relaxed
    expect(folded.isRelaxed, "Folded instance should be relaxed")
}

private func testFoldVerificationRejectsBadRelaxation() {
    let circuitSize = 4
    let inst0 = makeVerifierInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
    let inst1 = makeVerifierInstance(publicInput: [frFromInt(4)],
                                      beta: frFromInt(5), gamma: frFromInt(6),
                                      scaleFactor: frFromInt(2))
    let wit0 = makeVerifierWitness(size: circuitSize, seed: 10)
    let wit1 = makeVerifierWitness(size: circuitSize, seed: 20)

    let (folded, proof) = foldPair(inst0: inst0, inst1: inst1,
                                    wit0: wit0, wit1: wit1, circuitSize: circuitSize)

    // Tamper: wrong u
    let tampered = ProtogalaxyInstance(
        witnessCommitments: folded.witnessCommitments,
        publicInput: folded.publicInput,
        beta: folded.beta,
        gamma: folded.gamma,
        errorTerm: folded.errorTerm,
        u: frAdd(folded.u, Fr.one))

    let engine = GPUProtogalaxyVerifierEngine()
    let result = engine.verifyFoldDetailed(instances: [inst0, inst1],
                                            folded: tampered, proof: proof)
    expect(!result.valid, "Tampered u should fail")
    expect(!result.relaxationValid, "Relaxation check should fail specifically")
}

private func testFreshInstancesHaveZeroError() {
    let inst = makeVerifierInstance(publicInput: [frFromInt(1)],
                                     beta: frFromInt(2), gamma: frFromInt(3))
    expect(frEq(inst.errorTerm, Fr.zero), "Fresh instance should have zero error term")
    expect(frEq(inst.u, Fr.one), "Fresh instance should have u = 1")
    expect(!inst.isRelaxed, "Fresh instance should not be relaxed")

    // Also test mismatched instance count
    let engine = GPUProtogalaxyVerifierEngine()
    let proof = ProtogalaxyFoldingProof(fCoefficients: [Fr.zero, Fr.zero], instanceCount: 3)
    let ok = engine.verifyFold(instances: [inst], folded: inst, proof: proof)
    expect(!ok, "Mismatched instance count should fail")
}

// MARK: - Test Runner

public func runGPUProtogalaxyVerifierTests() {
    suite("GPU Protogalaxy Verifier Engine")

    testBasicFoldVerification()
    testDetailedFoldVerification()
    testFoldVerificationRejectsTampered()
    testFoldVerificationRejectsBadPublicInput()
    testFoldVerificationRejectsBadChallenges()
    testVanishingPolyEvaluation()
    testVerifyVanishingPoly()
    testLagrangePartitionOfUnity()
    testLagrangeAtDomainPoints()
    testLagrangeBasisComputation()
    testChallengeRederivation()
    testChallengeConsistencyVerification()
    testChallengeDiffersForDifferentInstances()
    testBatchVerification()
    testBatchVerificationWithInvalid()
    testIVCChainVerification()
    testIVCChainRejectsTamperedFinal()
    testSingleInstanceChain()
    testErrorTermComputation()
    testErrorPropagationVerification()
    testStateHashing()
    testChainedStateHash()
    testGPUAvailability()
    testGPUInnerProduct()
    testDeciderProofVerification()
    testMultiplePublicInputs()
    testLargerCircuitSize()
    testConfigDefaults()
    testConfigSkipCommitments()
    testResultTypes()
    testVersionExists()
    testEdgeCaseTwoInstances()
    testFoldVerificationRejectsBadRelaxation()
    testFreshInstancesHaveZeroError()
}
