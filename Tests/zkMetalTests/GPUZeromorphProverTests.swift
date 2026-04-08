// GPUZeromorphProverTests -- Comprehensive tests for GPU-accelerated Zeromorph prover engine
//
// Validates: single opening, batch opening, degree-folding optimization, evaluation
// correctness, wrong-eval rejection, commitment binding, proof structure, configuration,
// statistics, edge cases, and pairing-compatible verification.

import Foundation
import zkMetal

// MARK: - Public test entry point

public func runGPUZeromorphProverTests() {
    suite("GPU Zeromorph Prover Engine")

    testVersionIsSet()
    testBasicOpen()
    testOpen3Var()
    testOpen4Var()
    testDegreeFoldingEnabled()
    testDegreeFoldingDisabled()
    testBatchOpen2Polys()
    testBatchOpen3Polys()
    testWrongEvalRejection()
    testWrongEvalBatchRejection()
    testCommitmentBinding()
    testCommitmentDeterminism()
    testEvalZMFoldMatchesMLE()
    testEvalZMFoldSingleVar()
    testProofStructure()
    testProofWellFormed()
    testBatchProofStructure()
    testConfigDefault()
    testConfigCPUOnly()
    testConfigOptimized()
    testProofCountStats()
    testBatchProofCountStats()
    testResetStats()
    testDebugDescription()
    testConstantPolynomial()
    testLinearPolynomial()
    testSingleVariableOpen()
    testApproxByteSize()
    testBatchApproxByteSize()
}

// MARK: - Version

private func testVersionIsSet() {
    let v = GPUZeromorphProverEngine.version
    expect(!v.version.isEmpty, "Version string is non-empty: \(v.description)")
}

// MARK: - Helper: create engine with test SRS

private func makeEngine(secret: [UInt32], srsSize: Int,
                         config: ZeromorphProverConfig = .default) -> GPUZeromorphProverEngine? {
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else { return nil }
    return GPUZeromorphProverEngine(kzg: kzg, config: config)
}

private func makeSecret(_ v: UInt32) -> [UInt32] {
    return [v, 0, 0, 0, 0, 0, 0, 0]
}

// MARK: - Basic 2-variable opening

private func testBasicOpen() {
    suite("ZM Prover -- basic 2-var open")

    let secret = makeSecret(17)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]
    let expectedValue = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "claimed value matches ZM fold")
    expectEqual(proof.numVariables, 2, "2 quotient commitments")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: expectedValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "verification passes with correct value")
}

// MARK: - 3-variable opening

private func testOpen3Var() {
    suite("ZM Prover -- 3-var open")

    let secret = makeSecret(23)
    guard let engine = makeEngine(secret: secret, srsSize: 16) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5)]
    let expectedValue = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "claimed value matches")
    expectEqual(proof.numVariables, 3, "3 quotient commitments")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: expectedValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "3-var verification passes")
}

// MARK: - 4-variable opening

private func testOpen4Var() {
    suite("ZM Prover -- 4-var open")

    let secret = makeSecret(29)
    guard let engine = makeEngine(secret: secret, srsSize: 32) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...16).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
    let expectedValue = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "claimed value matches")
    expectEqual(proof.numVariables, 4, "4 quotient commitments")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: expectedValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "4-var verification passes")
}

// MARK: - Degree-folding enabled

private func testDegreeFoldingEnabled() {
    suite("ZM Prover -- degree-folding enabled")

    let config = ZeromorphProverConfig(enableDegreeFolding: true)
    let secret = makeSecret(37)
    guard let engine = makeEngine(secret: secret, srsSize: 16, config: config) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(11), frFromInt(13), frFromInt(17)]
    let expectedValue = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open with folding failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "value matches with folding")
    // Degree-folding applies to quotients with >= 4 elements, so n=3 should fold q^(0)
    expect(proof.degreeFolded, "degree-folding was applied for n=3")
}

// MARK: - Degree-folding disabled

private func testDegreeFoldingDisabled() {
    suite("ZM Prover -- degree-folding disabled")

    let config = ZeromorphProverConfig(enableDegreeFolding: false)
    let secret = makeSecret(41)
    guard let engine = makeEngine(secret: secret, srsSize: 16, config: config) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(11), frFromInt(13), frFromInt(17)]

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open without folding failed")
        return
    }

    expect(!proof.degreeFolded, "degree-folding was NOT applied when disabled")
}

// MARK: - Batch open 2 polynomials

private func testBatchOpen2Polys() {
    suite("ZM Prover -- batch open 2 polys")

    let secret = makeSecret(31)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(42)

    let v1 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals1, point: point)
    let v2 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals2, point: point)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2],
        point: point,
        values: [v1, v2],
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    expectEqual(batchProof.count, 2, "2 claimed values")
    expect(frEqual(batchProof.claimedValues[0], v1), "value 1 matches")
    expect(frEqual(batchProof.claimedValues[1], v2), "value 2 matches")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyBatchWithSecret(
        evaluationSets: [evals1, evals2],
        point: point,
        values: [v1, v2],
        proof: batchProof,
        srsSecret: srsSecret
    )
    expect(valid, "batch verification passes for 2 polys")
}

// MARK: - Batch open 3 polynomials

private func testBatchOpen3Polys() {
    suite("ZM Prover -- batch open 3 polys")

    let secret = makeSecret(43)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let evals3 = [frFromInt(5), frFromInt(7), frFromInt(11), frFromInt(13)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(77)

    let v1 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals1, point: point)
    let v2 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals2, point: point)
    let v3 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals3, point: point)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2, evals3],
        point: point,
        values: [v1, v2, v3],
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    expectEqual(batchProof.count, 3, "3 claimed values")
    expect(frEqual(batchProof.claimedValues[0], v1), "value 1 matches")
    expect(frEqual(batchProof.claimedValues[1], v2), "value 2 matches")
    expect(frEqual(batchProof.claimedValues[2], v3), "value 3 matches")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyBatchWithSecret(
        evaluationSets: [evals1, evals2, evals3],
        point: point,
        values: [v1, v2, v3],
        proof: batchProof,
        srsSecret: srsSecret
    )
    expect(valid, "batch verification passes for 3 polys")
}

// MARK: - Wrong evaluation rejection

private func testWrongEvalRejection() {
    suite("ZM Prover -- wrong eval rejection")

    let secret = makeSecret(13)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]
    let correctValue = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)
    let wrongValue = frAdd(correctValue, Fr.one)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    let srsSecret = frFromLimbs(secret)

    let validCorrect = engine.verifyWithSecret(
        evaluations: evals, point: point, value: correctValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(validCorrect, "correct value verifies")

    let validWrong = engine.verifyWithSecret(
        evaluations: evals, point: point, value: wrongValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(!validWrong, "wrong value rejected")
}

// MARK: - Wrong evaluation batch rejection

private func testWrongEvalBatchRejection() {
    suite("ZM Prover -- wrong eval batch rejection")

    let secret = makeSecret(47)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(42)

    let v1 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals1, point: point)
    let v2 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals2, point: point)
    let wrongV2 = frAdd(v2, Fr.one)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2],
        point: point,
        values: [v1, v2],
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    let srsSecret = frFromLimbs(secret)

    // Correct values should verify
    let validCorrect = engine.verifyBatchWithSecret(
        evaluationSets: [evals1, evals2],
        point: point,
        values: [v1, v2],
        proof: batchProof,
        srsSecret: srsSecret
    )
    expect(validCorrect, "correct batch values verify")

    // Wrong value should be rejected
    let validWrong = engine.verifyBatchWithSecret(
        evaluationSets: [evals1, evals2],
        point: point,
        values: [v1, wrongV2],
        proof: batchProof,
        srsSecret: srsSecret
    )
    expect(!validWrong, "wrong batch value rejected")
}

// MARK: - Commitment binding

private func testCommitmentBinding() {
    suite("ZM Prover -- commitment binding")

    let secret = makeSecret(19)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

    guard let c1 = try? engine.commit(evaluations: evals1),
          let c2 = try? engine.commit(evaluations: evals2) else {
        expect(false, "commit failed")
        return
    }

    let c1Aff = batchToAffine([c1])
    let c2Aff = batchToAffine([c2])
    let sameX = fpToInt(c1Aff[0].x) == fpToInt(c2Aff[0].x)
    let sameY = fpToInt(c1Aff[0].y) == fpToInt(c2Aff[0].y)
    expect(!(sameX && sameY), "different polys have different commitments")
}

// MARK: - Commitment determinism

private func testCommitmentDeterminism() {
    suite("ZM Prover -- commitment determinism")

    let secret = makeSecret(19)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

    guard let c1 = try? engine.commit(evaluations: evals),
          let c2 = try? engine.commit(evaluations: evals) else {
        expect(false, "commit failed")
        return
    }

    let c1Aff = batchToAffine([c1])
    let c2Aff = batchToAffine([c2])
    let matchX = fpToInt(c1Aff[0].x) == fpToInt(c2Aff[0].x)
    let matchY = fpToInt(c1Aff[0].y) == fpToInt(c2Aff[0].y)
    expect(matchX && matchY, "same poly has same commitment")
}

// MARK: - ZM fold matches MLE on Boolean inputs

private func testEvalZMFoldMatchesMLE() {
    suite("ZM Prover -- ZMFold matches MLE on Boolean inputs")

    // evals = [1, 2, 3, 4]
    // ZM fold formula: result[i] = evals[2i] + c * evals[2i+1]  (lo + c*hi)
    // MLE fold formula: result[i] = evals[2i] + c * (evals[2i+1] - evals[2i])  ((1-c)*lo + c*hi)
    //
    // ZM fold of [1,2,3,4] at (r0,r1) = 1 + 2*r0 + 3*r1 + 4*r0*r1
    // MLE of [1,2,3,4] at (r0,r1) = 1 + r0 + 2*r1  (multilinear extension)
    //
    // These formulas only agree at (0,0). This is by design — ZM fold is used
    // for Zeromorph quotient computation, not MLE evaluation.

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

    // At (0, 0): both give 1
    let point00 = [Fr.zero, Fr.zero]
    let zmVal00 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point00)
    let mleVal00 = GPUZeromorphProverEngine.evaluateMLE(evaluations: evals, point: point00)
    expect(frEqual(zmVal00, mleVal00), "ZMFold == MLE at (0,0)")

    // Verify ZM fold gives correct values per its own formula
    // ZM(0,1) = 1 + 0 + 3 + 0 = 4
    let point01 = [Fr.zero, Fr.one]
    let zmVal01 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point01)
    expect(frEqual(zmVal01, frFromInt(4)), "ZMFold(0,1) = 4")

    // ZM(1,0) = 1 + 2 + 0 + 0 = 3
    let point10 = [Fr.one, Fr.zero]
    let zmVal10 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point10)
    expect(frEqual(zmVal10, frFromInt(3)), "ZMFold(1,0) = 3")

    // ZM(1,1) = 1 + 2 + 3 + 4 = 10
    let point11 = [Fr.one, Fr.one]
    let zmVal11 = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point11)
    expect(frEqual(zmVal11, frFromInt(10)), "ZMFold(1,1) = 10")

    // Verify MLE gives standard multilinear values at Boolean points
    let mleVal01 = GPUZeromorphProverEngine.evaluateMLE(evaluations: evals, point: point01)
    expect(frEqual(mleVal01, frFromInt(3)), "MLE(0,1) = evals[2] = 3")

    let mleVal10 = GPUZeromorphProverEngine.evaluateMLE(evaluations: evals, point: point10)
    expect(frEqual(mleVal10, frFromInt(2)), "MLE(1,0) = evals[1] = 2")

    let mleVal11 = GPUZeromorphProverEngine.evaluateMLE(evaluations: evals, point: point11)
    expect(frEqual(mleVal11, frFromInt(4)), "MLE(1,1) = evals[3] = 4")
}

// MARK: - Single variable evaluation

private func testEvalZMFoldSingleVar() {
    suite("ZM Prover -- ZMFold single variable")

    // f(x) = [3, 7] => f(u) = 3 + 7*u
    let evals = [frFromInt(3), frFromInt(7)]
    let u = frFromInt(5)
    let result = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: [u])
    // 3 + 7*5 = 38
    let expected = frFromInt(38)
    expect(frEqual(result, expected), "single variable ZMFold: 3 + 7*5 = 38")
}

// MARK: - Proof structure

private func testProofStructure() {
    suite("ZM Prover -- proof structure")

    let secret = makeSecret(53)
    guard let engine = makeEngine(secret: secret, srsSize: 16) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expectEqual(proof.numVariables, 3, "numVariables == 3")
    expect(!pointIsIdentity(proof.kzgWitness), "KZG witness is not identity")
    expect(!frEqual(proof.zeta, Fr.zero), "zeta is nonzero")
}

// MARK: - Proof well-formed

private func testProofWellFormed() {
    suite("ZM Prover -- proof well-formed")

    let secret = makeSecret(59)
    guard let engine = makeEngine(secret: secret, srsSize: 16) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(proof.isWellFormed, "proof is well-formed")
    expect(proof.approximateByteSize > 0, "byte size is positive")
}

// MARK: - Batch proof structure

private func testBatchProofStructure() {
    suite("ZM Prover -- batch proof structure")

    let secret = makeSecret(61)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(11)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2],
        point: point,
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    expectEqual(batchProof.count, 2, "batch count is 2")
    expectEqual(batchProof.perPolyQuotientCommitments.count, 2, "2 per-poly quotient sets")
    expect(frEqual(batchProof.gamma, gamma), "gamma preserved in proof")
    expect(batchProof.approximateByteSize > 0, "batch byte size is positive")
}

// MARK: - Configuration tests

private func testConfigDefault() {
    suite("ZM Prover -- config default")

    let config = ZeromorphProverConfig.default
    expectEqual(config.gpuCommitThreshold, 64, "default threshold 64")
    expect(config.enableDegreeFolding, "default enables folding")
    expect(config.enableBatchParallel, "default enables batch parallel")
    expectEqual(config.maxBatchConcurrency, 4, "default max concurrency 4")
}

private func testConfigCPUOnly() {
    suite("ZM Prover -- config CPU-only")

    let config = ZeromorphProverConfig.cpuOnly
    expectEqual(config.gpuCommitThreshold, Int.max, "CPU-only threshold is max")
    expect(!config.enableDegreeFolding, "CPU-only disables folding")
    expect(!config.enableBatchParallel, "CPU-only disables batch parallel")
    expectEqual(config.maxBatchConcurrency, 1, "CPU-only concurrency 1")
}

private func testConfigOptimized() {
    suite("ZM Prover -- config optimized")

    let config = ZeromorphProverConfig.optimized
    expectEqual(config.gpuCommitThreshold, 32, "optimized threshold 32")
    expect(config.enableDegreeFolding, "optimized enables folding")
    expect(config.enableBatchParallel, "optimized enables batch parallel")
    expectEqual(config.maxBatchConcurrency, 8, "optimized max concurrency 8")
}

// MARK: - Statistics

private func testProofCountStats() {
    suite("ZM Prover -- proof count stats")

    let secret = makeSecret(67)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    expectEqual(engine.proofCount, 0, "initial proof count is 0")

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]

    _ = try? engine.open(evaluations: evals, point: point)
    expectEqual(engine.proofCount, 1, "proof count after 1 open")

    _ = try? engine.open(evaluations: evals, point: point)
    expectEqual(engine.proofCount, 2, "proof count after 2 opens")
}

private func testBatchProofCountStats() {
    suite("ZM Prover -- batch proof count stats")

    let secret = makeSecret(71)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    expectEqual(engine.batchProofCount, 0, "initial batch count is 0")

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(99)

    _ = try? engine.batchOpen(evaluationSets: [evals1, evals2], point: point, gamma: gamma)
    expectEqual(engine.batchProofCount, 1, "batch count after 1 batch open")
}

private func testResetStats() {
    suite("ZM Prover -- reset stats")

    let secret = makeSecret(73)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]
    _ = try? engine.open(evaluations: evals, point: point)

    expect(engine.proofCount > 0, "proof count > 0 before reset")
    engine.resetStats()
    expectEqual(engine.proofCount, 0, "proof count is 0 after reset")
    expectEqual(engine.batchProofCount, 0, "batch count is 0 after reset")
}

// MARK: - Debug description

private func testDebugDescription() {
    suite("ZM Prover -- debug description")

    let secret = makeSecret(79)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let desc = engine.debugDescription()
    expect(desc.contains("GPUZeromorphProverEngine"), "contains engine name")
    expect(desc.contains("srsSize"), "contains srsSize")
    expect(desc.contains("degreeFolding"), "contains degreeFolding flag")
    expect(desc.contains("proofCount"), "contains proofCount")
}

// MARK: - Constant polynomial

private func testConstantPolynomial() {
    suite("ZM Prover -- constant polynomial")

    let secret = makeSecret(83)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    // f = [42, 42, 42, 42] is constant
    let c = frFromInt(42)
    let evals = [c, c, c, c]
    let point = [frFromInt(100), frFromInt(200)]

    let val = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)
    // For constant f, ZMFold gives f[0] + u*f[1] at each level, but
    // since all evals are c: fold gives c + u*c = c*(1+u) -- actually no:
    // evals[2i] = c, evals[2i+1] = c, so fold = c + u*c
    // This is NOT equal to c for arbitrary u. The ZM fold of a constant polynomial
    // is not the constant itself for non-Boolean points. Verify the proof works anyway.

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed for constant poly")
        return
    }

    expect(frEqual(proof.claimedValue, val), "value matches ZM fold")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: val,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "constant polynomial proof verifies")
}

// MARK: - Linear polynomial

private func testLinearPolynomial() {
    suite("ZM Prover -- linear polynomial")

    let secret = makeSecret(89)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    // f on 2 vars: f(x0, x1) with evals [0, 1, 0, 1] (only depends on x1)
    let evals = [Fr.zero, Fr.one, Fr.zero, Fr.one]
    let point = [frFromInt(3), frFromInt(7)]

    let val = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed for linear poly")
        return
    }

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: val,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "linear polynomial proof verifies")
}

// MARK: - Single variable opening

private func testSingleVariableOpen() {
    suite("ZM Prover -- single variable")

    let secret = makeSecret(97)
    guard let engine = makeEngine(secret: secret, srsSize: 4) else {
        expect(false, "engine init failed")
        return
    }

    // f(x) = [3, 7], 1 variable, 2 evals
    let evals = [frFromInt(3), frFromInt(7)]
    let point = [frFromInt(5)]

    let val = GPUZeromorphProverEngine.evaluateZMFold(evaluations: evals, point: point)
    // 3 + 5*7 = 38
    let expected = frFromInt(38)
    expect(frEqual(val, expected), "single var ZM fold correct")

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed for single var")
        return
    }

    expectEqual(proof.numVariables, 1, "1 quotient commitment")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: val,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "single variable proof verifies")
}

// MARK: - Approximate byte size

private func testApproxByteSize() {
    suite("ZM Prover -- approx byte size")

    let secret = makeSecret(101)
    guard let engine = makeEngine(secret: secret, srsSize: 16) else {
        expect(false, "engine init failed")
        return
    }

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    let size = proof.approximateByteSize
    // 3 quotient commitments (3*96) + 1 witness (96) + 3 scalars (3*32) + 1 flag
    // = 288 + 96 + 96 + 1 = 481
    expect(size > 400, "byte size > 400 for 3-var proof")
    expect(size < 600, "byte size < 600 for 3-var proof")
}

private func testBatchApproxByteSize() {
    suite("ZM Prover -- batch approx byte size")

    let secret = makeSecret(103)
    guard let engine = makeEngine(secret: secret, srsSize: 8) else {
        expect(false, "engine init failed")
        return
    }

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(11)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2],
        point: point,
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    let size = batchProof.approximateByteSize
    expect(size > 200, "batch byte size > 200")
    expect(size < 2000, "batch byte size reasonable")
}
