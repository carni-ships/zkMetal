import zkMetal
import Foundation

// MARK: - Public test entry point

public func runGPUZeromorphTests() {
    testGPUZeromorphSimple2Var()
    testGPUZeromorphMultiVar()
    testGPUZeromorphBatchOpening()
    testGPUZeromorphWrongEvalRejection()
    testGPUZeromorphCommitmentBinding()
}

// MARK: - Simple 2-variable opening

/// f(x0, x1) on {0,1}^2 = [1, 2, 3, 4]
/// Open at point (u0, u1) = (5, 7) using ZM fold.
private func testGPUZeromorphSimple2Var() {
    suite("GPUZeromorph simple 2-var")

    // Set up KZG with test SRS
    let secret: [UInt32] = [17, 0, 0, 0, 0, 0, 0, 0]
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 8, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else {
        expect(false, "KZG init failed")
        return
    }
    let engine = GPUZeromorphEngine(kzg: kzg)

    // f = [1, 2, 3, 4] (4 evaluations = 2 variables)
    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]

    // Compute expected value
    let expectedValue = GPUZeromorphEngine.evaluateZMFold(evaluations: evals, point: point)

    // Open
    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "claimed value matches ZM fold")
    expect(proof.numVariables == 2, "2 quotient commitments")

    // Verify with secret
    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: expectedValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "verification passes with correct value")
}

// MARK: - Multi-variable opening (3 vars)

/// f on {0,1}^3 = [1,2,3,4,5,6,7,8], point = (2,3,5)
private func testGPUZeromorphMultiVar() {
    suite("GPUZeromorph multi-var (3 vars)")

    let secret: [UInt32] = [23, 0, 0, 0, 0, 0, 0, 0]
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 16, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else {
        expect(false, "KZG init failed")
        return
    }
    let engine = GPUZeromorphEngine(kzg: kzg)

    let evals = (1...8).map { frFromInt(UInt64($0)) }
    let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

    let expectedValue = GPUZeromorphEngine.evaluateZMFold(evaluations: evals, point: point)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    expect(frEqual(proof.claimedValue, expectedValue), "claimed value matches")
    expect(proof.numVariables == 3, "3 quotient commitments")

    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyWithSecret(
        evaluations: evals, point: point, value: expectedValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(valid, "3-var verification passes")
}

// MARK: - Batch opening

/// Open 3 polynomials at the same point using random linear combination.
private func testGPUZeromorphBatchOpening() {
    suite("GPUZeromorph batch opening")

    let secret: [UInt32] = [31, 0, 0, 0, 0, 0, 0, 0]
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 8, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else {
        expect(false, "KZG init failed")
        return
    }
    let engine = GPUZeromorphEngine(kzg: kzg)

    // 3 polynomials, 2 variables each (4 evals each)
    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let evals3 = [frFromInt(5), frFromInt(7), frFromInt(11), frFromInt(13)]

    let point = [frFromInt(3), frFromInt(9)]
    let gamma = frFromInt(42)

    let v1 = GPUZeromorphEngine.evaluateZMFold(evaluations: evals1, point: point)
    let v2 = GPUZeromorphEngine.evaluateZMFold(evaluations: evals2, point: point)
    let v3 = GPUZeromorphEngine.evaluateZMFold(evaluations: evals3, point: point)

    guard let batchProof = try? engine.batchOpen(
        evaluationSets: [evals1, evals2, evals3],
        point: point,
        values: [v1, v2, v3],
        gamma: gamma
    ) else {
        expect(false, "batch open failed")
        return
    }

    expect(batchProof.claimedValues.count == 3, "3 claimed values")
    expect(frEqual(batchProof.claimedValues[0], v1), "value 1 matches")
    expect(frEqual(batchProof.claimedValues[1], v2), "value 2 matches")
    expect(frEqual(batchProof.claimedValues[2], v3), "value 3 matches")

    // Verify batch
    let srsSecret = frFromLimbs(secret)
    let valid = engine.verifyBatchWithSecret(
        evaluationSets: [evals1, evals2, evals3],
        point: point,
        values: [v1, v2, v3],
        proof: batchProof,
        srsSecret: srsSecret
    )
    expect(valid, "batch verification passes")
}

// MARK: - Wrong evaluation rejection

/// Proof should fail verification if the claimed value is wrong.
private func testGPUZeromorphWrongEvalRejection() {
    suite("GPUZeromorph wrong eval rejection")

    let secret: [UInt32] = [13, 0, 0, 0, 0, 0, 0, 0]
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 8, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else {
        expect(false, "KZG init failed")
        return
    }
    let engine = GPUZeromorphEngine(kzg: kzg)

    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(5), frFromInt(7)]

    let correctValue = GPUZeromorphEngine.evaluateZMFold(evaluations: evals, point: point)
    let wrongValue = frAdd(correctValue, Fr.one)

    guard let proof = try? engine.open(evaluations: evals, point: point) else {
        expect(false, "open failed")
        return
    }

    let srsSecret = frFromLimbs(secret)

    // Correct value should verify
    let validCorrect = engine.verifyWithSecret(
        evaluations: evals, point: point, value: correctValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(validCorrect, "correct value verifies")

    // Wrong value should NOT verify (value mismatch detected early)
    let validWrong = engine.verifyWithSecret(
        evaluations: evals, point: point, value: wrongValue,
        proof: proof, srsSecret: srsSecret
    )
    expect(!validWrong, "wrong value rejected")
}

// MARK: - Commitment binding

/// Different polynomials should produce different commitments.
private func testGPUZeromorphCommitmentBinding() {
    suite("GPUZeromorph commitment binding")

    let secret: [UInt32] = [19, 0, 0, 0, 0, 0, 0, 0]
    let gen = bn254G1Generator()
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 8, generator: gen)
    guard let kzg = try? KZGEngine(srs: srs) else {
        expect(false, "KZG init failed")
        return
    }
    let engine = GPUZeromorphEngine(kzg: kzg)

    let evals1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let evals2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

    guard let c1 = try? engine.commit(evaluations: evals1),
          let c2 = try? engine.commit(evaluations: evals2) else {
        expect(false, "commit failed")
        return
    }

    // Different polynomials => different commitments
    let c1Aff = batchToAffine([c1])
    let c2Aff = batchToAffine([c2])

    let sameX = fpToInt(c1Aff[0].x) == fpToInt(c2Aff[0].x)
    let sameY = fpToInt(c1Aff[0].y) == fpToInt(c2Aff[0].y)
    expect(!(sameX && sameY), "different polys have different commitments")

    // Same polynomial => same commitment
    guard let c1Again = try? engine.commit(evaluations: evals1) else {
        expect(false, "recommit failed")
        return
    }
    let c1AgainAff = batchToAffine([c1Again])
    let matchX = fpToInt(c1Aff[0].x) == fpToInt(c1AgainAff[0].x)
    let matchY = fpToInt(c1Aff[0].y) == fpToInt(c1AgainAff[0].y)
    expect(matchX && matchY, "same poly has same commitment")
}
