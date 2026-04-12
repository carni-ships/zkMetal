// GPUBasefoldProverTests — Comprehensive tests for GPU-accelerated Basefold prover engine
//
// Tests commitment, opening, verification, batch operations, univariate mode,
// RS encoding, proximity testing, tamper detection, and edge cases.

import zkMetal
import Foundation

// MARK: - Public test entry point

public func runGPUBasefoldProverTests() {
    suite("GPU Basefold Prover Engine")
    testBasicCommitOpen()
    testTwoVarCommitOpen()
    testThreeVarCommitOpen()
    testFourVarCommitOpen()
    testEvaluationCorrectness()
    testMultilinearEvaluation()
    testUnivariateEvaluation()
    testRSEncodeRateOne()
    testRSEncodeBlowupTwo()
    testCommitmentDeterminism()
    testVerificationPasses()
    testWrongValueRejected()
    testWrongPointRejected()
    testBasefoldTamperedProofRejected()
    testProximityTest()
    testBatchCommitTwoPolys()
    testBatchCommitThreePolys()
    testBatchOpenVerify()
    testBatchWrongValueRejected()
    testUnivariateCommitOpen()
    testUnivariateLinearPoly()
    testUnivariateConstantPoly()
    testCommitmentStats()
    testProofStats()
    testConfigStandard()
    testConfigFast()
    testSingleEvalPoly()
    testLargerPolynomial()
    testFoldConsistency()
    testMerklePathVerification()
    testQueryProofStructure()
    testAllOnesPolynomial()
    testAllZerosPolynomial()
    testIdentityPolynomial()
}

// MARK: - Basic Commit/Open Tests

private func testBasicCommitOpen() {
    do {
        let engine = try GPUBasefoldProverEngine()
        // f(x) = 3 + 5*x on {0,1}^1 => evals = [3, 8]
        let evals = [frFromInt(3), frFromInt(8)]
        let commitment = try engine.commit(evaluations: evals)

        expect(!frEqual(commitment.root, Fr.zero), "commit: non-zero root")
        expectEqual(commitment.numVars, 1, "commit: 1 variable")
        expectEqual(commitment.evaluations.count, 2, "commit: 2 evaluations")

        // Open at point [7] => f(7) = 3 + 5*7 = 38
        let point = [frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expectedValue = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expectedValue), "open: final value matches evaluation")
    } catch {
        expect(false, "basic commit/open: unexpected error \(error)")
    }
}

private func testTwoVarCommitOpen() {
    do {
        let engine = try GPUBasefoldProverEngine()
        // f(x0, x1) on {0,1}^2 = [1, 2, 3, 4]
        let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 2, "2-var: numVars = 2")

        let point = [frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expected), "2-var: correct evaluation")
        // fold-by-4: 2 vars => 1 fused level (stride 2) = 1 committed level
        expect(proof.foldLayers.count == 1, "2-var: 1 fold level (fold-by-4)")
    } catch {
        expect(false, "2-var commit/open: error \(error)")
    }
}

private func testThreeVarCommitOpen() {
    do {
        let engine = try GPUBasefoldProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 3, "3-var: numVars = 3")
        expectEqual(commitment.evaluations.count, 8, "3-var: 8 evaluations")

        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expected), "3-var: correct evaluation")
        // fold-by-4: 3 vars => 1 fused (stride 2) + 1 single (stride 1) = 2 committed levels
        expect(proof.foldLayers.count == 2, "3-var: 2 fold levels (fold-by-4)")
        expect(proof.queries.count > 0, "3-var: has query proofs")
    } catch {
        expect(false, "3-var commit/open: error \(error)")
    }
}

private func testFourVarCommitOpen() {
    do {
        let engine = try GPUBasefoldProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 4, "4-var: numVars = 4")

        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expected), "4-var: correct evaluation")
        // fold-by-4: 4 vars => 2 fused levels (each stride 2) = 2 committed levels
        expectEqual(proof.foldLayers.count, 2, "4-var: 2 fold levels (fold-by-4)")
    } catch {
        expect(false, "4-var commit/open: error \(error)")
    }
}

// MARK: - Evaluation Tests

private func testEvaluationCorrectness() {
    // Test that multilinear evaluation is consistent with manual computation
    // f(x0, x1) = 1*(1-x0)(1-x1) + 2*x0*(1-x1) + 3*(1-x0)*x1 + 4*x0*x1
    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(2), frFromInt(3)]

    let result = GPUBasefoldProverEngine.evaluateMultilinear(
        evaluations: evals, point: point)

    // Manual: fold with x0=2: [1+2*(2-1), 3+2*(4-3)] = [3, 5]
    // fold with x1=3: 3 + 3*(5-3) = 9
    let expected = frFromInt(9)
    expect(frEqual(result, expected), "eval correctness: f(2,3) = 9")
}

private func testMultilinearEvaluation() {
    // f(x0) = a*(1-x0) + b*x0 = a + x0*(b-a)
    let a = frFromInt(10)
    let b = frFromInt(20)
    let evals = [a, b]

    // f(0) = 10
    let val0 = GPUBasefoldProverEngine.evaluateMultilinear(
        evaluations: evals, point: [Fr.zero])
    expect(frEqual(val0, a), "multilinear: f(0) = 10")

    // f(1) = 20
    let val1 = GPUBasefoldProverEngine.evaluateMultilinear(
        evaluations: evals, point: [Fr.one])
    expect(frEqual(val1, b), "multilinear: f(1) = 20")

    // f(3) = 10 + 3*(20-10) = 40
    let val3 = GPUBasefoldProverEngine.evaluateMultilinear(
        evaluations: evals, point: [frFromInt(3)])
    expect(frEqual(val3, frFromInt(40)), "multilinear: f(3) = 40")
}

private func testUnivariateEvaluation() {
    // f(X) = 2 + 3X + 5X^2
    let coeffs = [frFromInt(2), frFromInt(3), frFromInt(5)]

    // f(0) = 2
    let v0 = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: Fr.zero)
    expect(frEqual(v0, frFromInt(2)), "univariate: f(0) = 2")

    // f(1) = 2 + 3 + 5 = 10
    let v1 = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: Fr.one)
    expect(frEqual(v1, frFromInt(10)), "univariate: f(1) = 10")

    // f(2) = 2 + 6 + 20 = 28
    let v2 = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: frFromInt(2))
    expect(frEqual(v2, frFromInt(28)), "univariate: f(2) = 28")

    // f(10) = 2 + 30 + 500 = 532
    let v10 = GPUBasefoldProverEngine.evaluateUnivariate(coeffs: coeffs, at: frFromInt(10))
    expect(frEqual(v10, frFromInt(532)), "univariate: f(10) = 532")
}

// MARK: - RS Encoding Tests

private func testRSEncodeRateOne() {
    do {
        // Rate log = 0 means no blowup
        let engine = try GPUBasefoldProverEngine(config: BasefoldProverConfig(
            numQueries: 8, rateLog: 0, maxGPULogSize: 20))
        let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let encoded = engine.rsEncode(evaluations: evals)

        expectEqual(encoded.count, 4, "rate-1: no blowup, same size")
        for i in 0..<4 {
            expect(frEqual(encoded[i], evals[i]), "rate-1: element \(i) unchanged")
        }
    } catch {
        expect(false, "RS encode rate-1: error \(error)")
    }
}

private func testRSEncodeBlowupTwo() {
    do {
        let engine = try GPUBasefoldProverEngine(config: BasefoldProverConfig(
            numQueries: 8, rateLog: 1, maxGPULogSize: 20))
        let evals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let encoded = engine.rsEncode(evaluations: evals)

        expectEqual(encoded.count, 8, "blowup-2: doubled size")

        // Original evaluations preserved
        for i in 0..<4 {
            expect(frEqual(encoded[i], evals[i]), "blowup-2: original eval \(i) preserved")
        }

        // Extended evaluations should be non-trivial
        var allZero = true
        for i in 4..<8 {
            if !frEqual(encoded[i], Fr.zero) { allZero = false }
        }
        expect(!allZero, "blowup-2: extended evaluations are non-trivial")
    } catch {
        expect(false, "RS encode blowup-2: error \(error)")
    }
}

// MARK: - Commitment Properties

private func testCommitmentDeterminism() {
    do {
        let engine = try GPUBasefoldProverEngine()
        let evals = [frFromInt(42), frFromInt(99), frFromInt(7), frFromInt(13)]

        let c1 = try engine.commit(evaluations: evals)
        let c2 = try engine.commit(evaluations: evals)

        expect(frEqual(c1.root, c2.root), "determinism: same input gives same root")
        expectEqual(c1.numVars, c2.numVars, "determinism: same numVars")
        expectEqual(c1.evaluations.count, c2.evaluations.count, "determinism: same eval count")
    } catch {
        expect(false, "commitment determinism: error \(error)")
    }
}

// MARK: - Verification Tests

private func testVerificationPasses() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let proof = try engine.open(commitment: commitment, point: point)
        let claimedValue = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)

        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: claimedValue, proof: proof)
        expect(valid, "verification: correct proof passes")
    } catch {
        expect(false, "verification passes: error \(error)")
    }
}

private func testWrongValueRejected() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...4).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(5), frFromInt(7)]

        let proof = try engine.open(commitment: commitment, point: point)

        // Try to verify with wrong claimed value
        let wrongValue = frFromInt(999999)
        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: wrongValue, proof: proof)
        expect(!valid, "wrong value: verification rejects incorrect claimed value")
    } catch {
        expect(false, "wrong value rejected: error \(error)")
    }
}

private func testWrongPointRejected() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...4).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(5), frFromInt(7)]

        let proof = try engine.open(commitment: commitment, point: point)
        let correctValue = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)

        // Verify at wrong point — claimed value won't match final value
        let wrongPoint = [frFromInt(100), frFromInt(200)]
        let valid = engine.verify(
            root: commitment.root, point: wrongPoint,
            claimedValue: correctValue, proof: proof)
        // This should fail because the fold values in the proof were computed
        // with point [5,7], not [100,200]
        expect(!valid, "wrong point: verification rejects mismatched point")
    } catch {
        expect(false, "wrong point rejected: error \(error)")
    }
}

private func testBasefoldTamperedProofRejected() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...4).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(5), frFromInt(7)]

        let proof = try engine.open(commitment: commitment, point: point)

        // Tamper with the final value
        let tamperedValue = frAdd(proof.finalValue, Fr.one)
        let tamperedProof = BasefoldProverProof(
            foldRoots: proof.foldRoots,
            finalValue: tamperedValue,
            foldLayers: proof.foldLayers,
            queries: proof.queries,
            point: proof.point
        )

        let claimedValue = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: claimedValue, proof: tamperedProof)
        expect(!valid, "tampered proof: verification rejects modified final value")
    } catch {
        expect(false, "tampered proof rejected: error \(error)")
    }
}

// MARK: - Proximity Test

private func testProximityTest() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let proof = try engine.open(commitment: commitment, point: point)
        let passes = engine.proximityTest(commitment: commitment, proof: proof)
        expect(passes, "proximity test: honest proof passes")
    } catch {
        expect(false, "proximity test: error \(error)")
    }
}

// MARK: - Batch Tests

private func testBatchCommitTwoPolys() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let poly1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let poly2 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])

        expectEqual(batch.commitments.count, 2, "batch 2: two commitments")
        expect(!frEqual(batch.batchRoot, Fr.zero), "batch 2: non-zero batch root")

        // Individual roots should differ (different polynomials)
        expect(!frEqual(batch.commitments[0].root, batch.commitments[1].root),
               "batch 2: different roots for different polys")
    } catch {
        expect(false, "batch commit 2 polys: error \(error)")
    }
}

private func testBatchCommitThreePolys() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let polys = (0..<3).map { i in
            (0..<4).map { j in frFromInt(UInt64(i * 4 + j + 1)) }
        }

        let batch = try engine.batchCommit(polynomials: polys)

        expectEqual(batch.commitments.count, 3, "batch 3: three commitments")
        expect(!frEqual(batch.batchRoot, Fr.zero), "batch 3: non-zero batch root")

        // All roots should be distinct
        let r0 = batch.commitments[0].root
        let r1 = batch.commitments[1].root
        let r2 = batch.commitments[2].root
        expect(!frEqual(r0, r1), "batch 3: root 0 != root 1")
        expect(!frEqual(r1, r2), "batch 3: root 1 != root 2")
        expect(!frEqual(r0, r2), "batch 3: root 0 != root 2")
    } catch {
        expect(false, "batch commit 3 polys: error \(error)")
    }
}

private func testBatchOpenVerify() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let poly1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let poly2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let point = [frFromInt(3), frFromInt(5)]

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])
        let batchProof = try engine.batchOpen(batch: batch, point: point)

        expectEqual(batchProof.proofs.count, 2, "batch open: 2 proofs")
        expect(!frEqual(batchProof.batchChallenge, Fr.zero), "batch open: non-zero challenge")

        let v1 = GPUBasefoldProverEngine.evaluateMultilinear(evaluations: poly1, point: point)
        let v2 = GPUBasefoldProverEngine.evaluateMultilinear(evaluations: poly2, point: point)

        let valid = engine.verifyBatch(
            batch: batch, point: point,
            claimedValues: [v1, v2], proof: batchProof)
        expect(valid, "batch open: verification passes")
    } catch {
        expect(false, "batch open verify: error \(error)")
    }
}

private func testBatchWrongValueRejected() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let poly1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let poly2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let point = [frFromInt(3), frFromInt(5)]

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])
        let batchProof = try engine.batchOpen(batch: batch, point: point)

        // Provide wrong value for second polynomial
        let v1 = GPUBasefoldProverEngine.evaluateMultilinear(evaluations: poly1, point: point)
        let wrongV2 = frFromInt(12345)

        let valid = engine.verifyBatch(
            batch: batch, point: point,
            claimedValues: [v1, wrongV2], proof: batchProof)
        expect(!valid, "batch wrong value: verification rejects incorrect value")
    } catch {
        expect(false, "batch wrong value rejected: error \(error)")
    }
}

// MARK: - Univariate Mode Tests

private func testUnivariateCommitOpen() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // f(X) = 1 + 2X + 3X^2 + 4X^3
        let coeffs = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        let commitment = try engine.commitUnivariate(coeffs: coeffs)
        expect(!frEqual(commitment.root, Fr.zero), "univariate commit: non-zero root")
        expect(commitment.numVars >= 2, "univariate commit: at least 2 vars for 4 coeffs")
    } catch {
        expect(false, "univariate commit/open: error \(error)")
    }
}

private func testUnivariateLinearPoly() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // f(X) = 5 + 3X
        let coeffs = [frFromInt(5), frFromInt(3)]

        let commitment = try engine.commitUnivariate(coeffs: coeffs)
        expect(!frEqual(commitment.root, Fr.zero), "univariate linear: non-zero root")

        // The commitment should have evaluations at domain points
        // f(0) = 5, f(1) = 8
        expect(frEqual(commitment.evaluations[0], frFromInt(5)),
               "univariate linear: f(0) = 5")
        expect(frEqual(commitment.evaluations[1], frFromInt(8)),
               "univariate linear: f(1) = 8")
    } catch {
        expect(false, "univariate linear: error \(error)")
    }
}

private func testUnivariateConstantPoly() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // f(X) = 42, padded to 2 coefficients
        let coeffs = [frFromInt(42), Fr.zero]

        let commitment = try engine.commitUnivariate(coeffs: coeffs)
        expect(!frEqual(commitment.root, Fr.zero), "univariate constant: non-zero root")

        // f(0) = 42, f(1) = 42
        expect(frEqual(commitment.evaluations[0], frFromInt(42)),
               "univariate constant: f(0) = 42")
        expect(frEqual(commitment.evaluations[1], frFromInt(42)),
               "univariate constant: f(1) = 42")
    } catch {
        expect(false, "univariate constant: error \(error)")
    }
}

// MARK: - Stats Tests

private func testCommitmentStats() {
    do {
        let engine = try GPUBasefoldProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let stats = engine.commitmentStats(commitment)
        expectEqual(stats.numVars, 3, "stats: numVars = 3")
        expectEqual(stats.evalCount, 8, "stats: 8 evaluations")
        expect(stats.encodedCount >= 8, "stats: encoded count >= eval count")
        expect(stats.treeSize > 0, "stats: positive tree size")
    } catch {
        expect(false, "commitment stats: error \(error)")
    }
}

private func testProofStats() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let proof = try engine.open(commitment: commitment, point: point)

        let stats = engine.proofStats(proof)
        expectEqual(stats.numFoldLevels, 2, "proof stats: 2 fold levels (fold-by-4)")
        expectEqual(stats.numQueries, 16, "proof stats: fast config has 16 queries")
        expect(stats.totalAuthPathSize > 0, "proof stats: non-zero auth path size")
    } catch {
        expect(false, "proof stats: error \(error)")
    }
}

// MARK: - Config Tests

private func testConfigStandard() {
    let config = BasefoldProverConfig.standard
    expectEqual(config.numQueries, 40, "standard config: 40 queries")
    expectEqual(config.rateLog, 1, "standard config: rateLog = 1")
    expectEqual(config.maxGPULogSize, 24, "standard config: maxGPULogSize = 24")
}

private func testConfigFast() {
    let config = BasefoldProverConfig.fast
    expectEqual(config.numQueries, 16, "fast config: 16 queries")
    expectEqual(config.rateLog, 1, "fast config: rateLog = 1")
    expectEqual(config.maxGPULogSize, 20, "fast config: maxGPULogSize = 20")
}

// MARK: - Edge Cases

private func testSingleEvalPoly() {
    // A polynomial with a single evaluation (0 variables) is degenerate
    // but should still produce a valid commitment
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = [frFromInt(1), frFromInt(2)]
        let commitment = try engine.commit(evaluations: evals)

        expect(!frEqual(commitment.root, Fr.zero), "single var: non-zero root")
        expectEqual(commitment.numVars, 1, "single var: 1 variable")

        let point = [frFromInt(10)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expected), "single var: correct evaluation")
    } catch {
        expect(false, "single eval poly: error \(error)")
    }
}

private func testLargerPolynomial() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // 5 variables = 32 evaluations
        let n = 32
        let evals = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 5, "larger: 5 variables")

        let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.finalValue, expected), "larger: correct evaluation at (2,3,5,7,11)")

        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: expected, proof: proof)
        expect(valid, "larger: verification passes")
    } catch {
        expect(false, "larger polynomial: error \(error)")
    }
}

private func testFoldConsistency() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let proof = try engine.open(commitment: commitment, point: point)

        // fold-by-4: 3 vars => 2 committed levels
        // Level 0 (fused, stride 2): 8 -> 2 elements
        // Level 1 (single, stride 1): 2 -> 1 element
        expectEqual(proof.foldLayers.count, 2, "fold: 2 levels (fold-by-4)")
        if proof.foldLayers.count == 2 {
            expectEqual(proof.foldLayers[0].count, 2, "fold: layer 0 has 2 elements (8/4)")
            expectEqual(proof.foldLayers[1].count, 1, "fold: layer 1 has 1 element")
        }

        // Final fold layer should contain the final value
        if let lastLayer = proof.foldLayers.last {
            expect(frEqual(lastLayer[0], proof.finalValue),
                   "fold: last layer = final value")
        }
    } catch {
        expect(false, "fold consistency: error \(error)")
    }
}

private func testMerklePathVerification() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let commitment = try engine.commit(evaluations: evals)

        // The Merkle tree should be valid: verify path for leaf 0
        let tree = commitment.merkleTree
        let encoded = commitment.encodedEvals
        let root = commitment.root

        if encoded.count >= 2 && tree.count > 1 {
            let leaf = encoded[0]
            // Extract path for leaf 0
            var path: [Fr] = []
            var idx = 0
            var levelStart = 0
            var levelSize = encoded.count

            while levelSize > 1 {
                let sibIdx = idx ^ 1
                if levelStart + sibIdx < tree.count {
                    path.append(tree[levelStart + sibIdx])
                }
                idx /= 2
                levelStart += levelSize
                levelSize /= 2
            }

            let valid = engine.verifyMerklePath(leaf: leaf, path: path, index: 0, root: root)
            expect(valid, "merkle path: leaf 0 verifies to root")
        } else {
            expect(true, "merkle path: skipped for small tree")
        }
    } catch {
        expect(false, "merkle path verification: error \(error)")
    }
}

private func testQueryProofStructure() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let proof = try engine.open(commitment: commitment, point: point)

        expect(proof.queries.count == 16, "query structure: 16 queries (fast config)")

        for (qi, query) in proof.queries.enumerated() {
            // Each query should have eval pairs and fold values of same length
            expectEqual(query.evalPairs.count, query.foldValues.count,
                        "query \(qi): evalPairs.count == foldValues.count")

            // Auth paths should have same count as eval pairs
            expectEqual(query.authPaths.count, query.evalPairs.count,
                        "query \(qi): authPaths.count == evalPairs.count")

            // Query index should be valid
            expect(query.index >= 0 && query.index < evals.count / 2,
                   "query \(qi): valid index")
        }
    } catch {
        expect(false, "query proof structure: error \(error)")
    }
}

// MARK: - Special Polynomial Tests

private func testAllOnesPolynomial() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // Constant polynomial f = 1 everywhere on {0,1}^2
        let evals = [Fr.one, Fr.one, Fr.one, Fr.one]
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(99), frFromInt(42)]
        let proof = try engine.open(commitment: commitment, point: point)

        // Constant polynomial evaluates to 1 everywhere
        expect(frEqual(proof.finalValue, Fr.one), "all-ones: evaluates to 1")

        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: Fr.one, proof: proof)
        expect(valid, "all-ones: verification passes")
    } catch {
        expect(false, "all-ones polynomial: error \(error)")
    }
}

private func testAllZerosPolynomial() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // Zero polynomial
        let evals = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(7), frFromInt(13)]
        let proof = try engine.open(commitment: commitment, point: point)

        expect(frEqual(proof.finalValue, Fr.zero), "all-zeros: evaluates to 0")

        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: Fr.zero, proof: proof)
        expect(valid, "all-zeros: verification passes")
    } catch {
        expect(false, "all-zeros polynomial: error \(error)")
    }
}

private func testIdentityPolynomial() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .fast)
        // f(x0, x1) where f(0,0)=0, f(1,0)=1, f(0,1)=2, f(1,1)=3
        // This is the identity-like polynomial: f = x0 + 2*x1
        let evals = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
        let commitment = try engine.commit(evaluations: evals)

        // f(4, 5) = 4 + 2*5 = 14
        // Via multilinear: fold x0=4: [0+4*(1-0), 2+4*(3-2)] = [4, 6]
        // fold x1=5: 4 + 5*(6-4) = 14
        let point = [frFromInt(4), frFromInt(5)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBasefoldProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(expected, frFromInt(14)), "identity: manual check f(4,5)=14")
        expect(frEqual(proof.finalValue, expected), "identity: proof matches evaluation")

        let valid = engine.verify(
            root: commitment.root, point: point,
            claimedValue: expected, proof: proof)
        expect(valid, "identity: verification passes")
    } catch {
        expect(false, "identity polynomial: error \(error)")
    }
}

// MARK: - Benchmark

private func basefoldBenchmark() {
    do {
        let engine = try GPUBasefoldProverEngine(config: .standard)
        let n = 1 << 18  // 2^18 = 262144 evaluations
        let numVars = 18

        fputs(String(format: "\n  Basefold Prover Benchmark (n=2^%d, m=1):\n", numVars), stderr)

        // Create test polynomial
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            evals[i] = frFromInt(UInt64(i % 1000))
        }

        // Benchmark commit (5 runs)
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let commitment = try engine.commit(evaluations: evals)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            commitTimes.append(dt)
        }
        commitTimes.sort()
        fputs(String(format: "    Commit:    %.2f ms (median of 5)\n", commitTimes[2]), stderr)

        // Create a random point
        var point = [Fr](repeating: Fr.zero, count: numVars)
        for i in 0..<numVars {
            point[i] = frFromInt(UInt64((i + 1) * 12345 % 1000))
        }

        // Benchmark open (3 runs)
        var openTimes = [Double]()
        for _ in 0..<3 {
            let commitment = try engine.commit(evaluations: evals)
            let t0 = CFAbsoluteTimeGetCurrent()
            let proof = try engine.open(commitment: commitment, point: point)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            openTimes.append(dt)
        }
        openTimes.sort()
        fputs(String(format: "    Open:      %.2f ms (median of 3)\n", openTimes[1]), stderr)

        fputs(String(format: "  Basefold version: %@\n", GPUBasefoldProverEngine.version.description), stderr)
    } catch {
        fputs("  Basefold benchmark error: \(error)\n", stderr)
    }
}

public func runBasefoldBenchmark() {
    basefoldBenchmark()
}
