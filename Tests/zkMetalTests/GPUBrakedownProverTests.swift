// GPUBrakedownProverTests — Comprehensive tests for GPU-accelerated Brakedown prover engine
//
// Tests commitment, opening, verification, batch operations, expander encoding,
// code consistency, tensor products, tamper detection, diagnostics, and edge cases.

import zkMetal
import Foundation

// MARK: - Public test entry point

public func runGPUBrakedownProverTests() {
    suite("GPU Brakedown Prover Engine")
    testBasicCommitOpen()
    testTwoVarCommitOpen()
    testThreeVarCommitOpen()
    testFourVarCommitOpen()
    testFiveVarCommitOpen()
    testEvaluationCorrectness()
    testMultilinearAtBooleanVertices()
    testMultilinearExtrapolation()
    testTensorProductOneBit()
    testTensorProductTwoBit()
    testTensorProductThreeBit()
    testTensorProductAtZero()
    testTensorProductAtOne()
    testCommitmentDeterminism()
    testCommitmentNonZeroRoot()
    testCommitmentDimensions()
    testOpenValueConsistency()
    testVerificationPasses()
    testVerificationWrongValue()
    testVerificationWrongPoint()
    testVerificationTamperedColumn()
    testVerificationTamperedTVector()
    testCodeConsistencyPassesOnValidProof()
    testCodeConsistencyFailsOnTamperedT()
    testBatchCommitTwo()
    testBatchCommitThree()
    testBatchOpenVerify()
    testBatchVerifyRejectsWrong()
    testQueryIndicesDeterministic()
    testQueryIndicesDistinct()
    testQueryIndicesInRange()
    testMerkleProofExtraction()
    testProofSize()
    testCommitmentStats()
    testProofStats()
    testCodeParams()
    testConfigStandard()
    testConfigHighSecurity()
    testConfigFast()
    testSoundnessBits()
    testAllOnesPolynomial()
    testAllZerosPolynomial()
    testIdentityPolynomial()
    testSingleVariablePoly()
    testLargePolynomial()
    testOpenAtOrigin()
    testOpenAtAllOnes()
}

// MARK: - Basic Commit/Open Tests

private func testBasicCommitOpen() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // f(x) = 3*(1-x) + 8*x on {0,1}^1 => evals = [3, 8]
        let evals = [frFromInt(3), frFromInt(8)]
        let commitment = try engine.commit(evaluations: evals)

        expect(!frEqual(commitment.merkleRoot, Fr.zero), "commit: non-zero root")
        expectEqual(commitment.numVars, 1, "commit: 1 variable")
        expectEqual(commitment.evaluations.count, 2, "commit: 2 evaluations")

        // Open at point [7] => f(7) = 3 + 7*(8-3) = 38
        let point = [frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expectedValue = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expectedValue),
               "open: claimed value matches multilinear evaluation")
    } catch {
        expect(false, "basic commit/open: unexpected error \(error)")
    }
}

private func testTwoVarCommitOpen() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 2, "2-var: numVars = 2")
        expectEqual(commitment.numRows * commitment.numCols, 4, "2-var: matrix product = 4")

        let point = [frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "2-var: correct evaluation")
    } catch {
        expect(false, "2-var commit/open: error \(error)")
    }
}

private func testThreeVarCommitOpen() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 3, "3-var: numVars = 3")
        expectEqual(commitment.evaluations.count, 8, "3-var: 8 evaluations")

        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "3-var: correct evaluation")
        expect(proof.queryIndices.count > 0, "3-var: has query indices")
    } catch {
        expect(false, "3-var commit/open: error \(error)")
    }
}

private func testFourVarCommitOpen() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 4, "4-var: numVars = 4")

        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "4-var: correct evaluation")
        expect(proof.tVector.count == commitment.numCols, "4-var: t-vector has numCols entries")
    } catch {
        expect(false, "4-var commit/open: error \(error)")
    }
}

private func testFiveVarCommitOpen() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...32).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 5, "5-var: numVars = 5")

        let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "5-var: correct evaluation")
    } catch {
        expect(false, "5-var commit/open: error \(error)")
    }
}

// MARK: - Multilinear Evaluation Tests

private func testEvaluationCorrectness() {
    // f(x0, x1) = 1*(1-x0)(1-x1) + 2*x0*(1-x1) + 3*(1-x0)*x1 + 4*x0*x1
    let evals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let point = [frFromInt(2), frFromInt(3)]

    let result = GPUBrakedownProverEngine.evaluateMultilinear(
        evaluations: evals, point: point)

    // Manual: fold with x0=2: [1+2*(2-1), 3+2*(4-3)] = [3, 5]
    // fold with x1=3: 3 + 3*(5-3) = 9
    let expected = frFromInt(9)
    expect(frEqual(result, expected), "eval correctness: f(2,3) = 9")
}

private func testMultilinearAtBooleanVertices() {
    let evals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let v00 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.zero, Fr.zero])
    let v10 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.one, Fr.zero])
    let v01 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.zero, Fr.one])
    let v11 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.one, Fr.one])
    expect(frEqual(v00, frFromInt(10)), "boolean (0,0) = evals[0]")
    expect(frEqual(v10, frFromInt(20)), "boolean (1,0) = evals[1]")
    expect(frEqual(v01, frFromInt(30)), "boolean (0,1) = evals[2]")
    expect(frEqual(v11, frFromInt(40)), "boolean (1,1) = evals[3]")
}

private func testMultilinearExtrapolation() {
    let evals = [frFromInt(10), frFromInt(20)]
    let val0 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.zero])
    expect(frEqual(val0, frFromInt(10)), "extrapolation: f(0) = 10")
    let val1 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [Fr.one])
    expect(frEqual(val1, frFromInt(20)), "extrapolation: f(1) = 20")
    let val3 = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: [frFromInt(3)])
    expect(frEqual(val3, frFromInt(40)), "extrapolation: f(3) = 40")
}

// MARK: - Tensor Product Tests

private func testTensorProductOneBit() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let z = frFromInt(3)
        let tensor = engine.computeTensor([z])
        // tensor[0] = 1 - 3 = -2, tensor[1] = 3
        let expected0 = frSub(Fr.one, z)
        let expected1 = z
        expect(frEqual(tensor[0], expected0), "tensor 1-bit: t[0] = 1-z")
        expect(frEqual(tensor[1], expected1), "tensor 1-bit: t[1] = z")
    } catch {
        expect(false, "tensor 1-bit: error \(error)")
    }
}

private func testTensorProductTwoBit() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let z0 = frFromInt(2); let z1 = frFromInt(5)
        let tensor = engine.computeTensor([z0, z1])
        expectEqual(tensor.count, 4, "tensor 2-bit: 4 elements")
        let m0 = frSub(Fr.one, z0); let m1 = frSub(Fr.one, z1)
        expect(frEqual(tensor[0], frMul(m0, m1)), "tensor[0] = (1-z0)(1-z1)")
        expect(frEqual(tensor[1], frMul(z0, m1)), "tensor[1] = z0*(1-z1)")
        expect(frEqual(tensor[2], frMul(m0, z1)), "tensor[2] = (1-z0)*z1")
        expect(frEqual(tensor[3], frMul(z0, z1)), "tensor[3] = z0*z1")
    } catch {
        expect(false, "tensor 2-bit: error \(error)")
    }
}

private func testTensorProductThreeBit() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let tensor = engine.computeTensor(point)
        expectEqual(tensor.count, 8, "tensor 3-bit: 8 elements")
        // Check: <tensor, evals> == MLE(evals, point)
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        var dot = Fr.zero
        for i in 0..<8 { dot = frAdd(dot, frMul(tensor[i], evals[i])) }
        let mlEval = GPUBrakedownProverEngine.evaluateMultilinear(evaluations: evals, point: point)
        expect(frEqual(dot, mlEval), "tensor 3-bit: <tensor, evals> = MLE(evals, point)")
    } catch {
        expect(false, "tensor 3-bit: error \(error)")
    }
}

private func testTensorProductAtZero() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let tensor = engine.computeTensor([Fr.zero, Fr.zero])
        // All zeros except tensor[0] = 1
        expect(frEqual(tensor[0], Fr.one), "tensor at origin: t[0] = 1")
        expect(frEqual(tensor[1], Fr.zero), "tensor at origin: t[1] = 0")
        expect(frEqual(tensor[2], Fr.zero), "tensor at origin: t[2] = 0")
        expect(frEqual(tensor[3], Fr.zero), "tensor at origin: t[3] = 0")
    } catch {
        expect(false, "tensor at origin: error \(error)")
    }
}

private func testTensorProductAtOne() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let tensor = engine.computeTensor([Fr.one, Fr.one])
        // Only tensor[3] = 1, rest = 0
        expect(frEqual(tensor[0], Fr.zero), "tensor at (1,1): t[0] = 0")
        expect(frEqual(tensor[1], Fr.zero), "tensor at (1,1): t[1] = 0")
        expect(frEqual(tensor[2], Fr.zero), "tensor at (1,1): t[2] = 0")
        expect(frEqual(tensor[3], Fr.one), "tensor at (1,1): t[3] = 1")
    } catch {
        expect(false, "tensor at (1,1): error \(error)")
    }
}

// MARK: - Commitment Properties

private func testCommitmentDeterminism() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }

        let c1 = try engine.commit(evaluations: evals)
        let c2 = try engine.commit(evaluations: evals)

        expect(frEqual(c1.merkleRoot, c2.merkleRoot),
               "determinism: same evaluations produce same root")
        expectEqual(c1.numRows, c2.numRows, "determinism: same numRows")
        expectEqual(c1.numCols, c2.numCols, "determinism: same numCols")
        expectEqual(c1.numEncodedCols, c2.numEncodedCols, "determinism: same encodedCols")
    } catch {
        expect(false, "commitment determinism: error \(error)")
    }
}

private func testCommitmentNonZeroRoot() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = [frFromInt(42), frFromInt(99)]
        let commitment = try engine.commit(evaluations: evals)
        expect(!frEqual(commitment.merkleRoot, Fr.zero),
               "non-zero root: commitment root is non-zero")
    } catch {
        expect(false, "non-zero root: error \(error)")
    }
}

private func testCommitmentDimensions() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // 16 evaluations => 4x4 matrix (logN=4, logRows=2, logCols=2)
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)

        expectEqual(c.numRows * c.numCols, 16, "dimensions: rows * cols = n")
        expect(c.numEncodedCols >= c.numCols, "dimensions: encodedCols >= numCols")
        expectEqual(c.numEncodedCols, c.numCols * engine.config.rateInverse,
                    "dimensions: encodedCols = numCols * rateInverse")
        expectEqual(c.encodedMatrix.count, c.numRows * c.numEncodedCols,
                    "dimensions: encoded matrix size correct")
    } catch {
        expect(false, "commitment dimensions: error \(error)")
    }
}

// MARK: - Open and Verify Tests

private func testOpenValueConsistency() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected),
               "value consistency: proof.claimedValue matches MLE")
    } catch {
        expect(false, "value consistency: error \(error)")
    }
}

private func testVerificationPasses() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let proof = try engine.open(commitment: commitment, point: point)

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "verification passes on honest proof")
    } catch {
        expect(false, "verification passes: error \(error)")
    }
}

private func testVerificationWrongValue() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        let tampered = BrakedownProverOpeningProof(
            tVector: p.tVector, columnOpenings: p.columnOpenings,
            merkleProofs: p.merkleProofs, queryIndices: p.queryIndices,
            claimedValue: frAdd(p.claimedValue, Fr.one), point: p.point)
        expect(!engine.verify(commitment: c, proof: tampered), "wrong value rejected")
    } catch {
        expect(false, "wrong value rejected: error \(error)")
    }
}

private func testVerificationWrongPoint() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        let wrongPt = [frFromInt(4), frFromInt(7), frFromInt(11), frFromInt(13)]
        let tampered = BrakedownProverOpeningProof(
            tVector: p.tVector, columnOpenings: p.columnOpenings,
            merkleProofs: p.merkleProofs, queryIndices: p.queryIndices,
            claimedValue: p.claimedValue, point: wrongPt)
        expect(!engine.verify(commitment: c, proof: tampered), "wrong point rejected")
    } catch {
        expect(false, "wrong point rejected: error \(error)")
    }
}

private func testVerificationTamperedColumn() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        guard !p.columnOpenings.isEmpty, !p.columnOpenings[0].isEmpty else {
            expect(false, "tampered column: no columns"); return
        }
        var cols = p.columnOpenings
        cols[0][0] = frAdd(cols[0][0], Fr.one)
        let tampered = BrakedownProverOpeningProof(
            tVector: p.tVector, columnOpenings: cols,
            merkleProofs: p.merkleProofs, queryIndices: p.queryIndices,
            claimedValue: p.claimedValue, point: p.point)
        expect(!engine.verify(commitment: c, proof: tampered), "tampered column rejected")
    } catch {
        expect(false, "tampered column: error \(error)")
    }
}

private func testVerificationTamperedTVector() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        guard !p.tVector.isEmpty else { expect(false, "no t-vector"); return }
        var t = p.tVector; t[0] = frAdd(t[0], Fr.one)
        let tampered = BrakedownProverOpeningProof(
            tVector: t, columnOpenings: p.columnOpenings,
            merkleProofs: p.merkleProofs, queryIndices: p.queryIndices,
            claimedValue: p.claimedValue, point: p.point)
        expect(!engine.verify(commitment: c, proof: tampered), "tampered t-vector rejected")
    } catch {
        expect(false, "tampered t-vector: error \(error)")
    }
}

// MARK: - Code Consistency Tests

private func testCodeConsistencyPassesOnValidProof() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        let logRows = Int(log2(Double(c.numRows)))
        let ok = engine.codeConsistencyCheck(proof: p, numRows: c.numRows,
            numCols: c.numCols, pointRows: Array(point.prefix(logRows)))
        expect(ok, "code consistency passes on valid proof")
    } catch {
        expect(false, "code consistency valid: error \(error)")
    }
}

private func testCodeConsistencyFailsOnTamperedT() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let c = try engine.commit(evaluations: evals)
        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let p = try engine.open(commitment: c, point: point)
        guard !p.tVector.isEmpty else { expect(false, "no t-vector"); return }
        var t = p.tVector; t[0] = frAdd(t[0], frFromInt(100))
        let tampered = BrakedownProverOpeningProof(
            tVector: t, columnOpenings: p.columnOpenings, merkleProofs: p.merkleProofs,
            queryIndices: p.queryIndices, claimedValue: p.claimedValue, point: p.point)
        let logRows = Int(log2(Double(c.numRows)))
        let ok = engine.codeConsistencyCheck(proof: tampered, numRows: c.numRows,
            numCols: c.numCols, pointRows: Array(point.prefix(logRows)))
        expect(!ok, "code consistency fails on tampered t-vector")
    } catch {
        expect(false, "code consistency tampered: error \(error)")
    }
}

// MARK: - Batch Tests

private func testBatchCommitTwo() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let poly1 = (1...8).map { frFromInt(UInt64($0)) }
        let poly2 = (10...17).map { frFromInt(UInt64($0)) }

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])
        expectEqual(batch.commitments.count, 2, "batch 2: two commitments")
        expect(!frEqual(batch.batchRoot, Fr.zero), "batch 2: non-zero batch root")
        expect(!frEqual(batch.commitments[0].merkleRoot,
                        batch.commitments[1].merkleRoot),
               "batch 2: different polynomials have different roots")
    } catch {
        expect(false, "batch commit 2: error \(error)")
    }
}

private func testBatchCommitThree() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let p1 = (1...4).map { frFromInt(UInt64($0)) }
        let p2 = (5...8).map { frFromInt(UInt64($0)) }
        let p3 = (9...12).map { frFromInt(UInt64($0)) }

        let batch = try engine.batchCommit(polynomials: [p1, p2, p3])
        expectEqual(batch.commitments.count, 3, "batch 3: three commitments")
        expect(!frEqual(batch.batchRoot, Fr.zero), "batch 3: non-zero batch root")
    } catch {
        expect(false, "batch commit 3: error \(error)")
    }
}

private func testBatchOpenVerify() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let poly1 = (1...8).map { frFromInt(UInt64($0)) }
        let poly2 = (10...17).map { frFromInt(UInt64($0)) }

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let batchProof = try engine.batchOpen(batch: batch, point: point)
        expectEqual(batchProof.proofs.count, 2, "batch open: two proofs")

        // Verify each individual proof
        for i in 0..<batch.commitments.count {
            let valid = engine.verify(commitment: batch.commitments[i],
                                      proof: batchProof.proofs[i])
            expect(valid, "batch open: proof \(i) verifies individually")
        }

        // Verify via batch verify
        let batchValid = engine.verifyBatch(batch: batch, proof: batchProof)
        expect(batchValid, "batch open: batch verification passes")
    } catch {
        expect(false, "batch open/verify: error \(error)")
    }
}

private func testBatchVerifyRejectsWrong() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let poly1 = (1...8).map { frFromInt(UInt64($0)) }
        let poly2 = (10...17).map { frFromInt(UInt64($0)) }

        let batch = try engine.batchCommit(polynomials: [poly1, poly2])
        let point = [frFromInt(2), frFromInt(3), frFromInt(5)]

        let batchProof = try engine.batchOpen(batch: batch, point: point)

        // Tamper: swap proofs
        let tamperedBatchProof = BrakedownProverBatchProof(
            proofs: [batchProof.proofs[1], batchProof.proofs[0]],
            batchChallenge: batchProof.batchChallenge
        )
        let valid = engine.verifyBatch(batch: batch, proof: tamperedBatchProof)
        expect(!valid, "batch verify rejects swapped proofs")
    } catch {
        expect(false, "batch verify rejects wrong: error \(error)")
    }
}

// MARK: - Query Index Tests

private func testQueryIndicesDeterministic() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let root = frFromInt(12345)
        let idx1 = engine.generateQueryIndices(root: root, numQueries: 10, maxCol: 100)
        let idx2 = engine.generateQueryIndices(root: root, numQueries: 10, maxCol: 100)
        expectEqual(idx1.count, idx2.count, "query indices deterministic: same count")
        for i in 0..<idx1.count {
            expectEqual(idx1[i], idx2[i], "query indices deterministic: index \(i) matches")
        }
    } catch {
        expect(false, "query indices deterministic: error \(error)")
    }
}

private func testQueryIndicesDistinct() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let root = frFromInt(67890)
        let indices = engine.generateQueryIndices(root: root, numQueries: 20, maxCol: 1000)
        let unique = Set(indices)
        expectEqual(indices.count, unique.count, "query indices distinct: all unique")
    } catch {
        expect(false, "query indices distinct: error \(error)")
    }
}

private func testQueryIndicesInRange() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let root = frFromInt(11111)
        let maxCol = 50
        let indices = engine.generateQueryIndices(root: root, numQueries: 15, maxCol: maxCol)
        for idx in indices {
            expect(idx >= 0 && idx < maxCol, "query indices in range: \(idx) < \(maxCol)")
        }
    } catch {
        expect(false, "query indices in range: error \(error)")
    }
}

// MARK: - Merkle Proof Tests

private func testMerkleProofExtraction() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let proof = try engine.open(commitment: commitment, point: point)

        // Each queried column should have a Merkle proof
        expectEqual(proof.merkleProofs.count, proof.queryIndices.count,
                    "merkle proofs: one proof per query")

        // Merkle proofs should be non-empty for non-trivial trees
        for i in 0..<proof.merkleProofs.count {
            expect(proof.merkleProofs[i].count > 0,
                   "merkle proofs: proof \(i) is non-empty")
        }
    } catch {
        expect(false, "merkle proof extraction: error \(error)")
    }
}

// MARK: - Proof Size and Stats Tests

private func testProofSize() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        expect(proof.proofSizeBytes > 0, "proof size: positive bytes")
        expect(proof.tVector.count == commitment.numCols,
               "proof size: t-vector length = numCols")
        expect(proof.columnOpenings.count == proof.queryIndices.count,
               "proof size: column count = query count")
    } catch {
        expect(false, "proof size: error \(error)")
    }
}

private func testCommitmentStats() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let stats = engine.commitmentStats(commitment)
        expectEqual(stats.numVars, 4, "commitment stats: numVars = 4")
        expectEqual(stats.evalCount, 16, "commitment stats: evalCount = 16")
        expect(stats.encodedCols > 0, "commitment stats: positive encodedCols")
        expect(stats.treeSize > 0, "commitment stats: positive treeSize")
        expect(stats.commitmentBytes > 0, "commitment stats: positive commitmentBytes")
    } catch {
        expect(false, "commitment stats: error \(error)")
    }
}

private func testProofStats() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...16).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        let stats = engine.proofStats(proof)
        expect(stats.tVectorLen > 0, "proof stats: positive t-vector length")
        expect(stats.numQueries > 0, "proof stats: positive query count")
        expect(stats.proofBytes > 0, "proof stats: positive proof bytes")
        expect(stats.avgMerklePathLen >= 0, "proof stats: non-negative avg path length")
    } catch {
        expect(false, "proof stats: error \(error)")
    }
}

private func testCodeParams() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let params = engine.codeParams(forNumCols: 8)
        expectEqual(params.codewordLength, 8 * engine.config.rateInverse,
                    "code params: codeword = numCols * rateInverse")
        expectEqual(params.redundancyLength, params.codewordLength - 8,
                    "code params: redundancy = codeword - message")
        expect(params.rate > 0 && params.rate < 1, "code params: 0 < rate < 1")
        expect(params.degree > 0, "code params: positive degree")
    } catch {
        expect(false, "code params: error \(error)")
    }
}

// MARK: - Configuration Tests

private func testConfigStandard() {
    let cfg = BrakedownProverConfig.standard
    expectEqual(cfg.rateInverse, 4, "standard config: rateInverse = 4")
    expectEqual(cfg.numQueries, 30, "standard config: numQueries = 30")
    expectEqual(cfg.expanderDegree, 10, "standard config: expanderDegree = 10")
}

private func testConfigHighSecurity() {
    let cfg = BrakedownProverConfig.highSecurity
    expectEqual(cfg.rateInverse, 8, "high security config: rateInverse = 8")
    expectEqual(cfg.numQueries, 50, "high security config: numQueries = 50")
    expectEqual(cfg.expanderDegree, 16, "high security config: expanderDegree = 16")
}

private func testConfigFast() {
    let cfg = BrakedownProverConfig.fast
    expectEqual(cfg.rateInverse, 4, "fast config: rateInverse = 4")
    expectEqual(cfg.numQueries, 16, "fast config: numQueries = 16")
    expectEqual(cfg.expanderDegree, 8, "fast config: expanderDegree = 8")
}

private func testSoundnessBits() {
    let standard = BrakedownProverConfig.standard
    // 30 * log2(4) = 30 * 2 = 60
    expect(abs(standard.soundnessBits - 60.0) < 0.01,
           "soundness bits: standard ~ 60")

    let highSec = BrakedownProverConfig.highSecurity
    // 50 * log2(8) = 50 * 3 = 150
    expect(abs(highSec.soundnessBits - 150.0) < 0.01,
           "soundness bits: high security ~ 150")

    let fast = BrakedownProverConfig.fast
    // 16 * log2(4) = 16 * 2 = 32
    expect(abs(fast.soundnessBits - 32.0) < 0.01,
           "soundness bits: fast ~ 32")
}

// MARK: - Edge Case Polynomials

private func testAllOnesPolynomial() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // Constant polynomial f = 1 everywhere
        let evals = [Fr](repeating: Fr.one, count: 8)
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(3), frFromInt(7), frFromInt(11)]
        let proof = try engine.open(commitment: commitment, point: point)

        // Constant 1 polynomial evaluates to 1 at any point
        expect(frEqual(proof.claimedValue, Fr.one), "all-ones poly: evaluates to 1")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "all-ones poly: verification passes")
    } catch {
        expect(false, "all-ones polynomial: error \(error)")
    }
}

private func testAllZerosPolynomial() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = [Fr](repeating: Fr.zero, count: 4)
        let commitment = try engine.commit(evaluations: evals)

        let point = [frFromInt(5), frFromInt(7)]
        let proof = try engine.open(commitment: commitment, point: point)

        // Zero polynomial evaluates to 0 everywhere
        expect(frEqual(proof.claimedValue, Fr.zero), "all-zeros poly: evaluates to 0")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "all-zeros poly: verification passes")
    } catch {
        expect(false, "all-zeros polynomial: error \(error)")
    }
}

private func testIdentityPolynomial() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // f(x) = x, evals on {0,1} = [0, 1]
        let evals = [Fr.zero, Fr.one]
        let commitment = try engine.commit(evaluations: evals)

        // f(5) = 5
        let point = [frFromInt(5)]
        let proof = try engine.open(commitment: commitment, point: point)

        expect(frEqual(proof.claimedValue, frFromInt(5)), "identity poly: f(5) = 5")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "identity poly: verification passes")
    } catch {
        expect(false, "identity polynomial: error \(error)")
    }
}

private func testSingleVariablePoly() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // f(x) = 3 + 7x => evals = [3, 10]
        let evals = [frFromInt(3), frFromInt(10)]
        let commitment = try engine.commit(evaluations: evals)

        // f(4) = 3 + 7*4 = 31  ... but in MLE: f(4) = 3*(1-4) + 10*4 = -9 + 40 = 31
        let point = [frFromInt(4)]
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "single var: correct value")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "single var: verification passes")
    } catch {
        expect(false, "single variable polynomial: error \(error)")
    }
}

private func testLargePolynomial() {
    do {
        let engine = try GPUBrakedownProverEngine()
        // 2^8 = 256 evaluations
        let evals = (1...256).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        expectEqual(commitment.numVars, 8, "large poly: numVars = 8")
        expect(commitment.numRows * commitment.numCols == 256,
               "large poly: matrix product = 256")

        let point = (1...8).map { frFromInt(UInt64($0)) }
        let proof = try engine.open(commitment: commitment, point: point)

        let expected = GPUBrakedownProverEngine.evaluateMultilinear(
            evaluations: evals, point: point)
        expect(frEqual(proof.claimedValue, expected), "large poly: correct evaluation")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "large poly: verification passes")
    } catch {
        expect(false, "large polynomial: error \(error)")
    }
}

private func testOpenAtOrigin() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        // Evaluating at origin (0,0,0) should give evals[0] = 1
        let point = [Fr.zero, Fr.zero, Fr.zero]
        let proof = try engine.open(commitment: commitment, point: point)

        expect(frEqual(proof.claimedValue, frFromInt(1)),
               "open at origin: f(0,0,0) = evals[0] = 1")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "open at origin: verification passes")
    } catch {
        expect(false, "open at origin: error \(error)")
    }
}

private func testOpenAtAllOnes() {
    do {
        let engine = try GPUBrakedownProverEngine()
        let evals = (1...8).map { frFromInt(UInt64($0)) }
        let commitment = try engine.commit(evaluations: evals)

        // Evaluating at (1,1,1) should give evals[7] = 8
        let point = [Fr.one, Fr.one, Fr.one]
        let proof = try engine.open(commitment: commitment, point: point)

        expect(frEqual(proof.claimedValue, frFromInt(8)),
               "open at all-ones: f(1,1,1) = evals[7] = 8")

        let valid = engine.verify(commitment: commitment, proof: proof)
        expect(valid, "open at all-ones: verification passes")
    } catch {
        expect(false, "open at all-ones: error \(error)")
    }
}
