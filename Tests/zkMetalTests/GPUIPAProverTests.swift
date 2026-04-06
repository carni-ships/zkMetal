// GPUIPAProverTests — Comprehensive tests for GPU-accelerated IPA prover engine
//
// Validates Pedersen vector commitment, IPA recursive halving, proof verification,
// batch proving, polynomial evaluation proofs, tamper rejection, and edge cases.

import Foundation
import Metal
import zkMetal

public func runGPUIPAProverTests() {
    suite("GPU IPA Prover Engine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    testVersionIsSet()
    testPedersenCommitBasic()
    testPedersenCommitZeroVector()
    testPedersenCommitWithBlinding()
    testBasicIPAProof()
    testIPAProofWithOnes()
    testIPAProofSmallVectors()
    testIPAProofRoundCount()
    testIPAProofTamperedRejection()
    testIPAProofTamperedFinalAB()
    testIPAProofWrongCommitment()
    testIPAProofUnblinded()
    testPolyEvaluationProof()
    testPolyEvaluationAtZero()
    testPolyEvaluationAtOne()
    testBatchProveSequential()
    testBatchProveParallel()
    testBatchVerifyRejectsWrong()
    testBatchProveUnblinded()
    testWeightedInnerProduct()
    testProofWellFormed()
    testBatchStatistics()
    testProofCountStats()
    testDebugDescription()
    testConfigOptions()
    testLargerVectorSize()
    testIdentityInnerProduct()
    testEngineWithExplicitGenerators()
}

// MARK: - Version

private func testVersionIsSet() {
    let v = GPUIPAProverEngine.version
    expect(!v.version.isEmpty, "Version string is non-empty: \(v.description)")
}

// MARK: - Pedersen Commitment

private func testPedersenCommitBasic() {
    suite("GPU IPA Prover — Pedersen commit basic")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create GPUIPAProverEngine")
        return
    }

    let a: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let r = frFromInt(42)
    guard let commitment = try? engine.pedersenCommit(vector: a, blindingFactor: r) else {
        expect(false, "pedersenCommit should not throw")
        return
    }

    expect(!pointIsIdentity(commitment.point), "Commitment is not identity")
    expectEqual(commitment.vectorLength, 8, "Vector length recorded correctly")
    expect(frEqual(commitment.blindingFactor, r), "Blinding factor stored correctly")
}

private func testPedersenCommitZeroVector() {
    suite("GPU IPA Prover — Pedersen commit zero vector")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    // Zero vector with zero blinding: should be identity
    let zero: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]
    guard let c = try? engine.commitUnblinded(vector: zero) else {
        expect(false, "commitUnblinded should not throw")
        return
    }
    expect(pointIsIdentity(c.point), "Commitment to zero vector with zero blinding is identity")
}

private func testPedersenCommitWithBlinding() {
    suite("GPU IPA Prover — Pedersen commit with blinding")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let r1 = frFromInt(100)
    let r2 = frFromInt(200)

    guard let c1 = try? engine.pedersenCommit(vector: a, blindingFactor: r1),
          let c2 = try? engine.pedersenCommit(vector: a, blindingFactor: r2) else {
        expect(false, "pedersenCommit should not throw")
        return
    }

    // Different blinding factors should produce different commitments
    let eq = pointEqual(c1.point, c2.point)
    expect(!eq, "Different blinding factors produce different commitments")
}

// MARK: - Basic IPA Proof

private func testBasicIPAProof() {
    suite("GPU IPA Prover — Basic IPA proof")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11),
                    Fr.zero, Fr.zero, Fr.zero, Fr.zero]
    let b: [Fr] = [frFromInt(2), frFromInt(4), frFromInt(6), frFromInt(8),
                    Fr.zero, Fr.zero, Fr.zero, Fr.zero]

    guard let commitment = try? engine.commitUnblinded(vector: a) else {
        expect(false, "commitUnblinded should not throw")
        return
    }

    guard let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // <a, b> = 3*2 + 5*4 + 7*6 + 11*8 = 6 + 20 + 42 + 88 = 156
    let expectedIP = frFromInt(156)
    expect(frEqual(proof.innerProduct, expectedIP), "Inner product = 156")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "Basic IPA proof verifies correctly")
}

private func testIPAProofWithOnes() {
    suite("GPU IPA Prover — IPA proof with ones vector")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let n = 8
    let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
    let b: [Fr] = [Fr](repeating: Fr.one, count: n)

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // <a, 1> = 1 + 2 + ... + 8 = 36
    let expectedIP = frFromInt(36)
    expect(frEqual(proof.innerProduct, expectedIP), "<a, 1> = sum(a) = 36")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "IPA proof with ones vector verifies")
}

private func testIPAProofSmallVectors() {
    suite("GPU IPA Prover — IPA proof n=2 (minimum)")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(10), frFromInt(20)]
    let b: [Fr] = [frFromInt(3), frFromInt(7)]

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // <a, b> = 10*3 + 20*7 = 30 + 140 = 170
    let expectedIP = frFromInt(170)
    expect(frEqual(proof.innerProduct, expectedIP), "Inner product = 170")
    expectEqual(proof.rounds, 1, "n=2 has 1 round")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "IPA proof n=2 verifies")
}

// MARK: - Round Count

private func testIPAProofRoundCount() {
    suite("GPU IPA Prover — Round count check")

    guard let engine = try? GPUIPAProverEngine(maxSize: 16) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    // n=4 -> 2 rounds
    do {
        let a: [Fr] = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<4).map { frFromInt(UInt64($0 + 5)) }
        guard let c = try? engine.commitUnblinded(vector: a),
              let p = try? engine.prove(a: a, b: b, commitment: c) else {
            expect(false, "prove should not throw")
            return
        }
        expectEqual(p.rounds, 2, "n=4: 2 rounds")
        expectEqual(p.Ls.count, 2, "n=4: 2 L values")
        expectEqual(p.Rs.count, 2, "n=4: 2 R values")
    }

    // n=8 -> 3 rounds
    do {
        let a: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 5)) }
        guard let c = try? engine.commitUnblinded(vector: a),
              let p = try? engine.prove(a: a, b: b, commitment: c) else {
            expect(false, "prove should not throw")
            return
        }
        expectEqual(p.rounds, 3, "n=8: 3 rounds")
    }

    // n=16 -> 4 rounds
    do {
        let a: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 5)) }
        guard let c = try? engine.commitUnblinded(vector: a),
              let p = try? engine.prove(a: a, b: b, commitment: c) else {
            expect(false, "prove should not throw")
            return
        }
        expectEqual(p.rounds, 4, "n=16: 4 rounds")
    }
}

// MARK: - Tamper Rejection

private func testIPAProofTamperedRejection() {
    suite("GPU IPA Prover — Tampered inner product rejected")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                    Fr.zero, Fr.zero, Fr.zero, Fr.zero]
    let b: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8),
                    Fr.zero, Fr.zero, Fr.zero, Fr.zero]

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // Tamper with inner product
    let tampered = IPAProverProof(
        Ls: proof.Ls, Rs: proof.Rs,
        finalA: proof.finalA, finalB: proof.finalB,
        innerProduct: frFromInt(999)
    )
    let invalid = engine.verify(proof: tampered, commitment: commitment.point, b: b)
    expect(!invalid, "Tampered inner product rejected by verifier")
}

private func testIPAProofTamperedFinalAB() {
    suite("GPU IPA Prover — Tampered finalA/finalB/LR rejected")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let b: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 10)) }

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // Tamper finalA
    let tamperedA = IPAProverProof(
        Ls: proof.Ls, Rs: proof.Rs,
        finalA: frFromInt(77777), finalB: proof.finalB,
        innerProduct: proof.innerProduct
    )
    expect(!engine.verify(proof: tamperedA, commitment: commitment.point, b: b),
           "Tampered finalA rejected")

    // Tamper finalB
    let tamperedB = IPAProverProof(
        Ls: proof.Ls, Rs: proof.Rs,
        finalA: proof.finalA, finalB: frFromInt(88888),
        innerProduct: proof.innerProduct
    )
    expect(!engine.verify(proof: tamperedB, commitment: commitment.point, b: b),
           "Tampered finalB rejected")

    // Swap first L and R
    var badLs = proof.Ls
    var badRs = proof.Rs
    if badLs.count > 0 && badRs.count > 0 {
        let tmp = badLs[0]
        badLs[0] = badRs[0]
        badRs[0] = tmp
    }
    let tamperedLR = IPAProverProof(
        Ls: badLs, Rs: badRs,
        finalA: proof.finalA, finalB: proof.finalB,
        innerProduct: proof.innerProduct
    )
    expect(!engine.verify(proof: tamperedLR, commitment: commitment.point, b: b),
           "Swapped L/R rejected")
}

private func testIPAProofWrongCommitment() {
    suite("GPU IPA Prover — Wrong commitment rejected")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let b: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 10)) }

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    // Use a different commitment (add a generator to it)
    let wrongC = pointAdd(commitment.point, pointFromAffine(engine.generators[0]))
    let invalid = engine.verify(proof: proof, commitment: wrongC, b: b)
    expect(!invalid, "Wrong commitment rejected by verifier")
}

private func testIPAProofUnblinded() {
    suite("GPU IPA Prover — Unblinded convenience API")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
    let b: [Fr] = [frFromInt(11), frFromInt(13), frFromInt(17), frFromInt(19)]

    guard let proof = try? engine.proveUnblinded(a: a, b: b) else {
        expect(false, "proveUnblinded should not throw")
        return
    }

    // <a, b> = 2*11 + 3*13 + 5*17 + 7*19 = 22 + 39 + 85 + 133 = 279
    let expectedIP = frFromInt(279)
    expect(frEqual(proof.innerProduct, expectedIP), "Inner product = 279")

    guard let commitment = try? engine.commitUnblinded(vector: a) else {
        expect(false, "commitUnblinded should not throw")
        return
    }
    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "Unblinded proof verifies correctly")
}

// MARK: - Polynomial Evaluation

private func testPolyEvaluationProof() {
    suite("GPU IPA Prover — Polynomial evaluation proof")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    // p(x) = 3 + 5x + 7x^2 + 11x^3
    let coeffs: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
    let z = frFromInt(2)

    guard let proof = try? engine.proveEvaluation(coeffs: coeffs, at: z) else {
        expect(false, "proveEvaluation should not throw")
        return
    }

    // p(2) = 3 + 10 + 28 + 88 = 129
    // The inner product should be <coeffs, [1, 2, 4, 8]> = 129
    let expectedIP = frFromInt(129)
    expect(frEqual(proof.innerProduct, expectedIP), "p(2) = 129")

    guard let commitment = try? engine.commitUnblinded(vector: coeffs) else {
        expect(false, "commitUnblinded should not throw")
        return
    }
    let valid = engine.verifyEvaluation(
        proof: proof, commitment: commitment.point, at: z, length: coeffs.count
    )
    expect(valid, "Polynomial evaluation proof verifies")
}

private func testPolyEvaluationAtZero() {
    suite("GPU IPA Prover — Polynomial evaluation at z=0")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let coeffs: [Fr] = [frFromInt(42), frFromInt(7), frFromInt(3), frFromInt(1)]
    let z = Fr.zero

    guard let proof = try? engine.proveEvaluation(coeffs: coeffs, at: z) else {
        expect(false, "proveEvaluation should not throw")
        return
    }

    // p(0) = 42 (constant term)
    let expectedIP = frFromInt(42)
    expect(frEqual(proof.innerProduct, expectedIP), "p(0) = 42")

    guard let commitment = try? engine.commitUnblinded(vector: coeffs) else {
        expect(false, "commitUnblinded should not throw")
        return
    }
    let valid = engine.verifyEvaluation(
        proof: proof, commitment: commitment.point, at: z, length: coeffs.count
    )
    expect(valid, "Evaluation at z=0 verifies")
}

private func testPolyEvaluationAtOne() {
    suite("GPU IPA Prover — Polynomial evaluation at z=1")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let z = Fr.one

    guard let proof = try? engine.proveEvaluation(coeffs: coeffs, at: z) else {
        expect(false, "proveEvaluation should not throw")
        return
    }

    // p(1) = 1 + 2 + 3 + 4 = 10
    let expectedIP = frFromInt(10)
    expect(frEqual(proof.innerProduct, expectedIP), "p(1) = sum of coefficients = 10")

    guard let commitment = try? engine.commitUnblinded(vector: coeffs) else {
        expect(false, "commitUnblinded should not throw")
        return
    }
    let valid = engine.verifyEvaluation(
        proof: proof, commitment: commitment.point, at: z, length: coeffs.count
    )
    expect(valid, "Evaluation at z=1 verifies")
}

// MARK: - Batch Proving

private func testBatchProveSequential() {
    suite("GPU IPA Prover — Batch prove sequential")

    guard let engine = try? GPUIPAProverEngine(
        maxSize: 8,
        config: IPAProverConfig(enableBatchParallel: false)
    ) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let b1: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
    let a2: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let b2: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(4), frFromInt(5)]

    guard let c1 = try? engine.commitUnblinded(vector: a1),
          let c2 = try? engine.commitUnblinded(vector: a2) else {
        expect(false, "commitUnblinded should not throw")
        return
    }

    let items: [(a: [Fr], b: [Fr], commitment: PedersenVectorCommitment)] = [
        (a: a1, b: b1, commitment: c1),
        (a: a2, b: b2, commitment: c2)
    ]

    guard let batch = try? engine.batchProve(items: items) else {
        expect(false, "batchProve should not throw")
        return
    }

    expectEqual(batch.count, 2, "Batch has 2 proofs")
    expectEqual(batch.commitments.count, 2, "Batch has 2 commitments")

    // Verify each proof individually
    let valid1 = engine.verify(proof: batch.proofs[0], commitment: c1.point, b: b1)
    expect(valid1, "Batch proof[0] verifies")

    let valid2 = engine.verify(proof: batch.proofs[1], commitment: c2.point, b: b2)
    expect(valid2, "Batch proof[1] verifies")
}

private func testBatchProveParallel() {
    suite("GPU IPA Prover — Batch prove parallel")

    guard let engine = try? GPUIPAProverEngine(
        maxSize: 8,
        config: IPAProverConfig(enableBatchParallel: true, maxBatchConcurrency: 4)
    ) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    var items = [(a: [Fr], b: [Fr], commitment: PedersenVectorCommitment)]()
    var bVectors = [[Fr]]()

    for k in 0..<4 {
        let a: [Fr] = (0..<4).map { frFromInt(UInt64($0 + k * 10 + 1)) }
        let b: [Fr] = (0..<4).map { frFromInt(UInt64($0 + k * 5 + 1)) }
        guard let c = try? engine.commitUnblinded(vector: a) else {
            expect(false, "commitUnblinded should not throw")
            return
        }
        items.append((a: a, b: b, commitment: c))
        bVectors.append(b)
    }

    guard let batch = try? engine.batchProve(items: items) else {
        expect(false, "batchProve should not throw")
        return
    }

    expectEqual(batch.count, 4, "Batch has 4 proofs")

    // Verify all proofs via batchVerify
    let valid = engine.batchVerify(batch: batch, bVectors: bVectors)
    expect(valid, "All parallel batch proofs verify")
}

private func testBatchVerifyRejectsWrong() {
    suite("GPU IPA Prover — Batch verify rejects wrong b-vector")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let b: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

    guard let c = try? engine.commitUnblinded(vector: a) else {
        expect(false, "commitUnblinded should not throw")
        return
    }

    let items: [(a: [Fr], b: [Fr], commitment: PedersenVectorCommitment)] = [
        (a: a, b: b, commitment: c)
    ]

    guard let batch = try? engine.batchProve(items: items) else {
        expect(false, "batchProve should not throw")
        return
    }

    // Verify with wrong b-vector
    let wrongB: [Fr] = [frFromInt(99), frFromInt(98), frFromInt(97), frFromInt(96)]
    let invalid = engine.batchVerify(batch: batch, bVectors: [wrongB])
    expect(!invalid, "Batch verify rejects wrong b-vector")
}

private func testBatchProveUnblinded() {
    suite("GPU IPA Prover — Batch prove unblinded convenience")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let items: [(a: [Fr], b: [Fr])] = [
        (a: [frFromInt(1), frFromInt(2)], b: [frFromInt(3), frFromInt(4)]),
        (a: [frFromInt(5), frFromInt(6)], b: [frFromInt(7), frFromInt(8)])
    ]

    guard let batch = try? engine.batchProveUnblinded(items: items) else {
        expect(false, "batchProveUnblinded should not throw")
        return
    }

    expectEqual(batch.count, 2, "Batch has 2 proofs")

    // Check inner products: <[1,2], [3,4]> = 11, <[5,6], [7,8]> = 83
    expect(frEqual(batch.proofs[0].innerProduct, frFromInt(11)), "Batch[0] ip = 11")
    expect(frEqual(batch.proofs[1].innerProduct, frFromInt(83)), "Batch[1] ip = 83")

    // Verify
    let bVectors = items.map { $0.b }
    let valid = engine.batchVerify(batch: batch, bVectors: bVectors)
    expect(valid, "Batch unblinded proofs verify")
}

// MARK: - Weighted Inner Product

private func testWeightedInnerProduct() {
    suite("GPU IPA Prover — Weighted inner product proof")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
    let b: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(1), frFromInt(1)]
    let w: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

    guard let commitment = try? engine.commitUnblinded(vector: a) else {
        expect(false, "commitUnblinded should not throw")
        return
    }

    guard let proof = try? engine.proveWeighted(
        a: a, b: b, weights: w, commitment: commitment
    ) else {
        expect(false, "proveWeighted should not throw")
        return
    }

    // <a, b*w> = <a, w> = 2*10 + 3*20 + 5*30 + 7*40 = 20 + 60 + 150 + 280 = 510
    let expectedIP = frFromInt(510)
    expect(frEqual(proof.innerProduct, expectedIP), "Weighted IP = 510")

    // Verify with b*w
    var bw = [Fr]()
    for i in 0..<b.count { bw.append(frMul(b[i], w[i])) }
    let valid = engine.verify(proof: proof, commitment: commitment.point, b: bw)
    expect(valid, "Weighted inner product proof verifies")
}

// MARK: - Proof Properties

private func testProofWellFormed() {
    suite("GPU IPA Prover — Proof well-formed check")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let a: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let b: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 10)) }

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    expect(proof.isWellFormed, "Valid proof is well-formed")

    // Construct an invalid proof (mismatched L/R counts)
    let badProof = IPAProverProof(
        Ls: [proof.Ls[0]], Rs: [],
        finalA: proof.finalA, finalB: proof.finalB,
        innerProduct: proof.innerProduct
    )
    expect(!badProof.isWellFormed, "Mismatched L/R counts => not well-formed")

    // Empty proof
    let emptyProof = IPAProverProof(
        Ls: [], Rs: [],
        finalA: Fr.zero, finalB: Fr.zero,
        innerProduct: Fr.zero
    )
    expect(!emptyProof.isWellFormed, "Empty proof is not well-formed")
}

private func testBatchStatistics() {
    suite("GPU IPA Prover — Batch statistics")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let items: [(a: [Fr], b: [Fr])] = [
        (a: [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
         b: [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]),
        (a: [frFromInt(10), frFromInt(20)],
         b: [frFromInt(30), frFromInt(40)])
    ]

    guard let batch = try? engine.batchProveUnblinded(items: items) else {
        expect(false, "batchProveUnblinded should not throw")
        return
    }

    // First proof: 4 elements -> padded to 4 -> 2 rounds
    // Second proof: 2 elements -> padded to 2 -> 1 round
    let totalRounds = batch.totalRounds
    expect(totalRounds >= 3, "Total rounds >= 3 (got \(totalRounds))")

    let totalBytes = batch.approximateByteSize
    expect(totalBytes > 0, "Total byte size > 0")
}

// MARK: - Statistics

private func testProofCountStats() {
    suite("GPU IPA Prover — Proof count stats")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    engine.resetStats()
    expectEqual(engine.proofCount, 0, "Proof count starts at 0")

    let a: [Fr] = [frFromInt(1), frFromInt(2)]
    let b: [Fr] = [frFromInt(3), frFromInt(4)]
    _ = try? engine.proveUnblinded(a: a, b: b)
    expect(engine.proofCount >= 1, "Proof count incremented after prove")

    _ = try? engine.proveUnblinded(a: a, b: b)
    expect(engine.proofCount >= 2, "Proof count incremented again")

    engine.resetStats()
    expectEqual(engine.proofCount, 0, "Proof count reset to 0")
}

// MARK: - Debug Description

private func testDebugDescription() {
    suite("GPU IPA Prover — Debug description")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let desc = engine.debugDescription()
    expect(desc.contains("GPUIPAProverEngine"), "Debug description contains engine name")
    expect(desc.contains("maxSize: 8"), "Debug description contains maxSize")
    expect(desc.contains("generators: 8"), "Debug description contains generator count")
}

// MARK: - Configuration

private func testConfigOptions() {
    suite("GPU IPA Prover — Config options")

    let defaultConfig = IPAProverConfig.default
    expectEqual(defaultConfig.gpuThreshold, 64, "Default GPU threshold = 64")
    expect(defaultConfig.enableBatchParallel, "Default batch parallel enabled")
    expectEqual(defaultConfig.maxBatchConcurrency, 4, "Default max concurrency = 4")

    let cpuConfig = IPAProverConfig.cpuOnly
    expectEqual(cpuConfig.gpuThreshold, Int.max, "CPU-only GPU threshold = Int.max")
    expect(!cpuConfig.enableBatchParallel, "CPU-only batch parallel disabled")
    expectEqual(cpuConfig.maxBatchConcurrency, 1, "CPU-only max concurrency = 1")

    // Custom config
    let custom = IPAProverConfig(gpuThreshold: 128, enableBatchParallel: true, maxBatchConcurrency: 8)
    expectEqual(custom.gpuThreshold, 128, "Custom GPU threshold = 128")
    expectEqual(custom.maxBatchConcurrency, 8, "Custom max concurrency = 8")
}

// MARK: - Larger Vectors

private func testLargerVectorSize() {
    suite("GPU IPA Prover — Larger vector (n=32)")

    guard let engine = try? GPUIPAProverEngine(maxSize: 32) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let n = 32
    let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 * 3 + 1)) }
    let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 * 7 + 2)) }

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw for n=32")
        return
    }

    expectEqual(proof.rounds, 5, "n=32: 5 rounds")
    expect(proof.isWellFormed, "n=32 proof is well-formed")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "n=32 proof verifies correctly")
}

// MARK: - Identity Inner Product

private func testIdentityInnerProduct() {
    suite("GPU IPA Prover — Identity inner product <a, 0> = 0")

    guard let engine = try? GPUIPAProverEngine(maxSize: 8) else {
        print("  [SKIP] Failed to create engine")
        return
    }

    let n = 4
    let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
    let b: [Fr] = [Fr](repeating: Fr.zero, count: n)

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw")
        return
    }

    expect(frEqual(proof.innerProduct, Fr.zero), "<a, 0> = 0")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "Zero b-vector proof verifies")
}

// MARK: - Explicit Generators

private func testEngineWithExplicitGenerators() {
    suite("GPU IPA Prover — Engine with explicit generators")

    // Build generators manually from a reference engine
    guard let ref = try? GPUIPAProverEngine(maxSize: 4) else {
        print("  [SKIP] Failed to create reference engine")
        return
    }

    let engine = GPUIPAProverEngine(
        generators: ref.generators,
        H: ref.H,
        U: ref.U,
        config: .default
    )

    expectEqual(engine.maxSize, 4, "Explicit engine maxSize = 4")
    expectEqual(engine.generators.count, 4, "Explicit engine has 4 generators")

    // Prove and verify with the explicit engine
    let a: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let b: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

    guard let commitment = try? engine.commitUnblinded(vector: a),
          let proof = try? engine.prove(a: a, b: b, commitment: commitment) else {
        expect(false, "prove should not throw for explicit engine")
        return
    }

    // <a, b> = 10 + 40 + 90 + 160 = 300
    let expectedIP = frFromInt(300)
    expect(frEqual(proof.innerProduct, expectedIP), "Explicit engine IP = 300")

    let valid = engine.verify(proof: proof, commitment: commitment.point, b: b)
    expect(valid, "Explicit engine proof verifies")
}
