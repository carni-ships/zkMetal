// Sparse Polynomial Commitment Tests

import Foundation
import zkMetal

public func runSparsePolyCommitTests() {
    suite("SparsePolyCommit")

    // Shared test SRS
    let secret: [UInt32] = [7, 0, 0, 0, 0, 0, 0, 0]
    let srsSecret = frFromLimbs(secret)
    let generator = PointAffine(x: .one, y: Fp(v: (
        0x2611_5a1b, 0xa722_44a0, 0x1f23_2048, 0x66b3_ff14,
        0x3250_d688, 0x4eb1_de7e, 0x29e7_52c2, 0x1971_ff07
    )))
    let srsSize = 1024
    let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)

    // -------------------------------------------------------
    // 1. SparsePoly basic operations
    // -------------------------------------------------------
    do {
        // Create sparse poly: 3x^2 + 5x^10
        let three = frFromInt(3)
        let five = frFromInt(5)
        let sp = SparsePoly(terms: [(index: 2, coeff: three), (index: 10, coeff: five)], degreeBound: 16)
        expectEqual(sp.nnz, 2, "sparse nnz")

        // Convert to dense and back
        let dense = sp.toDense()
        expectEqual(dense.count, 16, "dense length")
        expect(frToInt(dense[2]) == frToInt(three), "dense[2] == 3")
        expect(frToInt(dense[10]) == frToInt(five), "dense[10] == 5")
        expect(frToInt(dense[0]) == frToInt(Fr.zero), "dense[0] == 0")

        let sp2 = SparsePoly(dense: dense)
        expectEqual(sp2.nnz, 2, "roundtrip nnz")
    }

    // -------------------------------------------------------
    // 2. SparsePoly evaluation
    // -------------------------------------------------------
    do {
        let three = frFromInt(3)
        let five = frFromInt(5)
        let sp = SparsePoly(terms: [(index: 2, coeff: three), (index: 10, coeff: five)], degreeBound: 16)

        let z = frFromInt(2)
        let sparseEval = sp.evaluate(at: z)

        // Dense evaluation for comparison via Horner manually:
        // 3*2^2 + 5*2^10 = 12 + 5120 = 5132
        let dense = sp.toDense()
        var denseEval = Fr.zero
        var zPow = Fr.one
        for c in dense {
            denseEval = frAdd(denseEval, frMul(c, zPow))
            zPow = frMul(zPow, z)
        }
        expect(frToInt(sparseEval) == frToInt(denseEval), "sparse eval matches dense eval")

        // Check numeric value
        let expected = frFromInt(5132)
        expect(frToInt(sparseEval) == frToInt(expected), "eval = 5132")
    }

    // -------------------------------------------------------
    // 3. Sparse commit matches dense commit for same polynomial
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)

        // Build a sparse polynomial with known non-zero entries
        let terms: [(index: Int, coeff: Fr)] = [
            (0, frFromInt(7)),
            (3, frFromInt(11)),
            (15, frFromInt(42)),
            (100, frFromInt(99)),
        ]
        let sp = SparsePoly(terms: terms, degreeBound: 128)

        let sparseCommit = sparseEngine.commit(sp)
        let denseCommit = sparseEngine.commitDense(sp.toDense())

        expect(pointEqual(sparseCommit, denseCommit),
               "sparse KZG commit matches dense commit")
    }

    // -------------------------------------------------------
    // 4. Opening proof verification
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)
        let openingEngine = SparseOpening(srs: srs)

        let terms: [(index: Int, coeff: Fr)] = [
            (0, frFromInt(1)),
            (1, frFromInt(2)),
            (5, frFromInt(3)),
        ]
        let sp = SparsePoly(terms: terms, degreeBound: 64)
        let commitment = sparseEngine.commit(sp)

        let z = frFromInt(4)
        let proof = openingEngine.open(sp, at: z)

        // Verify: p(4) = 1 + 2*4 + 3*4^5 = 1 + 8 + 3072 = 3081
        let expectedEval = frFromInt(3081)
        expect(frToInt(proof.evaluation) == frToInt(expectedEval), "opening eval = 3081")

        // Verify the proof
        let valid = openingEngine.verify(
            commitment: commitment,
            point: z,
            proof: proof,
            srsSecret: srsSecret
        )
        expect(valid, "sparse opening proof verifies")

        // Verify with wrong evaluation fails
        let badProof = SparseOpeningProof(evaluation: frFromInt(9999), witness: proof.witness)
        let invalid = openingEngine.verify(
            commitment: commitment,
            point: z,
            proof: badProof,
            srsSecret: srsSecret
        )
        expect(!invalid, "bad opening proof rejected")
    }

    // -------------------------------------------------------
    // 5. Batch commitment correctness
    // -------------------------------------------------------
    do {
        let batchEngine = BatchSparseCommit(srs: srs)
        let sparseEngine = SparseKZG(srs: srs)

        let p1 = SparsePoly(terms: [(0, frFromInt(5)), (10, frFromInt(3))], degreeBound: 32)
        let p2 = SparsePoly(terms: [(1, frFromInt(7)), (10, frFromInt(2)), (20, frFromInt(1))], degreeBound: 32)
        let p3 = SparsePoly(terms: [(5, frFromInt(11))], degreeBound: 32)

        // Individual commits
        let c1 = sparseEngine.commit(p1)
        let c2 = sparseEngine.commit(p2)
        let c3 = sparseEngine.commit(p3)

        // Batch commits (basic)
        let batchBasic = batchEngine.commitAll([p1, p2, p3])
        expect(pointEqual(batchBasic[0], c1), "batch[0] matches individual")
        expect(pointEqual(batchBasic[1], c2), "batch[1] matches individual")
        expect(pointEqual(batchBasic[2], c3), "batch[2] matches individual")

        // Batch commits (shared-index optimization)
        let batchShared = batchEngine.commitShared([p1, p2, p3])
        expect(pointEqual(batchShared[0], c1), "shared[0] matches individual")
        expect(pointEqual(batchShared[1], c2), "shared[1] matches individual")
        expect(pointEqual(batchShared[2], c3), "shared[2] matches individual")
    }

    // -------------------------------------------------------
    // 6. Performance: sparse vs dense commit for 1% density
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)
        let degree = 512
        let density = 0.01  // 1% density -> ~5 non-zero coeffs

        // Build sparse polynomial
        var terms = [(index: Int, coeff: Fr)]()
        let nnz = max(1, Int(Double(degree) * density))
        // Deterministic "random" indices
        for i in 0..<nnz {
            let idx = (i * 97 + 13) % degree  // pseudo-random spread
            terms.append((index: idx, coeff: frFromInt(UInt64(i + 1))))
        }
        let sp = SparsePoly(terms: terms, degreeBound: degree)
        let dense = sp.toDense()

        // Time sparse commit
        let t0 = CFAbsoluteTimeGetCurrent()
        var sparseResult = pointIdentity()
        for _ in 0..<100 {
            sparseResult = sparseEngine.commit(sp)
        }
        let sparseTime = CFAbsoluteTimeGetCurrent() - t0

        // Time dense commit
        let t1 = CFAbsoluteTimeGetCurrent()
        var denseResult = pointIdentity()
        for _ in 0..<100 {
            denseResult = sparseEngine.commitDense(dense)
        }
        let denseTime = CFAbsoluteTimeGetCurrent() - t1

        // Results should match
        expect(pointEqual(sparseResult, denseResult), "sparse/dense commit match for 1% density")

        // Sparse should be faster (at 1% density, ~100x fewer MSM elements)
        let speedup = denseTime / max(sparseTime, 1e-9)
        print("  sparse vs dense (1% density, n=\(degree)): sparse=\(String(format: "%.3f", sparseTime*10))ms, dense=\(String(format: "%.3f", denseTime*10))ms, speedup=\(String(format: "%.1f", speedup))x")
        expect(speedup > 1.0, "sparse commit faster than dense at 1% density")
    }

    // -------------------------------------------------------
    // 7. R1CS matrix commitment use case
    // -------------------------------------------------------
    do {
        let batchEngine = BatchSparseCommit(srs: srs)

        // Simulate R1CS A, B, C matrices as sparse column polynomials
        // Typical R1CS: each row has 2-3 non-zero entries across A, B, C
        let numVars = 128

        // Matrix A: identity-like with some structure
        var aTerms = [(index: Int, coeff: Fr)]()
        for i in stride(from: 0, to: numVars, by: 4) {
            aTerms.append((index: i, coeff: Fr.one))
        }
        let polyA = SparsePoly(terms: aTerms, degreeBound: numVars)

        // Matrix B: sparse with scattered entries
        var bTerms = [(index: Int, coeff: Fr)]()
        for i in stride(from: 1, to: numVars, by: 7) {
            bTerms.append((index: i, coeff: frFromInt(UInt64(i + 1))))
        }
        let polyB = SparsePoly(terms: bTerms, degreeBound: numVars)

        // Matrix C: very sparse (output wires)
        let polyC = SparsePoly(terms: [
            (0, frFromInt(1)),
            (numVars - 1, frFromInt(1)),
        ], degreeBound: numVars)

        // Commit all three using shared-index optimization
        let commitments = batchEngine.commitShared([polyA, polyB, polyC])
        expectEqual(commitments.count, 3, "R1CS 3 matrix commitments")

        // Verify each matches individual commit
        let sparseEngine = SparseKZG(srs: srs)
        expect(pointEqual(commitments[0], sparseEngine.commit(polyA)), "R1CS A commit")
        expect(pointEqual(commitments[1], sparseEngine.commit(polyB)), "R1CS B commit")
        expect(pointEqual(commitments[2], sparseEngine.commit(polyC)), "R1CS C commit")

        print("  R1CS matrix commit: A(\(polyA.nnz) nnz), B(\(polyB.nnz) nnz), C(\(polyC.nnz) nnz) / \(numVars) total")
    }

    // -------------------------------------------------------
    // 8. Selector polynomial commitment (mostly zeros)
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)
        let openingEngine = SparseOpening(srs: srs)

        // Plonk selector: q_M is 1 only at multiplication gates
        // Simulate: 256 gates, only 10% are multiplications
        let numGates = 256
        var selectorTerms = [(index: Int, coeff: Fr)]()
        for i in stride(from: 0, to: numGates, by: 10) {
            selectorTerms.append((index: i, coeff: Fr.one))
        }
        let qM = SparsePoly(terms: selectorTerms, degreeBound: numGates)
        expect(qM.density < 0.15, "selector density < 15%")

        // Commit
        let commitment = sparseEngine.commit(qM)
        expect(!pointIsIdentity(commitment), "selector commit non-trivial")

        // Verify matches dense
        let denseCommit = sparseEngine.commitDense(qM.toDense())
        expect(pointEqual(commitment, denseCommit), "selector sparse == dense commit")

        // Open at a random point and verify
        let z = frFromInt(17)
        let proof = openingEngine.open(qM, at: z)
        let valid = openingEngine.verify(
            commitment: commitment,
            point: z,
            proof: proof,
            srsSecret: srsSecret
        )
        expect(valid, "selector opening proof verifies")

        print("  selector q_M: \(qM.nnz)/\(numGates) non-zero (\(String(format: "%.0f", qM.density * 100))% density)")
    }

    // -------------------------------------------------------
    // 9. Batch verify multiple openings
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)
        let openingEngine = SparseOpening(srs: srs)

        // Create 3 sparse polynomials
        let polys = [
            SparsePoly(terms: [(0, frFromInt(1)), (3, frFromInt(5))], degreeBound: 32),
            SparsePoly(terms: [(1, frFromInt(2)), (7, frFromInt(3))], degreeBound: 32),
            SparsePoly(terms: [(0, frFromInt(4)), (15, frFromInt(6))], degreeBound: 32),
        ]

        let commitments = polys.map { sparseEngine.commit($0) }
        let points = [frFromInt(3), frFromInt(5), frFromInt(7)]
        let proofs = zip(polys, points).map { openingEngine.open($0.0, at: $0.1) }

        let gamma = frFromInt(13)
        let batchValid = SparseVerification.batchVerify(
            commitments: commitments,
            points: points,
            proofs: proofs,
            gamma: gamma,
            srs: srs,
            srsSecret: srsSecret
        )
        expect(batchValid, "batch verify multiple sparse openings")
    }

    // -------------------------------------------------------
    // 10. SparsePoly arithmetic
    // -------------------------------------------------------
    do {
        let p1 = SparsePoly(terms: [(0, frFromInt(3)), (5, frFromInt(7))], degreeBound: 8)
        let p2 = SparsePoly(terms: [(0, frFromInt(2)), (3, frFromInt(4))], degreeBound: 8)

        let sum = p1.add(p2)
        let z = frFromInt(2)
        let sumEval = sum.evaluate(at: z)

        // p1(2) = 3 + 7*32 = 227, p2(2) = 2 + 4*8 = 34, sum(2) = 261
        let expected = frFromInt(261)
        expect(frToInt(sumEval) == frToInt(expected), "sparse add eval correct")

        // Scale
        let scaled = p1.scale(by: frFromInt(10))
        let scaledEval = scaled.evaluate(at: z)
        let expectedScaled = frFromInt(2270)  // 10 * 227
        expect(frToInt(scaledEval) == frToInt(expectedScaled), "sparse scale eval correct")
    }

    // -------------------------------------------------------
    // 11. Linear combination commitment
    // -------------------------------------------------------
    do {
        let batchEngine = BatchSparseCommit(srs: srs)
        let sparseEngine = SparseKZG(srs: srs)

        let p1 = SparsePoly(terms: [(0, frFromInt(5)), (3, frFromInt(2))], degreeBound: 8)
        let p2 = SparsePoly(terms: [(1, frFromInt(3)), (3, frFromInt(1))], degreeBound: 8)

        let gamma = frFromInt(7)

        // Linear combination: p1 + gamma * p2
        let lcCommit = batchEngine.commitLinearCombination([p1, p2], gamma: gamma)

        // Manual: combined = p1 + 7*p2
        // = 5 + 21*x + 2*x^3 + 7*x^3 = 5 + 21*x + 9*x^3
        let combined = SparsePoly(terms: [
            (0, frFromInt(5)),
            (1, frMul(frFromInt(7), frFromInt(3))),
            (3, frAdd(frFromInt(2), frMul(frFromInt(7), frFromInt(1)))),
        ], degreeBound: 8)
        let directCommit = sparseEngine.commit(combined)

        expect(pointEqual(lcCommit, directCommit), "linear combination commit correct")
    }

    // -------------------------------------------------------
    // 12. Empty and trivial cases
    // -------------------------------------------------------
    do {
        let sparseEngine = SparseKZG(srs: srs)

        // Empty polynomial
        let empty = SparsePoly(terms: [], degreeBound: 16)
        expectEqual(empty.nnz, 0, "empty nnz")
        let emptyCommit = sparseEngine.commit(empty)
        expect(pointIsIdentity(emptyCommit), "empty commit is identity")

        // Single term
        let single = SparsePoly(terms: [(0, Fr.one)], degreeBound: 1)
        let singleCommit = sparseEngine.commit(single)
        expect(!pointIsIdentity(singleCommit), "single-term commit non-trivial")

        // Zero coefficient filtered out
        let withZero = SparsePoly(terms: [(0, Fr.zero), (1, frFromInt(5))], degreeBound: 4)
        expectEqual(withZero.nnz, 1, "zero coeff filtered")
    }

    // -------------------------------------------------------
    // 13. Sparse IPA commitment matches dense
    // -------------------------------------------------------
    do {
        // Use a power-of-2 size for IPA generators
        let n = 64
        let ipaGens = Array(srs.prefix(n))
        let qPoint = srs[n]  // use an extra SRS point as Q

        let sparseIPA = SparseIPA(generators: ipaGens, Q: qPoint)

        let terms: [(index: Int, coeff: Fr)] = [
            (0, frFromInt(3)),
            (7, frFromInt(11)),
            (31, frFromInt(5)),
            (63, frFromInt(2)),
        ]
        let sp = SparsePoly(terms: terms, degreeBound: n)

        let sparseCommit = sparseIPA.commit(sp)

        // Dense commit using cPippengerMSMFlat with frToLimbs
        let dense = sp.toDense()
        var flatScalars = [UInt32]()
        flatScalars.reserveCapacity(n * 8)
        for c in dense {
            flatScalars.append(contentsOf: frToLimbs(c))
        }
        let denseCommit = cPippengerMSMFlat(points: ipaGens, flatScalars: flatScalars)

        expect(pointEqual(sparseCommit, denseCommit), "sparse IPA matches dense IPA commit")
    }
}
