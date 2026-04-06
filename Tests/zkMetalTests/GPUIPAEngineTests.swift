// GPUIPAEngineTests — Tests for GPU-accelerated IPA polynomial commitment engine
//
// Validates commit correctness, single opening, batch opening,
// wrong value rejection, and round count.

import Foundation
import Metal
@testable import zkMetal

public func runGPUIPAEngineTests() {
    suite("GPUIPAEngine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    // Use n=8 (power of 2) for fast tests
    let n = 8
    guard let engine = try? GPUIPAEngine(maxDegree: n) else {
        print("  [SKIP] Failed to create GPUIPAEngine")
        return
    }

    // --- Test 1: Commit correctness ---
    // Commit should match manual MSM over generators
    do {
        let coeffs: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let c = try! engine.commit(coeffs)

        // Manual MSM: sum(coeff_i * G_i)
        var manual = pointIdentity()
        for i in 0..<n {
            let term = cPointScalarMul(pointFromAffine(engine.generators[i]), coeffs[i])
            manual = pointAdd(manual, term)
        }
        expect(pointEqual(c, manual), "Commit matches manual MSM over generators")
    }

    // --- Test 2: Commit with zero polynomial ---
    do {
        let zero: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let c = try! engine.commit(zero)
        expect(pointIsIdentity(c), "Commit to zero polynomial is identity point")
    }

    // --- Test 3: Single opening proof — valid ---
    do {
        let coeffs: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11),
                             Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = frFromInt(2)
        let c = try! engine.commit(coeffs)
        let proof = try! engine.open(coeffs, at: z)

        // p(2) = 3 + 5*2 + 7*4 + 11*8 = 3 + 10 + 28 + 88 = 129
        let expected = frFromInt(129)
        expect(frEqual(proof.evaluation, expected), "Evaluation p(2) = 129")

        let valid = engine.verify(commitment: c, point: z, proof: proof)
        expect(valid, "Single opening proof verifies correctly")
    }

    // --- Test 4: Single opening — wrong value rejection ---
    do {
        let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                             Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = frFromInt(5)
        let c = try! engine.commit(coeffs)
        let proof = try! engine.open(coeffs, at: z)

        // Tamper with the evaluation
        let tamperedProof = IPAOpeningProof(
            Ls: proof.Ls, Rs: proof.Rs,
            finalA: proof.finalA,
            evaluation: frFromInt(999)
        )
        let invalid = engine.verify(commitment: c, point: z, proof: tamperedProof)
        expect(!invalid, "Tampered evaluation rejected by verifier")
    }

    // --- Test 5: Round count check ---
    do {
        let coeffs: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let z = frFromInt(3)
        let proof = try! engine.open(coeffs, at: z)

        let expectedRounds = Int(log2(Double(n)))
        expectEqual(proof.rounds, expectedRounds,
                    "Proof has log2(\(n)) = \(expectedRounds) rounds")
        expectEqual(proof.Ls.count, expectedRounds, "L count = \(expectedRounds)")
        expectEqual(proof.Rs.count, expectedRounds, "R count = \(expectedRounds)")
    }

    // --- Test 6: Opening at z=0 ---
    do {
        let coeffs: [Fr] = [frFromInt(42), frFromInt(7), frFromInt(3), frFromInt(1),
                             Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = Fr.zero
        let c = try! engine.commit(coeffs)
        let proof = try! engine.open(coeffs, at: z)

        // p(0) = constant term = 42
        expect(frEqual(proof.evaluation, frFromInt(42)), "p(0) = constant term 42")
        let valid = engine.verify(commitment: c, point: z, proof: proof)
        expect(valid, "Opening at z=0 verifies")
    }

    // --- Test 7: Opening at z=1 ---
    do {
        let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                             Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = Fr.one
        let c = try! engine.commit(coeffs)
        let proof = try! engine.open(coeffs, at: z)

        // p(1) = 1 + 2 + 3 + 4 = 10
        expect(frEqual(proof.evaluation, frFromInt(10)), "p(1) = sum of coefficients = 10")
        let valid = engine.verify(commitment: c, point: z, proof: proof)
        expect(valid, "Opening at z=1 verifies")
    }

    // --- Test 8: Batch opening — valid ---
    do {
        let p1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                         Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let p2: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8),
                         Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = frFromInt(2)
        let gamma = frFromInt(13)

        let c1 = try! engine.commit(p1)
        let c2 = try! engine.commit(p2)

        let batchProof = try! engine.batchOpen(polys: [p1, p2], at: z, gamma: gamma)

        // p1(2) = 1 + 4 + 12 + 32 = 49
        // p2(2) = 5 + 12 + 28 + 64 = 109
        expect(frEqual(batchProof.evaluations[0], frFromInt(49)), "Batch: p1(2) = 49")
        expect(frEqual(batchProof.evaluations[1], frFromInt(109)), "Batch: p2(2) = 109")

        let valid = engine.batchVerify(commitments: [c1, c2], proof: batchProof)
        expect(valid, "Batch opening proof verifies correctly")
    }

    // --- Test 9: Batch opening — wrong commitment rejected ---
    do {
        let p1: [Fr] = [frFromInt(10), frFromInt(20), Fr.zero, Fr.zero,
                         Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let p2: [Fr] = [frFromInt(30), frFromInt(40), Fr.zero, Fr.zero,
                         Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        let z = frFromInt(3)
        let gamma = frFromInt(7)

        let c1 = try! engine.commit(p1)
        let c2 = try! engine.commit(p2)

        let batchProof = try! engine.batchOpen(polys: [p1, p2], at: z, gamma: gamma)

        // Use wrong commitment for p1
        let wrongC1 = pointAdd(c1, pointFromAffine(engine.generators[0]))
        let invalid = engine.batchVerify(commitments: [wrongC1, c2], proof: batchProof)
        expect(!invalid, "Batch verify rejects wrong commitment")
    }

    // --- Test 10: Version is set ---
    do {
        let v = GPUIPAEngine.version
        expect(!v.version.isEmpty, "Version string is non-empty: \(v.description)")
    }
}
