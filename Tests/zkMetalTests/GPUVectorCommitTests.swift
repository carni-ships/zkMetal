import zkMetal
import Foundation

// MARK: - GPU Vector Commit Engine Tests

public func runGPUVectorCommitTests() {
    suite("GPU Vector Commit Engine")

    let engine = GPUVectorCommitEngine()

    // Helper: generate n distinct affine generator points from BN254 G1
    func makeGenerators(_ n: Int) -> [PointAffine] {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))
        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(n)
        var acc = g
        for _ in 0..<n {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }
        return batchToAffine(projPoints)
    }

    // --- Test 1: Basic commit and verify by recomputation ---
    do {
        let n = 8
        let gens = makeGenerators(n)
        let values: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }

        let c = engine.commit(vector: values, generators: gens)

        // Recompute manually: sum(v_i * G_i)
        var manual = pointIdentity()
        for i in 0..<n {
            let term = cPointScalarMul(pointFromAffine(gens[i]), values[i])
            manual = pointAdd(manual, term)
        }
        expect(pointEqual(c, manual), "Basic commit matches manual MSM")
    }

    // --- Test 2: Commit with single element ---
    do {
        let gens = makeGenerators(1)
        let values: [Fr] = [frFromInt(42)]
        let c = engine.commit(vector: values, generators: gens)
        let expected = cPointScalarMul(pointFromAffine(gens[0]), frFromInt(42))
        expect(pointEqual(c, expected), "Single-element commit: 42 * G_0")
    }

    // --- Test 3: Batch commit consistency ---
    do {
        let n = 8
        let gens = makeGenerators(n)
        let v1: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let v2: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 10)) }
        let v3: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 100)) }

        let batchResults = engine.batchCommit(vectors: [v1, v2, v3], generators: gens)
        let c1 = engine.commit(vector: v1, generators: gens)
        let c2 = engine.commit(vector: v2, generators: gens)
        let c3 = engine.commit(vector: v3, generators: gens)

        expect(pointEqual(batchResults[0], c1), "Batch commit[0] matches individual")
        expect(pointEqual(batchResults[1], c2), "Batch commit[1] matches individual")
        expect(pointEqual(batchResults[2], c3), "Batch commit[2] matches individual")
    }

    // --- Test 4: Batch commit empty ---
    do {
        let gens = makeGenerators(4)
        let results = engine.batchCommit(vectors: [], generators: gens)
        expectEqual(results.count, 0, "Batch commit of empty list returns empty")
    }

    // --- Test 5: Open at index — valid opening ---
    do {
        let n = 8
        let gens = makeGenerators(n)
        let values: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let c = engine.commit(vector: values, generators: gens)

        for idx in [0, 3, 7] {
            let proof = engine.openAt(vector: values, index: idx, generators: gens)
            let valid = engine.verifyOpening(commitment: c, index: idx, value: values[idx],
                                              proof: proof, generators: gens)
            expect(valid, "Open at index \(idx): valid opening accepted")
        }
    }

    // --- Test 6: Open at index — wrong value rejected ---
    do {
        let n = 4
        let gens = makeGenerators(n)
        let values: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let c = engine.commit(vector: values, generators: gens)

        let proof = engine.openAt(vector: values, index: 2, generators: gens)
        // Claim wrong value at index 2
        let invalid = engine.verifyOpening(commitment: c, index: 2, value: frFromInt(999),
                                            proof: proof, generators: gens)
        expect(!invalid, "Open at index 2: wrong value (999 instead of 7) rejected")
    }

    // --- Test 7: Open at index — wrong proof rejected ---
    do {
        let n = 4
        let gens = makeGenerators(n)
        let values: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let c = engine.commit(vector: values, generators: gens)

        // Get proof for index 1, try to use it for index 2
        let proofFor1 = engine.openAt(vector: values, index: 1, generators: gens)
        // Build a fake proof with wrong partial but correct index
        let fakeProof = VectorOpeningProof(partialCommitment: proofFor1.partialCommitment,
                                            openedIndex: 2)
        let invalid = engine.verifyOpening(commitment: c, index: 2, value: values[2],
                                            proof: fakeProof, generators: gens)
        expect(!invalid, "Open at index 2: wrong partial commitment rejected")
    }

    // --- Test 8: Open at index — wrong commitment rejected ---
    do {
        let n = 4
        let gens = makeGenerators(n)
        let values: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let c = engine.commit(vector: values, generators: gens)

        let proof = engine.openAt(vector: values, index: 0, generators: gens)

        // Perturb commitment
        let wrongC = pointAdd(c, pointFromAffine(gens[0]))
        let invalid = engine.verifyOpening(commitment: wrongC, index: 0, value: values[0],
                                            proof: proof, generators: gens)
        expect(!invalid, "Open at index 0: wrong commitment rejected")
    }

    // --- Test 9: Open every index in a vector ---
    do {
        let n = 16
        let gens = makeGenerators(n)
        let values: [Fr] = (0..<n).map { frFromInt(UInt64($0 * 3 + 7)) }
        let c = engine.commit(vector: values, generators: gens)

        var allValid = true
        for i in 0..<n {
            let proof = engine.openAt(vector: values, index: i, generators: gens)
            if !engine.verifyOpening(commitment: c, index: i, value: values[i],
                                      proof: proof, generators: gens) {
                allValid = false
                break
            }
        }
        expect(allValid, "All 16 index openings verify correctly")
    }

    // --- Test 10: Linearity — commit(a+b) == commit(a) + commit(b) ---
    do {
        let n = 8
        let gens = makeGenerators(n)
        let a: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let b: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 10)) }
        let ab: [Fr] = (0..<n).map { frAdd(a[$0], b[$0]) }

        let ca = engine.commit(vector: a, generators: gens)
        let cb = engine.commit(vector: b, generators: gens)
        let cab = engine.commit(vector: ab, generators: gens)

        let sum = pointAdd(ca, cb)
        expect(pointEqual(sum, cab), "Linearity: commit(a) + commit(b) == commit(a+b)")
    }

    // --- Test 11: Larger vector (n=64) commit and open ---
    do {
        let n = 64
        let gens = makeGenerators(n)
        let values: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let c = engine.commit(vector: values, generators: gens)

        let proof = engine.openAt(vector: values, index: 33, generators: gens)
        let valid = engine.verifyOpening(commitment: c, index: 33, value: values[33],
                                          proof: proof, generators: gens)
        expect(valid, "n=64: open at index 33 verifies")
    }

    // --- Test 12: Version is set ---
    do {
        let v = GPUVectorCommitEngine.version
        expect(!v.version.isEmpty, "Version string is non-empty: \(v.description)")
    }
}
