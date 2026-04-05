import zkMetal
import Foundation

// MARK: - Pedersen Hash Engine (BN254) Tests

public func runPedersenHashTests() {
    suite("Pedersen Hash Engine (BN254)")

    // --- Test 1: Basic commit and open ---
    do {
        let engine = PedersenHashEngine(size: 8)
        let values: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let r = frFromInt(42)
        let c = engine.commit(values: values, blinding: r)

        expect(!pointIsIdentity(c), "Commit produces non-identity point")
        expect(engine.open(values: values, blinding: r, commitment: c),
               "Basic commit and open: valid opening accepted")
    }

    // --- Test 2: Wrong values rejected ---
    do {
        let engine = PedersenHashEngine(size: 4)
        let values: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let r = frFromInt(99)
        let c = engine.commit(values: values, blinding: r)

        let wrong: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(13)]
        expect(!engine.open(values: wrong, blinding: r, commitment: c),
               "Wrong values rejected")
    }

    // --- Test 3: Wrong blinding rejected ---
    do {
        let engine = PedersenHashEngine(size: 4)
        let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let r = frFromInt(42)
        let c = engine.commit(values: values, blinding: r)

        expect(!engine.open(values: values, blinding: frFromInt(43), commitment: c),
               "Wrong blinding rejected")
    }

    // --- Test 4: Homomorphic addition ---
    // Commit(a, r1) + Commit(b, r2) == Commit(a+b, r1+r2)
    do {
        let engine = PedersenHashEngine(size: 4)
        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let b: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let r1 = frFromInt(7)
        let r2 = frFromInt(13)

        let ca = engine.commit(values: a, blinding: r1)
        let cb = engine.commit(values: b, blinding: r2)
        let cSum = PedersenHashEngine.add(ca, cb)

        let ab: [Fr] = (0..<4).map { frAdd(a[$0], b[$0]) }
        let rSum = frAdd(r1, r2)
        let cDirect = engine.commit(values: ab, blinding: rSum)

        expect(pointEqual(cSum, cDirect),
               "Homomorphic addition: Commit(a,r1) + Commit(b,r2) = Commit(a+b, r1+r2)")
    }

    // --- Test 5: Scalar multiplication of commitment ---
    // s * Commit(a, r) == Commit(s*a, s*r)
    do {
        let engine = PedersenHashEngine(size: 4)
        let a: [Fr] = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let r = frFromInt(17)
        let s = frFromInt(5)

        let ca = engine.commit(values: a, blinding: r)
        let cScaled = PedersenHashEngine.scalarMul(s, ca)

        let sa: [Fr] = a.map { frMul($0, s) }
        let sr = frMul(s, r)
        let cDirect = engine.commit(values: sa, blinding: sr)

        expect(pointEqual(cScaled, cDirect),
               "Scalar mul: s * Commit(a,r) = Commit(s*a, s*r)")
    }

    // --- Test 6: Batch commitment ---
    do {
        let engine = PedersenHashEngine(size: 4)
        let v1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let v2: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let v3: [Fr] = [frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)]

        let batch = engine.batchCommit(vectors: [v1, v2, v3])

        expect(batch.count == 3, "Batch commit returns 3 commitments")

        // Each batch result should match individual commit
        let c1 = engine.commit(values: v1)
        let c2 = engine.commit(values: v2)
        let c3 = engine.commit(values: v3)

        expect(pointEqual(batch[0], c1), "Batch[0] matches individual commit")
        expect(pointEqual(batch[1], c2), "Batch[1] matches individual commit")
        expect(pointEqual(batch[2], c3), "Batch[2] matches individual commit")
    }

    // --- Test 7: Deterministic generators ---
    // Same index should produce the same generator every time
    do {
        let g0a = PedersenHashEngine.deriveGenerator(index: 0)
        let g0b = PedersenHashEngine.deriveGenerator(index: 0)
        let g1 = PedersenHashEngine.deriveGenerator(index: 1)

        expect(fpToInt(g0a.x) == fpToInt(g0b.x) && fpToInt(g0a.y) == fpToInt(g0b.y),
               "Deterministic generators: same index -> same point")
        expect(fpToInt(g0a.x) != fpToInt(g1.x) || fpToInt(g0a.y) != fpToInt(g1.y),
               "Different indices -> different generators")
    }

    // --- Test 8: Blinding factor hides commitment ---
    // Same values with different blinding should give different commitments
    do {
        let engine = PedersenHashEngine(size: 4)
        let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let c1 = engine.commit(values: values, blinding: frFromInt(1))
        let c2 = engine.commit(values: values, blinding: frFromInt(2))

        expect(!pointEqual(c1, c2),
               "Different blinding factors produce different commitments")
    }

    // --- Test 9: Zero vector commitment ---
    do {
        let engine = PedersenHashEngine(size: 4)
        let zeros: [Fr] = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]

        // Zero vector with zero blinding should give identity
        let c0 = engine.commit(values: zeros, blinding: Fr.zero)
        expect(pointIsIdentity(c0),
               "Zero vector with zero blinding = identity")

        // Zero vector with non-zero blinding should give r * H
        let r = frFromInt(42)
        let cBlind = engine.commit(values: zeros, blinding: r)
        expect(!pointIsIdentity(cBlind),
               "Zero vector with blinding is non-identity")

        // Verify it opens correctly
        expect(engine.open(values: zeros, blinding: r, commitment: cBlind),
               "Zero vector with blinding opens correctly")
    }

    // --- Test 10: No blinding (nil) commitment ---
    do {
        let engine = PedersenHashEngine(size: 4)
        let values: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let c = engine.commit(values: values, blinding: nil)

        // Should equal commit with zero blinding
        let c0 = engine.commit(values: values, blinding: Fr.zero)
        expect(pointEqual(c, c0),
               "Nil blinding equals zero blinding")
    }

    // --- Test 11: BGMW scalar mul correctness ---
    // Verify that BGMW table-based scalar mul matches naive scalar mul
    do {
        let g = PedersenHashEngine.deriveGenerator(index: 0)
        let table = bgmwBuildTable(generator: g, windowBits: 4)
        let scalar = frFromInt(12345)

        let bgmwResult = bgmwScalarMul(table: table, scalar: scalar)
        let naiveResult = cPointScalarMul(pointFromAffine(g), scalar)

        expect(pointEqual(bgmwResult, naiveResult),
               "BGMW scalar mul matches naive for small scalar")

        // Test with larger scalar
        let bigScalar = frMul(frFromInt(999999), frFromInt(888888))
        let bgmwBig = bgmwScalarMul(table: table, scalar: bigScalar)
        let naiveBig = cPointScalarMul(pointFromAffine(g), bigScalar)

        expect(pointEqual(bgmwBig, naiveBig),
               "BGMW scalar mul matches naive for large scalar")
    }

    // --- Test 12: Partial vector (fewer values than generators) ---
    do {
        let engine = PedersenHashEngine(size: 8)
        let values: [Fr] = [frFromInt(1), frFromInt(2)]  // only 2 of 8
        let r = frFromInt(7)
        let c = engine.commit(values: values, blinding: r)
        expect(engine.open(values: values, blinding: r, commitment: c),
               "Partial vector commit and open works")
    }

    // --- Test 13: Performance benchmark ---
    do {
        let n = 256
        let engine = PedersenHashEngine(size: n)
        let values: [Fr] = (0..<n).map { frFromInt(UInt64($0 + 1)) }
        let r = frFromInt(42)

        let t0 = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            _ = engine.commit(values: values, blinding: r)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let perCommit = elapsed / Double(iterations) * 1000.0

        print("  Pedersen commit (n=\(n), BGMW w=4): \(String(format: "%.2f", perCommit)) ms/commit")
        expect(true, "Performance benchmark completed")
    }
}
