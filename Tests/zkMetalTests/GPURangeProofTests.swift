import zkMetal

// MARK: - GPU Range Proof Tests

public func runGPURangeProofTests() {
    suite("GPU Range Proof Engine")

    let engine = GPURangeProofEngine()

    // Test 1: Valid 8-bit range proof (value in [0, 256))
    do {
        let generators = IPAGenerators.generate(n: 8)
        let blinding = frFromInt(42)
        let value: UInt64 = 200

        let proof = engine.prove(value: value, blinding: blinding, generators: generators)

        // Verify the commitment matches V = value*g + blinding*h
        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)
        let expectedV = pointAdd(cPointScalarMul(gProj, frFromInt(value)),
                                 cPointScalarMul(hProj, blinding))
        expect(pointEqual(proof.V, expectedV), "8-bit range proof: commitment correct")

        let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
        expect(valid, "8-bit range proof: valid value 200")
    }

    // Test 2: Valid 16-bit range proof
    do {
        let generators = IPAGenerators.generate(n: 16)
        let blinding = frFromInt(1337)
        let value: UInt64 = 50000

        let proof = engine.prove(value: value, blinding: blinding, generators: generators)
        let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
        expect(valid, "16-bit range proof: valid value 50000")
    }

    // Test 3: Edge case — value = 0
    do {
        let generators = IPAGenerators.generate(n: 8)
        let blinding = frFromInt(99)

        let proof = engine.prove(value: 0, blinding: blinding, generators: generators)
        let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
        expect(valid, "8-bit range proof: value = 0")
    }

    // Test 4: Edge case — value = 2^n - 1 (maximum)
    do {
        let generators = IPAGenerators.generate(n: 8)
        let blinding = frFromInt(77)
        let value: UInt64 = 255  // 2^8 - 1

        let proof = engine.prove(value: value, blinding: blinding, generators: generators)
        let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
        expect(valid, "8-bit range proof: value = 255 (max)")
    }

    // Test 5: Verify rejects wrong commitment
    do {
        let generators = IPAGenerators.generate(n: 8)
        let blinding = frFromInt(42)
        let value: UInt64 = 100

        let proof = engine.prove(value: value, blinding: blinding, generators: generators)

        // Create a different commitment (wrong value)
        let gProj = pointFromAffine(generators.g)
        let hProj = pointFromAffine(generators.h)
        let wrongV = pointAdd(cPointScalarMul(gProj, frFromInt(101)),
                              cPointScalarMul(hProj, blinding))

        let valid = engine.verify(proof: proof, commitment: wrongV, generators: generators)
        expect(!valid, "8-bit range proof: rejects wrong commitment")
    }

    // Test 6: Batch verification — all valid
    do {
        let generators = IPAGenerators.generate(n: 8)
        let values: [UInt64] = [10, 100, 200, 50]
        let blindings: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        var proofs = [RangeProof]()
        var commitments = [PointProjective]()
        for i in 0..<values.count {
            let proof = engine.prove(value: values[i], blinding: blindings[i],
                                     generators: generators)
            proofs.append(proof)
            commitments.append(proof.V)
        }

        let valid = engine.batchVerify(proofs: proofs, commitments: commitments,
                                        generators: generators)
        expect(valid, "Batch verify: 4 valid proofs")
    }

    // Test 7: Batch verification — one bad commitment
    do {
        let generators = IPAGenerators.generate(n: 8)
        let blindings: [Fr] = [frFromInt(10), frFromInt(20)]

        let proof0 = engine.prove(value: 50, blinding: blindings[0], generators: generators)
        let proof1 = engine.prove(value: 150, blinding: blindings[1], generators: generators)

        // Swap a commitment to make batch invalid
        let valid = engine.batchVerify(
            proofs: [proof0, proof1],
            commitments: [proof0.V, proof0.V],  // wrong second commitment
            generators: generators)
        expect(!valid, "Batch verify: rejects mismatched commitment")
    }

    // Test 8: Empty batch
    do {
        let generators = IPAGenerators.generate(n: 8)
        let valid = engine.batchVerify(proofs: [], commitments: [], generators: generators)
        expect(valid, "Batch verify: empty batch is valid")
    }

    // Test 9: Proof bit size is recorded correctly
    do {
        let generators = IPAGenerators.generate(n: 16)
        let proof = engine.prove(value: 1000, blinding: frFromInt(55), generators: generators)
        expectEqual(proof.bitSize, 16, "Proof records bit size = 16")
    }

    // Test 10: Different blinding factors produce different proofs
    do {
        let generators = IPAGenerators.generate(n: 8)
        let proof1 = engine.prove(value: 42, blinding: frFromInt(1), generators: generators)
        let proof2 = engine.prove(value: 42, blinding: frFromInt(2), generators: generators)

        // Same value, different blinding -> different V commitments
        expect(!pointEqual(proof1.V, proof2.V),
               "Different blindings produce different commitments")

        // Both should still verify
        let v1 = engine.verify(proof: proof1, commitment: proof1.V, generators: generators)
        let v2 = engine.verify(proof: proof2, commitment: proof2.V, generators: generators)
        expect(v1, "Proof with blinding=1 verifies")
        expect(v2, "Proof with blinding=2 verifies")
    }

    print("  GPU Range Proof engine version: \(GPURangeProofEngine.version.description)")
}
