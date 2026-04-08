// GPUGroth16VKEngine tests: VK serialization, compression, preprocessing,
// IC computation, validation, and diff utilities
import zkMetal
import Foundation

public func runGPUGroth16VKTests() {
    suite("GPU Groth16 VK Engine")

    // Helper: create a VK from a setup
    func makeVK() -> (Groth16VerificationKey, Groth16ProvingKey, R1CSInstance) {
        let r1cs = buildExampleCircuit()
        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        return (vk, pk, r1cs)
    }

    let engine = GPUGroth16VKEngine()

    // --- Test 1: Binary serialization round-trip ---
    do {
        let (vk, _, _) = makeVK()
        let data = engine.serializeBinary(vk)
        expect(data.count > 0, "Binary serialization produces non-empty data")
        guard let restored = engine.deserializeBinary(data) else {
            expect(false, "Binary deserialization should succeed")
            return
        }
        expect(engine.isEqual(vk, restored), "Binary round-trip preserves VK")
    }

    // --- Test 2: Binary serialization size is correct ---
    do {
        let (vk, _, _) = makeVK()
        let data = engine.serializeBinary(vk)
        let expectedSize = engine.uncompressedSize(vk)
        expectEqual(data.count, expectedSize, "Binary data size matches uncompressed size")
    }

    // --- Test 3: Deserialize invalid data returns nil ---
    do {
        let tooShort = Data([0, 0, 0, 1])  // claims 1 IC but no data
        let result = engine.deserializeBinary(tooShort)
        expect(result == nil, "Deserializing truncated data returns nil")
    }

    // --- Test 4: JSON serialization round-trip ---
    do {
        let (vk, pk, r1cs) = makeVK()
        let json = engine.serializeJSON(vk)
        expect(json["protocol"] as? String == "groth16", "JSON has protocol field")
        expect(json["curve"] as? String == "bn254", "JSON has curve field")

        guard let restored = engine.deserializeJSON(json) else {
            expect(false, "JSON deserialization should succeed")
            return
        }
        // JSON round-trip goes through affine->hex->parse->Montgomery, so check
        // that the VK is functionally equivalent (validates same proofs).
        // We must use the pk from the SAME setup that produced this VK,
        // otherwise the proof won't verify (different toxic waste).
        let (pubInputs, witness) = computeExampleWitness(x: 3)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs,
                                      publicInputs: pubInputs, witness: witness)
        let verifier = Groth16Verifier()
        let valid = verifier.verify(proof: proof, vk: restored, publicInputs: pubInputs)
        expect(valid, "JSON round-trip VK verifies proofs correctly")
    } catch {
        expect(false, "JSON round-trip error: \(error)")
    }

    // --- Test 5: JSON has expected structure ---
    do {
        let (vk, _, _) = makeVK()
        let json = engine.serializeJSON(vk)
        expect(json["alpha_g1"] is [String], "JSON alpha_g1 is string array")
        expect(json["beta_g2"] is [String], "JSON beta_g2 is string array")
        expect(json["ic"] is [[String]], "JSON ic is array of string arrays")
        let nPublic = json["nPublic"] as? Int
        expect(nPublic == vk.ic.count - 1, "JSON nPublic matches ic count - 1")
    }

    // --- Test 6: G1 point compression round-trip ---
    do {
        let (vk, _, _) = makeVK()
        guard let compressed = engine.compressG1(vk.alpha_g1) else {
            expect(false, "G1 compression should succeed")
            return
        }
        expect(compressed.x.count == 4, "Compressed G1 has 4 limbs")
        guard let decompressed = engine.decompressG1(compressed) else {
            expect(false, "G1 decompression should succeed")
            return
        }
        // Check affine coordinates match
        if let origAff = pointToAffine(vk.alpha_g1),
           let decAff = pointToAffine(decompressed) {
            let xMatch = fpSub(origAff.x, decAff.x).isZero
            let yMatch = fpSub(origAff.y, decAff.y).isZero
            expect(xMatch && yMatch, "G1 compression round-trip preserves point")
        } else {
            expect(false, "Affine conversion should succeed")
        }
    }

    // --- Test 7: G2 point compression ---
    do {
        let (vk, _, _) = makeVK()
        guard let compressed = engine.compressG2(vk.beta_g2) else {
            expect(false, "G2 compression should succeed")
            return
        }
        expect(compressed.xC0.count == 4, "Compressed G2 xC0 has 4 limbs")
        expect(compressed.xC1.count == 4, "Compressed G2 xC1 has 4 limbs")
    }

    // --- Test 8: Full VK compression ---
    do {
        let (vk, _, _) = makeVK()
        guard let compressed = engine.compressVK(vk) else {
            expect(false, "VK compression should succeed")
            return
        }
        expect(compressed.ic.count == vk.ic.count, "Compressed VK IC count matches")

        let compSize = engine.compressedSize(vk)
        let uncompSize = engine.uncompressedSize(vk)
        expect(compSize < uncompSize, "Compressed size is smaller than uncompressed")
    }

    // --- Test 9: VK preprocessing ---
    do {
        let (vk, _, _) = makeVK()
        guard let pvk = engine.preprocess(vk) else {
            expect(false, "VK preprocessing should succeed")
            return
        }
        expect(pvk.ic.count == vk.ic.count, "Preprocessed VK IC count matches")
        // Verify the negated alpha is correct
        let sum = pointAdd(pointFromAffine(pvk.alpha_g1),
                           pointFromAffine(pvk.negAlpha_g1))
        expect(pointIsIdentity(sum), "negAlpha + alpha = identity")
    }

    // --- Test 10: Batch preprocessing ---
    do {
        let (vk1, _, _) = makeVK()
        let (vk2, _, _) = makeVK()
        let results = engine.batchPreprocess([vk1, vk2])
        expectEqual(results.count, 2, "Batch preprocess returns correct count")
        expect(results[0] != nil, "First VK preprocessed successfully")
        expect(results[1] != nil, "Second VK preprocessed successfully")
    }

    // --- Test 11: IC computation matches verifier ---
    do {
        let (vk, _, _) = makeVK()
        let (pubInputs, _) = computeExampleWitness(x: 3)

        guard let ic = engine.computeIC(vk: vk, publicInputs: pubInputs) else {
            expect(false, "IC computation should succeed")
            return
        }
        // Manually compute IC the same way the verifier does
        var expected = vk.ic[0]
        for i in 0..<pubInputs.count {
            if !pubInputs[i].isZero {
                expected = pointAdd(expected, pointScalarMul(vk.ic[i + 1], pubInputs[i]))
            }
        }
        expect(pointEqual(ic, expected), "IC computation matches manual calculation")
    }

    // --- Test 12: IC computation with preprocessed VK ---
    do {
        let (vk, _, _) = makeVK()
        let (pubInputs, _) = computeExampleWitness(x: 5)

        guard let pvk = engine.preprocess(vk) else {
            expect(false, "Preprocessing should succeed")
            return
        }
        guard let icFromVK = engine.computeIC(vk: vk, publicInputs: pubInputs),
              let icFromPVK = engine.computeICPreprocessed(pvk: pvk, publicInputs: pubInputs) else {
            expect(false, "IC computation should succeed")
            return
        }
        expect(pointEqual(icFromVK, icFromPVK),
               "IC from VK and preprocessed VK match")
    }

    // --- Test 13: IC computation with zero inputs ---
    do {
        let (vk, _, _) = makeVK()
        let zeroInputs = [Fr](repeating: .zero, count: vk.ic.count - 1)
        guard let ic = engine.computeIC(vk: vk, publicInputs: zeroInputs) else {
            expect(false, "IC computation with zeros should succeed")
            return
        }
        // With all zero inputs, IC should equal IC[0]
        expect(pointEqual(ic, vk.ic[0]), "IC with zero inputs equals IC[0]")
    }

    // --- Test 14: IC computation with wrong count returns nil ---
    do {
        let (vk, _, _) = makeVK()
        let wrongInputs = [Fr.one, Fr.one, Fr.one, Fr.one, Fr.one]  // wrong count
        let result = engine.computeIC(vk: vk, publicInputs: wrongInputs)
        expect(result == nil, "IC with wrong input count returns nil")
    }

    // --- Test 15: VK validation on valid key ---
    do {
        let (vk, _, _) = makeVK()
        let result = engine.validate(vk)
        expect(result.valid, "Valid VK passes validation")
        expect(result.alphaOnCurve, "Alpha is on curve")
        expect(result.betaOnCurve, "Beta is on curve")
        expect(result.gammaOnCurve, "Gamma is on curve")
        expect(result.deltaOnCurve, "Delta is on curve")
        expect(result.icAllOnCurve, "All IC points on curve")
        expect(result.icCountValid, "IC count valid")
        expect(result.pairingConsistent, "Pairing is consistent")
    }

    // --- Test 16: VK validation detects invalid alpha ---
    do {
        // Create a VK with a bogus alpha (not on curve)
        let (vk, _, _) = makeVK()
        let badAlpha = PointProjective(x: fpFromInt(42), y: fpFromInt(99), z: .one)
        let badVK = Groth16VerificationKey(alpha_g1: badAlpha, beta_g2: vk.beta_g2,
                                            gamma_g2: vk.gamma_g2, delta_g2: vk.delta_g2,
                                            ic: vk.ic)
        let result = engine.validate(badVK)
        expect(!result.valid, "VK with bad alpha fails validation")
        expect(!result.alphaOnCurve, "Bad alpha detected as off-curve")
    }

    // --- Test 17: VK validation detects empty IC ---
    do {
        let (vk, _, _) = makeVK()
        let emptyVK = Groth16VerificationKey(alpha_g1: vk.alpha_g1, beta_g2: vk.beta_g2,
                                              gamma_g2: vk.gamma_g2, delta_g2: vk.delta_g2,
                                              ic: [])
        let result = engine.validate(emptyVK)
        expect(!result.icCountValid, "Empty IC detected")
    }

    // --- Test 18: VK diff -- identical VKs ---
    do {
        let (vk, _, _) = makeVK()
        let d = engine.diff(vk, vk)
        expect(d.isIdentical, "Diff of same VK is identical")
        expect(!d.alphaG1Differs, "Alpha matches")
        expect(!d.betaG2Differs, "Beta matches")
        expect(!d.gammaG2Differs, "Gamma matches")
        expect(!d.deltaG2Differs, "Delta matches")
        expect(!d.icCountDiffers, "IC count matches")
        expect(d.icDiffIndices.isEmpty, "No IC differences")
    }

    // --- Test 19: VK diff -- different VKs ---
    do {
        let (vk1, _, _) = makeVK()
        let (vk2, _, _) = makeVK()
        // Two independent setups will produce different keys
        // (due to random toxic waste in setup)
        let d = engine.diff(vk1, vk2)
        // At least some fields should differ since setup uses random tau/alpha/beta
        let anyDiff = d.alphaG1Differs || d.betaG2Differs || d.gammaG2Differs ||
                      d.deltaG2Differs || !d.icDiffIndices.isEmpty
        expect(anyDiff, "Different setups produce different VKs")
    }

    // --- Test 20: VK isEqual ---
    do {
        let (vk, _, _) = makeVK()
        expect(engine.isEqual(vk, vk), "VK equals itself")
    }

    // --- Test 21: VK diff with different IC counts ---
    do {
        let (vk, _, _) = makeVK()
        // Create a VK with one fewer IC point
        let shorterIC = Array(vk.ic.dropLast())
        let shortVK = Groth16VerificationKey(alpha_g1: vk.alpha_g1, beta_g2: vk.beta_g2,
                                              gamma_g2: vk.gamma_g2, delta_g2: vk.delta_g2,
                                              ic: shorterIC)
        let d = engine.diff(vk, shortVK)
        expect(d.icCountDiffers, "Different IC counts detected")
        expect(!d.isIdentical, "VK with different IC count is not identical")
    }

    // --- Test 22: Compression size savings ---
    do {
        let (vk, _, _) = makeVK()
        let compSize = engine.compressedSize(vk)
        let uncompSize = engine.uncompressedSize(vk)
        let savings = Double(uncompSize - compSize) / Double(uncompSize) * 100
        expect(savings > 30.0, "Compression saves >30% (\(String(format: "%.1f", savings))%)")
    }

    // --- Test 23: End-to-end: IC computation + verification ---
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 7)
        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)

        // Compute IC via engine
        guard let ic = engine.computeIC(vk: vk, publicInputs: pubInputs) else {
            expect(false, "IC computation should succeed")
            return
        }

        // Generate a proof and verify using our IC
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs,
                                      publicInputs: pubInputs, witness: witness)

        // Manual pairing check using engine's IC
        guard let pA = pointToAffine(proof.a), let pC = pointToAffine(proof.c),
              let al = pointToAffine(vk.alpha_g1), let vx = pointToAffine(ic) else {
            expect(false, "Affine conversion should succeed")
            return
        }
        guard let pB = g2ToAffine(proof.b), let be = g2ToAffine(vk.beta_g2),
              let ga = g2ToAffine(vk.gamma_g2), let de = g2ToAffine(vk.delta_g2) else {
            expect(false, "G2 affine conversion should succeed")
            return
        }
        let ok = cBN254PairingCheck([(pointNegateAffine(pA), pB), (al, be), (vx, ga), (pC, de)])
        expect(ok, "Verification with engine IC succeeds")
    } catch {
        expect(false, "End-to-end error: \(error)")
    }

    // --- Test 24: Identity point handling ---
    do {
        let identity = pointIdentity()
        let compressed = engine.compressG1(identity)
        // Identity has z=0, so affine conversion returns nil -> compression returns nil
        expect(compressed == nil, "Identity point compression returns nil")
    }

    // --- Test 25: Version is set ---
    do {
        let v = GPUGroth16VKEngine.version
        expect(!v.version.isEmpty, "GPU Groth16 VK Engine version is set")
    }
}
