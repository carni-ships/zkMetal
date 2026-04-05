import zkMetal
import Foundation

/// Compare two Fp values for equality (bitwise on internal limbs).
private func fpEqual(_ a: Fp, _ b: Fp) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// Compare two Fr values for equality.
private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// Compare two G1 affine points (by converting both to affine decimal strings).
private func g1ProjectiveEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
    guard let aa = pointToAffine(a), let ba = pointToAffine(b) else {
        return pointIsIdentity(a) && pointIsIdentity(b)
    }
    return fpEqual(aa.x, ba.x) && fpEqual(aa.y, ba.y)
}

/// Compare two G2 projective points via affine.
private func g2ProjectiveEqual(_ a: G2ProjectivePoint, _ b: G2ProjectivePoint) -> Bool {
    guard let aa = g2ToAffine(a), let ba = g2ToAffine(b) else {
        return g2IsIdentity(a) && g2IsIdentity(b)
    }
    return fpEqual(aa.x.c0, ba.x.c0) && fpEqual(aa.x.c1, ba.x.c1) &&
           fpEqual(aa.y.c0, ba.y.c0) && fpEqual(aa.y.c1, ba.y.c1)
}

func runSerializationTests() {
    suite("Decimal Conversion")

    // Test fpToDecimal / fpFromDecimal roundtrip
    let fp1 = fpFromInt(42)
    let fp1Dec = fpToDecimal(fp1)
    expectEqual(fp1Dec, "42", "fpToDecimal(42)")
    if let fp1Back = fpFromDecimal(fp1Dec) {
        let fp1BackDec = fpToDecimal(fp1Back)
        expectEqual(fp1BackDec, "42", "fpFromDecimal roundtrip")
    } else { expect(false, "fpFromDecimal returned nil") }

    // Test frToDecimal / frFromDecimal roundtrip
    let fr1 = frFromInt(12345)
    let fr1Dec = frToDecimal(fr1)
    expectEqual(fr1Dec, "12345", "frToDecimal(12345)")
    if let fr1Back = frFromDecimal(fr1Dec) {
        let fr1BackDec = frToDecimal(fr1Back)
        expectEqual(fr1BackDec, "12345", "frFromDecimal roundtrip")
    } else { expect(false, "frFromDecimal returned nil") }

    // Test zero
    expectEqual(fpToDecimal(.zero), "0", "fpToDecimal(0)")
    expectEqual(frToDecimal(.zero), "0", "frToDecimal(0)")

    // Test one
    expectEqual(fpToDecimal(.one), "1", "fpToDecimal(1)")
    expectEqual(frToDecimal(.one), "1", "frToDecimal(1)")

    // Test large value roundtrip
    let frLarge = frMul(frFromInt(1_000_000_007), frFromInt(1_000_000_009))
    let frLargeDec = frToDecimal(frLarge)
    if let frLargeBack = frFromDecimal(frLargeDec) {
        expectEqual(frToDecimal(frLargeBack), frLargeDec, "large Fr roundtrip")
    } else { expect(false, "frFromDecimal large returned nil") }

    // Test known product: 1000000007 * 1000000009 = 1000000016000000063
    expectEqual(frLargeDec, "1000000016000000063", "fr product value")

    suite("snarkjs Groth16 Serialization")

    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)  // 3^3 + 3 + 5 = 35

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        // Serialize to snarkjs JSON
        let sjProof = proof.toSnarkjs()
        expectEqual(sjProof.protocol_type, "groth16", "Protocol is groth16")
        expect(sjProof.pi_a.count == 3, "pi_a has 3 elements")
        expect(sjProof.pi_b.count == 3, "pi_b has 3 elements")
        expect(sjProof.pi_c.count == 3, "pi_c has 3 elements")
        expectEqual(sjProof.pi_a[2], "1", "pi_a z-coord is 1")

        // pi_b inner arrays should have 2 elements each (Fp2 components)
        for (i, bComp) in sjProof.pi_b.enumerated() {
            expect(bComp.count == 2, "pi_b[\(i)] has 2 components")
        }
        expectEqual(sjProof.pi_b[2], ["1", "0"], "pi_b z-coord is [1,0]")

        // Roundtrip: serialize to JSON data and back, compare field values
        if let jsonData = proof.toSnarkjsJSON() {
            // Verify it's valid JSON
            expect(JSONSerialization.isValidJSONObject(
                try! JSONSerialization.jsonObject(with: jsonData)), "Valid JSON")

            if let proofBack = Groth16Proof.fromSnarkjsJSON(jsonData) {
                // Compare proof elements directly (not pairing verification)
                expect(g1ProjectiveEqual(proof.a, proofBack.a), "pi_a roundtrip exact")
                expect(g2ProjectiveEqual(proof.b, proofBack.b), "pi_b roundtrip exact")
                expect(g1ProjectiveEqual(proof.c, proofBack.c), "pi_c roundtrip exact")
            } else { expect(false, "Groth16Proof.fromSnarkjsJSON returned nil") }
        } else { expect(false, "toSnarkjsJSON returned nil") }

        // VK roundtrip (compare field values)
        if let vkJson = vk.toSnarkjsJSON() {
            if let vkBack = Groth16VerificationKey.fromSnarkjsJSON(vkJson) {
                expect(g1ProjectiveEqual(vk.alpha_g1, vkBack.alpha_g1), "VK alpha roundtrip")
                expect(g2ProjectiveEqual(vk.beta_g2, vkBack.beta_g2), "VK beta roundtrip")
                expect(g2ProjectiveEqual(vk.gamma_g2, vkBack.gamma_g2), "VK gamma roundtrip")
                expect(g2ProjectiveEqual(vk.delta_g2, vkBack.delta_g2), "VK delta roundtrip")
                expectEqual(vkBack.ic.count, vk.ic.count, "VK IC count")
                for i in 0..<vk.ic.count {
                    expect(g1ProjectiveEqual(vk.ic[i], vkBack.ic[i]), "VK IC[\(i)] roundtrip")
                }
                let sjVk = vkBack.toSnarkjs()
                expectEqual(sjVk.nPublic, 2, "VK nPublic")
                expectEqual(sjVk.IC.count, 3, "VK IC snarkjs count")
            } else { expect(false, "Groth16VerificationKey.fromSnarkjsJSON returned nil") }
        } else { expect(false, "VK toSnarkjsJSON returned nil") }

        // Public inputs roundtrip
        let signals = pubInputs.toSnarkjsSignals()
        expectEqual(signals.signals.count, 2, "Signals count")
        expectEqual(signals.signals[0], "3", "Public input x=3")
        expectEqual(signals.signals[1], "35", "Public input y=35")
        if let inputsBack = frArrayFromSnarkjsSignals(signals) {
            expectEqual(frToDecimal(inputsBack[0]), "3", "Recovered x")
            expectEqual(frToDecimal(inputsBack[1]), "35", "Recovered y")
        } else { expect(false, "frArrayFromSnarkjsSignals returned nil") }

    } catch { expect(false, "snarkjs serialization error: \(error)") }

    suite("Ethereum ABI Encoding")

    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)

        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        // Encode to ABI
        if let abiBytes = EthereumABIEncoder.encodeProof(proof: proof, publicInputs: pubInputs) {
            let expectedSize = (2 + 4 + 2 + pubInputs.count) * 32
            expectEqual(abiBytes.count, expectedSize, "ABI size = \(expectedSize)")

            // Decode and compare field values
            if let (proofBack, inputsBack) = EthereumABIEncoder.decodeProof(
                data: abiBytes, numPublicInputs: pubInputs.count) {
                expect(g1ProjectiveEqual(proof.a, proofBack.a), "ABI pi_a roundtrip")
                expect(g2ProjectiveEqual(proof.b, proofBack.b), "ABI pi_b roundtrip")
                expect(g1ProjectiveEqual(proof.c, proofBack.c), "ABI pi_c roundtrip")
                for i in 0..<pubInputs.count {
                    expect(frEqual(pubInputs[i], inputsBack[i]), "ABI input[\(i)] roundtrip")
                }
            } else { expect(false, "ABI decodeProof returned nil") }

            // Hex encoding
            if let hex = EthereumABIEncoder.encodeProofHex(proof: proof, publicInputs: pubInputs) {
                expect(hex.hasPrefix("0x"), "Hex has 0x prefix")
                expectEqual(hex.count, 2 + expectedSize * 2, "Hex length")
            } else { expect(false, "encodeProofHex returned nil") }

        } else { expect(false, "ABI encodeProof returned nil") }

    } catch { expect(false, "ABI encoding error: \(error)") }

    suite("Proof Envelope")

    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)

        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        // Create Groth16 envelope
        let envelope = ProofEnvelope.groth16(
            proof: proof, publicInputs: pubInputs,
            metadata: ["prover": "zkMetal", "version": "1.0"]
        )
        expectEqual(envelope.scheme, .groth16, "Envelope scheme")
        expectEqual(envelope.curve, .bn254, "Envelope curve")
        expectEqual(envelope.version, 1, "Envelope version")
        expectEqual(envelope.publicInputs.count, 2, "Envelope public inputs count")

        // JSON roundtrip
        if let envJson = envelope.toJSON() {
            if let envBack = ProofEnvelope.fromJSON(envJson) {
                expectEqual(envBack.scheme, .groth16, "Roundtrip scheme")
                expectEqual(envBack.curve, .bn254, "Roundtrip curve")
                expectEqual(envBack.publicInputs, envelope.publicInputs, "Roundtrip public inputs")
                expectEqual(envBack.metadata?["prover"], "zkMetal", "Roundtrip metadata")

                // Extract and compare field values
                if let proofBack = envBack.extractGroth16Proof(),
                   let inputsBack = envBack.extractPublicInputs() {
                    expect(g1ProjectiveEqual(proof.a, proofBack.a), "Envelope pi_a roundtrip")
                    expect(g2ProjectiveEqual(proof.b, proofBack.b), "Envelope pi_b roundtrip")
                    expect(g1ProjectiveEqual(proof.c, proofBack.c), "Envelope pi_c roundtrip")
                    for i in 0..<pubInputs.count {
                        expect(frEqual(pubInputs[i], inputsBack[i]), "Envelope input[\(i)] roundtrip")
                    }
                } else { expect(false, "Envelope extraction returned nil") }
            } else { expect(false, "ProofEnvelope.fromJSON returned nil") }
        } else { expect(false, "Envelope toJSON returned nil") }

        // Binary envelope (using KZG proof as example)
        let fakeBytes: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]
        let binEnvelope = ProofEnvelope.binary(
            scheme: .kzg, proofBytes: fakeBytes, publicInputs: pubInputs
        )
        if let binJson = binEnvelope.toJSON(),
           let binBack = ProofEnvelope.fromJSON(binJson) {
            expectEqual(binBack.scheme, .kzg, "Binary envelope scheme")
            if let extracted = binBack.extractBinaryProof() {
                expectEqual(extracted, fakeBytes, "Binary payload roundtrip")
            } else { expect(false, "extractBinaryProof returned nil") }
        } else { expect(false, "Binary envelope JSON roundtrip failed") }

    } catch { expect(false, "Proof envelope error: \(error)") }
}
