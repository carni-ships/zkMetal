// Groth16 Solidity verifier generation and serialization tests
import zkMetal
import Foundation

func runGroth16SolidityTests() {
    suite("Groth16 Solidity Verifier Generation")

    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)

        let setup = Groth16Setup()
        let (pk, vk) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        // --- Solidity contract generation ---
        let solidity = generateSolidityVerifier(vk: vk)
        expect(solidity.contains("pragma solidity"), "Solidity has pragma")
        expect(solidity.contains("contract Groth16Verifier"), "Solidity has contract name")
        expect(solidity.contains("ecAdd precompile at 0x06"), "Uses ecAdd precompile")
        expect(solidity.contains("ecMul precompile at 0x07"), "Uses ecMul precompile")
        expect(solidity.contains("ecPairing precompile at 0x08"), "Uses ecPairing precompile")
        expect(solidity.contains("ALPHA_X"), "Has alpha VK constant")
        expect(solidity.contains("BETA_X1"), "Has beta VK constant")
        expect(solidity.contains("GAMMA_X1"), "Has gamma VK constant")
        expect(solidity.contains("DELTA_X1"), "Has delta VK constant")
        expect(solidity.contains("IC0_X"), "Has IC[0] constant")
        expect(solidity.contains("IC1_X"), "Has IC[1] constant")
        expect(solidity.contains("IC2_X"), "Has IC[2] constant")
        expect(solidity.contains("verifyProof("), "Has verifyProof function")
        expect(solidity.contains("uint256[2] memory a"), "Correct a parameter type")
        expect(solidity.contains("uint256[2][2] memory b"), "Correct b parameter type")
        expect(solidity.contains("uint256[2] memory input"), "Correct input size for 2 public inputs")
        expect(solidity.contains("negate(a[1])"), "Negates proof A")
        expect(solidity.contains("pairing_result[0] == 1"), "Checks pairing result")

        // --- Solidity export via exportVerificationKey ---
        if let solidityData = exportVerificationKey(vk: vk, format: .solidity) {
            let solidityStr = String(data: solidityData, encoding: .utf8) ?? ""
            expect(solidityStr.contains("contract Groth16Verifier"), "exportVK .solidity works")
        } else { expect(false, "exportVK .solidity returned nil") }

        // --- Calldata generation ---
        let calldata = generateCalldata(proof: proof, publicInputs: pubInputs)
        expect(calldata.hasPrefix("0x43753b4d"), "Calldata has function selector")
        // 4 bytes selector + (2+4+2+2)*32 = 324 bytes = 648 hex chars + 8 selector = 656 + "0x"
        let expectedHexLen = 2 + 8 + (2 + 4 + 2 + pubInputs.count) * 64
        expectEqual(calldata.count, expectedHexLen, "Calldata hex length")
        // All chars should be hex
        let calldataBody = String(calldata.dropFirst(2))
        expect(calldataBody.allSatisfy { $0.isHexDigit }, "Calldata is valid hex")

        // --- JSON proof export ---
        if let jsonData = exportProof(proof: proof, publicInputs: pubInputs, format: .json) {
            let jsonObj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
            expect(jsonObj != nil, "JSON proof is valid JSON object")
            if let obj = jsonObj {
                expect(obj["proof"] != nil, "JSON has proof field")
                expect(obj["publicSignals"] != nil, "JSON has publicSignals field")

                if let proofDict = obj["proof"] as? [String: Any] {
                    expect(proofDict["protocol"] as? String == "groth16", "Protocol is groth16")
                    expect(proofDict["pi_a"] != nil, "Has pi_a")
                    expect(proofDict["pi_b"] != nil, "Has pi_b")
                    expect(proofDict["pi_c"] != nil, "Has pi_c")

                    // pi_b should be array of arrays
                    if let piB = proofDict["pi_b"] as? [[String]] {
                        expectEqual(piB.count, 3, "pi_b has 3 elements")
                        for subArr in piB {
                            expectEqual(subArr.count, 2, "pi_b sub-array has 2 elements")
                        }
                    } else { expect(false, "pi_b is not [[String]]") }
                }

                if let signals = obj["publicSignals"] as? [String] {
                    expectEqual(signals.count, 2, "2 public signals")
                    expectEqual(signals[0], "3", "Public input x=3")
                    expectEqual(signals[1], "35", "Public input y=35")
                }
            }
        } else { expect(false, "exportProof .json returned nil") }

        // --- VK JSON roundtrip ---
        suite("Groth16 Solidity VK Roundtrip")

        if let vkJsonData = exportVerificationKey(vk: vk, format: .json) {
            if let vkBack = importVerificationKey(data: vkJsonData, format: .json) {
                // Compare alpha
                if let origAlpha = pointToAffine(vk.alpha_g1),
                   let backAlpha = pointToAffine(vkBack.alpha_g1) {
                    expectEqual(fpToDecimal(origAlpha.x), fpToDecimal(backAlpha.x), "VK JSON alpha.x roundtrip")
                    expectEqual(fpToDecimal(origAlpha.y), fpToDecimal(backAlpha.y), "VK JSON alpha.y roundtrip")
                } else { expect(false, "VK alpha affine conversion failed") }

                expectEqual(vkBack.ic.count, vk.ic.count, "VK JSON IC count roundtrip")
            } else { expect(false, "importVK .json returned nil") }
        } else { expect(false, "exportVK .json returned nil") }

        // --- VK Binary roundtrip ---
        if let vkBinData = exportVerificationKey(vk: vk, format: .binary) {
            // Expected size: 4 + 64 + 128*3 + 3*64 = 4 + 64 + 384 + 192 = 644
            let expectedSize = 4 + 64 + 128 * 3 + vk.ic.count * 64
            expectEqual(vkBinData.count, expectedSize, "VK binary size")

            if let vkBack = importVerificationKey(data: vkBinData, format: .binary) {
                if let origAlpha = pointToAffine(vk.alpha_g1),
                   let backAlpha = pointToAffine(vkBack.alpha_g1) {
                    expectEqual(fpToDecimal(origAlpha.x), fpToDecimal(backAlpha.x), "VK binary alpha.x roundtrip")
                    expectEqual(fpToDecimal(origAlpha.y), fpToDecimal(backAlpha.y), "VK binary alpha.y roundtrip")
                } else { expect(false, "VK binary alpha affine conversion failed") }

                expectEqual(vkBack.ic.count, vk.ic.count, "VK binary IC count roundtrip")

                // Verify the round-tripped VK still works for verification
                let verifier = Groth16Verifier()
                let validWithOrigVK = verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs)
                let validWithRoundtrippedVK = verifier.verify(proof: proof, vk: vkBack, publicInputs: pubInputs)
                expect(validWithOrigVK == validWithRoundtrippedVK,
                       "Binary VK roundtrip preserves verification result")
            } else { expect(false, "importVK .binary returned nil") }
        } else { expect(false, "exportVK .binary returned nil") }

        // --- Proof binary roundtrip ---
        suite("Groth16 Solidity Proof Binary")

        if let proofBinData = exportProof(proof: proof, publicInputs: pubInputs, format: .binary) {
            // Expected: 64 (A) + 128 (B) + 64 (C) + 2*32 (inputs) = 320
            let expectedSize = 64 + 128 + 64 + pubInputs.count * 32
            expectEqual(proofBinData.count, expectedSize, "Proof binary size")
        } else { expect(false, "exportProof .binary returned nil") }

        // --- importVerificationKey .solidity returns nil ---
        if let solidityData = exportVerificationKey(vk: vk, format: .solidity) {
            let result = importVerificationKey(data: solidityData, format: .solidity)
            expect(result == nil, "importVK .solidity returns nil (not reversible)")
        }

        // --- Calldata via exportProof .solidity ---
        if let calldataData = exportProof(proof: proof, publicInputs: pubInputs, format: .solidity) {
            let calldataStr = String(data: calldataData, encoding: .utf8) ?? ""
            expect(calldataStr.hasPrefix("0x43753b4d"), "exportProof .solidity gives calldata")
        } else { expect(false, "exportProof .solidity returned nil") }

    } catch {
        expect(false, "Groth16 Solidity tests error: \(error)")
    }
}
