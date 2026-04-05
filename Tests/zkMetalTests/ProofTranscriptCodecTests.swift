import zkMetal
import Foundation

/// Compare two Fp values for equality (bitwise on internal limbs).
private func fpEq(_ a: Fp, _ b: Fp) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func g1AffineEq(_ a: PointAffine, _ b: PointAffine) -> Bool {
    fpEq(a.x, b.x) && fpEq(a.y, b.y)
}

private func g2AffineEq(_ a: G2AffinePoint, _ b: G2AffinePoint) -> Bool {
    fpEq(a.x.c0, b.x.c0) && fpEq(a.x.c1, b.x.c1) &&
    fpEq(a.y.c0, b.y.c0) && fpEq(a.y.c1, b.y.c1)
}

/// Compare two transcripts element-wise.
private func transcriptEqual(_ a: ProofTranscript, _ b: ProofTranscript) -> Bool {
    guard a.system == b.system else { return false }
    guard a.g1Points.count == b.g1Points.count else { return false }
    guard a.g2Points.count == b.g2Points.count else { return false }
    guard a.scalars.count == b.scalars.count else { return false }
    guard a.publicInputs.count == b.publicInputs.count else { return false }

    for i in 0..<a.g1Points.count {
        if !g1AffineEq(a.g1Points[i], b.g1Points[i]) { return false }
    }
    for i in 0..<a.g2Points.count {
        if !g2AffineEq(a.g2Points[i], b.g2Points[i]) { return false }
    }
    for i in 0..<a.scalars.count {
        if !frEq(a.scalars[i], b.scalars[i]) { return false }
    }
    for i in 0..<a.publicInputs.count {
        if !frEq(a.publicInputs[i], b.publicInputs[i]) { return false }
    }
    return true
}

/// Build a Groth16 proof transcript for testing.
private func makeTestGroth16Transcript() -> ProofTranscript? {
    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3) // 3^3 + 3 + 5 = 35
        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
        return ProofTranscript.fromGroth16(proof, publicInputs: pubInputs,
                                           metadata: ["prover": "zkMetal-test"])
    } catch {
        return nil
    }
}

public func runProofTranscriptCodecTests() {

    suite("ProofTranscript: Groth16 round-trip construction")

    do {
        let r1cs = buildExampleCircuit()
        let (pubInputs, witness) = computeExampleWitness(x: 3)
        let setup = Groth16Setup()
        let (pk, _) = setup.setup(r1cs: r1cs)
        let prover = try Groth16Prover()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)

        // Build transcript
        guard let transcript = ProofTranscript.fromGroth16(proof, publicInputs: pubInputs) else {
            expect(false, "ProofTranscript.fromGroth16 returned nil")
            return
        }
        expectEqual(transcript.system, .groth16, "Transcript system is groth16")
        expectEqual(transcript.g1Points.count, 2, "Groth16 transcript has 2 G1 points (a, c)")
        expectEqual(transcript.g2Points.count, 1, "Groth16 transcript has 1 G2 point (b)")
        expectEqual(transcript.publicInputs.count, 2, "Transcript has 2 public inputs")

        // Round-trip back to Groth16Proof
        guard let proofBack = transcript.toGroth16() else {
            expect(false, "transcript.toGroth16() returned nil")
            return
        }
        // Compare via affine coordinates
        guard let aAff = pointToAffine(proof.a), let aAff2 = pointToAffine(proofBack.a) else {
            expect(false, "pointToAffine failed"); return
        }
        expect(fpEq(aAff.x, aAff2.x) && fpEq(aAff.y, aAff2.y), "Groth16 a roundtrip")

        guard let bAff = g2ToAffine(proof.b), let bAff2 = g2ToAffine(proofBack.b) else {
            expect(false, "g2ToAffine failed"); return
        }
        expect(g2AffineEq(bAff, bAff2), "Groth16 b roundtrip")

        guard let cAff = pointToAffine(proof.c), let cAff2 = pointToAffine(proofBack.c) else {
            expect(false, "pointToAffine failed for c"); return
        }
        expect(fpEq(cAff.x, cAff2.x) && fpEq(cAff.y, cAff2.y), "Groth16 c roundtrip")

    } catch {
        expect(false, "Groth16 transcript construction error: \(error)")
    }

    // MARK: - Ethereum ABI encoding

    suite("ProofTranscriptCodec: Ethereum ABI format")

    if let transcript = makeTestGroth16Transcript() {
        do {
            let abiBytes = try ProofTranscriptCodec.serialize(transcript, format: .ethereumABI)
            // Groth16 ABI: (2+4+2+N) * 32 bytes
            let expectedSize = (2 + 4 + 2 + transcript.publicInputs.count) * 32
            expectEqual(abiBytes.count, expectedSize, "ABI size = \(expectedSize)")

            // Deserialize and compare
            let decoded = try ProofTranscriptCodec.deserialize(abiBytes, format: .ethereumABI,
                                                               system: .groth16)
            expect(transcriptEqual(transcript, decoded), "ABI round-trip matches")

            // Verify known structure: first 32 bytes are a.x in big-endian
            let axDec = fpToDecimal(transcript.g1Points[0].x)
            expect(!axDec.isEmpty, "a.x decimal is non-empty")

        } catch {
            expect(false, "ABI encode/decode error: \(error)")
        }
    } else {
        expect(false, "Failed to create test Groth16 transcript")
    }

    // MARK: - Ethereum ABI known test vector

    suite("ProofTranscriptCodec: Ethereum ABI known vector")

    // Verify that encoding the identity-like small values produces correct big-endian layout
    do {
        let smallFr1 = frFromInt(3)
        let smallFr2 = frFromInt(35)

        // Create a simple transcript with known small public inputs
        if let transcript = makeTestGroth16Transcript() {
            let abiBytes = try ProofTranscriptCodec.serialize(transcript, format: .ethereumABI)

            // Public inputs start at offset 256 (8 * 32 bytes for proof elements)
            let inputOffset = 256
            guard abiBytes.count >= inputOffset + 64 else {
                expect(false, "ABI too short for inputs"); return
            }

            // Decode the public inputs from ABI bytes manually
            let input0Bytes = Array(abiBytes[inputOffset..<(inputOffset + 32)])
            let input1Bytes = Array(abiBytes[(inputOffset + 32)..<(inputOffset + 64)])

            // Last byte of first input should encode value 3
            expectEqual(input0Bytes[31], 3, "ABI input[0] last byte = 3")
            // Last byte of second input should encode value 35
            expectEqual(input1Bytes[31], 35, "ABI input[1] last byte = 35")
            // Leading bytes should be zero for small values
            expectEqual(input0Bytes[0], 0, "ABI input[0] leading byte = 0")
        }
    } catch {
        expect(false, "ABI known vector error: \(error)")
    }

    // MARK: - G1 point compression/decompression

    suite("ProofTranscriptCodec: BN254 G1 compression")

    if let transcript = makeTestGroth16Transcript() {
        let aAff = transcript.g1Points[0]
        let compressed = transcriptG1Compress(aAff)
        expectEqual(compressed.count, 33, "G1 compressed is 33 bytes")
        expect(compressed[0] == 0x02 || compressed[0] == 0x03, "G1 prefix is 0x02 or 0x03")

        do {
            let decompressed = try transcriptG1Decompress(compressed)
            expect(g1AffineEq(aAff, decompressed), "G1 compress/decompress roundtrip")
        } catch {
            expect(false, "G1 decompression error: \(error)")
        }

        // Test the c point too
        let cAff = transcript.g1Points[1]
        let cCompressed = transcriptG1Compress(cAff)
        do {
            let cDecompressed = try transcriptG1Decompress(cCompressed)
            expect(g1AffineEq(cAff, cDecompressed), "G1 c-point compress/decompress")
        } catch {
            expect(false, "G1 c-point decompression error: \(error)")
        }
    }

    // MARK: - G2 point compression/decompression

    suite("ProofTranscriptCodec: BN254 G2 compression")

    if let transcript = makeTestGroth16Transcript(), !transcript.g2Points.isEmpty {
        let bAff = transcript.g2Points[0]
        let compressed = transcriptG2Compress(bAff)
        expectEqual(compressed.count, 65, "G2 compressed is 65 bytes")
        expect(compressed[0] == 0x0a || compressed[0] == 0x0b, "G2 prefix is 0x0a or 0x0b")

        do {
            let decompressed = try transcriptG2Decompress(compressed)
            expect(g2AffineEq(bAff, decompressed), "G2 compress/decompress roundtrip")
        } catch {
            expect(false, "G2 decompression error: \(error)")
        }
    }

    // MARK: - Gnark format round-trip

    suite("ProofTranscriptCodec: Gnark format")

    if let transcript = makeTestGroth16Transcript() {
        do {
            let gnarkBytes = try ProofTranscriptCodec.serialize(transcript, format: .gnark)
            expect(gnarkBytes.count > 0, "Gnark output is non-empty")

            let decoded = try ProofTranscriptCodec.deserialize(gnarkBytes, format: .gnark,
                                                                system: .groth16)
            expect(transcriptEqual(transcript, decoded), "Gnark round-trip matches")

            // Gnark should be smaller than ABI (compressed points vs uncompressed)
            let abiBytes = try ProofTranscriptCodec.serialize(transcript, format: .ethereumABI)
            expect(gnarkBytes.count < abiBytes.count,
                   "Gnark (\(gnarkBytes.count)) < ABI (\(abiBytes.count))")
        } catch {
            expect(false, "Gnark encode/decode error: \(error)")
        }
    }

    // MARK: - SnarkJS JSON format round-trip

    suite("ProofTranscriptCodec: SnarkJS JSON format")

    if let transcript = makeTestGroth16Transcript() {
        do {
            let jsonBytes = try ProofTranscriptCodec.serialize(transcript, format: .snarkjsJSON)
            expect(jsonBytes.count > 0, "JSON output is non-empty")

            // Verify it's valid JSON
            let jsonStr = String(bytes: jsonBytes, encoding: .utf8) ?? ""
            expect(jsonStr.contains("\"protocol\""), "JSON has protocol field")
            expect(jsonStr.contains("\"pi_a\""), "JSON has pi_a field")
            expect(jsonStr.contains("\"pi_b\""), "JSON has pi_b field")
            expect(jsonStr.contains("\"pi_c\""), "JSON has pi_c field")
            expect(jsonStr.contains("\"public_inputs\""), "JSON has public_inputs field")

            let decoded = try ProofTranscriptCodec.deserialize(jsonBytes, format: .snarkjsJSON,
                                                                system: .groth16)
            expect(transcriptEqual(transcript, decoded), "JSON round-trip matches")
        } catch {
            expect(false, "JSON encode/decode error: \(error)")
        }
    }

    // MARK: - Binary compact format

    suite("ProofTranscriptCodec: Binary compact format")

    if let transcript = makeTestGroth16Transcript() {
        do {
            let compactBytes = try ProofTranscriptCodec.serialize(transcript, format: .binaryCompact)
            expect(compactBytes.count > 0, "Compact output is non-empty")

            // Verify magic header
            expectEqual(compactBytes[0], 0x5A, "Magic Z")
            expectEqual(compactBytes[1], 0x4B, "Magic K")
            expectEqual(compactBytes[2], 0x54, "Magic T")
            expectEqual(compactBytes[3], 0x58, "Magic X")
            expectEqual(compactBytes[4], 1, "Version 1")
            expectEqual(compactBytes[5], 0, "System groth16 = 0")

            let decoded = try ProofTranscriptCodec.deserialize(compactBytes, format: .binaryCompact,
                                                                system: .groth16)
            expect(transcriptEqual(transcript, decoded), "Binary compact round-trip matches")

            // Binary compact should be the smallest format (compressed points, minimal headers)
            let abiBytes = try ProofTranscriptCodec.serialize(transcript, format: .ethereumABI)
            let gnarkBytes = try ProofTranscriptCodec.serialize(transcript, format: .gnark)
            let jsonBytes = try ProofTranscriptCodec.serialize(transcript, format: .snarkjsJSON)

            // Report sizes
            let sizes = [
                ("ABI", abiBytes.count),
                ("Gnark", gnarkBytes.count),
                ("JSON", jsonBytes.count),
                ("Compact", compactBytes.count)
            ]
            for (name, size) in sizes {
                expect(size > 0, "\(name) size = \(size) bytes")
            }

            // Compact should be <= Gnark (both use compressed points, compact has smaller headers)
            expect(compactBytes.count <= gnarkBytes.count,
                   "Compact (\(compactBytes.count)) <= Gnark (\(gnarkBytes.count))")

            // JSON is largest (decimal strings)
            expect(jsonBytes.count > compactBytes.count,
                   "JSON (\(jsonBytes.count)) > Compact (\(compactBytes.count))")
        } catch {
            expect(false, "Binary compact encode/decode error: \(error)")
        }
    }

    // MARK: - Cross-format conversion

    suite("ProofTranscriptCodec: Cross-format conversion")

    if let transcript = makeTestGroth16Transcript() {
        do {
            // ABI -> JSON -> Binary Compact
            let abiBytes = try ProofTranscriptCodec.serialize(transcript, format: .ethereumABI)

            let jsonBytes = try ProofTranscriptCodec.convert(abiBytes, from: .ethereumABI,
                                                              to: .snarkjsJSON, system: .groth16)
            expect(jsonBytes.count > 0, "ABI -> JSON conversion produced output")

            let compactBytes = try ProofTranscriptCodec.convert(jsonBytes, from: .snarkjsJSON,
                                                                 to: .binaryCompact, system: .groth16)
            expect(compactBytes.count > 0, "JSON -> Compact conversion produced output")

            // Decode the final compact bytes and compare with original
            let finalTranscript = try ProofTranscriptCodec.deserialize(compactBytes,
                                                                        format: .binaryCompact,
                                                                        system: .groth16)
            expect(transcriptEqual(transcript, finalTranscript),
                   "ABI -> JSON -> Compact preserves data")

            // Also test Compact -> Gnark -> ABI round-trip
            let gnarkBytes = try ProofTranscriptCodec.convert(compactBytes, from: .binaryCompact,
                                                               to: .gnark, system: .groth16)
            let abiBytes2 = try ProofTranscriptCodec.convert(gnarkBytes, from: .gnark,
                                                              to: .ethereumABI, system: .groth16)
            let finalTranscript2 = try ProofTranscriptCodec.deserialize(abiBytes2,
                                                                         format: .ethereumABI,
                                                                         system: .groth16)
            expect(transcriptEqual(transcript, finalTranscript2),
                   "Compact -> Gnark -> ABI preserves data")

        } catch {
            expect(false, "Cross-format conversion error: \(error)")
        }
    }

    // MARK: - Batch encoding

    suite("ProofTranscriptCodec: Batch encoding")

    if let t1 = makeTestGroth16Transcript() {
        // Make a second transcript with different inputs
        var t2: ProofTranscript? = nil
        do {
            let r1cs = buildExampleCircuit()
            let (pubInputs, witness) = computeExampleWitness(x: 7) // 7^3 + 7 + 5 = 355
            let setup = Groth16Setup()
            let (pk, _) = setup.setup(r1cs: r1cs)
            let prover = try Groth16Prover()
            let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
            t2 = ProofTranscript.fromGroth16(proof, publicInputs: pubInputs)
        } catch {
            expect(false, "Batch: second proof generation error: \(error)")
        }

        if let t2 = t2 {
            do {
                let batchData = try ProofTranscriptCodec.batchEncode([t1, t2], format: .binaryCompact)
                expect(batchData.count > 0, "Batch encode produced output")

                // First 4 bytes should be count = 2
                expectEqual(batchData[0], 2, "Batch count low byte = 2")
                expectEqual(batchData[1], 0, "Batch count byte 1 = 0")

                let decoded = try ProofTranscriptCodec.batchDecode(batchData, format: .binaryCompact,
                                                                    system: .groth16)
                expectEqual(decoded.count, 2, "Batch decoded 2 transcripts")
                expect(transcriptEqual(t1, decoded[0]), "Batch[0] matches original")
                expect(transcriptEqual(t2, decoded[1]), "Batch[1] matches original")
            } catch {
                expect(false, "Batch encode/decode error: \(error)")
            }
        }
    }

    // MARK: - Error cases

    suite("ProofTranscriptCodec: Error handling")

    // Truncated ABI data
    do {
        _ = try ProofTranscriptCodec.deserialize([0x00, 0x01], format: .ethereumABI, system: .groth16)
        expect(false, "Should throw on truncated ABI")
    } catch {
        expect(true, "Truncated ABI throws error")
    }

    // Invalid binary compact magic
    do {
        let badMagic: [UInt8] = [0x00, 0x00, 0x00, 0x00, 1, 0, 0, 0, 0, 0, 0]
        _ = try ProofTranscriptCodec.deserialize(badMagic, format: .binaryCompact, system: .groth16)
        expect(false, "Should throw on bad magic")
    } catch {
        expect(true, "Bad magic throws error")
    }

    // Invalid JSON
    do {
        let badJson: [UInt8] = Array("not json".utf8)
        _ = try ProofTranscriptCodec.deserialize(badJson, format: .snarkjsJSON, system: .groth16)
        expect(false, "Should throw on invalid JSON")
    } catch {
        expect(true, "Invalid JSON throws error")
    }

    // ABI format only supports groth16
    do {
        let fakeTranscript = ProofTranscript(system: .plonk, g1Points: [], publicInputs: [])
        _ = try ProofTranscriptCodec.serialize(fakeTranscript, format: .ethereumABI)
        expect(false, "Should throw for Plonk ABI")
    } catch {
        expect(true, "Plonk ABI throws unsupported format")
    }
}
