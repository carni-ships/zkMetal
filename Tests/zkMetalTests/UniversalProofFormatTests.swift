import zkMetal
import Foundation

// MARK: - Test helpers

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func fpEqual(_ a: Fp, _ b: Fp) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func pointEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
    fpEqual(a.x, b.x) && fpEqual(a.y, b.y) && fpEqual(a.z, b.z)
}

// MARK: - Tests

func runUniversalProofFormatTests() {
    suite("UniversalProofFormat — Binary Round-Trip")

    // Test 1: Simple round-trip with raw data
    do {
        let original = UniversalProof(
            type: .groth16, version: 1,
            curveId: .bn254, fieldId: .bn254Fr,
            proofData: [1, 2, 3, 4, 5, 6, 7, 8],
            publicInputs: [[10, 20, 30], [40, 50, 60]],
            metadata: ["prover": "zkMetal", "version": "1.0"]
        )
        let serialized = original.serialize()
        let decoded = try UniversalProof.deserialize(serialized)
        expect(decoded.type == .groth16, "type round-trip")
        expect(decoded.version == 1, "version round-trip")
        expect(decoded.curveId == .bn254, "curveId round-trip")
        expect(decoded.fieldId == .bn254Fr, "fieldId round-trip")
        expect(decoded.proofData == [1, 2, 3, 4, 5, 6, 7, 8], "proofData round-trip")
        expect(decoded.publicInputs.count == 2, "publicInputs count")
        expect(decoded.publicInputs[0] == [10, 20, 30], "publicInputs[0] round-trip")
        expect(decoded.publicInputs[1] == [40, 50, 60], "publicInputs[1] round-trip")
        expect(decoded.metadata["prover"] == "zkMetal", "metadata prover")
        expect(decoded.metadata["version"] == "1.0", "metadata version")
    } catch {
        expect(false, "Binary round-trip failed: \(error)")
    }

    // Test 2: All proof types serialize/deserialize
    for pt in ProofSystemType.allCases {
        do {
            let proof = UniversalProof(
                type: pt, version: 1,
                curveId: .bls12_381, fieldId: .babybear,
                proofData: [0xDE, 0xAD]
            )
            let bytes = proof.serialize()
            let back = try UniversalProof.deserialize(bytes)
            expect(back.type == pt, "proof type \(pt) round-trip")
        } catch {
            expect(false, "proof type \(pt) round-trip failed: \(error)")
        }
    }

    // Test 3: Empty proof data and metadata
    do {
        let proof = UniversalProof(type: .stark, proofData: [])
        let bytes = proof.serialize()
        let back = try UniversalProof.deserialize(bytes)
        expect(back.proofData.isEmpty, "empty proofData round-trip")
        expect(back.publicInputs.isEmpty, "empty publicInputs round-trip")
        expect(back.metadata.isEmpty, "empty metadata round-trip")
    } catch {
        expect(false, "empty proof round-trip failed: \(error)")
    }

    // Test 4: Invalid magic bytes
    do {
        let badData: [UInt8] = [0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        _ = try UniversalProof.deserialize(badData)
        expect(false, "should reject invalid magic")
    } catch {
        expect(true, "rejected invalid magic")
    }

    // Test 5: Truncated data
    do {
        let shortData: [UInt8] = [0x5A, 0x4B, 0x50, 0x46]
        _ = try UniversalProof.deserialize(shortData)
        expect(false, "should reject truncated header")
    } catch {
        expect(true, "rejected truncated header")
    }

    // Test 6: Header magic bytes check
    do {
        let proof = UniversalProof(type: .kzg, proofData: [42])
        let bytes = proof.serialize()
        expect(bytes[0] == 0x5A && bytes[1] == 0x4B && bytes[2] == 0x50 && bytes[3] == 0x46,
               "ZKPF magic bytes present")
    }

    suite("UniversalProofFormat — JSON Round-Trip")

    // Test 7: JSON encode/decode
    do {
        let original = UniversalProof(
            type: .plonk, version: 1,
            curveId: .pallas, fieldId: .goldilocks,
            proofData: [0xCA, 0xFE, 0xBA, 0xBE],
            publicInputs: [[1, 2], [3, 4]],
            metadata: ["circuit": "test"]
        )
        let json = original.toJSON()
        let decoded = try UniversalProof.fromJSON(json)
        expect(decoded.type == .plonk, "JSON type round-trip")
        expect(decoded.curveId == .pallas, "JSON curveId round-trip")
        expect(decoded.fieldId == .goldilocks, "JSON fieldId round-trip")
        expect(decoded.proofData == [0xCA, 0xFE, 0xBA, 0xBE], "JSON proofData round-trip")
        expect(decoded.publicInputs.count == 2, "JSON publicInputs count")
        expect(decoded.metadata["circuit"] == "test", "JSON metadata round-trip")
    } catch {
        expect(false, "JSON round-trip failed: \(error)")
    }

    suite("UniversalProofFormat — Proof Hash")

    // Test 8: Same proof produces same hash
    do {
        let proof1 = UniversalProof(type: .groth16, curveId: .bn254, fieldId: .bn254Fr,
                                     proofData: [1, 2, 3])
        let proof2 = UniversalProof(type: .groth16, curveId: .bn254, fieldId: .bn254Fr,
                                     proofData: [1, 2, 3])
        let hash1 = proof1.proofHash()
        let hash2 = proof2.proofHash()
        expect(hash1 == hash2, "identical proofs produce same hash")
        expect(hash1.count == 32, "SHA-256 hash is 32 bytes")
    }

    // Test 9: Different proofs produce different hashes
    do {
        let proof1 = UniversalProof(type: .groth16, proofData: [1, 2, 3])
        let proof2 = UniversalProof(type: .groth16, proofData: [1, 2, 4])
        let hash1 = proof1.proofHash()
        let hash2 = proof2.proofHash()
        expect(hash1 != hash2, "different proofs produce different hashes")
    }

    suite("UniversalProofFormat — Groth16 Wrap/Unwrap")

    // Test 10: Groth16Proof round-trip through UniversalProof
    do {
        // Create a dummy Groth16 proof with known points
        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())
        let original = Groth16Proof(a: g1, b: g2, c: g1)
        let publicInputs = [frFromInt(42), frFromInt(7)]

        let universal = UniversalProof.fromGroth16(original, publicInputs: publicInputs,
                                                    metadata: ["test": "groth16"])
        expect(universal.type == .groth16, "wrapped type is groth16")
        expect(universal.curveId == .bn254, "wrapped curve is bn254")

        // Binary round-trip
        let bytes = universal.serialize()
        let decoded = try UniversalProof.deserialize(bytes)
        let recovered = try decoded.toGroth16()

        expect(pointEqual(recovered.a, original.a), "Groth16 a point round-trip")
        expect(pointEqual(recovered.c, original.c), "Groth16 c point round-trip")
        // G2 check: compare Fp2 components
        expect(fpEqual(recovered.b.x.c0, original.b.x.c0), "Groth16 b.x.c0 round-trip")
        expect(fpEqual(recovered.b.x.c1, original.b.x.c1), "Groth16 b.x.c1 round-trip")
        expect(fpEqual(recovered.b.y.c0, original.b.y.c0), "Groth16 b.y.c0 round-trip")
        expect(fpEqual(recovered.b.y.c1, original.b.y.c1), "Groth16 b.y.c1 round-trip")

        // Public inputs
        expect(decoded.publicInputs.count == 2, "Groth16 public inputs count")
    } catch {
        expect(false, "Groth16 wrap/unwrap failed: \(error)")
    }

    suite("UniversalProofFormat — PlonkProof Wrap/Unwrap")

    // Test 11: PlonkProof round-trip through UniversalProof
    do {
        let g1 = pointFromAffine(bn254G1Generator())
        let fr1 = frFromInt(123)
        let fr2 = frFromInt(456)
        let original = PlonkProof(
            aCommit: g1, bCommit: g1, cCommit: g1,
            zCommit: g1,
            tLoCommit: g1, tMidCommit: g1, tHiCommit: g1,
            tExtraCommits: [],
            aEval: fr1, bEval: fr1, cEval: fr1,
            sigma1Eval: fr2, sigma2Eval: fr2,
            zOmegaEval: fr1,
            openingProof: g1,
            shiftedOpeningProof: g1,
            publicInputs: [fr1, fr2]
        )

        let universal = UniversalProof.fromPlonk(original, metadata: ["test": "plonk"])
        let bytes = universal.serialize()
        let decoded = try UniversalProof.deserialize(bytes)
        let recovered = try decoded.toPlonk()

        expect(pointEqual(recovered.aCommit, original.aCommit), "Plonk aCommit round-trip")
        expect(frEqual(recovered.aEval, original.aEval), "Plonk aEval round-trip")
        expect(frEqual(recovered.sigma1Eval, original.sigma1Eval), "Plonk sigma1Eval round-trip")
        expect(recovered.publicInputs.count == 2, "Plonk public inputs count")
        expect(frEqual(recovered.publicInputs[0], fr1), "Plonk public input 0")
        expect(frEqual(recovered.publicInputs[1], fr2), "Plonk public input 1")
    } catch {
        expect(false, "Plonk wrap/unwrap failed: \(error)")
    }

    suite("UniversalProofFormat — ProofRegistry")

    // Test 12: Registry lookup
    do {
        let registry = ProofRegistry.shared
        let types = registry.registeredTypes
        expect(types.contains(.groth16), "registry has groth16 codec")
        expect(types.contains(.plonk), "registry has plonk codec")
        expect(types.contains(.kzg), "registry has kzg codec")
        expect(types.contains(.ipa), "registry has ipa codec")
    }

    // Test 13: Registry encode/decode round-trip for Groth16
    do {
        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())
        let original = Groth16Proof(a: g1, b: g2, c: g1)

        let registry = ProofRegistry.shared
        let universal = try registry.encode(original, type: .groth16)
        expect(universal.type == .groth16, "registry encoded type")
        expect(universal.curveId == .bn254, "registry encoded curve")

        let decoded = try registry.decode(universal)
        guard let recovered = decoded as? Groth16Proof else {
            expect(false, "registry decoded wrong type")
            return
        }
        expect(pointEqual(recovered.a, original.a), "registry Groth16 a round-trip")
        expect(pointEqual(recovered.c, original.c), "registry Groth16 c round-trip")
    } catch {
        expect(false, "ProofRegistry round-trip failed: \(error)")
    }

    // Test 14: Registry rejects unregistered type
    do {
        let registry = ProofRegistry()  // fresh empty registry
        _ = try registry.encode("dummy", type: .nova)
        expect(false, "should reject unregistered type")
    } catch {
        expect(true, "rejected unregistered proof type")
    }

    // Test 15: Custom codec registration
    do {
        let registry = ProofRegistry()
        let customCodec = DummyCodec()
        try registry.register(customCodec)
        let types = registry.registeredTypes
        expect(types.contains(.nova), "custom codec registered")

        let universal = try registry.encode([UInt8(99)], type: .nova)
        let back = try registry.decode(universal) as! [UInt8]
        expect(back == [99], "custom codec round-trip")
    } catch {
        expect(false, "custom codec test failed: \(error)")
    }

    // Test 16: Large proof data
    do {
        let largeData = [UInt8](repeating: 0xAB, count: 100_000)
        let proof = UniversalProof(type: .stark, proofData: largeData)
        let bytes = proof.serialize()
        let decoded = try UniversalProof.deserialize(bytes)
        expect(decoded.proofData.count == 100_000, "large proof data preserved")
        expect(decoded.proofData[0] == 0xAB && decoded.proofData[99_999] == 0xAB,
               "large proof data content correct")
    } catch {
        expect(false, "large proof data round-trip failed: \(error)")
    }

    // Test 17: All curve/field IDs round-trip
    for curve in CurveId.allCases {
        for field in FieldId.allCases {
            do {
                let proof = UniversalProof(type: .groth16, curveId: curve, fieldId: field,
                                           proofData: [0xFF])
                let bytes = proof.serialize()
                let back = try UniversalProof.deserialize(bytes)
                expect(back.curveId == curve && back.fieldId == field,
                       "curve \(curve) field \(field) round-trip")
            } catch {
                expect(false, "curve \(curve) field \(field) failed: \(error)")
            }
        }
    }
}

// Dummy codec for testing custom registration
private struct DummyCodec: ProofCodec {
    let proofType: ProofSystemType = .nova
    func encode(_ proof: Any) throws -> [UInt8] {
        guard let bytes = proof as? [UInt8] else {
            throw ProofRegistryError.typeMismatch(expected: "[UInt8]",
                                                   got: String(describing: Swift.type(of: proof)))
        }
        return bytes
    }
    func decode(_ data: [UInt8]) throws -> Any { data }
}
