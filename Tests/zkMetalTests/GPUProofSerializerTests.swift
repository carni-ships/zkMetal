import zkMetal
import Foundation

// MARK: - Helpers

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func pointsEqualAffine(_ a: PointProjective, _ b: PointProjective) -> Bool {
    if pointIsIdentity(a) && pointIsIdentity(b) { return true }
    guard let aa = pointToAffine(a), let ba = pointToAffine(b) else {
        return false
    }
    return aa.x.to64() == ba.x.to64() && aa.y.to64() == ba.y.to64()
}

private func makeTestPoint(_ scalar: UInt64) -> PointProjective {
    let g1Aff = bn254G1Generator()
    let g1 = pointFromAffine(g1Aff)
    return cPointScalarMul(g1, frFromInt(scalar))
}

private func makeTestHash(_ seed: UInt8) -> [UInt8] {
    var h = [UInt8](repeating: 0, count: 32)
    h[0] = seed
    h[31] = seed &+ 1
    h[15] = seed &* 3
    return h
}

// MARK: - Tests

public func runGPUProofSerializerTests() {
    suite("GPU Proof Serializer Engine")

    // ---- Test 1: Groth16 roundtrip (default flags) ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = makeTestPoint(3)
        let b = makeTestPoint(7)
        let c = makeTestPoint(11)
        let pubInputs = [frFromInt(42), frFromInt(99), frFromInt(7)]

        let serialized = engine.serializeGroth16(a: a, b: b, c: c, publicInputs: pubInputs)

        expect(serialized.proofType == .groth16, "Groth16 roundtrip: proof type is groth16")
        expect(serialized.curveId == .bn254, "Groth16 roundtrip: curve is bn254")
        expect(serialized.data.count > 0, "Groth16 roundtrip: non-empty data")

        let (da, db, dc, dpub) = try engine.deserializeGroth16(serialized)
        expect(pointsEqualAffine(da, a), "Groth16 roundtrip: point A matches")
        expect(pointsEqualAffine(db, b), "Groth16 roundtrip: point B matches")
        expect(pointsEqualAffine(dc, c), "Groth16 roundtrip: point C matches")
        expect(dpub.count == 3, "Groth16 roundtrip: 3 public inputs")
        for i in 0..<3 {
            expect(frEqual(dpub[i], pubInputs[i]), "Groth16 roundtrip: pubInput[\(i)] matches")
        }
    } catch {
        expect(false, "Groth16 roundtrip threw: \(error)")
    }

    // ---- Test 2: Groth16 with identity point ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = pointIdentity()
        let b = makeTestPoint(5)
        let c = pointIdentity()
        let pubInputs = [Fr.zero, frFromInt(1)]

        let serialized = engine.serializeGroth16(a: a, b: b, c: c, publicInputs: pubInputs)
        let (da, db, dc, dpub) = try engine.deserializeGroth16(serialized)

        expect(pointIsIdentity(da), "Groth16 identity: point A is identity")
        expect(pointsEqualAffine(db, b), "Groth16 identity: point B matches")
        expect(pointIsIdentity(dc), "Groth16 identity: point C is identity")
        expect(dpub.count == 2, "Groth16 identity: 2 public inputs")
        expect(frEqual(dpub[0], Fr.zero), "Groth16 identity: pubInput[0] is zero")
    } catch {
        expect(false, "Groth16 identity threw: \(error)")
    }

    // ---- Test 3: Groth16 no compression ----
    do {
        let engine = GPUProofSerializerEngine(flags: .none)
        let a = makeTestPoint(13)
        let b = makeTestPoint(17)
        let c = makeTestPoint(19)
        let pubInputs = [frFromInt(100)]

        let serialized = engine.serializeGroth16(a: a, b: b, c: c, publicInputs: pubInputs)

        // No compression => raw 96-byte points
        expect(serialized.data.count > serialized.uncompressedSize / 2,
               "Groth16 no-compress: raw size is substantial")

        let (da, db, dc, dpub) = try engine.deserializeGroth16(serialized)
        expect(pointsEqualAffine(da, a), "Groth16 no-compress: point A matches")
        expect(pointsEqualAffine(db, b), "Groth16 no-compress: point B matches")
        expect(pointsEqualAffine(dc, c), "Groth16 no-compress: point C matches")
        expect(dpub.count == 1, "Groth16 no-compress: 1 public input")
        expect(frEqual(dpub[0], frFromInt(100)), "Groth16 no-compress: pubInput matches")
    } catch {
        expect(false, "Groth16 no-compress threw: \(error)")
    }

    // ---- Test 4: Compression ratio (compressed < uncompressed) ----
    do {
        let engine = GPUProofSerializerEngine(flags: .default)
        let a = makeTestPoint(3)
        let b = makeTestPoint(7)
        let c = makeTestPoint(11)

        let serialized = engine.serializeGroth16(a: a, b: b, c: c, publicInputs: [])
        expect(serialized.compressionRatio < 1.0,
               "Compression ratio: compressed < uncompressed for points")

        let engineRaw = GPUProofSerializerEngine(flags: .none)
        let serRaw = engineRaw.serializeGroth16(a: a, b: b, c: c, publicInputs: [])
        expect(serialized.data.count < serRaw.data.count,
               "Compression ratio: compressed data smaller than raw")
    }

    // ---- Test 5: Plonk roundtrip ----
    do {
        let engine = GPUProofSerializerEngine()
        let wires = [makeTestPoint(2), makeTestPoint(3), makeTestPoint(5)]
        let gp = makeTestPoint(7)
        let quotients = [makeTestPoint(11), makeTestPoint(13), makeTestPoint(17)]
        let opening = makeTestPoint(19)
        let shiftedOpening = makeTestPoint(23)
        let evals = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]

        let serialized = engine.serializePlonk(
            wireCommitments: wires,
            grandProductCommitment: gp,
            quotientCommitments: quotients,
            openingProof: opening,
            shiftedOpeningProof: shiftedOpening,
            evaluations: evals
        )

        expect(serialized.proofType == .plonk, "Plonk roundtrip: proof type is plonk")

        let result = try engine.deserializePlonk(serialized)
        expect(result.wireCommitments.count == 3, "Plonk roundtrip: 3 wire commitments")
        for i in 0..<3 {
            expect(pointsEqualAffine(result.wireCommitments[i], wires[i]),
                   "Plonk roundtrip: wire[\(i)] matches")
        }
        expect(pointsEqualAffine(result.grandProductCommitment, gp),
               "Plonk roundtrip: grand product matches")
        expect(result.quotientCommitments.count == 3, "Plonk roundtrip: 3 quotient commitments")
        for i in 0..<3 {
            expect(pointsEqualAffine(result.quotientCommitments[i], quotients[i]),
                   "Plonk roundtrip: quotient[\(i)] matches")
        }
        expect(pointsEqualAffine(result.openingProof, opening),
               "Plonk roundtrip: opening proof matches")
        expect(pointsEqualAffine(result.shiftedOpeningProof, shiftedOpening),
               "Plonk roundtrip: shifted opening proof matches")
        expect(result.evaluations.count == 4, "Plonk roundtrip: 4 evaluations")
        for i in 0..<4 {
            expect(frEqual(result.evaluations[i], evals[i]),
                   "Plonk roundtrip: eval[\(i)] matches")
        }
    } catch {
        expect(false, "Plonk roundtrip threw: \(error)")
    }

    // ---- Test 6: STARK roundtrip ----
    do {
        let engine = GPUProofSerializerEngine()
        let traceCommits = [makeTestHash(0x01), makeTestHash(0x02), makeTestHash(0x03)]
        let constraintEvals = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let friLayers: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
            [frFromInt(5), frFromInt(6)]
        ]
        let queryPaths = [
            [makeTestHash(0xA0), makeTestHash(0xA1)],
            [makeTestHash(0xB0), makeTestHash(0xB1)]
        ]

        let serialized = engine.serializeSTARK(
            traceCommitments: traceCommits,
            constraintEvaluations: constraintEvals,
            friLayers: friLayers,
            queryPaths: queryPaths
        )

        expect(serialized.proofType == .stark, "STARK roundtrip: proof type is stark")
        expect(serialized.curveId == .none, "STARK roundtrip: curve is none")

        let result = try engine.deserializeSTARK(serialized)
        expect(result.traceCommitments.count == 3, "STARK roundtrip: 3 trace commitments")
        for i in 0..<3 {
            expect(result.traceCommitments[i] == traceCommits[i],
                   "STARK roundtrip: trace commit[\(i)] matches")
        }
        expect(result.constraintEvaluations.count == 3, "STARK roundtrip: 3 constraint evals")
        for i in 0..<3 {
            expect(frEqual(result.constraintEvaluations[i], constraintEvals[i]),
                   "STARK roundtrip: constraint eval[\(i)] matches")
        }
        expect(result.friLayers.count == 2, "STARK roundtrip: 2 FRI layers")
        expect(result.friLayers[0].count == 4, "STARK roundtrip: FRI layer 0 has 4 elements")
        expect(result.friLayers[1].count == 2, "STARK roundtrip: FRI layer 1 has 2 elements")
        for i in 0..<4 {
            expect(frEqual(result.friLayers[0][i], friLayers[0][i]),
                   "STARK roundtrip: FRI layer 0[\(i)] matches")
        }
        expect(result.queryPaths.count == 2, "STARK roundtrip: 2 query paths")
        expect(result.queryPaths[0].count == 2, "STARK roundtrip: path 0 has 2 hashes")
        expect(result.queryPaths[0][0] == queryPaths[0][0], "STARK roundtrip: path[0][0] matches")
    } catch {
        expect(false, "STARK roundtrip threw: \(error)")
    }

    // ---- Test 7: FRI roundtrip ----
    do {
        let engine = GPUProofSerializerEngine()
        let layers: [[Fr]] = [
            [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)],
            [frFromInt(50), frFromInt(60)],
            [frFromInt(70)]
        ]
        let queryPaths = [
            [makeTestHash(0x10), makeTestHash(0x11), makeTestHash(0x12)],
            [makeTestHash(0x20), makeTestHash(0x21), makeTestHash(0x22)]
        ]
        let finalPoly = [frFromInt(99), frFromInt(88)]

        let serialized = engine.serializeFRI(layers: layers, queryPaths: queryPaths,
                                             finalPoly: finalPoly)

        expect(serialized.proofType == .fri, "FRI roundtrip: proof type is fri")

        let result = try engine.deserializeFRI(serialized)
        expect(result.layers.count == 3, "FRI roundtrip: 3 layers")
        expect(result.layers[0].count == 4, "FRI roundtrip: layer 0 has 4 elements")
        expect(result.layers[1].count == 2, "FRI roundtrip: layer 1 has 2 elements")
        expect(result.layers[2].count == 1, "FRI roundtrip: layer 2 has 1 element")
        for i in 0..<4 {
            expect(frEqual(result.layers[0][i], layers[0][i]),
                   "FRI roundtrip: layer 0[\(i)] matches")
        }
        expect(result.queryPaths.count == 2, "FRI roundtrip: 2 query paths")
        expect(result.queryPaths[0].count == 3, "FRI roundtrip: path 0 has 3 hashes")
        expect(result.finalPoly.count == 2, "FRI roundtrip: final poly has 2 coefficients")
        expect(frEqual(result.finalPoly[0], frFromInt(99)), "FRI roundtrip: finalPoly[0] matches")
        expect(frEqual(result.finalPoly[1], frFromInt(88)), "FRI roundtrip: finalPoly[1] matches")
    } catch {
        expect(false, "FRI roundtrip threw: \(error)")
    }

    // ---- Test 8: Run-length encoding roundtrip ----
    do {
        let engine = GPUProofSerializerEngine(flags: [.runLengthEncode])
        // Create data with many repeated values (ideal for RLE)
        var pubInputs = [Fr]()
        for _ in 0..<50 { pubInputs.append(Fr.zero) }
        for _ in 0..<20 { pubInputs.append(frFromInt(42)) }
        for _ in 0..<30 { pubInputs.append(Fr.one) }

        let a = makeTestPoint(3)
        let serialized = engine.serializeGroth16(a: a, b: a, c: a, publicInputs: pubInputs)

        let (_, _, _, dpub) = try engine.deserializeGroth16(serialized)
        expect(dpub.count == 100, "RLE roundtrip: 100 public inputs")
        for i in 0..<50 {
            expect(frEqual(dpub[i], Fr.zero), "RLE roundtrip: first 50 are zero")
        }
        for i in 50..<70 {
            expect(frEqual(dpub[i], frFromInt(42)), "RLE roundtrip: next 20 are 42")
        }
        for i in 70..<100 {
            expect(frEqual(dpub[i], Fr.one), "RLE roundtrip: last 30 are one")
        }

        // RLE should compress well for repeated data
        let engineNoRLE = GPUProofSerializerEngine(flags: .none)
        let serNoRLE = engineNoRLE.serializeGroth16(a: a, b: a, c: a, publicInputs: pubInputs)
        expect(serialized.data.count < serNoRLE.data.count,
               "RLE roundtrip: RLE version is smaller than raw for repeated data")
    } catch {
        expect(false, "RLE roundtrip threw: \(error)")
    }

    // ---- Test 9: Proof type detection ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = makeTestPoint(3)

        let groth16 = engine.serializeGroth16(a: a, b: a, c: a, publicInputs: [])
        let plonk = engine.serializePlonk(wireCommitments: [a], grandProductCommitment: a,
                                           quotientCommitments: [a], openingProof: a,
                                           shiftedOpeningProof: a, evaluations: [frFromInt(1)])
        let stark = engine.serializeSTARK(traceCommitments: [makeTestHash(1)],
                                           constraintEvaluations: [frFromInt(1)],
                                           friLayers: [[frFromInt(1)]],
                                           queryPaths: [[makeTestHash(2)]])
        let fri = engine.serializeFRI(layers: [[frFromInt(1)]], queryPaths: [[makeTestHash(1)]],
                                       finalPoly: [frFromInt(1)])

        expect(GPUProofSerializerEngine.detectProofType(groth16.data) == .groth16,
               "Detect proof type: groth16")
        expect(GPUProofSerializerEngine.detectProofType(plonk.data) == .plonk,
               "Detect proof type: plonk")
        expect(GPUProofSerializerEngine.detectProofType(stark.data) == .stark,
               "Detect proof type: stark")
        expect(GPUProofSerializerEngine.detectProofType(fri.data) == .fri,
               "Detect proof type: fri")
        expect(GPUProofSerializerEngine.detectProofType([]) == nil,
               "Detect proof type: empty data returns nil")
        expect(GPUProofSerializerEngine.detectProofType([0x00, 0x00]) == nil,
               "Detect proof type: invalid magic returns nil")
    }

    // ---- Test 10: Curve ID detection ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = makeTestPoint(3)

        let groth16 = engine.serializeGroth16(a: a, b: a, c: a, publicInputs: [])
        let stark = engine.serializeSTARK(traceCommitments: [makeTestHash(1)],
                                           constraintEvaluations: [frFromInt(1)],
                                           friLayers: [[frFromInt(1)]],
                                           queryPaths: [[makeTestHash(2)]])

        expect(GPUProofSerializerEngine.detectCurveId(groth16.data) == .bn254,
               "Detect curve: bn254 for Groth16")
        expect(GPUProofSerializerEngine.detectCurveId(stark.data) == .none,
               "Detect curve: none for STARK")
    }

    // ---- Test 11: Format version compatibility ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = makeTestPoint(3)
        let serialized = engine.serializeGroth16(a: a, b: a, c: a, publicInputs: [])

        expect(GPUProofSerializerEngine.isCompatible(serialized.data),
               "Version compat: current version is compatible")
        expect(!GPUProofSerializerEngine.isCompatible([]),
               "Version compat: empty data is not compatible")
        expect(!GPUProofSerializerEngine.isCompatible([0xFF, 0xFF, 0xFF, 0xFF]),
               "Version compat: wrong magic is not compatible")

        // Tamper with version to a future version
        var tampered = serialized.data
        tampered[4] = 0xFF  // version = 255
        tampered[5] = 0x00
        expect(!GPUProofSerializerEngine.isCompatible(tampered),
               "Version compat: future version is not compatible")
    }

    // ---- Test 12: Size estimator ----
    do {
        let estGroth16 = ProofSizeEstimator.estimateGroth16(publicInputCount: 3)
        expect(estGroth16 > 0, "Size estimator: Groth16 estimate > 0")
        // 3 compressed points (33 bytes each) + 3 public inputs (32 each) + overhead
        expect(estGroth16 >= 3 * 33 + 3 * 32, "Size estimator: Groth16 >= minimum")

        let estPlonk = ProofSizeEstimator.estimatePlonk(numWires: 3, numEvaluations: 5)
        expect(estPlonk > estGroth16, "Size estimator: Plonk > Groth16 (more data)")

        let estSTARK = ProofSizeEstimator.estimateSTARK(traceWidth: 4, numFRILayers: 8,
                                                         friLayerSize: 16, numQueries: 32,
                                                         merkleDepth: 10)
        expect(estSTARK > 0, "Size estimator: STARK estimate > 0")

        let estFRI = ProofSizeEstimator.estimateFRI(numLayers: 5, layerSize: 16,
                                                     numQueries: 16, merkleDepth: 10)
        expect(estFRI > 0, "Size estimator: FRI estimate > 0")

        // Raw vs compressed estimates
        let estRaw = ProofSizeEstimator.estimateGroth16(publicInputCount: 3, flags: .none)
        let estCompressed = ProofSizeEstimator.estimateGroth16(publicInputCount: 3, flags: .default)
        expect(estRaw > estCompressed,
               "Size estimator: raw estimate > compressed estimate")
    }

    // ---- Test 13: Streaming serialization buffer ----
    do {
        let stream = StreamingSerializationBuffer(capacity: 256)

        stream.beginSection(.fieldElements)
        stream.writeFr(frFromInt(42))
        stream.writeFr(frFromInt(99))
        stream.endSection()

        stream.beginSection(.groupPoints)
        let p = makeTestPoint(7)
        stream.writeCompressedPoint(p)
        stream.endSection()

        let data = stream.finalize()
        expect(data.count > 0, "Streaming buffer: non-empty output")
        expect(stream.bytesWritten > 0, "Streaming buffer: bytes written > 0")

        // Verify section structure: tag (1 byte) + length (4 bytes) + data
        expect(data[0] == SectionTag.fieldElements.rawValue,
               "Streaming buffer: first section tag is fieldElements")
        let section0Len = UInt32(data[1]) | (UInt32(data[2]) << 8) |
                          (UInt32(data[3]) << 16) | (UInt32(data[4]) << 24)
        expect(section0Len == 64, "Streaming buffer: first section is 64 bytes (2 Fr elements)")
    }

    // ---- Test 14: Streaming Merkle path ----
    do {
        let stream = StreamingSerializationBuffer(capacity: 256)

        let path = [makeTestHash(0xAA), makeTestHash(0xBB), makeTestHash(0xCC)]
        stream.beginSection(.merklePath)
        stream.writeMerklePath(path)
        stream.endSection()

        let data = stream.finalize()
        expect(data[0] == SectionTag.merklePath.rawValue,
               "Merkle path streaming: correct section tag")
        // 4 bytes count + 3 * 32 bytes = 100 bytes section data
        let sectionLen = UInt32(data[1]) | (UInt32(data[2]) << 8) |
                         (UInt32(data[3]) << 16) | (UInt32(data[4]) << 24)
        expect(sectionLen == 100, "Merkle path streaming: section length is 100")
    }

    // ---- Test 15: Custom proof serialization ----
    do {
        let engine = GPUProofSerializerEngine()
        let serialized = engine.serializeCustom(proofType: .spartan, curveId: .bn254,
                                                 estimatedSize: 128) { stream in
            stream.beginSection(.fieldElements)
            stream.writeFr(frFromInt(1))
            stream.writeFr(frFromInt(2))
            stream.writeFr(frFromInt(3))
            stream.endSection()

            stream.beginSection(.customData)
            stream.writeBytes([0xDE, 0xAD, 0xBE, 0xEF])
            stream.endSection()
        }

        expect(serialized.proofType == .spartan, "Custom serialize: proof type is spartan")
        expect(serialized.curveId == .bn254, "Custom serialize: curve is bn254")
        expect(serialized.data.count > 20, "Custom serialize: data has header + body")

        // Verify we can read sections back
        let reader = try SerializedProofReader(serialized.data)
        expect(reader.proofType == .spartan, "Custom deserialize: proof type matches")

        guard let (tag1, sec1) = try reader.readSection() else {
            expect(false, "Custom deserialize: missing section 1")
            return
        }
        expect(tag1 == .fieldElements, "Custom deserialize: section 1 is fieldElements")
        expect(sec1.count == 96, "Custom deserialize: section 1 has 96 bytes (3 Fr)")

        guard let (tag2, sec2) = try reader.readSection() else {
            expect(false, "Custom deserialize: missing section 2")
            return
        }
        expect(tag2 == .customData, "Custom deserialize: section 2 is customData")
        expect(sec2.count == 4, "Custom deserialize: section 2 has 4 bytes")
        expect(sec2 == [0xDE, 0xAD, 0xBE, 0xEF], "Custom deserialize: custom data matches")

        expect(!reader.hasMoreSections, "Custom deserialize: no more sections")
    } catch {
        expect(false, "Custom serialize threw: \(error)")
    }

    // ---- Test 16: Deserialization error handling (truncated data) ----
    do {
        let shortData: [UInt8] = [0x5A, 0x4B, 0x53, 0x5A] // Just magic, no rest
        do {
            _ = try SerializedProofReader(shortData)
            expect(false, "Truncated header: should throw")
        } catch {
            expect(true, "Truncated header: threw as expected")
        }
    }

    // ---- Test 17: Deserialization error handling (wrong magic) ----
    do {
        var badMagic = [UInt8](repeating: 0, count: 20)
        badMagic[0] = 0xFF
        do {
            _ = try SerializedProofReader(badMagic)
            expect(false, "Wrong magic: should throw")
        } catch {
            expect(true, "Wrong magic: threw as expected")
        }
    }

    // ---- Test 18: Deserialization error handling (wrong proof type) ----
    do {
        let engine = GPUProofSerializerEngine()
        let a = makeTestPoint(3)
        let serialized = engine.serializeGroth16(a: a, b: a, c: a, publicInputs: [])

        do {
            _ = try engine.deserializePlonk(serialized)
            expect(false, "Wrong proof type: should throw on Plonk deserialize of Groth16")
        } catch {
            expect(true, "Wrong proof type: threw as expected")
        }
    } catch {
        expect(false, "Wrong proof type test threw unexpectedly: \(error)")
    }

    // ---- Test 19: Max compression vs no compression size comparison ----
    do {
        let engineMax = GPUProofSerializerEngine(flags: .maxCompression)
        let engineNone = GPUProofSerializerEngine(flags: .none)

        // Data with lots of repeated values
        var pubInputs = [Fr]()
        for _ in 0..<100 { pubInputs.append(Fr.zero) }

        let a = makeTestPoint(3)
        let serMax = engineMax.serializeGroth16(a: a, b: a, c: a, publicInputs: pubInputs)
        let serNone = engineNone.serializeGroth16(a: a, b: a, c: a, publicInputs: pubInputs)

        expect(serMax.data.count < serNone.data.count,
               "Max compression: compressed size < raw size for repeated data")
    }

    // ---- Test 20: Empty proof roundtrip ----
    do {
        let engine = GPUProofSerializerEngine()
        let serialized = engine.serializeGroth16(a: pointIdentity(), b: pointIdentity(),
                                                  c: pointIdentity(), publicInputs: [])

        let (da, db, dc, dpub) = try engine.deserializeGroth16(serialized)
        expect(pointIsIdentity(da), "Empty proof: A is identity")
        expect(pointIsIdentity(db), "Empty proof: B is identity")
        expect(pointIsIdentity(dc), "Empty proof: C is identity")
        expect(dpub.isEmpty, "Empty proof: no public inputs")
    } catch {
        expect(false, "Empty proof threw: \(error)")
    }

    // ---- Test 21: Large Plonk proof roundtrip ----
    do {
        let engine = GPUProofSerializerEngine()
        var wires = [PointProjective]()
        for i: UInt64 in 1...8 { wires.append(makeTestPoint(i)) }
        var quotients = [PointProjective]()
        for i: UInt64 in 10...17 { quotients.append(makeTestPoint(i)) }
        var evals = [Fr]()
        for i: UInt64 in 1...20 { evals.append(frFromInt(i * 100)) }

        let serialized = engine.serializePlonk(
            wireCommitments: wires,
            grandProductCommitment: makeTestPoint(99),
            quotientCommitments: quotients,
            openingProof: makeTestPoint(200),
            shiftedOpeningProof: makeTestPoint(201),
            evaluations: evals
        )

        let result = try engine.deserializePlonk(serialized)
        expect(result.wireCommitments.count == 8, "Large Plonk: 8 wire commitments")
        expect(result.quotientCommitments.count == 8, "Large Plonk: 8 quotient commitments")
        expect(result.evaluations.count == 20, "Large Plonk: 20 evaluations")

        for i in 0..<8 {
            expect(pointsEqualAffine(result.wireCommitments[i], wires[i]),
                   "Large Plonk: wire[\(i)] matches")
        }
        for i in 0..<20 {
            expect(frEqual(result.evaluations[i], evals[i]),
                   "Large Plonk: eval[\(i)] matches")
        }
    } catch {
        expect(false, "Large Plonk threw: \(error)")
    }

    // ---- Test 22: SerializedProofReader section parsing ----
    do {
        // Read Fr from raw bytes
        let fr = frFromInt(12345)
        var buf = [UInt8]()
        let limbs = [fr.v.0, fr.v.1, fr.v.2, fr.v.3,
                     fr.v.4, fr.v.5, fr.v.6, fr.v.7]
        for limb in limbs {
            buf.append(UInt8(limb & 0xFF))
            buf.append(UInt8((limb >> 8) & 0xFF))
            buf.append(UInt8((limb >> 16) & 0xFF))
            buf.append(UInt8((limb >> 24) & 0xFF))
        }
        var pos = 0
        let readBack = try SerializedProofReader.readFr(from: buf, at: &pos)
        expect(frEqual(readBack, fr), "Reader helper: readFr roundtrip")
        expect(pos == 32, "Reader helper: readFr advanced pos by 32")
    } catch {
        expect(false, "Reader helper threw: \(error)")
    }

    // ---- Test 23: STARK with RLE compression ----
    do {
        let engine = GPUProofSerializerEngine(flags: [.runLengthEncode])
        // Lots of repeated constraint evaluations
        var evals = [Fr]()
        for _ in 0..<50 { evals.append(Fr.zero) }
        for _ in 0..<25 { evals.append(frFromInt(42)) }

        let serialized = engine.serializeSTARK(
            traceCommitments: [makeTestHash(1)],
            constraintEvaluations: evals,
            friLayers: [[frFromInt(1), frFromInt(1), frFromInt(1)]],
            queryPaths: [[makeTestHash(0xAA)]]
        )

        let result = try engine.deserializeSTARK(serialized)
        expect(result.constraintEvaluations.count == 75, "STARK RLE: 75 constraint evals")
        expect(frEqual(result.constraintEvaluations[0], Fr.zero), "STARK RLE: first is zero")
        expect(frEqual(result.constraintEvaluations[49], Fr.zero), "STARK RLE: 50th is zero")
        expect(frEqual(result.constraintEvaluations[50], frFromInt(42)), "STARK RLE: 51st is 42")
        expect(result.friLayers[0].count == 3, "STARK RLE: FRI layer has 3 elements")
    } catch {
        expect(false, "STARK RLE threw: \(error)")
    }

    // ---- Test 24: Version field ----
    do {
        expect(GPUProofSerializerEngine.version.version == "1.0.0",
               "Engine version: 1.0.0")
    }
}
