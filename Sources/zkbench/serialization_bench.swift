// Proof Serialization Benchmark & Roundtrip Tests
import zkMetal
import Foundation

public func runSerializationBench() {
    fputs("\n=== Proof Serialization Benchmark ===\n", stderr)

    var rng: UInt64 = 0xDEAD_BEEF_CAFE_1234

    // Helper: pseudo-random Fr
    func randFr() -> Fr {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(rng >> 16)
    }

    // Helper: pseudo-random Fp
    func randFp() -> Fp {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return fpFromInt(rng >> 16)
    }

    // Helper: pseudo-random PointProjective
    func randPoint() -> PointProjective {
        PointProjective(x: randFp(), y: randFp(), z: randFp())
    }

    // Helper: Fr equality check
    func frEq(_ a: Fr, _ b: Fr) -> Bool {
        a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
        a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    func fpEq(_ a: Fp, _ b: Fp) -> Bool {
        a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
        a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    func ptEq(_ a: PointProjective, _ b: PointProjective) -> Bool {
        fpEq(a.x, b.x) && fpEq(a.y, b.y) && fpEq(a.z, b.z)
    }

    var allPass = true

    // ========================================================================
    // 1. ProofWriter/ProofReader primitives
    // ========================================================================
    fputs("\n--- Primitive roundtrip tests ---\n", stderr)

    do {
        // Fr roundtrip
        let testFr = randFr()
        let w = ProofWriter()
        w.writeFr(testFr)
        let r = ProofReader(w.finalize())
        let got = try r.readFr()
        let pass = frEq(testFr, got)
        fputs("  Fr roundtrip: \(pass ? "PASS" : "FAIL")\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  Fr roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    do {
        // Fr array roundtrip
        let arr = (0..<16).map { _ in randFr() }
        let w = ProofWriter()
        w.writeFrArray(arr)
        let r = ProofReader(w.finalize())
        let got = try r.readFrArray()
        let pass = got.count == arr.count && zip(arr, got).allSatisfy { frEq($0, $1) }
        fputs("  FrArray[16] roundtrip: \(pass ? "PASS" : "FAIL")\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  FrArray roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    do {
        // PointProjective roundtrip
        let pt = randPoint()
        let w = ProofWriter()
        w.writePointProjective(pt)
        let r = ProofReader(w.finalize())
        let got = try r.readPointProjective()
        let pass = ptEq(pt, got)
        fputs("  PointProjective roundtrip: \(pass ? "PASS" : "FAIL")\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  PointProjective roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    do {
        // UInt32/UInt64/Label roundtrip
        let w = ProofWriter()
        w.writeUInt32(0xDEADBEEF)
        w.writeUInt64(0xCAFEBABE_12345678)
        w.writeLabel("test-label-v1")
        w.writeBytes([1, 2, 3, 4, 5])
        let r = ProofReader(w.finalize())
        let u32 = try r.readUInt32()
        let u64 = try r.readUInt64()
        let label = try r.readLabel()
        let bytes = try r.readBytes()
        let pass = u32 == 0xDEADBEEF && u64 == 0xCAFEBABE_12345678 &&
                   label == "test-label-v1" && bytes == [1, 2, 3, 4, 5] && r.isAtEnd
        fputs("  UInt32/UInt64/Label/Bytes roundtrip: \(pass ? "PASS" : "FAIL")\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  Primitives roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    // Hex/Base64 roundtrip
    do {
        let original: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF]
        let hex = original.toHex()
        let fromHex = [UInt8].fromHex(hex)
        let hexPass = fromHex == original && hex == "deadbeef00ff"
        fputs("  Hex roundtrip: \(hexPass ? "PASS" : "FAIL")\n", stderr)
        if !hexPass { allPass = false }

        let b64 = original.toBase64()
        let fromB64 = [UInt8].fromBase64(b64)
        let b64Pass = fromB64 == original
        fputs("  Base64 roundtrip: \(b64Pass ? "PASS" : "FAIL")\n", stderr)
        if !b64Pass { allPass = false }
    }

    // Error handling
    do {
        let r = ProofReader([0x01, 0x02])  // only 2 bytes
        _ = try r.readUInt32()  // needs 4
        fputs("  Truncation detection: FAIL (no error thrown)\n", stderr)
        allPass = false
    } catch is ProofSerializationError {
        fputs("  Truncation detection: PASS\n", stderr)
    } catch {
        fputs("  Truncation detection: FAIL (wrong error: \(error))\n", stderr)
        allPass = false
    }

    do {
        let w = ProofWriter()
        w.writeLabel("WRONG-v1")
        let r = ProofReader(w.finalize())
        try r.expectLabel("RIGHT-v1")
        fputs("  Label mismatch detection: FAIL (no error thrown)\n", stderr)
        allPass = false
    } catch let e as ProofSerializationError {
        if case .wrongLabel = e {
            fputs("  Label mismatch detection: PASS\n", stderr)
        } else {
            fputs("  Label mismatch detection: FAIL (wrong error variant)\n", stderr)
            allPass = false
        }
    } catch {
        fputs("  Label mismatch detection: FAIL (wrong error: \(error))\n", stderr)
        allPass = false
    }

    // ========================================================================
    // 2. KZG Proof roundtrip
    // ========================================================================
    fputs("\n--- KZG Proof serialization ---\n", stderr)
    do {
        let proof = KZGProof(evaluation: randFr(), witness: randPoint())
        let bytes = proof.serialize()
        let restored = try KZGProof.deserialize(bytes)
        let pass = frEq(proof.evaluation, restored.evaluation) &&
                   ptEq(proof.witness, restored.witness)
        fputs("  KZG roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
        if !pass { allPass = false }

        // Throughput
        let runs = 1000
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runs {
            let _ = proof.serialize()
        }
        let serTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runs {
            let _ = try KZGProof.deserialize(bytes)
        }
        let desTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs(String(format: "  KZG %dx: ser %.2f ms, deser %.2f ms\n", runs, serTime, desTime), stderr)
    } catch {
        fputs("  KZG roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    // ========================================================================
    // 3. IPA Proof roundtrip
    // ========================================================================
    fputs("\n--- IPA Proof serialization ---\n", stderr)
    for logN in [4, 8] {
        do {
            let numRounds = logN
            let L = (0..<numRounds).map { _ in randPoint() }
            let R = (0..<numRounds).map { _ in randPoint() }
            let proof = IPAProof(L: L, R: R, a: randFr())
            let bytes = proof.serialize()
            let restored = try IPAProof.deserialize(bytes)
            let pass = restored.L.count == L.count && restored.R.count == R.count &&
                       frEq(proof.a, restored.a) &&
                       zip(L, restored.L).allSatisfy { ptEq($0, $1) } &&
                       zip(R, restored.R).allSatisfy { ptEq($0, $1) }
            fputs("  IPA (rounds=\(numRounds)) roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
            if !pass { allPass = false }
        } catch {
            fputs("  IPA (rounds=\(logN)) roundtrip: FAIL (\(error))\n", stderr)
            allPass = false
        }
    }

    // ========================================================================
    // 4. Sumcheck Proof roundtrip
    // ========================================================================
    fputs("\n--- Sumcheck Proof serialization ---\n", stderr)
    for numVars in [8, 16] {
        do {
            let rounds: [(Fr, Fr, Fr)] = (0..<numVars).map { _ in (randFr(), randFr(), randFr()) }
            let proof = SumcheckProof(rounds: rounds, finalEval: randFr())
            let bytes = proof.serialize()
            let restored = try SumcheckProof.deserialize(bytes)
            let pass = restored.rounds.count == rounds.count &&
                       frEq(proof.finalEval, restored.finalEval) &&
                       zip(rounds, restored.rounds).allSatisfy { frEq($0.0, $1.0) && frEq($0.1, $1.1) && frEq($0.2, $1.2) }
            fputs("  Sumcheck (vars=\(numVars)) roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
            if !pass { allPass = false }
        } catch {
            fputs("  Sumcheck (vars=\(numVars)) roundtrip: FAIL (\(error))\n", stderr)
            allPass = false
        }
    }

    // ========================================================================
    // 5. Lookup Proof roundtrip
    // ========================================================================
    fputs("\n--- Lookup Proof serialization ---\n", stderr)
    for tableSize in [8, 64] {
        do {
            let mult = (0..<tableSize).map { _ in randFr() }
            let numRounds = Int(log2(Double(tableSize)))
            let lookupRounds: [(Fr, Fr, Fr)] = (0..<numRounds).map { _ in (randFr(), randFr(), randFr()) }
            let tableRounds: [(Fr, Fr, Fr)] = (0..<numRounds).map { _ in (randFr(), randFr(), randFr()) }
            let proof = LookupProof(
                multiplicities: mult, beta: randFr(),
                lookupSumcheckRounds: lookupRounds,
                tableSumcheckRounds: tableRounds,
                claimedSum: randFr(), lookupFinalEval: randFr(), tableFinalEval: randFr())
            let bytes = proof.serialize()
            let restored = try LookupProof.deserialize(bytes)
            let pass = restored.multiplicities.count == mult.count &&
                       frEq(proof.beta, restored.beta) &&
                       frEq(proof.claimedSum, restored.claimedSum) &&
                       frEq(proof.lookupFinalEval, restored.lookupFinalEval) &&
                       frEq(proof.tableFinalEval, restored.tableFinalEval) &&
                       zip(mult, restored.multiplicities).allSatisfy { frEq($0, $1) } &&
                       restored.lookupSumcheckRounds.count == lookupRounds.count &&
                       restored.tableSumcheckRounds.count == tableRounds.count
            fputs("  Lookup (N=\(tableSize)) roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
            if !pass { allPass = false }
        } catch {
            fputs("  Lookup (N=\(tableSize)) roundtrip: FAIL (\(error))\n", stderr)
            allPass = false
        }
    }

    // ========================================================================
    // 6. Lasso Proof roundtrip
    // ========================================================================
    fputs("\n--- Lasso Proof serialization ---\n", stderr)
    do {
        let numChunks = 4
        let chunkSize = 16
        let m = 32
        var subtableProofs = [SubtableProof]()
        for k in 0..<numChunks {
            let readCounts = (0..<chunkSize).map { _ in randFr() }
            let numRounds = Int(log2(Double(m)))
            let readRounds: [(Fr, Fr, Fr)] = (0..<numRounds).map { _ in (randFr(), randFr(), randFr()) }
            let tableRounds: [(Fr, Fr, Fr)] = (0..<numRounds).map { _ in (randFr(), randFr(), randFr()) }
            subtableProofs.append(SubtableProof(
                chunkIndex: k, readCounts: readCounts, beta: randFr(),
                readSumcheckRounds: readRounds, tableSumcheckRounds: tableRounds,
                claimedSum: randFr(), readFinalEval: randFr(), tableFinalEval: randFr()))
        }
        let indices = (0..<numChunks).map { _ in (0..<m).map { _ in Int.random(in: 0..<chunkSize) } }
        let proof = LassoProof(numChunks: numChunks, subtableProofs: subtableProofs, indices: indices)
        let bytes = proof.serialize()
        let restored = try LassoProof.deserialize(bytes)
        let pass = restored.numChunks == numChunks &&
                   restored.subtableProofs.count == numChunks &&
                   restored.indices.count == numChunks &&
                   (0..<numChunks).allSatisfy { k in
                       restored.subtableProofs[k].chunkIndex == k &&
                       restored.subtableProofs[k].readCounts.count == chunkSize &&
                       restored.indices[k] == indices[k]
                   }
        fputs("  Lasso (chunks=\(numChunks), m=\(m)) roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  Lasso roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    // ========================================================================
    // 7. FRI Commitment roundtrip
    // ========================================================================
    fputs("\n--- FRI serialization ---\n", stderr)
    for logN in [8, 14] {
        do {
            let numLayers = logN
            let layers = (0..<numLayers).map { l in
                (0..<(1 << (logN - l))).map { _ in randFr() }
            }
            let roots = (0..<numLayers).map { _ in randFr() }
            let betas = (0..<numLayers).map { _ in randFr() }
            let commitment = FRICommitment(layers: layers, roots: roots, betas: betas, finalValue: randFr())
            let bytes = commitment.serialize()
            let restored = try FRICommitment.deserialize(bytes)
            let pass = restored.layers.count == numLayers &&
                       frEq(commitment.finalValue, restored.finalValue) &&
                       zip(roots, restored.roots).allSatisfy { frEq($0, $1) } &&
                       zip(betas, restored.betas).allSatisfy { frEq($0, $1) } &&
                       (0..<numLayers).allSatisfy { l in
                           layers[l].count == restored.layers[l].count &&
                           zip(layers[l], restored.layers[l]).allSatisfy { frEq($0, $1) }
                       }
            fputs("  FRI Commitment (2^\(logN)) roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
            if !pass { allPass = false }

            // Throughput for serialization
            let runs = logN <= 10 ? 100 : 5
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<runs {
                let _ = commitment.serialize()
            }
            let serTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(runs)
            let t1 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<runs {
                let _ = try FRICommitment.deserialize(bytes)
            }
            let desTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(runs)
            fputs(String(format: "  FRI 2^%d: ser %.2f ms, deser %.2f ms, size %.1f KB\n",
                         logN, serTime, desTime, Double(bytes.count) / 1024.0), stderr)
        } catch {
            fputs("  FRI (2^\(logN)) roundtrip: FAIL (\(error))\n", stderr)
            allPass = false
        }
    }

    // FRI QueryProof roundtrip
    do {
        let numLayers = 4
        let layerEvals: [(Fr, Fr)] = (0..<numLayers).map { _ in (randFr(), randFr()) }
        let merklePaths: [[[Fr]]] = (0..<numLayers).map { _ in
            (0..<4).map { _ in  // 4 siblings per layer
                (0..<4).map { _ in randFr() }  // 4 Fr per sibling
            }
        }
        let queryProof = FRIQueryProof(initialIndex: 42, layerEvals: layerEvals, merklePaths: merklePaths)
        let bytes = queryProof.serialize()
        let restored = try FRIQueryProof.deserialize(bytes)
        let pass = restored.initialIndex == 42 &&
                   restored.layerEvals.count == numLayers &&
                   zip(layerEvals, restored.layerEvals).allSatisfy { frEq($0.0, $1.0) && frEq($0.1, $1.1) } &&
                   restored.merklePaths.count == numLayers
        fputs("  FRI QueryProof roundtrip: \(pass ? "PASS" : "FAIL") (\(bytes.count) bytes)\n", stderr)
        if !pass { allPass = false }
    } catch {
        fputs("  FRI QueryProof roundtrip: FAIL (\(error))\n", stderr)
        allPass = false
    }

    // ========================================================================
    // 8. Serialization size summary
    // ========================================================================
    fputs("\n--- Proof sizes ---\n", stderr)
    func sizeRow(_ name: String, _ size: Int) {
        let padded = name.padding(toLength: 25, withPad: " ", startingAt: 0)
        fputs("  \(padded) \(String(format: "%7d", size)) B\n", stderr)
    }
    fputs("  Proof type                   Size\n", stderr)
    fputs("  " + String(repeating: "-", count: 37) + "\n", stderr)

    // KZG
    let kzgBytes = KZGProof(evaluation: randFr(), witness: randPoint()).serialize()
    sizeRow("KZG", kzgBytes.count)

    // IPA (8 rounds)
    let ipaBytes = IPAProof(L: (0..<8).map { _ in randPoint() },
                            R: (0..<8).map { _ in randPoint() },
                            a: randFr()).serialize()
    sizeRow("IPA (8 rounds)", ipaBytes.count)

    // Sumcheck (16 vars)
    let scBytes = SumcheckProof(
        rounds: (0..<16).map { _ in (randFr(), randFr(), randFr()) },
        finalEval: randFr()).serialize()
    sizeRow("Sumcheck (16 vars)", scBytes.count)

    // Lookup (N=64)
    let lookupBytes = LookupProof(
        multiplicities: (0..<64).map { _ in randFr() }, beta: randFr(),
        lookupSumcheckRounds: (0..<6).map { _ in (randFr(), randFr(), randFr()) },
        tableSumcheckRounds: (0..<6).map { _ in (randFr(), randFr(), randFr()) },
        claimedSum: randFr(), lookupFinalEval: randFr(), tableFinalEval: randFr()).serialize()
    sizeRow("Lookup (N=64)", lookupBytes.count)

    // ========================================================================
    // 9. Hex/Base64 encoding throughput
    // ========================================================================
    fputs("\n--- Encoding throughput ---\n", stderr)
    let bigPayload = (0..<10000).map { _ -> UInt8 in
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return UInt8(truncatingIfNeeded: rng >> 32)
    }
    let hexRuns = 1000
    let ht0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<hexRuns { let _ = bigPayload.toHex() }
    let hexEncTime = (CFAbsoluteTimeGetCurrent() - ht0) * 1000
    let hexStr = bigPayload.toHex()
    let ht1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<hexRuns { let _ = [UInt8].fromHex(hexStr) }
    let hexDecTime = (CFAbsoluteTimeGetCurrent() - ht1) * 1000
    fputs(String(format: "  Hex  (10KB, %dx): enc %.1f ms, dec %.1f ms\n", hexRuns, hexEncTime, hexDecTime), stderr)

    let bt0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<hexRuns { let _ = bigPayload.toBase64() }
    let b64EncTime = (CFAbsoluteTimeGetCurrent() - bt0) * 1000
    let b64Str = bigPayload.toBase64()
    let bt1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<hexRuns { let _ = [UInt8].fromBase64(b64Str) }
    let b64DecTime = (CFAbsoluteTimeGetCurrent() - bt1) * 1000
    fputs(String(format: "  B64  (10KB, %dx): enc %.1f ms, dec %.1f ms\n", hexRuns, b64EncTime, b64DecTime), stderr)

    // ========================================================================
    // Summary
    // ========================================================================
    fputs("\n\(allPass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")\n", stderr)
}
