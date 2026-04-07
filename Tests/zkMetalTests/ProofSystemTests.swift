import zkMetal

func runProofSystemTests() {
    suite("IPA")
    expect(frToInt(IPAEngine.innerProduct(
        [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
        [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]))[0] == 70, "Inner product")

    for logN in [2, 4, 6] {
        let n = 1 << logN
        do {
            let (gens, Q) = IPAEngine.generateTestGenerators(count: n)
            let engine = try IPAEngine(generators: gens, Q: Q)
            var a = [Fr](); var bv = [Fr]()
            for i in 0..<n { a.append(frFromInt(UInt64(i + 1))); bv.append(frFromInt(UInt64(n - i))) }
            let v = IPAEngine.innerProduct(a, bv)
            let C = try engine.commit(a)
            let proof = try engine.createProof(a: a, b: bv)
            expect(engine.verify(commitment: C, b: bv, innerProductValue: v, proof: proof), "IPA n=\(n)")
            let wrongV = frFromInt(999)
            expect(!engine.verify(commitment: C, b: bv, innerProductValue: wrongV, proof: proof), "IPA reject n=\(n)")
        } catch { expect(false, "IPA n=\(n) error: \(error)") }
    }

    suite("Verkle Trees")
    do {
        let engine = try VerkleEngine(width: 4)
        let vals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let C = try engine.commit(vals)
        expect(!pointIsIdentity(C), "Commit non-trivial")
        for idx in 0..<4 {
            let proof = try engine.createOpeningProof(values: vals, index: idx)
            expect(engine.verifyOpeningProof(proof), "Opening \(idx)")
        }
        let numLeaves = 8
        var leaves = [Fr]()
        for i in 0..<numLeaves { leaves.append(frFromInt(UInt64(i + 1))) }
        let (levels, _) = try engine.buildTree(leaves: leaves)
        let root = levels.last![0]
        let p0 = try engine.createPathProof(leaves: leaves, leafIndex: 0)
        expect(engine.verifyPathProof(p0, root: root), "Path proof leaf 0")
        expect(!engine.verifyPathProof(p0, root: pointDouble(root)), "Reject wrong root")
    } catch { expect(false, "Verkle error: \(error)") }

    suite("LogUp Lookup")
    do {
        let engine = try LookupEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let lookups = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let proof = try engine.prove(table: table, lookups: lookups, beta: frFromInt(12345))
        expect(try engine.verify(proof: proof, table: table, lookups: lookups), "Simple lookup")

        let table2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let lookups2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }
        let proof2 = try engine.prove(table: table2, lookups: lookups2, beta: frFromInt(99999))
        expect(try engine.verify(proof: proof2, table: table2, lookups: lookups2), "Repeated lookups")

        // Reject tampered
        let tampered = LookupProof(
            multiplicities: proof.multiplicities, beta: proof.beta,
            lookupSumcheckRounds: proof.lookupSumcheckRounds,
            tableSumcheckRounds: proof.tableSumcheckRounds,
            claimedSum: frAdd(proof.claimedSum, Fr.one),
            lookupFinalEval: proof.lookupFinalEval, tableFinalEval: proof.tableFinalEval)
        expect(try !engine.verify(proof: tampered, table: table, lookups: lookups), "Reject tampered")
    } catch { expect(false, "LogUp error: \(error)") }

    suite("cq Cached Quotients Lookup")
    do {
        // Generate test SRS
        let cqGen = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let cqSecret: [UInt32] = [0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1111, 0x2222, 0x3333, 0x0001]
        let cqSrsSecret = frFromLimbs(cqSecret)
        let cqSrs = KZGEngine.generateTestSRS(secret: cqSecret, size: 256, generator: cqGen)
        print("  SRS generated: \(cqSrs.count) points")
        let cqEngine = try CQEngine(srs: cqSrs)
        print("  CQEngine initialized")

        // Test 1: Simple lookup (N=4, |T|=4)
        let cqTable1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        print("  Preprocessing table...")
        let cqTc1 = try cqEngine.preprocessTable(table: cqTable1)
        print("  Table preprocessed, cached quotients: \(cqTc1.cachedQuotientCommitments.count)")
        let cqLookups1: [Fr] = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        print("  Proving...")
        let cqProof1 = try cqEngine.prove(lookups: cqLookups1, table: cqTc1)
        print("  Proof generated, verifying...")
        let cqValid1 = cqEngine.verify(proof: cqProof1, table: cqTc1, numLookups: 4, srsSecret: cqSrsSecret)
        print("  Verify result: \(cqValid1)")
        expect(cqValid1, "cq simple lookup")

        // Test 2: Repeated lookups (N=8, |T|=8)
        let cqTable2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let cqTc2: CQTableCommitment
        let cqLookups2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }
        do {
            cqTc2 = try cqEngine.preprocessTable(table: cqTable2)
            let cqProof2 = try cqEngine.prove(lookups: cqLookups2, table: cqTc2)
            let cqValid2 = cqEngine.verify(proof: cqProof2, table: cqTc2, numLookups: 8, srsSecret: cqSrsSecret)
            expect(cqValid2, "cq repeated lookups")
            expect(cqTc2.cachedQuotientCommitments.count == 8, "cq cached quotients count 8")
        } catch { print("  cq test2 error: \(error)"); expect(false, "cq test2: \(error)"); return }

        // Test 3: Cached quotients exist and have correct count
        expect(cqTc1.cachedQuotientCommitments.count == 4, "cq cached quotients count")
        expect(cqTc2.cachedQuotientCommitments.count == 8, "cq cached quotients count 8")

        // Test 4: Multiplicities correct
        let cqMult = CQEngine.computeMultiplicities(table: cqTable2, lookups: cqLookups2)
        let cqExpMult: [UInt64] = [2, 0, 2, 0, 2, 0, 2, 0]
        var cqMultOk = true
        for i in 0..<8 {
            if !frEqual(cqMult[i], frFromInt(cqExpMult[i])) { cqMultOk = false; break }
        }
        expect(cqMultOk, "cq multiplicities correct")

        // Test 5: Asymmetric (N < |T|) -- lookups are a small subset of table
        let cqTable3: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let cqTc3 = try cqEngine.preprocessTable(table: cqTable3)
        let cqLookups3: [Fr] = (0..<4).map { cqTable3[$0] }
        let cqProof3 = try cqEngine.prove(lookups: cqLookups3, table: cqTc3)
        let cqValid3 = cqEngine.verify(proof: cqProof3, table: cqTc3, numLookups: 4, srsSecret: cqSrsSecret)
        expect(cqValid3, "cq asymmetric N=4 T=16")

        // Test 6: N > |T| (many lookups into small table)
        let cqTable4: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]
        let cqTc4 = try cqEngine.preprocessTable(table: cqTable4)
        var cqLookups4 = [Fr]()
        for i in 0..<16 { cqLookups4.append(cqTable4[i % 4]) }
        let cqProof4 = try cqEngine.prove(lookups: cqLookups4, table: cqTc4)
        let cqValid4 = cqEngine.verify(proof: cqProof4, table: cqTc4, numLookups: 16, srsSecret: cqSrsSecret)
        expect(cqValid4, "cq N>T (N=16 T=4)")

        // Test 7: Reject tampered multiplicity sum
        let cqTampered = CQProof(
            phiCommitment: cqProof1.phiCommitment,
            quotientCommitment: cqProof1.quotientCommitment,
            multiplicities: cqProof1.multiplicities,
            multiplicitySum: frAdd(cqProof1.multiplicitySum, Fr.one),
            challengeZ: cqProof1.challengeZ,
            phiOpening: cqProof1.phiOpening,
            tOpening: cqProof1.tOpening
        )
        let cqRejected = !cqEngine.verify(proof: cqTampered, table: cqTc1, numLookups: 4, srsSecret: cqSrsSecret)
        expect(cqRejected, "cq reject tampered sum")

        // Test 8: proveAndVerify convenience
        let (_, cqRtValid) = try cqEngine.proveAndVerify(lookups: cqLookups1, table: cqTc1, srsSecret: cqSrsSecret)
        expect(cqRtValid, "cq proveAndVerify round-trip")

        // Test 9: Larger table (|T|=64, N=32)
        let cqTable5: [Fr] = (0..<64).map { frFromInt(UInt64($0 * 11 + 5)) }
        let cqTc5 = try cqEngine.preprocessTable(table: cqTable5)
        var cqRng: UInt64 = 0xCAFE_BABE
        var cqLookups5 = [Fr]()
        for _ in 0..<32 {
            cqRng = cqRng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(cqRng >> 32) % 64
            cqLookups5.append(cqTable5[idx])
        }
        let (_, cqValid5) = try cqEngine.proveAndVerify(lookups: cqLookups5, table: cqTc5, srsSecret: cqSrsSecret)
        expect(cqValid5, "cq larger (N=32 T=64)")

    } catch { print("  cq error: \(error)"); expect(false, "cq Lookup error: \(error)") }

    suite("ECDSA")
    do {
        let engine = try ECDSAEngine()
        let d = secpFrFromInt(42)
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)
        let Q = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(d)))
        let k = secpFrFromInt(137); let z = secpFrFromInt(12345)
        let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
        let rAff = secpPointToAffine(rProj)
        var rModN = secpToInt(rAff.x)
        if gte256(rModN, SecpFr.N) { (rModN, _) = sub256(rModN, SecpFr.N) }
        let rFr = secpFrFromRaw(rModN)
        let sFr = secpFrMul(secpFrInverse(k), secpFrAdd(z, secpFrMul(rFr, d)))
        let sig = ECDSASignature(r: rFr, s: sFr, z: z)

        expect(engine.verify(sig: sig, pubkey: Q), "Single verify")
        expect(!engine.verify(sig: ECDSASignature(r: rFr, s: sFr, z: secpFrFromInt(99999)), pubkey: Q), "Reject wrong z")

        // Batch verify
        let batchN = 16
        var sigs = [ECDSASignature](); var pks = [SecpPointAffine]()
        for i in 0..<batchN {
            let di = secpFrFromInt(UInt64(100 + i))
            let qi = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(di)))
            let ki = secpFrFromInt(UInt64(1000 + i * 7))
            let zi = secpFrFromInt(UInt64(50000 + i * 13))
            let ri = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(ki)))
            var rm = secpToInt(ri.x)
            if gte256(rm, SecpFr.N) { (rm, _) = sub256(rm, SecpFr.N) }
            let rf = secpFrFromRaw(rm)
            let sf = secpFrMul(secpFrInverse(ki), secpFrAdd(zi, secpFrMul(rf, di)))
            sigs.append(ECDSASignature(r: rf, s: sf, z: zi)); pks.append(qi)
        }
        let results = try engine.batchVerify(signatures: sigs, pubkeys: pks)
        expect(results.allSatisfy { $0 }, "Batch verify 16 valid")
        var bad = sigs
        bad[batchN / 2] = ECDSASignature(r: sigs[batchN / 2].r, s: sigs[batchN / 2].s, z: secpFrFromInt(99999))
        let badR = try engine.batchVerify(signatures: bad, pubkeys: pks)
        expect(!badR[batchN / 2], "Batch detect invalid")
    } catch { expect(false, "ECDSA error: \(error)") }

    suite("Radix Sort")
    do {
        let engine = try RadixSortEngine()
        expect(try engine.sort([1, 2, 3, 4, 5, 6, 7, 8]) == [1, 2, 3, 4, 5, 6, 7, 8], "Already sorted")
        expect(try engine.sort([8, 7, 6, 5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5, 6, 7, 8], "Reverse sorted")
        expect(try engine.sort([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9], "Duplicates")
        expect(try engine.sort([UInt32]()).isEmpty, "Empty")
        expect(try engine.sort([42]) == [42], "Single")

        var rng: UInt64 = 0xDEAD_BEEF_CAFE
        var keys = [UInt32]()
        for _ in 0..<10000 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; keys.append(UInt32(truncatingIfNeeded: rng >> 32)) }
        expect(try engine.sort(keys) == keys.sorted(), "Random 10K")

        let (sk, sv) = try engine.sortKV(keys: [30, 10, 20, 40], values: [300, 100, 200, 400])
        expect(sk == [10, 20, 30, 40] && sv == [100, 200, 300, 400], "Key-value sort")
    } catch { expect(false, "Sort error: \(error)") }
}
