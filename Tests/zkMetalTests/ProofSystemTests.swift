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
            let Cbound = pointAdd(C, pointScalarMul(pointFromAffine(Q), v))
            let proof = try engine.createProof(a: a, b: bv)
            expect(engine.verify(commitment: Cbound, b: bv, innerProductValue: v, proof: proof), "IPA n=\(n)")
            let wrongV = frFromInt(999)
            let Cwrong = pointAdd(C, pointScalarMul(pointFromAffine(Q), wrongV))
            expect(!engine.verify(commitment: Cwrong, b: bv, innerProductValue: wrongV, proof: proof), "IPA reject n=\(n)")
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
