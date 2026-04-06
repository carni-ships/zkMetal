import zkMetal

// MARK: - Multi-Scalar Inner Product Tests

public func runMultiScalarInnerProductTests() {
    suite("Multi-Scalar Inner Product Engine")

    // Test 1: Field inner product correctness (small)
    do {
        let engine = try! MultiScalarInnerProduct()

        let a: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let b: [Fr] = [frFromInt(2), frFromInt(4), frFromInt(6), frFromInt(8)]
        // <a, b> = 3*2 + 5*4 + 7*6 + 11*8 = 6 + 20 + 42 + 88 = 156
        let result = engine.fieldInnerProduct(a: a, b: b)
        let expected = frFromInt(156)
        expect(frEqual(result, expected), "fieldInnerProduct small vectors: 156")
    }

    // Test 2: Field inner product with zeros
    do {
        let engine = try! MultiScalarInnerProduct()

        let a: [Fr] = [frFromInt(1), frFromInt(0), frFromInt(3), frFromInt(0)]
        let b: [Fr] = [frFromInt(0), frFromInt(2), frFromInt(0), frFromInt(4)]
        // <a, b> = 0 + 0 + 0 + 0 = 0
        let result = engine.fieldInnerProduct(a: a, b: b)
        expect(frEqual(result, Fr.zero), "fieldInnerProduct with zeros: 0")
    }

    // Test 3: ipaFold correctness
    do {
        let engine = try! MultiScalarInnerProduct()

        // a = [1, 2, 3, 4], challenge x = 5
        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let b: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let x = frFromInt(5)

        let (aFolded, bFolded) = engine.ipaFold(a: a, b: b, challenge: x)

        // a'[0] = 1 + 5*3 = 16, a'[1] = 2 + 5*4 = 22
        expect(aFolded.count == 2, "ipaFold output length is n/2")
        expect(frEqual(aFolded[0], frFromInt(16)), "ipaFold a[0] = 1 + 5*3 = 16")
        expect(frEqual(aFolded[1], frFromInt(22)), "ipaFold a[1] = 2 + 5*4 = 22")

        // b is folded with x^{-1}, verify it's halved correctly
        expect(bFolded.count == 2, "ipaFold b output length is n/2")

        // Verify: b'[i] = b[i] + x^{-1} * b[2+i]
        let xInv = frInverse(x)
        let bExpected0 = frAdd(b[0], frMul(xInv, b[2]))
        let bExpected1 = frAdd(b[1], frMul(xInv, b[3]))
        expect(frEqual(bFolded[0], bExpected0), "ipaFold b[0] correct")
        expect(frEqual(bFolded[1], bExpected1), "ipaFold b[1] correct")
    }

    // Test 4: foldVector single vector
    do {
        let engine = try! MultiScalarInnerProduct()

        let v: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let challenge = frFromInt(3)
        let folded = engine.foldVector(v, challenge: challenge)

        // folded[0] = 10 + 3*30 = 100, folded[1] = 20 + 3*40 = 140
        expect(folded.count == 2, "foldVector output length")
        expect(frEqual(folded[0], frFromInt(100)), "foldVector [0] = 10 + 3*30 = 100")
        expect(frEqual(folded[1], frFromInt(140)), "foldVector [1] = 20 + 3*40 = 140")
    }

    // Test 5: IPA prove and verify (n=4)
    do {
        let n = 4
        let (gens, Q) = IPAEngine.generateTestGenerators(count: n + 1)
        let G = Array(gens.prefix(n))
        let qAffine = gens[n]

        let engine = try! MultiScalarInnerProduct()

        let a: [Fr] = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]
        let b: [Fr] = [frFromInt(2), frFromInt(4), frFromInt(6), frFromInt(8)]

        let (proof, commitment, ip) = engine.ipaProve(a: a, b: b, G: G, Q: qAffine)

        // Verify the inner product value: 3*2 + 5*4 + 7*6 + 11*8 = 156
        expect(frEqual(ip, frFromInt(156)), "IPA inner product = 156")

        // Verify the proof
        let valid = engine.ipaVerify(proof: proof, commitment: commitment,
                                     innerProduct: ip, G: G, Q: qAffine)
        expect(valid, "IPA prove/verify n=4")
    }

    // Test 6: IPA prove and verify (n=8)
    do {
        let n = 8
        let (gens, _) = IPAEngine.generateTestGenerators(count: n + 1)
        let G = Array(gens.prefix(n))
        let qAffine = gens[n]

        let engine = try! MultiScalarInnerProduct()

        var a = [Fr]()
        var b = [Fr]()
        for i in 0..<n {
            a.append(frFromInt(UInt64(i + 1)))
            b.append(frFromInt(UInt64(n - i)))
        }

        let (proof, commitment, ip) = engine.ipaProve(a: a, b: b, G: G, Q: qAffine)

        // <a, b> = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        expect(frEqual(ip, frFromInt(120)), "IPA inner product n=8: 120")

        let valid = engine.ipaVerify(proof: proof, commitment: commitment,
                                     innerProduct: ip, G: G, Q: qAffine)
        expect(valid, "IPA prove/verify n=8")
    }

    // Test 7: IPA should fail with wrong inner product
    do {
        let n = 4
        let (gens, _) = IPAEngine.generateTestGenerators(count: n + 1)
        let G = Array(gens.prefix(n))
        let qAffine = gens[n]

        let engine = try! MultiScalarInnerProduct()

        let a: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let b: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

        let (proof, _, _) = engine.ipaProve(a: a, b: b, G: G, Q: qAffine)

        // Try to verify with wrong inner product value
        let wrongIP = frFromInt(999)
        // Recompute the commitment with the wrong IP to create a mismatch
        let wrongCommitment = pointIdentity()  // clearly wrong
        let invalid = engine.ipaVerify(proof: proof, commitment: wrongCommitment,
                                       innerProduct: wrongIP, G: G, Q: qAffine)
        expect(!invalid, "IPA rejects wrong commitment/inner product")
    }

    // Test 8: IPA prove/verify with n=16
    do {
        let n = 16
        let (gens, _) = IPAEngine.generateTestGenerators(count: n + 1)
        let G = Array(gens.prefix(n))
        let qAffine = gens[n]

        let engine = try! MultiScalarInnerProduct()

        var a = [Fr]()
        var b = [Fr]()
        for i in 0..<n {
            a.append(frFromInt(UInt64(i * 3 + 1)))
            b.append(frFromInt(UInt64(i * 2 + 1)))
        }

        let (proof, commitment, ip) = engine.ipaProve(a: a, b: b, G: G, Q: qAffine)

        let valid = engine.ipaVerify(proof: proof, commitment: commitment,
                                     innerProduct: ip, G: G, Q: qAffine)
        expect(valid, "IPA prove/verify n=16")
    }

    // Test 9: Field inner product commutativity
    do {
        let engine = try! MultiScalarInnerProduct()

        let a: [Fr] = [frFromInt(7), frFromInt(13), frFromInt(19), frFromInt(23)]
        let b: [Fr] = [frFromInt(3), frFromInt(11), frFromInt(17), frFromInt(29)]

        let ab = engine.fieldInnerProduct(a: a, b: b)
        let ba = engine.fieldInnerProduct(a: b, b: a)
        expect(frEqual(ab, ba), "fieldInnerProduct is commutative")
    }

    // Test 10: Proof size is log(n)
    do {
        let n = 16
        let (gens, _) = IPAEngine.generateTestGenerators(count: n + 1)
        let G = Array(gens.prefix(n))
        let qAffine = gens[n]

        let engine = try! MultiScalarInnerProduct()

        var a = [Fr]()
        var b = [Fr]()
        for i in 0..<n {
            a.append(frFromInt(UInt64(i + 1)))
            b.append(frFromInt(UInt64(1)))
        }

        let (proof, _, _) = engine.ipaProve(a: a, b: b, G: G, Q: qAffine)

        // log2(16) = 4 rounds
        expectEqual(proof.Ls.count, 4, "Proof has log2(n)=4 L commitments")
        expectEqual(proof.Rs.count, 4, "Proof has log2(n)=4 R commitments")
    }

    print("  Multi-Scalar Inner Product tests done")
}
