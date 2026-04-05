import zkMetal

func runVerkleProofTests() {
    suite("Banderwagon Curve")

    // Test 1: Generator is on curve
    let genAff = banderwagonGenerator
    expect(bwIsOnCurve(genAff), "Generator on curve")

    // Test 2: Identity check
    let id = BanderwagonExtended.identity
    expect(bwIsIdentity(id), "Identity is identity")
    expect(!bwIsIdentity(bwFromAffine(genAff)), "Generator is not identity")

    // Test 3: Point addition and doubling consistency
    let g = bwFromAffine(genAff)
    let g2_add = bwAdd(g, g)
    let g2_dbl = bwDouble(g)
    expect(bwEqual(g2_add, g2_dbl), "Add self == Double")

    // Test 4: Addition with identity
    let gPlusId = bwAdd(g, id)
    expect(bwEqual(gPlusId, g), "G + Identity == G")

    // Test 5: Scalar multiplication
    let g3_scalar = bwScalarMul(g, fr381FromInt(3))
    let g3_add = bwAdd(bwAdd(g, g), g)
    expect(bwEqual(g3_scalar, g3_add), "3*G == G+G+G")

    // Test 6: Negation
    let negG = bwNegate(g)
    let gPlusNeg = bwAdd(g, negG)
    expect(bwIsIdentity(gPlusNeg), "G + (-G) == Identity")

    // Test 7: Serialization round-trip
    let serialized = bwSerialize(g)
    expect(serialized.count == 32, "Serialized point is 32 bytes")
    if let deserialized = bwDeserialize(serialized) {
        expect(bwEqual(deserialized, g), "Serialize-deserialize round-trip")
    } else {
        expect(false, "Deserialization failed")
    }

    // Test 8: Map to field
    let fieldVal = bwMapToField(g)
    expect(!fieldVal.isZero, "Map to field non-zero")
    let fieldValId = bwMapToField(id)
    expect(fieldValId.isZero, "Map to field of identity is zero")

    // Test 9: Batch to affine
    let points = [g, g2_dbl, g3_scalar]
    let affines = bwBatchToAffine(points)
    expect(affines.count == 3, "Batch to affine count")
    for i in 0..<3 {
        let reconverted = bwFromAffine(affines[i])
        expect(bwEqual(reconverted, points[i]), "Batch to affine point \(i)")
    }

    // Test: scalar mul of 1
    let gTimes1 = bwScalarMul(g, Fr381.one)
    expect(bwEqual(gTimes1, g), "1*G == G")

    // Test: scalar mul of 2
    let gTimes2 = bwScalarMul(g, fr381FromInt(2))
    expect(bwEqual(gTimes2, bwDouble(g)), "2*G == Double(G)")

    // Test: scalar mul with large number
    let s100 = fr381FromInt(100)
    let g100a = bwScalarMul(g, s100)
    var g100b = BanderwagonExtended.identity
    for _ in 0..<100 { g100b = bwAdd(g100b, g) }
    expect(bwEqual(g100a, g100b), "100*G == G+G+...+G")

    // Test: scalar mul distributivity
    let s5 = fr381FromInt(5)
    let g1sum = bwAdd(g, bwDouble(g))  // g + 2g = 3g
    let scaledSum = bwScalarMul(g1sum, s5)  // 5 * 3g = 15g
    let direct15 = bwScalarMul(g, fr381FromInt(15))
    expect(bwEqual(scaledSum, direct15), "5*(G+2G) == 15G")

    // Test: scalar mul with sum of two points
    let g0 = g
    let g1pt = bwDouble(g)  // 2G
    let sum01 = bwAdd(g0, g1pt)  // 3G
    let s7 = fr381FromInt(7)
    let scaledSumResult = bwScalarMul(sum01, s7)  // 7*3G = 21G
    let distResult = bwAdd(bwScalarMul(g0, s7), bwScalarMul(g1pt, s7))  // 7G + 14G = 21G
    expect(bwEqual(scaledSumResult, distResult), "Scalar mul distributes over add")

    // Test: large scalar associativity using BwScalar (mod q)
    // Check: (a*b mod q)*G == a*(b*G) using BwScalar arithmetic
    let bigAq = bwScalarInverse(bwScalarFromInt(2))  // (q+1)/2 mod q
    let bigBq = bwScalarFromInt(20)
    let bigABq = bwScalarMul(bigAq, bigBq)  // (q+1)/2 * 20 mod q = 10
    let result1q = bwScalarMulQ(g, bigABq)   // should be 10*G
    let result2q = bwScalarMulQ(bwScalarMulQ(g, bigBq), bigAq)  // bigA * (20*G)
    let result3q = bwScalarMulQ(g, bwScalarFromInt(10))  // 10*G directly

    // Check: 2*(10*G) == 20*G
    let g10q = bwScalarMulQ(g, bwScalarFromInt(10))
    let g20q = bwScalarMulQ(g, bwScalarFromInt(20))
    let g10x2q = bwScalarMulQ(g10q, bwScalarFromInt(2))
    expect(bwEqual(g10x2q, g20q), "2*(10*G) == 20*G (BwScalar)")

    // Check: 2 * ((q+1)/2 * G) == G
    let bigAGq = bwScalarMulQ(g, bigAq)
    let twoBigAGq = bwScalarMulQ(bigAGq, bwScalarFromInt(2))
    expect(bwEqual(twoBigAGq, g), "2 * ((q+1)/2 * G) == G")

    // Check: (q+1)/2 * (2*G) == G
    let g2q = bwDouble(g)
    let bigAG2q = bwScalarMulQ(g2q, bigAq)
    expect(bwEqual(bigAG2q, g), "(q+1)/2 * 2G == G")

    // Verify q*G == identity
    let qScalar = BwScalar(banderwagonOrder[0], banderwagonOrder[1],
                            banderwagonOrder[2], banderwagonOrder[3])
    let qG = bwScalarMulQ(g, qScalar)
    expect(bwIsIdentity(qG), "q*G == identity")

    expect(bwEqual(result1q, result3q), "(a*b)*G == 10*G where a*b=10 (BwScalar)")
    expect(bwEqual(result2q, result3q), "a*(b*G) == 10*G (BwScalar)")
    expect(bwEqual(result1q, result2q), "(a*b)*G == a*(b*G) (BwScalar)")

    // Test: Manual IPA-like check with known x=2 (using BwScalar)
    do {
        let eng2 = BanderwagonIPAEngine(width: 2)
        let G0 = eng2.generators[0]
        let G1 = eng2.generators[1]
        let QQ = eng2.Q
        let s0 = bwScalarFromInt(10)
        let s1 = bwScalarFromInt(20)
        let CC = eng2.commit([s0, s1])
        let vv = bwScalarFromInt(10)  // <[10,20], [1,0]> = 10
        let Cb = bwAdd(CC, bwScalarMulQ(QQ, vv))

        let cL = BwScalar.zero   // <[10], [0]> = 0
        let cR = bwScalarFromInt(20) // <[20], [1]> = 20
        let LL = bwAdd(bwScalarMulQ(G1, s0), bwScalarMulQ(QQ, cL))  // 10*G1 + 0*Q
        let RR = bwAdd(bwScalarMulQ(G0, s1), bwScalarMulQ(QQ, cR))  // 20*G0 + 20*Q

        let xx = bwScalarFromInt(2)
        let xxi = bwScalarInverse(xx)
        let xx2 = bwScalarSqr(xx)
        let xxi2 = bwScalarSqr(xxi)

        let cprime = bwAdd(Cb, bwAdd(bwScalarMulQ(LL, xx2), bwScalarMulQ(RR, xxi2)))

        let af = bwScalarAdd(bwScalarMul(xx, s0), bwScalarMul(xxi, s1))
        let bf = xxi
        let gf0 = bwScalarMul(af, xxi)
        let gf1 = bwScalarMul(af, xx)
        let aGfinal = bwAdd(bwScalarMulQ(G0, gf0), bwScalarMulQ(G1, gf1))
        let abf = bwScalarMul(af, bf)
        let expectedVal = bwAdd(aGfinal, bwScalarMulQ(QQ, abf))

        // Debug each component
        let cLQ = bwScalarMulQ(QQ, cL)
        expect(bwIsIdentity(cLQ), "0*Q == identity")

        // Check LL = 10*G1 (since cL=0)
        let LL_manual = bwScalarMulQ(G1, s0)
        expect(bwEqual(LL, LL_manual), "L == 10*G1")

        // Check: does Cb == CC + vv*Q correctly?
        let CC_check = bwAdd(bwScalarMulQ(G0, s0), bwScalarMulQ(G1, s1))
        expect(bwEqual(CC_check, CC), "commit == 10*G0 + 20*G1")

        let Cb_check = bwAdd(CC_check, bwScalarMulQ(QQ, vv))
        expect(bwEqual(Cb_check, Cb), "Cbound check")

        // Check coefficient consistency
        let coeff_G0 = bwScalarAdd(bwScalarFromInt(10), bwScalarMul(bwScalarFromInt(20), xxi2))
        let coeff_G1 = bwScalarAdd(bwScalarFromInt(20), bwScalarMul(bwScalarFromInt(10), xx2))
        let coeff_Q = bwScalarAdd(bwScalarFromInt(10), bwScalarMul(bwScalarFromInt(20), xxi2))

        let cprime_manual = bwAdd(bwAdd(bwScalarMulQ(G0, coeff_G0),
                                         bwScalarMulQ(G1, coeff_G1)),
                                   bwScalarMulQ(QQ, coeff_Q))
        expect(bwEqual(cprime, cprime_manual), "cprime == manual decomposition")

        // gf0 should equal coeff_G0, gf1 should equal coeff_G1, abf should equal coeff_Q
        expect(gf0 == coeff_G0, "gf0 == coeff_G0")
        expect(gf1 == coeff_G1, "gf1 == coeff_G1")
        expect(abf == coeff_Q, "abf == coeff_Q")

        expect(bwEqual(cprime, expectedVal), "Manual IPA check x=2")
    }

    suite("Banderwagon IPA")

    // Test with small width for speed
    let width = 4
    let ipaEngine = BanderwagonIPAEngine(width: width)

    // Test 10: Commitment is non-trivial
    let vals: [BwScalar] = [bwScalarFromInt(10), bwScalarFromInt(20), bwScalarFromInt(30), bwScalarFromInt(40)]
    let C = ipaEngine.commit(vals)
    expect(!bwIsIdentity(C), "Commitment non-trivial")

    // Test 11: IPA proof creation and verification
    var b = [BwScalar](repeating: .zero, count: width)
    b[1] = .one  // open at index 1
    let v = BanderwagonIPAEngine.innerProduct(vals, b)
    // v should equal vals[1] = 20
    expect(v == bwScalarFromInt(20), "Inner product at index 1")

    let vQ = bwScalarMulQ(ipaEngine.Q, v)
    let Cbound = bwAdd(C, vQ)
    let proof = ipaEngine.createProof(a: vals, b: b)

    // Test: n=2 IPA proof
    let smallEngine2 = BanderwagonIPAEngine(width: 2)
    let smallVals: [BwScalar] = [bwScalarFromInt(10), bwScalarFromInt(20)]
    let smallC = smallEngine2.commit(smallVals)
    var smallB = [BwScalar](repeating: .zero, count: 2)
    smallB[0] = .one
    let smallV = BanderwagonIPAEngine.innerProduct(smallVals, smallB)
    let smallVQ = bwScalarMulQ(smallEngine2.Q, smallV)
    let smallCbound = bwAdd(smallC, smallVQ)
    let smallProof = smallEngine2.createProof(a: smallVals, b: smallB)
    let smallVerified = smallEngine2.verify(commitment: smallCbound, b: smallB,
                                            innerProductValue: smallV, proof: smallProof)
    expect(smallVerified, "IPA n=2 proof verified")

    let verified = ipaEngine.verify(commitment: Cbound, b: b, innerProductValue: v, proof: proof)
    expect(verified, "IPA proof verified")

    // Test 12: Wrong value should fail verification
    let wrongV = bwScalarFromInt(999)
    let CwrongBound = bwAdd(C, bwScalarMulQ(ipaEngine.Q, wrongV))
    let rejectedWrong = !ipaEngine.verify(commitment: CwrongBound, b: b,
                                           innerProductValue: wrongV, proof: proof)
    expect(rejectedWrong, "IPA rejects wrong value")

    // Test 13: Different opening index
    for idx in 0..<width {
        var bIdx = [BwScalar](repeating: .zero, count: width)
        bIdx[idx] = .one
        let vIdx = BanderwagonIPAEngine.innerProduct(vals, bIdx)
        let CboundIdx = bwAdd(C, bwScalarMulQ(ipaEngine.Q, vIdx))
        let proofIdx = ipaEngine.createProof(a: vals, b: bIdx)
        let ok = ipaEngine.verify(commitment: CboundIdx, b: bIdx,
                                  innerProductValue: vIdx, proof: proofIdx)
        expect(ok, "IPA proof index \(idx)")
    }

    suite("Verkle Tree Proofs")

    // Build a small Verkle tree with width-4 IPA
    let smallEngine = BanderwagonIPAEngine(width: 4)
    let tree = VerkleTree(ipaEngine: smallEngine)

    // Insert some key-value pairs
    // Keys are 32 bytes: first 31 = stem, last byte = suffix
    var key1 = [UInt8](repeating: 0, count: 32)
    key1[0] = 1; key1[31] = 0
    let val1 = fr381FromInt(42)

    var key2 = [UInt8](repeating: 0, count: 32)
    key2[0] = 2; key2[31] = 1
    let val2 = fr381FromInt(100)

    tree.insert(key: key1, value: val1)
    tree.insert(key: key2, value: val2)

    // Test 14: Lookup works
    if let got1 = tree.get(key: key1) {
        expect(fr381ToInt(got1) == fr381ToInt(val1), "Tree get key1")
    } else {
        expect(false, "Tree get key1 returned nil")
    }

    if let got2 = tree.get(key: key2) {
        expect(fr381ToInt(got2) == fr381ToInt(val2), "Tree get key2")
    } else {
        expect(false, "Tree get key2 returned nil")
    }

    // Test 15: Absent key returns nil
    var keyAbsent = [UInt8](repeating: 0, count: 32)
    keyAbsent[0] = 3; keyAbsent[31] = 2
    expect(tree.get(key: keyAbsent) == nil, "Tree get absent key is nil")

    // Test 16: Compute commitments
    tree.computeCommitments()
    let rootC = tree.rootCommitment()
    expect(!bwIsIdentity(rootC), "Root commitment non-trivial")

    // Test 17: Generate proof for existing key
    let proof1 = generateVerkleProof(tree: tree, key: key1)
    expect(proof1.extensionStatus == .present, "Proof key1 status is present")
    expect(!proof1.commitments.isEmpty, "Proof key1 has commitments")
    expect(!proof1.ipaProofs.isEmpty, "Proof key1 has IPA proofs")

    // Test 18: Generate proof for absent key
    let proofAbsent = generateVerkleProof(tree: tree, key: keyAbsent)
    expect(proofAbsent.extensionStatus == .absent, "Proof absent key status")

    // Test 19: Multi-proof generation
    let multiProof = generateVerkleMultiProof(tree: tree, keys: [key1, key2])
    expect(!multiProof.commitments.isEmpty, "Multi-proof has commitments")
    expect(multiProof.extensionStatuses.count == 2, "Multi-proof has 2 statuses")

    // Verkle proof tests complete
}
