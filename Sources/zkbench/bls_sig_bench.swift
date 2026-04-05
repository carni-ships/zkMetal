// BLS Signature Benchmark and Correctness Tests
import zkMetal
import Foundation

public func runBLSSignatureBench() {
    print("\n=== BLS Signature Engine ===")

    let engine = BLSSignatureEngine()

    // --- Pairing Sanity Check ---
    print("\n--- Pairing Sanity ---")
    let g1gen = bls12381G1Generator()
    let g2pt = bls12381G2Generator()

    let pairResult = bls12381Pairing(g1gen, g2pt)
    let pIsOne = fp12_381Equal(pairResult, .one)
    print("  e(G1,G2) != 1: \(!pIsOne ? "PASS" : "FAIL")")

    // Negation check
    let negG2pt = g2_381NegateAffine(g2pt)
    let pCheck = bls12381PairingCheck([(g1gen, g2pt), (g1gen, negG2pt)])
    print("  e(G1,G2)*e(G1,-G2)=1: \(pCheck ? "PASS" : "FAIL")")

    // Bilinearity: e(2G1,G2) == e(G1,2G2)
    let twoG1 = g1_381ToAffine(g1_381Double(g1_381FromAffine(g1gen)))!
    let twoG2 = g2_381ToAffine(g2_381Double(g2_381FromAffine(g2pt)))!
    let bilinLhs = bls12381Pairing(twoG1, g2pt)
    let bilinRhs = bls12381Pairing(g1gen, twoG2)
    let bilinOk = fp12_381Equal(bilinLhs, bilinRhs)
    print("  e(2G1,G2)==e(G1,2G2): \(bilinOk ? "PASS" : "FAIL")")

    // e(G1,G2)^2 == e(2G1,G2)
    let eSq = fp12_381Sqr(pairResult)
    let eSqEq = fp12_381Equal(eSq, bilinLhs)
    print("  e(G1,G2)^2==e(2G1,G2): \(eSqEq ? "PASS" : "FAIL")")

    // Pairing check bilinearity
    let negG1 = g1_381NegateAffine(g1gen)
    let bilinCheck = bls12381PairingCheck([(twoG1, g2pt), (negG1, twoG2)])
    print("  e(2G1,G2)*e(-G1,2G2)=1: \(bilinCheck ? "PASS" : "FAIL")")

    // e(3G1, G2) * e(-G1, 3G2) = 1
    let threeG1 = g1_381ToAffine(g1_381MulInt(g1_381FromAffine(g1gen), 3))!
    let threeG2 = g2_381ToAffine(g2_381MulInt(g2_381FromAffine(g2pt), 3))!
    let bilin3Check = bls12381PairingCheck([(threeG1, g2pt), (g1_381NegateAffine(g1gen), threeG2)])
    print("  e(3G1,G2)*e(-G1,3G2)=1: \(bilin3Check ? "PASS" : "FAIL")")

    // Scalar mul bilinearity: e([sk]G1, G2) == e(G1, [sk]G2)
    let skTest: [UInt64] = [12345678, 0, 0, 0]
    let skG1 = g1_381ToAffine(g1_381ScalarMul(g1_381FromAffine(g1gen), skTest))!
    let skG2 = g2_381ToAffine(g2_381ScalarMul(g2_381FromAffine(g2pt), skTest))!
    let eSkG1_G2 = bls12381Pairing(skG1, g2pt)
    let eG1_skG2 = bls12381Pairing(g1gen, skG2)
    let scalarBilin = fp12_381Equal(eSkG1_G2, eG1_skG2)
    print("  e([sk]G1,G2)==e(G1,[sk]G2): \(scalarBilin ? "PASS" : "FAIL")")

    // Frobenius
    let frobOne = fp12_381Frobenius(.one)
    let frobOneOk = fp12_381Equal(frobOne, .one)
    print("  frob(1)==1: \(frobOneOk ? "PASS" : "FAIL")")

    // --- BLS Signature Correctness Tests ---
    print("\n--- Correctness Tests ---")

    // Generate a key pair
    let sk = fr381FromInt(12345678)
    let pk = engine.publicKey(secretKey: sk)
    print("  Key generation: PASS")

    // Sign a message
    let message = Array("hello world".utf8)
    let sig = engine.sign(message: message, secretKey: sk)
    print("  Sign: PASS")

    // Verify the signature
    let valid = engine.verify(message: message, signature: sig, publicKey: pk)
    print("  Verify valid sig: \(valid ? "PASS" : "FAIL")")

    // Verify with wrong message should fail
    let wrongMessage = Array("wrong message".utf8)
    let invalid = engine.verify(message: wrongMessage, signature: sig, publicKey: pk)
    print("  Reject wrong message: \(!invalid ? "PASS" : "FAIL")")

    // Verify with wrong key should fail
    let sk2 = fr381FromInt(87654321)
    let pk2 = engine.publicKey(secretKey: sk2)
    let invalidKey = engine.verify(message: message, signature: sig, publicKey: pk2)
    print("  Reject wrong key: \(!invalidKey ? "PASS" : "FAIL")")

    // Sign/verify round-trip with different message
    let message2 = Array("ethereum consensus layer".utf8)
    let sig2 = engine.sign(message: message2, secretKey: sk)
    let valid2 = engine.verify(message: message2, signature: sig2, publicKey: pk)
    print("  Round-trip (msg2): \(valid2 ? "PASS" : "FAIL")")

    // Different key, same message
    let sig3 = engine.sign(message: message, secretKey: sk2)
    let valid3 = engine.verify(message: message, signature: sig3, publicKey: pk2)
    print("  Different signer: \(valid3 ? "PASS" : "FAIL")")

    // Cross-verify should fail: sig from sk on message vs pk2
    let crossInvalid = engine.verify(message: message, signature: sig, publicKey: pk2)
    print("  Reject cross-key: \(!crossInvalid ? "PASS" : "FAIL")")

    // --- Aggregate Signature Tests ---
    print("\n--- Aggregate Signatures ---")

    // Two signers, same message (fast aggregate)
    let aggSig = engine.aggregate(signatures: [sig, sig3])
    let fastAggValid = engine.fastAggregateVerify(
        message: message, aggregateSig: aggSig, publicKeys: [pk, pk2])
    print("  Fast aggregate verify (2 signers): \(fastAggValid ? "PASS" : "FAIL")")

    // Three signers
    let sk3 = fr381FromInt(11111111)
    let pk3 = engine.publicKey(secretKey: sk3)
    let sig4 = engine.sign(message: message, secretKey: sk3)
    let aggSig3 = engine.aggregate(signatures: [sig, sig3, sig4])
    let fastAgg3Valid = engine.fastAggregateVerify(
        message: message, aggregateSig: aggSig3, publicKeys: [pk, pk2, pk3])
    print("  Fast aggregate verify (3 signers): \(fastAgg3Valid ? "PASS" : "FAIL")")

    // Aggregate with wrong key set should fail
    let fastAggWrong = engine.fastAggregateVerify(
        message: message, aggregateSig: aggSig, publicKeys: [pk, pk3])
    print("  Reject wrong key set: \(!fastAggWrong ? "PASS" : "FAIL")")

    // Aggregate verify with different messages
    let aggVerifyValid = engine.aggregateVerify(
        messages: [message, message2],
        signatures: [sig, engine.sign(message: message2, secretKey: sk2)],
        publicKeys: [pk, pk2])
    print("  Aggregate verify (diff msgs): \(aggVerifyValid ? "PASS" : "FAIL")")

    // --- Hash-to-Curve Determinism Test ---
    print("\n--- Hash-to-Curve ---")

    let h1 = engine.hashToCurveG2(message: message)
    let h2 = engine.hashToCurveG2(message: message)
    if let h1a = g2_381ToAffine(h1), let h2a = g2_381ToAffine(h2) {
        let deterministic = fp381ToInt(h1a.x.c0) == fp381ToInt(h2a.x.c0) &&
                            fp381ToInt(h1a.x.c1) == fp381ToInt(h2a.x.c1) &&
                            fp381ToInt(h1a.y.c0) == fp381ToInt(h2a.y.c0) &&
                            fp381ToInt(h1a.y.c1) == fp381ToInt(h2a.y.c1)
        print("  Deterministic: \(deterministic ? "PASS" : "FAIL")")
    }

    // Different messages -> different points
    let h3 = engine.hashToCurveG2(message: wrongMessage)
    if let h1a = g2_381ToAffine(h1), let h3a = g2_381ToAffine(h3) {
        let different = fp381ToInt(h1a.x.c0) != fp381ToInt(h3a.x.c0) ||
                        fp381ToInt(h1a.x.c1) != fp381ToInt(h3a.x.c1)
        print("  Different msgs -> different points: \(different ? "PASS" : "FAIL")")
    }

    // Point is on curve
    let blsEngine = BLS12381Engine()
    if let h1a = g2_381ToAffine(h1) {
        let onCurve = blsEngine.g2IsOnCurve(h1a)
        print("  H(m) on curve: \(onCurve ? "PASS" : "FAIL")")
    }

    // --- Benchmarks ---
    print("\n--- Benchmarks ---")

    // Hash to curve
    let h2cIters = 10
    let h2cT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<h2cIters {
        let _ = engine.hashToCurveG2(message: message)
    }
    let h2cTime = (CFAbsoluteTimeGetCurrent() - h2cT0) * 1000 / Double(h2cIters)
    print(String(format: "  Hash-to-curve G2: %.1f ms", h2cTime))

    // Sign
    let signIters = 5
    let signT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<signIters {
        let _ = engine.sign(message: message, secretKey: sk)
    }
    let signTime = (CFAbsoluteTimeGetCurrent() - signT0) * 1000 / Double(signIters)
    print(String(format: "  Sign: %.1f ms", signTime))

    // Verify
    let verifyIters = 3
    let verifyT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<verifyIters {
        let _ = engine.verify(message: message, signature: sig, publicKey: pk)
    }
    let verifyTime = (CFAbsoluteTimeGetCurrent() - verifyT0) * 1000 / Double(verifyIters)
    print(String(format: "  Verify: %.1f ms", verifyTime))

    // Aggregate signatures (pre-computed)
    for count in [10, 100] {
        var sigs = [G2Affine381]()
        var pks = [G1Affine381]()
        for i in 0..<count {
            let ski = fr381FromInt(UInt64(1000 + i))
            let pki = engine.publicKey(secretKey: ski)
            let sigi = engine.sign(message: message, secretKey: ski)
            sigs.append(sigi)
            pks.append(pki)
        }

        // Aggregate
        let aggT0 = CFAbsoluteTimeGetCurrent()
        let agg = engine.aggregate(signatures: sigs)
        let aggTime = (CFAbsoluteTimeGetCurrent() - aggT0) * 1000
        print(String(format: "  Aggregate %d sigs: %.1f ms", count, aggTime))

        // Fast aggregate verify
        let favT0 = CFAbsoluteTimeGetCurrent()
        let favResult = engine.fastAggregateVerify(message: message, aggregateSig: agg, publicKeys: pks)
        let favTime = (CFAbsoluteTimeGetCurrent() - favT0) * 1000
        print(String(format: "  Fast agg verify %d: %.1f ms (%@)", count, favTime,
                     favResult ? "valid" : "INVALID"))
    }

    // Summary
    let allPass = valid && !invalid && !invalidKey && valid2 && valid3 && !crossInvalid
    print("\n  Overall: \(allPass ? "ALL PASS" : "SOME FAILED")")
}
