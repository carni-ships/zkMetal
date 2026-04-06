import zkMetal
import Foundation

// MARK: - GPU BLS Signature Engine Tests

public func runGPUBLSSignatureTests() {
    suite("GPU BLS Signature Engine — Initialization")

    let engine = GPUBLSSignatureEngine()

    // Test 1: Engine initializes successfully
    expect(true, "Engine initialized (GPU or CPU fallback)")

    // Test 2: Version is set correctly
    let ver = GPUBLSSignatureEngine.version
    expect(ver.version == "1.0.0", "Version is 1.0.0")

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Deterministic Key Generation")

    testDeterministicKeyPair(engine)
    testDifferentSeedsProduceDifferentKeys(engine)
    testPublicKeyFromKnownScalar(engine)
    testKeyPairFromBytes(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Random Key Generation")

    testRandomKeyPair(engine)
    testTwoRandomKeyPairsDiffer(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Sign and Verify")

    testSignAndVerifySingle(engine)
    testWrongMessageFails(engine)
    testWrongPublicKeyFails(engine)
    testSignWithDST(engine)
    testVerifyTimed(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Hash to Curve")

    testHashToCurveProducesValidPoint(engine)
    testDifferentMessagesHashDifferently(engine)
    testSameMessageHashesDeterministically(engine)
    testHashToCurveWithCustomDST(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Signature Aggregation")

    testAggregateTwoSignatures(engine)
    testAggregateSingleReturnsItself(engine)
    testAggregateEightSignatures(engine)
    testAggregationIsCommutative(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Aggregate Verification")

    testVerifyAggregateDistinctMessages(engine)
    testAggregateVerifyFailsWrongMessage(engine)
    testAggregateVerifyFailsSwappedMessages(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Fast Aggregate Verify")

    testFastAggregateVerify(engine)
    testFastAggregateVerifyMissingSigner(engine)
    testFastAggregateVerifyEmptyKeys(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Multi-Pairing Batch Verification")

    testBatchVerifyMultiplePairings(engine)
    testBatchVerifyWithInvalid(engine)
    testBatchVerifyEmpty(engine)
    testBatchVerifySingleEntry(engine)
    testBatchVerifyReturnsTiming(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Subgroup Checks")

    testG1GeneratorInSubgroup(engine)
    testG2GeneratorInSubgroup(engine)
    testDerivedG1InSubgroup(engine)
    testG1OnCurve(engine)
    testG2OnCurve(engine)
    testG1FullValidation(engine)
    testG2FullValidation(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Batch Subgroup Checks")

    testBatchG1SubgroupCheck(engine)
    testBatchG2SubgroupCheck(engine)
    testBatchG1EmptyReturnsEmpty(engine)
    testBatchG2EmptyReturnsEmpty(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Proof of Possession")

    testProofOfPossession(engine)
    testProofOfPossessionWrongKey(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Threshold Signatures")

    testLagrangeCoefficient(engine)
    testThresholdCombineShares(engine)
    testThresholdInsufficientShares(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Public Key Aggregation")

    testAggregatePublicKeys(engine)
    testAggregatePublicKeysEmpty(engine)
    testPublicKeyAggregationCommutative(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Serialization")

    testSerializeG1(engine)
    testSerializeG2(engine)
    testSerializationDeterministic(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Pairing Utilities")

    testPairingSingle(engine)
    testPairingCheck(engine)
    testMillerLoopAndFinalExp(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — Signature Validation")

    testIsValidSignature(engine)
    testIsValidPublicKey(engine)
    testSignDetailed(engine)

    // ---------------------------------------------------------------
    suite("GPU BLS Signature Engine — End-to-End Workflows")

    testEthereumValidatorWorkflow(engine)
    testDistinctMessageWorkflow(engine)
    testRogueKeyMitigationWorkflow(engine)
}

// MARK: - Key Generation Tests

private func testDeterministicKeyPair(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 42)
    expect(!kp.secretKey.isZero, "Secret key is non-zero")
    expect(engine.g1IsOnCurve(kp.publicKey), "Public key is on G1 curve")
}

private func testDifferentSeedsProduceDifferentKeys(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 1)
    let kp2 = engine.deterministicKeyPair(seed: 2)
    let pk1x = fp381ToInt(kp1.publicKey.x)
    let pk2x = fp381ToInt(kp2.publicKey.x)
    expect(pk1x != pk2x, "Different seeds produce different public keys")
}

private func testPublicKeyFromKnownScalar(_ engine: GPUBLSSignatureEngine) {
    let sk = fr381FromInt(7)
    let pk = engine.generatePublicKey(secretKey: sk)
    expect(engine.g1IsOnCurve(pk), "pk from scalar 7 is on curve")
    let gen = bls12381G1Generator()
    expect(fp381ToInt(pk.x) != fp381ToInt(gen.x),
           "pk from scalar 7 differs from generator")
}

private func testKeyPairFromBytes(_ engine: GPUBLSSignatureEngine) {
    let ikm = Array("test key material".utf8)
    let kp = engine.keyPairFromBytes(ikm)
    expect(!kp.secretKey.isZero, "Key from bytes: secret key non-zero")
    expect(engine.g1IsOnCurve(kp.publicKey), "Key from bytes: public key on curve")

    // Deterministic: same input gives same key
    let kp2 = engine.keyPairFromBytes(ikm)
    let sk1 = fr381ToInt(kp.secretKey)
    let sk2 = fr381ToInt(kp2.secretKey)
    expectEqual(sk1, sk2, "Key from bytes is deterministic")
}

// MARK: - Random Key Tests

private func testRandomKeyPair(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.generateKeyPair()
    expect(!kp.secretKey.isZero, "Random secret key is non-zero")
    expect(engine.g1IsOnCurve(kp.publicKey), "Random public key is on curve")
}

private func testTwoRandomKeyPairsDiffer(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.generateKeyPair()
    let kp2 = engine.generateKeyPair()
    let sk1 = fr381ToInt(kp1.secretKey)
    let sk2 = fr381ToInt(kp2.secretKey)
    expect(sk1 != sk2, "Two random key pairs differ")
}

// MARK: - Sign and Verify Tests

private func testSignAndVerifySingle(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 100)
    let msg = Array("hello BLS signature".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let valid = engine.verify(message: msg, signature: sig, publicKey: kp.publicKey)
    expect(valid, "Single signature verifies correctly")
}

private func testWrongMessageFails(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 200)
    let msg = Array("correct message".utf8)
    let wrongMsg = Array("wrong message".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let valid = engine.verify(message: wrongMsg, signature: sig, publicKey: kp.publicKey)
    expect(!valid, "Wrong message fails verification")
}

private func testWrongPublicKeyFails(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 300)
    let wrongKp = engine.deterministicKeyPair(seed: 301)
    let msg = Array("test message".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let valid = engine.verify(message: msg, signature: sig, publicKey: wrongKp.publicKey)
    expect(!valid, "Wrong public key fails verification")
}

private func testSignWithDST(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 310)
    let msg = Array("DST test message".utf8)
    let dst = Array("MY_APP_v1".utf8)
    let sig = engine.signWithDST(message: msg, secretKey: kp.secretKey, dst: dst)
    let valid = engine.verifyWithDST(message: msg, signature: sig,
                                      publicKey: kp.publicKey, dst: dst)
    expect(valid, "DST sign/verify works correctly")

    // Wrong DST should fail
    let wrongDst = Array("WRONG_DST".utf8)
    let invalid = engine.verifyWithDST(message: msg, signature: sig,
                                        publicKey: kp.publicKey, dst: wrongDst)
    expect(!invalid, "Wrong DST fails verification")
}

private func testVerifyTimed(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 320)
    let msg = Array("timed verify".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let result = engine.verifyTimed(message: msg, signature: sig, publicKey: kp.publicKey)
    expect(result.valid, "Timed verify returns valid")
    expect(result.elapsedMs > 0, "Timed verify has positive elapsed time")
}

// MARK: - Hash to Curve Tests

private func testHashToCurveProducesValidPoint(_ engine: GPUBLSSignatureEngine) {
    let msg = Array("hash me to G2".utf8)
    let hm = engine.hashToCurveG2(message: msg)
    expect(engine.g2IsOnCurve(hm), "Hash-to-curve result is on G2 curve")
}

private func testDifferentMessagesHashDifferently(_ engine: GPUBLSSignatureEngine) {
    let hm1 = engine.hashToCurveG2(message: Array("message A".utf8))
    let hm2 = engine.hashToCurveG2(message: Array("message B".utf8))
    let x1c0 = fp381ToInt(hm1.x.c0)
    let x2c0 = fp381ToInt(hm2.x.c0)
    expect(x1c0 != x2c0, "Different messages hash to different G2 points")
}

private func testSameMessageHashesDeterministically(_ engine: GPUBLSSignatureEngine) {
    let msg = Array("deterministic hash".utf8)
    let hm1 = engine.hashToCurveG2(message: msg)
    let hm2 = engine.hashToCurveG2(message: msg)
    let x1c0 = fp381ToInt(hm1.x.c0)
    let x2c0 = fp381ToInt(hm2.x.c0)
    expectEqual(x1c0, x2c0, "Same message hashes to same G2 point")
}

private func testHashToCurveWithCustomDST(_ engine: GPUBLSSignatureEngine) {
    let msg = Array("custom DST hash".utf8)
    let dst1 = Array("DST_A".utf8)
    let dst2 = Array("DST_B".utf8)
    let hm1 = engine.hashToCurveG2(message: msg, dst: dst1)
    let hm2 = engine.hashToCurveG2(message: msg, dst: dst2)
    let x1c0 = fp381ToInt(hm1.x.c0)
    let x2c0 = fp381ToInt(hm2.x.c0)
    expect(x1c0 != x2c0, "Different DSTs produce different hash points")
}

// MARK: - Signature Aggregation Tests

private func testAggregateTwoSignatures(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 10)
    let kp2 = engine.deterministicKeyPair(seed: 11)
    let sig1 = engine.sign(message: Array("msg1".utf8), secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: Array("msg2".utf8), secretKey: kp2.secretKey)
    let agg = engine.aggregateSignatures([sig1, sig2])
    expect(engine.g2IsOnCurve(agg), "Aggregated signature is on G2 curve")
}

private func testAggregateSingleReturnsItself(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 20)
    let sig = engine.sign(message: Array("solo".utf8), secretKey: kp.secretKey)
    let agg = engine.aggregateSignatures([sig])
    let sigX = fp381ToInt(sig.x.c0)
    let aggX = fp381ToInt(agg.x.c0)
    expectEqual(sigX, aggX, "Aggregate of 1 equals original")
}

private func testAggregateEightSignatures(_ engine: GPUBLSSignatureEngine) {
    var sigs = [G2Affine381]()
    for i in 0..<8 {
        let kp = engine.deterministicKeyPair(seed: UInt64(50 + i))
        sigs.append(engine.sign(message: Array("batch \(i)".utf8), secretKey: kp.secretKey))
    }
    let agg = engine.aggregateSignatures(sigs)
    expect(engine.g2IsOnCurve(agg), "Aggregate of 8 is on G2 curve")
}

private func testAggregationIsCommutative(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 30)
    let kp2 = engine.deterministicKeyPair(seed: 31)
    let kp3 = engine.deterministicKeyPair(seed: 32)
    let sig1 = engine.sign(message: Array("a".utf8), secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: Array("b".utf8), secretKey: kp2.secretKey)
    let sig3 = engine.sign(message: Array("c".utf8), secretKey: kp3.secretKey)

    let agg123 = engine.aggregateSignatures([sig1, sig2, sig3])
    let agg321 = engine.aggregateSignatures([sig3, sig2, sig1])
    let x123 = fp381ToInt(agg123.x.c0)
    let x321 = fp381ToInt(agg321.x.c0)
    expectEqual(x123, x321, "Aggregation is order-independent")
}

// MARK: - Aggregate Verification Tests

private func testVerifyAggregateDistinctMessages(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 70)
    let kp2 = engine.deterministicKeyPair(seed: 71)
    let msg1 = Array("message one".utf8)
    let msg2 = Array("message two".utf8)
    let sig1 = engine.sign(message: msg1, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg2, secretKey: kp2.secretKey)
    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.verifyAggregate(
        messages: [msg1, msg2],
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(valid, "Aggregate verification of 2 distinct-message sigs succeeds")
}

private func testAggregateVerifyFailsWrongMessage(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 80)
    let kp2 = engine.deterministicKeyPair(seed: 81)
    let msg1 = Array("good 1".utf8)
    let msg2 = Array("good 2".utf8)
    let badMsg = Array("bad msg".utf8)
    let sig1 = engine.sign(message: msg1, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg2, secretKey: kp2.secretKey)
    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.verifyAggregate(
        messages: [badMsg, msg2],
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(!valid, "Aggregate verify fails with wrong message")
}

private func testAggregateVerifyFailsSwappedMessages(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 82)
    let kp2 = engine.deterministicKeyPair(seed: 83)
    let msg1 = Array("first".utf8)
    let msg2 = Array("second".utf8)
    let sig1 = engine.sign(message: msg1, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg2, secretKey: kp2.secretKey)
    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.verifyAggregate(
        messages: [msg2, msg1],
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(!valid, "Aggregate verify fails with swapped messages")
}

// MARK: - Fast Aggregate Verify Tests

private func testFastAggregateVerify(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 90)
    let kp2 = engine.deterministicKeyPair(seed: 91)
    let kp3 = engine.deterministicKeyPair(seed: 92)
    let msg = Array("beacon block root".utf8)
    let sig1 = engine.sign(message: msg, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg, secretKey: kp2.secretKey)
    let sig3 = engine.sign(message: msg, secretKey: kp3.secretKey)
    let aggSig = engine.aggregateSignatures([sig1, sig2, sig3])

    let valid = engine.fastAggregateVerify(
        message: msg,
        publicKeys: [kp1.publicKey, kp2.publicKey, kp3.publicKey],
        aggregateSignature: aggSig
    )
    expect(valid, "Fast aggregate verify with 3 signers succeeds")
}

private func testFastAggregateVerifyMissingSigner(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 93)
    let kp2 = engine.deterministicKeyPair(seed: 94)
    let kp3 = engine.deterministicKeyPair(seed: 95)
    let msg = Array("consensus msg".utf8)
    let sig1 = engine.sign(message: msg, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg, secretKey: kp2.secretKey)
    // Only aggregate 2 sigs but verify with 3 keys
    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.fastAggregateVerify(
        message: msg,
        publicKeys: [kp1.publicKey, kp2.publicKey, kp3.publicKey],
        aggregateSignature: aggSig
    )
    expect(!valid, "Fast aggregate verify fails with missing signer")
}

private func testFastAggregateVerifyEmptyKeys(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 96)
    let sig = engine.sign(message: Array("empty".utf8), secretKey: kp.secretKey)
    let valid = engine.fastAggregateVerify(
        message: Array("empty".utf8),
        publicKeys: [],
        aggregateSignature: sig
    )
    expect(!valid, "Fast aggregate verify with empty keys returns false")
}

// MARK: - Multi-Pairing Batch Verification Tests

private func testBatchVerifyMultiplePairings(_ engine: GPUBLSSignatureEngine) {
    var entries: [(message: [UInt8], signature: G2Affine381, publicKey: G1Affine381)] = []
    for i in 0..<3 {
        let kp = engine.deterministicKeyPair(seed: UInt64(400 + i))
        let msg = Array("batch entry \(i)".utf8)
        let sig = engine.sign(message: msg, secretKey: kp.secretKey)
        entries.append((message: msg, signature: sig, publicKey: kp.publicKey))
    }

    let result = engine.batchVerifyMultiPairing(entries: entries)
    expect(result.allValid, "Multi-pairing batch verify of 3 valid sigs succeeds")
    expectEqual(result.perEntry.count, 3, "Batch result has 3 per-entry results")
    expect(result.perEntry.allSatisfy { $0 }, "All per-entry results are true")
    expect(result.pairingCount > 0, "Pairing count is positive")
}

private func testBatchVerifyWithInvalid(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 410)
    let kp2 = engine.deterministicKeyPair(seed: 411)
    let kp3 = engine.deterministicKeyPair(seed: 412)
    let msg1 = Array("valid 1".utf8)
    let msg2 = Array("valid 2".utf8)
    let msg3 = Array("valid 3".utf8)
    let sig1 = engine.sign(message: msg1, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg2, secretKey: kp2.secretKey)
    // sig3 signed with kp1 but verified with kp3 -> invalid
    let sig3 = engine.sign(message: msg3, secretKey: kp1.secretKey)

    let entries: [(message: [UInt8], signature: G2Affine381, publicKey: G1Affine381)] = [
        (message: msg1, signature: sig1, publicKey: kp1.publicKey),
        (message: msg2, signature: sig2, publicKey: kp2.publicKey),
        (message: msg3, signature: sig3, publicKey: kp3.publicKey),
    ]

    let result = engine.batchVerifyMultiPairing(entries: entries)
    expect(!result.allValid, "Multi-pairing batch detects invalid signature")
    expect(result.perEntry[0] == true, "First sig is valid in per-entry")
    expect(result.perEntry[1] == true, "Second sig is valid in per-entry")
    expect(result.perEntry[2] == false, "Third sig (wrong key) is invalid")
}

private func testBatchVerifyEmpty(_ engine: GPUBLSSignatureEngine) {
    let result = engine.batchVerifyMultiPairing(entries: [])
    expect(result.allValid, "Empty batch is valid")
    expect(result.perEntry.isEmpty, "Empty batch has empty per-entry")
    expectEqual(result.pairingCount, 0, "Empty batch has 0 pairings")
}

private func testBatchVerifySingleEntry(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 420)
    let msg = Array("single batch".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let result = engine.batchVerifyMultiPairing(entries: [
        (message: msg, signature: sig, publicKey: kp.publicKey)
    ])
    expect(result.allValid, "Single-entry batch is valid")
    expectEqual(result.perEntry.count, 1, "Single-entry batch has 1 result")
}

private func testBatchVerifyReturnsTiming(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 430)
    let msg = Array("timing".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let result = engine.batchVerifyMultiPairing(entries: [
        (message: msg, signature: sig, publicKey: kp.publicKey)
    ])
    expect(result.elapsedMs >= 0, "Batch verify has non-negative elapsed time")
}

// MARK: - Subgroup Check Tests

private func testG1GeneratorInSubgroup(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G1Generator()
    expect(engine.g1SubgroupCheck(gen), "G1 generator is in r-torsion subgroup")
}

private func testG2GeneratorInSubgroup(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G2Generator()
    expect(engine.g2SubgroupCheck(gen), "G2 generator is in r-torsion subgroup")
}

private func testDerivedG1InSubgroup(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 150)
    expect(engine.g1SubgroupCheck(kp.publicKey), "Derived public key is in subgroup")
}

private func testG1OnCurve(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G1Generator()
    expect(engine.g1IsOnCurve(gen), "G1 generator is on curve y^2 = x^3 + 4")
}

private func testG2OnCurve(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G2Generator()
    expect(engine.g2IsOnCurve(gen), "G2 generator is on twist curve")
}

private func testG1FullValidation(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G1Generator()
    expect(engine.g1Validate(gen), "G1 generator passes full validation")
}

private func testG2FullValidation(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G2Generator()
    expect(engine.g2Validate(gen), "G2 generator passes full validation")
}

// MARK: - Batch Subgroup Check Tests

private func testBatchG1SubgroupCheck(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G1Generator()
    let kp1 = engine.deterministicKeyPair(seed: 160)
    let kp2 = engine.deterministicKeyPair(seed: 161)
    let kp3 = engine.deterministicKeyPair(seed: 162)

    let results = engine.batchG1SubgroupCheck([gen, kp1.publicKey, kp2.publicKey, kp3.publicKey])
    expectEqual(results.count, 4, "Batch G1 check returns 4 results")
    expect(results.allSatisfy { $0 }, "All G1 points are in subgroup")
}

private func testBatchG2SubgroupCheck(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G2Generator()
    let kp1 = engine.deterministicKeyPair(seed: 170)
    let kp2 = engine.deterministicKeyPair(seed: 171)
    let sig1 = engine.sign(message: Array("sub1".utf8), secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: Array("sub2".utf8), secretKey: kp2.secretKey)

    let results = engine.batchG2SubgroupCheck([gen, sig1, sig2])
    expectEqual(results.count, 3, "Batch G2 check returns 3 results")
    expect(results.allSatisfy { $0 }, "All G2 points are in subgroup")
}

private func testBatchG1EmptyReturnsEmpty(_ engine: GPUBLSSignatureEngine) {
    let results = engine.batchG1SubgroupCheck([])
    expect(results.isEmpty, "Empty batch G1 check returns empty")
}

private func testBatchG2EmptyReturnsEmpty(_ engine: GPUBLSSignatureEngine) {
    let results = engine.batchG2SubgroupCheck([])
    expect(results.isEmpty, "Empty batch G2 check returns empty")
}

// MARK: - Proof of Possession Tests

private func testProofOfPossession(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 500)
    let pop = engine.generateProofOfPossession(secretKey: kp.secretKey)

    // PoP public key should match
    let popPkX = fp381ToInt(pop.publicKey.x)
    let kpPkX = fp381ToInt(kp.publicKey.x)
    expectEqual(popPkX, kpPkX, "PoP public key matches key pair")

    // PoP should verify
    expect(engine.verifyProofOfPossession(pop), "Proof of possession verifies")
}

private func testProofOfPossessionWrongKey(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 510)
    let kp2 = engine.deterministicKeyPair(seed: 511)
    let pop = engine.generateProofOfPossession(secretKey: kp1.secretKey)

    // Replace public key with a different one -> should fail
    let badPop = BLSProofOfPossession(publicKey: kp2.publicKey, proof: pop.proof)
    expect(!engine.verifyProofOfPossession(badPop),
           "PoP fails with wrong public key")
}

// MARK: - Threshold Signature Tests

private func testLagrangeCoefficient(_ engine: GPUBLSSignatureEngine) {
    // For 2-of-3 threshold at indices [1, 2, 3]:
    // lambda_1 = (2*3) / ((2-1)*(3-1)) = 6 / 2 = 3
    let lambda = engine.lagrangeCoefficient(index: 1, indices: [1, 2, 3])
    let lambdaLimbs = fr381ToInt(lambda)
    let expected = fr381ToInt(fr381FromInt(3))
    expectEqual(lambdaLimbs, expected, "Lagrange coefficient for index 1 in [1,2,3] is 3")
}

private func testThresholdCombineShares(_ engine: GPUBLSSignatureEngine) {
    // Create a 2-of-3 threshold scheme:
    // f(x) = sk + a1*x where sk is the secret, a1 is random
    // Shares: f(1), f(2), f(3) for signers at indices 1, 2, 3
    let sk = fr381FromInt(42)
    let a1 = fr381FromInt(7)  // coefficient

    // Secret key shares: f(i) = sk + a1 * i
    let sk1 = fr381Add(sk, fr381Mul(a1, fr381FromInt(1)))
    let sk2 = fr381Add(sk, fr381Mul(a1, fr381FromInt(2)))
    let sk3 = fr381Add(sk, fr381Mul(a1, fr381FromInt(3)))

    let msg = Array("threshold message".utf8)

    // Each signer creates a partial signature
    let sig1 = engine.sign(message: msg, secretKey: sk1)
    let sig2 = engine.sign(message: msg, secretKey: sk2)
    let sig3 = engine.sign(message: msg, secretKey: sk3)

    let shares = [
        BLSThresholdShare(index: 1, partialSignature: sig1),
        BLSThresholdShare(index: 2, partialSignature: sig2),
        BLSThresholdShare(index: 3, partialSignature: sig3),
    ]

    // Combine any 2 shares
    let combined = engine.combineThresholdShares(Array(shares[0..<2]), threshold: 2)
    expect(combined != nil, "Threshold combination succeeds with 2 shares")

    // The combined signature should verify against pk = [sk]G1
    if let combinedSig = combined {
        let pk = engine.generatePublicKey(secretKey: sk)
        let valid = engine.verify(message: msg, signature: combinedSig, publicKey: pk)
        expect(valid, "Threshold combined signature verifies against master public key")
    }
}

private func testThresholdInsufficientShares(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 600)
    let sig = engine.sign(message: Array("threshold".utf8), secretKey: kp.secretKey)
    let share = BLSThresholdShare(index: 1, partialSignature: sig)

    // Need 2 shares but only have 1
    let result = engine.combineThresholdShares([share], threshold: 2)
    expect(result == nil, "Threshold fails with insufficient shares")
}

// MARK: - Public Key Aggregation Tests

private func testAggregatePublicKeys(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 180)
    let kp2 = engine.deterministicKeyPair(seed: 181)
    let aggPk = engine.aggregatePublicKeys([kp1.publicKey, kp2.publicKey])
    expect(aggPk != nil, "Aggregated public key is non-nil")
    expect(engine.g1IsOnCurve(aggPk!), "Aggregated public key is on curve")
}

private func testAggregatePublicKeysEmpty(_ engine: GPUBLSSignatureEngine) {
    let aggPk = engine.aggregatePublicKeys([])
    expect(aggPk == nil, "Aggregate of empty keys returns nil")
}

private func testPublicKeyAggregationCommutative(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 190)
    let kp2 = engine.deterministicKeyPair(seed: 191)
    let kp3 = engine.deterministicKeyPair(seed: 192)
    let agg123 = engine.aggregatePublicKeys([kp1.publicKey, kp2.publicKey, kp3.publicKey])!
    let agg321 = engine.aggregatePublicKeys([kp3.publicKey, kp2.publicKey, kp1.publicKey])!
    let x123 = fp381ToInt(agg123.x)
    let x321 = fp381ToInt(agg321.x)
    expectEqual(x123, x321, "Public key aggregation is commutative")
}

// MARK: - Serialization Tests

private func testSerializeG1(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G1Generator()
    let bytes = engine.serializeG1(gen)
    expectEqual(bytes.count, 48, "G1 serialization is 48 bytes")
    expect(bytes.contains(where: { $0 != 0 }), "G1 serialization is non-zero")
}

private func testSerializeG2(_ engine: GPUBLSSignatureEngine) {
    let gen = bls12381G2Generator()
    let bytes = engine.serializeG2(gen)
    expectEqual(bytes.count, 96, "G2 serialization is 96 bytes")
    expect(bytes.contains(where: { $0 != 0 }), "G2 serialization is non-zero")
}

private func testSerializationDeterministic(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 700)
    let bytes1 = engine.serializeG1(kp.publicKey)
    let bytes2 = engine.serializeG1(kp.publicKey)
    expectEqual(bytes1, bytes2, "G1 serialization is deterministic")
}

// MARK: - Pairing Utility Tests

private func testPairingSingle(_ engine: GPUBLSSignatureEngine) {
    let g1 = bls12381G1Generator()
    let g2 = bls12381G2Generator()
    let e = engine.pairing(g1, g2)
    // e(G1, G2) should not be identity
    expect(!e.c0.c0.c0.isZero || !e.c1.c0.c0.isZero,
           "Pairing e(G1, G2) is non-trivial")
}

private func testPairingCheck(_ engine: GPUBLSSignatureEngine) {
    // e(pk, H(m)) * e(-G1, sig) == 1 for valid signature
    let kp = engine.deterministicKeyPair(seed: 800)
    let msg = Array("pairing check".utf8)
    let sig = engine.sign(message: msg, secretKey: kp.secretKey)
    let hm = engine.hashToCurveG2(message: msg)
    let gen = bls12381G1Generator()
    let negGen = g1_381NegateAffine(gen)

    let ok = engine.pairingCheck([(kp.publicKey, hm), (negGen, sig)])
    expect(ok, "Pairing check for valid signature succeeds")
}

private func testMillerLoopAndFinalExp(_ engine: GPUBLSSignatureEngine) {
    let g1 = bls12381G1Generator()
    let g2 = bls12381G2Generator()

    // miller(g1, g2) followed by final exp should equal pairing(g1, g2)
    let ml = engine.millerLoop(g1, g2)
    let result = engine.finalExponentiation(ml)
    let direct = engine.pairing(g1, g2)

    let eq = fp12_381Equal(result, direct)
    expect(eq, "Miller loop + final exp equals direct pairing")
}

// MARK: - Signature Validation Tests

private func testIsValidSignature(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 850)
    let sig = engine.sign(message: Array("valid sig".utf8), secretKey: kp.secretKey)
    expect(engine.isValidSignature(sig), "Valid signature passes validation")
}

private func testIsValidPublicKey(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 860)
    expect(engine.isValidPublicKey(kp.publicKey), "Valid public key passes validation")
}

private func testSignDetailed(_ engine: GPUBLSSignatureEngine) {
    let kp = engine.deterministicKeyPair(seed: 870)
    let msg = Array("detailed sign".utf8)
    let (sig, hashPt) = engine.signDetailed(message: msg, secretKey: kp.secretKey)

    expect(engine.g2IsOnCurve(sig), "Detailed sig is on G2 curve")
    expect(engine.g2IsOnCurve(hashPt), "Detailed hash point is on G2 curve")

    // sig should verify
    let valid = engine.verify(message: msg, signature: sig, publicKey: kp.publicKey)
    expect(valid, "Detailed signature verifies")

    // hashPt should match independent hash-to-curve
    let hmIndep = engine.hashToCurveG2(message: msg)
    let hpXc0 = fp381ToInt(hashPt.x.c0)
    let hmXc0 = fp381ToInt(hmIndep.x.c0)
    expectEqual(hpXc0, hmXc0, "Detailed hash point matches independent hash-to-curve")
}

// MARK: - End-to-End Workflow Tests

private func testEthereumValidatorWorkflow(_ engine: GPUBLSSignatureEngine) {
    // Simulate Ethereum beacon chain: 3 validators sign same block root
    let kp1 = engine.deterministicKeyPair(seed: 1000)
    let kp2 = engine.deterministicKeyPair(seed: 1001)
    let kp3 = engine.deterministicKeyPair(seed: 1002)

    let blockRoot = Array("0xdeadbeef".utf8)

    // Each validator signs
    let sig1 = engine.sign(message: blockRoot, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: blockRoot, secretKey: kp2.secretKey)
    let sig3 = engine.sign(message: blockRoot, secretKey: kp3.secretKey)

    // Aggregate signatures
    let aggSig = engine.aggregateSignatures([sig1, sig2, sig3])

    // Fast aggregate verify (same message)
    let valid = engine.fastAggregateVerify(
        message: blockRoot,
        publicKeys: [kp1.publicKey, kp2.publicKey, kp3.publicKey],
        aggregateSignature: aggSig
    )
    expect(valid, "Ethereum validator workflow: fast aggregate verify succeeds")
}

private func testDistinctMessageWorkflow(_ engine: GPUBLSSignatureEngine) {
    let kp1 = engine.deterministicKeyPair(seed: 2000)
    let kp2 = engine.deterministicKeyPair(seed: 2001)

    let msg1 = Array("transfer 100 ETH".utf8)
    let msg2 = Array("transfer 200 ETH".utf8)

    let sig1 = engine.sign(message: msg1, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg2, secretKey: kp2.secretKey)

    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.verifyAggregate(
        messages: [msg1, msg2],
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(valid, "Distinct message workflow: aggregate verify succeeds")

    // Tampered messages should fail
    let tampered = engine.verifyAggregate(
        messages: [msg2, msg1],
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(!tampered, "Distinct message workflow: swapped messages fail")
}

private func testRogueKeyMitigationWorkflow(_ engine: GPUBLSSignatureEngine) {
    // PoP-based rogue key mitigation:
    // 1. Each signer generates a PoP
    // 2. Verifier checks all PoPs before accepting aggregate

    let kp1 = engine.deterministicKeyPair(seed: 3000)
    let kp2 = engine.deterministicKeyPair(seed: 3001)

    let pop1 = engine.generateProofOfPossession(secretKey: kp1.secretKey)
    let pop2 = engine.generateProofOfPossession(secretKey: kp2.secretKey)

    // Verify PoPs
    expect(engine.verifyProofOfPossession(pop1), "PoP 1 verifies")
    expect(engine.verifyProofOfPossession(pop2), "PoP 2 verifies")

    // Now safe to aggregate
    let msg = Array("safe aggregate".utf8)
    let sig1 = engine.sign(message: msg, secretKey: kp1.secretKey)
    let sig2 = engine.sign(message: msg, secretKey: kp2.secretKey)
    let aggSig = engine.aggregateSignatures([sig1, sig2])

    let valid = engine.fastAggregateVerify(
        message: msg,
        publicKeys: [kp1.publicKey, kp2.publicKey],
        aggregateSignature: aggSig
    )
    expect(valid, "Rogue key mitigation workflow: aggregate with PoP succeeds")
}
