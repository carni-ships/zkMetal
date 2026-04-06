import zkMetal

// MARK: - GPU BLS Aggregate Signature Tests

public func runGPUBLSAggregateTests() {
    suite("GPU BLS Aggregate Engine — Initialization")

    let engine = GPUBLSAggregateEngine()

    // Test 1: Engine initializes (GPU or CPU fallback)
    expect(true, "Engine initialized successfully")

    // Test 2: Version is set
    let ver = GPUBLSAggregateEngine.version
    expect(ver.version == "1.0.0", "Version is 1.0.0")

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Key Generation")

    // Test 3: Deterministic key pair generation
    do {
        let (sk, pk) = engine.deterministicKeyPair(seed: 42)
        expect(!sk.isZero, "Secret key is non-zero")
        expect(engine.g1IsOnCurve(pk), "Public key is on G1 curve")
    }

    // Test 4: Different seeds produce different keys
    do {
        let (_, pk1) = engine.deterministicKeyPair(seed: 1)
        let (_, pk2) = engine.deterministicKeyPair(seed: 2)
        let pk1x = fp381ToInt(pk1.x)
        let pk2x = fp381ToInt(pk2.x)
        expect(pk1x != pk2x, "Different seeds produce different public keys")
    }

    // Test 5: Public key from known scalar
    do {
        let sk = fr381FromInt(7)
        let pk = engine.generatePublicKey(secretKey: sk)
        expect(engine.g1IsOnCurve(pk), "pk from scalar 7 is on curve")

        // [7]G1 should not be identity
        let gen = bls12381G1Generator()
        expect(fp381ToInt(pk.x) != fp381ToInt(gen.x),
               "pk from scalar 7 differs from generator")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Sign and Verify")

    // Test 6: Sign and verify a single message
    do {
        let (sk, pk) = engine.deterministicKeyPair(seed: 100)
        let msg = Array("hello BLS".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        let valid = engine.verify(message: msg, signature: sig, publicKey: pk)
        expect(valid, "Single signature verifies correctly")
    }

    // Test 7: Wrong message fails verification
    do {
        let (sk, pk) = engine.deterministicKeyPair(seed: 200)
        let msg = Array("correct message".utf8)
        let wrongMsg = Array("wrong message".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        let valid = engine.verify(message: wrongMsg, signature: sig, publicKey: pk)
        expect(!valid, "Wrong message fails verification")
    }

    // Test 8: Wrong public key fails verification
    do {
        let (sk, _) = engine.deterministicKeyPair(seed: 300)
        let (_, wrongPk) = engine.deterministicKeyPair(seed: 301)
        let msg = Array("test message".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        let valid = engine.verify(message: msg, signature: sig, publicKey: wrongPk)
        expect(!valid, "Wrong public key fails verification")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Hash to Curve")

    // Test 9: Hash to curve produces valid G2 point
    do {
        let msg = Array("hash me to G2".utf8)
        let hm = engine.hashToCurveG2(message: msg)
        expect(engine.g2IsOnCurve(hm), "Hash-to-curve result is on G2 curve")
    }

    // Test 10: Different messages produce different curve points
    do {
        let hm1 = engine.hashToCurveG2(message: Array("message A".utf8))
        let hm2 = engine.hashToCurveG2(message: Array("message B".utf8))
        let x1c0 = fp381ToInt(hm1.x.c0)
        let x2c0 = fp381ToInt(hm2.x.c0)
        expect(x1c0 != x2c0, "Different messages hash to different G2 points")
    }

    // Test 11: Same message produces same curve point (deterministic)
    do {
        let msg = Array("deterministic".utf8)
        let hm1 = engine.hashToCurveG2(message: msg)
        let hm2 = engine.hashToCurveG2(message: msg)
        let x1c0 = fp381ToInt(hm1.x.c0)
        let x2c0 = fp381ToInt(hm2.x.c0)
        expect(x1c0 == x2c0, "Same message hashes to same G2 point")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Signature Aggregation")

    // Test 12: Aggregate 2 signatures
    do {
        let (sk1, _) = engine.deterministicKeyPair(seed: 10)
        let (sk2, _) = engine.deterministicKeyPair(seed: 11)
        let msg1 = Array("msg1".utf8)
        let msg2 = Array("msg2".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)

        let agg = engine.aggregateSignatures([sig1, sig2])
        expect(agg.count == 2, "Aggregate count is 2")
        expect(engine.g2IsOnCurve(agg.aggregateSignature),
               "Aggregated signature is on G2 curve")
    }

    // Test 13: Aggregate single signature returns itself
    do {
        let (sk, _) = engine.deterministicKeyPair(seed: 20)
        let msg = Array("solo".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        let agg = engine.aggregateSignatures([sig])
        expect(agg.count == 1, "Aggregate of 1 has count 1")

        let sigX = fp381ToInt(sig.x.c0)
        let aggX = fp381ToInt(agg.aggregateSignature.x.c0)
        expect(sigX == aggX, "Aggregate of 1 equals original signature")
    }

    // Test 14: Aggregate 8 signatures (may trigger GPU path)
    do {
        var signatures = [G2Affine381]()
        for i in 0..<8 {
            let (sk, _) = engine.deterministicKeyPair(seed: UInt64(50 + i))
            let msg = Array("batch message \(i)".utf8)
            signatures.append(engine.sign(message: msg, secretKey: sk))
        }

        let agg = engine.aggregateSignatures(signatures)
        expect(agg.count == 8, "Aggregate count is 8")
        expect(engine.g2IsOnCurve(agg.aggregateSignature),
               "Large aggregate signature is on G2 curve")
    }

    // Test 15: Aggregation is order-independent (commutative)
    do {
        let (sk1, _) = engine.deterministicKeyPair(seed: 30)
        let (sk2, _) = engine.deterministicKeyPair(seed: 31)
        let (sk3, _) = engine.deterministicKeyPair(seed: 32)
        let msg1 = Array("a".utf8)
        let msg2 = Array("b".utf8)
        let msg3 = Array("c".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)
        let sig3 = engine.sign(message: msg3, secretKey: sk3)

        let agg_123 = engine.aggregateSignatures([sig1, sig2, sig3])
        let agg_321 = engine.aggregateSignatures([sig3, sig2, sig1])
        let x123 = fp381ToInt(agg_123.aggregateSignature.x.c0)
        let x321 = fp381ToInt(agg_321.aggregateSignature.x.c0)
        expect(x123 == x321, "Aggregation is order-independent")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Aggregate Verification")

    // Test 16: Verify aggregate of 2 distinct message/signer pairs
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 70)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 71)
        let msg1 = Array("message one".utf8)
        let msg2 = Array("message two".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)

        let aggSig = engine.aggregateSignatures([sig1, sig2]).aggregateSignature
        let valid = engine.verifyAggregate(
            messages: [msg1, msg2],
            publicKeys: [pk1, pk2],
            aggregateSignature: aggSig
        )
        expect(valid, "Aggregate verification of 2 sigs succeeds")
    }

    // Test 17: Aggregate verify fails with wrong message
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 80)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 81)
        let msg1 = Array("good msg 1".utf8)
        let msg2 = Array("good msg 2".utf8)
        let badMsg = Array("bad msg".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)

        let aggSig = engine.aggregateSignatures([sig1, sig2]).aggregateSignature
        let valid = engine.verifyAggregate(
            messages: [badMsg, msg2],
            publicKeys: [pk1, pk2],
            aggregateSignature: aggSig
        )
        expect(!valid, "Aggregate verify fails with wrong message")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Fast Aggregate Verify")

    // Test 18: Fast aggregate verify (same message, multiple signers)
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 90)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 91)
        let (sk3, pk3) = engine.deterministicKeyPair(seed: 92)
        let msg = Array("beacon block root".utf8)
        let sig1 = engine.sign(message: msg, secretKey: sk1)
        let sig2 = engine.sign(message: msg, secretKey: sk2)
        let sig3 = engine.sign(message: msg, secretKey: sk3)

        let aggSig = engine.aggregateSignatures([sig1, sig2, sig3]).aggregateSignature
        let valid = engine.fastAggregateVerify(
            message: msg,
            publicKeys: [pk1, pk2, pk3],
            aggregateSignature: aggSig
        )
        expect(valid, "Fast aggregate verify with 3 signers succeeds")
    }

    // Test 19: Fast aggregate verify fails with missing signer
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 93)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 94)
        let (_, pk3) = engine.deterministicKeyPair(seed: 95)
        let msg = Array("consensus message".utf8)
        let sig1 = engine.sign(message: msg, secretKey: sk1)
        let sig2 = engine.sign(message: msg, secretKey: sk2)
        // Only aggregate sig1 + sig2, but verify with pk1 + pk2 + pk3
        let aggSig = engine.aggregateSignatures([sig1, sig2]).aggregateSignature
        let valid = engine.fastAggregateVerify(
            message: msg,
            publicKeys: [pk1, pk2, pk3],
            aggregateSignature: aggSig
        )
        expect(!valid, "Fast aggregate verify fails with missing signer")
    }

    // Test 20: Fast aggregate verify with empty public keys returns false
    do {
        let (sk, _) = engine.deterministicKeyPair(seed: 96)
        let msg = Array("empty".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        let valid = engine.fastAggregateVerify(
            message: msg,
            publicKeys: [],
            aggregateSignature: sig
        )
        expect(!valid, "Fast aggregate verify with empty keys returns false")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Batch Verification")

    // Test 21: Batch verify 3 valid signatures
    do {
        var entries: [(message: [UInt8], signature: G2Affine381,
                       publicKey: G1Affine381)] = []
        for i in 0..<3 {
            let (sk, pk) = engine.deterministicKeyPair(seed: UInt64(110 + i))
            let msg = Array("batch entry \(i)".utf8)
            let sig = engine.sign(message: msg, secretKey: sk)
            entries.append((message: msg, signature: sig, publicKey: pk))
        }

        let result = engine.batchVerify(entries: entries)
        expect(result.allValid, "Batch verify of 3 valid sigs succeeds")
        expect(result.results.count == 3, "Batch result count is 3")
        expect(result.results.allSatisfy { $0 }, "All individual results are true")
    }

    // Test 22: Batch verify with one invalid signature
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 120)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 121)
        let (_, pk3) = engine.deterministicKeyPair(seed: 122)
        let msg1 = Array("valid 1".utf8)
        let msg2 = Array("valid 2".utf8)
        let msg3 = Array("valid 3".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)
        // sig3 is signed by sk1 but verified with pk3 -> invalid
        let sig3 = engine.sign(message: msg3, secretKey: sk1)

        let entries: [(message: [UInt8], signature: G2Affine381,
                       publicKey: G1Affine381)] = [
            (message: msg1, signature: sig1, publicKey: pk1),
            (message: msg2, signature: sig2, publicKey: pk2),
            (message: msg3, signature: sig3, publicKey: pk3),
        ]

        let result = engine.batchVerify(entries: entries)
        expect(!result.allValid, "Batch verify detects invalid signature")
        // Individual results should identify which one failed
        expect(result.results[0] == true, "First sig is valid")
        expect(result.results[1] == true, "Second sig is valid")
        expect(result.results[2] == false, "Third sig (wrong key) is invalid")
    }

    // Test 23: Batch verify empty list
    do {
        let result = engine.batchVerify(entries: [])
        expect(result.allValid, "Batch verify of empty list succeeds")
        expect(result.results.isEmpty, "Empty batch has empty results")
    }

    // Test 24: Batch verify single entry
    do {
        let (sk, pk) = engine.deterministicKeyPair(seed: 130)
        let msg = Array("single batch".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        let result = engine.batchVerify(entries: [
            (message: msg, signature: sig, publicKey: pk)
        ])
        expect(result.allValid, "Batch verify of single entry succeeds")
        expect(result.results.count == 1, "Single batch has 1 result")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Subgroup Checks")

    // Test 25: G1 generator is in subgroup
    do {
        let gen = bls12381G1Generator()
        expect(engine.g1SubgroupCheck(gen), "G1 generator is in r-torsion subgroup")
    }

    // Test 26: G2 generator is in subgroup
    do {
        let gen = bls12381G2Generator()
        expect(engine.g2SubgroupCheck(gen), "G2 generator is in r-torsion subgroup")
    }

    // Test 27: Derived G1 points are in subgroup
    do {
        let (_, pk) = engine.deterministicKeyPair(seed: 150)
        expect(engine.g1SubgroupCheck(pk), "Derived G1 public key is in subgroup")
    }

    // Test 28: G1 generator is on curve
    do {
        let gen = bls12381G1Generator()
        expect(engine.g1IsOnCurve(gen), "G1 generator is on curve")
    }

    // Test 29: G2 generator is on curve
    do {
        let gen = bls12381G2Generator()
        expect(engine.g2IsOnCurve(gen), "G2 generator is on twist curve")
    }

    // Test 30: Full G1 validation (on-curve + subgroup)
    do {
        let gen = bls12381G1Generator()
        expect(engine.g1Validate(gen), "G1 generator passes full validation")
    }

    // Test 31: Full G2 validation (on-curve + subgroup)
    do {
        let gen = bls12381G2Generator()
        expect(engine.g2Validate(gen), "G2 generator passes full validation")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Batch Subgroup Checks")

    // Test 32: Batch G1 subgroup check
    do {
        let gen = bls12381G1Generator()
        let (_, pk1) = engine.deterministicKeyPair(seed: 160)
        let (_, pk2) = engine.deterministicKeyPair(seed: 161)
        let (_, pk3) = engine.deterministicKeyPair(seed: 162)

        let results = engine.batchG1SubgroupCheck([gen, pk1, pk2, pk3])
        expect(results.count == 4, "Batch G1 subgroup check returns 4 results")
        expect(results.allSatisfy { $0 }, "All G1 points are in subgroup")
    }

    // Test 33: Batch G2 subgroup check
    do {
        let gen = bls12381G2Generator()
        let (sk1, _) = engine.deterministicKeyPair(seed: 170)
        let (sk2, _) = engine.deterministicKeyPair(seed: 171)
        let msg1 = Array("sub1".utf8)
        let msg2 = Array("sub2".utf8)
        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)

        let results = engine.batchG2SubgroupCheck([gen, sig1, sig2])
        expect(results.count == 3, "Batch G2 subgroup check returns 3 results")
        expect(results.allSatisfy { $0 }, "All G2 points are in subgroup")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Public Key Aggregation")

    // Test 34: Aggregate 2 public keys
    do {
        let (_, pk1) = engine.deterministicKeyPair(seed: 180)
        let (_, pk2) = engine.deterministicKeyPair(seed: 181)
        let aggPk = engine.aggregatePublicKeys([pk1, pk2])
        expect(aggPk != nil, "Aggregated public key is non-nil")
        expect(engine.g1IsOnCurve(aggPk!), "Aggregated public key is on curve")
    }

    // Test 35: Aggregate empty public keys returns nil
    do {
        let aggPk = engine.aggregatePublicKeys([])
        expect(aggPk == nil, "Aggregate of empty keys returns nil")
    }

    // Test 36: Public key aggregation is order-independent
    do {
        let (_, pk1) = engine.deterministicKeyPair(seed: 190)
        let (_, pk2) = engine.deterministicKeyPair(seed: 191)
        let (_, pk3) = engine.deterministicKeyPair(seed: 192)
        let agg_123 = engine.aggregatePublicKeys([pk1, pk2, pk3])!
        let agg_321 = engine.aggregatePublicKeys([pk3, pk2, pk1])!
        let x123 = fp381ToInt(agg_123.x)
        let x321 = fp381ToInt(agg_321.x)
        expect(x123 == x321, "Public key aggregation is order-independent")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — Random Key Pair")

    // Test 37: Random key pair produces valid key
    do {
        let (sk, pk) = engine.generateKeyPair()
        expect(!sk.isZero, "Random secret key is non-zero")
        expect(engine.g1IsOnCurve(pk), "Random public key is on curve")
    }

    // Test 38: Two random key pairs are different
    do {
        let (sk1, _) = engine.generateKeyPair()
        let (sk2, _) = engine.generateKeyPair()
        let sk1Limbs = fr381ToInt(sk1)
        let sk2Limbs = fr381ToInt(sk2)
        expect(sk1Limbs != sk2Limbs, "Two random key pairs differ")
    }

    // ---------------------------------------------------------------
    suite("GPU BLS Aggregate Engine — End-to-End Workflow")

    // Test 39: Full Ethereum-style workflow: 3 validators sign same message
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 1000)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 1001)
        let (sk3, pk3) = engine.deterministicKeyPair(seed: 1002)

        let blockRoot = Array("0xdeadbeef".utf8)

        // Each validator signs
        let sig1 = engine.sign(message: blockRoot, secretKey: sk1)
        let sig2 = engine.sign(message: blockRoot, secretKey: sk2)
        let sig3 = engine.sign(message: blockRoot, secretKey: sk3)

        // Aggregate
        let aggResult = engine.aggregateSignatures([sig1, sig2, sig3])
        expect(aggResult.count == 3, "Workflow: aggregated 3 signatures")

        // Fast aggregate verify
        let valid = engine.fastAggregateVerify(
            message: blockRoot,
            publicKeys: [pk1, pk2, pk3],
            aggregateSignature: aggResult.aggregateSignature
        )
        expect(valid, "Workflow: fast aggregate verify succeeds")
    }

    // Test 40: Full workflow with distinct messages
    do {
        let (sk1, pk1) = engine.deterministicKeyPair(seed: 2000)
        let (sk2, pk2) = engine.deterministicKeyPair(seed: 2001)

        let msg1 = Array("transfer 100 ETH".utf8)
        let msg2 = Array("transfer 200 ETH".utf8)

        let sig1 = engine.sign(message: msg1, secretKey: sk1)
        let sig2 = engine.sign(message: msg2, secretKey: sk2)

        let aggSig = engine.aggregateSignatures([sig1, sig2]).aggregateSignature

        let valid = engine.verifyAggregate(
            messages: [msg1, msg2],
            publicKeys: [pk1, pk2],
            aggregateSignature: aggSig
        )
        expect(valid, "Workflow: distinct-message aggregate verify succeeds")

        // Tamper with messages -> should fail
        let tamperedValid = engine.verifyAggregate(
            messages: [msg2, msg1],  // swapped
            publicKeys: [pk1, pk2],
            aggregateSignature: aggSig
        )
        expect(!tamperedValid, "Workflow: swapped messages fail verification")
    }
}
