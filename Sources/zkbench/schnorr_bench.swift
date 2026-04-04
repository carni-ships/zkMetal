// Schnorr BIP 340 Signature Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runSchnorrBench() {
    print("\n=== Schnorr BIP 340 (secp256k1) ===")

    do {
        let engine = try SchnorrEngine(withMSM: false)

        // --- Correctness Tests ---
        print("\n--- Correctness Tests ---")

        // Test 1: Key generation round-trip
        guard let (sk, pk) = engine.keyGen() else {
            print("  Key generation: FAIL (nil)")
            return
        }
        let pk2 = engine.publicKey(secretKey: sk)
        let keyGenOk = pk2 != nil && pk2! == pk
        print("  Key generation round-trip: \(keyGenOk ? "PASS" : "FAIL")")

        // Test 2: Sign/verify round-trip
        let msg = Array("Hello, Schnorr!".utf8)
        guard let sig = engine.sign(message: msg, secretKey: sk) else {
            print("  Sign: FAIL (nil)")
            return
        }
        let valid = engine.verify(message: msg, signature: sig, publicKey: pk)
        print("  Sign/verify round-trip: \(valid ? "PASS" : "FAIL")")

        // Test 3: Wrong message rejected
        let wrongMsg = Array("Wrong message".utf8)
        let invalid1 = engine.verify(message: wrongMsg, signature: sig, publicKey: pk)
        print("  Wrong message rejected: \(!invalid1 ? "PASS" : "FAIL")")

        // Test 4: Wrong key rejected
        guard let (_, wrongPk) = engine.keyGen() else {
            print("  Wrong key test: FAIL (nil)")
            return
        }
        let invalid2 = engine.verify(message: msg, signature: sig, publicKey: wrongPk)
        print("  Wrong key rejected: \(!invalid2 ? "PASS" : "FAIL")")

        // Test 5: Tampered signature rejected
        var tamperedSig = sig.bytes
        tamperedSig[0] ^= 0x01
        let invalid3 = engine.verify(
            message: msg,
            signature: SchnorrSignature.fromBytes(tamperedSig)!,
            publicKey: pk)
        print("  Tampered signature rejected: \(!invalid3 ? "PASS" : "FAIL")")

        // Test 6: Signature serialization round-trip
        let sigBytes = sig.bytes
        let sigParsed = SchnorrSignature.fromBytes(sigBytes)
        let serOk = sigParsed != nil && sigParsed! == sig
        print("  Signature serialization: \(serOk ? "PASS" : "FAIL")")

        // Test 7: BIP 340 test vectors
        // Vector 0: secret key 0x03, msg 0x00...00
        let bip340ok = runBIP340Vectors(engine: engine)

        // Test 8: Batch verification
        let batchN = 16
        var batchMsgs = [[UInt8]]()
        var batchSigs = [SchnorrSignature]()
        var batchPks = [SchnorrPublicKey]()

        for i in 0..<batchN {
            // Deterministic secret key
            var skBytes = [UInt8](repeating: 0, count: 32)
            skBytes[31] = UInt8(i + 10)
            guard let bpk = engine.publicKey(secretKey: skBytes) else {
                print("  Batch key gen \(i): FAIL")
                return
            }
            let bMsg = Array("batch message \(i)".utf8)
            guard let bSig = engine.sign(message: bMsg, secretKey: skBytes) else {
                print("  Batch sign \(i): FAIL")
                return
            }
            batchMsgs.append(bMsg)
            batchSigs.append(bSig)
            batchPks.append(bpk)
        }

        let batchResult = engine.batchVerify(messages: batchMsgs, signatures: batchSigs, publicKeys: batchPks)
        print("  Batch verify \(batchN) valid: \(batchResult ? "PASS" : "FAIL")")

        // Batch with one invalid
        var badSigs = batchSigs
        var tamperedBytes = badSigs[batchN / 2].bytes
        tamperedBytes[63] ^= 0x01
        badSigs[batchN / 2] = SchnorrSignature.fromBytes(tamperedBytes)!
        let badBatch = engine.batchVerify(messages: batchMsgs, signatures: badSigs, publicKeys: batchPks)
        print("  Batch detect 1 invalid: \(!badBatch ? "PASS" : "FAIL")")

        // Test 9: Tagged hash consistency
        let tag1 = SchnorrEngine.taggedHash(tag: "BIP0340/challenge", data: [0x01, 0x02])
        let tag2 = SchnorrEngine.taggedHash(tag: "BIP0340/challenge", data: [0x01, 0x02])
        let tag3 = SchnorrEngine.taggedHash(tag: "BIP0340/challenge", data: [0x01, 0x03])
        let tagOk = tag1 == tag2 && tag1 != tag3
        print("  Tagged hash consistency: \(tagOk ? "PASS" : "FAIL")")

        let allPass = keyGenOk && valid && !invalid1 && !invalid2 && !invalid3 &&
                      serOk && bip340ok && batchResult && !badBatch && tagOk
        print("\n  Schnorr overall: \(allPass ? "ALL PASS" : "SOME FAILED")")

        // --- Performance ---
        if !skipCPU {
            print("\n--- Performance ---")

            // Deterministic key for benchmarks
            var benchSk = [UInt8](repeating: 0, count: 32)
            benchSk[31] = 42
            let benchPk = engine.publicKey(secretKey: benchSk)!
            let benchMsg = Array("benchmark message for schnorr signatures".utf8)
            let benchSig = engine.sign(message: benchMsg, secretKey: benchSk)!

            // Sign timing
            let signRuns = 200
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<signRuns {
                let _ = engine.sign(message: benchMsg, secretKey: benchSk)
            }
            let signTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(signRuns)
            fputs(String(format: "  Schnorr sign:   %.3f ms/sig\n", signTime), stderr)

            // Verify timing
            let verifyRuns = 200
            let t1 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<verifyRuns {
                let _ = engine.verify(message: benchMsg, signature: benchSig, publicKey: benchPk)
            }
            let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(verifyRuns)
            fputs(String(format: "  Schnorr verify: %.3f ms/sig\n", verifyTime), stderr)

            // Batch verify timing
            for batchSize in [10, 64, 256] {
                var msgs = [[UInt8]]()
                var sigs = [SchnorrSignature]()
                var pks = [SchnorrPublicKey]()

                for i in 0..<batchSize {
                    var ski = [UInt8](repeating: 0, count: 32)
                    ski[31] = UInt8(i & 0xFF)
                    ski[30] = UInt8((i >> 8) & 0xFF)
                    guard let pki = engine.publicKey(secretKey: ski) else { continue }
                    let mi = Array("batch bench msg \(i)".utf8)
                    guard let si = engine.sign(message: mi, secretKey: ski) else { continue }
                    msgs.append(mi)
                    sigs.append(si)
                    pks.append(pki)
                }

                let runs = max(1, 50 / batchSize + 1)
                let t2 = CFAbsoluteTimeGetCurrent()
                for _ in 0..<runs {
                    let _ = engine.batchVerify(messages: msgs, signatures: sigs, publicKeys: pks)
                }
                let batchTime = (CFAbsoluteTimeGetCurrent() - t2) * 1000 / Double(runs)
                fputs(String(format: "  Batch verify %3d: %7.1f ms (%5.3f ms/sig)\n",
                      batchSize, batchTime, batchTime / Double(batchSize)), stderr)
            }

            // Compare with ECDSA
            fputs("\n  --- Comparison with ECDSA ---\n", stderr)
            fputs(String(format: "  Schnorr sign:   %.3f ms  (no field inversion)\n", signTime), stderr)
            fputs(String(format: "  Schnorr verify: %.3f ms  (2 scalar muls, no inversion)\n", verifyTime), stderr)
        }

    } catch {
        print("  ERROR: \(error)")
    }
}

// MARK: - BIP 340 Test Vectors

/// Run BIP 340 official test vectors.
/// Source: https://github.com/bitcoin/bips/blob/master/bip-0340/test-vectors.csv
private func runBIP340Vectors(engine: SchnorrEngine) -> Bool {
    // Test vector 0: known secret key, message, expected signature
    // sk = 0x0000...0003, msg = 0x00...00 (32 bytes)
    // From BIP 340 reference:
    // Vector 0:
    //   sk  = 0000000000000000000000000000000000000000000000000000000000000003
    //   pk  = F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
    //   msg = 0000000000000000000000000000000000000000000000000000000000000000
    //   aux = 0000000000000000000000000000000000000000000000000000000000000000
    //   sig = E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215
    //         25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0

    let sk0 = hexToBytes("0000000000000000000000000000000000000000000000000000000000000003")
    let pk0Expected = hexToBytes("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9")
    let msg0 = [UInt8](repeating: 0, count: 32)
    let aux0 = [UInt8](repeating: 0, count: 32)
    let sigExpected0 = hexToBytes(
        "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215" +
        "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0")

    // Check public key derivation
    guard let pk0 = engine.publicKey(secretKey: sk0) else {
        print("  BIP340 vec 0 pubkey: FAIL (nil)")
        return false
    }
    let pk0ok = pk0.bytes == pk0Expected
    print("  BIP340 vec 0 pubkey: \(pk0ok ? "PASS" : "FAIL")")

    // Check signing
    guard let sig0 = engine.sign(message: msg0, secretKey: sk0, auxRand: aux0) else {
        print("  BIP340 vec 0 sign: FAIL (nil)")
        return false
    }
    let sig0ok = sig0.bytes == sigExpected0
    print("  BIP340 vec 0 sign: \(sig0ok ? "PASS" : "FAIL")")
    if !sig0ok {
        print("    expected: \(sigExpected0.map { String(format: "%02x", $0) }.joined())")
        print("    got:      \(sig0.bytes.map { String(format: "%02x", $0) }.joined())")
    }

    // Check verification
    let ver0 = engine.verify(message: msg0, signature: SchnorrSignature.fromBytes(sigExpected0)!, publicKey: SchnorrPublicKey(bytes: pk0Expected))
    print("  BIP340 vec 0 verify: \(ver0 ? "PASS" : "FAIL")")

    // Vector 1:
    //   sk  = B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF
    //   pk  = DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659
    //   msg = 243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89
    //   aux = 0000000000000000000000000000000000000000000000000000000000000001
    //   sig = 6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE3341
    //         8906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0F

    let sk1 = hexToBytes("B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF")
    let pk1Expected = hexToBytes("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659")
    let msg1 = hexToBytes("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89")
    let aux1 = hexToBytes("0000000000000000000000000000000000000000000000000000000000000001")
    let sigExpected1 = hexToBytes(
        "6896BD60EEAE296DB48A229FF71DFE071BDE413E6D43F917DC8DCF8C78DE3341" +
        "8906D11AC976ABCCB20B091292BFF4EA897EFCB639EA871CFA95F6DE339E4B0A")

    guard let pk1 = engine.publicKey(secretKey: sk1) else {
        print("  BIP340 vec 1 pubkey: FAIL (nil)")
        return false
    }
    let pk1ok = pk1.bytes == pk1Expected
    print("  BIP340 vec 1 pubkey: \(pk1ok ? "PASS" : "FAIL")")

    guard let sig1 = engine.sign(message: msg1, secretKey: sk1, auxRand: aux1) else {
        print("  BIP340 vec 1 sign: FAIL (nil)")
        return false
    }
    let sig1ok = sig1.bytes == sigExpected1
    print("  BIP340 vec 1 sign: \(sig1ok ? "PASS" : "FAIL")")
    if !sig1ok {
        print("    expected: \(sigExpected1.map { String(format: "%02x", $0) }.joined())")
        print("    got:      \(sig1.bytes.map { String(format: "%02x", $0) }.joined())")
    }

    let ver1 = engine.verify(message: msg1, signature: SchnorrSignature.fromBytes(sigExpected1)!, publicKey: SchnorrPublicKey(bytes: pk1Expected))
    print("  BIP340 vec 1 verify: \(ver1 ? "PASS" : "FAIL")")

    // Vector 2:
    //   sk  = C90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B14E5C9
    //   pk  = DD308AFEC5777E13121FA72B9CC1B7CC0139715309B086C960E18FD7D6C0B60D
    //   msg = 7E2D58D8B3BCDF1ABADEC7829054F90DDA9805AAB56C77333024B9D0A508B75C
    //   aux = C87AA53824B4D7AE2EB035A2B5BBBCCC080E76CDC6D1692C4B0B62D798E6D906
    //   sig = 5831AAEED7B44BB74E5EAB94BA9D4294C49BCF2A60728D8B4C200F50DD313C1B
    //         AB745879A5AD954A72C45A91C3A51D3C7ADEA98D82F8481E0E1E03674A6F3FB7

    let sk2 = hexToBytes("C90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B14E5C9")
    let pk2Expected = hexToBytes("DD308AFEC5777E13121FA72B9CC1B7CC0139715309B086C960E18FD969774EB8")
    let msg2 = hexToBytes("7E2D58D8B3BCDF1ABADEC7829054F90DDA9805AAB56C77333024B9D0A508B75C")
    let aux2 = hexToBytes("C87AA53824B4D7AE2EB035A2B5BBBCCC080E76CDC6D1692C4B0B62D798E6D906")
    let sigExpected2 = hexToBytes(
        "5831AAEED7B44BB74E5EAB94BA9D4294C49BCF2A60728D8B4C200F50DD313C1B" +
        "AB745879A5AD954A72C45A91C3A51D3C7ADEA98D82F8481E0E1E03674A6F3FB7")

    guard let pk2v = engine.publicKey(secretKey: sk2) else {
        print("  BIP340 vec 2 pubkey: FAIL (nil)")
        return false
    }
    let pk2ok = pk2v.bytes == pk2Expected
    print("  BIP340 vec 2 pubkey: \(pk2ok ? "PASS" : "FAIL")")

    guard let sig2 = engine.sign(message: msg2, secretKey: sk2, auxRand: aux2) else {
        print("  BIP340 vec 2 sign: FAIL (nil)")
        return false
    }
    let sig2ok = sig2.bytes == sigExpected2
    print("  BIP340 vec 2 sign: \(sig2ok ? "PASS" : "FAIL")")
    if !sig2ok {
        print("    expected: \(sigExpected2.map { String(format: "%02x", $0) }.joined())")
        print("    got:      \(sig2.bytes.map { String(format: "%02x", $0) }.joined())")
    }

    let ver2 = engine.verify(message: msg2, signature: SchnorrSignature.fromBytes(sigExpected2)!, publicKey: SchnorrPublicKey(bytes: pk2Expected))
    print("  BIP340 vec 2 verify: \(ver2 ? "PASS" : "FAIL")")

    // Vector 3: edge case (msg and aux all 0xFF)
    let sk3 = hexToBytes("0B432B2677937381AEF05BB02A66ECD012773062CF3FA2549E44F58ED2401710")
    let pk3Expected = hexToBytes("25D1DFF95105F5253C4022F628A996AD3A0D95FBF21D468A1B33F8C160D8F517")
    let msg3 = [UInt8](repeating: 0xFF, count: 32)
    let aux3 = [UInt8](repeating: 0xFF, count: 32)
    let sigExpected3 = hexToBytes(
        "7EB0509757E246F19449885651611CB965ECC1A187DD51B64FDA1EDC9637D5EC" +
        "97582B9CB13DB3933705B32BA982AF5AF25FD78881EBB32771FC5922EFC66EA3")

    guard let pk3 = engine.publicKey(secretKey: sk3) else {
        print("  BIP340 vec 3 pubkey: FAIL (nil)")
        return false
    }
    let pk3ok = pk3.bytes == pk3Expected
    print("  BIP340 vec 3 pubkey: \(pk3ok ? "PASS" : "FAIL")")

    guard let sig3 = engine.sign(message: msg3, secretKey: sk3, auxRand: aux3) else {
        print("  BIP340 vec 3 sign: FAIL (nil)")
        return false
    }
    let sig3ok = sig3.bytes == sigExpected3
    print("  BIP340 vec 3 sign: \(sig3ok ? "PASS" : "FAIL")")
    if !sig3ok {
        print("    expected: \(sigExpected3.map { String(format: "%02x", $0) }.joined())")
        print("    got:      \(sig3.bytes.map { String(format: "%02x", $0) }.joined())")
    }

    let ver3 = engine.verify(message: msg3, signature: SchnorrSignature.fromBytes(sigExpected3)!, publicKey: SchnorrPublicKey(bytes: pk3Expected))
    print("  BIP340 vec 3 verify: \(ver3 ? "PASS" : "FAIL")")

    // Verification-only vectors (no secret key)
    // Vector 4: valid signature, verify-only
    let pk4 = hexToBytes("D69C3509BB99E412E68B0FE8544E72837DFA30746D8BE2AA65975F29D22DC7B9")
    let msg4 = hexToBytes("4DF3C3F68FCC83B27E9D42C90431A72499F17875C81A599B566C9889B9696703")
    let sig4 = hexToBytes(
        "00000000000000000000003B78CE563F89A0ED9414F5AA28AD0D96D6795F9C63" +
        "76AFB1548AF603B3EB45C9F8207DEE1060CB71C04E80F593060B07D28308D7F4")
    let ver4 = engine.verify(message: msg4,
                              signature: SchnorrSignature.fromBytes(sig4)!,
                              publicKey: SchnorrPublicKey(bytes: pk4))
    print("  BIP340 vec 4 verify (valid): \(ver4 ? "PASS" : "FAIL")")

    // Vector 6: invalid — negated message hash (wrong sig for this pk/msg)
    let pk6 = hexToBytes("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659")
    let msg6 = hexToBytes("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89")
    let sig6 = hexToBytes(
        "FFF97BD5755EEEA420453A14355235D382F6472F8568A18B2F057A1460297556" +
        "3CC27944640AC607CD107AE10923D9EF7A73C643E166BE5EBEAFA34B1AC553E2")
    let ver6 = engine.verify(message: msg6,
                              signature: SchnorrSignature.fromBytes(sig6)!,
                              publicKey: SchnorrPublicKey(bytes: pk6))
    print("  BIP340 vec 6 verify (invalid): \(!ver6 ? "PASS" : "FAIL")")

    // Vector 13: invalid — s == n (curve order)
    let pk13 = hexToBytes("DFF1D77F2A671C5F36183726DB2341BE58FEAE1DA2DECED843240F7B502BA659")
    let msg13 = hexToBytes("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89")
    let sig13 = hexToBytes(
        "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769" +
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141")
    let ver13 = engine.verify(message: msg13,
                               signature: SchnorrSignature.fromBytes(sig13)!,
                               publicKey: SchnorrPublicKey(bytes: pk13))
    print("  BIP340 vec 13 verify (s == n): \(!ver13 ? "PASS" : "FAIL")")

    // Vector 14: invalid — public key exceeds field size
    let pk14 = hexToBytes("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC30")
    let msg14 = hexToBytes("243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89")
    let sig14 = hexToBytes(
        "6CFF5C3BA86C69EA4B7376F31A9BCB4F74C1976089B2D9963DA2E5543E177769" +
        "69E89B4C5564D00349106B8497785DD7D1D713A8AE82B32FA79D5F7FC407D39B")
    let ver14 = engine.verify(message: msg14,
                               signature: SchnorrSignature.fromBytes(sig14)!,
                               publicKey: SchnorrPublicKey(bytes: pk14))
    print("  BIP340 vec 14 verify (pk >= p): \(!ver14 ? "PASS" : "FAIL")")

    return pk0ok && sig0ok && ver0 && pk1ok && sig1ok && ver1 &&
           pk2ok && sig2ok && ver2 && pk3ok && sig3ok && ver3 &&
           ver4 && !ver6 && !ver13 && !ver14
}

/// Hex string to bytes
private func hexToBytes(_ hex: String) -> [UInt8] {
    var bytes = [UInt8]()
    bytes.reserveCapacity(hex.count / 2)
    var idx = hex.startIndex
    while idx < hex.endIndex {
        let next = hex.index(idx, offsetBy: 2)
        if let byte = UInt8(hex[idx..<next], radix: 16) {
            bytes.append(byte)
        }
        idx = next
    }
    return bytes
}
