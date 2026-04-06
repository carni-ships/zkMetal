// Batch Ed25519 Signature Verification — Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runBatchEd25519Bench() {
    print("\n=== Batch Ed25519 Verification ===")
    print("  Version: \(BatchEd25519Verifier.version.description)")
    fflush(stdout)

    let engine = EdDSAEngine()
    let verifier = BatchEd25519Verifier()

    // --- Correctness Tests ---
    print("\n--- Correctness Tests ---")

    // Test 1: Valid single signature
    do {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 42
        let sk = EdDSASecretKey(seed: seed)
        let msg = Array("Hello, Ed25519!".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        let valid = verifier.verifySingle(message: msg, signature: sig, publicKey: sk.publicKey)
        print("  Single valid signature: \(valid ? "PASS" : "FAIL")")
    }

    // Test 2: Invalid signature rejection (wrong message)
    do {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 42
        let sk = EdDSASecretKey(seed: seed)
        let msg = Array("Hello, Ed25519!".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        let wrongMsg = Array("Wrong message".utf8)
        let invalid = verifier.verifySingle(message: wrongMsg, signature: sig, publicKey: sk.publicKey)
        print("  Reject wrong message: \(!invalid ? "PASS" : "FAIL")")
    }

    // Test 3: Invalid signature rejection (wrong key)
    do {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 42
        let sk = EdDSASecretKey(seed: seed)
        let msg = Array("Hello, Ed25519!".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        var seed2 = [UInt8](repeating: 0, count: 32)
        seed2[0] = 99
        let sk2 = EdDSASecretKey(seed: seed2)
        let invalid = verifier.verifySingle(message: msg, signature: sig, publicKey: sk2.publicKey)
        print("  Reject wrong key: \(!invalid ? "PASS" : "FAIL")")
    }

    // Test 4: Invalid signature rejection (corrupted S)
    do {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 42
        let sk = EdDSASecretKey(seed: seed)
        let msg = Array("Hello, Ed25519!".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        // Corrupt S bytes
        var badS = sig.s
        badS[0] ^= 0xFF
        let badSig = Ed25519Signature(r: sig.r, s: badS)
        let invalid = verifier.verifySingle(message: msg, signature: badSig, publicKey: sk.publicKey)
        print("  Reject corrupted S: \(!invalid ? "PASS" : "FAIL")")
    }

    // Test 5: Batch verification with all valid signatures
    do {
        let batchN = 16
        var sigs = [Ed25519Signature]()
        var msgs = [[UInt8]]()
        var pks = [Ed25519PublicKey]()

        for i in 0..<batchN {
            var si = [UInt8](repeating: 0, count: 32)
            si[0] = UInt8(i + 1)
            let ski = EdDSASecretKey(seed: si)
            let msgi = Array("Batch message \(i)".utf8)
            let sigi = engine.sign(message: msgi, secretKey: ski)
            sigs.append(sigi)
            msgs.append(msgi)
            pks.append(ski.publicKey)
        }

        let allValid = verifier.verifyBatch(messages: msgs, signatures: sigs, publicKeys: pks)
        print("  Batch \(batchN) valid: \(allValid ? "PASS" : "FAIL")")
    }

    // Test 6: Batch verification detects one invalid signature
    do {
        let batchN = 16
        var sigs = [Ed25519Signature]()
        var msgs = [[UInt8]]()
        var pks = [Ed25519PublicKey]()

        for i in 0..<batchN {
            var si = [UInt8](repeating: 0, count: 32)
            si[0] = UInt8(i + 1)
            let ski = EdDSASecretKey(seed: si)
            let msgi = Array("Batch message \(i)".utf8)
            let sigi = engine.sign(message: msgi, secretKey: ski)
            sigs.append(sigi)
            msgs.append(msgi)
            pks.append(ski.publicKey)
        }

        // Corrupt one signature (replace S with zeros, which is still < q)
        let badIdx = batchN / 2
        var badS = [UInt8](repeating: 0, count: 32)
        badS[0] = 1  // non-zero but wrong
        sigs[badIdx] = Ed25519Signature(r: sigs[badIdx].r, s: badS)

        let batchResult = verifier.verifyBatch(messages: msgs, signatures: sigs, publicKeys: pks)
        print("  Batch detect invalid: \(!batchResult ? "PASS" : "FAIL")")
    }

    // Test 7: Detailed batch identifies which signature is bad
    do {
        let batchN = 8
        var sigs = [Ed25519Signature]()
        var msgs = [[UInt8]]()
        var pks = [Ed25519PublicKey]()

        for i in 0..<batchN {
            var si = [UInt8](repeating: 0, count: 32)
            si[0] = UInt8(i + 10)
            let ski = EdDSASecretKey(seed: si)
            let msgi = Array("Detail msg \(i)".utf8)
            let sigi = engine.sign(message: msgi, secretKey: ski)
            sigs.append(sigi)
            msgs.append(msgi)
            pks.append(ski.publicKey)
        }

        // Corrupt signature at index 3
        let badIdx = 3
        var badS = sigs[badIdx].s
        badS[0] ^= 0xFF
        sigs[badIdx] = Ed25519Signature(r: sigs[badIdx].r, s: badS)

        let results = verifier.verifyBatchDetailed(messages: msgs, signatures: sigs, publicKeys: pks)
        var detailOk = true
        for i in 0..<batchN {
            if i == badIdx {
                if results[i] { detailOk = false }
            } else {
                if !results[i] { detailOk = false }
            }
        }
        print("  Detailed batch identifies bad sig: \(detailOk ? "PASS" : "FAIL")")
    }

    // Test 8: Empty batch
    do {
        let result = verifier.verifyBatch(messages: [], signatures: [], publicKeys: [])
        print("  Empty batch: \(result ? "PASS" : "FAIL")")
    }

    // Test 9: Single-element batch
    do {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 77
        let sk = EdDSASecretKey(seed: seed)
        let msg = Array("Single batch".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        let result = verifier.verifyBatch(messages: [msg], signatures: [sig], publicKeys: [sk.publicKey])
        print("  Single-element batch: \(result ? "PASS" : "FAIL")")
    }

    // Test 10: Strategy descriptions
    print("\n--- Strategy ---")
    print("  N=1:   \(BatchEd25519Verifier.strategyDescription(batchSize: 1))")
    print("  N=8:   \(BatchEd25519Verifier.strategyDescription(batchSize: 8))")
    print("  N=256: \(BatchEd25519Verifier.strategyDescription(batchSize: 256))")
    print("  N=1K:  \(BatchEd25519Verifier.strategyDescription(batchSize: 1024))")

    // --- Performance ---
    if !skipCPU {
        print("\n--- Performance ---")

        // Generate signatures for benchmarking
        let benchSizes = [1, 4, 16, 64, 256]
        for batchN in benchSizes {
            var sigs = [Ed25519Signature]()
            var msgs = [[UInt8]]()
            var pks = [Ed25519PublicKey]()

            for i in 0..<batchN {
                var si = [UInt8](repeating: 0, count: 32)
                si[0] = UInt8(i & 0xFF)
                si[1] = UInt8((i >> 8) & 0xFF)
                let ski = EdDSASecretKey(seed: si)
                let msgi = Array("Bench message \(i)".utf8)
                let sigi = engine.sign(message: msgi, secretKey: ski)
                sigs.append(sigi)
                msgs.append(msgi)
                pks.append(ski.publicKey)
            }

            let runs = batchN <= 16 ? 100 : 20
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<runs {
                let _ = verifier.verifyBatch(messages: msgs, signatures: sigs, publicKeys: pks)
            }
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(runs)
            let perSig = elapsed / Double(batchN)
            fputs(String(format: "  Batch %4d: %7.2f ms total, %6.2f ms/sig\n", batchN, elapsed, perSig), stderr)
        }

        // Individual verification baseline
        do {
            var seed = [UInt8](repeating: 0, count: 32)
            seed[0] = 42
            let sk = EdDSASecretKey(seed: seed)
            let msg = Array("Baseline".utf8)
            let sig = engine.sign(message: msg, secretKey: sk)

            let runs = 200
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<runs {
                let _ = verifier.verifySingle(message: msg, signature: sig, publicKey: sk.publicKey)
            }
            let perSig = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(runs)
            fputs(String(format: "  Single verify baseline: %.2f ms/sig\n", perSig), stderr)
        }
    }

    print("\n  Batch Ed25519 tests complete.")
    fflush(stdout)
}
