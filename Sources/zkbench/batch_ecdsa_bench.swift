// Batch ECDSA Verification Benchmark
import zkMetal
import Foundation

public func runBatchECDSABench() {
    print("\n=== Batch ECDSA Verifier (secp256k1) ===")
    print("  Version: \(BatchECDSAVerifier.version.description)")

    do {
        let verifier = try BatchECDSAVerifier()

        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)

        // --- Helper: create a valid (sig, pubkey, recoveryBit) tuple ---
        func makeSignature(privKey: UInt64, nonce: UInt64, msgHash: UInt64)
            -> (ECDSASignature, SecpPointAffine, UInt8)
        {
            let d = secpFrFromInt(privKey)
            let Q = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(d)))
            let k = secpFrFromInt(nonce)
            let z = secpFrFromInt(msgHash)

            let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
            let rAff = secpPointToAffine(rProj)
            let rXraw = secpToInt(rAff.x)
            var rModN = rXraw
            if gte256(rModN, SecpFr.N) {
                (rModN, _) = sub256(rModN, SecpFr.N)
            }
            let rFr = secpFrFromRaw(rModN)

            let kInv = secpFrInverse(k)
            let sFr = secpFrMul(kInv, secpFrAdd(z, secpFrMul(rFr, d)))

            let ry = secpToInt(rAff.y)
            let parity = UInt8(ry[0] & 1)

            return (ECDSASignature(r: rFr, s: sFr, z: z), Q, parity)
        }

        // --- Correctness: single verification ---
        print("\n--- Correctness Tests ---")

        let (validSig, validPK, _) = makeSignature(privKey: 42, nonce: 137, msgHash: 12345)
        let singleOk = verifier.verifySingle(sig: validSig, pubkey: validPK)
        print("  Single verify (valid):   \(singleOk ? "PASS" : "FAIL")")

        // Wrong message hash
        let wrongZ = ECDSASignature(r: validSig.r, s: validSig.s, z: secpFrFromInt(99999))
        let wrongZResult = verifier.verifySingle(sig: wrongZ, pubkey: validPK)
        print("  Single verify (bad z):   \(!wrongZResult ? "PASS" : "FAIL")")

        // Wrong public key
        let wrongPK = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(secpFrFromInt(99))))
        let wrongPKResult = verifier.verifySingle(sig: validSig, pubkey: wrongPK)
        print("  Single verify (bad pk):  \(!wrongPKResult ? "PASS" : "FAIL")")

        // --- Batch with all valid ---
        let batchN = 64
        var sigs = [ECDSASignature]()
        var pks = [SecpPointAffine]()
        var recov = [UInt8]()

        for i in 0..<batchN {
            let (s, pk, rb) = makeSignature(
                privKey: UInt64(100 + i), nonce: UInt64(1000 + i * 7),
                msgHash: UInt64(50000 + i * 13))
            sigs.append(s)
            pks.append(pk)
            recov.append(rb)
        }

        let batchResults = try verifier.verifyBatch(
            signatures: sigs, publicKeys: pks, recoveryBits: recov)
        let allValid = batchResults.allSatisfy { $0 }
        print("  Batch verify \(batchN) valid:  \(allValid ? "PASS" : "FAIL")")

        // --- Batch with one invalid ---
        var badSigs = sigs
        badSigs[batchN / 2] = ECDSASignature(
            r: sigs[batchN / 2].r, s: sigs[batchN / 2].s, z: secpFrFromInt(99999))

        let badResults = try verifier.verifyBatch(
            signatures: badSigs, publicKeys: pks, recoveryBits: recov)
        let detectedBad = !badResults[batchN / 2]
        let othersOk = badResults.enumerated()
            .filter { $0.offset != batchN / 2 }
            .allSatisfy { $0.element }
        print("  Batch detect 1 invalid:  \(detectedBad && othersOk ? "PASS" : "FAIL")")

        // --- Probabilistic batch (all valid) ---
        let probOk = try verifier.batchVerifyProbabilistic(
            signatures: sigs, publicKeys: pks, recoveryBits: recov)
        print("  Probabilistic all valid: \(probOk ? "PASS" : "FAIL")")

        // --- Probabilistic batch (one invalid) ---
        let probBad = try verifier.batchVerifyProbabilistic(
            signatures: badSigs, publicKeys: pks, recoveryBits: recov)
        print("  Probabilistic detect bad:\(!probBad ? "PASS" : "FAIL")")

        // --- Strategy descriptions ---
        print("\n--- Strategy Selection ---")
        for n in [1, 3, 4, 64, 256, 1024] {
            print("  N=\(String(format: "%4d", n)): \(BatchECDSAVerifier.strategyDescription(batchSize: n))")
        }

        // --- Performance ---
        if !skipCPU {
            print("\n--- Performance ---")

            // Single verification
            let singleRuns = 100
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<singleRuns {
                let _ = verifier.verifySingle(sig: validSig, pubkey: validPK)
            }
            let singleMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(singleRuns)
            fputs(String(format: "  Single verify:          %.2f ms/sig\n", singleMs), stderr)

            // Batch verification at various sizes
            for batchSize in [16, 64, 256] {
                if batchSize > batchN { break }
                let bSigs = Array(sigs.prefix(batchSize))
                let bKeys = Array(pks.prefix(batchSize))
                let bRecov = Array(recov.prefix(batchSize))

                // warmup
                let _ = try verifier.verifyBatch(
                    signatures: bSigs, publicKeys: bKeys, recoveryBits: bRecov)

                let runs = 5
                var times = [Double]()
                for _ in 0..<runs {
                    let start = CFAbsoluteTimeGetCurrent()
                    let _ = try verifier.verifyBatch(
                        signatures: bSigs, publicKeys: bKeys, recoveryBits: bRecov)
                    times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
                }
                times.sort()
                let median = times[runs / 2]
                let perSig = median / Double(batchSize)
                let speedup = singleMs / perSig
                fputs(String(format: "  Batch %4d sigs: %7.1f ms total, %.2f ms/sig (%.1fx vs single)\n",
                      batchSize, median, perSig, speedup), stderr)
            }
        }

        let allPass = singleOk && !wrongZResult && !wrongPKResult && allValid &&
                      detectedBad && othersOk && probOk && !probBad
        print("\n  Batch ECDSA overall: \(allPass ? "ALL PASS" : "SOME FAILED")")

    } catch {
        print("  ERROR: \(error)")
    }
}
