// ECDSA Batch Verification Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runECDSABench() {
    print("\n=== ECDSA secp256k1 Verification ===")

    // --- Scalar field tests ---
    print("\n--- Scalar Field (Fr) Tests ---")

    let one = secpFrFromInt(1)
    let oneOut = secpFrToInt(one)
    let oneOk = oneOut[0] == 1 && oneOut[1] == 0 && oneOut[2] == 0 && oneOut[3] == 0
    print("  Fr fromInt(1) round-trip: \(oneOk ? "PASS" : "FAIL")")

    let a = secpFrFromInt(42)
    let b = secpFrFromInt(7)
    let ab = secpFrMul(a, b)
    let abOut = secpFrToInt(ab)
    let mulOk = abOut[0] == 294 && abOut[1] == 0
    print("  Fr 42*7 = 294: \(mulOk ? "PASS" : "FAIL")")

    let aInv = secpFrInverse(a)
    let aInvA = secpFrMul(a, aInv)
    let invOk = aInvA == SecpFr.one
    print("  Fr a * a^(-1) = 1: \(invOk ? "PASS" : "FAIL")")

    // Batch inverse test
    let testElems = [secpFrFromInt(3), secpFrFromInt(7), secpFrFromInt(42), secpFrFromInt(100)]
    let batchInvs = secpFrBatchInverse(testElems)
    var batchOk = true
    for i in 0..<testElems.count {
        let prod = secpFrMul(testElems[i], batchInvs[i])
        if prod != SecpFr.one {
            batchOk = false
            print("  Batch inverse FAIL at index \(i)")
        }
    }
    print("  Fr batch inverse: \(batchOk ? "PASS" : "FAIL")")

    // --- Square root test ---
    let x = secpFromInt(4)
    if let sqrtX = secpSqrt(x) {
        let sqrtVal = secpToInt(sqrtX)
        let sqrtOk = sqrtVal[0] == 2 && sqrtVal[1] == 0
        print("  Fp sqrt(4) = 2: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  Fp sqrt(4) = NONE: FAIL")
    }

    // Test sqrt of a known non-residue
    let nr = secpFromInt(3)
    let sqrtNR = secpSqrt(nr)
    print("  Fp sqrt(3) = none: \(sqrtNR == nil ? "PASS" : "FAIL")")

    // --- ECDSA single verification ---
    print("\n--- ECDSA Verification ---")

    // Generate a known signature using deterministic values
    // Private key d = 42
    let d = secpFrFromInt(42)
    let gen = secp256k1Generator()
    let gProj = secpPointFromAffine(gen)

    // Public key Q = d*G
    let qProj = secpPointMulScalar(gProj, secpFrToInt(d))
    let Q = secpPointToAffine(qProj)

    // Verify Q is on curve
    let qx3 = secpMul(secpSqr(Q.x), Q.x)
    let qRhs = secpAdd(qx3, secpFromInt(7))
    let qOnCurve = secpToInt(secpSqr(Q.y)) == secpToInt(qRhs)
    print("  Q = 42*G on curve: \(qOnCurve ? "PASS" : "FAIL")")

    // Create a signature: k = 137 (nonce), z = hash = 12345
    let k = secpFrFromInt(137)
    let z = secpFrFromInt(12345)

    // R = k*G
    let rProj = secpPointMulScalar(gProj, secpFrToInt(k))
    let rAff = secpPointToAffine(rProj)
    let rXraw = secpToInt(rAff.x)  // Fp value
    // r = R.x mod n
    var rModN = rXraw
    if gte256(rModN, SecpFr.N) {
        (rModN, _) = sub256(rModN, SecpFr.N)
    }
    let rFr = secpFrFromRaw(rModN)

    // s = k^(-1) * (z + r*d) mod n
    let kInv = secpFrInverse(k)
    let rd = secpFrMul(rFr, d)
    let zPlusRD = secpFrAdd(z, rd)
    let sFr = secpFrMul(kInv, zPlusRD)

    print("  Signature (d=42, k=137, z=12345):")
    let rInt = secpFrToInt(rFr)
    let sInt = secpFrToInt(sFr)
    print("    r = 0x\(rInt.reversed().map { String(format: "%016llx", $0) }.joined().prefix(32))...")
    print("    s = 0x\(sInt.reversed().map { String(format: "%016llx", $0) }.joined().prefix(32))...")

    // Verify the signature
    do {
        let engine = try ECDSAEngine()
        let sig = ECDSASignature(r: rFr, s: sFr, z: z)

        let valid = engine.verify(sig: sig, pubkey: Q)
        print("  Single verify (valid sig): \(valid ? "PASS" : "FAIL")")

        // Test with wrong message
        let wrongZ = secpFrFromInt(99999)
        let wrongSig = ECDSASignature(r: rFr, s: sFr, z: wrongZ)
        let invalid = engine.verify(sig: wrongSig, pubkey: Q)
        print("  Single verify (wrong z): \(!invalid ? "PASS" : "FAIL")")

        // Test with wrong key
        let wrongQ = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(secpFrFromInt(99))))
        let invalidKey = engine.verify(sig: sig, pubkey: wrongQ)
        print("  Single verify (wrong Q): \(!invalidKey ? "PASS" : "FAIL")")

        // --- Batch verification ---
        print("\n--- Batch Verification ---")

        // Generate N signatures with different keys
        let batchN = 64
        var sigs = [ECDSASignature]()
        var pubkeys = [SecpPointAffine]()

        for i in 0..<batchN {
            let di = secpFrFromInt(UInt64(100 + i))
            let qi = secpPointToAffine(secpPointMulScalar(gProj, secpFrToInt(di)))
            let ki = secpFrFromInt(UInt64(1000 + i * 7))
            let zi = secpFrFromInt(UInt64(50000 + i * 13))

            let ri_proj = secpPointMulScalar(gProj, secpFrToInt(ki))
            let ri_aff = secpPointToAffine(ri_proj)
            let ri_x = secpToInt(ri_aff.x)
            var ri_mod = ri_x
            if gte256(ri_mod, SecpFr.N) {
                (ri_mod, _) = sub256(ri_mod, SecpFr.N)
            }
            let ri_fr = secpFrFromRaw(ri_mod)

            let ki_inv = secpFrInverse(ki)
            let si_fr = secpFrMul(ki_inv, secpFrAdd(zi, secpFrMul(ri_fr, di)))

            sigs.append(ECDSASignature(r: ri_fr, s: si_fr, z: zi))
            pubkeys.append(qi)
        }

        // Individual batch verification
        let batchResults = try engine.batchVerify(signatures: sigs, pubkeys: pubkeys)
        let allValid = batchResults.allSatisfy { $0 }
        print("  Batch verify \(batchN) valid sigs: \(allValid ? "PASS" : "FAIL (\(batchResults.filter { !$0 }.count) failed)")")

        // Probabilistic batch verify
        // First, compute y-parity for each R point
        var recoveryBits = [UInt8]()
        for i in 0..<batchN {
            let ki = secpFrFromInt(UInt64(1000 + i * 7))
            let ri_proj = secpPointMulScalar(gProj, secpFrToInt(ki))
            let ri_aff = secpPointToAffine(ri_proj)
            let ry = secpToInt(ri_aff.y)
            recoveryBits.append(UInt8(ry[0] & 1))
        }

        let probResult = try engine.batchVerifyProbabilistic(
            signatures: sigs, pubkeys: pubkeys, recoveryBits: recoveryBits)
        print("  Probabilistic batch verify \(batchN) valid: \(probResult ? "PASS" : "FAIL")")

        // Test with one invalid signature
        var badSigs = sigs
        badSigs[batchN / 2] = ECDSASignature(r: sigs[batchN / 2].r, s: sigs[batchN / 2].s,
                                               z: secpFrFromInt(99999))
        let badBatch = try engine.batchVerify(signatures: badSigs, pubkeys: pubkeys)
        let detectedBad = !badBatch[batchN / 2] && badBatch.enumerated().filter({ $0.offset != batchN / 2 }).allSatisfy({ $0.element })
        print("  Batch detect 1 invalid: \(detectedBad ? "PASS" : "FAIL")")

        let badProbResult = try engine.batchVerifyProbabilistic(
            signatures: badSigs, pubkeys: pubkeys, recoveryBits: recoveryBits)
        print("  Probabilistic detect 1 invalid: \(!badProbResult ? "PASS" : "FAIL")")

        // --- Performance ---
        if !skipCPU {
            print("\n--- Verification Performance ---")

            // Single verification timing
            let singleRuns = 100
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<singleRuns {
                let _ = engine.verify(sig: sig, pubkey: Q)
            }
            let singleTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(singleRuns)
            fputs(String(format: "  Single verify: %.2f ms/sig\n", singleTime), stderr)

            // Probabilistic batch timing
            for batchSize in [64, 256, 1024] {
                if batchSize > batchN { break }
                let bSigs = Array(sigs.prefix(batchSize))
                let bKeys = Array(pubkeys.prefix(batchSize))
                let bRecov = Array(recoveryBits.prefix(batchSize))

                let _ = try engine.batchVerifyProbabilistic(
                    signatures: bSigs, pubkeys: bKeys, recoveryBits: bRecov)  // warmup

                let runs = 3
                var times = [Double]()
                for _ in 0..<runs {
                    let start = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.batchVerifyProbabilistic(
                        signatures: bSigs, pubkeys: bKeys, recoveryBits: bRecov)
                    times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
                }
                times.sort()
                let median = times[runs / 2]
                fputs(String(format: "  Batch probabilistic %4d sigs: %7.1f ms (%.2f ms/sig)\n",
                      batchSize, median, median / Double(batchSize)), stderr)
            }
        }

        print("\n  ECDSA overall: \(oneOk && mulOk && invOk && batchOk && qOnCurve && valid && !invalid && !invalidKey && allValid && probResult && detectedBad && !badProbResult ? "ALL PASS ✓" : "SOME FAILED ✗")")

    } catch {
        print("  ERROR: \(error)")
    }
}
