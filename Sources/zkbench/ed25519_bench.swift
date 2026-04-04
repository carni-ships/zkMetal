// Ed25519 / Curve25519 Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runEd25519Bench() {
    print("\n=== Ed25519 / Curve25519 Tests ===")
    fflush(stdout)

    // --- Field Fp tests ---
    print("\n--- Field Fp (2^255-19) Tests ---")

    let one = ed25519FpFromInt(1)
    let oneOut = ed25519FpToInt(one)
    let oneOk = oneOut[0] == 1 && oneOut[1] == 0 && oneOut[2] == 0 && oneOut[3] == 0
    print("  Fp fromInt(1) round-trip: \(oneOk ? "PASS" : "FAIL")")

    let zero = ed25519FpFromInt(0)
    let zeroOk = zero.isZero
    print("  Fp fromInt(0) is zero: \(zeroOk ? "PASS" : "FAIL")")

    let a = ed25519FpFromInt(42)
    let addZero = ed25519FpAdd(a, Ed25519Fp.zero)
    let addZeroOk = ed25519FpToInt(addZero) == ed25519FpToInt(a)
    print("  Fp a + 0 = a: \(addZeroOk ? "PASS" : "FAIL")")

    let negA = ed25519FpNeg(a)
    let addNeg = ed25519FpAdd(a, negA)
    let addNegOk = addNeg.isZero
    print("  Fp a + (-a) = 0: \(addNegOk ? "PASS" : "FAIL")")

    let mulOne = ed25519FpMul(a, one)
    let mulOneOk = ed25519FpToInt(mulOne) == ed25519FpToInt(a)
    print("  Fp a * 1 = a: \(mulOneOk ? "PASS" : "FAIL")")

    let aInv = ed25519FpInverse(a)
    let mulInv = ed25519FpMul(a, aInv)
    let mulInvOut = ed25519FpToInt(mulInv)
    let mulInvOk = mulInvOut[0] == 1 && mulInvOut[1] == 0 && mulInvOut[2] == 0 && mulInvOut[3] == 0
    print("  Fp a * a^(-1) = 1: \(mulInvOk ? "PASS" : "FAIL")")

    // Distributivity
    let b = ed25519FpFromInt(123456789)
    let c = ed25519FpFromInt(987654321)
    let lhs = ed25519FpMul(ed25519FpAdd(a, b), c)
    let rhs = ed25519FpAdd(ed25519FpMul(a, c), ed25519FpMul(b, c))
    let distOk = ed25519FpToInt(lhs) == ed25519FpToInt(rhs)
    print("  Fp (a+b)*c = a*c + b*c: \(distOk ? "PASS" : "FAIL")")

    // Square root
    let four = ed25519FpFromInt(4)
    if let sqrtFour = ed25519FpSqrt(four) {
        let sqrtVal = ed25519FpToInt(sqrtFour)
        let sqrtCheck = ed25519FpToInt(ed25519FpSqr(sqrtFour))
        let sqrtOk = sqrtCheck == ed25519FpToInt(four)
        print("  Fp sqrt(4)^2 = 4: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  Fp sqrt(4) = NONE: FAIL")
    }

    // --- Scalar field Fq tests ---
    print("\n--- Scalar Field Fq Tests ---")

    let fqOne = ed25519FqFromInt(1)
    let fqOneOut = ed25519FqToInt(fqOne)
    let fqOneOk = fqOneOut[0] == 1 && fqOneOut[1] == 0 && fqOneOut[2] == 0 && fqOneOut[3] == 0
    print("  Fq fromInt(1) round-trip: \(fqOneOk ? "PASS" : "FAIL")")

    let fqA = ed25519FqFromInt(42)
    let fqB = ed25519FqFromInt(7)
    let fqAB = ed25519FqMul(fqA, fqB)
    let fqABOut = ed25519FqToInt(fqAB)
    let fqMulOk = fqABOut[0] == 294 && fqABOut[1] == 0
    print("  Fq 42*7 = 294: \(fqMulOk ? "PASS" : "FAIL")")

    let fqAInv = ed25519FqInverse(fqA)
    let fqAInvA = ed25519FqMul(fqA, fqAInv)
    let fqInvOk = fqAInvA == Ed25519Fq.one
    print("  Fq a * a^(-1) = 1: \(fqInvOk ? "PASS" : "FAIL")")

    // --- Curve tests ---
    print("\n--- Ed25519 Curve Tests ---")

    let gen = ed25519Generator()
    let genOnCurve = ed25519PointOnCurve(gen)
    print("  Generator on curve: \(genOnCurve ? "PASS" : "FAIL")")

    // Identity
    let id = ed25519PointIdentity()
    let isId = ed25519PointIsIdentity(id)
    print("  Identity check: \(isId ? "PASS" : "FAIL")")

    // G + identity = G
    let genExt = ed25519PointFromAffine(gen)
    let gPlusId = ed25519PointAdd(genExt, id)
    let gPlusIdAff = ed25519PointToAffine(gPlusId)
    let gxI = ed25519FpToInt(gPlusIdAff.x)
    let gyI = ed25519FpToInt(gPlusIdAff.y)
    let genxI = ed25519FpToInt(gen.x)
    let genyI = ed25519FpToInt(gen.y)
    let gPlusIdOk = gxI == genxI && gyI == genyI
    print("  G + O = G: \(gPlusIdOk ? "PASS" : "FAIL")")

    // 2*G = G + G
    let g2 = ed25519PointDouble(genExt)
    let g2add = ed25519PointAdd(genExt, genExt)
    let g2Aff = ed25519PointToAffine(g2)
    let g2addAff = ed25519PointToAffine(g2add)
    let dblOk = ed25519FpToInt(g2Aff.x) == ed25519FpToInt(g2addAff.x) &&
                ed25519FpToInt(g2Aff.y) == ed25519FpToInt(g2addAff.y)
    print("  2*G == G+G: \(dblOk ? "PASS" : "FAIL")")

    // 2G on curve
    let g2OnCurve = ed25519PointOnCurve(g2Aff)
    print("  2*G on curve: \(g2OnCurve ? "PASS" : "FAIL")")

    // Scalar mul: 5*G = G + G + G + G + G
    let g5 = ed25519PointMulInt(genExt, 5)
    var g5add = ed25519PointIdentity()
    for _ in 0..<5 { g5add = ed25519PointAdd(g5add, genExt) }
    let g5Aff = ed25519PointToAffine(g5)
    let g5addAff = ed25519PointToAffine(g5add)
    let scalarMulOk = ed25519FpToInt(g5Aff.x) == ed25519FpToInt(g5addAff.x) &&
                      ed25519FpToInt(g5Aff.y) == ed25519FpToInt(g5addAff.y)
    print("  5*G == G+G+G+G+G: \(scalarMulOk ? "PASS" : "FAIL")")

    // Negation: G + (-G) = identity
    let negG = ed25519PointNeg(genExt)
    let gPlusNegG = ed25519PointAdd(genExt, negG)
    let negOk = ed25519PointIsIdentity(gPlusNegG)
    print("  G + (-G) = O: \(negOk ? "PASS" : "FAIL")")

    // Point encoding/decoding round-trip
    let encoded = ed25519PointEncode(gen)
    if let decoded = ed25519PointDecode(encoded) {
        let encDecOk = ed25519FpToInt(decoded.x) == ed25519FpToInt(gen.x) &&
                       ed25519FpToInt(decoded.y) == ed25519FpToInt(gen.y)
        print("  Encode/decode round-trip: \(encDecOk ? "PASS" : "FAIL")")
    } else {
        print("  Encode/decode round-trip: FAIL (decode returned nil)")
    }

    // --- EdDSA Sign/Verify ---
    print("\n--- EdDSA Sign/Verify ---")
    do {
        let engine = EdDSAEngine()

        // Deterministic test key
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = 42
        let sk = EdDSASecretKey(seed: seed)
        let pk = sk.publicKey

        // Verify public key is on curve
        let pkOnCurve = ed25519PointOnCurve(pk.point)
        print("  Public key on curve: \(pkOnCurve ? "PASS" : "FAIL")")

        // Sign a message
        let msg = Array("Hello, Ed25519!".utf8)
        let sig = engine.sign(message: msg, secretKey: sk)

        // Verify the signature
        let valid = engine.verify(signature: sig, message: msg, publicKey: pk)
        print("  Sign/verify (valid): \(valid ? "PASS" : "FAIL")")

        // Wrong message
        let wrongMsg = Array("Wrong message".utf8)
        let invalidMsg = engine.verify(signature: sig, message: wrongMsg, publicKey: pk)
        print("  Verify (wrong message): \(!invalidMsg ? "PASS" : "FAIL")")

        // Wrong key
        var seed2 = [UInt8](repeating: 0, count: 32)
        seed2[0] = 99
        let sk2 = EdDSASecretKey(seed: seed2)
        let invalidKey = engine.verify(signature: sig, message: msg, publicKey: sk2.publicKey)
        print("  Verify (wrong key): \(!invalidKey ? "PASS" : "FAIL")")

        // --- Batch verification ---
        print("\n--- Batch Verification ---")
        let batchN = 16
        var sigs = [EdDSASignature]()
        var msgs = [[UInt8]]()
        var pks = [EdDSAPublicKey]()

        for i in 0..<batchN {
            var si = [UInt8](repeating: 0, count: 32)
            si[0] = UInt8(i + 1)
            let ski = EdDSASecretKey(seed: si)
            let msgi = Array("Message \(i)".utf8)
            let sigi = engine.sign(message: msgi, secretKey: ski)
            sigs.append(sigi)
            msgs.append(msgi)
            pks.append(ski.publicKey)
        }

        let batchValid = engine.batchVerify(signatures: sigs, messages: msgs, publicKeys: pks)
        print("  Batch verify \(batchN) valid sigs: \(batchValid ? "PASS" : "FAIL")")

        // Corrupt one signature
        var badSigs = sigs
        let badS = [UInt8](repeating: 0xFF, count: 32)
        badSigs[batchN / 2] = EdDSASignature(r: sigs[batchN / 2].r, s: badS)
        let batchInvalid = engine.batchVerify(signatures: badSigs, messages: msgs, publicKeys: pks)
        print("  Batch detect 1 invalid: \(!batchInvalid ? "PASS" : "FAIL")")

        // --- RFC 8032 Test Vectors ---
        print("\n--- RFC 8032 Test Vectors ---")
        // Test vector 1: empty message
        let tv1Seed: [UInt8] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
            0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x60
        ]
        let tv1ExpectedPK: [UInt8] = [
            0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
            0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a
        ]
        // Debug: verify SHA-512 using known test vector
        // SHA-512("abc") should start with ddaf35a193617aba...
        let abcHash = sha512(Array("abc".utf8))
        let abcExpected = "ddaf35a1"
        let abcGot = abcHash[0..<4].map { String(format: "%02x", $0) }.joined()
        print("  SHA-512('abc') prefix: \(abcGot == abcExpected ? "PASS" : "FAIL: got \(abcGot)")")

        // Check TV1 seed SHA-512
        let tv1h = sha512(tv1Seed)
        let expectedH = "357c83864f2833cb427a2ef1c00a013cfdff2768d980c0a3a520f006904de90f"
        let gotH = tv1h[0..<32].map { String(format: "%02x", $0) }.joined()
        print("  TV1 SHA-512(seed) lo32: \(gotH == expectedH ? "PASS" : "FAIL")")
        if gotH != expectedH {
            print("    got:      \(gotH)")
            print("    expected: \(expectedH)")
            // Print the full seed to verify
            print("    seed:     \(tv1Seed.map { String(format: "%02x", $0) }.joined())")
        }

        let tv1SK = EdDSASecretKey(seed: tv1Seed)

        // Debug: check generator against known value
        let genX = ed25519FpToInt(ed25519Generator().x)
        let expectedGenX: [UInt64] = [0xc9562d608f25d51a, 0x692cc7609525a7b2, 0xc0a4e231fdd6dc5c, 0x216936d3cd6e53fe]
        print("  Gen.x match RFC: \(genX == expectedGenX ? "PASS" : "FAIL: got \(genX.map { String(format: "%016llx", $0) })")")

        let tv1PkMatch = tv1SK.publicKey.encoded == tv1ExpectedPK
        if !tv1PkMatch {
            print("  TV1 public key: FAIL")
            print("    got:      \(tv1SK.publicKey.encoded.map { String(format: "%02x", $0) }.joined())")
            print("    expected: \(tv1ExpectedPK.map { String(format: "%02x", $0) }.joined())")
            // Check scalar
            let scalarInt = ed25519FqToInt(tv1SK.scalar)
            print("    scalar lo: \(String(format: "%016llx", scalarInt[0]))")
        } else {
            print("  TV1 public key: PASS")
        }

        let tv1Sig = engine.sign(message: [], secretKey: tv1SK)
        let tv1Valid = engine.verify(signature: tv1Sig, message: [], publicKey: tv1SK.publicKey)
        print("  TV1 sign+verify empty msg: \(tv1Valid ? "PASS" : "FAIL")")

        // Verify signature bytes match RFC 8032
        let tv1ExpectedSig: [UInt8] = [
            0xe5, 0x56, 0x43, 0x00, 0xc3, 0x60, 0xac, 0x72, 0x90, 0x86, 0xe2, 0xcc, 0x80, 0x6e, 0x82, 0x8a,
            0x84, 0x87, 0x7f, 0x1e, 0xb8, 0xe5, 0xd9, 0x74, 0xd8, 0x73, 0xe0, 0x65, 0x22, 0x49, 0x01, 0x55,
            0x5f, 0xb8, 0x82, 0x15, 0x90, 0xa3, 0x3b, 0xac, 0xc6, 0x1e, 0x39, 0x70, 0x1c, 0xf9, 0xb4, 0x6b,
            0xd2, 0x5b, 0xf5, 0xf0, 0x59, 0x5b, 0xbe, 0x24, 0x65, 0x51, 0x41, 0x43, 0x8e, 0x7a, 0x10, 0x0b
        ]
        let tv1SigMatch = tv1Sig.toBytes() == tv1ExpectedSig
        print("  TV1 signature bytes: \(tv1SigMatch ? "PASS" : "FAIL")")
        if !tv1SigMatch {
            print("    got: \(tv1Sig.toBytes().map { String(format: "%02x", $0) }.joined())")
        }

        // --- Performance ---
        if !skipCPU {
            print("\n--- Performance ---")

            // Field ops
            let fieldRuns = 100000
            let t0 = CFAbsoluteTimeGetCurrent()
            var acc = ed25519FpFromInt(42)
            for _ in 0..<fieldRuns {
                acc = ed25519FpMul(acc, acc)
            }
            _ = acc
            let fieldTime = (CFAbsoluteTimeGetCurrent() - t0) * 1e9 / Double(fieldRuns)
            fputs(String(format: "  Fp mul: %.0f ns/op\n", fieldTime), stderr)

            // Point ops
            let pointRuns = 1000
            let t1 = CFAbsoluteTimeGetCurrent()
            var pAcc = genExt
            for _ in 0..<pointRuns {
                pAcc = ed25519PointDouble(pAcc)
            }
            _ = pAcc
            let dblTime = (CFAbsoluteTimeGetCurrent() - t1) * 1e6 / Double(pointRuns)
            fputs(String(format: "  Point double: %.1f us/op\n", dblTime), stderr)

            let t2 = CFAbsoluteTimeGetCurrent()
            var pAcc2 = genExt
            for _ in 0..<pointRuns {
                pAcc2 = ed25519PointAdd(pAcc2, genExt)
            }
            _ = pAcc2
            let addTime = (CFAbsoluteTimeGetCurrent() - t2) * 1e6 / Double(pointRuns)
            fputs(String(format: "  Point add: %.1f us/op\n", addTime), stderr)

            // Sign/verify
            let sigRuns = 100
            let t3 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<sigRuns {
                let _ = engine.sign(message: msg, secretKey: sk)
            }
            let signTime = (CFAbsoluteTimeGetCurrent() - t3) * 1000 / Double(sigRuns)
            fputs(String(format: "  Sign: %.2f ms/op\n", signTime), stderr)

            let t4 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<sigRuns {
                let _ = engine.verify(signature: sig, message: msg, publicKey: pk)
            }
            let verifyTime = (CFAbsoluteTimeGetCurrent() - t4) * 1000 / Double(sigRuns)
            fputs(String(format: "  Verify: %.2f ms/op\n", verifyTime), stderr)
        }

        // Flush output before MSM
        fflush(stdout)

        // --- MSM ---
        print("\n--- Ed25519 MSM ---")
        do {
            let msmEngine = try Ed25519MSM()

            // Generate test points
            let testSizes = [256, 1024, 4096, 16384]
            var projPoints = [Ed25519PointExtended]()
            projPoints.reserveCapacity(testSizes.last!)
            var pAcc3 = genExt
            for _ in 0..<testSizes.last! {
                projPoints.append(pAcc3)
                pAcc3 = ed25519PointAdd(pAcc3, genExt)
            }
            let allPoints = ed25519BatchToAffine(projPoints)

            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            var allScalars = [[UInt32]]()
            for _ in 0..<testSizes.last! {
                var limbs = [UInt32](repeating: 0, count: 8)
                for j in 0..<8 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
                }
                // Keep scalar < q (clear top bits)
                limbs[7] &= 0x0FFFFFFF
                allScalars.append(limbs)
            }

            for n in testSizes {
                let pts = Array(allPoints.prefix(n))
                let scs = Array(allScalars.prefix(n))

                // Warmup
                let _ = try msmEngine.msm(points: pts, scalars: scs)

                let runs = 3
                var times = [Double]()
                for _ in 0..<runs {
                    let start = CFAbsoluteTimeGetCurrent()
                    let _ = try msmEngine.msm(points: pts, scalars: scs)
                    times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
                }
                times.sort()
                let median = times[runs / 2]
                fputs(String(format: "  MSM %5d pts: %7.1f ms\n", n, median), stderr)
            }
        } catch {
            print("  MSM error: \(error)")
        }

        // Summary
        let allTests = oneOk && zeroOk && addZeroOk && addNegOk && mulOneOk && mulInvOk && distOk &&
                       fqOneOk && fqMulOk && fqInvOk && genOnCurve && isId && gPlusIdOk && dblOk &&
                       g2OnCurve && scalarMulOk && negOk && valid && !invalidMsg && !invalidKey &&
                       batchValid && !batchInvalid && tv1PkMatch && tv1Valid && tv1SigMatch
        print("\n  Ed25519 overall: \(allTests ? "ALL PASS" : "SOME FAILED")")

    }
}
