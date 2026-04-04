// BabyJubjub Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runBabyJubjubBench() {
    print("\n=== BabyJubjub Curve Tests ===")
    fflush(stdout)

    // --- Curve constants ---
    print("\n--- Curve Constants ---")
    let aConst = bjjA()
    let dConst = bjjD()
    let aInt = frToInt(aConst)
    let dInt = frToInt(dConst)
    print("  a = 168700: \(aInt[0] == 168700 && aInt[1] == 0 ? "PASS" : "FAIL")")
    print("  d = 168696: \(dInt[0] == 168696 && dInt[1] == 0 ? "PASS" : "FAIL")")

    // --- Generator point ---
    print("\n--- Generator Point ---")
    let gen = bjjGenerator()
    let genOnCurve = bjjPointOnCurve(gen)
    print("  Generator on curve: \(genOnCurve ? "PASS" : "FAIL")")

    // Check known x,y coordinates
    let gx = frToInt(gen.x)
    let gy = frToInt(gen.y)
    let gxOk = gx[0] == 0x2893f3f6bb957051 && gx[1] == 0x2ab8d8010534e0b6
    let gyOk = gy[0] == 0x4b3c257a872d7d8b && gy[1] == 0xfce0051fb9e13377
    print("  Generator x matches: \(gxOk ? "PASS" : "FAIL")")
    print("  Generator y matches: \(gyOk ? "PASS" : "FAIL")")

    // --- Identity ---
    print("\n--- Identity Point ---")
    let id = bjjPointIdentity()
    let isId = bjjPointIsIdentity(id)
    print("  Identity check: \(isId ? "PASS" : "FAIL")")

    // G + identity = G
    let genExt = bjjPointFromAffine(gen)
    let gPlusId = bjjPointAdd(genExt, id)
    let gPlusIdAff = bjjPointToAffine(gPlusId)
    let gPlusIdOk = frToInt(gPlusIdAff.x) == frToInt(gen.x) &&
                    frToInt(gPlusIdAff.y) == frToInt(gen.y)
    print("  G + identity = G: \(gPlusIdOk ? "PASS" : "FAIL")")

    // --- Point operations ---
    print("\n--- Point Operations ---")

    // G + G = 2G
    let g2 = bjjPointAdd(genExt, genExt)
    let g2d = bjjPointDouble(genExt)
    let g2aff = bjjPointToAffine(g2)
    let g2daff = bjjPointToAffine(g2d)
    let doubleOk = frToInt(g2aff.x) == frToInt(g2daff.x) &&
                   frToInt(g2aff.y) == frToInt(g2daff.y)
    print("  G + G == 2G (double): \(doubleOk ? "PASS" : "FAIL")")

    // 2G is on curve
    print("  2G on curve: \(bjjPointOnCurve(g2aff) ? "PASS" : "FAIL")")

    // G + (-G) = identity
    let negG = bjjPointNeg(genExt)
    let gPlusNegG = bjjPointAdd(genExt, negG)
    let zeroOk = bjjPointIsIdentity(gPlusNegG)
    print("  G + (-G) = identity: \(zeroOk ? "PASS" : "FAIL")")

    // Scalar mul: 3*G = G + G + G
    let g3 = bjjPointMulInt(genExt, 3)
    let g3manual = bjjPointAdd(g2, genExt)
    let g3aff = bjjPointToAffine(g3)
    let g3maff = bjjPointToAffine(g3manual)
    let scalMulOk = frToInt(g3aff.x) == frToInt(g3maff.x) &&
                    frToInt(g3aff.y) == frToInt(g3maff.y)
    print("  3*G == G+G+G: \(scalMulOk ? "PASS" : "FAIL")")

    // --- Point addition performance ---
    print("\n--- Performance ---")

    let warmup = 100
    for _ in 0..<warmup {
        _ = bjjPointAdd(genExt, g2)
    }

    // Point addition
    let addRuns = 10000
    var acc = genExt
    let addT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<addRuns {
        acc = bjjPointAdd(acc, genExt)
    }
    let addTime = (CFAbsoluteTimeGetCurrent() - addT0) * 1_000_000 / Double(addRuns)
    print(String(format: "  Point add:    %.2f us/op (%d ops)", addTime, addRuns))
    _ = acc  // prevent opt

    // Point doubling
    acc = genExt
    let dblT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<addRuns {
        acc = bjjPointDouble(acc)
    }
    let dblTime = (CFAbsoluteTimeGetCurrent() - dblT0) * 1_000_000 / Double(addRuns)
    print(String(format: "  Point double: %.2f us/op (%d ops)", dblTime, addRuns))
    _ = acc

    // Scalar multiplication
    let smRuns = 100
    let testScalar: [UInt64] = [0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF, 0, 0]
    let smT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<smRuns {
        _ = bjjPointMulScalar(genExt, testScalar)
    }
    let smTime = (CFAbsoluteTimeGetCurrent() - smT0) * 1000 / Double(smRuns)
    print(String(format: "  Scalar mul:   %.2f ms/op (%d ops)", smTime, smRuns))

    // --- frSqrt test ---
    print("\n--- Field sqrt ---")
    let four = frFromInt(4)
    if let sqrtFour = frSqrt(four) {
        let check = frSqr(sqrtFour)
        let sqrtOk = frToInt(check) == frToInt(four)
        print("  sqrt(4)^2 = 4: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  sqrt(4) = NONE: FAIL")
    }

    let nine = frFromInt(9)
    if let sqrtNine = frSqrt(nine) {
        let check = frSqr(sqrtNine)
        let sqrtOk = frToInt(check) == frToInt(nine)
        print("  sqrt(9)^2 = 9: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  sqrt(9) = NONE: FAIL")
    }

    // --- Pedersen Hash ---
    print("\n--- Pedersen Hash ---")

    let pedT0 = CFAbsoluteTimeGetCurrent()
    let pedersen = PedersenBJJ(numGenerators: 62)
    let pedInitTime = (CFAbsoluteTimeGetCurrent() - pedT0) * 1000
    print(String(format: "  Init (62 generators): %.1f ms", pedInitTime))

    // Determinism: hash same data twice
    let testData: [UInt8] = [1, 2, 3, 4, 5, 6, 7, 8]
    let h1 = pedersen.hash(testData)
    let h2 = pedersen.hash(testData)
    let detOk = frToInt(h1.x) == frToInt(h2.x) && frToInt(h1.y) == frToInt(h2.y)
    print("  Deterministic: \(detOk ? "PASS" : "FAIL")")

    // Result is on curve
    let hashOnCurve = bjjPointOnCurve(h1)
    print("  Hash on curve: \(hashOnCurve ? "PASS" : "FAIL")")

    // Different data -> different hash
    let testData2: [UInt8] = [8, 7, 6, 5, 4, 3, 2, 1]
    let h3 = pedersen.hash(testData2)
    let diffOk = frToInt(h1.x) != frToInt(h3.x) || frToInt(h1.y) != frToInt(h3.y)
    print("  Different inputs differ: \(diffOk ? "PASS" : "FAIL")")

    // Pedersen hash benchmarks
    for size in [8, 16, 31] {
        let data = [UInt8](repeating: 0xAB, count: size)
        let runs = 100
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runs {
            _ = pedersen.hash(data)
        }
        let time = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(runs)
        print(String(format: "  Hash %d bytes: %.2f ms/op", size, time))
    }

    // --- EdDSA ---
    print("\n--- EdDSA over BabyJubjub ---")

    let eddsa = BabyJubjubEdDSA()

    // Generate key from known scalar
    let sk = BJJSecretKey(scalarValue: 12345)
    let pk = sk.publicKey
    print("  Key generation: PASS")

    // Public key on curve
    let pkOnCurve = bjjPointOnCurve(pk.point)
    print("  Public key on curve: \(pkOnCurve ? "PASS" : "FAIL")")

    // Sign & verify
    let msg = [frFromInt(42), frFromInt(123)]
    let sig = eddsa.sign(message: msg, secretKey: sk)

    let verifyOk = eddsa.verify(signature: sig, message: msg, publicKey: pk)
    print("  Sign/verify round-trip: \(verifyOk ? "PASS" : "FAIL")")

    // Invalid signature rejection
    let wrongMsg = [frFromInt(999)]
    let verifyWrong = eddsa.verify(signature: sig, message: wrongMsg, publicKey: pk)
    print("  Invalid msg rejected: \(!verifyWrong ? "PASS" : "FAIL")")

    // Wrong public key
    let sk2 = BJJSecretKey(scalarValue: 99999)
    let verifyWrongPK = eddsa.verify(signature: sig, message: msg, publicKey: sk2.publicKey)
    print("  Wrong PK rejected: \(!verifyWrongPK ? "PASS" : "FAIL")")

    // Sign/verify performance
    let signRuns = 20
    let signT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<signRuns {
        _ = eddsa.sign(message: msg, secretKey: sk)
    }
    let signTime = (CFAbsoluteTimeGetCurrent() - signT0) * 1000 / Double(signRuns)
    print(String(format: "  Sign:   %.2f ms/op", signTime))

    let verifyRuns = 20
    let verifyT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<verifyRuns {
        _ = eddsa.verify(signature: sig, message: msg, publicKey: pk)
    }
    let verifyTime = (CFAbsoluteTimeGetCurrent() - verifyT0) * 1000 / Double(verifyRuns)
    print(String(format: "  Verify: %.2f ms/op", verifyTime))

    // Batch verify
    let batchSize = 10
    var sigs = [BJJSignature]()
    var msgs = [[Fr]]()
    var pks = [BJJPublicKey]()
    for i in 0..<batchSize {
        let ski = BJJSecretKey(scalarValue: UInt64(1000 + i))
        let msgi = [frFromInt(UInt64(i + 1))]
        let sigi = eddsa.sign(message: msgi, secretKey: ski)
        sigs.append(sigi)
        msgs.append(msgi)
        pks.append(ski.publicKey)
    }

    let batchOk = eddsa.batchVerify(signatures: sigs, messages: msgs, publicKeys: pks)
    print("  Batch verify (\(batchSize)): \(batchOk ? "PASS" : "FAIL")")

    let batchT0 = CFAbsoluteTimeGetCurrent()
    _ = eddsa.batchVerify(signatures: sigs, messages: msgs, publicKeys: pks)
    let batchTime = (CFAbsoluteTimeGetCurrent() - batchT0) * 1000
    print(String(format: "  Batch verify %d sigs: %.2f ms", batchSize, batchTime))

    print("\n=== BabyJubjub Tests Complete ===")
    fflush(stdout)
}
