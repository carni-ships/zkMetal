// Jubjub Curve Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runJubjubBench() {
    print("\n=== Jubjub Curve Tests ===")
    fflush(stdout)

    // --- Curve constants ---
    print("\n--- Curve Constants ---")
    let aConst = jubjubA()
    let dConst = jubjubD()
    let aInt = fr381ToInt(aConst)
    let dInt = fr381ToInt(dConst)
    // a = -1 mod p = p - 1
    let pMinus1: [UInt64] = [
        0xffffffff00000000, 0x53bda402fffe5bfe,
        0x3339d80809a1d805, 0x73eda753299d7d48
    ]
    let aOk = aInt[0] == pMinus1[0] && aInt[1] == pMinus1[1] &&
              aInt[2] == pMinus1[2] && aInt[3] == pMinus1[3]
    print("  a = -1 (p-1): \(aOk ? "PASS" : "FAIL")")

    // d = 19257038036680949359750312669786877991949435402254120286184196891950884077233
    let dExpected: [UInt64] = [
        0x01065fd6d6343eb1, 0x292d7f6d37579d26,
        0xf5fd9207e6bd7fd4, 0x2a9318e74bfa2b48
    ]
    let dOk = dInt[0] == dExpected[0] && dInt[1] == dExpected[1] &&
              dInt[2] == dExpected[2] && dInt[3] == dExpected[3]
    print("  d matches spec: \(dOk ? "PASS" : "FAIL")")

    // --- Generator point ---
    print("\n--- Generator Point ---")
    let gen = jubjubGenerator()
    let genOnCurve = jubjubPointOnCurve(gen)
    print("  Generator on curve: \(genOnCurve ? "PASS" : "FAIL")")

    // Check known coordinates
    let gx = fr381ToInt(gen.x)
    let gy = fr381ToInt(gen.y)
    let gxOk = gx[0] == 0x7f4ecf1a74f976c4 && gx[3] == 0x5183972af8eff38c
    let gyOk = gy[0] == 0x146bad709349702e && gy[3] == 0x3b43f8472ca2fc2c
    print("  Generator x matches: \(gxOk ? "PASS" : "FAIL")")
    print("  Generator y matches: \(gyOk ? "PASS" : "FAIL")")

    // --- Identity ---
    print("\n--- Identity Point ---")
    let id = jubjubPointIdentity()
    let isId = jubjubPointIsIdentity(id)
    print("  Identity check: \(isId ? "PASS" : "FAIL")")

    // G + identity = G
    let genExt = jubjubPointFromAffine(gen)
    let gPlusId = jubjubPointAdd(genExt, id)
    let gPlusIdAff = jubjubPointToAffine(gPlusId)
    let gPlusIdOk = fr381ToInt(gPlusIdAff.x) == fr381ToInt(gen.x) &&
                    fr381ToInt(gPlusIdAff.y) == fr381ToInt(gen.y)
    print("  G + identity = G: \(gPlusIdOk ? "PASS" : "FAIL")")

    // --- Point operations ---
    print("\n--- Point Operations ---")

    // G + G = 2G
    let g2 = jubjubPointAdd(genExt, genExt)
    let g2d = jubjubPointDouble(genExt)
    let g2aff = jubjubPointToAffine(g2)
    let g2daff = jubjubPointToAffine(g2d)
    let doubleOk = fr381ToInt(g2aff.x) == fr381ToInt(g2daff.x) &&
                   fr381ToInt(g2aff.y) == fr381ToInt(g2daff.y)
    print("  G + G == 2G (double): \(doubleOk ? "PASS" : "FAIL")")

    // 2G is on curve
    print("  2G on curve: \(jubjubPointOnCurve(g2aff) ? "PASS" : "FAIL")")

    // G + (-G) = identity
    let negG = jubjubPointNeg(genExt)
    let gPlusNegG = jubjubPointAdd(genExt, negG)
    let zeroOk = jubjubPointIsIdentity(gPlusNegG)
    print("  G + (-G) = identity: \(zeroOk ? "PASS" : "FAIL")")

    // Scalar mul: 3*G = G + G + G
    let g3 = jubjubPointMulInt(genExt, 3)
    let g3manual = jubjubPointAdd(g2, genExt)
    let g3aff = jubjubPointToAffine(g3)
    let g3maff = jubjubPointToAffine(g3manual)
    let scalMulOk = fr381ToInt(g3aff.x) == fr381ToInt(g3maff.x) &&
                    fr381ToInt(g3aff.y) == fr381ToInt(g3maff.y)
    print("  3*G == G+G+G: \(scalMulOk ? "PASS" : "FAIL")")

    // 3G on curve
    print("  3G on curve: \(jubjubPointOnCurve(g3aff) ? "PASS" : "FAIL")")

    // Subgroup order: r_s * G = identity
    let rsG = jubjubPointMulScalar(genExt, JUBJUB_SUBGROUP_ORDER)
    let orderOk = jubjubPointIsIdentity(rsG)
    print("  r_s * G = identity: \(orderOk ? "PASS" : "FAIL")")

    // --- Batch to affine ---
    print("\n--- Batch Operations ---")
    let pts = [genExt, g2, g3]
    let affPts = jubjubBatchToAffine(pts)
    let batchOk = fr381ToInt(affPts[0].x) == fr381ToInt(gen.x) &&
                  fr381ToInt(affPts[1].x) == fr381ToInt(g2aff.x) &&
                  fr381ToInt(affPts[2].x) == fr381ToInt(g3aff.x)
    print("  Batch to affine (3 pts): \(batchOk ? "PASS" : "FAIL")")

    // --- Encode / Decode ---
    print("\n--- Encoding ---")
    let encoded = jubjubPointEncode(gen)
    if let decoded = jubjubPointDecode(encoded) {
        let encOk = fr381ToInt(decoded.x) == fr381ToInt(gen.x) &&
                    fr381ToInt(decoded.y) == fr381ToInt(gen.y)
        print("  Encode/decode round-trip: \(encOk ? "PASS" : "FAIL")")
    } else {
        print("  Encode/decode round-trip: FAIL (decode returned nil)")
    }

    let compressed = jubjubPointCompress(gen)
    print("  Compressed size: \(compressed.count == 32 ? "PASS" : "FAIL") (\(compressed.count) bytes)")

    // --- Performance ---
    print("\n--- Performance ---")

    let warmup = 100
    for _ in 0..<warmup {
        _ = jubjubPointAdd(genExt, g2)
    }

    // Point addition
    let addRuns = 10000
    var acc = genExt
    let addT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<addRuns {
        acc = jubjubPointAdd(acc, genExt)
    }
    let addTime = (CFAbsoluteTimeGetCurrent() - addT0) * 1_000_000 / Double(addRuns)
    print(String(format: "  Point add:    %.2f us/op (%d ops)", addTime, addRuns))
    _ = acc

    // Point doubling
    acc = genExt
    let dblT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<addRuns {
        acc = jubjubPointDouble(acc)
    }
    let dblTime = (CFAbsoluteTimeGetCurrent() - dblT0) * 1_000_000 / Double(addRuns)
    print(String(format: "  Point double: %.2f us/op (%d ops)", dblTime, addRuns))
    _ = acc

    // Scalar multiplication
    let smRuns = 100
    let testScalar: [UInt64] = [0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF, 0, 0]
    let smT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<smRuns {
        _ = jubjubPointMulScalar(genExt, testScalar)
    }
    let smTime = (CFAbsoluteTimeGetCurrent() - smT0) * 1000 / Double(smRuns)
    print(String(format: "  Scalar mul:   %.2f ms/op (%d ops)", smTime, smRuns))

    // --- Field sqrt ---
    print("\n--- Field sqrt (Fr381) ---")
    let four = fr381FromInt(4)
    if let sqrtFour = fr381Sqrt(four) {
        let check = fr381Sqr(sqrtFour)
        let sqrtOk = fr381ToInt(check) == fr381ToInt(four)
        print("  sqrt(4)^2 = 4: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  sqrt(4) = NONE: FAIL")
    }

    let nine = fr381FromInt(9)
    if let sqrtNine = fr381Sqrt(nine) {
        let check = fr381Sqr(sqrtNine)
        let sqrtOk = fr381ToInt(check) == fr381ToInt(nine)
        print("  sqrt(9)^2 = 9: \(sqrtOk ? "PASS" : "FAIL")")
    } else {
        print("  sqrt(9) = NONE: FAIL")
    }

    print("\n=== Jubjub Tests Complete ===")
    fflush(stdout)
}
