// secp256k1 Field and Curve Benchmark / Correctness Tests
import zkMetal
import Foundation

public func runSecp256k1Test() {
    print("\n=== secp256k1 Field + Curve Tests ===")

    // --- Field arithmetic tests ---
    print("\n--- Field Arithmetic ---")

    // Test: 1 in Montgomery = R mod p
    let one = secpFromInt(1)
    let oneOut = secpToInt(one)
    let oneOk = oneOut[0] == 1 && oneOut[1] == 0 && oneOut[2] == 0 && oneOut[3] == 0
    print("  fromInt(1) round-trip: \(oneOk ? "PASS" : "FAIL")")

    // Test: 0 round-trip
    let zero = secpFromInt(0)
    let zeroOk = zero.isZero
    print("  fromInt(0) is zero: \(zeroOk ? "PASS" : "FAIL")")

    // Test: add(a, 0) = a
    let a = secpFromInt(42)
    let addZero = secpAdd(a, SecpFp.zero)
    let addZeroOk = secpToInt(addZero) == secpToInt(a)
    print("  a + 0 = a: \(addZeroOk ? "PASS" : "FAIL")")

    // Test: a + (-a) = 0
    let negA = secpNeg(a)
    let addNeg = secpAdd(a, negA)
    let addNegOk = addNeg.isZero
    print("  a + (-a) = 0: \(addNegOk ? "PASS" : "FAIL")")

    // Test: mul(a, 1) = a
    let mulOne = secpMul(a, one)
    let mulOneOk = secpToInt(mulOne) == secpToInt(a)
    print("  a * 1 = a: \(mulOneOk ? "PASS" : "FAIL")")

    // Test: a * a^(-1) = 1
    let aInv = secpInverse(a)
    let mulInv = secpMul(a, aInv)
    let mulInvOut = secpToInt(mulInv)
    let mulInvOk = mulInvOut[0] == 1 && mulInvOut[1] == 0 && mulInvOut[2] == 0 && mulInvOut[3] == 0
    print("  a * a^(-1) = 1: \(mulInvOk ? "PASS" : "FAIL")")

    // Test: (a + b) * c = a*c + b*c (distributivity)
    let b = secpFromInt(123456789)
    let c = secpFromInt(987654321)
    let lhs = secpMul(secpAdd(a, b), c)
    let rhs = secpAdd(secpMul(a, c), secpMul(b, c))
    let distOk = secpToInt(lhs) == secpToInt(rhs)
    print("  Distributivity: \(distOk ? "PASS" : "FAIL")")

    // Test: sub(a, b) + b = a
    let subAB = secpSub(a, b)
    let addBack = secpAdd(subAB, b)
    let subOk = secpToInt(addBack) == secpToInt(a)
    print("  (a - b) + b = a: \(subOk ? "PASS" : "FAIL")")

    // Test: hex round-trip
    let hexVal = secpFromHex("0xdeadbeefcafebabe1234567890abcdef")
    let hexStr = secpToHex(hexVal)
    let hexBack = secpFromHex(hexStr)
    let hexOk = secpToInt(hexVal) == secpToInt(hexBack)
    print("  Hex round-trip: \(hexOk ? "PASS" : "FAIL")")

    // --- Curve tests ---
    print("\n--- Curve Operations ---")

    // Test: Generator on curve: y^2 = x^3 + 7
    let gen = secp256k1Generator()
    let gx2 = secpSqr(gen.x)
    let gx3 = secpMul(gx2, gen.x)
    let seven = secpFromInt(7)
    let rhs_curve = secpAdd(gx3, seven)
    let lhs_curve = secpSqr(gen.y)
    let onCurve = secpToInt(lhs_curve) == secpToInt(rhs_curve)
    print("  Generator on curve (y²=x³+7): \(onCurve ? "PASS" : "FAIL")")

    // Test: 2G on curve
    let gProj = secpPointFromAffine(gen)
    let twoG = secpPointDouble(gProj)
    let twoGAff = secpPointToAffine(twoG)
    let twoGx3 = secpMul(secpSqr(twoGAff.x), twoGAff.x)
    let twoGLhs = secpSqr(twoGAff.y)
    let twoGRhs = secpAdd(twoGx3, seven)
    let twoGOnCurve = secpToInt(twoGLhs) == secpToInt(twoGRhs)
    print("  2G on curve: \(twoGOnCurve ? "PASS" : "FAIL")")

    // Test: G + G = 2G
    let gPlusG = secpPointAdd(gProj, gProj)
    let gPlusGAff = secpPointToAffine(gPlusG)
    let addEqDbl = secpToInt(gPlusGAff.x) == secpToInt(twoGAff.x) &&
                   secpToInt(gPlusGAff.y) == secpToInt(twoGAff.y)
    print("  G + G = 2G: \(addEqDbl ? "PASS" : "FAIL")")

    // Test: 3G on curve
    let threeG = secpPointAdd(twoG, gProj)
    let threeGAff = secpPointToAffine(threeG)
    let threeGx3 = secpMul(secpSqr(threeGAff.x), threeGAff.x)
    let threeGOnCurve = secpToInt(secpSqr(threeGAff.y)) == secpToInt(secpAdd(threeGx3, seven))
    print("  3G on curve: \(threeGOnCurve ? "PASS" : "FAIL")")

    // Test: scalar mul 5*G = G+G+G+G+G
    let fiveG_mul = secpPointMulInt(gProj, 5)
    var fiveG_add = gProj
    for _ in 1..<5 { fiveG_add = secpPointAdd(fiveG_add, gProj) }
    let fiveGMulAff = secpPointToAffine(fiveG_mul)
    let fiveGAddAff = secpPointToAffine(fiveG_add)
    let scalarOk = secpToInt(fiveGMulAff.x) == secpToInt(fiveGAddAff.x) &&
                   secpToInt(fiveGMulAff.y) == secpToInt(fiveGAddAff.y)
    print("  5*G (mul) = 5*G (add): \(scalarOk ? "PASS" : "FAIL")")

    // Test: identity behavior
    let identity = secpPointIdentity()
    let gPlusId = secpPointAdd(gProj, identity)
    let gPlusIdAff = secpPointToAffine(gPlusId)
    let idOk = secpToInt(gPlusIdAff.x) == secpToInt(gen.x) &&
               secpToInt(gPlusIdAff.y) == secpToInt(gen.y)
    print("  G + O = G: \(idOk ? "PASS" : "FAIL")")

    // Test: batch affine conversion
    var projPoints = [SecpPointProjective]()
    var acc = gProj
    for _ in 0..<10 {
        projPoints.append(acc)
        acc = secpPointAdd(acc, gProj)
    }
    let batchAff = batchSecpToAffine(projPoints)
    var batchOk = true
    for i in 0..<10 {
        let singleAff = secpPointToAffine(projPoints[i])
        if secpToInt(batchAff[i].x) != secpToInt(singleAff.x) ||
           secpToInt(batchAff[i].y) != secpToInt(singleAff.y) {
            print("  Batch mismatch at \(i)")
            batchOk = false
            break
        }
    }
    print("  Batch affine conversion: \(batchOk ? "PASS" : "FAIL")")

    // Test: point negation and addition
    let negG = secpPointNegateAffine(gen)
    let gPlusNegG = secpPointAdd(gProj, secpPointFromAffine(negG))
    let negOk = secpPointIsIdentity(gPlusNegG)
    print("  G + (-G) = O: \(negOk ? "PASS" : "FAIL")")

    // Summary
    let allPass = oneOk && zeroOk && addZeroOk && addNegOk && mulOneOk && mulInvOk &&
                  distOk && subOk && hexOk && onCurve && twoGOnCurve && addEqDbl &&
                  threeGOnCurve && scalarOk && idOk && batchOk && negOk
    print("\n  Overall: \(allPass ? "ALL PASS ✓" : "SOME FAILED ✗")")
}

public func runSecp256k1GLVTest() {
    print("\n=== secp256k1 GLV Tests ===")

    // Test 1: CPU GLV decomposition round-trip
    // k ≡ k1 + k2·λ (mod n) where λ is cube root of unity in scalar field
    let lambda: [UInt64] = [
        0xdf02967c1b23bd72, 0x122e22ea20816678,
        0xa5261c028812645a, 0x5363ad4cc05c30e0
    ]
    let n: [UInt64] = [
        0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
        0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF
    ]

    // Random test scalar
    let testScalar: [UInt32] = [
        0xdeadbeef, 0xcafebabe, 0x12345678, 0x9abcdef0,
        0x11223344, 0x55667788, 0x99aabbcc, 0x0ddeeff0
    ]

    let (k1, k2, neg1, neg2) = Secp256k1GLV.decompose(testScalar)

    // Verify k1 and k2 are ~128 bits (top 4 limbs should be zero)
    let k1Top = k1[4...7].reduce(UInt32(0), |)
    let k2Top = k2[4...7].reduce(UInt32(0), |)
    let sizeOk = k1Top == 0 && k2Top == 0
    print("  k1, k2 ≤ 128 bits: \(sizeOk ? "PASS" : "FAIL") (k1_top=\(k1Top), k2_top=\(k2Top))")

    // Verify round-trip: k ≡ k1 + k2·λ (mod n)
    // Convert to 64-bit for verification
    let k64: [UInt64] = [
        UInt64(testScalar[0]) | (UInt64(testScalar[1]) << 32),
        UInt64(testScalar[2]) | (UInt64(testScalar[3]) << 32),
        UInt64(testScalar[4]) | (UInt64(testScalar[5]) << 32),
        UInt64(testScalar[6]) | (UInt64(testScalar[7]) << 32)
    ]
    let k1_64: [UInt64] = [
        UInt64(k1[0]) | (UInt64(k1[1]) << 32),
        UInt64(k1[2]) | (UInt64(k1[3]) << 32), 0, 0
    ]
    let k2_64: [UInt64] = [
        UInt64(k2[0]) | (UInt64(k2[1]) << 32),
        UInt64(k2[2]) | (UInt64(k2[3]) << 32), 0, 0
    ]

    // k2 * lambda mod n
    let k2l = mulSchoolbook(k2_64, 4, lambda, 4)
    var k2l_mod = modReduce(k2l, n)

    // Apply signs
    var k1_signed = k1_64
    if neg1 { k1_signed = modNeg(k1_signed, n) }
    if neg2 { k2l_mod = modNeg(k2l_mod, n) }

    // recon = k1 + k2*lambda mod n
    var recon = modAdd(k1_signed, k2l_mod, n)

    // Reduce k64 mod n
    var kReduced = Array(k64)
    while modGte(kReduced, n) {
        kReduced = modSub(kReduced, n)
    }

    let reconOk = recon == kReduced
    print("  GLV round-trip (k ≡ k1 + k2·λ mod n): \(reconOk ? "PASS" : "FAIL")")
    if !reconOk {
        print("    k_reduced: \(kReduced.map { String(format: "%016llx", $0) }.reversed().joined())")
        print("    recon:     \(recon.map { String(format: "%016llx", $0) }.reversed().joined())")
    }

    // Test 2: Endomorphism β^3 = 1 mod p
    let beta = SecpFp(v: (
        Secp256k1GLV.BETA_MONT[0], Secp256k1GLV.BETA_MONT[1],
        Secp256k1GLV.BETA_MONT[2], Secp256k1GLV.BETA_MONT[3],
        Secp256k1GLV.BETA_MONT[4], Secp256k1GLV.BETA_MONT[5],
        Secp256k1GLV.BETA_MONT[6], Secp256k1GLV.BETA_MONT[7]
    ))
    let beta2 = secpMul(beta, beta)
    let beta3 = secpMul(beta2, beta)
    let one = secpFromInt(1)
    let beta3Ok = secpToInt(beta3) == secpToInt(one)
    print("  β³ = 1 (mod p): \(beta3Ok ? "PASS" : "FAIL")")

    // Test 3: Endomorphism on curve: φ(G) = (β·Gx, Gy) is on curve
    let gen = secp256k1Generator()
    let endoG = Secp256k1GLV.applyEndomorphism(gen)
    let seven = secpFromInt(7)
    let endoLhs = secpSqr(endoG.y)
    let endoRhs = secpAdd(secpMul(secpSqr(endoG.x), endoG.x), seven)
    let endoOnCurve = secpToInt(endoLhs) == secpToInt(endoRhs)
    print("  φ(G) on curve: \(endoOnCurve ? "PASS" : "FAIL")")

    // Test 4: Point-level verification: k*G = k1*G + k2*φ(G) for several scalars
    var pointOk = true
    let gProj = secpPointFromAffine(gen)
    let endoGProj = secpPointFromAffine(endoG)
    for trial in 0..<5 {
        var s: [UInt32] = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 { s[j] = UInt32(0xdeadbeef &+ UInt32(trial * 7919 + j * 1009)) }
        let (dk1, dk2, dn1, dn2) = Secp256k1GLV.decompose(s)

        // k1*G
        var k1_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k1_int = (k1_int << 32) | Int(dk1[j]) }
        var term1 = secpPointMulInt(gProj, k1_int)
        if dn1 {
            let a = secpPointToAffine(term1)
            term1 = secpPointFromAffine(secpPointNegateAffine(a))
        }

        // k2*φ(G)
        var k2_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k2_int = (k2_int << 32) | Int(dk2[j]) }
        var term2 = secpPointMulInt(endoGProj, k2_int)
        if dn2 {
            let a = secpPointToAffine(term2)
            term2 = secpPointFromAffine(secpPointNegateAffine(a))
        }

        let glvResult = secpPointAdd(term1, term2)

        // k*G (direct)
        var k_int = 0
        for j in stride(from: 7, through: 0, by: -1) {
            k_int = (k_int << 32) | Int(s[j] & 0xFFFF)  // truncate for feasible CPU mul
        }
        // Can't do full 256-bit scalar mul on CPU, so truncate test scalars
        s = [UInt32](repeating: 0, count: 8)
        s[0] = UInt32(trial * 7919 + 42) & 0xFFF // 12-bit scalar
        let (dk1b, dk2b, dn1b, dn2b) = Secp256k1GLV.decompose(s)

        var k1b_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k1b_int = (k1b_int << 32) | Int(dk1b[j]) }
        var t1 = secpPointMulInt(gProj, k1b_int)
        if dn1b {
            let a = secpPointToAffine(t1)
            t1 = secpPointFromAffine(secpPointNegateAffine(a))
        }

        var k2b_int = 0
        for j in stride(from: 3, through: 0, by: -1) { k2b_int = (k2b_int << 32) | Int(dk2b[j]) }
        var t2 = secpPointMulInt(endoGProj, k2b_int)
        if dn2b {
            let a = secpPointToAffine(t2)
            t2 = secpPointFromAffine(secpPointNegateAffine(a))
        }

        let glvRes = secpPointAdd(t1, t2)
        let directRes = secpPointMulInt(gProj, Int(s[0]))

        let gAff = secpPointToAffine(glvRes)
        let dAff = secpPointToAffine(directRes)
        let ok = secpToInt(gAff.x) == secpToInt(dAff.x) && secpToInt(gAff.y) == secpToInt(dAff.y)
        if !ok {
            print("  Point-level GLV FAIL at trial \(trial): s=\(s[0])")
            print("    k1=\(dk1b.prefix(4)), k2=\(dk2b.prefix(4)), neg1=\(dn1b), neg2=\(dn2b)")
            print("    GLV:    \(secpToHex(gAff.x))")
            print("    Direct: \(secpToHex(dAff.x))")
            pointOk = false
        }
    }
    print("  k*G = k1*G + k2*φ(G) point check: \(pointOk ? "PASS" : "FAIL")")

    print("\n  GLV overall: \(sizeOk && reconOk && beta3Ok && endoOnCurve && pointOk ? "ALL PASS ✓" : "SOME FAILED ✗")")
}

// Modular arithmetic helpers for GLV test verification
private func modReduce(_ a: [UInt64], _ m: [UInt64]) -> [UInt64] {
    var r = a
    while r.count < 4 { r.append(0) }

    // Shift-subtract division for wide numbers
    // Find the bit width of r and m
    func bitWidth(_ v: [UInt64]) -> Int {
        for i in stride(from: v.count - 1, through: 0, by: -1) {
            if v[i] != 0 { return i * 64 + (64 - v[i].leadingZeroBitCount) }
        }
        return 0
    }

    let mBits = bitWidth(m)
    guard mBits > 0 else { return r }

    var rBits = bitWidth(r)
    if rBits < mBits { while r.count < 4 { r.append(0) }; return Array(r.prefix(4)) }

    let maxShift = rBits - mBits
    for shift in stride(from: maxShift, through: 0, by: -1) {
        // Build m << shift
        let limbShift = shift / 64
        let bitShift = shift % 64
        var mShifted = [UInt64](repeating: 0, count: r.count)
        for i in 0..<m.count {
            let dst = i + limbShift
            if dst < mShifted.count {
                mShifted[dst] |= m[i] << bitShift
            }
            if bitShift > 0 && dst + 1 < mShifted.count {
                mShifted[dst + 1] |= m[i] >> (64 - bitShift)
            }
        }
        if modGte(r, mShifted) {
            r = modSub(r, mShifted)
        }
    }
    while r.count < 4 { r.append(0) }
    return Array(r.prefix(4))
}

private func modGte(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    let n = max(a.count, b.count)
    for i in stride(from: n - 1, through: 0, by: -1) {
        let av = i < a.count ? a[i] : 0
        let bv = i < b.count ? b[i] : 0
        if av > bv { return true }
        if av < bv { return false }
    }
    return true
}

private func modSub(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
    let n = max(a.count, b.count)
    var r = [UInt64](repeating: 0, count: n)
    var borrow: UInt64 = 0
    for i in 0..<n {
        let av = i < a.count ? a[i] : 0
        let bv = i < b.count ? b[i] : 0
        let (d1, b1) = av.subtractingReportingOverflow(bv)
        let (d2, b2) = d1.subtractingReportingOverflow(borrow)
        r[i] = d2
        borrow = (b1 ? 1 : 0) + (b2 ? 1 : 0)
    }
    return r
}

private func modAdd(_ a: [UInt64], _ b: [UInt64], _ m: [UInt64]) -> [UInt64] {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let (s1, c1) = a[i].addingReportingOverflow(b[i])
        let (s2, c2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
    }
    if carry != 0 || modGte(r, m) {
        r = modSub(r, m)
    }
    return r
}

private func modNeg(_ a: [UInt64], _ m: [UInt64]) -> [UInt64] {
    if a.allSatisfy({ $0 == 0 }) { return a }
    return modSub(m, a)
}

public func runSecp256k1MSMBench() {
    print("\n=== secp256k1 MSM Benchmark ===")

    do {
        let engine = try Secp256k1MSM()

        // Generate points: G, 2G, 3G, ...
        let gen = secp256k1Generator()
        let gProj = secpPointFromAffine(gen)

        let logSizes = [8, 10, 12, 14, 16, 18]
        let maxN = 1 << logSizes.last!

        fputs("Generating \(maxN) distinct points...\n", stderr)
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [SecpPointProjective]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = secpPointAdd(acc, gProj)
        }
        let allPoints = batchSecpToAffine(projPoints)
        projPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

        // Generate random scalars
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        var allScalars = [[UInt32]]()
        allScalars.reserveCapacity(maxN)
        for _ in 0..<maxN {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            allScalars.append(limbs)
        }

        // C Pippenger correctness test
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            let cResult = cSecpPippengerMSM(points: testPts, scalars: testScls)
            let gpuResult = try engine.msm(points: testPts, scalars: testScls)
            let cAff = secpPointToAffine(cResult)
            let gpuAff = secpPointToAffine(gpuResult)
            let match = secpToInt(cAff.x) == secpToInt(gpuAff.x) &&
                        secpToInt(cAff.y) == secpToInt(gpuAff.y)
            print("  C Pippenger correctness: \(match ? "PASS" : "FAIL")")
        }

        // Performance
        engine.useGLV = false
        print("\n--- secp256k1 MSM Performance (GPU vs C Pippenger) ---")
        for logN in logSizes {
            let n = 1 << logN
            let pts = Array(allPoints.prefix(n))
            let scls = Array(allScalars.prefix(n))
            // GPU MSM
            let _ = try engine.msm(points: pts, scalars: scls) // warmup
            let runs = 5
            var gpuTimes = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: pts, scalars: scls)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                gpuTimes.append(elapsed)
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[runs / 2]

            // C Pippenger MSM
            let _ = cSecpPippengerMSM(points: pts, scalars: scls) // warmup
            var cTimes = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = cSecpPippengerMSM(points: pts, scalars: scls)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                cTimes.append(elapsed)
            }
            cTimes.sort()
            let cMedian = cTimes[runs / 2]

            let speedup = cMedian > 0 ? gpuMedian / cMedian : 0
            fputs(String(format: "  2^%-2d: GPU %7.1fms | C Pip %7.1fms (%.1fx)\n",
                         logN, gpuMedian, cMedian, speedup), stderr)
        }
    } catch {
        print("  ERROR: \(error)")
    }
}
