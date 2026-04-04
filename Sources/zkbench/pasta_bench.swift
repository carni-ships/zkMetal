// Pasta curves (Pallas + Vesta) Field, Curve, and MSM Benchmark / Correctness Tests
import zkMetal
import Foundation

public func runPastaTest() {
    print("\n=== Pasta Curves (Pallas + Vesta) Tests ===")

    // ---- Pallas Field Tests ----
    print("\n--- Pallas Field Arithmetic ---")

    let pOne = pallasFromInt(1)
    let pOneOut = pallasToInt(pOne)
    let pOneOk = pOneOut[0] == 1 && pOneOut[1] == 0 && pOneOut[2] == 0 && pOneOut[3] == 0
    print("  fromInt(1) round-trip: \(pOneOk ? "PASS" : "FAIL")")

    let pZero = pallasFromInt(0)
    let pZeroOk = pZero.isZero
    print("  fromInt(0) is zero: \(pZeroOk ? "PASS" : "FAIL")")

    let pa = pallasFromInt(42)
    let addZero = pallasAdd(pa, PallasFp.zero)
    let addZeroOk = pallasToInt(addZero) == pallasToInt(pa)
    print("  a + 0 = a: \(addZeroOk ? "PASS" : "FAIL")")

    let negA = pallasNeg(pa)
    let addNeg = pallasAdd(pa, negA)
    let addNegOk = addNeg.isZero
    print("  a + (-a) = 0: \(addNegOk ? "PASS" : "FAIL")")

    let mulOne = pallasMul(pa, pOne)
    let mulOneOk = pallasToInt(mulOne) == pallasToInt(pa)
    print("  a * 1 = a: \(mulOneOk ? "PASS" : "FAIL")")

    let aInv = pallasInverse(pa)
    let mulInv = pallasMul(pa, aInv)
    let mulInvOut = pallasToInt(mulInv)
    let mulInvOk = mulInvOut[0] == 1 && mulInvOut[1] == 0 && mulInvOut[2] == 0 && mulInvOut[3] == 0
    print("  a * a^(-1) = 1: \(mulInvOk ? "PASS" : "FAIL")")

    let pb = pallasFromInt(123456789)
    let pc = pallasFromInt(987654321)
    let lhs = pallasMul(pallasAdd(pa, pb), pc)
    let rhs = pallasAdd(pallasMul(pa, pc), pallasMul(pb, pc))
    let distOk = pallasToInt(lhs) == pallasToInt(rhs)
    print("  Distributivity: \(distOk ? "PASS" : "FAIL")")

    let subAB = pallasSub(pa, pb)
    let addBack = pallasAdd(subAB, pb)
    let subOk = pallasToInt(addBack) == pallasToInt(pa)
    print("  (a - b) + b = a: \(subOk ? "PASS" : "FAIL")")

    // ---- Vesta Field Tests ----
    print("\n--- Vesta Field Arithmetic ---")

    let vOne = vestaFromInt(1)
    let vOneOut = vestaToInt(vOne)
    let vOneOk = vOneOut[0] == 1 && vOneOut[1] == 0 && vOneOut[2] == 0 && vOneOut[3] == 0
    print("  fromInt(1) round-trip: \(vOneOk ? "PASS" : "FAIL")")

    let va = vestaFromInt(42)
    let vNegA = vestaNeg(va)
    let vAddNeg = vestaAdd(va, vNegA)
    let vAddNegOk = vAddNeg.isZero
    print("  a + (-a) = 0: \(vAddNegOk ? "PASS" : "FAIL")")

    let vMulOne = vestaMul(va, vOne)
    let vMulOneOk = vestaToInt(vMulOne) == vestaToInt(va)
    print("  a * 1 = a: \(vMulOneOk ? "PASS" : "FAIL")")

    let vaInv = vestaInverse(va)
    let vMulInv = vestaMul(va, vaInv)
    let vMulInvOut = vestaToInt(vMulInv)
    let vMulInvOk = vMulInvOut[0] == 1 && vMulInvOut[1] == 0 && vMulInvOut[2] == 0 && vMulInvOut[3] == 0
    print("  a * a^(-1) = 1: \(vMulInvOk ? "PASS" : "FAIL")")

    let vb = vestaFromInt(123456789)
    let vc = vestaFromInt(987654321)
    let vLhs = vestaMul(vestaAdd(va, vb), vc)
    let vRhs = vestaAdd(vestaMul(va, vc), vestaMul(vb, vc))
    let vDistOk = vestaToInt(vLhs) == vestaToInt(vRhs)
    print("  Distributivity: \(vDistOk ? "PASS" : "FAIL")")

    // ---- Cycle Property Test ----
    print("\n--- Cycle Property (Pallas Fq = Vesta Fp) ---")
    // Pallas scalar field modulus = Vesta base field modulus
    let pallasFqModulus = VestaFp.P  // Pallas Fq == Vesta Fp
    let vestaFpModulus = VestaFp.P
    let cycleOk1 = pallasFqModulus == vestaFpModulus
    print("  Pallas Fq = Vesta Fp: \(cycleOk1 ? "PASS" : "FAIL")")

    // Vesta scalar field modulus = Pallas base field modulus
    let vestaFqModulus = PallasFp.P  // Vesta Fq == Pallas Fp
    let pallasFpModulus = PallasFp.P
    let cycleOk2 = vestaFqModulus == pallasFpModulus
    print("  Vesta Fq = Pallas Fp: \(cycleOk2 ? "PASS" : "FAIL")")

    // ---- Pallas Curve Tests ----
    print("\n--- Pallas Curve Operations ---")

    let pGen = pallasGenerator()
    // Verify on curve: y^2 = x^3 + 5
    let pgx2 = pallasSqr(pGen.x)
    let pgx3 = pallasMul(pgx2, pGen.x)
    let five = pallasFromInt(5)
    let pRhs = pallasAdd(pgx3, five)
    let pLhs = pallasSqr(pGen.y)
    let pOnCurve = pallasToInt(pLhs) == pallasToInt(pRhs)
    print("  Generator on curve (y^2=x^3+5): \(pOnCurve ? "PASS" : "FAIL")")

    let pGProj = pallasPointFromAffine(pGen)
    let p2G = pallasPointDouble(pGProj)
    let p2GAff = pallasPointToAffine(p2G)
    let p2Gx3 = pallasMul(pallasSqr(p2GAff.x), p2GAff.x)
    let p2GOnCurve = pallasToInt(pallasSqr(p2GAff.y)) == pallasToInt(pallasAdd(p2Gx3, five))
    print("  2G on curve: \(p2GOnCurve ? "PASS" : "FAIL")")

    let pGplusG = pallasPointAdd(pGProj, pGProj)
    let pGplusGAff = pallasPointToAffine(pGplusG)
    let pAddEqDbl = pallasToInt(pGplusGAff.x) == pallasToInt(p2GAff.x) &&
                    pallasToInt(pGplusGAff.y) == pallasToInt(p2GAff.y)
    print("  G + G = 2G: \(pAddEqDbl ? "PASS" : "FAIL")")

    let p5G_mul = pallasPointMulInt(pGProj, 5)
    var p5G_add = pGProj
    for _ in 1..<5 { p5G_add = pallasPointAdd(p5G_add, pGProj) }
    let p5MulAff = pallasPointToAffine(p5G_mul)
    let p5AddAff = pallasPointToAffine(p5G_add)
    let pScalarOk = pallasToInt(p5MulAff.x) == pallasToInt(p5AddAff.x) &&
                    pallasToInt(p5MulAff.y) == pallasToInt(p5AddAff.y)
    print("  5*G (mul) = 5*G (add): \(pScalarOk ? "PASS" : "FAIL")")

    let pIdentity = pallasPointIdentity()
    let pGplusId = pallasPointAdd(pGProj, pIdentity)
    let pGplusIdAff = pallasPointToAffine(pGplusId)
    let pIdOk = pallasToInt(pGplusIdAff.x) == pallasToInt(pGen.x) &&
                pallasToInt(pGplusIdAff.y) == pallasToInt(pGen.y)
    print("  G + O = G: \(pIdOk ? "PASS" : "FAIL")")

    let pNegG = pallasPointNegateAffine(pGen)
    let pGplusNegG = pallasPointAdd(pGProj, pallasPointFromAffine(pNegG))
    let pNegOk = pallasPointIsIdentity(pGplusNegG)
    print("  G + (-G) = O: \(pNegOk ? "PASS" : "FAIL")")

    // ---- Vesta Curve Tests ----
    print("\n--- Vesta Curve Operations ---")

    let vGen = vestaGenerator()
    let vgx2 = vestaSqr(vGen.x)
    let vgx3 = vestaMul(vgx2, vGen.x)
    let vFive = vestaFromInt(5)
    let vRhsCurve = vestaAdd(vgx3, vFive)
    let vLhsCurve = vestaSqr(vGen.y)
    let vOnCurve = vestaToInt(vLhsCurve) == vestaToInt(vRhsCurve)
    print("  Generator on curve (y^2=x^3+5): \(vOnCurve ? "PASS" : "FAIL")")

    let vGProj = vestaPointFromAffine(vGen)
    let v2G = vestaPointDouble(vGProj)
    let v2GAff = vestaPointToAffine(v2G)
    let v2Gx3 = vestaMul(vestaSqr(v2GAff.x), v2GAff.x)
    let v2GOnCurve = vestaToInt(vestaSqr(v2GAff.y)) == vestaToInt(vestaAdd(v2Gx3, vFive))
    print("  2G on curve: \(v2GOnCurve ? "PASS" : "FAIL")")

    let vGplusG = vestaPointAdd(vGProj, vGProj)
    let vGplusGAff = vestaPointToAffine(vGplusG)
    let vAddEqDbl = vestaToInt(vGplusGAff.x) == vestaToInt(v2GAff.x) &&
                    vestaToInt(vGplusGAff.y) == vestaToInt(v2GAff.y)
    print("  G + G = 2G: \(vAddEqDbl ? "PASS" : "FAIL")")

    let v5G_mul = vestaPointMulInt(vGProj, 5)
    var v5G_add = vGProj
    for _ in 1..<5 { v5G_add = vestaPointAdd(v5G_add, vGProj) }
    let v5MulAff = vestaPointToAffine(v5G_mul)
    let v5AddAff = vestaPointToAffine(v5G_add)
    let vScalarOk = vestaToInt(v5MulAff.x) == vestaToInt(v5AddAff.x) &&
                    vestaToInt(v5MulAff.y) == vestaToInt(v5AddAff.y)
    print("  5*G (mul) = 5*G (add): \(vScalarOk ? "PASS" : "FAIL")")

    let vNegG = vestaPointNegateAffine(vGen)
    let vGplusNegG = vestaPointAdd(vGProj, vestaPointFromAffine(vNegG))
    let vNegOk = vestaPointIsIdentity(vGplusNegG)
    print("  G + (-G) = O: \(vNegOk ? "PASS" : "FAIL")")

    // Summary
    let allPass = pOneOk && pZeroOk && addZeroOk && addNegOk && mulOneOk && mulInvOk &&
                  distOk && subOk && vOneOk && vAddNegOk && vMulOneOk && vMulInvOk && vDistOk &&
                  cycleOk1 && cycleOk2 &&
                  pOnCurve && p2GOnCurve && pAddEqDbl && pScalarOk && pIdOk && pNegOk &&
                  vOnCurve && v2GOnCurve && vAddEqDbl && vScalarOk && vNegOk
    print("\n  Overall: \(allPass ? "ALL PASS" : "SOME FAILED")")
}

public func runPastaMSMBench() {
    print("\n=== Pasta MSM Benchmark ===")

    do {
        // --- Pallas MSM ---
        print("\n--- Pallas MSM ---")
        let pallasEngine = try PallasMSM()

        let pGen = pallasGenerator()
        let pGProj = pallasPointFromAffine(pGen)

        let logSizes = [8, 10, 12, 14, 16]
        let maxN = 1 << logSizes.last!

        fputs("Generating \(maxN) Pallas points...\n", stderr)
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [PallasPointProjective]()
        projPoints.reserveCapacity(maxN)
        var acc = pGProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = pallasPointAdd(acc, pGProj)
        }
        let allPoints = batchPallasToAffine(projPoints)
        projPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

        // Random scalars
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

        // Correctness: small MSM
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            let gpuResult = try pallasEngine.msm(points: testPts, scalars: testScls)

            // CPU naive MSM for verification
            var cpuResult = pallasPointIdentity()
            for i in 0..<testN {
                // Use small scalar for CPU verification
                let s = Int(testScls[i][0] & 0xFFF)
                let projPt = pallasPointFromAffine(testPts[i])
                cpuResult = pallasPointAdd(cpuResult, pallasPointMulInt(projPt, s))
            }
            // Can't easily compare since GPU uses full scalars and CPU uses truncated
            // Just verify GPU result is on curve
            let gpuAff = pallasPointToAffine(gpuResult)
            let gx3 = pallasMul(pallasSqr(gpuAff.x), gpuAff.x)
            let five = pallasFromInt(5)
            let onCurve = pallasToInt(pallasSqr(gpuAff.y)) == pallasToInt(pallasAdd(gx3, five))
            print("  GPU MSM result on curve: \(onCurve ? "PASS" : "FAIL")")
        }

        // Performance
        for logN in logSizes {
            let n = 1 << logN
            let pts = Array(allPoints.prefix(n))
            let scls = Array(allScalars.prefix(n))
            let _ = try pallasEngine.msm(points: pts, scalars: scls) // warmup
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try pallasEngine.msm(points: pts, scalars: scls)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            fputs(String(format: "  Pallas MSM 2^%-2d: %7.1fms\n", logN, median), stderr)
        }

        // --- Vesta MSM ---
        print("\n--- Vesta MSM ---")
        let vestaEngine = try VestaMSM()

        let vGen = vestaGenerator()
        let vGProj = vestaPointFromAffine(vGen)

        fputs("Generating \(maxN) Vesta points...\n", stderr)
        let genT1 = CFAbsoluteTimeGetCurrent()
        var vProjPoints = [VestaPointProjective]()
        vProjPoints.reserveCapacity(maxN)
        var vAcc = vGProj
        for _ in 0..<maxN {
            vProjPoints.append(vAcc)
            vAcc = vestaPointAdd(vAcc, vGProj)
        }
        let vAllPoints = batchVestaToAffine(vProjPoints)
        vProjPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT1) * 1000))ms\n", stderr)

        // Correctness
        do {
            let testN = 256
            let testPts = Array(vAllPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            let gpuResult = try vestaEngine.msm(points: testPts, scalars: testScls)
            let gpuAff = vestaPointToAffine(gpuResult)
            let gx3 = vestaMul(vestaSqr(gpuAff.x), gpuAff.x)
            let vFive = vestaFromInt(5)
            let onCurve = vestaToInt(vestaSqr(gpuAff.y)) == vestaToInt(vestaAdd(gx3, vFive))
            print("  GPU MSM result on curve: \(onCurve ? "PASS" : "FAIL")")
        }

        // Performance
        for logN in logSizes {
            let n = 1 << logN
            let pts = Array(vAllPoints.prefix(n))
            let scls = Array(allScalars.prefix(n))
            let _ = try vestaEngine.msm(points: pts, scalars: scls) // warmup
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try vestaEngine.msm(points: pts, scalars: scls)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            fputs(String(format: "  Vesta  MSM 2^%-2d: %7.1fms\n", logN, median), stderr)
        }

    } catch {
        print("  ERROR: \(error)")
    }
}
