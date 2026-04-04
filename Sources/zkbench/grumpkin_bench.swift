// Grumpkin curve benchmark and correctness tests
import zkMetal
import Foundation

public func runGrumpkinTest() {
    print("\n=== Grumpkin Curve Tests ===")

    // --- Generator on curve ---
    print("\n--- Grumpkin Curve Operations ---")
    let gen = grumpkinGenerator()
    let onCurve = grumpkinPointIsOnCurve(gen)
    print("  Generator on curve (y^2=x^3-17): \(onCurve ? "PASS" : "FAIL")")

    // --- Cycle property: Grumpkin base field = BN254 Fr ---
    print("\n--- Cycle Property ---")
    // Grumpkin Fp = BN254 Fr (same modulus)
    // Grumpkin Fq = BN254 Fq (same modulus)
    print("  Grumpkin base field = BN254 Fr: PASS (by construction)")
    print("  Grumpkin scalar field = BN254 Fq: PASS (by construction)")

    // --- Point doubling ---
    let gProj = grumpkinPointFromAffine(gen)
    let g2 = grumpkinPointDouble(gProj)
    let g2Aff = grumpkinPointToAffine(g2)
    let g2OnCurve = grumpkinPointIsOnCurve(g2Aff)
    print("  2G on curve: \(g2OnCurve ? "PASS" : "FAIL")")

    // --- G + G = 2G ---
    let gPlusG = grumpkinPointAdd(gProj, gProj)
    let gPlusGAff = grumpkinPointToAffine(gPlusG)
    let addEqDbl = frToInt(gPlusGAff.x) == frToInt(g2Aff.x) &&
                   frToInt(gPlusGAff.y) == frToInt(g2Aff.y)
    print("  G + G = 2G: \(addEqDbl ? "PASS" : "FAIL")")

    // --- 5*G scalar mul = repeated addition ---
    let g5_mul = grumpkinPointMulInt(gProj, 5)
    var g5_add = gProj
    for _ in 1..<5 { g5_add = grumpkinPointAdd(g5_add, gProj) }
    let g5MulAff = grumpkinPointToAffine(g5_mul)
    let g5AddAff = grumpkinPointToAffine(g5_add)
    let scalarOk = frToInt(g5MulAff.x) == frToInt(g5AddAff.x) &&
                   frToInt(g5MulAff.y) == frToInt(g5AddAff.y)
    print("  5*G (mul) = 5*G (add): \(scalarOk ? "PASS" : "FAIL")")

    // --- G + O = G ---
    let identity = grumpkinPointIdentity()
    let gPlusId = grumpkinPointAdd(gProj, identity)
    let gPlusIdAff = grumpkinPointToAffine(gPlusId)
    let idOk = frToInt(gPlusIdAff.x) == frToInt(gen.x) &&
               frToInt(gPlusIdAff.y) == frToInt(gen.y)
    print("  G + O = G: \(idOk ? "PASS" : "FAIL")")

    // --- G + (-G) = O ---
    let negG = grumpkinPointNegateAffine(gen)
    let gPlusNegG = grumpkinPointAdd(gProj, grumpkinPointFromAffine(negG))
    let negOk = grumpkinPointIsIdentity(gPlusNegG)
    print("  G + (-G) = O: \(negOk ? "PASS" : "FAIL")")

    // --- Associativity: (G + 2G) + 3G = G + (2G + 3G) ---
    let g3 = grumpkinPointAdd(g2, gProj)
    let lhs = grumpkinPointAdd(grumpkinPointAdd(gProj, g2), g3)
    let rhs = grumpkinPointAdd(gProj, grumpkinPointAdd(g2, g3))
    let lhsAff = grumpkinPointToAffine(lhs)
    let rhsAff = grumpkinPointToAffine(rhs)
    let assocOk = frToInt(lhsAff.x) == frToInt(rhsAff.x) &&
                  frToInt(lhsAff.y) == frToInt(rhsAff.y)
    print("  Associativity: \(assocOk ? "PASS" : "FAIL")")

    // --- Commutativity: G + 2G = 2G + G ---
    let commLhs = grumpkinPointAdd(gProj, g2)
    let commRhs = grumpkinPointAdd(g2, gProj)
    let commLhsAff = grumpkinPointToAffine(commLhs)
    let commRhsAff = grumpkinPointToAffine(commRhs)
    let commOk = frToInt(commLhsAff.x) == frToInt(commRhsAff.x) &&
                 frToInt(commLhsAff.y) == frToInt(commRhsAff.y)
    print("  Commutativity: \(commOk ? "PASS" : "FAIL")")

    // --- Batch affine conversion ---
    let points = [gProj, g2, g3]
    let batchAff = batchGrumpkinToAffine(points)
    let batchOk = batchAff.count == 3 &&
                  frToInt(batchAff[0].x) == frToInt(gen.x) &&
                  grumpkinPointIsOnCurve(batchAff[1]) &&
                  grumpkinPointIsOnCurve(batchAff[2])
    print("  Batch affine: \(batchOk ? "PASS" : "FAIL")")

    // --- Point operation benchmarks ---
    print("\n--- Point Operation Benchmarks ---")
    do {
        let iterations = 10000
        var acc = gProj

        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            acc = grumpkinPointDouble(acc)
        }
        let dblTime = (CFAbsoluteTimeGetCurrent() - t0) * 1e6 / Double(iterations)
        fputs(String(format: "  Point double: %.2f us/op\n", dblTime), stderr)

        let t1 = CFAbsoluteTimeGetCurrent()
        var acc2 = gProj
        for _ in 0..<iterations {
            acc2 = grumpkinPointAdd(acc2, gProj)
        }
        let addTime = (CFAbsoluteTimeGetCurrent() - t1) * 1e6 / Double(iterations)
        fputs(String(format: "  Point add:    %.2f us/op\n", addTime), stderr)

        let t2 = CFAbsoluteTimeGetCurrent()
        let scalarMulIters = 100
        for i in 0..<scalarMulIters {
            _ = grumpkinPointMulInt(gProj, 12345 + i)
        }
        let mulTime = (CFAbsoluteTimeGetCurrent() - t2) * 1e3 / Double(scalarMulIters)
        fputs(String(format: "  Scalar mul:   %.2f ms/op\n", mulTime), stderr)
    }

    let allPass = onCurve && g2OnCurve && addEqDbl && scalarOk && idOk && negOk &&
                  assocOk && commOk && batchOk
    print("\n  Overall: \(allPass ? "ALL PASS" : "SOME FAILED")")
}

public func runGrumpkinMSMBench() {
    print("\n=== Grumpkin MSM Benchmark ===")

    do {
        let engine = try GrumpkinMSM()
        let gen = grumpkinGenerator()
        let gProj = grumpkinPointFromAffine(gen)

        let logSizes = [8, 12, 14, 16]
        let maxN = 1 << logSizes.last!

        fputs("Generating \(maxN) Grumpkin points...\n", stderr)
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [GrumpkinPointProjective]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = grumpkinPointAdd(acc, gProj)
        }
        let allPoints = batchGrumpkinToAffine(projPoints)
        projPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

        // Random scalars (in BN254 Fq / Grumpkin scalar field)
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

        // Correctness: small MSM result is on curve
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            let gpuResult = try engine.msm(points: testPts, scalars: testScls)
            let gpuAff = grumpkinPointToAffine(gpuResult)
            let onCurve = grumpkinPointIsOnCurve(gpuAff)
            print("  GPU MSM result on curve: \(onCurve ? "PASS" : "FAIL")")
        }

        // Correctness: MSM matches naive scalar mul sum (tiny size)
        do {
            let testN = 16
            let testPts = Array(allPoints.prefix(testN))
            // Use small scalars for CPU verification
            var smallScalars = [[UInt32]]()
            for i in 0..<testN {
                var limbs = [UInt32](repeating: 0, count: 8)
                limbs[0] = UInt32(i + 1)
                smallScalars.append(limbs)
            }
            let gpuResult = try engine.msm(points: testPts, scalars: smallScalars)

            // CPU naive: sum(scalar_i * P_i)
            var cpuResult = grumpkinPointIdentity()
            for i in 0..<testN {
                cpuResult = grumpkinPointAdd(cpuResult, grumpkinPointMulInt(grumpkinPointFromAffine(testPts[i]), Int(smallScalars[i][0])))
            }

            let gpuAff = grumpkinPointToAffine(gpuResult)
            let cpuAff = grumpkinPointToAffine(cpuResult)
            let matchOk = frToInt(gpuAff.x) == frToInt(cpuAff.x) &&
                          frToInt(gpuAff.y) == frToInt(cpuAff.y)
            print("  MSM matches naive sum: \(matchOk ? "PASS" : "FAIL")")
        }

        // Performance benchmarks
        for logN in logSizes {
            let n = 1 << logN
            let pts = Array(allPoints.prefix(n))
            let scls = Array(allScalars.prefix(n))
            let _ = try engine.msm(points: pts, scalars: scls) // warmup
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: pts, scalars: scls)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            fputs(String(format: "  Grumpkin MSM 2^%-2d: %7.1fms\n", logN, median), stderr)
        }
    } catch {
        print("  ERROR: \(error)")
    }
}
