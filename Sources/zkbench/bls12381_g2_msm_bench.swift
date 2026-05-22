// BLS12-381 G2 MSM Benchmark
// GPU-accelerated multi-scalar multiplication on BLS12-381 G2 curve.

import Foundation
import zkMetal

public func runBLS12381G2MSMBench() {
    print("\n=== BLS12-381 G2 MSM Benchmark ===")

    do {
        let engine = try BLS12381G2MSM()
        let gen = bls12381G2Generator()
        let gProj = g2_381FromAffine(gen)

        let logSizes = [8, 10, 12, 14, 16, 18]
        let sizes = logSizes.map { 1 << $0 }
        let maxN = sizes.last!

        fputs("Generating \(maxN) G2 points...\n", stderr)
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [G2Projective381]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = g2_381Add(acc, gProj)
        }
        var allPoints = [G2Affine381]()
        allPoints.reserveCapacity(maxN)
        for p in projPoints {
            if let aff = g2_381ToAffine(p) {
                allPoints.append(aff)
            }
        }
        projPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

        // Random scalars (Fr381 = 255-bit, stored as 8x32-bit limbs)
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

        // Correctness: small MSM (16 pts) against CPU reference
        do {
            let testN = 16
            let testPts = Array(allPoints.prefix(testN))
            var smallScalars = [[UInt32]]()
            for i in 0..<testN {
                var limbs = [UInt32](repeating: 0, count: 8)
                limbs[0] = UInt32(i + 1)
                smallScalars.append(limbs)
            }

            let gpuResult = try engine.msm(points: testPts, scalars: smallScalars)

            var cpuResult = g2_381Identity()
            for i in 0..<testN {
                let reduced = BLS12381G2MSM.reduceModR(smallScalars[i])
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                let term = g2_381ScalarMul(g2_381FromAffine(testPts[i]), u64)
                cpuResult = g2_381IsIdentity(cpuResult) ? term : g2_381Add(cpuResult, term)
            }

            let gpuAff = g2_381ToAffine(gpuResult)
            let cpuAff = g2_381ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2PointsEqual(ga.x, ca.x) && fp2PointsEqual(ga.y, ca.y)
            }
            print("  MSM 16pt (small scalars): \(matchOk ? "PASS" : "FAIL")")
        }

        // Correctness: GPU MSM with random scalars vs CPU reference
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))

            let gpuResult = try engine.msm(points: testPts, scalars: testScls)

            var cpuResult = g2_381Identity()
            for i in 0..<testN {
                let reduced = BLS12381G2MSM.reduceModR(testScls[i])
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                let term = g2_381ScalarMul(g2_381FromAffine(testPts[i]), u64)
                cpuResult = g2_381IsIdentity(cpuResult) ? term : g2_381Add(cpuResult, term)
            }

            let gpuAff = g2_381ToAffine(gpuResult)
            let cpuAff = g2_381ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2PointsEqual(ga.x, ca.x) && fp2PointsEqual(ga.y, ca.y)
            }
            print("  MSM 256pt (random scalars): \(matchOk ? "PASS" : "FAIL")")
        }

        // Correctness: small window size stress test
        do {
            let testN = 128
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))

            engine.windowBitsOverride = 4
            let gpuResult = try engine.msm(points: testPts, scalars: testScls)
            engine.windowBitsOverride = nil

            var cpuResult = g2_381Identity()
            for i in 0..<testN {
                let reduced = BLS12381G2MSM.reduceModR(testScls[i])
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                let term = g2_381ScalarMul(g2_381FromAffine(testPts[i]), u64)
                cpuResult = g2_381IsIdentity(cpuResult) ? term : g2_381Add(cpuResult, term)
            }

            let gpuAff = g2_381ToAffine(gpuResult)
            let cpuAff = g2_381ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2PointsEqual(ga.x, ca.x) && fp2PointsEqual(ga.y, ca.y)
            }
            print("  MSM 128pt wb=4 (stress test): \(matchOk ? "PASS" : "FAIL")")
        }

        // Performance benchmarks
        print("\n--- BLS12-381 G2 MSM Performance ---")
        for (idx, n) in sizes.enumerated() {
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
            let pps = Double(n) / (median / 1000.0)
            fputs(String(format: "  G2 MSM 2^%-2d = %7d pts: %7.1f ms  (%.0f pts/s)\n",
                         logSizes[idx], n, median, pps), stderr)
        }

        // Compare G1 vs G2 MSM at 256 points
        do {
            let testN = 256
            let g1Pts = try generateG1Points(testN)
            let g2Pts = Array(allPoints.prefix(testN))
            let g1Scls = generateG1Scalars(testN)
            let g2Scls = Array(allScalars.prefix(testN))

            let g1Engine = try BLS12381MSM()
            let g2Engine = engine

            // Warmup
            let _ = try g1Engine.msm(points: g1Pts, scalars: g1Scls)
            let _ = try g2Engine.msm(points: g2Pts, scalars: g2Scls)

            let runs = 5
            var g1Times = [Double]()
            var g2Times = [Double]()

            for _ in 0..<runs {
                let g1Start = CFAbsoluteTimeGetCurrent()
                let _ = try g1Engine.msm(points: g1Pts, scalars: g1Scls)
                g1Times.append((CFAbsoluteTimeGetCurrent() - g1Start) * 1000)

                let g2Start = CFAbsoluteTimeGetCurrent()
                let _ = try g2Engine.msm(points: g2Pts, scalars: g2Scls)
                g2Times.append((CFAbsoluteTimeGetCurrent() - g2Start) * 1000)
            }

            g1Times.sort()
            g2Times.sort()
            let g1Med = g1Times[runs / 2]
            let g2Med = g2Times[runs / 2]

            fputs(String(format: "  G1 vs G2 @ 256pts: G1=%.1fms G2=%.1fms ratio=%.1fx\n",
                         g1Med, g2Med, g2Med / g1Med), stderr)
        }

    } catch {
        print("  ERROR: \(error)")
    }
}

// MARK: - G1 helpers for comparison

private func fp2PointsEqual(_ a: Fp2_381, _ b: Fp2_381) -> Bool {
    // Compare all 24 limb slots across c0 and c1
    a.c0.v.0 == b.c0.v.0 && a.c0.v.1 == b.c0.v.1 &&
    a.c0.v.2 == b.c0.v.2 && a.c0.v.3 == b.c0.v.3 &&
    a.c0.v.4 == b.c0.v.4 && a.c0.v.5 == b.c0.v.5 &&
    a.c0.v.6 == b.c0.v.6 && a.c0.v.7 == b.c0.v.7 &&
    a.c0.v.8 == b.c0.v.8 && a.c0.v.9 == b.c0.v.9 &&
    a.c0.v.10 == b.c0.v.10 && a.c0.v.11 == b.c0.v.11 &&
    a.c1.v.0 == b.c1.v.0 && a.c1.v.1 == b.c1.v.1 &&
    a.c1.v.2 == b.c1.v.2 && a.c1.v.3 == b.c1.v.3 &&
    a.c1.v.4 == b.c1.v.4 && a.c1.v.5 == b.c1.v.5 &&
    a.c1.v.6 == b.c1.v.6 && a.c1.v.7 == b.c1.v.7 &&
    a.c1.v.8 == b.c1.v.8 && a.c1.v.9 == b.c1.v.9 &&
    a.c1.v.10 == b.c1.v.10 && a.c1.v.11 == b.c1.v.11
}

private func generateG1Points(_ count: Int) throws -> [G1Affine381] {
    let gen = bls12381G1Generator()
    let gProj = g1_381FromAffine(gen)
    var points = [G1Affine381]()
    points.reserveCapacity(count)
    var acc = gProj
    for _ in 0..<count {
        if let aff = g1_381ToAffine(acc) {
            points.append(aff)
        }
        acc = g1_381Add(acc, gProj)
    }
    return points
}

private func generateG1Scalars(_ count: Int) -> [[UInt32]] {
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var scalars = [[UInt32]]()
    scalars.reserveCapacity(count)
    for _ in 0..<count {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        scalars.append(limbs)
    }
    return scalars
}