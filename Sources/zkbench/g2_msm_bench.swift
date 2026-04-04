// BN254 G2 MSM benchmark and correctness tests

import Foundation
import zkMetal

public func runG2MSMBench() {
    print("\n=== BN254 G2 MSM Benchmark ===")

    do {
        let engine = try BN254G2MSM()
        let gen = bn254G2Generator()
        let gProj = g2FromAffine(gen)

        let logSizes = [8, 10, 12, 14]
        let maxN = 1 << logSizes.last!

        fputs("Generating \(maxN) G2 points...\n", stderr)
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [G2ProjectivePoint]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = g2Add(acc, gProj)
        }
        let allPoints = batchG2ToAffine(projPoints)
        projPoints = []
        fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

        // Random scalars (BN254 Fr)
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

        // Correctness: MSM matches naive scalar mul sum (small scalars, CPU path)
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

            var cpuResult = g2Identity()
            for i in 0..<testN {
                let scalU64: [UInt64] = [UInt64(smallScalars[i][0]), 0, 0, 0]
                let term = g2ScalarMul(g2FromAffine(testPts[i]), scalU64)
                cpuResult = g2IsIdentity(cpuResult) ? term : g2Add(cpuResult, term)
            }

            let gpuAff = g2ToAffine(gpuResult)
            let cpuAff = g2ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2Equal(ga.x, ca.x) && fp2Equal(ga.y, ca.y)
            }
            print("  MSM matches naive sum (16pt): \(matchOk ? "PASS" : "FAIL")")
        }

        // Correctness: GPU path with random scalars
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            let gpuResult = try engine.msm(points: testPts, scalars: testScls)

            var cpuResult = g2Identity()
            for i in 0..<testN {
                let reduced = BN254G2MSM.reduceModR(testScls[i])
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                let term = g2ScalarMul(g2FromAffine(testPts[i]), u64)
                cpuResult = g2IsIdentity(cpuResult) ? term : g2Add(cpuResult, term)
            }

            let gpuAff = g2ToAffine(gpuResult)
            let cpuAff = g2ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2Equal(ga.x, ca.x) && fp2Equal(ga.y, ca.y)
            }
            print("  MSM matches CPU (256pt random): \(matchOk ? "PASS" : "FAIL")")
        }

        // Correctness: small window size (wb=4) stress-tests CSM scratch sizing
        do {
            let testN = 128
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))
            engine.windowBitsOverride = 4
            let gpuResult = try engine.msm(points: testPts, scalars: testScls)
            engine.windowBitsOverride = nil

            var cpuResult = g2Identity()
            for i in 0..<testN {
                let reduced = BN254G2MSM.reduceModR(testScls[i])
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                let term = g2ScalarMul(g2FromAffine(testPts[i]), u64)
                cpuResult = g2IsIdentity(cpuResult) ? term : g2Add(cpuResult, term)
            }

            let gpuAff = g2ToAffine(gpuResult)
            let cpuAff = g2ToAffine(cpuResult)
            var matchOk = false
            if let ga = gpuAff, let ca = cpuAff {
                matchOk = fp2Equal(ga.x, ca.x) && fp2Equal(ga.y, ca.y)
            }
            print("  MSM 128pt random wb=4: \(matchOk ? "PASS" : "FAIL")")
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
            fputs(String(format: "  G2 MSM 2^%-2d: %7.1fms\n", logN, median), stderr)
        }

        // Compare with CPU
        do {
            let testN = 256
            let testPts = Array(allPoints.prefix(testN))
            let testScls = Array(allScalars.prefix(testN))

            let cpuStart = CFAbsoluteTimeGetCurrent()
            var scalsU64 = [[UInt64]]()
            for sc in testScls {
                let reduced = BN254G2MSM.reduceModR(sc)
                var u64 = [UInt64](repeating: 0, count: 4)
                for j in 0..<4 {
                    u64[j] = UInt64(reduced[j * 2]) | (UInt64(reduced[j * 2 + 1]) << 32)
                }
                scalsU64.append(u64)
            }
            let _ = g2PippengerMSM(
                points: testPts.map { g2FromAffine($0) },
                scalars: scalsU64)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000

            let gpuStart = CFAbsoluteTimeGetCurrent()
            let _ = try engine.msm(points: testPts, scalars: testScls)
            let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000

            fputs(String(format: "  CPU vs GPU @ 256pts: CPU=%.1fms GPU=%.1fms speedup=%.1fx\n",
                         cpuTime, gpuTime, cpuTime / gpuTime), stderr)
        }

    } catch {
        print("  ERROR: \(error)")
    }
}

/// Batch convert G2 projective points to affine using Montgomery's trick
public func batchG2ToAffine(_ points: [G2ProjectivePoint]) -> [G2AffinePoint] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fp2](repeating: .one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = g2IsIdentity(points[i]) ? prods[i-1] : fp2Mul(prods[i-1], points[i].z)
    }

    var inv = fp2Inverse(prods[n - 1])

    var result = [G2AffinePoint](repeating: G2AffinePoint(x: .one, y: .one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if g2IsIdentity(points[i]) { continue }
        let zinv = (i > 0) ? fp2Mul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fp2Mul(inv, points[i].z) }
        let zinv2 = fp2Sqr(zinv)
        let zinv3 = fp2Mul(zinv2, zinv)
        result[i] = G2AffinePoint(x: fp2Mul(points[i].x, zinv2), y: fp2Mul(points[i].y, zinv3))
    }
    return result
}

private func fp2Equal(_ a: Fp2, _ b: Fp2) -> Bool {
    a.c0.v.0 == b.c0.v.0 && a.c0.v.1 == b.c0.v.1 && a.c0.v.2 == b.c0.v.2 && a.c0.v.3 == b.c0.v.3 &&
    a.c0.v.4 == b.c0.v.4 && a.c0.v.5 == b.c0.v.5 && a.c0.v.6 == b.c0.v.6 && a.c0.v.7 == b.c0.v.7 &&
    a.c1.v.0 == b.c1.v.0 && a.c1.v.1 == b.c1.v.1 && a.c1.v.2 == b.c1.v.2 && a.c1.v.3 == b.c1.v.3 &&
    a.c1.v.4 == b.c1.v.4 && a.c1.v.5 == b.c1.v.5 && a.c1.v.6 == b.c1.v.6 && a.c1.v.7 == b.c1.v.7
}
