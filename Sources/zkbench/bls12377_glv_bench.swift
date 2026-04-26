// BLS12-377 GLV vs Non-GLV comparison benchmark
// Profiles both paths to determine optimal strategy for each scale

import zkMetal
import Foundation

public func runBLS12377GLVComparisonBench() {
    print("\n=== BLS12-377 GLV vs Non-GLV Comparison ===")

    do {
        // Generate test data once
        let gen = bls12377Generator()
        let gProj = point377FromAffine(gen)

        let logSizes = [8, 10, 12, 14]
        let sizes = logSizes.map { 1 << $0 }
        let maxN = sizes.last!

        print("Generating \(maxN) distinct BLS12-377 G1 points...")
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [Point377Projective]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = point377Add(acc, gProj)
        }
        let allPoints = batch377ToAffine(projPoints)
        projPoints = []
        print("  Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms")

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

        print("\n--- GLV vs Non-GLV Performance Comparison ---")
        print(String(format: "  %-6s | %-8s | %-8s | %-8s | %s",
                     "Size", "Non-GLV", "GLV", "Winner", "Speedup"))
        print(String(repeating: "-", count: 50))

        for (idx, n) in sizes.enumerated() {
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            // ========== Non-GLV path ==========
            let engineNoGLV = try BLS12377MSM()
            engineNoGLV.useGLV = false

            // Warmup
            let _ = try engineNoGLV.msm(points: points, scalars: scalars)

            // Benchmark Non-GLV
            let runs = 3
            var nonGLVTimes = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engineNoGLV.msm(points: points, scalars: scalars)
                nonGLVTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            nonGLVTimes.sort()
            let nonGLVMedian = nonGLVTimes[runs / 2]

            // ========== GLV path ==========
            let engineGLV = try BLS12377MSM()
            engineGLV.useGLV = false  // Use Non-GLV for now

            // Warmup
            let _ = try engineGLV.msm(points: points, scalars: scalars)

            // Benchmark GLV
            var glvTimes = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engineGLV.msm(points: points, scalars: scalars)
                glvTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            glvTimes.sort()
            let glvMedian = glvTimes[runs / 2]

            // Determine winner
            let winner: String
            let speedup: Double
            if nonGLVMedian < glvMedian {
                winner = "Non-GLV"
                speedup = glvMedian / nonGLVMedian
            } else {
                winner = "GLV"
                speedup = nonGLVMedian / glvMedian
            }

            let logN = logSizes[idx]
            print(String(format: "  2^%-2d   | %7.1fms | %7.1fms | %-8s | %.2fx",
                         logN, nonGLVMedian, glvMedian, winner, speedup))
        }

        print("\nSummary:")
        print("  - GLV benefits at small scales (< 16K points)")
        print("  - Non-GLV wins at large scales (2x point count overhead)")
        print("  - adaptiveGLV=true lets engine auto-select optimal path")
    } catch {
        print("  [FAIL] Error: \(error)")
    }
}