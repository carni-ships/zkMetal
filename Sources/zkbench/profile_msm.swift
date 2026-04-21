// MSM profiling tool - profiles MetalMSM at various sizes
import Foundation
import zkMetal

public func runMSMProfile() {
    do {
        let engine = try MetalMSM()
        engine.profileMSM = true  // Enable detailed profiling
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        let logSizes = [8, 10, 12, 14, 16, 18, 20]
        let maxN = 1 << logSizes.last!

        print("Generating \(maxN) points...")
        let genT0 = CFAbsoluteTimeGetCurrent()
        var projPts = [PointProjective]()
        projPts.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPts.append(acc)
            acc = pointAdd(acc, gProj)
        }
        let allPoints = batchToAffine(projPts)
        projPts = []
        print("Point generation: \((CFAbsoluteTimeGetCurrent() - genT0)*1000)ms")

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

        print("")
        print("MetalMSM Performance Profile")
        print(String(repeating: "=", count: 50))
        print("Size  |  Points  |  Time (ms)  |  Points/sec")
        print(String(repeating: "-", count: 50))

        for logN in logSizes {
            let n = 1 << logN
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            // Warmup
            let _ = try engine.msm(points: points, scalars: scalars)

            // GPU timing
            let iterations = logN >= 18 ? 1 : 3
            let gpuT0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                let _ = try engine.msm(points: points, scalars: scalars)
            }
            let gpuElapsed = (CFAbsoluteTimeGetCurrent() - gpuT0) / Double(iterations)

            let ptsPerSec = Double(n) / gpuElapsed
            let nStr = n >= 1000 ? String(format: "%6dK", n/1000) : "\(n)"
            print("2^\(logN)   |  \(nStr)  |  \(String(format: "%8.1f", gpuElapsed*1000))  |  \(String(format: "%.0f", ptsPerSec))")
        }

        print("")
        print("Note: CPU comparison not included - uses naive extrapolation")
        print("      Real CPU Pippenger is ~10x faster than naive double-and-add")

    } catch {
        print("Error: \(error)")
    }
}
