import zkMetal
import Foundation

public func runMSMDebugTest() {
    print("=== MSM Large Scale Debug Test ===")

    do {
        let engine = try MetalMSM()
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        // Test specific sizes that show MISMATCH
        let testSizes = [256, 1024, 16384, 262144, 1048576]  // 2^8, 2^10, 2^14, 2^18, 2^20
        let maxN = testSizes.max()!

        print("Generating \(maxN) points...")
        var projPts = [PointProjective]()
        projPts.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPts.append(acc)
            acc = pointAdd(acc, gProj)
        }
        let allPoints = batchToAffine(projPts)
        projPts = []

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

        for n in testSizes {
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            print("\nTesting n=\(n) (2^\(Int(log2(Double(n)))))...")

            // Get window bits for this size
            var windowBits: UInt32
            if n <= 256 {
                windowBits = 8
            } else if n <= 4096 {
                windowBits = 10
            } else if n <= 32768 {
                windowBits = 12
            } else {
                windowBits = 16  // Assuming M3 Pro
            }
            let nWindows = (256 + Int(windowBits) - 1) / Int(windowBits)
            print("  windowBits=\(windowBits), nWindows=\(nWindows)")

            // C Pippenger
            let t0 = CFAbsoluteTimeGetCurrent()
            let cR = cPippengerMSM(points: points, scalars: scalars)
            let cTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // GPU MSM
            let t1 = CFAbsoluteTimeGetCurrent()
            let gpuR = try engine.msm(points: points, scalars: scalars)
            let gpuTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            let match = pointEqual(cR, gpuR)
            print("  C Pippenger: \(String(format: "%.1f", cTime))ms")
            print("  GPU: \(String(format: "%.1f", gpuTime))ms")
            print("  Result: \(match ? "OK" : "MISMATCH")")

            if !match {
                let cAff = batchToAffine([cR])[0]
                let gpuAff = batchToAffine([gpuR])[0]
                print("  C x: \(fpToInt(cAff.x))")
                print("  GPU x: \(fpToInt(gpuAff.x))")
                print("  C y: \(fpToInt(cAff.y))")
                print("  GPU y: \(fpToInt(gpuAff.y))")
            }
        }
    } catch {
        print("Error: \(error)")
    }
}
