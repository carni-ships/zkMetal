#!/usr/bin/env swift
import Foundation
import zkMetal

func testLargeMSM() {
    do {
        let engine = try MetalMSM()
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        let logSizes = [8, 10, 12, 14, 16, 18, 20]
        let maxN = 1 << logSizes.last!

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

        for logN in logSizes {
            let n = 1 << logN
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            print("Testing 2^\(logN) (n=\(n))...")

            // C Pippenger
            let cR = cPippengerMSM(points: points, scalars: scalars)

            // GPU MSM
            let gpuR = try engine.msm(points: points, scalars: scalars)

            let match = pointEqual(cR, gpuR)
            print("  2^\(logN): \(match ? "OK" : "MISMATCH")")

            if !match {
                print("    C result: \(cR)")
                print("    GPU result: \(gpuR)")

                let cAff = batchToAffine([cR])[0]
                let gpuAff = batchToAffine([gpuR])[0]

                print("    C x: \(fpToInt(cAff.x))")
                print("    GPU x: \(fpToInt(gpuAff.x))")
                print("    C y: \(fpToInt(cAff.y))")
                print("    GPU y: \(fpToInt(gpuAff.y))")
            }
        }
    } catch {
        print("Error: \(error)")
    }
}

testLargeMSM()
