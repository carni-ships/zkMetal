import zkMetal
import Foundation

// Test Pippenger MSM with known scalars - runs at START of benchmark
public func runPippengerTest() {
    print("=== Pippenger MSM Correctness Test ===")

    // Create generator point (1, 2) in Mont (which means fp(1), fp(2))
    let gx = fpFromInt(1)
    let gy = fpFromInt(2)

    // Convert to flat format for C
    let flatPts = batchToAffine([PointProjective(x: gx, y: gy, z: Fp.one)])

    let testScalars: [[UInt32]] = [
        [1, 0, 0, 0, 0, 0, 0, 0],   // scalar = 1
        [2, 0, 0, 0, 0, 0, 0, 0],   // scalar = 2
        [3, 0, 0, 0, 0, 0, 0, 0],   // scalar = 3
        [10, 0, 0, 0, 0, 0, 0, 0],  // scalar = 10
    ]

    for scalars in testScalars {
        // C Pippenger
        let cResult = cPippengerMSM(points: flatPts, scalars: [scalars])

        // Expected via pointMulInt (sequential scalar multiplication)
        let expected = pointMulInt(PointProjective(x: gx, y: gy, z: Fp.one), Int(scalars[0]))

        let match = pointEqual(cResult, expected)
        print("scalar=\(scalars[0]): \(match ? "PASS" : "FAIL")")

        if !match {
            let cAff = batchToAffine([cResult])[0]
            let expAff = batchToAffine([expected])[0]
            print("  C result: x=\(fpToInt(cAff.x)), y=\(fpToInt(cAff.y))")
            print("  Expected: x=\(fpToInt(expAff.x)), y=\(fpToInt(expAff.y))")
        }
    }
    print("=== Pippenger Test Complete ===\n")
}