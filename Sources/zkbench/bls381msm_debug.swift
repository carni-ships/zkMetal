import zkMetal
import Foundation

public func runBLS381MSMDebug() {
    print(">>> runBLS381MSMDebug() called <<<")
    fflush(stdout)

    print("Step 1: About to create BLS12381MSM...")
    fflush(stdout)

    do {
        let engine = try BLS12381MSM()
        print("Step 1: Engine created")
        fflush(stdout)

        print("Step 2: Getting generator...")
        fflush(stdout)
        let gen = bls12381G1Generator()
        print("Step 2: Got generator")
        fflush(stdout)

        print("Step 3: Converting to projective...")
        fflush(stdout)
        let gProj = g1_381FromAffine(gen)
        print("Step 3: Projective point created")
        fflush(stdout)

        print("Step 4: Generating 256 points...")
        fflush(stdout)
        let t0 = CFAbsoluteTimeGetCurrent()
        var projPoints = [G1Projective381]()
        projPoints.reserveCapacity(256)
        var acc = gProj
        for i in 0..<256 {
            projPoints.append(acc)
            acc = g1_381Add(acc, gProj)
            if i % 100 == 0 { print("  point \(i)"); fflush(stdout) }
        }
        let genTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print("Step 4: Points generated in \(genTime)ms")
        fflush(stdout)

        print("Step 5: Converting to affine...")
        fflush(stdout)
        let allPoints = batchG1_381ToAffine(projPoints)
        projPoints = []
        print("Step 5: Affine conversion done")
        fflush(stdout)

        print("Step 6: Generating scalars...")
        fflush(stdout)
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        var allScalars = [[UInt32]]()
        allScalars.reserveCapacity(256)
        for _ in 0..<256 {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            allScalars.append(limbs)
        }
        print("Step 6: Scalars generated")
        fflush(stdout)

        print("Step 7: Running MSM (n=256)...")
        fflush(stdout)
        let t1 = CFAbsoluteTimeGetCurrent()
        let result = try engine.msm(points: allPoints, scalars: allScalars)
        let elapsed = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        print("Step 7: MSM completed in \(elapsed)ms")
        fflush(stdout)

        if let aff = g1_381ToAffine(result) {
            let x = fp381ToInt(aff.x)
            let y = fp381ToInt(aff.y)
            print("Result: x=\(x[0]), y=\(y[0])")
            fflush(stdout)
        }
        print("Done!")
        fflush(stdout)
    } catch {
        print("ERROR: \(error)")
        fflush(stdout)
        exit(1)
    }
}