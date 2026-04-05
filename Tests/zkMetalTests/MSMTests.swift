import zkMetal

func runMSMTests() {
    let gx = fpFromInt(1), gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

    suite("MSM BN254 GPU")
    do {
        let msm = try MetalMSM()
        let g = PointAffine(x: gx, y: gy)
        let r = try msm.msm(points: [g], scalars: [[5, 0, 0, 0, 0, 0, 0, 0]])
        expect(pointEqual(r, pointMulInt(pointFromAffine(g), 5)), "5*G")

        let n = 64
        var pts = [PointProjective](); var acc = gProj
        for _ in 0..<n { pts.append(acc); acc = pointAdd(acc, gProj) }
        let affPts = batchToAffine(pts)
        var scalars = [[UInt32]](); var rng: UInt64 = 0xDEAD_CAFE
        for _ in 0..<n {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
            scalars.append(limbs)
        }
        let gpuR = try msm.msm(points: affPts, scalars: scalars)
        var seqR = pointIdentity()
        for i in 0..<n { seqR = pointAdd(seqR, pointScalarMul(pts[i], frFromLimbs(scalars[i]))) }
        expect(pointEqual(gpuR, seqR), "GPU vs sequential 64pts")

        let aff = batchToAffine([gpuR])[0]
        let y2 = fpSqr(aff.y); let x3 = fpMul(fpSqr(aff.x), aff.x); let rhs = fpAdd(x3, fpFromInt(3))
        expect(fpToInt(y2) == fpToInt(rhs), "Result on curve")
    } catch { expect(false, "MSM error: \(error)") }

    suite("C Pippenger MSM")
    let n = 256
    var projPts = [PointProjective](); var acc = gProj
    for _ in 0..<n { projPts.append(acc); acc = pointAdd(acc, gProj) }
    let pts = batchToAffine(projPts)
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var scalars = [[UInt32]]()
    for _ in 0..<n {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
        scalars.append(limbs)
    }
    let cR = cPippengerMSM(points: pts, scalars: scalars)
    let swR = parallelMSM(points: pts, scalars: scalars)
    expect(pointEqual(cR, swR), "C Pippenger = Swift Pippenger 256pts")

    // Edge cases
    let zero = cPippengerMSM(points: [pts[0]], scalars: [[0, 0, 0, 0, 0, 0, 0, 0]])
    expect(pointIsIdentity(zero), "scalar=0 → identity")
    let one = cPippengerMSM(points: [pts[0]], scalars: [[1, 0, 0, 0, 0, 0, 0, 0]])
    expect(pointEqual(one, pointFromAffine(pts[0])), "scalar=1 → point")

    // Small-n cross check
    for testN in [1, 2, 4, 8, 16, 32, 64] {
        let sp = Array(pts.prefix(testN))
        let ss = Array(scalars.prefix(testN))
        let cSmall = cPippengerMSM(points: sp, scalars: ss)
        var naive = pointIdentity()
        for i in 0..<testN { naive = pointAdd(naive, cPointScalarMul(pointFromAffine(sp[i]), frFromLimbs(ss[i]))) }
        expect(pointEqual(cSmall, naive), "C Pippenger n=\(testN)")
    }
}
