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

    let zero = cPippengerMSM(points: [pts[0]], scalars: [[0, 0, 0, 0, 0, 0, 0, 0]])
    expect(pointIsIdentity(zero), "scalar=0 -> identity")
    let one = cPippengerMSM(points: [pts[0]], scalars: [[1, 0, 0, 0, 0, 0, 0, 0]])
    expect(pointEqual(one, pointFromAffine(pts[0])), "scalar=1 -> point")

    for testN in [1, 2, 4, 8, 16, 32, 64] {
        let sp = Array(pts.prefix(testN))
        let ss = Array(scalars.prefix(testN))
        let cSmall = cPippengerMSM(points: sp, scalars: ss)
        var naive = pointIdentity()
        for i in 0..<testN { naive = pointAdd(naive, cPointScalarMul(pointFromAffine(sp[i]), frFromLimbs(ss[i]))) }
        expect(pointEqual(cSmall, naive), "C Pippenger n=\(testN)")
    }

    // Large GPU MSM: determinism and on-curve check at n=32768
    suite("Large GPU MSM")
    do {
        let lgN = 32768
        var lgProjPts = [PointProjective]()
        lgProjPts.reserveCapacity(lgN)
        var lgAcc = gProj
        for _ in 0..<lgN {
            lgProjPts.append(lgAcc)
            lgAcc = pointAdd(lgAcc, gProj)
        }
        let lgAffPts = batchToAffine(lgProjPts)
        var lgRng: UInt64 = 0xC00_E1A_1E_0000
        var lgScalars = [[UInt32]]()
        lgScalars.reserveCapacity(lgN)
        for _ in 0..<lgN {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                lgRng = lgRng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: lgRng >> 32)
            }
            lgScalars.append(limbs)
        }

        let engine1 = try MetalMSM()
        let r1 = try engine1.msm(points: lgAffPts, scalars: lgScalars)

        // Determinism: second run same engine
        let r2 = try engine1.msm(points: lgAffPts, scalars: lgScalars)
        expect(pointEqual(r1, r2), "GPU MSM deterministic 32768pts")

        // Cross-engine determinism
        let engine2 = try MetalMSM()
        let r3 = try engine2.msm(points: lgAffPts, scalars: lgScalars)
        expect(pointEqual(r1, r3), "GPU MSM cross-engine 32768pts")

        // On-curve
        let rAff = batchToAffine([r1])[0]
        let ry2 = fpSqr(rAff.y)
        let rx3 = fpMul(fpSqr(rAff.x), rAff.x)
        let rrhs = fpAdd(rx3, fpFromInt(3))
        expect(fpToInt(ry2) == fpToInt(rrhs), "GPU MSM on curve 32768pts")
    } catch {
        expect(false, "Large GPU MSM error: \(error)")
    }
}
