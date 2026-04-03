// BLS12-377 NTT Benchmark
import zkMetal
import Foundation

public func runBLS12377NTTBench() {
    print("\n=== BLS12-377 NTT Benchmark ===")
    print("Fr377 stride: \(MemoryLayout<Fr377>.stride) bytes")

    do {
        let engine = try BLS12377NTTEngine()

        // Correctness: round-trip at 2^10
        let testN = 1024
        var testInput = [Fr377](repeating: Fr377.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = fr377FromInt(UInt64(i + 1))
        }
        let gpuNTT = try engine.ntt(testInput)
        let gpuRecovered = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            let expected = fr377ToInt(testInput[i])
            let got = fr377ToInt(gpuRecovered[i])
            if expected != got {
                print("  MISMATCH at \(i): expected \(expected), got \(got)")
                correct = false
                break
            }
        }
        print("  Round-trip (2^10): \(correct ? "PASS" : "FAIL")")

        // Round-trip at 2^20 (four-step path)
        let testN2 = 1 << 20
        var testInput2 = [Fr377](repeating: Fr377.zero, count: testN2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput2[i] = fr377FromInt(rng >> 32)
        }
        let gpuNTT2 = try engine.ntt(testInput2)
        let gpuRecovered2 = try engine.intt(gpuNTT2)
        var correct2 = true
        var mismatches2 = 0
        for i in 0..<testN2 {
            let expected = fr377ToInt(testInput2[i])
            let got = fr377ToInt(gpuRecovered2[i])
            if expected != got {
                if mismatches2 < 3 {
                    print("  RT MISMATCH at \(i): expected \(expected[0]), got \(got[0])")
                }
                correct2 = false
                mismatches2 += 1
            }
        }
        print("  Four-step round-trip (2^20): \(correct2 ? "PASS" : "FAIL") (\(mismatches2) mismatches)")

        // Performance benchmark
        print("\n--- BLS12-377 NTT Performance ---")
        let sizes = [10, 12, 14, 16, 18, 20, 22, 24]

        for logN in sizes {
            let n = 1 << logN
            var data = [Fr377](repeating: Fr377.zero, count: n)
            rng = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = fr377FromInt(rng >> 32)
            }

            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Fr377>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
            }

            // Warmup
            for _ in 0..<3 {
                try engine.ntt(data: dataBuf, logN: logN)
                try engine.intt(data: dataBuf, logN: logN)
            }

            // Reload fresh data
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
            }

            var nttTimes = [Double]()
            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)
            }

            nttTimes.sort()
            let nttMedian = nttTimes[5]
            let elemPerSec = Double(n) / (nttMedian / 1000)
            print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                        logN, n, nttMedian, elemPerSec / 1e6))
        }

    } catch {
        print("  [FAIL] BLS12-377 NTT error: \(error)")
    }

    // BLS12-377 Fq field + curve correctness
    print("\n--- BLS12-377 Fq Field + Curve Correctness ---")

    // Field arithmetic
    let a = fq377FromInt(42)
    let b = fq377FromInt(100)
    let c = fq377Add(a, b)
    let cInt = fq377ToInt(c)
    print("  42 + 100 = \(cInt[0]) \(cInt[0] == 142 ? "[pass]" : "[FAIL]")")

    let d = fq377Mul(a, b)
    let dInt = fq377ToInt(d)
    print("  42 * 100 = \(dInt[0]) \(dInt[0] == 4200 ? "[pass]" : "[FAIL]")")

    let e = fq377Sub(b, a)
    let eInt = fq377ToInt(e)
    print("  100 - 42 = \(eInt[0]) \(eInt[0] == 58 ? "[pass]" : "[FAIL]")")

    let ainv = fq377Inverse(a)
    let check = fq377Mul(a, ainv)
    let checkInt = fq377ToInt(check)
    print("  42 * 42^(-1) = \(checkInt[0]) \(checkInt[0] == 1 ? "[pass]" : "[FAIL]")")

    // Curve operations
    let g = bls12377Generator()
    let gProj = point377FromAffine(g)
    let g2 = point377Double(gProj)
    let g3 = point377Add(g2, gProj)

    if let g2a = point377ToAffine(g2) {
        let x2 = fq377ToInt(g2a.x)
        print("  2G computed, x[0] = \(String(format: "0x%016llx", x2[0]))")

        // Verify 2G is on curve: y^2 = x^3 + 1
        let y2 = fq377Sqr(g2a.y)
        let x3 = fq377Mul(fq377Sqr(g2a.x), g2a.x)
        let rhs = fq377Add(x3, fq377FromInt(1))
        let y2Int = fq377ToInt(y2)
        let rhsInt = fq377ToInt(rhs)
        let onCurve = y2Int == rhsInt
        print("  2G on curve: \(onCurve ? "[pass]" : "[FAIL]")")
    } else {
        print("  [FAIL] 2G is identity")
    }

    if let g3a = point377ToAffine(g3) {
        let y2 = fq377Sqr(g3a.y)
        let x3 = fq377Mul(fq377Sqr(g3a.x), g3a.x)
        let rhs = fq377Add(x3, fq377FromInt(1))
        let onCurve = fq377ToInt(y2) == fq377ToInt(rhs)
        print("  3G on curve: \(onCurve ? "[pass]" : "[FAIL]")")
    } else {
        print("  [FAIL] 3G is identity")
    }

    // Verify G + G = 2G (from addition)
    let g1p1 = point377Add(gProj, gProj)
    if let g1p1a = point377ToAffine(g1p1), let g2a = point377ToAffine(g2) {
        let match = fq377ToInt(g1p1a.x) == fq377ToInt(g2a.x) && fq377ToInt(g1p1a.y) == fq377ToInt(g2a.y)
        print("  G + G = 2G: \(match ? "[pass]" : "[FAIL]")")
    }
}

public func runBLS12377GLVTest() {
    print("\n=== BLS12-377 GLV Decomposition Test ===")

    // Test: for scalar k, verify k ≡ k1 + k2·λ (mod r)
    let gen = bls12377Generator()
    let gProj = point377FromAffine(gen)

    // λ in Fr377 Montgomery form
    let lambdaMont = fr377Mul(Fr377.from64(BLS12377GLV.LAMBDA), Fr377.from64(Fr377.R2_MOD_R))

    // β in Fq377 Montgomery form
    let betaMont = BLS12377GLV.betaMontgomery

    // Test 1: Verify β³ = 1 in Fq
    let beta2 = fq377Mul(betaMont, betaMont)
    let beta3 = fq377Mul(beta2, betaMont)
    let beta3Int = fq377ToInt(beta3)
    let isCubeRoot = beta3Int[0] == 1 && beta3Int[1] == 0 && beta3Int[2] == 0 &&
                     beta3Int[3] == 0 && beta3Int[4] == 0 && beta3Int[5] == 0
    print("  β³ = 1 in Fq: \(isCubeRoot ? "[pass]" : "[FAIL]")")
    if !isCubeRoot {
        print("    β³ = \(beta3Int.map { String(format: "0x%016llx", $0) })")
    }

    // Test 2: Verify λ² + λ + 1 = 0 in Fr
    let lambda2 = fr377Mul(lambdaMont, lambdaMont)
    let sum = fr377Add(fr377Add(lambda2, lambdaMont), Fr377.one)
    let sumIsZero = sum.isZero || fr377ToInt(sum) == Fr377.P.map { _ in UInt64(0) }
    let sumInt = fr377ToInt(sum)
    let sumAllZero = sumInt.allSatisfy { $0 == 0 }
    print("  λ² + λ + 1 = 0 in Fr: \(sumAllZero ? "[pass]" : "[FAIL]")")
    if !sumAllZero {
        print("    λ² + λ + 1 = \(sumInt.map { String(format: "0x%016llx", $0) })")
    }

    // Test 3: Verify φ(G) = λ·G (endomorphism matches scalar multiplication)
    // φ(x,y) = (β·x, y)
    let endoG = Point377Affine(x: fq377Mul(betaMont, gen.x), y: gen.y)
    let endoGProj = point377FromAffine(endoG)

    // Compute λ·G by scalar multiplication
    // λ is ~128 bits, so we need to use the full 253-bit scalar mul
    let lambdaLimbs = BLS12377GLV.LAMBDA
    // Convert to Fr377 scalar (8×32-bit)
    var lambdaScalar = [UInt32](repeating: 0, count: 8)
    for i in 0..<4 {
        lambdaScalar[2*i] = UInt32(lambdaLimbs[i] & 0xFFFFFFFF)
        lambdaScalar[2*i+1] = UInt32(lambdaLimbs[i] >> 32)
    }

    // Scalar mul using double-and-add with full scalar
    var lambdaG = point377Identity()
    var base = gProj
    for i in 0..<8 {
        var word = lambdaScalar[i]
        for _ in 0..<32 {
            if word & 1 == 1 {
                lambdaG = point377IsIdentity(lambdaG) ? base : point377Add(lambdaG, base)
            }
            base = point377Double(base)
            word >>= 1
        }
    }

    if let endoAff = point377ToAffine(endoGProj), let lambdaAff = point377ToAffine(lambdaG) {
        let match = fq377ToInt(endoAff.x) == fq377ToInt(lambdaAff.x) &&
                    fq377ToInt(endoAff.y) == fq377ToInt(lambdaAff.y)
        print("  φ(G) = λ·G: \(match ? "[pass]" : "[FAIL]")")
        if !match {
            print("    endo x[0]: \(String(format: "0x%016llx", fq377ToInt(endoAff.x)[0]))")
            print("    λ·G  x[0]: \(String(format: "0x%016llx", fq377ToInt(lambdaAff.x)[0]))")
        }
    }

    // Test 4: GLV decomposition roundtrip
    var rng: UInt64 = 0xCAFE_BABE_1234_5678
    var allPass = true
    for trial in 0..<10 {
        var k = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            k[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        k = BLS12377MSM.reduceModR(k)

        let (k1, k2, neg1, neg2) = BLS12377GLV.decompose(k)

        // Verify: k ≡ k1 + k2·λ (mod r) with sign adjustments
        // k1_val = neg1 ? (r - k1) : k1, then k1_contribution = neg1 ? -(r-k1) = k1-r ≡ k1 mod r ... hmm
        // Actually: the MSM computes k1·P + k2·φ(P) where we negate P or φ(P) based on neg flags
        // So the effective scalar is: (neg1 ? -k1 : k1) + (neg2 ? -k2 : k2)·λ

        // Convert k1, k2 to Fr377 Montgomery form
        var k1Mont = fr377Mul(Fr377.from64([
            UInt64(k1[0]) | (UInt64(k1[1]) << 32),
            UInt64(k1[2]) | (UInt64(k1[3]) << 32),
            UInt64(k1[4]) | (UInt64(k1[5]) << 32),
            UInt64(k1[6]) | (UInt64(k1[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))

        var k2Mont = fr377Mul(Fr377.from64([
            UInt64(k2[0]) | (UInt64(k2[1]) << 32),
            UInt64(k2[2]) | (UInt64(k2[3]) << 32),
            UInt64(k2[4]) | (UInt64(k2[5]) << 32),
            UInt64(k2[6]) | (UInt64(k2[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))

        if neg1 { k1Mont = fr377Neg(k1Mont) }
        if neg2 { k2Mont = fr377Neg(k2Mont) }

        // recomputed = k1 + k2 * lambda
        let k2lambda = fr377Mul(k2Mont, lambdaMont)
        let recomputed = fr377Add(k1Mont, k2lambda)

        let kMont = fr377Mul(Fr377.from64([
            UInt64(k[0]) | (UInt64(k[1]) << 32),
            UInt64(k[2]) | (UInt64(k[3]) << 32),
            UInt64(k[4]) | (UInt64(k[5]) << 32),
            UInt64(k[6]) | (UInt64(k[7]) << 32)
        ]), Fr377.from64(Fr377.R2_MOD_R))

        let kInt = fr377ToInt(kMont)
        let reInt = fr377ToInt(recomputed)
        if kInt != reInt {
            print("  [FAIL] Trial \(trial): k ≠ k1 + k2·λ")
            print("    k  = \(kInt.map { String(format: "0x%016llx", $0) })")
            print("    re = \(reInt.map { String(format: "0x%016llx", $0) })")
            allPass = false
        }
    }
    print("  GLV decomposition roundtrip (10 trials): \(allPass ? "[pass]" : "[FAIL]")")
}

public func runBLS12377MSMBench() {
    print("\n=== BLS12-377 MSM Benchmark ===")

    do {
        let engine = try BLS12377MSM()
        // engine.useGLV = true  // GLV available but adds overhead for 12-limb fields

        // Generate points: G, 2G, 3G, ...
        let gen = bls12377Generator()
        let gProj = point377FromAffine(gen)

        let logSizes = [8, 10, 12, 14, 16, 17, 18]
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
        let genTime = (CFAbsoluteTimeGetCurrent() - genT0) * 1000
        print("  Point generation: \(String(format: "%.1f", genTime))ms")

        // Verify a few points are on-curve
        for idx in [0, 1, maxN - 1] {
            let p = allPoints[idx]
            let y2 = fq377Sqr(p.y)
            let x3 = fq377Mul(fq377Sqr(p.x), p.x)
            let rhs = fq377Add(x3, fq377FromInt(1))
            let onCurve = fq377ToInt(y2) == fq377ToInt(rhs)
            if !onCurve {
                print("  [FAIL] Point \(idx) not on curve!")
                return
            }
        }
        print("  Sample point on-curve checks: [pass]")

        // Generate random scalars (Fr377 is 253-bit, stored as 8×32-bit)
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

        // Correctness check: small MSM (8 points) against CPU double-and-add
        let smallN = 8
        let smallPts = Array(allPoints.prefix(smallN))
        let smallScalars = allScalars.prefix(smallN).map { BLS12377MSM.reduceModR($0) }

        // CPU reference
        var cpuResult = point377Identity()
        for i in 0..<smallN {
            let s = smallScalars[i]
            let sInt = Int(s[0]) | (Int(s[1]) << 32)  // just low 64 bits for simple scalar mul
            let term = point377MulInt(point377FromAffine(smallPts[i]), sInt & 0xFFFF)
            cpuResult = point377Add(cpuResult, term)
        }

        // GPU result (using same low-bits-only scalars for comparison)
        let lowScalars = smallScalars.map { s -> [UInt32] in
            [s[0] & 0xFFFF, 0, 0, 0, 0, 0, 0, 0]
        }
        let gpuResult = try engine.msm(points: smallPts, scalars: lowScalars)

        if let cpuAff = point377ToAffine(cpuResult), let gpuAff = point377ToAffine(gpuResult) {
            let match = fq377ToInt(cpuAff.x) == fq377ToInt(gpuAff.x) &&
                        fq377ToInt(cpuAff.y) == fq377ToInt(gpuAff.y)
            print("  MSM correctness (8 pts, low scalars): \(match ? "[pass]" : "[FAIL]")")
            if !match {
                print("    CPU x[0]: \(String(format: "0x%016llx", fq377ToInt(cpuAff.x)[0]))")
                print("    GPU x[0]: \(String(format: "0x%016llx", fq377ToInt(gpuAff.x)[0]))")
            }
        } else {
            let cpuId = point377IsIdentity(cpuResult)
            let gpuId = point377IsIdentity(gpuResult)
            print("  MSM correctness: cpu_identity=\(cpuId) gpu_identity=\(gpuId) \(cpuId == gpuId ? "[pass]" : "[FAIL]")")
        }

        // Performance benchmark
        print("\n--- BLS12-377 MSM Performance ---")
        for (idx, n) in sizes.enumerated() {
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            // Warmup
            let _ = try engine.msm(points: points, scalars: scalars)

            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: points, scalars: scalars)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            print(String(format: "  MSM 2^%-2d = %7d pts: %7.1f ms", logSizes[idx], n, median))
        }

    } catch {
        print("  [FAIL] BLS12-377 MSM error: \(error)")
    }
}
