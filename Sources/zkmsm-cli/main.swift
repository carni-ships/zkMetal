// zkmsm-cli — Metal GPU Multi-Scalar Multiplication for BN254
//
// CLI tool that performs MSM on the GPU using Metal compute shaders.
// Usage:
//   echo '{"points": [...], "scalars": [...]}' | zkmsm
//   zkmsm --bench <n_points>
//   zkmsm --test
//   zkmsm --info

import Foundation
import Metal
import zkMetal

// MARK: - Correctness Test

func runCorrectnessTest() throws {
    fputs("=== BN254 Field Arithmetic Correctness Test ===\n", stderr)

    let a = fpFromInt(42)
    let aInt = fpToInt(a)
    assert(aInt[0] == 42 && aInt[1] == 0, "Montgomery round-trip failed for 42")
    fputs("  [pass] Montgomery form round-trip\n", stderr)

    let b = fpFromInt(100)
    let c = fpAdd(a, b)
    let cInt = fpToInt(c)
    assert(cInt[0] == 142, "42 + 100 should be 142, got \(cInt[0])")
    fputs("  [pass] Field addition: 42 + 100 = 142\n", stderr)

    let d = fpMul(a, b)
    let dInt = fpToInt(d)
    assert(dInt[0] == 4200, "42 * 100 should be 4200, got \(dInt[0])")
    fputs("  [pass] Field multiplication: 42 * 100 = 4200\n", stderr)

    let e = fpSub(b, a)
    let eInt = fpToInt(e)
    assert(eInt[0] == 58, "100 - 42 should be 58, got \(eInt[0])")
    fputs("  [pass] Field subtraction: 100 - 42 = 58\n", stderr)

    let aInv = fpInverse(a)
    let shouldBeOne = fpMul(a, aInv)
    let oneInt = fpToInt(shouldBeOne)
    assert(oneInt[0] == 1 && oneInt[1] == 0 && oneInt[2] == 0 && oneInt[3] == 0,
           "42 * 42^-1 should be 1")
    fputs("  [pass] Field inverse: 42 * 42^(-1) = 1\n", stderr)

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)
    let gProj = pointFromAffine(g)

    let y2 = fpSqr(gy)
    let x3 = fpMul(gx, fpSqr(gx))
    let three = fpFromInt(3)
    let rhs = fpAdd(x3, three)
    let y2Int = fpToInt(y2)
    let rhsInt = fpToInt(rhs)
    assert(y2Int[0] == rhsInt[0] && y2Int[1] == rhsInt[1] &&
           y2Int[2] == rhsInt[2] && y2Int[3] == rhsInt[3],
           "Generator not on curve!")
    fputs("  [pass] Generator G=(1,2) is on BN254 curve\n", stderr)

    let g2 = pointDouble(gProj)
    let g2Affine = pointToAffine(g2)!
    let g2y2 = fpSqr(g2Affine.y)
    let g2x3 = fpMul(g2Affine.x, fpSqr(g2Affine.x))
    let g2rhs = fpAdd(g2x3, three)
    let g2y2Int = fpToInt(g2y2)
    let g2rhsInt = fpToInt(g2rhs)
    assert(g2y2Int[0] == g2rhsInt[0] && g2y2Int[1] == g2rhsInt[1] &&
           g2y2Int[2] == g2rhsInt[2] && g2y2Int[3] == g2rhsInt[3],
           "2G not on curve!")
    fputs("  [pass] 2G is on curve\n", stderr)

    let g2xInt = fpToInt(g2Affine.x)
    fputs("  [info] 2G.x = 0x\(g2xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    fputs("  [info] 2G.y = 0x\(fpToInt(g2Affine.y).reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)

    let gPlusG = pointAdd(gProj, gProj)
    let gPlusGAffine = pointToAffine(gPlusG)!
    let gPlusGxInt = fpToInt(gPlusGAffine.x)
    assert(gPlusGxInt[0] == g2xInt[0] && gPlusGxInt[1] == g2xInt[1] &&
           gPlusGxInt[2] == g2xInt[2] && gPlusGxInt[3] == g2xInt[3],
           "G + G != 2G")
    fputs("  [pass] G + G = 2G\n", stderr)

    let g4 = pointDouble(g2)
    let g4Affine = pointToAffine(g4)!
    let g4y2 = fpSqr(g4Affine.y)
    let g4x3 = fpMul(g4Affine.x, fpSqr(g4Affine.x))
    let g4rhs = fpAdd(g4x3, three)
    let g4y2Int = fpToInt(g4y2)
    let g4rhsInt = fpToInt(g4rhs)
    assert(g4y2Int[0] == g4rhsInt[0] && g4y2Int[1] == g4rhsInt[1] &&
           g4y2Int[2] == g4rhsInt[2] && g4y2Int[3] == g4rhsInt[3],
           "4G not on curve!")
    fputs("  [pass] 4G is on curve\n", stderr)

    let g3 = pointAdd(g2, gProj)
    let g3Affine = pointToAffine(g3)!
    let g3y2 = fpSqr(g3Affine.y)
    let g3x3 = fpMul(g3Affine.x, fpSqr(g3Affine.x))
    let g3rhs = fpAdd(g3x3, three)
    let g3y2Int = fpToInt(g3y2)
    let g3rhsInt = fpToInt(g3rhs)
    assert(g3y2Int[0] == g3rhsInt[0] && g3y2Int[1] == g3rhsInt[1] &&
           g3y2Int[2] == g3rhsInt[2] && g3y2Int[3] == g3rhsInt[3],
           "3G not on curve!")
    fputs("  [pass] 3G = 2G + G is on curve\n", stderr)

    fputs("\n=== All correctness tests passed ===\n", stderr)
}

// MARK: - Stdin MSM

func runStdinMSM() throws {
    let inputData = FileHandle.standardInput.readDataToEndOfFile()
    guard let json = try JSONSerialization.jsonObject(with: inputData) as? [String: Any] else {
        fputs("Error: invalid JSON input\n", stderr)
        throw MSMError.invalidInput
    }

    guard let pointsArr = json["points"] as? [[String]],
          let scalarsArr = json["scalars"] as? [String] else {
        fputs("Error: expected {\"points\": [[\"0x..\",\"0x..\"], ...], \"scalars\": [\"0x..\", ...]}\n", stderr)
        throw MSMError.invalidInput
    }

    guard pointsArr.count == scalarsArr.count, !pointsArr.isEmpty else {
        fputs("Error: points and scalars must have equal non-zero length\n", stderr)
        throw MSMError.invalidInput
    }

    let n = pointsArr.count

    var points: [PointAffine] = []
    points.reserveCapacity(n)
    for pair in pointsArr {
        guard pair.count == 2 else { throw MSMError.invalidInput }
        points.append(PointAffine(x: fpFromHex(pair[0]), y: fpFromHex(pair[1])))
    }

    var scalars: [[UInt32]] = []
    scalars.reserveCapacity(n)
    for hexStr in scalarsArr {
        let clean = hexStr.hasPrefix("0x") ? String(hexStr.dropFirst(2)) : hexStr
        let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for i in 0..<8 {
            let start = padded.index(padded.startIndex, offsetBy: (7 - i) * 8)
            let end = padded.index(start, offsetBy: 8)
            limbs[i] = UInt32(padded[start..<end], radix: 16) ?? 0
        }
        scalars.append(limbs)
    }

    let engine = try MetalMSM()
    let start = CFAbsoluteTimeGetCurrent()
    let result = try engine.msm(points: points, scalars: scalars)
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

    var output: [String: Any] = ["time_ms": elapsed]
    if let affine = pointToAffine(result) {
        output["x"] = fpToHex(affine.x)
        output["y"] = fpToHex(affine.y)
        output["infinity"] = false
    } else {
        output["x"] = "0x0"
        output["y"] = "0x0"
        output["infinity"] = true
    }

    let outputData = try JSONSerialization.data(withJSONObject: output, options: .prettyPrinted)
    print(String(data: outputData, encoding: .utf8)!)
}

// MARK: - Benchmark

func runBenchmark(nPoints: Int) throws {
    fputs("zkmsm benchmark: \(nPoints) points on \(MTLCreateSystemDefaultDevice()?.name ?? "unknown GPU")\n", stderr)

    let engine = try MetalMSM()

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)

    fputs("Generating \(nPoints) distinct points...\n", stderr)
    let setupStart = CFAbsoluteTimeGetCurrent()
    var projPoints = [PointProjective]()
    projPoints.reserveCapacity(nPoints)
    let gProj = pointFromAffine(g)
    var acc = gProj
    for _ in 0..<nPoints {
        projPoints.append(acc)
        acc = pointAdd(acc, gProj)
    }
    let points = batchToAffine(projPoints)
    fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - setupStart) * 1000))ms\n", stderr)

    var scalars: [[UInt32]] = []
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    for _ in 0..<nPoints {
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        scalars.append(limbs)
    }

    fputs("Warmup...\n", stderr)
    let _ = try engine.msm(points: points, scalars: scalars)

    let nRuns = 5
    var times = [Double]()
    var result = pointIdentity()
    for _ in 0..<nRuns {
        let start = CFAbsoluteTimeGetCurrent()
        result = try engine.msm(points: points, scalars: scalars)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        times.append(elapsed * 1000)
    }
    times.sort()
    let median = times[nRuns / 2]
    fputs("MSM(\(nPoints)): \(String(format: "%.3f", median))ms median [\(times.map { String(format: "%.1f", $0) }.joined(separator: ", "))]ms\n", stderr)
    fputs("GPU: \(engine.device.name)\n", stderr)
    fputs("Max threadgroup: \(engine.reduceSortedFunction.maxTotalThreadsPerThreadgroup)\n", stderr)

    if let affine = pointToAffine(result) {
        let xInt = fpToInt(affine.x)
        fputs("Result.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    } else {
        fputs("Result: point at infinity\n", stderr)
    }

    if nPoints <= 256 {
        fputs("Computing CPU reference MSM...\n", stderr)
        var cpuResult = pointIdentity()
        for i in 0..<nPoints {
            var r = pointIdentity()
            let p = pointFromAffine(points[i])
            for bit in stride(from: 255, through: 0, by: -1) {
                r = pointDouble(r)
                let limbIdx = bit / 32
                let bitPos = bit % 32
                if (scalars[i][limbIdx] >> bitPos) & 1 == 1 {
                    r = pointIsIdentity(r) ? p : pointAdd(r, p)
                }
            }
            cpuResult = pointIsIdentity(cpuResult) ? r : pointAdd(cpuResult, r)
        }
        if let affine = pointToAffine(cpuResult) {
            let xInt = fpToInt(affine.x)
            fputs("CPU ref.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
        }
    }
}

// MARK: - GLV Test

func runGLVTest() throws {
    fputs("=== GLV Endomorphism Test ===\n", stderr)

    let beta2 = fpSqr(FP_BETA)
    let beta3 = fpMul(FP_BETA, beta2)
    let b3int = fpToInt(beta3)
    assert(b3int[0] == 1 && b3int[1] == 0 && b3int[2] == 0 && b3int[3] == 0,
           "β³ ≠ 1 mod p!")
    fputs("  [pass] β³ ≡ 1 mod p\n", stderr)

    let sum = fpAdd(fpAdd(beta2, FP_BETA), Fp.one)
    assert(sum.isZero, "β² + β + 1 ≠ 0 mod p")
    fputs("  [pass] β² + β + 1 ≡ 0 mod p\n", stderr)

    let gx = fpFromInt(1); let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)

    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var testK = [UInt32](repeating: 0, count: 8)
    for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; testK[j] = UInt32(truncatingIfNeeded: rng >> 32) }
    let (k1, neg1, k2, neg2) = glvDecompose(testK)
    fputs("  k1=\(k1[0...3]), neg1=\(neg1), k2=\(k2[0...3]), neg2=\(neg2)\n", stderr)

    let gProj = pointFromAffine(g)
    var r42 = pointIdentity()
    var baseG = gProj
    for bit in 0..<256 {
        let limbIdx = bit / 32
        let bitPos = bit % 32
        if (testK[limbIdx] >> bitPos) & 1 == 1 {
            r42 = pointIsIdentity(r42) ? baseG : pointAdd(r42, baseG)
        }
        baseG = pointDouble(baseG)
    }
    if let a42 = pointToAffine(r42) {
        fputs("  [k]G.x = 0x\(fpToInt(a42.x).reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
    }

    var p1 = g
    if neg1 { p1 = pointNegateAffine(p1) }
    var p2 = applyEndomorphism(g)
    if neg2 { p2 = pointNegateAffine(p2) }

    var rk1 = pointIdentity()
    var baseP1 = pointFromAffine(p1)
    for bit in 0..<128 {
        let limbIdx = bit / 32
        let bitPos = bit % 32
        if (k1[limbIdx] >> bitPos) & 1 == 1 {
            rk1 = pointIsIdentity(rk1) ? baseP1 : pointAdd(rk1, baseP1)
        }
        baseP1 = pointDouble(baseP1)
    }

    var rk2 = pointIdentity()
    var baseP2 = pointFromAffine(p2)
    for bit in 0..<128 {
        let limbIdx = bit / 32
        let bitPos = bit % 32
        if (k2[limbIdx] >> bitPos) & 1 == 1 {
            rk2 = pointIsIdentity(rk2) ? baseP2 : pointAdd(rk2, baseP2)
        }
        baseP2 = pointDouble(baseP2)
    }

    let rGlv = pointIsIdentity(rk1) ? rk2 : (pointIsIdentity(rk2) ? rk1 : pointAdd(rk1, rk2))
    if let a42 = pointToAffine(r42), let aGlv = pointToAffine(rGlv) {
        let match = fpToInt(a42.x) == fpToInt(aGlv.x)
        fputs("  \(match ? "[pass]" : "[FAIL]") GLV decomposition: [k]G == [k1]G + [k2]φ(G)\n", stderr)
    }

    fputs("=== GLV Test Done ===\n", stderr)
}

// MARK: - Main

func main() throws {
    let args = CommandLine.arguments

    if args.contains("--test") {
        try runCorrectnessTest()
        return
    }

    if args.contains("--glv-test") {
        try runGLVTest()
        return
    }

    if args.contains("--msm-test") {
        fputs("=== MSM Correctness Test v2 ===\n", stderr)
        let engine = try MetalMSM()
        let gx = fpFromInt(1); let gy = fpFromInt(2)
        let g = PointAffine(x: gx, y: gy)

        // Test 0: MSM([3], [G]) first — before anything else
        engine.useGLV = false
        let r0 = try engine.msm(points: [g], scalars: [[3,0,0,0,0,0,0,0]])
        let expected3G = pointToAffine(pointAdd(pointFromAffine(g), pointDouble(pointFromAffine(g))))!
        if let a0 = pointToAffine(r0) {
            let match = fpToInt(a0.x) == fpToInt(expected3G.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(3*G) == 3G (first call)\n", stderr)
            if !match {
                fputs("    got.x  = \(fpToInt(a0.x))\n", stderr)
                fputs("    want.x = \(fpToInt(expected3G.x))\n", stderr)
            }
        } else { fputs("  [FAIL] 3*G = infinity (first call)\n", stderr) }

        // Test 1: MSM([1], [G]) = G
        let r1 = try engine.msm(points: [g], scalars: [[1,0,0,0,0,0,0,0]])
        if let a1 = pointToAffine(r1) {
            let match = fpToInt(a1.x) == fpToInt(g.x) && fpToInt(a1.y) == fpToInt(g.y)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(1*G) == G\n", stderr)
            if !match {
                fputs("    got.x  = \(fpToInt(a1.x))\n", stderr)
                fputs("    want.x = \(fpToInt(g.x))\n", stderr)
            }
        } else { fputs("  [FAIL] 1*G = infinity\n", stderr) }

        // Test 2: MSM([42], [G]) = 42G
        let s42: [[UInt32]] = [[42,0,0,0,0,0,0,0]]
        let r42 = try engine.msm(points: [g], scalars: s42)
        var expected42G = pointIdentity()
        var base42 = pointFromAffine(g)
        for bit in 0..<8 {
            if (42 >> bit) & 1 == 1 {
                expected42G = pointIsIdentity(expected42G) ? base42 : pointAdd(expected42G, base42)
            }
            base42 = pointDouble(base42)
        }
        let a42opt = pointToAffine(r42)
        let e42opt = pointToAffine(expected42G)
        fputs("  42G gpu_inf=\(a42opt == nil) cpu_inf=\(e42opt == nil)\n", stderr)
        if let a42 = a42opt, let e42 = e42opt {
            let match = fpToInt(a42.x) == fpToInt(e42.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(42*G) == 42G\n", stderr)
            fputs("    got.x  = \(fpToInt(a42.x))\n", stderr)
            fputs("    want.x = \(fpToInt(e42.x))\n", stderr)
        } else {
            fputs("  [FAIL] 42*G = infinity or expected42G = infinity\n", stderr)
        }

        // Test 2b: MSM([2], [G]) = 2G
        let twoG = pointToAffine(pointDouble(pointFromAffine(g)))!
        let r2 = try engine.msm(points: [g], scalars: [[2,0,0,0,0,0,0,0]])
        if let a2 = pointToAffine(r2) {
            let match = fpToInt(a2.x) == fpToInt(twoG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(2*G) == 2G\n", stderr)
        } else { fputs("  [FAIL] 2*G = infinity\n", stderr) }

        // Test 2c: MSM([3], [G]) = 3G
        let threeG = pointToAffine(pointAdd(pointFromAffine(g), pointDouble(pointFromAffine(g))))!
        let r3a = try engine.msm(points: [g], scalars: [[3,0,0,0,0,0,0,0]])
        if let a3a = pointToAffine(r3a) {
            let match = fpToInt(a3a.x) == fpToInt(threeG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(3*G) == 3G\n", stderr)
        } else { fputs("  [FAIL] 3*G = infinity\n", stderr) }

        // Test 2d: MSM([4], [G]) = 4G
        let fourG = pointToAffine(pointDouble(pointDouble(pointFromAffine(g))))!
        let r4 = try engine.msm(points: [g], scalars: [[4,0,0,0,0,0,0,0]])
        if let a4 = pointToAffine(r4) {
            let match = fpToInt(a4.x) == fpToInt(fourG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(4*G) == 4G\n", stderr)
            if !match { fputs("    got.x=\(fpToInt(a4.x))\n    want.x=\(fpToInt(fourG.x))\n", stderr) }
        } else { fputs("  [FAIL] 4*G = infinity\n", stderr) }

        // Test 2e: MSM([5], [G]) = 5G
        let fiveG = pointToAffine(pointAdd(pointFromAffine(fourG), pointFromAffine(g)))!
        let r5 = try engine.msm(points: [g], scalars: [[5,0,0,0,0,0,0,0]])
        if let a5 = pointToAffine(r5) {
            let match = fpToInt(a5.x) == fpToInt(fiveG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(5*G) == 5G\n", stderr)
            if !match { fputs("    got.x=\(fpToInt(a5.x))\n    want.x=\(fpToInt(fiveG.x))\n", stderr) }
        } else { fputs("  [FAIL] 5*G = infinity\n", stderr) }

        // Test 2f: MSM([6], [G]) = 6G
        let sixG = pointToAffine(pointDouble(pointAdd(pointFromAffine(g), pointDouble(pointFromAffine(g)))))!
        let r6 = try engine.msm(points: [g], scalars: [[6,0,0,0,0,0,0,0]])
        if let a6 = pointToAffine(r6) {
            let match = fpToInt(a6.x) == fpToInt(sixG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(6*G) == 6G\n", stderr)
            if !match { fputs("    got.x=\(fpToInt(a6.x))\n    want.x=\(fpToInt(sixG.x))\n", stderr) }
        } else { fputs("  [FAIL] 6*G = infinity\n", stderr) }

        // Test 3: MSM([1,1], [G, 2G]) = 3G
        let r3 = try engine.msm(points: [g, twoG], scalars: [[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0]])
        if let a3 = pointToAffine(r3) {
            let match = fpToInt(a3.x) == fpToInt(threeG.x)
            fputs("  \(match ? "[pass]" : "[FAIL]") MSM(1*G + 1*2G) == 3G\n", stderr)
        } else { fputs("  [FAIL] 1*G + 1*2G = infinity\n", stderr) }

        fputs("=== MSM Test Done ===\n", stderr)
        engine.useGLV = true
        return
    }

    if args.count >= 3 && args[1] == "--bench" {
        let n = Int(args[2]) ?? 1024
        if args.contains("--sweep") {
            let engine = try MetalMSM()
            let gx = fpFromInt(1); let gy = fpFromInt(2)
            let g = PointAffine(x: gx, y: gy)
            var scalars: [[UInt32]] = []
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            for _ in 0..<n {
                var limbs: [UInt32] = Array(repeating: 0, count: 8)
                for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
                scalars.append(limbs)
            }
            let points = [PointAffine](repeating: g, count: n)
            for wb: UInt32 in [18, 17, 16, 15, 14, 13, 12] {
                engine.windowBitsOverride = wb
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: points, scalars: scalars)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                fputs("  w=\(wb): \(String(format: "%.1f", elapsed * 1000))ms\n", stderr)
            }
            return
        }
        if let wbIdx = args.firstIndex(of: "--wb"), wbIdx + 1 < args.count,
           let wb = UInt32(args[wbIdx + 1]) {
            let engine = try MetalMSM()
            engine.windowBitsOverride = wb
            let gx = fpFromInt(1); let gy = fpFromInt(2)
            let g = PointAffine(x: gx, y: gy)
            fputs("Generating \(n) distinct points...\n", stderr)
            var projPts = [PointProjective]()
            projPts.reserveCapacity(n)
            let gProj = pointFromAffine(g)
            var ac = gProj
            for _ in 0..<n { projPts.append(ac); ac = pointAdd(ac, gProj) }
            let pts = batchToAffine(projPts)
            var scalars: [[UInt32]] = []
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            for _ in 0..<n {
                var limbs: [UInt32] = Array(repeating: 0, count: 8)
                for j in 0..<8 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; limbs[j] = UInt32(truncatingIfNeeded: rng >> 32) }
                scalars.append(limbs)
            }
            fputs("Warmup...\n", stderr)
            let _ = try engine.msm(points: pts, scalars: scalars)
            let start = CFAbsoluteTimeGetCurrent()
            let result = try engine.msm(points: pts, scalars: scalars)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            fputs("MSM(\(n), w=\(wb)): \(String(format: "%.3f", elapsed * 1000))ms\n", stderr)
            if let affine = pointToAffine(result) {
                let xInt = fpToInt(affine.x)
                fputs("Result.x = 0x\(xInt.reversed().map { String(format: "%016llx", $0) }.joined())\n", stderr)
            }
            return
        }
        try runBenchmark(nPoints: n)
        return
    }

    if args.contains("--info") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("{\"error\": \"No Metal GPU available\"}")
            return
        }
        let info: [String: Any] = [
            "gpu": device.name,
            "unified_memory": device.hasUnifiedMemory,
            "max_buffer_length": device.maxBufferLength,
            "max_threadgroup_memory": device.maxThreadgroupMemoryLength,
        ]
        let data = try JSONSerialization.data(withJSONObject: info, options: .prettyPrinted)
        print(String(data: data, encoding: .utf8)!)
        return
    }

    if args.contains("--msm") || args.count == 1 {
        try runStdinMSM()
        return
    }

    fputs("zkmsm: Metal GPU MSM for BN254\n", stderr)
    fputs("Usage:\n", stderr)
    fputs("  echo '{...}' | zkmsm     Compute MSM from stdin JSON\n", stderr)
    fputs("  zkmsm --msm              Same as above (explicit flag)\n", stderr)
    fputs("  zkmsm --test             Run correctness tests\n", stderr)
    fputs("  zkmsm --bench <n_points> Benchmark MSM\n", stderr)
    fputs("  zkmsm --info             Show GPU info\n", stderr)
}

try main()
