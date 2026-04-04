// Circle STARK benchmark: M31 field ops, CM31, Circle NTT
import zkMetal
import Foundation

public func runCircleBench() {
    print("=== Circle STARK Benchmark (Mersenne31) ===")

    // ---- M31 field correctness ----
    print("\n--- Mersenne31 Field (p = 2^31 - 1) ---")

    let a = M31(v: 42), b = M31(v: 100)
    let sum = m31Add(a, b)
    if sum.v == 142 { print("  [pass] M31 add: 42 + 100 = 142") }
    else { print("  [FAIL] M31 add: \(sum.v)"); return }

    let prod = m31Mul(a, b)
    if prod.v == 4200 { print("  [pass] M31 mul: 42 * 100 = 4200") }
    else { print("  [FAIL] M31 mul: \(prod.v)"); return }

    let diff = m31Sub(b, a)
    if diff.v == 58 { print("  [pass] M31 sub: 100 - 42 = 58") }
    else { print("  [FAIL] M31 sub: \(diff.v)"); return }

    // Inverse
    let aInv = m31Inverse(a)
    let aTimesInv = m31Mul(a, aInv)
    if aTimesInv.v == 1 { print("  [pass] M31 inverse: 42 * 42^(-1) = 1") }
    else { print("  [FAIL] M31 inverse: \(aTimesInv.v)"); return }

    // Edge cases
    let big1 = M31(v: M31.P - 1)
    let big2 = M31(v: M31.P - 2)
    let bigProd = m31Mul(big1, big2)
    if bigProd.v == 2 { print("  [pass] M31 large mul: (p-1)*(p-2) = 2") }
    else { print("  [FAIL] M31 large mul: \(bigProd.v)"); return }

    // Zero handling
    let z = M31(v: 0)
    let addZ = m31Add(a, z)
    if addZ.v == 42 { print("  [pass] M31 add zero: 42 + 0 = 42") }
    else { print("  [FAIL] M31 add zero: \(addZ.v)"); return }

    let negA = m31Neg(a)
    let sumNeg = m31Add(a, negA)
    if sumNeg.v == 0 { print("  [pass] M31 negation: 42 + (-42) = 0") }
    else { print("  [FAIL] M31 negation: \(sumNeg.v)"); return }

    // p-1 + 1 = 0
    let pMinus1 = M31(v: M31.P - 1)
    let wrapSum = m31Add(pMinus1, M31.one)
    if wrapSum.v == 0 { print("  [pass] M31 wraparound: (p-1) + 1 = 0") }
    else { print("  [FAIL] M31 wraparound: \(wrapSum.v)"); return }

    // ---- CM31 tests ----
    print("\n--- CM31 (Gaussian integers mod p) ---")

    let c1 = CM31(a: M31(v: 3), b: M31(v: 4))
    let c2 = CM31(a: M31(v: 1), b: M31(v: 2))
    let cProd = cm31Mul(c1, c2)
    // (3+4i)(1+2i) = 3+6i+4i+8i^2 = 3+10i-8 = -5+10i
    let expectedReal = m31Sub(M31.zero, M31(v: 5))  // p - 5
    if cProd.a.v == expectedReal.v && cProd.b.v == 10 {
        print("  [pass] CM31 mul: (3+4i)(1+2i) = (-5+10i)")
    } else {
        print("  [FAIL] CM31 mul: got (\(cProd.a.v)+\(cProd.b.v)i)"); return
    }

    let cInv = cm31Inverse(c1)
    let cOne = cm31Mul(c1, cInv)
    if cOne.a.v == 1 && cOne.b.v == 0 {
        print("  [pass] CM31 inverse: (3+4i) * (3+4i)^(-1) = 1")
    } else {
        print("  [FAIL] CM31 inverse: got (\(cOne.a.v)+\(cOne.b.v)i)"); return
    }

    // ---- Circle group tests ----
    print("\n--- Circle Group ---")

    let gen = CirclePoint.generator
    if gen.isOnCircle { print("  [pass] Generator is on circle: x^2 + y^2 = 1") }
    else { print("  [FAIL] Generator not on circle"); return }

    // Test group operation
    let g2 = circleGroupMul(gen, gen)
    if g2.isOnCircle { print("  [pass] g^2 is on circle") }
    else { print("  [FAIL] g^2 not on circle"); return }

    // Generator order: g^(2^31) should be identity
    var gPow = gen
    for _ in 0..<30 { gPow = circleGroupMul(gPow, gPow) }
    if gPow != CirclePoint.identity {
        gPow = circleGroupMul(gPow, gPow)
        if gPow == CirclePoint.identity {
            print("  [pass] Circle generator order: g^(2^31) = identity")
        } else {
            print("  [FAIL] g^(2^31) != identity: (\(gPow.x.v), \(gPow.y.v))"); return
        }
    } else {
        print("  [WARN] g^(2^30) = identity (order too small)")
    }

    // Subgroup generator
    let gen3 = circleSubgroupGenerator(logN: 3)
    let gen3pow8 = circleGroupPow(gen3, 8)
    if gen3pow8 == CirclePoint.identity {
        print("  [pass] Subgroup gen order 2^3: g_3^8 = identity")
    } else {
        print("  [FAIL] g_3^8 != identity"); return
    }
    let gen3pow4 = circleGroupPow(gen3, 4)
    if gen3pow4 != CirclePoint.identity {
        print("  [pass] g_3^4 != identity (primitive)")
    } else {
        print("  [FAIL] g_3 is not primitive"); return
    }

    // ---- CPU Circle NTT correctness ----
    print("\n--- Circle NTT (CPU reference) ---")

    for logN in 1...8 {
        let n = 1 << logN
        var coeffs = [M31](repeating: M31.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF + UInt64(logN)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = M31(v: UInt32(rng >> 33))  // values in [0, 2^31)
            if coeffs[i].v >= M31.P { coeffs[i].v = coeffs[i].v - M31.P }
        }

        let evals = CircleNTTEngine.cpuNTT(coeffs, logN: logN)
        let recovered = CircleNTTEngine.cpuINTT(evals, logN: logN)

        var match = true
        for i in 0..<n {
            if recovered[i].v != coeffs[i].v { match = false; break }
        }
        if match { print("  [pass] CPU Circle NTT roundtrip: N = \(n)") }
        else {
            print("  [FAIL] CPU Circle NTT roundtrip: N = \(n)")
            if n <= 8 {
                print("    coeffs:   \(coeffs.map { $0.v })")
                print("    evals:    \(evals.map { $0.v })")
                print("    recover:  \(recovered.map { $0.v })")
            }
            return
        }
    }

    // ---- GPU Circle NTT ----
    print("\n--- Circle NTT (GPU) ---")
    do {
        let engine = try CircleNTTEngine()

        for logN in 1...12 {
            let n = 1 << logN
            var coeffs = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0xCAFE_BABE + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = M31(v: UInt32(rng >> 33))
                if coeffs[i].v >= M31.P { coeffs[i].v = coeffs[i].v - M31.P }
            }

            let cpuEvals = CircleNTTEngine.cpuNTT(coeffs, logN: logN)
            let gpuEvals = try engine.ntt(coeffs)

            var fwdMatch = true
            for i in 0..<n {
                if gpuEvals[i].v != cpuEvals[i].v { fwdMatch = false; break }
            }

            let gpuRecovered = try engine.intt(gpuEvals)
            var invMatch = true
            for i in 0..<n {
                if gpuRecovered[i].v != coeffs[i].v { invMatch = false; break }
            }

            if fwdMatch && invMatch {
                print("  [pass] GPU Circle NTT: N = \(n)")
            } else {
                print("  [FAIL] GPU Circle NTT: N = \(n) (fwd=\(fwdMatch), inv=\(invMatch))")
                if n <= 8 {
                    print("    cpu evals: \(cpuEvals.map { $0.v })")
                    print("    gpu evals: \(gpuEvals.map { $0.v })")
                }
            }
        }

        // ---- Benchmarks ----
        print("\n--- Circle NTT Benchmarks ---")

        let benchSizes = [10, 12, 14, 16, 18, 20]
        for logN in benchSizes {
            let n = 1 << logN
            var data = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0x1234_5678 + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = M31(v: UInt32(rng >> 33) % M31.P)
            }

            // Warmup
            let _ = try engine.ntt(data)

            // Timed runs
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.ntt(data)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            print(String(format: "  Circle NTT 2^%-2d = %7d: %7.2f ms", logN, n, median))
        }
    } catch {
        print("  [FAIL] GPU init error: \(error)")
    }

    // ---- M31 throughput benchmark ----
    print("\n--- M31 Throughput ---")

    var warmup = M31.one
    for _ in 0..<10000 { warmup = m31Mul(warmup, a) }
    let iters = 1_000_000
    let mulStart = CFAbsoluteTimeGetCurrent()
    var acc = M31.one
    for _ in 0..<iters { acc = m31Mul(acc, a) }
    let mulElapsed = (CFAbsoluteTimeGetCurrent() - mulStart) * 1e9 / Double(iters)
    print(String(format: "  M31 mul: %.1f ns/op (%.0f M ops/s)", mulElapsed, 1e3 / mulElapsed))

    let addStart = CFAbsoluteTimeGetCurrent()
    var addAcc = M31.one
    for _ in 0..<iters { addAcc = m31Add(addAcc, a) }
    let addElapsed = (CFAbsoluteTimeGetCurrent() - addStart) * 1e9 / Double(iters)
    print(String(format: "  M31 add: %.1f ns/op (%.0f M ops/s)", addElapsed, 1e3 / addElapsed))

    let invStart = CFAbsoluteTimeGetCurrent()
    var invAcc = M31(v: 42)
    let invIters = 1000
    for _ in 0..<invIters { invAcc = m31Inverse(invAcc) }
    let invElapsed = (CFAbsoluteTimeGetCurrent() - invStart) * 1e6 / Double(invIters)
    print(String(format: "  M31 inv: %.1f us/op", invElapsed))

    // Use accumulators to prevent optimization
    if acc.v == 0 && addAcc.v == 0 && invAcc.v == 0 { print("  (anti-opt)") }

    print("\nCircle STARK benchmark complete.")
}
