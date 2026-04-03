// Polynomial Operations Benchmark
import zkMetal
import Foundation

public func runPolyBench() {
    print("=== Polynomial Operations Benchmark ===")

    do {
        let engine = try PolyEngine()

        // Correctness: multiply (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let a = [frFromInt(1), frFromInt(2)]
        let b = [frFromInt(3), frFromInt(4)]
        let c = try engine.multiply(a, b)
        let c0 = frToInt(c[0])[0], c1 = frToInt(c[1])[0], c2 = frToInt(c[2])[0]
        if c0 == 3 && c1 == 10 && c2 == 8 {
            print("  [pass] Poly multiply (1+2x)*(3+4x) = 3+10x+8x^2")
        } else {
            print("  [FAIL] Expected [3,10,8], got [\(c0),\(c1),\(c2)]")
            return
        }

        // Correctness: evaluate 1 + 2x + 3x^2 at x=5 → 1 + 10 + 75 = 86
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let points = [frFromInt(5)]
        let evals = try engine.evaluate(poly, at: points)
        let eval0 = frToInt(evals[0])[0]
        if eval0 == 86 {
            print("  [pass] Poly eval (1+2x+3x^2) at x=5 = 86")
        } else {
            print("  [FAIL] Expected 86, got \(eval0)")
            return
        }

        // Correctness: add/sub
        let sum = try engine.add([frFromInt(10), frFromInt(20)], [frFromInt(3), frFromInt(7)])
        if frToInt(sum[0])[0] == 13 && frToInt(sum[1])[0] == 27 {
            print("  [pass] Poly add")
        } else {
            print("  [FAIL] Poly add")
            return
        }

        let diff = try engine.sub([frFromInt(10), frFromInt(20)], [frFromInt(3), frFromInt(7)])
        if frToInt(diff[0])[0] == 7 && frToInt(diff[1])[0] == 13 {
            print("  [pass] Poly sub")
        } else {
            print("  [FAIL] Poly sub")
            return
        }

        // Benchmark: polynomial multiplication at various sizes
        print("\n--- Polynomial Multiply (NTT-based) ---")
        var rng: UInt64 = 0xDEAD_BEEF
        for logN in [10, 12, 14, 16] {
            let n = 1 << logN
            var pa = [Fr](repeating: Fr.zero, count: n)
            var pb = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                pa[i] = frFromInt(rng >> 32)
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                pb[i] = frFromInt(rng >> 32)
            }

            // CPU baseline: NTT-based poly multiply using vanilla cpuNTT
            let resultLen = pa.count + pb.count - 1
            var nPad = 1
            while nPad < resultLen { nPad <<= 1 }
            let padLogN = Int(log2(Double(nPad)))
            let aPad = pa + [Fr](repeating: Fr.zero, count: nPad - pa.count)
            let bPad = pb + [Fr](repeating: Fr.zero, count: nPad - pb.count)

            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let aNTT = NTTEngine.cpuNTT(aPad, logN: padLogN)
            let bNTT = NTTEngine.cpuNTT(bPad, logN: padLogN)
            var cNTT = [Fr](repeating: Fr.zero, count: nPad)
            for i in 0..<nPad { cNTT[i] = frMul(aNTT[i], bNTT[i]) }
            let _ = NTTEngine.cpuINTT(cNTT, logN: padLogN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // GPU (warmup + timed)
            let _ = try engine.multiply(pa, pb)
            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.multiply(pa, pb)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuTime = times[2]
            let speedup = cpuTime / gpuTime
            print(String(format: "  deg 2^%-2d | Vanilla CPU: %8.1fms | GPU: %6.2fms | GPU vs Vanilla: **%.0fx**",
                        logN, cpuTime, gpuTime, speedup))
        }

        // Correctness check for chunked eval (degree >= 256)
        do {
            let cDeg = 1024
            var cCoeffs = [Fr](repeating: Fr.zero, count: cDeg)
            for i in 0..<cDeg {
                cCoeffs[i] = frFromInt(UInt64(i + 1))
            }
            let cPts = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let gpuResults = try engine.evaluate(cCoeffs, at: cPts)
            // p(1) = 1+2+...+1024 = 1024*1025/2 = 524800
            let expected1 = frFromInt(524800)
            let match = frToInt(gpuResults[0]) == frToInt(expected1)
            print("  [" + (match ? "pass" : "FAIL") + "] Chunked eval (deg 1024 at x=1)")
        }

        // Benchmark: multi-point evaluation
        print("\n--- Multi-point Evaluation (Horner) ---")
        for logN in [10, 12, 14] {
            let deg = 1 << logN
            let numPts = 1 << logN
            var coeffs = [Fr](repeating: Fr.zero, count: deg)
            var pts = [Fr](repeating: Fr.zero, count: numPts)
            for i in 0..<deg {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }
            for i in 0..<numPts {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                pts[i] = frFromInt(rng >> 32)
            }

            let _ = try engine.evaluate(coeffs, at: pts)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluate(coeffs, at: pts)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            print(String(format: "  deg 2^%-2d, %d points: %7.2f ms (Horner)", logN, numPts, times[2]))
        }

        // Benchmark: subproduct tree evaluation
        print("\n--- Multi-point Evaluation (Subproduct Tree) ---")
        for logN in [10, 12, 14] {
            let deg = 1 << logN
            let numPts = 1 << logN
            var coeffs = [Fr](repeating: Fr.zero, count: deg)
            var pts = [Fr](repeating: Fr.zero, count: numPts)
            for i in 0..<deg {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }
            for i in 0..<numPts {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                pts[i] = frFromInt(rng >> 32)
            }

            // Correctness: compare tree vs Horner
            let hornerResults = try engine.evaluate(coeffs, at: pts)
            let treeResults = try engine.evaluateTree(coeffs, at: pts)
            var match = true
            for i in 0..<numPts {
                if frToInt(hornerResults[i]) != frToInt(treeResults[i]) {
                    match = false
                    print("  [FAIL] Mismatch at point \(i)")
                    break
                }
            }
            if match && logN <= 12 {
                print("  [pass] Tree eval matches Horner (deg 2^\(logN))")
            }

            let _ = try engine.evaluateTree(coeffs, at: pts)  // warmup

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluateTree(coeffs, at: pts)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            print(String(format: "  deg 2^%-2d, %d points: %7.2f ms (Tree)", logN, numPts, times[2]))
        }

    } catch {
        print("  [FAIL] Error: \(error)")
    }

    print("\nPolynomial benchmark complete.")
}
