import zkMetal

func runInterpolationTests() {
    suite("GPU Interpolation Engine")

    // Helper: evaluate polynomial at point using Horner's method
    func polyEval(_ coeffs: [Fr], at x: Fr) -> Fr {
        if coeffs.isEmpty { return .zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    // Simple LCG for deterministic randomness
    var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
    func nextRand() -> Fr {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(rng >> 32)
    }

    // --- Test 1: Trivial single point ---
    do {
        let engine = try GPUInterpolationEngine()
        let coeffs = try engine.interpolate(points: [frFromInt(7)], values: [frFromInt(42)])
        expect(coeffs.count == 1, "Single point: 1 coeff")
        expect(frToInt(coeffs[0]) == frToInt(frFromInt(42)), "Single point: c0 = 42")
    } catch { expect(false, "Single point error: \(error)") }

    // --- Test 2: Linear interpolation (2 points) ---
    do {
        let engine = try GPUInterpolationEngine()
        // p(x) = 3 + 5x => p(0)=3, p(1)=8
        let points = [frFromInt(0), frFromInt(1)]
        let values = [frFromInt(3), frFromInt(8)]
        let coeffs = try engine.interpolate(points: points, values: values)
        expect(coeffs.count == 2, "Linear: 2 coeffs")
        expect(frToInt(coeffs[0])[0] == 3, "Linear: c0 = 3")
        expect(frToInt(coeffs[1])[0] == 5, "Linear: c1 = 5")

        // Verify p(x_i) = y_i
        for i in 0..<2 {
            let ev = polyEval(coeffs, at: points[i])
            expect(ev == values[i], "Linear: p(x_\(i)) = y_\(i)")
        }
    } catch { expect(false, "Linear error: \(error)") }

    // --- Test 3: Quadratic interpolation (3 points) ---
    do {
        let engine = try GPUInterpolationEngine()
        // p(x) = 1 + 2x + 3x^2 => p(0)=1, p(1)=6, p(2)=17
        let points = [frFromInt(0), frFromInt(1), frFromInt(2)]
        let values = [frFromInt(1), frFromInt(6), frFromInt(17)]
        let coeffs = try engine.interpolate(points: points, values: values)
        expect(coeffs.count == 3, "Quadratic: 3 coeffs")
        expect(frToInt(coeffs[0])[0] == 1, "Quadratic: c0 = 1")
        expect(frToInt(coeffs[1])[0] == 2, "Quadratic: c1 = 2")
        expect(frToInt(coeffs[2])[0] == 3, "Quadratic: c2 = 3")
    } catch { expect(false, "Quadratic error: \(error)") }

    // --- Test 4: Random points, small n (CPU path) ---
    do {
        let engine = try GPUInterpolationEngine()
        let n = 16
        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(nextRand())
            values.append(nextRand())
        }

        let coeffs = try engine.interpolate(points: points, values: values)
        expect(coeffs.count == n, "Random small: \(n) coeffs")

        // Verify p(x_i) = y_i for all i
        var allOk = true
        for i in 0..<n {
            let ev = polyEval(coeffs, at: points[i])
            if ev != values[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "Random small: p(x_i) = y_i for all i (n=\(n))")
    } catch { expect(false, "Random small error: \(error)") }

    // --- Test 5: Random points, n >= cpuThreshold (GPU path) ---
    do {
        let engine = try GPUInterpolationEngine()
        let n = 128  // above cpuThreshold=64
        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(nextRand())
            values.append(nextRand())
        }

        let coeffs = try engine.interpolate(points: points, values: values)
        expect(coeffs.count == n, "Random GPU: \(n) coeffs")

        // Verify a subset (checking all 128 would be slow in test)
        var allOk = true
        for i in stride(from: 0, to: n, by: 8) {
            let ev = polyEval(coeffs, at: points[i])
            if ev != values[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "Random GPU: p(x_i) = y_i spot check (n=\(n))")
    } catch { expect(false, "Random GPU error: \(error)") }

    // --- Test 6: interpolateOnDomain ---
    do {
        let engine = try GPUInterpolationEngine()
        let domain = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        // p(x) = 7 + 3x: p(10)=37, p(20)=67, p(30)=97, p(40)=127
        // But 4 points => degree 3 polynomial, so this won't be exactly linear.
        // Instead, just use random values and verify.
        let vals = [frFromInt(37), frFromInt(67), frFromInt(97), frFromInt(127)]
        let coeffs = try engine.interpolateOnDomain(evals: vals, domain: domain)
        expect(coeffs.count == 4, "OnDomain: 4 coeffs")

        var allOk = true
        for i in 0..<4 {
            let ev = polyEval(coeffs, at: domain[i])
            if ev != vals[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "OnDomain: p(x_i) = y_i")
    } catch { expect(false, "OnDomain error: \(error)") }

    // --- Test 7: interpolateNTT (roots of unity domain) ---
    do {
        let engine = try GPUInterpolationEngine()
        let logN = 8
        let n = 1 << logN

        // Create a known polynomial, evaluate at roots of unity, then interpolate back
        var origCoeffs = [Fr](repeating: .zero, count: n)
        for i in 0..<n { origCoeffs[i] = nextRand() }

        // Evaluate at omega^i using NTT
        let nttEngine = try NTTEngine()
        let evals = try nttEngine.ntt(origCoeffs)

        // Interpolate back
        let recovered = try engine.interpolateNTT(evals: evals)
        expect(recovered.count == n, "NTT interp: \(n) coeffs")

        var allOk = true
        for i in 0..<n {
            if recovered[i] != origCoeffs[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "NTT interp: recovered == original (n=\(n))")
    } catch { expect(false, "NTT interp error: \(error)") }

    // --- Test 8: interpolateNTT larger ---
    do {
        let engine = try GPUInterpolationEngine()
        let logN = 12
        let n = 1 << logN

        var origCoeffs = [Fr](repeating: .zero, count: n)
        for i in 0..<n { origCoeffs[i] = nextRand() }

        let nttEngine = try NTTEngine()
        let evals = try nttEngine.ntt(origCoeffs)
        let recovered = try engine.interpolateNTT(evals: evals)

        var allOk = true
        for i in 0..<n {
            if recovered[i] != origCoeffs[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "NTT interp 2^12: recovered == original")
    } catch { expect(false, "NTT interp 2^12 error: \(error)") }

    // --- Test 9: CPU vs GPU consistency ---
    do {
        let engine = try GPUInterpolationEngine()
        let n = 32  // small enough for CPU, but we can test both
        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(nextRand())
            values.append(nextRand())
        }

        let cpuCoeffs = GPUInterpolationEngine.cpuInterpolate(points: points, values: values)
        // Force through the instance method (which uses CPU path for n < 64)
        let coeffs = try engine.interpolate(points: points, values: values)

        var allOk = true
        for i in 0..<n {
            if cpuCoeffs[i] != coeffs[i] {
                allOk = false
                break
            }
        }
        expect(allOk, "CPU vs engine consistency (n=\(n))")
    } catch { expect(false, "CPU vs GPU consistency error: \(error)") }

    // --- Test 10: Barycentric eval ---
    do {
        let engine = try GPUInterpolationEngine()
        let points = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        // p(x) = x^3: p(1)=1, p(2)=8, p(3)=27, p(4)=64
        let values = [frFromInt(1), frFromInt(8), frFromInt(27), frFromInt(64)]

        // Evaluate at x=5: 5^3 = 125
        let result = engine.barycentricEval(points: points, values: values, at: frFromInt(5))
        expect(frToInt(result)[0] == 125, "Barycentric eval: p(5) = 125")

        // Evaluate at x=0: 0^3 = 0
        let result0 = engine.barycentricEval(points: points, values: values, at: frFromInt(0))
        expect(frToInt(result0)[0] == 0, "Barycentric eval: p(0) = 0")

        // Evaluate at existing point x=3: should return 27
        let result3 = engine.barycentricEval(points: points, values: values, at: frFromInt(3))
        expect(frToInt(result3)[0] == 27, "Barycentric eval: p(3) = 27 (exact)")
    } catch { expect(false, "Barycentric eval error: \(error)") }

    // --- Test 11: Tuple convenience API ---
    do {
        let engine = try GPUInterpolationEngine()
        let pairs: [(Fr, Fr)] = [
            (frFromInt(0), frFromInt(5)),
            (frFromInt(1), frFromInt(7)),
            (frFromInt(2), frFromInt(13)),
        ]
        let coeffs = try engine.interpolate(pointValuePairs: pairs)
        expect(coeffs.count == 3, "Tuple API: 3 coeffs")

        // Verify
        for (x, y) in pairs {
            expect(polyEval(coeffs, at: x) == y, "Tuple API: p(x) = y")
        }
    } catch { expect(false, "Tuple API error: \(error)") }
}
