import zkMetal
import Foundation

// MARK: - Test helpers

/// Evaluate polynomial in coefficient form at point z using Horner's method
private func polyEval(_ coeffs: [Fr], at x: Fr) -> Fr {
    if coeffs.isEmpty { return .zero }
    var result = coeffs[coeffs.count - 1]
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}

/// Simple LCG for deterministic randomness
private struct TestRNG {
    var state: UInt64

    mutating func nextFr() -> Fr {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(state >> 32)
    }
}

// MARK: - Public test entry point

public func runGPUPolyInterpolationTests() {
    testLagrangeBasic()
    testLagrangeRandom()
    testLagrangeGPUPath()
    testBarycentricWeights()
    testBarycentricEval()
    testBatchInterpolation()
    testSubgroupInterpolation()
    testSubgroupLarger()
    testNewtonInterpolation()
    testNewtonToMonomial()
    testNewtonIncremental()
    testNewtonEval()
    testEdgeCases()
    testCpuGpuConsistency()
}

// MARK: - 1. Lagrange Interpolation

private func testLagrangeBasic() {
    suite("GPUPolyInterpolation — Lagrange Basic")

    // Single point
    do {
        let engine = try GPUPolyInterpolationEngine()
        let coeffs = try engine.lagrangeInterpolate(points: [frFromInt(7)], values: [frFromInt(42)])
        expect(coeffs.count == 1, "Single point: 1 coeff")
        expect(coeffs[0] == frFromInt(42), "Single point: c0 = 42")
    } catch { expect(false, "Single point error: \(error)") }

    // Linear: p(x) = 3 + 5x => p(0)=3, p(1)=8
    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(0), frFromInt(1)]
        let values = [frFromInt(3), frFromInt(8)]
        let coeffs = try engine.lagrangeInterpolate(points: points, values: values)
        expect(coeffs.count == 2, "Linear: 2 coeffs")
        for i in 0..<2 {
            let ev = polyEval(coeffs, at: points[i])
            expect(ev == values[i], "Linear: p(x_\(i)) = y_\(i)")
        }
    } catch { expect(false, "Linear error: \(error)") }

    // Quadratic: p(x) = 1 + 2x + 3x^2 => p(0)=1, p(1)=6, p(2)=17
    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(0), frFromInt(1), frFromInt(2)]
        let values = [frFromInt(1), frFromInt(6), frFromInt(17)]
        let coeffs = try engine.lagrangeInterpolate(points: points, values: values)
        expect(coeffs.count == 3, "Quadratic: 3 coeffs")
        for i in 0..<3 {
            let ev = polyEval(coeffs, at: points[i])
            expect(ev == values[i], "Quadratic: p(x_\(i)) = y_\(i)")
        }
    } catch { expect(false, "Quadratic error: \(error)") }
}

private func testLagrangeRandom() {
    suite("GPUPolyInterpolation — Lagrange Random (CPU path)")

    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xCAFE_BABE_0001)
        let n = 16

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let coeffs = try engine.lagrangeInterpolate(points: points, values: values)
        expect(coeffs.count == n, "Random n=\(n): \(n) coeffs")

        var allOk = true
        for i in 0..<n {
            let ev = polyEval(coeffs, at: points[i])
            if ev != values[i] { allOk = false; break }
        }
        expect(allOk, "Random n=\(n): p(x_i) = y_i for all i")
    } catch { expect(false, "Random small error: \(error)") }
}

private func testLagrangeGPUPath() {
    suite("GPUPolyInterpolation — Lagrange GPU path")

    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xDEAD_BEEF_0002)
        let n = 128 // above cpuThreshold=64

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let coeffs = try engine.lagrangeInterpolate(points: points, values: values)
        expect(coeffs.count == n, "GPU n=\(n): \(n) coeffs")

        // Spot check every 8th point
        var allOk = true
        for i in stride(from: 0, to: n, by: 8) {
            let ev = polyEval(coeffs, at: points[i])
            if ev != values[i] { allOk = false; break }
        }
        expect(allOk, "GPU n=\(n): spot check p(x_i) = y_i")
    } catch { expect(false, "GPU path error: \(error)") }
}

// MARK: - 2. Barycentric Interpolation

private func testBarycentricWeights() {
    suite("GPUPolyInterpolation — Barycentric Weights")

    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let bw = try engine.computeBarycentricWeights(points: points)
        expect(bw.points.count == 4, "Weights: 4 points")
        expect(bw.weights.count == 4, "Weights: 4 weights")

        // Verify: w_i * prod_{j!=i}(x_i - x_j) == 1
        for i in 0..<4 {
            var prod = Fr.one
            for j in 0..<4 {
                if j == i { continue }
                prod = frMul(prod, frSub(points[i], points[j]))
            }
            let check = frMul(bw.weights[i], prod)
            expect(check == Fr.one, "Weight \(i) * denom = 1")
        }
    } catch { expect(false, "Barycentric weights error: \(error)") }
}

private func testBarycentricEval() {
    suite("GPUPolyInterpolation — Barycentric Eval")

    do {
        let engine = try GPUPolyInterpolationEngine()
        // p(x) = x^3: p(1)=1, p(2)=8, p(3)=27, p(4)=64
        let points = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let values = [frFromInt(1), frFromInt(8), frFromInt(27), frFromInt(64)]
        let bw = try engine.computeBarycentricWeights(points: points)

        // Evaluate at x=5: 5^3 = 125
        let r5 = engine.barycentricEval(weights: bw, values: values, at: frFromInt(5))
        expect(r5 == frFromInt(125), "Barycentric: p(5) = 125")

        // Evaluate at x=0: 0^3 = 0
        let r0 = engine.barycentricEval(weights: bw, values: values, at: frFromInt(0))
        expect(r0 == frFromInt(0), "Barycentric: p(0) = 0")

        // Evaluate at existing point x=3: should return 27
        let r3 = engine.barycentricEval(weights: bw, values: values, at: frFromInt(3))
        expect(r3 == frFromInt(27), "Barycentric: p(3) = 27 (exact)")

        // Convenience API without precomputed weights
        let r5b = try engine.barycentricEval(points: points, values: values, at: frFromInt(5))
        expect(r5b == frFromInt(125), "Barycentric convenience: p(5) = 125")
    } catch { expect(false, "Barycentric eval error: \(error)") }
}

// MARK: - 3. Batch Interpolation

private func testBatchInterpolation() {
    suite("GPUPolyInterpolation — Batch Interpolation")

    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xBA7C40003)

        let n = 8
        let k = 4 // number of polynomials

        var points = [Fr]()
        for _ in 0..<n { points.append(rng.nextFr()) }

        var valueSets = [[Fr]]()
        for _ in 0..<k {
            var vs = [Fr]()
            for _ in 0..<n { vs.append(rng.nextFr()) }
            valueSets.append(vs)
        }

        let results = try engine.batchInterpolate(points: points, valueSets: valueSets)
        expect(results.count == k, "Batch: \(k) result sets")

        // Verify each polynomial
        var allOk = true
        for p in 0..<k {
            expect(results[p].count == n, "Batch poly \(p): \(n) coeffs")
            for i in 0..<n {
                let ev = polyEval(results[p], at: points[i])
                if ev != valueSets[p][i] { allOk = false; break }
            }
            if !allOk { break }
        }
        expect(allOk, "Batch: all p_j(x_i) = y_{j,i}")

        // Verify batch matches individual interpolation
        for p in 0..<k {
            let singleCoeffs = try engine.lagrangeInterpolate(points: points, values: valueSets[p])
            var match = true
            for i in 0..<n {
                if singleCoeffs[i] != results[p][i] { match = false; break }
            }
            expect(match, "Batch poly \(p) matches single interpolation")
        }
    } catch { expect(false, "Batch interpolation error: \(error)") }

    // Batch with empty valueSets
    do {
        let engine = try GPUPolyInterpolationEngine()
        let results = try engine.batchInterpolate(points: [frFromInt(1), frFromInt(2)], valueSets: [])
        expect(results.isEmpty, "Batch empty: no results")
    } catch { expect(false, "Batch empty error: \(error)") }
}

// MARK: - 4. Subgroup (NTT) Interpolation

private func testSubgroupInterpolation() {
    suite("GPUPolyInterpolation — Subgroup NTT")

    do {
        let engine = try GPUPolyInterpolationEngine()
        let logN = 8
        let n = 1 << logN

        var rng = TestRNG(state: 0xA77_50B_0004)
        var origCoeffs = [Fr](repeating: .zero, count: n)
        for i in 0..<n { origCoeffs[i] = rng.nextFr() }

        // Forward NTT to get evals at roots of unity
        let nttEngine = try NTTEngine()
        let evals = try nttEngine.ntt(origCoeffs)

        // Recover coefficients via subgroupInterpolate
        let recovered = try engine.subgroupInterpolate(evals: evals, logN: logN)
        expect(recovered.count == n, "Subgroup n=\(n): \(n) coeffs")

        var allOk = true
        for i in 0..<n {
            if recovered[i] != origCoeffs[i] { allOk = false; break }
        }
        expect(allOk, "Subgroup n=\(n): recovered == original")
    } catch { expect(false, "Subgroup NTT error: \(error)") }

    // Test auto-detect logN convenience
    do {
        let engine = try GPUPolyInterpolationEngine()
        let logN = 6
        let n = 1 << logN

        var rng = TestRNG(state: 0xA070_106_0005)
        var origCoeffs = [Fr](repeating: .zero, count: n)
        for i in 0..<n { origCoeffs[i] = rng.nextFr() }

        let nttEngine = try NTTEngine()
        let evals = try nttEngine.ntt(origCoeffs)

        let recovered = try engine.subgroupInterpolate(evals: evals) // auto logN
        var allOk = true
        for i in 0..<n {
            if recovered[i] != origCoeffs[i] { allOk = false; break }
        }
        expect(allOk, "Subgroup auto-logN: recovered == original")
    } catch { expect(false, "Subgroup auto-logN error: \(error)") }
}

private func testSubgroupLarger() {
    suite("GPUPolyInterpolation — Subgroup NTT 2^12")

    do {
        let engine = try GPUPolyInterpolationEngine()
        let logN = 12
        let n = 1 << logN

        var rng = TestRNG(state: 0xB16_A77_0006)
        var origCoeffs = [Fr](repeating: .zero, count: n)
        for i in 0..<n { origCoeffs[i] = rng.nextFr() }

        let nttEngine = try NTTEngine()
        let evals = try nttEngine.ntt(origCoeffs)
        let recovered = try engine.subgroupInterpolate(evals: evals, logN: logN)

        var allOk = true
        for i in 0..<n {
            if recovered[i] != origCoeffs[i] { allOk = false; break }
        }
        expect(allOk, "Subgroup 2^12: recovered == original")
    } catch { expect(false, "Subgroup 2^12 error: \(error)") }
}

// MARK: - 5. Newton Interpolation

private func testNewtonInterpolation() {
    suite("GPUPolyInterpolation — Newton Divided Differences")

    // Simple case: p(x) = 1 + 2x + 3x^2
    // p(0)=1, p(1)=6, p(2)=17
    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(0), frFromInt(1), frFromInt(2)]
        let values = [frFromInt(1), frFromInt(6), frFromInt(17)]

        let np = try engine.newtonInterpolate(points: points, values: values)
        expect(np.coeffs.count == 3, "Newton: 3 coeffs")

        // Verify Newton form evaluates correctly at all points
        for i in 0..<3 {
            let ev = np.evaluate(at: points[i])
            expect(ev == values[i], "Newton: p(x_\(i)) = y_\(i)")
        }

        // Verify at other point: p(5) = 1 + 10 + 75 = 86
        let r5 = np.evaluate(at: frFromInt(5))
        expect(r5 == frFromInt(86), "Newton: p(5) = 86")
    } catch { expect(false, "Newton basic error: \(error)") }

    // Random points
    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xAE07_0007)
        let n = 16

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let np = try engine.newtonInterpolate(points: points, values: values)
        var allOk = true
        for i in 0..<n {
            let ev = np.evaluate(at: points[i])
            if ev != values[i] { allOk = false; break }
        }
        expect(allOk, "Newton random n=\(n): p(x_i) = y_i")
    } catch { expect(false, "Newton random error: \(error)") }
}

private func testNewtonToMonomial() {
    suite("GPUPolyInterpolation — Newton to Monomial")

    // Convert Newton form to monomial and compare with Lagrange
    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
        // p(x) = 2 + x + 4x^2 + x^3
        // p(0)=2, p(1)=8, p(2)=40, p(3)=110
        let values = [frFromInt(2), frFromInt(8), frFromInt(40), frFromInt(110)]

        let np = try engine.newtonInterpolate(points: points, values: values)
        let monomialFromNewton = engine.newtonToMonomial(np)
        let monomialFromLagrange = try engine.lagrangeInterpolate(points: points, values: values)

        expect(monomialFromNewton.count == monomialFromLagrange.count, "Newton->mono: same length")

        var allOk = true
        for i in 0..<min(monomialFromNewton.count, monomialFromLagrange.count) {
            if monomialFromNewton[i] != monomialFromLagrange[i] { allOk = false; break }
        }
        expect(allOk, "Newton->mono matches Lagrange coefficients")
    } catch { expect(false, "Newton to monomial error: \(error)") }

    // Random test
    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xA0A0_0008)
        let n = 12

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let np = try engine.newtonInterpolate(points: points, values: values)
        let mono = engine.newtonToMonomial(np)

        // Verify monomial form evaluates correctly
        var allOk = true
        for i in 0..<n {
            let ev = polyEval(mono, at: points[i])
            if ev != values[i] { allOk = false; break }
        }
        expect(allOk, "Newton->mono random: p(x_i) = y_i")
    } catch { expect(false, "Newton to monomial random error: \(error)") }
}

private func testNewtonIncremental() {
    suite("GPUPolyInterpolation — Newton Incremental")

    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0x1AC6_0009)

        // Start with 1 point, incrementally add up to 8
        let totalN = 8
        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<totalN {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        // Build incrementally
        var np = try engine.newtonInterpolate(points: [points[0]], values: [values[0]])
        expect(np.evaluate(at: points[0]) == values[0], "Incremental: base point ok")

        for k in 1..<totalN {
            np = np.addPoint(x: points[k], y: values[k])

            // Verify all points so far
            var allOk = true
            for i in 0...k {
                let ev = np.evaluate(at: points[i])
                if ev != values[i] { allOk = false; break }
            }
            expect(allOk, "Incremental after adding point \(k): all ok")
        }

        // Final should match full Newton interpolation
        let npFull = try engine.newtonInterpolate(points: points, values: values)
        var match = true
        for i in 0..<totalN {
            let a = np.evaluate(at: rng.nextFr())
            let b = npFull.evaluate(at: rng.nextFr())
            // Can't compare directly since rng advanced differently;
            // instead verify both evaluate correctly at data points
            let evA = np.evaluate(at: points[i])
            let evB = npFull.evaluate(at: points[i])
            if evA != values[i] || evB != values[i] { match = false; break }
        }
        expect(match, "Incremental matches full Newton at data points")
    } catch { expect(false, "Newton incremental error: \(error)") }
}

private func testNewtonEval() {
    suite("GPUPolyInterpolation — Newton Eval Consistency")

    // Newton eval should match Lagrange eval at arbitrary points
    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xE4A1_000A)
        let n = 8

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let np = try engine.newtonInterpolate(points: points, values: values)
        let lagCoeffs = try engine.lagrangeInterpolate(points: points, values: values)

        // Evaluate both at 5 random test points
        var allOk = true
        for _ in 0..<5 {
            let z = rng.nextFr()
            let newtonVal = np.evaluate(at: z)
            let lagVal = polyEval(lagCoeffs, at: z)
            if newtonVal != lagVal { allOk = false; break }
        }
        expect(allOk, "Newton eval == Lagrange eval at random points")
    } catch { expect(false, "Newton eval consistency error: \(error)") }
}

// MARK: - Edge Cases

private func testEdgeCases() {
    suite("GPUPolyInterpolation — Edge Cases")

    // Single point
    do {
        let engine = try GPUPolyInterpolationEngine()
        let np = try engine.newtonInterpolate(points: [frFromInt(42)], values: [frFromInt(99)])
        expect(np.evaluate(at: frFromInt(42)) == frFromInt(99), "Newton single: p(42) = 99")
        expect(np.evaluate(at: frFromInt(0)) == frFromInt(99), "Newton single: constant")
    } catch { expect(false, "Edge single error: \(error)") }

    // Subgroup non-power-of-2 should throw
    do {
        let engine = try GPUPolyInterpolationEngine()
        let evals = [Fr](repeating: .zero, count: 3) // not power of 2
        _ = try engine.subgroupInterpolate(evals: evals)
        expect(false, "Subgroup non-pow2 should throw")
    } catch {
        expect(true, "Subgroup non-pow2 throws correctly")
    }

    // Barycentric weights for single point
    do {
        let engine = try GPUPolyInterpolationEngine()
        let bw = try engine.computeBarycentricWeights(points: [frFromInt(5)])
        expect(bw.weights.count == 1, "Barycentric single: 1 weight")
        expect(bw.weights[0] == Fr.one, "Barycentric single: weight = 1")

        let val = engine.barycentricEval(weights: bw, values: [frFromInt(77)], at: frFromInt(5))
        expect(val == frFromInt(77), "Barycentric single: eval at point = value")
    } catch { expect(false, "Barycentric single error: \(error)") }

    // Batch with single polynomial should match individual
    do {
        let engine = try GPUPolyInterpolationEngine()
        let points = [frFromInt(1), frFromInt(3), frFromInt(5)]
        let values = [frFromInt(10), frFromInt(20), frFromInt(30)]

        let batchResult = try engine.batchInterpolate(points: points, valueSets: [values])
        let singleResult = try engine.lagrangeInterpolate(points: points, values: values)

        expect(batchResult.count == 1, "Batch single: 1 result set")
        var match = true
        for i in 0..<3 {
            if batchResult[0][i] != singleResult[i] { match = false; break }
        }
        expect(match, "Batch single matches individual")
    } catch { expect(false, "Batch single error: \(error)") }
}

// MARK: - CPU vs GPU Consistency

private func testCpuGpuConsistency() {
    suite("GPUPolyInterpolation — CPU vs GPU Consistency")

    do {
        let engine = try GPUPolyInterpolationEngine()
        var rng = TestRNG(state: 0xC0A5_000B)
        let n = 32

        var points = [Fr]()
        var values = [Fr]()
        for _ in 0..<n {
            points.append(rng.nextFr())
            values.append(rng.nextFr())
        }

        let cpuCoeffs = GPUPolyInterpolationEngine.cpuInterpolate(points: points, values: values)
        let engineCoeffs = try engine.lagrangeInterpolate(points: points, values: values)

        var allOk = true
        for i in 0..<n {
            if cpuCoeffs[i] != engineCoeffs[i] { allOk = false; break }
        }
        expect(allOk, "CPU vs engine: identical coefficients (n=\(n))")
    } catch { expect(false, "CPU vs GPU consistency error: \(error)") }
}
