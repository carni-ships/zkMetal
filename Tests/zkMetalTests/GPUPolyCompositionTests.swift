import zkMetal
import Foundation

// MARK: - Test helpers

/// Simple LCG for test reproducibility
private struct CompRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

/// CPU reference Horner evaluation
private func cpuHorner(coeffs: [Fr], x: Fr) -> Fr {
    guard !coeffs.isEmpty else { return Fr.zero }
    var result = coeffs[coeffs.count - 1]
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}

/// CPU reference polynomial multiplication (schoolbook)
private func cpuPolyMul(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    guard !a.isEmpty && !b.isEmpty else { return [] }
    let resultLen = a.count + b.count - 1
    var result = [Fr](repeating: Fr.zero, count: resultLen)
    for i in 0..<a.count {
        for j in 0..<b.count {
            result[i + j] = frAdd(result[i + j], frMul(a[i], b[j]))
        }
    }
    return result
}

// MARK: - Public test entry point

public func runGPUPolyCompositionTests() {
    testComposeLinear()
    testComposeQuadratic()
    testComposeIdentity()
    testComposeConstant()
    testEvaluateCompositionCPU()
    testEvaluateCompositionGPU()
    testEvaluateCompositionConsistency()
    testDeepComposition()
    testDeepCompositionSingleColumn()
    testComposeRandomSmall()
    testPerformance()
}

// MARK: - compose: linear f into linear g

private func testComposeLinear() {
    suite("GPUPolyComposition compose linear")
    do {
        let engine = try GPUPolyCompositionEngine()

        // f(x) = 2 + 3x, g(x) = 4 + 5x
        // h(x) = f(g(x)) = 2 + 3*(4 + 5x) = 2 + 12 + 15x = 14 + 15x
        let two = frFromInt(2)
        let three = frFromInt(3)
        let four = frFromInt(4)
        let five = frFromInt(5)
        let fourteen = frFromInt(14)
        let fifteen = frFromInt(15)

        let f = [two, three]
        let g = [four, five]

        let h = try engine.compose(f: f, g: g)

        expectEqual(h.count, 2, "compose linear: result length")
        expectEqual(h[0], fourteen, "compose linear: h[0] = 14")
        expectEqual(h[1], fifteen, "compose linear: h[1] = 15")

        // Verify by evaluation: h(x) == f(g(x)) at a random point
        var rng = CompRNG(state: 0xC0DE0001)
        let x = rng.nextFr()
        let gx = cpuHorner(coeffs: g, x: x)
        let fgx = cpuHorner(coeffs: f, x: gx)
        let hx = cpuHorner(coeffs: h, x: x)
        expectEqual(hx, fgx, "compose linear: h(x) == f(g(x))")

    } catch { expect(false, "compose linear error: \(error)") }
}

// MARK: - compose: quadratic f into linear g

private func testComposeQuadratic() {
    suite("GPUPolyComposition compose quadratic")
    do {
        let engine = try GPUPolyCompositionEngine()

        // f(x) = 1 + x + x^2, g(x) = 2 + x
        // h(x) = f(g(x)) = 1 + (2+x) + (2+x)^2 = 1 + 2 + x + 4 + 4x + x^2 = 7 + 5x + x^2
        let one = frFromInt(1)
        let two = frFromInt(2)
        let five = frFromInt(5)
        let seven = frFromInt(7)

        let f = [one, one, one]  // 1 + x + x^2
        let g = [two, one]       // 2 + x

        let h = try engine.compose(f: f, g: g)

        expectEqual(h.count, 3, "compose quadratic: result length")
        expectEqual(h[0], seven, "compose quadratic: h[0] = 7")
        expectEqual(h[1], five, "compose quadratic: h[1] = 5")
        expectEqual(h[2], one, "compose quadratic: h[2] = 1")

        // Verify by evaluation at random points
        var rng = CompRNG(state: 0xC0DE0002)
        for _ in 0..<5 {
            let x = rng.nextFr()
            let gx = cpuHorner(coeffs: g, x: x)
            let fgx = cpuHorner(coeffs: f, x: gx)
            let hx = cpuHorner(coeffs: h, x: x)
            expectEqual(hx, fgx, "compose quadratic: h(x) == f(g(x))")
        }

    } catch { expect(false, "compose quadratic error: \(error)") }
}

// MARK: - compose: f = identity

private func testComposeIdentity() {
    suite("GPUPolyComposition compose identity")
    do {
        let engine = try GPUPolyCompositionEngine()

        // f(x) = x (identity), g(x) = arbitrary
        // h(x) = f(g(x)) = g(x)
        let zero = Fr.zero
        let one = frFromInt(1)

        var rng = CompRNG(state: 0xC0DE0003)
        let g = (0..<5).map { _ in rng.nextFr() }
        let f = [zero, one]  // f(x) = x

        let h = try engine.compose(f: f, g: g)

        // h should equal g (padded with zeros to deg(f)*deg(g) + 1)
        for i in 0..<g.count {
            expectEqual(h[i], g[i], "compose identity: h[\(i)] == g[\(i)]")
        }
        // Remaining coefficients should be zero
        for i in g.count..<h.count {
            expectEqual(h[i], Fr.zero, "compose identity: h[\(i)] == 0")
        }

    } catch { expect(false, "compose identity error: \(error)") }
}

// MARK: - compose: f = constant

private func testComposeConstant() {
    suite("GPUPolyComposition compose constant")
    do {
        let engine = try GPUPolyCompositionEngine()

        // f(x) = 42 (constant), g(x) = anything
        // h(x) = f(g(x)) = 42
        let c = frFromInt(42)

        var rng = CompRNG(state: 0xC0DE0004)
        let g = (0..<4).map { _ in rng.nextFr() }
        let f = [c]

        let h = try engine.compose(f: f, g: g)

        expectEqual(h[0], c, "compose constant: h[0] = 42")
        for i in 1..<h.count {
            expectEqual(h[i], Fr.zero, "compose constant: h[\(i)] = 0")
        }

    } catch { expect(false, "compose constant error: \(error)") }
}

// MARK: - evaluateComposition CPU path

private func testEvaluateCompositionCPU() {
    suite("GPUPolyComposition evaluateComposition CPU")
    do {
        let engine = try GPUPolyCompositionEngine()
        engine.gpuThreshold = Int.max  // Force CPU

        var rng = CompRNG(state: 0xC0DE0005)

        // f(x) = 1 + 2x + 3x^2
        let one = frFromInt(1)
        let two = frFromInt(2)
        let three = frFromInt(3)
        let f = [one, two, three]

        // Generate random g-evaluations
        let numPoints = 32
        let gEvals = (0..<numPoints).map { _ in rng.nextFr() }

        let result = try engine.evaluateComposition(f: f, gEvals: gEvals)
        expectEqual(result.count, numPoints, "evaluateComposition CPU: result count")

        // Verify each: result[i] = f(gEvals[i])
        var allMatch = true
        for i in 0..<numPoints {
            let expected = cpuHorner(coeffs: f, x: gEvals[i])
            if result[i] != expected {
                allMatch = false
                print("  [DETAIL] CPU mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "evaluateComposition CPU: all points match")

    } catch { expect(false, "evaluateComposition CPU error: \(error)") }
}

// MARK: - evaluateComposition GPU path

private func testEvaluateCompositionGPU() {
    suite("GPUPolyComposition evaluateComposition GPU")
    do {
        let engine = try GPUPolyCompositionEngine()
        engine.gpuThreshold = 1  // Force GPU

        var rng = CompRNG(state: 0xC0DE0006)

        let degF = 64
        let f = (0..<degF).map { _ in rng.nextFr() }

        let numPoints = 512
        let gEvals = (0..<numPoints).map { _ in rng.nextFr() }

        let result = try engine.evaluateComposition(f: f, gEvals: gEvals)
        expectEqual(result.count, numPoints, "evaluateComposition GPU: result count")

        // Verify against CPU reference
        var allMatch = true
        for i in 0..<numPoints {
            let expected = cpuHorner(coeffs: f, x: gEvals[i])
            if result[i] != expected {
                allMatch = false
                print("  [DETAIL] GPU mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "evaluateComposition GPU: all \(numPoints) points match CPU")

    } catch { expect(false, "evaluateComposition GPU error: \(error)") }
}

// MARK: - GPU vs CPU consistency

private func testEvaluateCompositionConsistency() {
    suite("GPUPolyComposition GPU vs CPU consistency")
    do {
        let engine = try GPUPolyCompositionEngine()
        var rng = CompRNG(state: 0xC0DE0007)

        let degF = 128
        let f = (0..<degF).map { _ in rng.nextFr() }
        let numPoints = 256
        let gEvals = (0..<numPoints).map { _ in rng.nextFr() }

        engine.gpuThreshold = 1
        let gpuResult = try engine.evaluateComposition(f: f, gEvals: gEvals)

        engine.gpuThreshold = Int.max
        let cpuResult = try engine.evaluateComposition(f: f, gEvals: gEvals)

        expectEqual(gpuResult.count, cpuResult.count, "GPU/CPU result count match")

        var allMatch = true
        for i in 0..<numPoints {
            if gpuResult[i] != cpuResult[i] {
                allMatch = false
                print("  [DETAIL] GPU/CPU mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "evaluateComposition GPU and CPU produce identical results")

    } catch { expect(false, "GPU/CPU consistency error: \(error)") }
}

// MARK: - deepComposition

private func testDeepComposition() {
    suite("GPUPolyComposition deepComposition")
    do {
        let engine = try GPUPolyCompositionEngine()
        var rng = CompRNG(state: 0xC0DE0008)

        let K = 3  // trace columns
        let N = 64 // domain size

        // Generate random trace columns and constraint poly
        var trace = [[Fr]]()
        for _ in 0..<K {
            trace.append((0..<N).map { _ in rng.nextFr() })
        }
        let constraintPoly = (0..<N).map { _ in rng.nextFr() }
        let alpha = rng.nextFr()
        let z = rng.nextFr()

        let result = try engine.deepComposition(trace: trace, constraintPoly: constraintPoly,
                                                 alpha: alpha, z: z)
        expectEqual(result.count, N, "deepComposition: result length = N")

        // Verify manually: result[j] = sum_i alpha^i * trace_i[j] + alpha^K * C[j]
        var alphaPow = Fr.one
        for j in 0..<N {
            var expected = Fr.zero
            var ap = Fr.one
            for i in 0..<K {
                expected = frAdd(expected, frMul(ap, trace[i][j]))
                ap = frMul(ap, alpha)
            }
            expected = frAdd(expected, frMul(ap, constraintPoly[j]))
            expectEqual(result[j], expected, "deepComposition: point \(j)")
        }

    } catch { expect(false, "deepComposition error: \(error)") }
}

// MARK: - deepComposition single column

private func testDeepCompositionSingleColumn() {
    suite("GPUPolyComposition deepComposition single column")
    do {
        let engine = try GPUPolyCompositionEngine()
        var rng = CompRNG(state: 0xC0DE0009)

        let N = 32
        let trace = [(0..<N).map { _ in rng.nextFr() }]
        let constraintPoly = (0..<N).map { _ in rng.nextFr() }
        let alpha = rng.nextFr()
        let z = rng.nextFr()

        let result = try engine.deepComposition(trace: trace, constraintPoly: constraintPoly,
                                                 alpha: alpha, z: z)
        expectEqual(result.count, N, "single column: result length")

        // result[j] = trace[0][j] + alpha * C[j]
        for j in 0..<N {
            let expected = frAdd(trace[0][j], frMul(alpha, constraintPoly[j]))
            expectEqual(result[j], expected, "single column: point \(j)")
        }

    } catch { expect(false, "single column deepComposition error: \(error)") }
}

// MARK: - compose random small polynomials, verify by evaluation

private func testComposeRandomSmall() {
    suite("GPUPolyComposition compose random small")
    do {
        let engine = try GPUPolyCompositionEngine()
        var rng = CompRNG(state: 0xC0DE000A)

        // Test several random compositions with small degrees
        let testCases: [(Int, Int)] = [(2, 2), (3, 2), (2, 3), (4, 3), (3, 4), (5, 2)]

        for (degF, degG) in testCases {
            let f = (0...degF).map { _ in rng.nextFr() }
            let g = (0...degG).map { _ in rng.nextFr() }

            let h = try engine.compose(f: f, g: g)

            // Verify at 10 random points
            var allMatch = true
            for _ in 0..<10 {
                let x = rng.nextFr()
                let gx = cpuHorner(coeffs: g, x: x)
                let fgx = cpuHorner(coeffs: f, x: gx)
                let hx = cpuHorner(coeffs: h, x: x)
                if hx != fgx {
                    allMatch = false
                    break
                }
            }
            expect(allMatch, "compose random (deg \(degF), \(degG)): h(x) == f(g(x)) at all test points")
        }

    } catch { expect(false, "compose random error: \(error)") }
}

// MARK: - Performance benchmark

private func testPerformance() {
    suite("GPUPolyComposition performance")
    do {
        let engine = try GPUPolyCompositionEngine()
        engine.gpuThreshold = 1
        var rng = CompRNG(state: 0xC0DE000B)

        // Benchmark evaluateComposition: deg-256 f, 8192 g-evaluation points
        let degF = 256
        let numPoints = 8192

        let f = (0..<degF).map { _ in rng.nextFr() }
        let gEvals = (0..<numPoints).map { _ in rng.nextFr() }

        // Warm up
        _ = try engine.evaluateComposition(f: f, gEvals: gEvals)

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try engine.evaluateComposition(f: f, gEvals: gEvals)
        let gpuTime = CFAbsoluteTimeGetCurrent() - t0
        expectEqual(result.count, numPoints, "Performance: result count")
        print(String(format: "  GPU evaluateComposition deg=%d pts=%d: %.2fms", degF, numPoints, gpuTime * 1000))

        // Benchmark compose: deg-8 f, deg-8 g (result deg 64)
        let compF = (0..<9).map { _ in rng.nextFr() }
        let compG = (0..<9).map { _ in rng.nextFr() }

        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = try engine.compose(f: compF, g: compG)
        }
        let compTime = (CFAbsoluteTimeGetCurrent() - t1) / 100.0
        print(String(format: "  CPU compose deg=8 x deg=8: %.2fms", compTime * 1000))

        // Benchmark deepComposition: 4 columns, 4096 points
        let K = 4
        let N = 4096
        var trace = [[Fr]]()
        for _ in 0..<K { trace.append((0..<N).map { _ in rng.nextFr() }) }
        let constraintPoly = (0..<N).map { _ in rng.nextFr() }
        let alpha = rng.nextFr()
        let z = rng.nextFr()

        let t2 = CFAbsoluteTimeGetCurrent()
        _ = try engine.deepComposition(trace: trace, constraintPoly: constraintPoly,
                                        alpha: alpha, z: z)
        let deepTime = CFAbsoluteTimeGetCurrent() - t2
        print(String(format: "  CPU deepComposition K=%d N=%d: %.2fms", K, N, deepTime * 1000))

    } catch { expect(false, "Performance benchmark error: \(error)") }
}
