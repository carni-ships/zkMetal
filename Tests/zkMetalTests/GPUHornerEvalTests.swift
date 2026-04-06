import zkMetal
import Foundation

// MARK: - Test helpers

/// Simple LCG for test reproducibility
private struct HornerRNG {
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

// MARK: - Public test entry point

public func runGPUHornerEvalTests() {
    testEvaluateSingleCPU()
    testEvaluateMultiplePointsGPU()
    testEvaluateGPUvsCPUConsistency()
    testBatchEvaluateGPU()
    testCachedKernelLargePoly()
    testFallbackForLargePolys()
    testEdgeCases()
    testHornerPerformance()
}

// MARK: - evaluateSingle CPU correctness

private func testEvaluateSingleCPU() {
    suite("GPUHornerEval evaluateSingle CPU")
    do {
        let engine = try GPUHornerEval()
        var rng = HornerRNG(state: 0xDEAD0001)

        // Known polynomial: p(x) = 1 + 2x + 3x^2 in Montgomery form
        // Use Fr.one for coefficient 1, etc.
        let one = Fr.from64([1, 0, 0, 0])  // toMontgomery(1)
        let two = Fr.from64([2, 0, 0, 0])
        let three = Fr.from64([3, 0, 0, 0])
        let coeffs = [one, two, three]

        // Evaluate at x=0: should give a_0 = 1
        let r0 = engine.evaluateSingle(coeffs: coeffs, point: Fr.zero)
        expectEqual(r0, one, "p(0) = 1")

        // Evaluate at x=1 (Montgomery form of 1): p(1) = 1+2+3 = 6
        let oneM = Fr.from64([1, 0, 0, 0])
        let r1 = engine.evaluateSingle(coeffs: coeffs, point: oneM)
        let six = Fr.from64([6, 0, 0, 0])
        expectEqual(r1, six, "p(1) = 6")

        // Random polynomial, random point -- compare against CPU reference
        let degree = 64
        var randomCoeffs = [Fr]()
        for _ in 0..<degree { randomCoeffs.append(rng.nextFr()) }
        let x = rng.nextFr()

        let result = engine.evaluateSingle(coeffs: randomCoeffs, point: x)
        let expected = cpuHorner(coeffs: randomCoeffs, x: x)
        expectEqual(result, expected, "Random poly deg=64 evaluateSingle matches CPU")

    } catch { expect(false, "evaluateSingle error: \(error)") }
}

// MARK: - evaluate (GPU path) correctness

private func testEvaluateMultiplePointsGPU() {
    suite("GPUHornerEval evaluate GPU multi-point")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1  // Force GPU path
        var rng = HornerRNG(state: 0xBEEF0002)

        let degree = 128
        var coeffs = [Fr]()
        for _ in 0..<degree { coeffs.append(rng.nextFr()) }

        let numPoints = 256
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        let results = try engine.evaluate(coeffs: coeffs, points: points)
        expectEqual(results.count, numPoints, "GPU evaluate result count = \(numPoints)")

        var allMatch = true
        for i in 0..<numPoints {
            let expected = cpuHorner(coeffs: coeffs, x: points[i])
            if results[i] != expected {
                allMatch = false
                print("  [DETAIL] GPU mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "GPU evaluate all \(numPoints) points match CPU reference")

    } catch { expect(false, "GPU evaluate error: \(error)") }
}

// MARK: - GPU vs CPU consistency (same engine, different thresholds)

private func testEvaluateGPUvsCPUConsistency() {
    suite("GPUHornerEval GPU vs CPU consistency")
    do {
        let engine = try GPUHornerEval()
        var rng = HornerRNG(state: 0xCAFE0003)

        let degree = 32
        var coeffs = [Fr]()
        for _ in 0..<degree { coeffs.append(rng.nextFr()) }

        let numPoints = 64
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        // Force GPU path
        engine.gpuWorkThreshold = 1
        let gpuResults = try engine.evaluate(coeffs: coeffs, points: points)

        // Force CPU path
        engine.gpuWorkThreshold = Int.max
        let cpuResults = try engine.evaluate(coeffs: coeffs, points: points)

        expectEqual(gpuResults.count, cpuResults.count, "GPU/CPU result count match")

        var allMatch = true
        for i in 0..<numPoints {
            if gpuResults[i] != cpuResults[i] {
                allMatch = false
                print("  [DETAIL] GPU/CPU mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "GPU and CPU paths produce identical results for all \(numPoints) points")

    } catch { expect(false, "GPU/CPU consistency error: \(error)") }
}

// MARK: - batchEvaluate

private func testBatchEvaluateGPU() {
    suite("GPUHornerEval batchEvaluate")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1
        var rng = HornerRNG(state: 0xF00D0004)

        let numPolys = 128
        let degree = 64
        var polys = [[Fr]]()
        for _ in 0..<numPolys {
            var coeffs = [Fr]()
            for _ in 0..<degree { coeffs.append(rng.nextFr()) }
            polys.append(coeffs)
        }

        let point = rng.nextFr()

        let results = try engine.batchEvaluate(polys: polys, point: point)
        expectEqual(results.count, numPolys, "batchEvaluate result count = \(numPolys)")

        var allMatch = true
        for m in 0..<numPolys {
            let expected = cpuHorner(coeffs: polys[m], x: point)
            if results[m] != expected {
                allMatch = false
                print("  [DETAIL] Batch mismatch at poly \(m)")
                break
            }
        }
        expect(allMatch, "batchEvaluate all \(numPolys) polys match CPU reference")

    } catch { expect(false, "batchEvaluate error: \(error)") }
}

// MARK: - Cached kernel for medium polynomials

private func testCachedKernelLargePoly() {
    suite("GPUHornerEval cached kernel (deg <= 512)")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1
        var rng = HornerRNG(state: 0xA1B20005)

        // degree 512 should use the cached threadgroup kernel
        let degree = 512
        var coeffs = [Fr]()
        for _ in 0..<degree { coeffs.append(rng.nextFr()) }

        let numPoints = 128
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        let results = try engine.evaluate(coeffs: coeffs, points: points)
        expectEqual(results.count, numPoints, "Cached kernel result count")

        // Spot-check 16 points
        var allMatch = true
        let step = numPoints / 16
        for i in stride(from: 0, to: numPoints, by: step) {
            let expected = cpuHorner(coeffs: coeffs, x: points[i])
            if results[i] != expected {
                allMatch = false
                print("  [DETAIL] Cached kernel mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "Cached kernel deg=512 spot-check matches CPU")

    } catch { expect(false, "Cached kernel error: \(error)") }
}

// MARK: - Fallback for polynomials > 512 degree (uncached kernel)

private func testFallbackForLargePolys() {
    suite("GPUHornerEval large poly (deg > 512, uncached)")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1
        var rng = HornerRNG(state: 0x77880006)

        // degree 1024 should use the uncached (device memory) kernel
        let degree = 1024
        var coeffs = [Fr]()
        for _ in 0..<degree { coeffs.append(rng.nextFr()) }

        let numPoints = 64
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        let results = try engine.evaluate(coeffs: coeffs, points: points)
        expectEqual(results.count, numPoints, "Large poly result count")

        var allMatch = true
        for i in 0..<numPoints {
            let expected = cpuHorner(coeffs: coeffs, x: points[i])
            if results[i] != expected {
                allMatch = false
                print("  [DETAIL] Large poly mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "Large poly deg=1024 all \(numPoints) points match CPU")

    } catch { expect(false, "Large poly error: \(error)") }
}

// MARK: - Edge cases

private func testEdgeCases() {
    suite("GPUHornerEval edge cases")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1
        var rng = HornerRNG(state: 0x99AA0007)

        // Degree 1 (constant polynomial)
        let c = rng.nextFr()
        let x = rng.nextFr()
        let r1 = try engine.evaluate(coeffs: [c], points: [x])
        expectEqual(r1.count, 1, "Constant poly result count")
        expectEqual(r1[0], c, "Constant poly p(x) = c for any x")

        // Degree 2 (linear): p(x) = a + b*x
        let a = rng.nextFr()
        let b = rng.nextFr()
        let r2 = engine.evaluateSingle(coeffs: [a, b], point: Fr.zero)
        expectEqual(r2, a, "Linear poly p(0) = a")

        // Single point via evaluate
        let coeffs3 = (0..<16).map { _ in rng.nextFr() }
        let pt = rng.nextFr()
        let r3 = try engine.evaluate(coeffs: coeffs3, points: [pt])
        let expected3 = cpuHorner(coeffs: coeffs3, x: pt)
        expectEqual(r3[0], expected3, "Single-point evaluate matches CPU")

    } catch { expect(false, "Edge case error: \(error)") }
}

// MARK: - Performance benchmark

private func testHornerPerformance() {
    suite("GPUHornerEval performance")
    do {
        let engine = try GPUHornerEval()
        engine.gpuWorkThreshold = 1
        var rng = HornerRNG(state: 0xBB110008)

        // Benchmark: degree 4096, 16384 points
        let degree = 1 << 12
        let numPoints = 1 << 14

        var coeffs = [Fr]()
        coeffs.reserveCapacity(degree)
        for _ in 0..<degree { coeffs.append(rng.nextFr()) }

        var points = [Fr]()
        points.reserveCapacity(numPoints)
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        // Warm up
        _ = try engine.evaluate(coeffs: coeffs, points: points)

        // Timed GPU evaluation
        let t0 = CFAbsoluteTimeGetCurrent()
        let results = try engine.evaluate(coeffs: coeffs, points: points)
        let gpuTime = CFAbsoluteTimeGetCurrent() - t0
        expectEqual(results.count, numPoints, "Performance result count")
        print(String(format: "  GPU Horner deg=%d pts=%d: %.2fms", degree, numPoints, gpuTime * 1000))

        // Timed CPU evaluation (smaller problem for reasonable time)
        let cpuDeg = 256
        let cpuPts = 256
        var cpuCoeffs = [Fr]()
        for _ in 0..<cpuDeg { cpuCoeffs.append(rng.nextFr()) }
        var cpuPoints = [Fr]()
        for _ in 0..<cpuPts { cpuPoints.append(rng.nextFr()) }

        let t1 = CFAbsoluteTimeGetCurrent()
        for pt in cpuPoints {
            _ = engine.evaluateSingle(coeffs: cpuCoeffs, point: pt)
        }
        let cpuTime = CFAbsoluteTimeGetCurrent() - t1
        print(String(format: "  CPU Horner deg=%d pts=%d: %.2fms", cpuDeg, cpuPts, cpuTime * 1000))

        // Batch benchmark: 1024 polys, degree 128
        let batchPolys = 1024
        let batchDeg = 128
        var polys = [[Fr]]()
        for _ in 0..<batchPolys {
            var c = [Fr]()
            for _ in 0..<batchDeg { c.append(rng.nextFr()) }
            polys.append(c)
        }
        let batchPt = rng.nextFr()
        _ = try engine.batchEvaluate(polys: polys, point: batchPt) // warm up

        let t2 = CFAbsoluteTimeGetCurrent()
        let batchResults = try engine.batchEvaluate(polys: polys, point: batchPt)
        let batchTime = CFAbsoluteTimeGetCurrent() - t2
        expectEqual(batchResults.count, batchPolys, "Batch perf result count")
        print(String(format: "  GPU Batch %d polys deg=%d: %.2fms", batchPolys, batchDeg, batchTime * 1000))

    } catch { expect(false, "Performance benchmark error: \(error)") }
}
