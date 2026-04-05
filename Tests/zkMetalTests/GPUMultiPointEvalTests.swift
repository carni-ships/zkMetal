import zkMetal
import Foundation

// MARK: - Test helpers

/// CPU Horner evaluation for BabyBear (reference implementation)
private func cpuHornerBb(coeffs: [UInt32], x: UInt32) -> UInt32 {
    let p = UInt64(Bb.P)
    var result = UInt64(coeffs.last!)
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = (result * UInt64(x) + UInt64(coeffs[i])) % p
    }
    return UInt32(result)
}

/// CPU Horner evaluation for BN254 Fr (reference)
private func cpuHornerFr(coeffs: [Fr], x: Fr) -> Fr {
    var result = coeffs.last!
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}

/// Simple LCG for test reproducibility
private struct MPERNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextBb() -> UInt32 {
        return next32() % Bb.P
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

private func frToU32(_ f: Fr) -> [UInt32] {
    [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
}

private func u32ToFr(_ arr: [UInt32], _ base: Int) -> Fr {
    Fr(v: (arr[base], arr[base+1], arr[base+2], arr[base+3],
            arr[base+4], arr[base+5], arr[base+6], arr[base+7]))
}

// MARK: - Public test entry point

public func runGPUMultiPointEvalTests() {
    testSinglePolyMultiplePointsBabyBear()
    testSinglePolyMultiplePointsBN254()
    testBatchEvalMultiplePolys()
    testBatchEvalBN254()
    testCrossEvaluationMatrix()
    testCrossEvaluationBN254()
    testLargeEvaluation()
    testPerformanceBenchmark()
}

// MARK: - Single poly at multiple points

private func testSinglePolyMultiplePointsBabyBear() {
    suite("GPUMultiPointEval single poly BabyBear")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1 // Force GPU path even for small inputs
        var rng = MPERNG(state: 0xBB010001)

        // Known polynomial: p(x) = 1 + 2x + 3x^2 + 4x^3
        let coeffs: [UInt32] = [1, 2, 3, 4]

        // Generate 128 random points (well above any threshold)
        var points = [UInt32]()
        for _ in 0..<128 { points.append(rng.nextBb()) }

        let results = try engine.evaluate(poly: coeffs, points: points, field: .babybear)
        expectEqual(results.count, 128, "BabyBear single eval result count")

        // Verify all against CPU reference
        var allMatch = true
        for i in 0..<128 {
            let expected = cpuHornerBb(coeffs: coeffs, x: points[i])
            if results[i] != expected {
                allMatch = false
                print("  [DETAIL] Mismatch at i=\(i): got \(results[i]), expected \(expected)")
                break
            }
        }
        expect(allMatch, "BabyBear single poly all 128 points match CPU")

        // Also test CPU fallback path
        engine.gpuThreshold = 1000
        let cpuResults = try engine.evaluate(poly: coeffs, points: points, field: .babybear)
        var cpuMatch = true
        for i in 0..<128 {
            if results[i] != cpuResults[i] { cpuMatch = false; break }
        }
        expect(cpuMatch, "GPU and CPU paths produce identical results")

    } catch { expect(false, "BabyBear single poly error: \(error)") }
}

private func testSinglePolyMultiplePointsBN254() {
    suite("GPUMultiPointEval single poly BN254")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1
        var rng = MPERNG(state: 0x12345678)

        let degree = 32
        var coeffsFr = [Fr]()
        for _ in 0..<degree { coeffsFr.append(rng.nextFr()) }

        var coeffsU32 = [UInt32]()
        for c in coeffsFr { coeffsU32.append(contentsOf: frToU32(c)) }

        let numPoints = 128
        var pointsFr = [Fr]()
        var pointsU32 = [UInt32]()
        for _ in 0..<numPoints {
            let p = rng.nextFr()
            pointsFr.append(p)
            pointsU32.append(contentsOf: frToU32(p))
        }

        let results = try engine.evaluate(poly: coeffsU32, points: pointsU32, field: .bn254)
        expectEqual(results.count, numPoints * 8, "BN254 single eval result count")

        var allMatch = true
        for i in 0..<numPoints {
            let expected = cpuHornerFr(coeffs: coeffsFr, x: pointsFr[i])
            let got = u32ToFr(results, i * 8)
            if got != expected {
                allMatch = false
                print("  [DETAIL] BN254 mismatch at i=\(i)")
                break
            }
        }
        expect(allMatch, "BN254 single poly all \(numPoints) points match CPU")

    } catch { expect(false, "BN254 single poly error: \(error)") }
}

// MARK: - Batch evaluate multiple polys at one point

private func testBatchEvalMultiplePolys() {
    suite("GPUMultiPointEval batch BabyBear")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1
        var rng = MPERNG(state: 0xBA7C4BB0)

        let numPolys = 256
        let degree = 64

        var polys = [[UInt32]]()
        for _ in 0..<numPolys {
            var coeffs = [UInt32]()
            for _ in 0..<degree { coeffs.append(rng.nextBb()) }
            polys.append(coeffs)
        }

        let point: [UInt32] = [rng.nextBb()]

        let results = try engine.batchEvaluate(polys: polys, point: point, field: .babybear)
        expectEqual(results.count, numPolys, "BabyBear batch eval result count")

        var allMatch = true
        for m in 0..<numPolys {
            let expected = cpuHornerBb(coeffs: polys[m], x: point[0])
            if results[m] != expected {
                allMatch = false
                print("  [DETAIL] Batch mismatch at poly \(m): got \(results[m]), expected \(expected)")
                break
            }
        }
        expect(allMatch, "BabyBear batch eval all \(numPolys) polys match CPU")

    } catch { expect(false, "BabyBear batch eval error: \(error)") }
}

private func testBatchEvalBN254() {
    suite("GPUMultiPointEval batch BN254")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1
        var rng = MPERNG(state: 0xBA7C4F50)

        let numPolys = 128
        let degree = 16

        var polysFr = [[Fr]]()
        var polysU32 = [[UInt32]]()
        for _ in 0..<numPolys {
            var fr = [Fr]()
            var u32 = [UInt32]()
            for _ in 0..<degree {
                let c = rng.nextFr()
                fr.append(c)
                u32.append(contentsOf: frToU32(c))
            }
            polysFr.append(fr)
            polysU32.append(u32)
        }

        let pointFr = rng.nextFr()
        let pointU32 = frToU32(pointFr)

        let results = try engine.batchEvaluate(polys: polysU32, point: pointU32, field: .bn254)
        expectEqual(results.count, numPolys * 8, "BN254 batch eval result count")

        var allMatch = true
        for m in 0..<numPolys {
            let expected = cpuHornerFr(coeffs: polysFr[m], x: pointFr)
            let got = u32ToFr(results, m * 8)
            if got != expected {
                allMatch = false
                print("  [DETAIL] BN254 batch mismatch at poly \(m)")
                break
            }
        }
        expect(allMatch, "BN254 batch eval all \(numPolys) polys match CPU")

    } catch { expect(false, "BN254 batch eval error: \(error)") }
}

// MARK: - Cross-evaluation matrix

private func testCrossEvaluationMatrix() {
    suite("GPUMultiPointEval cross BabyBear")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1
        var rng = MPERNG(state: 0xC505BB01)

        let numPolys = 8
        let numPoints = 128
        let degree = 32

        var polys = [[UInt32]]()
        for _ in 0..<numPolys {
            var coeffs = [UInt32]()
            for _ in 0..<degree { coeffs.append(rng.nextBb()) }
            polys.append(coeffs)
        }

        var points = [UInt32]()
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        let matrix = try engine.crossEvaluate(polys: polys, points: points, field: .babybear)
        expectEqual(matrix.count, numPolys, "Cross matrix row count")

        var allMatch = true
        for m in 0..<numPolys {
            expectEqual(matrix[m].count, numPoints, "Cross matrix row[\(m)] col count")
            for p in 0..<numPoints {
                let expected = cpuHornerBb(coeffs: polys[m], x: points[p])
                if matrix[m][p] != expected {
                    allMatch = false
                    print("  [DETAIL] Cross mismatch at (\(m),\(p)): got \(matrix[m][p]), expected \(expected)")
                    break
                }
            }
            if !allMatch { break }
        }
        expect(allMatch, "BabyBear cross eval \(numPolys)x\(numPoints) matrix matches CPU")

    } catch { expect(false, "BabyBear cross eval error: \(error)") }
}

private func testCrossEvaluationBN254() {
    suite("GPUMultiPointEval cross BN254")
    do {
        let engine = try GPUMultiPointEval()
        engine.gpuThreshold = 1
        var rng = MPERNG(state: 0xC505F501)

        let numPolys = 4
        let numPoints = 64
        let degree = 16

        var polysFr = [[Fr]]()
        var polysU32 = [[UInt32]]()
        for _ in 0..<numPolys {
            var fr = [Fr]()
            var u32 = [UInt32]()
            for _ in 0..<degree {
                let c = rng.nextFr()
                fr.append(c)
                u32.append(contentsOf: frToU32(c))
            }
            polysFr.append(fr)
            polysU32.append(u32)
        }

        var pointsFr = [Fr]()
        var pointsU32 = [UInt32]()
        for _ in 0..<numPoints {
            let p = rng.nextFr()
            pointsFr.append(p)
            pointsU32.append(contentsOf: frToU32(p))
        }

        let matrix = try engine.crossEvaluate(polys: polysU32, points: pointsU32, field: .bn254)
        expectEqual(matrix.count, numPolys, "BN254 cross matrix row count")

        var allMatch = true
        for m in 0..<numPolys {
            expectEqual(matrix[m].count, numPoints * 8, "BN254 cross row[\(m)] size")
            for p in 0..<numPoints {
                let expected = cpuHornerFr(coeffs: polysFr[m], x: pointsFr[p])
                let got = u32ToFr(matrix[m], p * 8)
                if got != expected {
                    allMatch = false
                    print("  [DETAIL] BN254 cross mismatch at (\(m),\(p))")
                    break
                }
            }
            if !allMatch { break }
        }
        expect(allMatch, "BN254 cross eval \(numPolys)x\(numPoints) matrix matches CPU")

    } catch { expect(false, "BN254 cross eval error: \(error)") }
}

// MARK: - Large evaluation (2^16 points)

private func testLargeEvaluation() {
    suite("GPUMultiPointEval large 2^16 BabyBear")
    do {
        let engine = try GPUMultiPointEval()
        var rng = MPERNG(state: 0x1A56E001)

        let degree = 256
        let numPoints = 1 << 16  // 65536

        var coeffs = [UInt32]()
        coeffs.reserveCapacity(degree)
        for _ in 0..<degree { coeffs.append(rng.nextBb()) }

        var points = [UInt32]()
        points.reserveCapacity(numPoints)
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        let results = try engine.evaluate(poly: coeffs, points: points, field: .babybear)
        expectEqual(results.count, numPoints, "Large eval result count = 65536")

        // Spot-check 16 points evenly spaced
        var allMatch = true
        let step = numPoints / 16
        for i in stride(from: 0, to: numPoints, by: step) {
            let expected = cpuHornerBb(coeffs: coeffs, x: points[i])
            if results[i] != expected {
                allMatch = false
                print("  [DETAIL] Large eval mismatch at i=\(i): got \(results[i]), expected \(expected)")
                break
            }
        }
        expect(allMatch, "Large eval 2^16 points spot-check matches CPU")

    } catch { expect(false, "Large eval error: \(error)") }
}

// MARK: - Performance benchmark

private func testPerformanceBenchmark() {
    suite("GPUMultiPointEval performance")
    do {
        let engine = try GPUMultiPointEval()
        var rng = MPERNG(state: 0x9E5F0001)

        // Horner: degree 2^12 poly at 2^14 points
        let degree = 1 << 12
        let numPoints = 1 << 14

        var coeffs = [UInt32]()
        coeffs.reserveCapacity(degree)
        for _ in 0..<degree { coeffs.append(rng.nextBb()) }

        var points = [UInt32]()
        points.reserveCapacity(numPoints)
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        // Warm up
        _ = try engine.evaluate(poly: coeffs, points: points, field: .babybear)

        // Timed Horner
        let t0 = CFAbsoluteTimeGetCurrent()
        let r1 = try engine.evaluate(poly: coeffs, points: points, field: .babybear)
        let hornerTime = CFAbsoluteTimeGetCurrent() - t0
        expectEqual(r1.count, numPoints, "Horner perf result count")
        print(String(format: "  Horner deg=%d pts=%d: %.2fms", degree, numPoints, hornerTime * 1000))

        // Cross: 16 polys x 1024 points, degree 256
        let crossPolys = 16
        let crossPoints = 1024
        let crossDeg = 256

        var polys = [[UInt32]]()
        for _ in 0..<crossPolys {
            var c = [UInt32]()
            for _ in 0..<crossDeg { c.append(rng.nextBb()) }
            polys.append(c)
        }
        var cpts = [UInt32]()
        for _ in 0..<crossPoints { cpts.append(rng.nextBb()) }

        // Warm up
        _ = try engine.crossEvaluate(polys: polys, points: cpts, field: .babybear)

        let t1 = CFAbsoluteTimeGetCurrent()
        let r2 = try engine.crossEvaluate(polys: polys, points: cpts, field: .babybear)
        let crossTime = CFAbsoluteTimeGetCurrent() - t1
        expectEqual(r2.count, crossPolys, "Cross perf result count")
        print(String(format: "  Cross %dx%d deg=%d: %.2fms", crossPolys, crossPoints, crossDeg, crossTime * 1000))

        // Batch: 4096 polys at 1 point, degree 64
        let batchPolys = 4096
        let batchDeg = 64
        var bpolys = [[UInt32]]()
        for _ in 0..<batchPolys {
            var c = [UInt32]()
            for _ in 0..<batchDeg { c.append(rng.nextBb()) }
            bpolys.append(c)
        }
        let bpt: [UInt32] = [rng.nextBb()]

        _ = try engine.batchEvaluate(polys: bpolys, point: bpt, field: .babybear)

        let t2 = CFAbsoluteTimeGetCurrent()
        let r3 = try engine.batchEvaluate(polys: bpolys, point: bpt, field: .babybear)
        let batchTime = CFAbsoluteTimeGetCurrent() - t2
        expectEqual(r3.count, batchPolys, "Batch perf result count")
        print(String(format: "  Batch %d polys deg=%d: %.2fms", batchPolys, batchDeg, batchTime * 1000))

    } catch { expect(false, "Performance benchmark error: \(error)") }
}
