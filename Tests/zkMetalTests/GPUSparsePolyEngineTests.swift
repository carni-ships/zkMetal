import zkMetal
import Foundation

// MARK: - Test helpers

private struct SparsePolyRNG {
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

    mutating func nextIndex(bound: Int) -> Int {
        return Int(next32()) % bound
    }
}

/// CPU reference: evaluate sparse poly at a point using incremental powers
private func cpuSparseEval(_ poly: SparsePoly, at z: Fr) -> Fr {
    return poly.evaluate(at: z)
}

/// CPU reference: multiply sparse by dense
private func cpuMulDense(_ sparse: SparsePoly, _ dense: [Fr]) -> [Fr] {
    guard sparse.nnz > 0 && !dense.isEmpty else { return [] }
    let maxIdx = sparse.terms.last!.index
    let outLen = maxIdx + dense.count
    var result = [Fr](repeating: Fr.zero, count: outLen)
    for (sIdx, sCoeff) in sparse.terms {
        for (dIdx, dCoeff) in dense.enumerated() {
            let k = sIdx + dIdx
            if k < outLen {
                result[k] = frAdd(result[k], frMul(sCoeff, dCoeff))
            }
        }
    }
    return result
}

// MARK: - Public test entry point

public func runGPUSparsePolyEngineTests() {
    testSparseEvalSingleCPU()
    testSparseEvalMultiGPU()
    testSparseToDense()
    testSparseMulDenseGPU()
    testSparseEdgeCases()
    testSparsePerformance()
}

// MARK: - evaluate (CPU single point)

private func testSparseEvalSingleCPU() {
    suite("GPUSparsePolyEngine evaluate (CPU)")
    do {
        let engine = try GPUSparsePolyEngine()
        var rng = SparsePolyRNG(state: 0x5BAE5E01)

        // Known polynomial: p(x) = 3 + 5*x^3 + 2*x^7
        let three = frFromInt(3)
        let five  = frFromInt(5)
        let two   = frFromInt(2)

        let poly = SparsePoly(terms: [
            (index: 0, coeff: three),
            (index: 3, coeff: five),
            (index: 7, coeff: two)
        ], degreeBound: 8)

        // p(0) = 3
        let r0 = engine.evaluate(poly: poly, point: Fr.zero)
        expectEqual(r0, three, "p(0) = 3")

        // p(1) = 3 + 5 + 2 = 10
        let r1 = engine.evaluate(poly: poly, point: Fr.one)
        let ten = frFromInt(10)
        expectEqual(r1, ten, "p(1) = 10")

        // Random sparse poly, random point
        let degree = 1024
        let nnz = 20
        var terms = [(index: Int, coeff: Fr)]()
        var usedIndices = Set<Int>()
        for _ in 0..<nnz {
            var idx: Int
            repeat { idx = rng.nextIndex(bound: degree) } while usedIndices.contains(idx)
            usedIndices.insert(idx)
            terms.append((index: idx, coeff: rng.nextFr()))
        }
        let randomPoly = SparsePoly(terms: terms, degreeBound: degree)
        let x = rng.nextFr()

        let result = engine.evaluate(poly: randomPoly, point: x)
        let expected = cpuSparseEval(randomPoly, at: x)
        expectEqual(result, expected, "Random sparse eval matches CPU reference")

    } catch { expect(false, "evaluate error: \(error)") }
}

// MARK: - evaluateMulti (GPU)

private func testSparseEvalMultiGPU() {
    suite("GPUSparsePolyEngine evaluateMulti (GPU)")
    do {
        let engine = try GPUSparsePolyEngine()
        engine.gpuWorkThreshold = 1  // Force GPU path
        var rng = SparsePolyRNG(state: 0x5BAE5E02)

        // Build a sparse polynomial with known structure
        let degree = 512
        let nnz = 15
        var terms = [(index: Int, coeff: Fr)]()
        var usedIndices = Set<Int>()
        for _ in 0..<nnz {
            var idx: Int
            repeat { idx = rng.nextIndex(bound: degree) } while usedIndices.contains(idx)
            usedIndices.insert(idx)
            terms.append((index: idx, coeff: rng.nextFr()))
        }
        let poly = SparsePoly(terms: terms, degreeBound: degree)

        // Generate random evaluation points
        let numPoints = 64
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        // GPU evaluation
        let gpuResults = try engine.evaluateMulti(poly: poly, points: points)

        // CPU reference
        let cpuResults = points.map { cpuSparseEval(poly, at: $0) }

        expect(gpuResults.count == numPoints, "GPU result count matches")
        var allMatch = true
        for i in 0..<numPoints {
            if gpuResults[i] != cpuResults[i] {
                allMatch = false
                expect(false, "Mismatch at point \(i)")
                break
            }
        }
        if allMatch {
            expect(true, "All \(numPoints) GPU evaluations match CPU")
        }

        // Test with more points to exercise cached kernel
        let manyPoints = 256
        var pts2 = [Fr]()
        for _ in 0..<manyPoints { pts2.append(rng.nextFr()) }
        let gpuRes2 = try engine.evaluateMulti(poly: poly, points: pts2)
        let cpuRes2 = pts2.map { cpuSparseEval(poly, at: $0) }
        var match2 = true
        for i in 0..<manyPoints {
            if gpuRes2[i] != cpuRes2[i] { match2 = false; break }
        }
        expect(match2, "256-point GPU sparse eval matches CPU")

    } catch { expect(false, "evaluateMulti error: \(error)") }
}

// MARK: - toDense

private func testSparseToDense() {
    suite("GPUSparsePolyEngine toDense")
    do {
        let engine = try GPUSparsePolyEngine()

        let one = frFromInt(1)
        let two = frFromInt(2)
        let three = frFromInt(3)

        let poly = SparsePoly(terms: [
            (index: 0, coeff: one),
            (index: 2, coeff: two),
            (index: 5, coeff: three)
        ], degreeBound: 8)

        // Default degree (uses degreeBound)
        let dense = engine.toDense(poly: poly)
        expectEqual(dense.count, 8, "Dense has 8 coefficients")
        expectEqual(dense[0], one, "dense[0] = 1")
        expectEqual(dense[1], Fr.zero, "dense[1] = 0")
        expectEqual(dense[2], two, "dense[2] = 2")
        expectEqual(dense[3], Fr.zero, "dense[3] = 0")
        expectEqual(dense[4], Fr.zero, "dense[4] = 0")
        expectEqual(dense[5], three, "dense[5] = 3")

        // Custom degree
        let dense4 = engine.toDense(poly: poly, degree: 4)
        expectEqual(dense4.count, 4, "Dense with degree=4")
        expectEqual(dense4[0], one, "dense4[0] = 1")
        expectEqual(dense4[2], two, "dense4[2] = 2")

        // Round-trip: from dense, back to sparse, back to dense
        let dense2 = engine.toDense(poly: SparsePoly(dense: dense))
        for i in 0..<8 {
            expectEqual(dense2[i], dense[i], "Round-trip coeff[\(i)]")
        }

    } catch { expect(false, "toDense error: \(error)") }
}

// MARK: - mulDense (GPU)

private func testSparseMulDenseGPU() {
    suite("GPUSparsePolyEngine mulDense (GPU)")
    do {
        let engine = try GPUSparsePolyEngine()
        engine.gpuWorkThreshold = 1  // Force GPU path
        var rng = SparsePolyRNG(state: 0x5BAE5E03)

        // Simple known case: (1 + x^2) * (1 + x) = 1 + x + x^2 + x^3
        let one = frFromInt(1)
        let sparse = SparsePoly(terms: [
            (index: 0, coeff: one),
            (index: 2, coeff: one)
        ], degreeBound: 3)
        let dense = [one, one] // 1 + x

        let result = try engine.mulDense(sparse: sparse, dense: dense)
        // Expected: [1, 1, 1, 1]
        expectEqual(result.count, 4, "Output length 4")
        expectEqual(result[0], one, "result[0] = 1")
        expectEqual(result[1], one, "result[1] = 1")
        expectEqual(result[2], one, "result[2] = 1")
        expectEqual(result[3], one, "result[3] = 1")

        // Random test: sparse * dense vs CPU reference
        let sparseNnz = 10
        let denseLen = 32
        let sparseDeg = 128
        var terms = [(index: Int, coeff: Fr)]()
        var usedIdx = Set<Int>()
        for _ in 0..<sparseNnz {
            var idx: Int
            repeat { idx = rng.nextIndex(bound: sparseDeg) } while usedIdx.contains(idx)
            usedIdx.insert(idx)
            terms.append((index: idx, coeff: rng.nextFr()))
        }
        let randSparse = SparsePoly(terms: terms, degreeBound: sparseDeg)

        var randDense = [Fr]()
        for _ in 0..<denseLen { randDense.append(rng.nextFr()) }

        let gpuResult = try engine.mulDense(sparse: randSparse, dense: randDense)
        let cpuResult = cpuMulDense(randSparse, randDense)

        expectEqual(gpuResult.count, cpuResult.count, "Output lengths match")
        var allMatch = true
        for i in 0..<min(gpuResult.count, cpuResult.count) {
            if gpuResult[i] != cpuResult[i] {
                allMatch = false
                expect(false, "mulDense mismatch at index \(i)")
                break
            }
        }
        if allMatch { expect(true, "Random sparse*dense GPU matches CPU") }

    } catch { expect(false, "mulDense error: \(error)") }
}

// MARK: - Edge cases

private func testSparseEdgeCases() {
    suite("GPUSparsePolyEngine edge cases")
    do {
        let engine = try GPUSparsePolyEngine()
        engine.gpuWorkThreshold = 1

        // Empty polynomial
        let empty = SparsePoly(terms: [], degreeBound: 0)
        let r0 = engine.evaluate(poly: empty, point: Fr.one)
        expectEqual(r0, Fr.zero, "Empty poly evaluates to zero")

        let multi = try engine.evaluateMulti(poly: empty, points: [Fr.one, Fr.zero])
        expectEqual(multi.count, 2, "Empty poly multi-eval returns correct count")
        expectEqual(multi[0], Fr.zero, "Empty poly multi-eval[0] = 0")

        // Single term: p(x) = 5*x^0 = 5
        let five = frFromInt(5)
        let single = SparsePoly(terms: [(index: 0, coeff: five)], degreeBound: 1)
        let r1 = engine.evaluate(poly: single, point: frFromInt(42))
        expectEqual(r1, five, "Constant poly = 5 at any point")

        // toDense with zero degree
        let d0 = engine.toDense(poly: empty)
        expectEqual(d0.count, 0, "Empty toDense has 0 elements")

        // mulDense with empty sparse
        let mulEmpty = try engine.mulDense(sparse: empty, dense: [Fr.one])
        expectEqual(mulEmpty.count, 0, "Empty sparse * dense = empty")

    } catch { expect(false, "Edge case error: \(error)") }
}

// MARK: - Performance

private func testSparsePerformance() {
    suite("GPUSparsePolyEngine performance")
    do {
        let engine = try GPUSparsePolyEngine()
        engine.gpuWorkThreshold = 1  // Force GPU
        var rng = SparsePolyRNG(state: 0xBEEF0001)

        // Build a sparse polynomial: 50 non-zero terms in degree 100k
        let degree = 100_000
        let nnz = 50
        var terms = [(index: Int, coeff: Fr)]()
        var usedIdx = Set<Int>()
        for _ in 0..<nnz {
            var idx: Int
            repeat { idx = rng.nextIndex(bound: degree) } while usedIdx.contains(idx)
            usedIdx.insert(idx)
            terms.append((index: idx, coeff: rng.nextFr()))
        }
        let poly = SparsePoly(terms: terms, degreeBound: degree)

        // Evaluate at 1024 points
        let numPoints = 1024
        var points = [Fr]()
        for _ in 0..<numPoints { points.append(rng.nextFr()) }

        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.evaluateMulti(poly: poly, points: points)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print("  Sparse eval (\(nnz) terms, deg \(degree)) at \(numPoints) points: \(String(format: "%.2f", dt * 1000))ms")
        expect(true, "Sparse eval perf benchmark ran")

        // mulDense benchmark: 50 sparse terms * 4096 dense
        let denseLen = 4096
        var dense = [Fr]()
        for _ in 0..<denseLen { dense.append(rng.nextFr()) }

        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.mulDense(sparse: poly, dense: dense)
        let dt1 = CFAbsoluteTimeGetCurrent() - t1
        print("  Sparse*Dense (\(nnz) sparse, \(denseLen) dense): \(String(format: "%.2f", dt1 * 1000))ms")
        expect(true, "mulDense perf benchmark ran")

    } catch { expect(false, "Performance test error: \(error)") }
}
