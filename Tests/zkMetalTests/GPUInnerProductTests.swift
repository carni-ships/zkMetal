// GPUInnerProductTests — Tests for GPU-accelerated inner product engine
//
// Validates correctness of field inner products across CPU and GPU paths,
// including batch operations, edge cases, and performance benchmarks.

import Foundation
import Metal
import zkMetal

public func runGPUInnerProductTests() {
    suite("GPUInnerProduct")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUInnerProductEngine() else {
        print("  [SKIP] Failed to create GPUInnerProductEngine")
        return
    }

    // Helper: CPU reference inner product
    func cpuInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }

    // Helper: generate random-ish Fr values from index (deterministic)
    func makeFr(_ seed: UInt64) -> Fr {
        // Use frFromInt with various seeds for deterministic test vectors
        return frFromInt(seed &+ 1)
    }

    // ================================================================
    // MARK: - Single element
    // ================================================================
    suite("GPUInnerProduct — Single element")

    do {
        let a = [frFromInt(7)]
        let b = [frFromInt(11)]
        let expected = frMul(frFromInt(7), frFromInt(11))

        // Force CPU path
        let saved = engine.cpuThreshold
        engine.cpuThreshold = 999999
        let cpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(cpuResult, expected, "single element CPU")

        // Force GPU path
        engine.cpuThreshold = 0
        let gpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(gpuResult, expected, "single element GPU")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Zero vector
    // ================================================================
    suite("GPUInnerProduct — Zero vectors")

    do {
        // Empty vectors
        let emptyResult = engine.fieldInnerProduct(a: [], b: [])
        expectEqual(emptyResult, Fr.zero, "empty vectors")

        // All-zero a
        let n = 128
        let zeros = [Fr](repeating: Fr.zero, count: n)
        let ones = (0..<n).map { _ in frFromInt(42) }

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let result1 = engine.fieldInnerProduct(a: zeros, b: ones)
        expectEqual(result1, Fr.zero, "zero a vector GPU")

        engine.cpuThreshold = 999999
        let result2 = engine.fieldInnerProduct(a: ones, b: zeros)
        expectEqual(result2, Fr.zero, "zero b vector CPU")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Small vector correctness
    // ================================================================
    suite("GPUInnerProduct — Small vector (n=16)")

    do {
        let n = 16
        let a = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let b = (0..<n).map { frFromInt(UInt64($0) + 100) }
        let expected = cpuInnerProduct(a, b)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 999999
        let cpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(cpuResult, expected, "n=16 CPU path")

        engine.cpuThreshold = 0
        let gpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(gpuResult, expected, "n=16 GPU path")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Medium vector (n=1024)
    // ================================================================
    suite("GPUInnerProduct — Medium vector (n=1024)")

    do {
        let n = 1024
        let a = (0..<n).map { frFromInt(UInt64($0) * 3 + 7) }
        let b = (0..<n).map { frFromInt(UInt64($0) * 5 + 13) }
        let expected = cpuInnerProduct(a, b)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let gpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(gpuResult, expected, "n=1024 GPU vs CPU")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Large vector GPU vs CPU (2^16)
    // ================================================================
    suite("GPUInnerProduct — Large vector (n=65536)")

    do {
        let n = 1 << 16
        let a = (0..<n).map { frFromInt(UInt64($0 &* 7 &+ 3)) }
        let b = (0..<n).map { frFromInt(UInt64($0 &* 11 &+ 5)) }

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 999999
        let cpuResult = cpuInnerProduct(a, b)

        engine.cpuThreshold = 0
        let gpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(gpuResult, cpuResult, "n=65536 GPU vs CPU")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - weightedSum API
    // ================================================================
    suite("GPUInnerProduct — weightedSum")

    do {
        let n = 256
        let values = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let weights = (0..<n).map { frFromInt(UInt64($0) + 50) }
        let expected = cpuInnerProduct(values, weights)

        let result = engine.weightedSum(values: values, weights: weights)
        expectEqual(result, expected, "weightedSum matches inner product")
    }

    // ================================================================
    // MARK: - multiEqInnerProduct API
    // ================================================================
    suite("GPUInnerProduct — multiEqInnerProduct")

    do {
        let n = 512
        let evals = (0..<n).map { frFromInt(UInt64($0) * 2 + 1) }
        let eq = (0..<n).map { frFromInt(UInt64($0) + 7) }
        let expected = cpuInnerProduct(evals, eq)

        let result = engine.multiEqInnerProduct(evals: evals, eq: eq)
        expectEqual(result, expected, "multiEqInnerProduct correctness")
    }

    // ================================================================
    // MARK: - Batch inner product
    // ================================================================
    suite("GPUInnerProduct — Batch inner product")

    do {
        let batchSize = 8
        var pairs = [([Fr], [Fr])]()
        var expected = [Fr]()

        for k in 0..<batchSize {
            let n = 64 + k * 16  // varying sizes: 64, 80, 96, ...
            let a = (0..<n).map { frFromInt(UInt64($0 + k * 1000 + 1)) }
            let b = (0..<n).map { frFromInt(UInt64($0 + k * 2000 + 1)) }
            pairs.append((a, b))
            expected.append(cpuInnerProduct(a, b))
        }

        let results = engine.batchFieldInnerProduct(pairs: pairs)
        expectEqual(results.count, batchSize, "batch count")
        for k in 0..<batchSize {
            expectEqual(results[k], expected[k], "batch[\(k)]")
        }
    }

    // ================================================================
    // MARK: - Batch with uniform sizes
    // ================================================================
    suite("GPUInnerProduct — Batch uniform sizes")

    do {
        let batchSize = 16
        let n = 128
        var pairs = [([Fr], [Fr])]()
        var expected = [Fr]()

        for k in 0..<batchSize {
            let a = (0..<n).map { frFromInt(UInt64($0 &+ k &* 100 &+ 1)) }
            let b = (0..<n).map { frFromInt(UInt64($0 &+ k &* 200 &+ 1)) }
            pairs.append((a, b))
            expected.append(cpuInnerProduct(a, b))
        }

        let results = engine.batchFieldInnerProduct(pairs: pairs)
        for k in 0..<batchSize {
            expectEqual(results[k], expected[k], "batch uniform[\(k)]")
        }
    }

    // ================================================================
    // MARK: - Pre-staged buffers
    // ================================================================
    suite("GPUInnerProduct — Pre-staged buffers")

    do {
        let n = 2048
        engine.stageBuffers(maxSize: n)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0

        let a = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let b = (0..<n).map { frFromInt(UInt64($0) + 500) }
        let expected = cpuInnerProduct(a, b)

        let result = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(result, expected, "pre-staged n=2048")

        // Second call reuses staged buffers
        let a2 = (0..<n).map { frFromInt(UInt64($0) * 3 + 1) }
        let b2 = (0..<n).map { frFromInt(UInt64($0) * 7 + 1) }
        let expected2 = cpuInnerProduct(a2, b2)
        let result2 = engine.fieldInnerProduct(a: a2, b: b2)
        expectEqual(result2, expected2, "pre-staged reuse")

        engine.releaseStaged()
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Identity: <a, 1> = sum(a)
    // ================================================================
    suite("GPUInnerProduct — Identity <a, ones> = sum(a)")

    do {
        let n = 512
        let a = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let ones = [Fr](repeating: Fr.one, count: n)

        let ipResult = engine.fieldInnerProduct(a: a, b: ones)
        var sumResult = Fr.zero
        for x in a { sumResult = frAdd(sumResult, x) }
        expectEqual(ipResult, sumResult, "<a, 1> = sum(a)")
    }

    // ================================================================
    // MARK: - Non-power-of-2 size
    // ================================================================
    suite("GPUInnerProduct — Non-power-of-2 (n=1000)")

    do {
        let n = 1000
        let a = (0..<n).map { frFromInt(UInt64($0) * 13 + 1) }
        let b = (0..<n).map { frFromInt(UInt64($0) * 17 + 1) }
        let expected = cpuInnerProduct(a, b)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let gpuResult = engine.fieldInnerProduct(a: a, b: b)
        expectEqual(gpuResult, expected, "n=1000 non-power-of-2")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Performance benchmark (2^16)
    // ================================================================
    suite("GPUInnerProduct — Performance 2^16")

    do {
        let n = 1 << 16
        let a = (0..<n).map { frFromInt(UInt64($0 &* 7 &+ 3)) }
        let b = (0..<n).map { frFromInt(UInt64($0 &* 11 &+ 5)) }

        let saved = engine.cpuThreshold

        // Warmup
        engine.cpuThreshold = 0
        _ = engine.fieldInnerProduct(a: a, b: b)

        // GPU timing
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        for _ in 0..<iterations {
            _ = engine.fieldInnerProduct(a: a, b: b)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / Double(iterations) * 1000.0

        // CPU timing
        engine.cpuThreshold = 999999
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = engine.fieldInnerProduct(a: a, b: b)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / Double(iterations) * 1000.0

        print(String(format: "  n=2^16: GPU %.2fms, CPU %.2fms, speedup %.1fx", gpuTime, cpuTime, cpuTime / gpuTime))
        expect(true, "benchmark completed")

        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Performance benchmark (2^20)
    // ================================================================
    suite("GPUInnerProduct — Performance 2^20")

    do {
        let n = 1 << 20
        let a = (0..<n).map { frFromInt(UInt64($0 &* 7 &+ 3)) }
        let b = (0..<n).map { frFromInt(UInt64($0 &* 11 &+ 5)) }

        let saved = engine.cpuThreshold

        // Warmup
        engine.cpuThreshold = 0
        _ = engine.fieldInnerProduct(a: a, b: b)

        // GPU timing
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let iterations = 5
        for _ in 0..<iterations {
            _ = engine.fieldInnerProduct(a: a, b: b)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / Double(iterations) * 1000.0

        // CPU timing
        engine.cpuThreshold = 999999
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = engine.fieldInnerProduct(a: a, b: b)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / Double(iterations) * 1000.0

        print(String(format: "  n=2^20: GPU %.2fms, CPU %.2fms, speedup %.1fx", gpuTime, cpuTime, cpuTime / gpuTime))
        expect(true, "benchmark completed")

        engine.cpuThreshold = saved
    }
}
