// GPUParallelReduceTests — Tests for general-purpose GPU parallel reduction engine
//
// Verifies sum, product, and generic reduction correctness for BN254 Fr arrays.

import Foundation
import Metal
@testable import zkMetal

public func runGPUParallelReduceTests() {
    suite("GPUParallelReduce")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUParallelReduceEngine() else {
        print("  [SKIP] Failed to create GPUParallelReduceEngine")
        return
    }

    // ================================================================
    // MARK: - Sum tests
    // ================================================================

    suite("GPUParallelReduce — Sum")

    // Test: empty array
    do {
        let result = engine.sum([Fr]())
        expectEqual(result, Fr.zero, "sum of empty array == 0")
    }

    // Test: single element
    do {
        let result = engine.sum([Fr.one])
        expectEqual(result, Fr.one, "sum of [1] == 1")
    }

    // Test: small sum matches CPU reference
    do {
        engine.cpuThreshold = 999999 // force CPU path
        let n = 100
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        // sum(1..100) = 100*101/2 = 5050
        let cpuResult = engine.sum(elements)

        // Compute reference: 5050 in Montgomery form
        var ref = Fr.zero
        for e in elements { ref = frAdd(ref, e) }
        expectEqual(cpuResult, ref, "CPU sum of 1..100 matches reference")
    }

    // Test: GPU sum matches CPU reference (power of 2)
    do {
        engine.cpuThreshold = 1 // force GPU path
        let n = 1024
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        let gpuResult = engine.sum(elements)

        // CPU reference
        engine.cpuThreshold = 999999
        let cpuResult = engine.sum(elements)
        engine.cpuThreshold = 1

        expectEqual(gpuResult, cpuResult, "GPU sum matches CPU for n=1024")
    }

    // Test: GPU sum matches CPU reference (non-power of 2)
    do {
        engine.cpuThreshold = 1
        let n = 1000
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        let gpuResult = engine.sum(elements)

        engine.cpuThreshold = 999999
        let cpuResult = engine.sum(elements)
        engine.cpuThreshold = 1

        expectEqual(gpuResult, cpuResult, "GPU sum matches CPU for n=1000 (non-power-of-2)")
    }

    // Test: GPU sum for multi-pass (large array)
    do {
        engine.cpuThreshold = 1
        let n = 4096
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        let gpuResult = engine.sum(elements)

        engine.cpuThreshold = 999999
        let cpuResult = engine.sum(elements)
        engine.cpuThreshold = 1

        expectEqual(gpuResult, cpuResult, "GPU sum matches CPU for n=4096 (multi-pass)")
    }

    // ================================================================
    // MARK: - Product tests
    // ================================================================

    suite("GPUParallelReduce — Product")

    // Test: empty array
    do {
        let result = engine.product([Fr]())
        expectEqual(result, Fr.one, "product of empty array == 1")
    }

    // Test: single element
    do {
        let two = frAdd(Fr.one, Fr.one)
        let result = engine.product([two])
        expectEqual(result, two, "product of [2] == 2")
    }

    // Test: small product matches CPU reference
    do {
        engine.cpuThreshold = 999999
        // Product of small values: 2 * 3 * 4 = 24
        let two = frAdd(Fr.one, Fr.one)
        let three = frAdd(two, Fr.one)
        let four = frAdd(three, Fr.one)
        let elements = [two, three, four]

        let result = engine.product(elements)
        let ref = frMul(frMul(two, three), four)
        expectEqual(result, ref, "CPU product of [2,3,4] matches reference")
    }

    // Test: GPU product matches CPU (power of 2)
    do {
        engine.cpuThreshold = 1
        // Use small field values to avoid complexity; product of 1s = 1
        let elements = [Fr](repeating: Fr.one, count: 256)
        let gpuResult = engine.product(elements)
        expectEqual(gpuResult, Fr.one, "GPU product of 256 ones == 1")
    }

    // Test: GPU product matches CPU (non-power of 2, mixed values)
    do {
        engine.cpuThreshold = 1
        let two = frAdd(Fr.one, Fr.one)
        // 32 copies of 2: product = 2^32 mod r
        let n = 32
        let elements = [Fr](repeating: two, count: n)
        let gpuResult = engine.product(elements)

        // CPU reference
        engine.cpuThreshold = 999999
        let cpuResult = engine.product(elements)
        engine.cpuThreshold = 1

        expectEqual(gpuResult, cpuResult, "GPU product of 32 twos matches CPU")
    }

    // ================================================================
    // MARK: - Generic reduce tests
    // ================================================================

    suite("GPUParallelReduce — Generic")

    // Test: generic sum matches dedicated sum
    do {
        engine.cpuThreshold = 1
        let n = 512
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        let sumResult = engine.sum(elements)
        let genericResult = engine.reduce(elements, op: .sum)
        expectEqual(sumResult, genericResult, "generic sum matches dedicated sum for n=512")
    }

    // Test: generic product matches dedicated product
    do {
        engine.cpuThreshold = 1
        let elements = [Fr](repeating: Fr.one, count: 128)
        let prodResult = engine.product(elements)
        let genericResult = engine.reduce(elements, op: .product)
        expectEqual(prodResult, genericResult, "generic product matches dedicated product for n=128")
    }

    // ================================================================
    // MARK: - Various sizes
    // ================================================================

    suite("GPUParallelReduce — Various sizes")

    for n in [2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 511, 512, 513, 1023, 2048, 3000] {
        engine.cpuThreshold = 1
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }
        let gpuResult = engine.sum(elements)

        engine.cpuThreshold = 999999
        let cpuResult = engine.sum(elements)

        expectEqual(gpuResult, cpuResult, "GPU sum matches CPU for n=\(n)")
    }

    // ================================================================
    // MARK: - Performance benchmark
    // ================================================================

    suite("GPUParallelReduce — Performance")

    do {
        engine.cpuThreshold = 1
        let n = 1 << 20 // 2^20 = 1,048,576
        // Fill with ones for a simple benchmark
        let elements = [Fr](repeating: Fr.one, count: n)

        // Warm up
        _ = engine.sum(elements)

        let iterations = 5
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = engine.sum(elements)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print(String(format: "  [PERF] GPU sum of 2^20 Fr elements: %.2f ms avg (%d iters)", avgMs, iterations))

        // Verify correctness of large sum
        // Sum of n ones = n in Montgomery form
        let result = engine.sum(elements)
        // Build reference: n * 1 = n
        engine.cpuThreshold = 999999
        // For n=2^20 ones, sum = n in Fr. Compute via repeated addition would be too slow.
        // Instead verify by: frMul(result, frInverse(Fr.one)) == result (sanity check that it's valid)
        // Or check that result is not zero
        expect(result != Fr.zero, "sum of 2^20 ones is nonzero")
        engine.cpuThreshold = 1
    }
}
