// GPUGrandProductEngineTests — Tests for GPU-accelerated grand product engine
//
// Verifies correctness of:
//   - grandProduct (full product reduction)
//   - partialProducts (exclusive prefix product)
//   - permutationProduct (numerator/denominator prefix product)
// Tests both CPU fallback (small arrays) and GPU path (large arrays).

import Foundation
import Metal
import zkMetal

public func runGPUGrandProductEngineTests() {
    suite("GPUGrandProductEngine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUGrandProductEngine() else {
        print("  [SKIP] Failed to create GPUGrandProductEngine")
        return
    }

    // ================================================================
    // MARK: - grandProduct: basic correctness
    // ================================================================

    suite("GPUGrandProductEngine — grandProduct basics")

    // Empty array -> 1
    do {
        let result = engine.grandProduct(values: [])
        expectEqual(result, Fr.one, "grandProduct([]) == 1")
    }

    // Single element
    do {
        let two = frAdd(Fr.one, Fr.one)
        let result = engine.grandProduct(values: [two])
        expectEqual(result, two, "grandProduct([2]) == 2")
    }

    // Product of [1, 2, 3, ..., 8]
    do {
        var elements = [Fr]()
        var x = Fr.zero
        for _ in 0..<8 {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        // Expected: 8! = 40320
        let result = engine.grandProduct(values: elements)
        var expected = Fr.one
        for e in elements { expected = frMul(expected, e) }
        expectEqual(result, expected, "grandProduct([1..8]) == 40320")
    }

    // ================================================================
    // MARK: - partialProducts: basic correctness (CPU path)
    // ================================================================

    suite("GPUGrandProductEngine — partialProducts basics")

    // Empty
    do {
        let result = engine.partialProducts(values: [])
        expectEqual(result.count, 0, "partialProducts([]) is empty")
    }

    // Single element: z[0] = 1
    do {
        let two = frAdd(Fr.one, Fr.one)
        let result = engine.partialProducts(values: [two])
        expectEqual(result.count, 1, "partialProducts([2]) has count 1")
        expectEqual(result[0], Fr.one, "partialProducts([2])[0] == 1")
    }

    // [a, b, c] -> [1, a, a*b]
    do {
        let a = frAdd(Fr.one, Fr.one)                    // 2
        let b = frAdd(a, Fr.one)                          // 3
        let c = frAdd(b, Fr.one)                          // 4
        let result = engine.partialProducts(values: [a, b, c])

        expectEqual(result.count, 3, "partialProducts count")
        expectEqual(result[0], Fr.one, "z[0] == 1")
        expectEqual(result[1], a, "z[1] == a")
        expectEqual(result[2], frMul(a, b), "z[2] == a*b")
    }

    // Larger test: 128 elements, compare CPU reference
    do {
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 999999  // force CPU path

        let n = 128
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let result = engine.partialProducts(values: elements)
        expectEqual(result.count, n, "partialProducts count == \(n)")
        expectEqual(result[0], Fr.one, "z[0] == 1")

        // Verify against sequential computation
        var running = Fr.one
        for i in 0..<n {
            expectEqual(result[i], running, "CPU partialProducts[\(i)] correct")
            running = frMul(running, elements[i])
        }

        engine.gpuThreshold = savedThreshold
    }

    // ================================================================
    // MARK: - partialProducts: GPU path (large array)
    // ================================================================

    suite("GPUGrandProductEngine — partialProducts GPU path")

    do {
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 256  // force GPU path for moderate sizes

        let n = 8192
        var elements = [Fr]()
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        for _ in 0..<n {
            // Generate pseudo-random non-zero Fr elements
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            var limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
            withUnsafeMutableBytes(of: &limbs) { buf in
                let ptr = buf.bindMemory(to: UInt64.self)
                for j in 0..<4 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    ptr[j] = rng
                }
            }
            // Ensure it's in valid range by clearing top bits
            limbs.7 &= 0x0FFFFFFF
            var elem = Fr(v: limbs)
            if elem.isZero { elem = Fr.one }
            elements.append(elem)
        }

        let gpuResult = engine.partialProducts(values: elements)

        // Verify first, last, and spot-check a few positions
        expectEqual(gpuResult.count, n, "GPU partialProducts count == \(n)")
        expectEqual(gpuResult[0], Fr.one, "GPU z[0] == 1")

        // Compute CPU reference
        var cpuRef = [Fr](repeating: Fr.zero, count: n)
        cpuRef[0] = Fr.one
        for i in 1..<n {
            cpuRef[i] = frMul(cpuRef[i - 1], elements[i - 1])
        }

        // Check a sample of positions (checking all 8K would spam output)
        let checkIndices = [0, 1, 2, 100, 511, 512, 513, 1000, 4095, 4096, 4097, n - 2, n - 1]
        var allMatch = true
        for idx in checkIndices where idx < n {
            if gpuResult[idx] != cpuRef[idx] {
                print("  [FAIL] GPU partialProducts[\(idx)] mismatch")
                allMatch = false
            }
        }
        expect(allMatch, "GPU partialProducts matches CPU reference at sample points")

        // Also verify the full product is consistent
        let gpuFullProd = engine.grandProduct(values: elements)
        var cpuFullProd = Fr.one
        for e in elements { cpuFullProd = frMul(cpuFullProd, e) }
        expectEqual(gpuFullProd, cpuFullProd, "GPU grandProduct matches CPU for \(n) elements")

        engine.gpuThreshold = savedThreshold
    }

    // ================================================================
    // MARK: - permutationProduct: correctness
    // ================================================================

    suite("GPUGrandProductEngine — permutationProduct")

    // Basic: numerators = [2, 3, 4], denominators = [1, 1, 1]
    // ratios = [2, 3, 4], prefix = [1, 2, 6]
    do {
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 999999  // force CPU

        let one = Fr.one
        let two = frAdd(one, one)
        let three = frAdd(two, one)
        let four = frAdd(three, one)

        let nums = [two, three, four]
        let dens = [one, one, one]

        let result = engine.permutationProduct(numerators: nums, denominators: dens)
        expectEqual(result.count, 3, "permutationProduct count")
        expectEqual(result[0], Fr.one, "z[0] == 1")
        expectEqual(result[1], two, "z[1] == 2/1 = 2")
        expectEqual(result[2], frMul(two, three), "z[2] == 2*3 = 6")

        engine.gpuThreshold = savedThreshold
    }

    // numerators == denominators -> all partial products == 1
    do {
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 999999

        let n = 32
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let result = engine.permutationProduct(numerators: elements, denominators: elements)
        expectEqual(result.count, n, "permutationProduct identity count")
        for i in 0..<n {
            expectEqual(result[i], Fr.one, "z[\(i)] == 1 when num == den")
        }

        engine.gpuThreshold = savedThreshold
    }

    // GPU path for permutationProduct
    do {
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 256

        let n = 4096
        var nums = [Fr]()
        var dens = [Fr]()
        var rng: UInt64 = 0x1234_5678_9ABC_DEF0

        for _ in 0..<n {
            // Generate pseudo-random non-zero Fr
            var limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
            withUnsafeMutableBytes(of: &limbs) { buf in
                let ptr = buf.bindMemory(to: UInt64.self)
                for j in 0..<4 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    ptr[j] = rng
                }
            }
            limbs.7 &= 0x0FFFFFFF
            var num = Fr(v: limbs)
            if num.isZero { num = Fr.one }
            nums.append(num)

            withUnsafeMutableBytes(of: &limbs) { buf in
                let ptr = buf.bindMemory(to: UInt64.self)
                for j in 0..<4 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    ptr[j] = rng
                }
            }
            limbs.7 &= 0x0FFFFFFF
            var den = Fr(v: limbs)
            if den.isZero { den = Fr.one }
            dens.append(den)
        }

        let gpuResult = engine.permutationProduct(numerators: nums, denominators: dens)

        // CPU reference
        var invDens = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { invDens[i] = frInverse(dens[i]) }
        var ratios = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { ratios[i] = frMul(nums[i], invDens[i]) }
        var cpuRef = [Fr](repeating: Fr.zero, count: n)
        cpuRef[0] = Fr.one
        for i in 1..<n { cpuRef[i] = frMul(cpuRef[i - 1], ratios[i - 1]) }

        expectEqual(gpuResult.count, n, "GPU permutationProduct count")

        let checkIndices = [0, 1, 2, 100, 511, 512, 1000, 2048, n - 1]
        var allMatch = true
        for idx in checkIndices where idx < n {
            if gpuResult[idx] != cpuRef[idx] {
                print("  [FAIL] GPU permutationProduct[\(idx)] mismatch")
                allMatch = false
            }
        }
        expect(allMatch, "GPU permutationProduct matches CPU reference at sample points")

        engine.gpuThreshold = savedThreshold
    }

    // ================================================================
    // MARK: - Performance (brief)
    // ================================================================

    suite("GPUGrandProductEngine — Performance")

    do {
        let n = 1 << 16  // 65536
        var elements = [Fr]()
        var rng: UInt64 = 0xABCD_1234
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            var limbs: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
            withUnsafeMutableBytes(of: &limbs) { buf in
                let ptr = buf.bindMemory(to: UInt64.self)
                for j in 0..<4 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    ptr[j] = rng
                }
            }
            limbs.7 &= 0x0FFFFFFF
            var elem = Fr(v: limbs)
            if elem.isZero { elem = Fr.one }
            elements.append(elem)
        }

        // Warmup
        _ = engine.partialProducts(values: elements)

        // Benchmark GPU
        let t0 = CFAbsoluteTimeGetCurrent()
        let iterations = 5
        for _ in 0..<iterations {
            _ = engine.partialProducts(values: elements)
        }
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iterations)

        // Benchmark CPU
        let savedThreshold = engine.gpuThreshold
        engine.gpuThreshold = 999999
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = engine.partialProducts(values: elements)
        }
        let cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0 / Double(iterations)
        engine.gpuThreshold = savedThreshold

        print(String(format: "  partialProducts(2^16): GPU %.1fms, CPU %.1fms (%.1fx)",
                     gpuMs, cpuMs, cpuMs / max(gpuMs, 0.001)))
        expect(true, "Performance benchmark completed")
    }
}
