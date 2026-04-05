// GPUBatchInverseEngineTests — Tests for GPU-accelerated batch modular inverse engine
//
// Verifies correctness of Montgomery's trick for BN254 Fr field on GPU.
// Tests: small array correctness, large array (2^16, 2^20) vs CPU reference,
// zero element handling, and performance comparison CPU vs GPU.

import Foundation
import Metal
@testable import zkMetal

public func runGPUBatchInverseEngineTests() {
    suite("GPUBatchInverseEngine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUBatchInverseEngine() else {
        print("  [SKIP] Failed to create GPUBatchInverseEngine")
        return
    }

    let savedThreshold = engine.cpuThreshold

    // ================================================================
    // MARK: - Small Array Correctness (CPU path)
    // ================================================================

    suite("GPUBatchInverseEngine — Small Array Correctness")

    // Test: inverse of 1 == 1
    do {
        engine.cpuThreshold = 999999  // force CPU path
        let result = try! engine.batchInverseFr([Fr.one])
        expectEqual(result[0], Fr.one, "inverse of 1 == 1")
    }

    // Test: batch inverse of [2, 3, ..., 17] — all products a[i]*inv[i] == 1
    do {
        engine.cpuThreshold = 999999
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<16 {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let inverses = try! engine.batchInverseFr(elements)
        expectEqual(inverses.count, elements.count, "output count matches input")
        for i in 0..<elements.count {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "CPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Test: consistency with frInverse (single-element Fermat inverse)
    do {
        engine.cpuThreshold = 999999
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<8 {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let batchResult = try! engine.batchInverseFr(elements)
        for i in 0..<elements.count {
            let singleInv = frInverse(elements[i])
            expectEqual(batchResult[i], singleInv, "batch inv[\(i)] == frInverse(a[\(i)])")
        }
    }

    // Test: empty array
    do {
        engine.cpuThreshold = 999999
        let result = try! engine.batchInverseFr([])
        expectEqual(result.count, 0, "empty input -> empty output")
    }

    // ================================================================
    // MARK: - Zero Element Handling
    // ================================================================

    suite("GPUBatchInverseEngine — Zero Handling")

    // Test: zero in middle — should return 0 for that position, correct inverses elsewhere
    do {
        engine.cpuThreshold = 999999
        let two = frAdd(Fr.one, Fr.one)
        let three = frAdd(two, Fr.one)
        let elements = [two, Fr.zero, three]

        let inverses = try! engine.batchInverseFr(elements)
        let prod0 = frMul(elements[0], inverses[0])
        expectEqual(prod0, Fr.one, "inv(2) * 2 == 1")
        expectEqual(inverses[1], Fr.zero, "inv(0) == 0")
        let prod2 = frMul(elements[2], inverses[2])
        expectEqual(prod2, Fr.one, "inv(3) * 3 == 1")
    }

    // Test: all zeros
    do {
        engine.cpuThreshold = 999999
        let elements = [Fr.zero, Fr.zero, Fr.zero]
        let inverses = try! engine.batchInverseFr(elements)
        for i in 0..<3 {
            expectEqual(inverses[i], Fr.zero, "all-zero: inv[\(i)] == 0")
        }
    }

    // Test: zero at start and end
    do {
        engine.cpuThreshold = 999999
        let five = frAdd(frAdd(frAdd(frAdd(Fr.one, Fr.one), Fr.one), Fr.one), Fr.one)
        let elements = [Fr.zero, five, Fr.zero]
        let inverses = try! engine.batchInverseFr(elements)
        expectEqual(inverses[0], Fr.zero, "zero at start: inv[0] == 0")
        let prod1 = frMul(elements[1], inverses[1])
        expectEqual(prod1, Fr.one, "middle element: inv(5) * 5 == 1")
        expectEqual(inverses[2], Fr.zero, "zero at end: inv[2] == 0")
    }

    // ================================================================
    // MARK: - Large Array Correctness (GPU path, 2^16)
    // ================================================================

    suite("GPUBatchInverseEngine — Large Array 2^16 (GPU)")

    do {
        engine.cpuThreshold = 256  // force GPU path for N >= 256
        let n = 1 << 16
        var elements = [Fr]()
        elements.reserveCapacity(n)
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let inverses = try! engine.batchInverseFr(elements)
        expectEqual(inverses.count, n, "GPU 2^16: output count == \(n)")

        // Spot-check first 32 and last 32 elements
        for i in 0..<32 {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "GPU 2^16: a[\(i)] * inv[\(i)] == 1")
        }
        for i in (n - 32)..<n {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "GPU 2^16: a[\(i)] * inv[\(i)] == 1")
        }

        // Verify against CPU reference for a sample
        engine.cpuThreshold = 999999  // force CPU
        let cpuRef = try! engine.batchInverseFr(Array(elements[0..<256]))
        engine.cpuThreshold = 256
        for i in 0..<256 {
            expectEqual(inverses[i], cpuRef[i], "GPU vs CPU ref: inv[\(i)] match")
        }
    }

    // ================================================================
    // MARK: - Large Array Correctness (GPU path, 2^20)
    // ================================================================

    suite("GPUBatchInverseEngine — Large Array 2^20 (GPU)")

    do {
        engine.cpuThreshold = 256
        let n = 1 << 20
        var elements = [Fr]()
        elements.reserveCapacity(n)
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let inverses = try! engine.batchInverseFr(elements)
        let dtGPU = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        expectEqual(inverses.count, n, "GPU 2^20: output count == \(n)")

        // Spot-check first 16 and last 16
        for i in 0..<16 {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "GPU 2^20: a[\(i)] * inv[\(i)] == 1")
        }
        for i in (n - 16)..<n {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "GPU 2^20: a[\(i)] * inv[\(i)] == 1")
        }

        print("  BN254 Fr GPU 2^20 batch inverse: \(String(format: "%.2f", dtGPU))ms")
    }

    // ================================================================
    // MARK: - GPU vs CPU Reference Correctness (2^16 with zeros)
    // ================================================================

    suite("GPUBatchInverseEngine — GPU vs CPU Reference with Zeros")

    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Fr]()
        elements.reserveCapacity(n)
        var x = Fr.one
        for i in 0..<n {
            x = frAdd(x, Fr.one)
            // Insert zeros at every 100th position
            if i % 100 == 50 {
                elements.append(Fr.zero)
            } else {
                elements.append(x)
            }
        }

        // GPU result
        let gpuResult = try! engine.batchInverseFr(elements)

        // CPU reference result
        engine.cpuThreshold = 999999
        let cpuResult = try! engine.batchInverseFr(elements)
        engine.cpuThreshold = 256

        // Compare all elements
        var mismatches = 0
        for i in 0..<n {
            if gpuResult[i] != cpuResult[i] {
                mismatches += 1
                if mismatches <= 5 {
                    print("  [MISMATCH] i=\(i) GPU!=CPU")
                }
            }
        }
        expectEqual(mismatches, 0, "GPU vs CPU: 0 mismatches out of \(n)")
    }

    // ================================================================
    // MARK: - Performance: CPU vs GPU
    // ================================================================

    suite("GPUBatchInverseEngine — Performance CPU vs GPU")

    // 2^16 elements: batch GPU vs sequential CPU Fermat inverse
    do {
        let n = 1 << 16
        var elements = [Fr]()
        elements.reserveCapacity(n)
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        // GPU batch (Montgomery's trick on GPU)
        engine.cpuThreshold = 256
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try! engine.batchInverseFr(elements)
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // CPU batch (Montgomery's trick on CPU)
        engine.cpuThreshold = 999999
        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try! engine.batchInverseFr(elements)
        let cpuBatchMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        // CPU sequential (individual Fermat inverses)
        let t2 = CFAbsoluteTimeGetCurrent()
        for i in 0..<min(n, 1024) {
            let _ = frInverse(elements[i])
        }
        let cpuSeqPartialMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        let cpuSeqEstMs = cpuSeqPartialMs * Double(n) / 1024.0

        print("  BN254 Fr 2^16 batch inverse:")
        print("    GPU batch:          \(String(format: "%.2f", gpuMs))ms")
        print("    CPU batch:          \(String(format: "%.2f", cpuBatchMs))ms")
        print("    CPU sequential est: \(String(format: "%.0f", cpuSeqEstMs))ms (extrapolated from 1024)")
        if gpuMs > 0 {
            print("    GPU vs CPU-batch:   \(String(format: "%.1f", cpuBatchMs / gpuMs))x")
            print("    GPU vs CPU-seq:     \(String(format: "%.0f", cpuSeqEstMs / gpuMs))x")
        }
        expect(true, "performance logged")
    }

    // ================================================================
    // MARK: - In-place MTLBuffer API
    // ================================================================

    suite("GPUBatchInverseEngine — MTLBuffer API")

    do {
        engine.cpuThreshold = 256
        let n = 1024
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let byteSize = n * MemoryLayout<Fr>.stride
        guard let inputBuf = engine.device.makeBuffer(bytes: elements, length: byteSize, options: .storageModeShared) else {
            expect(false, "failed to create input buffer")
            engine.cpuThreshold = savedThreshold
            return
        }

        let outBuf = try! engine.batchInverseBuffer(inputBuf, count: n, field: .bn254)
        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: n)
        for i in 0..<16 {
            let product = frMul(elements[i], ptr[i])
            expectEqual(product, Fr.one, "MTLBuffer: a[\(i)] * inv[\(i)] == 1")
        }
        engine.releaseBuffer(outBuf)
    }

    engine.cpuThreshold = savedThreshold
}
