// GPUBatchInverseTests — Tests for GPU-accelerated batch modular inverse engine
//
// Verifies correctness of Montgomery's trick across BN254, BabyBear, and Goldilocks fields.

import Foundation
import Metal
import zkMetal

public func runGPUBatchInverseTests() {
    suite("GPUBatchInverse")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUBatchInverseEngine() else {
        print("  [SKIP] Failed to create GPUBatchInverseEngine")
        return
    }

    // Force CPU path for small tests, GPU for large
    let savedThreshold = engine.cpuThreshold

    // ================================================================
    // MARK: - BN254 Fr Tests
    // ================================================================

    suite("GPUBatchInverse — BN254 Fr")

    // Test: single element inverse
    do {
        engine.cpuThreshold = 999999  // force CPU path
        let a = Fr.one
        let result = try! engine.batchInverseFr([a])
        // 1^(-1) = 1
        expectEqual(result[0], Fr.one, "BN254: inverse of 1 == 1")
    }

    // Test: correctness a[i] * inv[i] == 1
    do {
        engine.cpuThreshold = 999999  // force CPU
        // Small test values in Montgomery form: use frMul to create elements
        var elements = [Fr]()
        var x = Fr.one
        for _ in 0..<16 {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let inverses = try! engine.batchInverseFr(elements)
        for i in 0..<elements.count {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "BN254 CPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Test: zero handling
    do {
        engine.cpuThreshold = 999999
        var elements = [Fr]()
        elements.append(frAdd(Fr.one, Fr.one))  // 2
        elements.append(Fr.zero)                  // 0
        elements.append(frAdd(frAdd(Fr.one, Fr.one), Fr.one))  // 3

        let inverses = try! engine.batchInverseFr(elements)
        let prod0 = frMul(elements[0], inverses[0])
        expectEqual(prod0, Fr.one, "BN254: inv(2) * 2 == 1")
        expectEqual(inverses[1], Fr.zero, "BN254: inv(0) == 0")
        let prod2 = frMul(elements[2], inverses[2])
        expectEqual(prod2, Fr.one, "BN254: inv(3) * 3 == 1")
    }

    // Test: consistency with single-element inverse
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
            expectEqual(batchResult[i], singleInv, "BN254: batch inv[\(i)] == single inv[\(i)]")
        }
    }

    // Test: GPU path (2^16 elements)
    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Fr]()
        elements.reserveCapacity(n)
        var x = Fr.one
        for _ in 0..<n {
            x = frAdd(x, Fr.one)
            elements.append(x)
        }

        let inverses = try! engine.batchInverseFr(elements)
        expectEqual(inverses.count, n, "BN254 GPU: got \(n) inverses")

        // Spot-check first 16 and last 16
        for i in 0..<16 {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "BN254 GPU: a[\(i)] * inv[\(i)] == 1")
        }
        for i in (n - 16)..<n {
            let product = frMul(elements[i], inverses[i])
            expectEqual(product, Fr.one, "BN254 GPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // ================================================================
    // MARK: - BabyBear Tests
    // ================================================================

    suite("GPUBatchInverse — BabyBear")

    // Test: single element
    do {
        engine.cpuThreshold = 999999
        let a = [Bb(v: 1)]
        let result = try! engine.batchInverseBb(a)
        expectEqual(result[0].v, 1, "BB: inverse of 1 == 1")
    }

    // Test: correctness
    do {
        engine.cpuThreshold = 999999
        var elements = [Bb]()
        for i in 1...32 {
            elements.append(Bb(v: UInt32(i)))
        }

        let inverses = try! engine.batchInverseBb(elements)
        for i in 0..<elements.count {
            let product = bbMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "BB CPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Test: zero handling
    do {
        engine.cpuThreshold = 999999
        let elements = [Bb(v: 5), Bb(v: 0), Bb(v: 7)]
        let inverses = try! engine.batchInverseBb(elements)

        let prod0 = bbMul(elements[0], inverses[0])
        expectEqual(prod0.v, 1, "BB: inv(5) * 5 == 1")
        expectEqual(inverses[1].v, 0, "BB: inv(0) == 0")
        let prod2 = bbMul(elements[2], inverses[2])
        expectEqual(prod2.v, 1, "BB: inv(7) * 7 == 1")
    }

    // Test: consistency with single inverse
    do {
        engine.cpuThreshold = 999999
        var elements = [Bb]()
        for i in 1...16 {
            elements.append(Bb(v: UInt32(i)))
        }

        let batchResult = try! engine.batchInverseBb(elements)
        for i in 0..<elements.count {
            let singleInv = bbInverse(elements[i])
            expectEqual(batchResult[i].v, singleInv.v, "BB: batch inv[\(i)] == single inv[\(i)]")
        }
    }

    // Test: GPU path (2^16 elements)
    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Bb]()
        elements.reserveCapacity(n)
        for i in 1...n {
            elements.append(Bb(v: UInt32(i % Int(Bb.P))))
        }
        // Ensure no zeros from modular wraparound
        for i in 0..<n where elements[i].v == 0 {
            elements[i] = Bb(v: 1)
        }

        let inverses = try! engine.batchInverseBb(elements)
        expectEqual(inverses.count, n, "BB GPU: got \(n) inverses")

        for i in 0..<16 {
            let product = bbMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "BB GPU: a[\(i)] * inv[\(i)] == 1")
        }
        for i in (n - 16)..<n {
            let product = bbMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "BB GPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Test: GPU path (2^20 elements)
    do {
        engine.cpuThreshold = 256
        let n = 1 << 20
        var elements = [Bb]()
        elements.reserveCapacity(n)
        for i in 1...n {
            elements.append(Bb(v: UInt32(i % Int(Bb.P))))
        }
        for i in 0..<n where elements[i].v == 0 {
            elements[i] = Bb(v: 1)
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let inverses = try! engine.batchInverseBb(elements)
        let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        expectEqual(inverses.count, n, "BB GPU 2^20: got \(n) inverses")

        // Spot-check
        for i in 0..<8 {
            let product = bbMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "BB GPU 2^20: a[\(i)] * inv[\(i)] == 1")
        }
        print("  BB GPU 2^20 batch inverse: \(String(format: "%.2f", dt))ms")
    }

    // ================================================================
    // MARK: - Goldilocks Tests
    // ================================================================

    suite("GPUBatchInverse — Goldilocks")

    // Test: single element
    do {
        engine.cpuThreshold = 999999
        let a = [Gl(v: 1)]
        let result = try! engine.batchInverseGl(a)
        expectEqual(result[0].v, 1, "GL: inverse of 1 == 1")
    }

    // Test: correctness
    do {
        engine.cpuThreshold = 999999
        var elements = [Gl]()
        for i: UInt64 in 1...32 {
            elements.append(Gl(v: i))
        }

        let inverses = try! engine.batchInverseGl(elements)
        for i in 0..<elements.count {
            let product = glMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "GL CPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Test: zero handling
    do {
        engine.cpuThreshold = 999999
        let elements = [Gl(v: 5), Gl(v: 0), Gl(v: 7)]
        let inverses = try! engine.batchInverseGl(elements)

        let prod0 = glMul(elements[0], inverses[0])
        expectEqual(prod0.v, 1, "GL: inv(5) * 5 == 1")
        expectEqual(inverses[1].v, 0, "GL: inv(0) == 0")
        let prod2 = glMul(elements[2], inverses[2])
        expectEqual(prod2.v, 1, "GL: inv(7) * 7 == 1")
    }

    // Test: consistency with single inverse
    do {
        engine.cpuThreshold = 999999
        var elements = [Gl]()
        for i: UInt64 in 1...16 {
            elements.append(Gl(v: i))
        }

        let batchResult = try! engine.batchInverseGl(elements)
        for i in 0..<elements.count {
            let singleInv = glInverse(elements[i])
            expectEqual(batchResult[i].v, singleInv.v, "GL: batch inv[\(i)] == single inv[\(i)]")
        }
    }

    // Test: GPU path (2^16 elements)
    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Gl]()
        elements.reserveCapacity(n)
        for i: UInt64 in 1...UInt64(n) {
            elements.append(Gl(v: i))
        }

        let inverses = try! engine.batchInverseGl(elements)
        expectEqual(inverses.count, n, "GL GPU: got \(n) inverses")

        for i in 0..<16 {
            let product = glMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "GL GPU: a[\(i)] * inv[\(i)] == 1")
        }
        for i in (n - 16)..<n {
            let product = glMul(elements[i], inverses[i])
            expectEqual(product.v, 1, "GL GPU: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // ================================================================
    // MARK: - Cross-field UInt32 API Tests
    // ================================================================

    suite("GPUBatchInverse — UInt32 API")

    // BabyBear via UInt32 API
    do {
        engine.cpuThreshold = 999999
        let elements: [UInt32] = [3, 7, 11, 13]
        let inverses = try! engine.batchInverse(elements, field: .babybear)
        for i in 0..<elements.count {
            let product = bbMul(Bb(v: elements[i]), Bb(v: inverses[i]))
            expectEqual(product.v, 1, "UInt32 BB: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // Goldilocks via UInt32 API (2 UInt32 per element)
    do {
        engine.cpuThreshold = 999999
        // Encode Goldilocks values as pairs of UInt32 (little-endian)
        let vals: [UInt64] = [3, 7, 11, 13]
        var elements = [UInt32]()
        for v in vals {
            elements.append(UInt32(v & 0xFFFFFFFF))
            elements.append(UInt32(v >> 32))
        }
        let inverses = try! engine.batchInverse(elements, field: .goldilocks)
        for i in 0..<vals.count {
            let invVal = UInt64(inverses[2 * i]) | (UInt64(inverses[2 * i + 1]) << 32)
            let product = glMul(Gl(v: vals[i]), Gl(v: invVal))
            expectEqual(product.v, 1, "UInt32 GL: a[\(i)] * inv[\(i)] == 1")
        }
    }

    // ================================================================
    // MARK: - Performance Comparison
    // ================================================================

    suite("GPUBatchInverse — Performance")

    // BabyBear: GPU batch vs sequential CPU inverse
    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Bb]()
        elements.reserveCapacity(n)
        for i in 1...n {
            elements.append(Bb(v: UInt32(i)))
        }

        // GPU batch
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try! engine.batchInverseBb(elements)
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // Sequential CPU inverse (single-element Fermat for each)
        let t1 = CFAbsoluteTimeGetCurrent()
        var cpuResult = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n {
            cpuResult[i] = bbInverse(elements[i])
        }
        let cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        print("  BB 2^16 batch inverse: GPU=\(String(format: "%.2f", gpuMs))ms, CPU_seq=\(String(format: "%.2f", cpuMs))ms, speedup=\(String(format: "%.1f", cpuMs / max(gpuMs, 0.01)))x")
        expect(true, "performance logged")
    }

    // Goldilocks: GPU batch vs sequential CPU inverse
    do {
        engine.cpuThreshold = 256
        let n = 1 << 16
        var elements = [Gl]()
        elements.reserveCapacity(n)
        for i: UInt64 in 1...UInt64(n) {
            elements.append(Gl(v: i))
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try! engine.batchInverseGl(elements)
        let gpuMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let t1 = CFAbsoluteTimeGetCurrent()
        var cpuResult = [Gl](repeating: Gl.zero, count: n)
        for i in 0..<n {
            cpuResult[i] = glInverse(elements[i])
        }
        let cpuMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        print("  GL 2^16 batch inverse: GPU=\(String(format: "%.2f", gpuMs))ms, CPU_seq=\(String(format: "%.2f", cpuMs))ms, speedup=\(String(format: "%.1f", cpuMs / max(gpuMs, 0.01)))x")
        expect(true, "performance logged")
    }

    // ================================================================
    // MARK: - MTLBuffer API Test
    // ================================================================

    suite("GPUBatchInverse — MTLBuffer API")

    do {
        engine.cpuThreshold = 256
        let n = 1024
        var elements = [Bb]()
        for i in 1...n {
            elements.append(Bb(v: UInt32(i)))
        }

        let byteSize = n * MemoryLayout<Bb>.stride
        guard let inputBuf = engine.device.makeBuffer(bytes: elements, length: byteSize, options: .storageModeShared) else {
            expect(false, "failed to create input buffer")
            return
        }

        let outBuf = try! engine.batchInverseBuffer(inputBuf, count: n, field: .babybear)
        let ptr = outBuf.contents().bindMemory(to: Bb.self, capacity: n)
        for i in 0..<16 {
            let product = bbMul(elements[i], ptr[i])
            expectEqual(product.v, 1, "MTLBuffer BB: a[\(i)] * inv[\(i)] == 1")
        }
        engine.releaseBuffer(outBuf)
    }

    engine.cpuThreshold = savedThreshold
}
