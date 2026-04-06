// GPUMatrixTransposeTests — Tests for GPU-accelerated matrix transpose engine
//
// Validates correctness of tiled GPU transpose against CPU reference,
// including out-of-place rectangular, in-place square, batch, and typed Fr APIs.

import Foundation
import Metal
@testable import zkMetal

public func runGPUMatrixTransposeTests() {
    suite("GPUMatrixTranspose")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUMatrixTransposeEngine() else {
        print("  [SKIP] Failed to create GPUMatrixTransposeEngine")
        return
    }

    // Helper: create deterministic Fr from index
    func makeFr(_ seed: UInt64) -> Fr {
        return frFromInt(seed &+ 1)
    }

    // Helper: CPU reference transpose
    func cpuTranspose(_ a: [Fr], rows: Int, cols: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[j * rows + i] = a[i * cols + j]
            }
        }
        return result
    }

    // ================================================================
    // MARK: - Small matrix (CPU path)
    // ================================================================
    suite("GPUMatrixTranspose — Small matrix (CPU path)")

    do {
        let rows = 4
        let cols = 3
        let n = rows * cols
        let a = (0..<n).map { makeFr(UInt64($0)) }

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 999999  // force CPU path
        let result = try! engine.transposeFr(a, rows: rows, cols: cols)
        let expected = cpuTranspose(a, rows: rows, cols: cols)

        expect(result.count == expected.count, "output size matches")
        var allMatch = true
        for i in 0..<result.count {
            if result[i] != expected[i] { allMatch = false; break }
        }
        expect(allMatch, "small 4x3 transpose matches CPU reference")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - GPU out-of-place rectangular
    // ================================================================
    suite("GPUMatrixTranspose — GPU out-of-place rectangular")

    do {
        let rows = 64
        let cols = 32
        let n = rows * cols
        let a = (0..<n).map { makeFr(UInt64($0)) }
        let expected = cpuTranspose(a, rows: rows, cols: cols)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0  // force GPU path
        let result = try! engine.transposeFr(a, rows: rows, cols: cols)

        expect(result.count == expected.count, "output size \(result.count) == \(expected.count)")
        var mismatches = 0
        for i in 0..<result.count {
            if result[i] != expected[i] { mismatches += 1 }
        }
        expect(mismatches == 0, "GPU 64x32 transpose: \(mismatches) mismatches")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - GPU out-of-place non-tile-aligned
    // ================================================================
    suite("GPUMatrixTranspose — Non-tile-aligned dimensions")

    do {
        // 17x23 — not divisible by tile size 16
        let rows = 17
        let cols = 23
        let n = rows * cols
        let a = (0..<n).map { makeFr(UInt64($0 * 7 + 3)) }
        let expected = cpuTranspose(a, rows: rows, cols: cols)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let result = try! engine.transposeFr(a, rows: rows, cols: cols)

        var mismatches = 0
        for i in 0..<result.count {
            if result[i] != expected[i] { mismatches += 1 }
        }
        expect(mismatches == 0, "GPU 17x23 transpose: \(mismatches) mismatches")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - GPU in-place square
    // ================================================================
    suite("GPUMatrixTranspose — GPU in-place square")

    do {
        let n = 32
        var a = (0..<n * n).map { makeFr(UInt64($0)) }
        let expected = cpuTranspose(a, rows: n, cols: n)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        try! engine.transposeFrInPlace(&a, n: n)

        var mismatches = 0
        for i in 0..<a.count {
            if a[i] != expected[i] { mismatches += 1 }
        }
        expect(mismatches == 0, "GPU in-place 32x32 transpose: \(mismatches) mismatches")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - In-place non-tile-aligned square
    // ================================================================
    suite("GPUMatrixTranspose — In-place non-tile-aligned square")

    do {
        let n = 19
        var a = (0..<n * n).map { makeFr(UInt64($0 * 13 + 5)) }
        let expected = cpuTranspose(a, rows: n, cols: n)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        try! engine.transposeFrInPlace(&a, n: n)

        var mismatches = 0
        for i in 0..<a.count {
            if a[i] != expected[i] { mismatches += 1 }
        }
        expect(mismatches == 0, "GPU in-place 19x19 transpose: \(mismatches) mismatches")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Double transpose = identity
    // ================================================================
    suite("GPUMatrixTranspose — Double transpose = identity")

    do {
        let rows = 48
        let cols = 64
        let n = rows * cols
        let a = (0..<n).map { makeFr(UInt64($0)) }

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let transposed = try! engine.transposeFr(a, rows: rows, cols: cols)
        let back = try! engine.transposeFr(transposed, rows: cols, cols: rows)

        var mismatches = 0
        for i in 0..<a.count {
            if a[i] != back[i] { mismatches += 1 }
        }
        expect(mismatches == 0, "double transpose 48x64 is identity: \(mismatches) mismatches")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Batch transpose
    // ================================================================
    suite("GPUMatrixTranspose — Batch transpose")

    do {
        let rows = 32
        let cols = 16
        let n = rows * cols
        let batchSize = 4

        var matrices = [[Fr]]()
        for b in 0..<batchSize {
            matrices.append((0..<n).map { makeFr(UInt64($0 + b * n)) })
        }

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0

        // Use MTLBuffer API for batch
        let device = engine.device
        var srcBufs = [MTLBuffer]()
        let byteSize = n * MemoryLayout<Fr>.stride
        for m in matrices {
            let buf = device.makeBuffer(length: byteSize, options: .storageModeShared)!
            m.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, byteSize) }
            srcBufs.append(buf)
        }

        let dstBufs = try! engine.batchTranspose(matrices: srcBufs, rows: rows, cols: cols)

        var allCorrect = true
        for b in 0..<batchSize {
            let expected = cpuTranspose(matrices[b], rows: rows, cols: cols)
            let ptr = dstBufs[b].contents().bindMemory(to: Fr.self, capacity: n)
            let result = Array(UnsafeBufferPointer(start: ptr, count: n))
            for i in 0..<n {
                if result[i] != expected[i] { allCorrect = false; break }
            }
            engine.releaseBuffer(dstBufs[b])
        }
        expect(allCorrect, "batch transpose of 4 matrices (32x16)")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Edge cases
    // ================================================================
    suite("GPUMatrixTranspose — Edge cases")

    do {
        // Empty matrix
        let empty = try! engine.transposeFr([], rows: 0, cols: 0)
        expect(empty.isEmpty, "empty matrix transpose")

        // 1x1 matrix
        let single = [makeFr(42)]
        let result = try! engine.transposeFr(single, rows: 1, cols: 1)
        expect(result.count == 1 && result[0] == single[0], "1x1 transpose")

        // Single row (1 x N)
        let rowVec = (0..<16).map { makeFr(UInt64($0)) }
        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let colVec = try! engine.transposeFr(rowVec, rows: 1, cols: 16)
        expect(colVec.count == 16, "1x16 -> 16x1 size")
        var rowMatch = true
        for i in 0..<16 {
            if colVec[i] != rowVec[i] { rowMatch = false; break }
        }
        expect(rowMatch, "1xN transpose preserves elements")
        engine.cpuThreshold = saved
    }

    // ================================================================
    // MARK: - Performance: GPU vs CPU
    // ================================================================
    suite("GPUMatrixTranspose — Performance")

    do {
        let rows = 256
        let cols = 256
        let n = rows * cols
        let a = (0..<n).map { makeFr(UInt64($0)) }

        // CPU timing
        let cpuT0 = CFAbsoluteTimeGetCurrent()
        let cpuIterations = 5
        for _ in 0..<cpuIterations {
            let _ = cpuTranspose(a, rows: rows, cols: cols)
        }
        let cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000.0 / Double(cpuIterations)

        // GPU timing
        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        // Warmup
        let _ = try! engine.transposeFr(a, rows: rows, cols: cols)

        let gpuT0 = CFAbsoluteTimeGetCurrent()
        let gpuIterations = 10
        for _ in 0..<gpuIterations {
            let _ = try! engine.transposeFr(a, rows: rows, cols: cols)
        }
        let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuT0) * 1000.0 / Double(gpuIterations)
        engine.cpuThreshold = saved

        let speedup = cpuMs / max(gpuMs, 0.001)
        print(String(format: "  256x256 Fr: CPU=%.2fms  GPU=%.2fms  (%.1fx)", cpuMs, gpuMs, speedup))
        // Just verify it produces correct results at this size
        let gpuResult = try! engine.transposeFr(a, rows: rows, cols: cols)
        let cpuResult = cpuTranspose(a, rows: rows, cols: cols)
        var ok = true
        for i in 0..<n { if gpuResult[i] != cpuResult[i] { ok = false; break } }
        expect(ok, "256x256 GPU transpose correctness")
    }
}
