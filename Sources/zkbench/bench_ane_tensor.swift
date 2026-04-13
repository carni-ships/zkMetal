// bench_ane_tensor.swift — ANE Tensor Operation benchmarks
//
// Benchmarks matrix-vector multiply, inner product, and matrix-matrix multiply
// using ANE-accelerated Metal kernels with SIMD4 vectorization.
//
// Tests both GPU (ANE when available) and scalar fallback paths.

import ANEOps
import Foundation

public func runANETensorBench() {
    print("=== ANE Tensor Operations Benchmark ===")

    // Initialize ANE tensor
    let initResult = ane_tensor_init()
    if initResult == 0 {
        print("ANE Tensor GPU initialized successfully")
        if ane_tensor_gpu_available() {
            print("GPU tensor operations are available")
        } else {
            print("Warning: GPU initialized but pipelines not available")
        }
    } else {
        print("ANE Tensor GPU init returned \(initResult), using scalar fallback")
    }

    var rng: UInt64 = 0xDEAD_BEEF_CAFE_0001

    // ============================================================
    // Matrix-Vector Multiply: result = M * vec
    // ============================================================
    print("\n--- Matrix-Vector Multiply (rows x cols) ---")

    let matvec_configs = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]

    for (rows, cols) in matvec_configs {
        // Generate random matrix M (rows x cols)
        var M = [UInt32](repeating: 0, count: rows * cols)
        for i in 0..<M.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            M[i] = UInt32(truncatingIfNeeded: rng) % 2013265921  // BabyBear prime
        }

        // Generate random vector vec (cols)
        var vec = [UInt32](repeating: 0, count: cols)
        for i in 0..<cols {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            vec[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        // Result buffer
        var result = [UInt32](repeating: 0, count: rows)

        // Warmup
        ane_tensor_matvec(M, vec, Int32(rows), Int32(cols), &result)

        // Benchmark
        let iters = 100
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            ane_tensor_matvec(M, vec, Int32(rows), Int32(cols), &result)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let totalMul = Int64(rows) * Int64(cols) * 2  // mul + add per element
        let nsPerOp = elapsed * 1e6 / (Double(iters) * Double(totalMul))
        let opsPerSec = 1e9 / nsPerOp
        let mElemPerSec = opsPerSec / 1e6

        print(String(format: "  %4d x %4d: %.2f ms (%d iters) -> %.0f M field ops/s",
                     rows, cols, elapsed / Double(iters), iters, mElemPerSec))
    }

    // ============================================================
    // Batch Matrix-Vector Multiply
    // ============================================================
    print("\n--- Batch Matrix-Vector Multiply (batch x rows x cols) ---")

    let batch_configs = [
        (64, 64, 8),
        (64, 64, 32),
        (128, 128, 4),
        (256, 256, 2),
    ]

    for (rows, cols, batch) in batch_configs {
        let matSize = batch * rows * cols
        let vecSize = batch * cols
        let resSize = batch * rows

        var M = [UInt32](repeating: 0, count: matSize)
        for i in 0..<matSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            M[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var vecs = [UInt32](repeating: 0, count: vecSize)
        for i in 0..<vecSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            vecs[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var result = [UInt32](repeating: 0, count: resSize)

        // Warmup
        ane_tensor_matvec_batch(M, vecs, Int32(rows), Int32(cols), Int32(batch), &result)

        // Benchmark
        let iters = 100
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            ane_tensor_matvec_batch(M, vecs, Int32(rows), Int32(cols), Int32(batch), &result)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let totalOps = Int64(batch) * Int64(rows) * Int64(cols) * 2
        let nsPerOp = elapsed * 1e6 / (Double(iters) * Double(totalOps))
        let opsPerSec = 1e9 / nsPerOp
        let mElemPerSec = opsPerSec / 1e6

        print(String(format: "  batch=%2d, %4d x %4d: %.2f ms -> %.0f M field ops/s",
                     batch, rows, cols, elapsed / Double(iters), mElemPerSec))
    }

    // ============================================================
    // Inner Product: sum = Σ a[i] * b[i]
    // ============================================================
    print("\n--- Inner Product (vector length n) ---")

    let inner_configs = [64, 128, 256, 512, 1024, 2048]

    for n in inner_configs {
        var a = [UInt32](repeating: 0, count: n)
        var b = [UInt32](repeating: 0, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            a[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            b[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var result: UInt32 = 0

        // Warmup
        result = ane_tensor_inner_product(a, b, Int32(n))

        // Benchmark
        let iters = 1000
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            result = ane_tensor_inner_product(a, b, Int32(n))
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let nsPerOp = elapsed * 1e6 / Double(iters)
        let opsPerSec = 1e9 / nsPerOp
        let mElemPerSec = opsPerSec / 1e6

        print(String(format: "  n=%5d: %.3f ms (%d iters) -> %.0f M elem/s",
                     n, elapsed / Double(iters), iters, mElemPerSec))
    }

    // ============================================================
    // Batch Inner Product
    // ============================================================
    print("\n--- Batch Inner Product (batch x n) ---")

    let batch_inner_configs = [
        (256, 8),
        (256, 32),
        (1024, 4),
        (1024, 16),
    ]

    for (n, batch) in batch_inner_configs {
        let aSize = batch * n
        let bSize = batch * n

        var a_batch = [UInt32](repeating: 0, count: aSize)
        var b_batch = [UInt32](repeating: 0, count: bSize)
        for i in 0..<aSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            a_batch[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
            b_batch[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var results = [UInt32](repeating: 0, count: batch)

        // Warmup
        ane_tensor_inner_product_batch(a_batch, b_batch, Int32(n), Int32(batch), &results)

        // Benchmark
        let iters = 500
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            ane_tensor_inner_product_batch(a_batch, b_batch, Int32(n), Int32(batch), &results)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let totalElems = Int64(batch) * Int64(n) * 2
        let nsPerOp = elapsed * 1e6 / (Double(iters) * Double(totalElems))
        let opsPerSec = 1e9 / nsPerOp
        let mElemPerSec = opsPerSec / 1e6

        print(String(format: "  batch=%3d, n=%5d: %.2f ms -> %.0f M elem/s",
                     batch, n, elapsed / Double(iters), mElemPerSec))
    }

    // ============================================================
    // Matrix-Matrix Multiply: C = A * B
    // ============================================================
    print("\n--- Matrix-Matrix Multiply (rowsA x colsA x colsB) ---")

    let matmul_configs = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ]

    for (rowsA, colsA, colsB) in matmul_configs {
        let aSize = rowsA * colsA
        let bSize = colsA * colsB
        let cSize = rowsA * colsB

        var A = [UInt32](repeating: 0, count: aSize)
        var B = [UInt32](repeating: 0, count: bSize)
        for i in 0..<aSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            A[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }
        for i in 0..<bSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            B[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var C = [UInt32](repeating: 0, count: cSize)

        // Warmup
        ane_tensor_matmul(A, B, Int32(rowsA), Int32(colsA), Int32(colsB), &C)

        // Benchmark
        let iters = 50
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            ane_tensor_matmul(A, B, Int32(rowsA), Int32(colsA), Int32(colsB), &C)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let totalMul = Int64(rowsA) * Int64(colsA) * Int64(colsB) * 2
        let nsPerOp = elapsed * 1e6 / (Double(iters) * Double(totalMul))
        let opsPerSec = 1e9 / nsPerOp
        let mElemPerSec = opsPerSec / 1e6

        print(String(format: "  %3d x %3d x %3d: %.2f ms -> %.0f M field ops/s",
                     rowsA, colsA, colsB, elapsed / Double(iters), mElemPerSec))
    }

    // Cleanup
    ane_tensor_shutdown()

    print("\nANE Tensor benchmark complete.")
}
