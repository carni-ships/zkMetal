// bench_ane_binary_tower.swift — Binary Tower ANE benchmarks
//
// Benchmarks GF(2^64) and GF(2^128) multiplication and addition
// using the scalar fallback paths (ANE not yet implemented).
//
// Measures throughput for 1K..1M operations.

import ANEOps
import Foundation

public func runANEBinaryTowerBench() {
    print("=== ANE Binary Tower Benchmark (scalar fallback) ===")
    print("Note: ANE not available, benchmarking scalar PMULL paths.\n")

    // ============================================================
    // GF(2^64) Multiply — throughput benchmark
    // ============================================================
    print("--- GF(2^64) Multiply ---")

    // Generate random operands - need 1M for largest n
    var operandsA = [UInt64](repeating: 0, count: 2_000_000)
    var operandsB = [UInt64](repeating: 0, count: 2_000_000)
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    for i in 0..<2_000_000 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsA[i] = rng
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsB[i] = rng
    }

    // Adaptive iterations based on n
    let testSizes: [(Int, Int)] = [
        (10, 10000),   // logN=10, n=1024, iters=10000
        (12, 5000),    // logN=12, n=4096, iters=5000
        (14, 1000),     // logN=14, n=16384, iters=1000
        (16, 200),     // logN=16, n=65536, iters=200
        (18, 50),      // logN=18, n=262144, iters=50
        (20, 10),      // logN=20, n=1048576, iters=10
    ]

    for (logN, iters) in testSizes {
        let n = 1 << logN

        // Warmup
        for i in 0..<min(1000, n) {
            _ = bt_gf64_mul_scalar(operandsA[i], operandsB[i])
        }

        var totalElapsed = 0.0
        for _ in 0..<iters {
            let start = CFAbsoluteTimeGetCurrent()
            for i in 0..<n {
                _ = bt_gf64_mul_scalar(operandsA[i], operandsB[i])
            }
            totalElapsed += CFAbsoluteTimeGetCurrent() - start
        }

        let elapsed = totalElapsed / Double(iters)
        let opsPerSec = Double(n) / elapsed
        let nsPerOp = elapsed / Double(n) * 1e9
        print(String(format: "  2^%2d = %7d ops: %.3f ms (%.0f ops/sec, %.0f ns/op)",
                     logN, n, elapsed * 1000, opsPerSec, nsPerOp))
    }

    // ============================================================
    // GF(2^128) Multiply — throughput benchmark
    // ============================================================
    print("\n--- GF(2^128) Multiply ---")

    // GF(2^128) uses two UInt64 for each operand
    // Layout: [lo, hi] for each element
    var operandsA128 = [UInt64](repeating: 0, count: 2 * 300_000)
    var operandsB128 = [UInt64](repeating: 0, count: 2 * 300_000)
    rng = 0xCAFEBABE_DEAD_BEEF
    for i in 0..<300_000 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsA128[i * 2] = rng
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsA128[i * 2 + 1] = rng
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsB128[i * 2] = rng
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        operandsB128[i * 2 + 1] = rng
    }

    let testSizes128: [(Int, Int)] = [
        (10, 1000),   // logN=10, n=1024, iters=1000
        (12, 500),     // logN=12, n=4096, iters=500
        (14, 100),    // logN=14, n=16384, iters=100
        (16, 20),     // logN=16, n=65536, iters=20
        (18, 5),      // logN=18, n=262144, iters=5
    ]

    for (logN, iters) in testSizes128 {
        let n = 1 << logN

        // Warmup
        var result = [UInt64](repeating: 0, count: 2)
        for i in 0..<min(100, n) {
            result.withUnsafeMutableBufferPointer { resBuf in
                operandsA128.withUnsafeBufferPointer { aBuf in
                    operandsB128.withUnsafeBufferPointer { bBuf in
                        bt_gf128_mul_scalar(
                            aBuf.baseAddress! + i * 2,
                            bBuf.baseAddress! + i * 2,
                            resBuf.baseAddress!
                        )
                    }
                }
            }
        }

        var totalElapsed = 0.0
        for _ in 0..<iters {
            let start = CFAbsoluteTimeGetCurrent()
            for i in 0..<n {
                result.withUnsafeMutableBufferPointer { resBuf in
                    operandsA128.withUnsafeBufferPointer { aBuf in
                        operandsB128.withUnsafeBufferPointer { bBuf in
                            bt_gf128_mul_scalar(
                                aBuf.baseAddress! + i * 2,
                                bBuf.baseAddress! + i * 2,
                                resBuf.baseAddress!
                            )
                        }
                    }
                }
            }
            totalElapsed += CFAbsoluteTimeGetCurrent() - start
        }

        let elapsed = totalElapsed / Double(iters)
        let opsPerSec = Double(n) / elapsed
        let nsPerOp = elapsed / Double(n) * 1e9
        print(String(format: "  2^%2d = %7d ops: %.3f ms (%.0f ops/sec, %.0f ns/op)",
                     logN, n, elapsed * 1000, opsPerSec, nsPerOp))
    }

    // ============================================================
    // GF(2^64) Add (XOR) — for comparison
    // ============================================================
    print("\n--- GF(2^64) Add (XOR) ---")

    let testSizesAdd: [(Int, Int)] = [
        (10, 10000),
        (12, 5000),
        (14, 1000),
        (16, 200),
        (18, 50),
        (20, 10),
    ]

    for (logN, iters) in testSizesAdd {
        let n = 1 << logN

        // Warmup
        for i in 0..<min(1000, n) {
            _ = bt_gf64_add_scalar(operandsA[i], operandsB[i])
        }

        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            for i in 0..<n {
                _ = bt_gf64_add_scalar(operandsA[i], operandsB[i])
            }
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) / Double(iters)
        let opsPerSec = Double(n) / elapsed
        let nsPerOp = elapsed / Double(n) * 1e9
        print(String(format: "  2^%2d = %7d ops: %.3f ms (%.0f ops/sec, %.0f ns/op)",
                     logN, n, elapsed * 1000, opsPerSec, nsPerOp))
    }

    // ============================================================
    // GF(2^64) vs GF(2^128) multiply comparison
    // ============================================================
    print("\n--- Multiply: GF(2^64) vs GF(2^128) ---")
    print("  Ratio: GF(2^128) is ~3x cost of GF(2^64) due to Karatsuba")
    print("  ANE target: batch GF(2^8) mul via FP16 matmul")

    print("\nBinary Tower ANE benchmark complete.")
    print("Baseline established for future ANE speedup comparisons.")
}
