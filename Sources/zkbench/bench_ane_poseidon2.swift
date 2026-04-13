// bench_ane_poseidon2.swift — Poseidon2 S-box ANE benchmarks
//
// Benchmarks the BabyBear (x^7) and M31 (x^5) Poseidon2 S-boxes
// using the scalar fallback paths (ANE not yet implemented).
//
// Results establish baseline for future ANE speedup comparisons.

import ANEOps
import Foundation

public func runANEPoseidon2Bench() {
    print("=== ANE Poseidon2 S-box Benchmark (scalar fallback) ===")
    print("Note: ANE returns -1 (not available), benchmarking scalar paths.\n")

    let warmup = 1000

    // ============================================================
    // BabyBear x^7 S-box (16 elements per permutation state)
    // ============================================================
    print("--- BabyBear x^7 S-box (16 elements) ---")

    // Generate test data: 16 BabyBear elements
    var stateBB = [UInt32](repeating: 0, count: 16)
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    for i in 0..<16 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        stateBB[i] = UInt32(truncatingIfNeeded: rng) % 2013265921  // BabyBear prime
    }

    // Warmup
    for _ in 0..<warmup {
        bb_poseidon2_sbox_ane(&stateBB)
    }

    // Benchmark single S-box application (16 elements)
    let iters = 50000
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        bb_poseidon2_sbox_ane(&stateBB)
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
    let usPerSbox = elapsed / Double(iters) * 1000
    let opsPerSec = Double(iters) / (elapsed / 1000)
    print(String(format: "  16x BabyBear x^7: %.2f us (%.0f ops/sec)",
                 usPerSbox, opsPerSec))

    // ============================================================
    // M31 x^5 S-box (16 elements per permutation state)
    // ============================================================
    print("\n--- M31 x^5 S-box (16 elements) ---")

    // Generate test data: 16 M31 elements
    var stateM31 = [UInt32](repeating: 0, count: 16)
    rng = 0xCAFE_BABE_DEAD_BEEF
    for i in 0..<16 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        stateM31[i] = UInt32(truncatingIfNeeded: rng) % 2147483647  // M31 prime
    }

    // Warmup
    for _ in 0..<warmup {
        m31_poseidon2_sbox_ane(&stateM31)
    }

    // Benchmark single S-box application (16 elements)
    let startM31 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        m31_poseidon2_sbox_ane(&stateM31)
    }
    let elapsedM31 = (CFAbsoluteTimeGetCurrent() - startM31) * 1000
    let usPerSboxM31 = elapsedM31 / Double(iters) * 1000
    let opsPerSecM31 = Double(iters) / (elapsedM31 / 1000)
    print(String(format: "  16x M31 x^5: %.2f us (%.0f ops/sec)",
                 usPerSboxM31, opsPerSecM31))

    // ============================================================
    // Per-element throughput analysis
    // ============================================================
    print("\n--- Per-element throughput ---")
    let elementsPerSbox = 16
    let nsPerElementBB = usPerSbox * 1000 / Double(elementsPerSbox)
    let nsPerElementM31 = usPerSboxM31 * 1000 / Double(elementsPerSbox)
    print(String(format: "  BabyBear x^7: %.0f ns/element (%.0f M elem/s)",
                 nsPerElementBB, 1e9 / nsPerElementBB / 1e6))
    print(String(format: "  M31 x^5:     %.0f ns/element (%.0f M elem/s)",
                 nsPerElementM31, 1e9 / nsPerElementM31 / 1e6))

    // ============================================================
    // Batch S-box benchmarks (1..256 elements)
    // ============================================================
    print("\n--- Batch S-box: 1..256 elements ---")

    for count in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        // Prepare input buffer
        var input = [UInt32](repeating: 0, count: count * 16)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        for i in 0..<input.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }

        var output = [UInt32](repeating: 0, count: count * 16)
        let countInt32 = Int32(count)

        // Warmup
        input.withUnsafeMutableBufferPointer { inp in
            output.withUnsafeMutableBufferPointer { out in
                bb_poseidon2_sbox_batch_ane(inp.baseAddress!, countInt32, out.baseAddress!)
            }
        }

        // Benchmark
        let batchIters = 10000
        let batchStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<batchIters {
            input.withUnsafeMutableBufferPointer { inp in
                output.withUnsafeMutableBufferPointer { out in
                    bb_poseidon2_sbox_batch_ane(inp.baseAddress!, countInt32, out.baseAddress!)
                }
            }
        }
        let batchElapsed = (CFAbsoluteTimeGetCurrent() - batchStart) * 1000
        let totalElements = Double(count * 16 * batchIters)
        let nsPerElem = batchElapsed * 1e6 / totalElements
        print(String(format: "  BB batch %3d x 16 = %5d elem: %.2f ms (%.0f ns/elem, %.0f M elem/s)",
                     count, count * 16, batchElapsed / Double(batchIters),
                     nsPerElem, 1e9 / nsPerElem / 1e6))
    }

    print("\nPoseidon2 ANE benchmark complete.")
    print("Baseline established for future ANE speedup comparisons.")
}
