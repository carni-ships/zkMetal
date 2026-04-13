// poseidon2_benchmark.swift — Comprehensive Poseidon2 BabyBear benchmark
//
// Compares performance of Poseidon2 permutation across:
// 1. CPU Scalar - pure Swift BabyBear arithmetic
// 2. GPU Metal - Metal compute with Poseidon2BabyBearEngine
// 3. ANE - Apple Neural Engine batch permutation
//
// Focuses on full permutation (21 rounds) not just S-box.

import Foundation
import Metal
import ANEOps
import zkMetal

public func runPoseidon2Benchmark() {
    print("=== Poseidon2 BabyBear Comprehensive Benchmark ===\n")

    // ============================================================
    // Initialize ANE
    // ============================================================
    print("--- Initialization ---")
    let aneInitResult = ane_poseidon2_init()
    let aneAvailable = ane_poseidon2_gpu_available()
    print("  ANE init: \(aneInitResult == 0 ? "success" : "failed")")
    print("  ANE GPU available: \(aneAvailable)")
    print("")

    var rng: UInt64 = 0xDEAD_BEEF_C0DE_0055

    // ============================================================
    // CPU Baseline: Pure Swift Poseidon2 permutation
    // ============================================================
    print("--- CPU Scalar Baseline ---")

    // Generate random BabyBear state (16 elements)
    func randomBbState() -> [Bb] {
        var state = [Bb](repeating: Bb.zero, count: 16)
        for i in 0..<16 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let val = UInt32(truncatingIfNeeded: rng) % 2013265921
            state[i] = Bb(v: val)
        }
        return state
    }

    // Warmup
    var warmupState = randomBbState()
    for _ in 0..<1000 {
        _ = poseidon2BbPermutation(warmupState)
    }

    // Benchmark CPU scalar
    let cpuIters = 10000
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<cpuIters {
        warmupState = randomBbState()
        _ = poseidon2BbPermutation(warmupState)
    }
    let cpuElapsed = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000
    let cpuMsPerPerm = cpuElapsed / Double(cpuIters)
    let cpuPermsPerSec = Double(cpuIters) / (cpuElapsed / 1000)

    print(String(format: "  CPU scalar: %.3f ms/perm (%.0f perms/sec)",
                 cpuMsPerPerm, cpuPermsPerSec))
    print(String(format: "  Throughput: %.0f M elem/sec (16 elements/perm)",
                 cpuPermsPerSec * 16 / 1e6))

    // ============================================================
    // GPU Metal: Poseidon2BabyBearEngine
    // ============================================================
    print("\n--- GPU Metal Poseidon2BabyBearEngine ---")

    var gpuMsPerBatch: Double = 0
    var gpuBatchSize: Int = 0
    var gpuElemsPerSec: Double = 0
    var gpuCostPerPerm: Double = 0
    var gpuPermsPerSec: Double = 0

    if let gpuEngine = try? Poseidon2BabyBearEngine() {
        // Prepare batch input (n pairs = n * 16 elements)
        let nPairs = 1024  // 1024 pairs = 16384 elements
        gpuBatchSize = nPairs
        var gpuInput = [Bb](repeating: Bb.zero, count: nPairs * 16)
        for i in 0..<gpuInput.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let val = UInt32(truncatingIfNeeded: rng) % 2013265921
            gpuInput[i] = Bb(v: val)
        }

        // Warmup
        _ = try? gpuEngine.hashPairs(Array(gpuInput[0..<32]))

        // Benchmark GPU hashPairs (2-to-1 compression)
        let gpuIters = 100
        let gpuT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<gpuIters {
            _ = try? gpuEngine.hashPairs(gpuInput)
        }
        let gpuElapsed = (CFAbsoluteTimeGetCurrent() - gpuT0) * 1000
        gpuMsPerBatch = gpuElapsed / Double(gpuIters)
        gpuElemsPerSec = Double(nPairs * 16 * gpuIters) / (gpuElapsed / 1000) / 1e6

        print(String(format: "  GPU Metal (hashPairs, %d pairs): %.3f ms/batch", nPairs, gpuMsPerBatch))
        print(String(format: "  Throughput: %.0f M elem/sec", gpuElemsPerSec))

        // Calculate per-permutation cost
        // hashPairs does 2-to-1 compression, so each output perm compresses 2 inputs
        gpuCostPerPerm = gpuMsPerBatch * 1000 / Double(nPairs)  // in microseconds
        gpuPermsPerSec = 1e6 / gpuCostPerPerm
        print(String(format: "  Per permutation: %.2f us (%.0f perms/sec)",
                     gpuCostPerPerm, gpuPermsPerSec))
    } else {
        print("  GPU Metal: not available (skipping)")
    }

    // ============================================================
    // ANE Batch Permutation
    // ============================================================
    print("\n--- ANE Batch Permutation ---")

    var aneMsPerBatch: Double = 0
    var aneBatchSize: Int = 0
    var aneElemsPerSec: Double = 0
    var aneCostPerPerm: Double = 0
    var anePermsPerSec: Double = 0

    if aneAvailable {
        // Generate batch input for ANE
        let batchSize = 1024
        aneBatchSize = batchSize
        var aneInput = [UInt32](repeating: 0, count: batchSize * 16)
        for i in 0..<aneInput.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            aneInput[i] = UInt32(truncatingIfNeeded: rng) % 2013265921
        }
        var aneOutput = [UInt32](repeating: 0, count: batchSize * 16)

        // Round constants (21 rounds * 16 elements = 336 UInt32)
        let rc = POSEIDON2_BB_ROUND_CONSTANTS
        var flatRC = [UInt32](repeating: 0, count: 336)
        var idx = 0
        for round in rc {
            for elem in round {
                flatRC[idx] = elem.v
                idx += 1
            }
        }

        // Internal diagonal constants (16 UInt32)
        let internalDiag: [UInt32] = [
            0x77ffffff, 0x00000001, 0x00000002, 0x3c000001,
            0x00000003, 0x00000004, 0x3c000000, 0x77fffffe,
            0x77fffffd, 0x77880001, 0x5a000001, 0x69000001,
            0x77fffff2, 0x00780000, 0x07800000, 0x0000000f
        ]

        // Warmup
        aneInput.withUnsafeMutableBufferPointer { inp in
            aneOutput.withUnsafeMutableBufferPointer { outp in
                flatRC.withUnsafeBufferPointer { rcp in
                    internalDiag.withUnsafeBufferPointer { diagp in
                        bb_poseidon2_permutation_batch_ane(
                            inp.baseAddress!, Int32(batchSize),
                            rcp.baseAddress!, diagp.baseAddress!,
                            outp.baseAddress!)
                    }
                }
            }
        }

        // Benchmark ANE batch permutation
        let aneIters = 100
        let aneT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<aneIters {
            aneInput.withUnsafeMutableBufferPointer { inp in
                aneOutput.withUnsafeMutableBufferPointer { outp in
                    flatRC.withUnsafeBufferPointer { rcp in
                        internalDiag.withUnsafeBufferPointer { diagp in
                            bb_poseidon2_permutation_batch_ane(
                                inp.baseAddress!, Int32(batchSize),
                                rcp.baseAddress!, diagp.baseAddress!,
                                outp.baseAddress!)
                        }
                    }
                }
            }
        }
        let aneElapsed = (CFAbsoluteTimeGetCurrent() - aneT0) * 1000
        aneMsPerBatch = aneElapsed / Double(aneIters)
        aneElemsPerSec = Double(batchSize * 16 * aneIters) / (aneElapsed / 1000) / 1e6

        print(String(format: "  ANE batch (%d perms): %.3f ms/batch", batchSize, aneMsPerBatch))
        print(String(format: "  Throughput: %.0f M elem/sec", aneElemsPerSec))

        aneCostPerPerm = aneMsPerBatch * 1000 / Double(batchSize)  // in microseconds
        anePermsPerSec = 1e6 / aneCostPerPerm
        print(String(format: "  Per permutation: %.2f us (%.0f perms/sec)",
                     aneCostPerPerm, anePermsPerSec))
    } else {
        print("  ANE GPU not available (ANE init returned \(aneInitResult))")
    }

    // ============================================================
    // Summary comparison
    // ============================================================
    print("\n=== Summary ===")
    print("Configuration: BabyBear Poseidon2, width=16, 21 rounds")
    print("")

    // CPU baseline
    let cpuUs = cpuMsPerPerm * 1000
    print(String(format: "CPU scalar:      %8.2f us/perm  (%8.0f perms/sec)",
                 cpuUs, cpuPermsPerSec))

    // GPU Metal
    if gpuPermsPerSec > 0 {
        print(String(format: "GPU Metal:       %8.2f us/perm  (%8.0f perms/sec)",
                     gpuCostPerPerm, gpuPermsPerSec))
    }

    // ANE
    if anePermsPerSec > 0 {
        print(String(format: "ANE batch:       %8.2f us/perm  (%8.0f perms/sec)",
                     aneCostPerPerm, anePermsPerSec))
    }

    // Speedup summary
    print("")
    if gpuPermsPerSec > 0 {
        let gpuSpeedup = cpuUs / gpuCostPerPerm
        print(String(format: "GPU Metal speedup vs CPU: %.1fx", gpuSpeedup))
    }
    if anePermsPerSec > 0 {
        let aneSpeedup = cpuUs / aneCostPerPerm
        print(String(format: "ANE speedup vs CPU:       %.1fx", aneSpeedup))
    }

    // ANE vs GPU comparison
    if gpuPermsPerSec > 0 && anePermsPerSec > 0 {
        let aneVsGpu = aneCostPerPerm / gpuCostPerPerm
        print(String(format: "ANE vs GPU Metal:        %.2fx (ANE/GPU)",
                     aneVsGpu))
    }

    print("\nBenchmark complete.")
}
