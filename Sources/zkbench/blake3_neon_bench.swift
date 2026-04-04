// Blake3 NEON Benchmark — CPU NEON vs GPU parent-node hashing
import zkMetal
import Foundation
import NeonFieldOps

public func runBlake3NeonBench() {
    print("=== Blake3 NEON Benchmark ===")

    // --- Correctness: compare NEON against CPU Swift reference ---
    // Test with known data: zeros
    var left = [UInt8](repeating: 0, count: 32)
    var right = [UInt8](repeating: 0, count: 32)
    var neonOut = [UInt8](repeating: 0, count: 32)

    blake3_hash_pair_neon(&left, &right, &neonOut)
    let swiftRef = blake3Parent(left + right)

    let neonHex = neonOut.map { String(format: "%02x", $0) }.joined()
    let swiftHex = swiftRef.map { String(format: "%02x", $0) }.joined()
    print("  NEON parent(0,0):  \(neonHex)")
    print("  Swift parent(0,0): \(swiftHex)")
    if neonOut == swiftRef {
        print("  [pass] Zeros test")
    } else {
        print("  [FAIL] Zeros test - mismatch!")
        return
    }

    // Test with sequential data
    for i in 0..<32 { left[i] = UInt8(i) }
    for i in 0..<32 { right[i] = UInt8(32 + i) }
    blake3_hash_pair_neon(&left, &right, &neonOut)
    let swiftRef2 = blake3Parent(left + right)
    if neonOut == swiftRef2 {
        print("  [pass] Sequential data test")
    } else {
        print("  [FAIL] Sequential data test")
        let nH = neonOut.map { String(format: "%02x", $0) }.joined()
        let sH = swiftRef2.map { String(format: "%02x", $0) }.joined()
        print("    NEON:  \(nH)")
        print("    Swift: \(sH)")
        return
    }

    // Test batch against individual
    let batchN = 100
    var batchInput = [UInt8](repeating: 0, count: batchN * 64)
    var rng: UInt64 = 0xB1A3E_CAFE_BABE
    for i in 0..<batchInput.count {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        batchInput[i] = UInt8(truncatingIfNeeded: rng >> 32)
    }
    var batchOutput = [UInt8](repeating: 0, count: batchN * 32)
    batchInput.withUnsafeBufferPointer { inp in
        batchOutput.withUnsafeMutableBufferPointer { out in
            blake3_batch_hash_pairs_neon(inp.baseAddress!, out.baseAddress!, batchN)
        }
    }

    var batchCorrect = true
    for i in 0..<batchN {
        let ref = blake3Parent(Array(batchInput[i*64..<(i+1)*64]))
        let got = Array(batchOutput[i*32..<(i+1)*32])
        if ref != got {
            print("  [FAIL] Batch pair \(i) mismatch")
            batchCorrect = false
            break
        }
    }
    if batchCorrect {
        print("  [pass] Batch \(batchN) pairs matches Swift reference")
    }

    // --- Correctness vs GPU ---
    do {
        let engine = try Blake3Engine()
        let gpuOut = try engine.hashParents(batchInput)
        var gpuMatch = true
        for i in 0..<batchN {
            let neon = Array(batchOutput[i*32..<(i+1)*32])
            let gpu = Array(gpuOut[i*32..<(i+1)*32])
            if neon != gpu {
                print("  [FAIL] NEON vs GPU mismatch at pair \(i)")
                let nH = neon.prefix(8).map { String(format: "%02x", $0) }.joined()
                let gH = gpu.prefix(8).map { String(format: "%02x", $0) }.joined()
                print("    NEON: \(nH)...")
                print("    GPU:  \(gH)...")
                gpuMatch = false
                break
            }
        }
        if gpuMatch {
            print("  [pass] NEON matches GPU for \(batchN) pairs")
        }
    } catch {
        print("  [skip] GPU comparison: \(error)")
    }

    // --- Single pair benchmark: NEON ---
    print("\n--- Single Pair Benchmark ---")
    // Warmup
    for _ in 0..<10000 {
        blake3_hash_pair_neon(&left, &right, &neonOut)
    }
    let singleIters = 100_000
    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<singleIters {
        blake3_hash_pair_neon(&left, &right, &neonOut)
    }
    let neonSingleTime = (CFAbsoluteTimeGetCurrent() - t0)
    let neonUsPerHash = neonSingleTime / Double(singleIters) * 1e6
    print(String(format: "  NEON: %.2f us/hash (%.0f hash/s)", neonUsPerHash,
                 Double(singleIters) / neonSingleTime))

    // Swift CPU reference
    if !skipCPU {
        let swiftInput = left + right
        for _ in 0..<5000 { let _ = blake3Parent(swiftInput) }
        let swiftIters = 10_000
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<swiftIters { let _ = blake3Parent(swiftInput) }
        let swiftTime = (CFAbsoluteTimeGetCurrent() - t1)
        let swiftUs = swiftTime / Double(swiftIters) * 1e6
        let speedup = swiftUs / neonUsPerHash
        print(String(format: "  Swift CPU: %.2f us/hash (%.1fx slower)", swiftUs, speedup))
    }

    // --- Batch benchmark: NEON vs GPU ---
    print("\n--- Batch Benchmark (NEON vs GPU) ---")
    for logN in [10, 12, 14, 16, 18] {
        let n = 1 << logN
        var input = [UInt8](repeating: 0, count: n * 64)
        var r: UInt64 = 0xDEAD_BEEF
        for i in stride(from: 0, to: input.count, by: 8) {
            r = r &* 6364136223846793005 &+ 1442695040888963407
            for b in 0..<min(8, input.count - i) {
                input[i + b] = UInt8((r >> (b * 8)) & 0xFF)
            }
        }
        var output = [UInt8](repeating: 0, count: n * 32)

        // NEON warmup + timed
        input.withUnsafeBufferPointer { inp in
            output.withUnsafeMutableBufferPointer { out in
                blake3_batch_hash_pairs_neon(inp.baseAddress!, out.baseAddress!, n)
            }
        }
        var neonTimes = [Double]()
        for _ in 0..<5 {
            let start = CFAbsoluteTimeGetCurrent()
            input.withUnsafeBufferPointer { inp in
                output.withUnsafeMutableBufferPointer { out in
                    blake3_batch_hash_pairs_neon(inp.baseAddress!, out.baseAddress!, n)
                }
            }
            neonTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        neonTimes.sort()
        let neonMedian = neonTimes[2]

        // GPU
        var gpuMedianStr = "N/A"
        do {
            let engine = try Blake3Engine()
            let _ = try engine.hashParents(input) // warmup
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashParents(input)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[2]
            let ratio = neonMedian / gpuMedian
            gpuMedianStr = String(format: "%.2f ms (NEON/GPU = %.1fx)", gpuMedian, ratio)
        } catch {
            gpuMedianStr = "error"
        }

        print(String(format: "  2^%-2d = %7d | NEON %8.2f ms | GPU %@",
                     logN, n, neonMedian, gpuMedianStr))
    }
}
