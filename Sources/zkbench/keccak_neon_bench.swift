// Keccak-256 NEON CPU Benchmark
import zkMetal
import NeonFieldOps
import Foundation

public func runKeccakNeonBench() {
    print("=== Keccak-256 NEON CPU Benchmark ===")

    // --- Correctness: hash of empty string ---
    var emptyOut = [UInt8](repeating: 0, count: 32)
    keccak256_hash_neon(nil, 0, &emptyOut)
    let emptyHex = emptyOut.map { String(format: "%02x", $0) }.joined()
    let expected = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    print("  keccak256_neon(\"\") = \(emptyHex)")
    if emptyHex == expected {
        print("  [pass] Empty hash matches known value")
    } else {
        print("  [FAIL] Expected \(expected)")
        return
    }

    // --- Correctness: hash of "abc" ---
    let abc: [UInt8] = [0x61, 0x62, 0x63]
    var abcOut = [UInt8](repeating: 0, count: 32)
    abc.withUnsafeBufferPointer { buf in
        keccak256_hash_neon(buf.baseAddress, 3, &abcOut)
    }
    let abcHex = abcOut.map { String(format: "%02x", $0) }.joined()
    let expectedAbc = "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    print("  keccak256_neon(\"abc\") = \(abcHex)")
    if abcHex == expectedAbc {
        print("  [pass] abc hash matches known value")
    } else {
        print("  [FAIL] Expected \(expectedAbc)")
        return
    }

    // --- Correctness: hash_pair matches hash of 64 concatenated bytes ---
    var rng: UInt64 = 0xCAFEBABE_DEADBEEF
    var testA = [UInt8](repeating: 0, count: 32)
    var testB = [UInt8](repeating: 0, count: 32)
    for i in 0..<32 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        testA[i] = UInt8(truncatingIfNeeded: rng >> 32)
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        testB[i] = UInt8(truncatingIfNeeded: rng >> 32)
    }

    var pairOut = [UInt8](repeating: 0, count: 32)
    testA.withUnsafeBufferPointer { aPtr in
        testB.withUnsafeBufferPointer { bPtr in
            keccak256_hash_pair_neon(aPtr.baseAddress, bPtr.baseAddress, &pairOut)
        }
    }

    // Compare with generic hash of concatenated input
    let concat = testA + testB
    var concatOut = [UInt8](repeating: 0, count: 32)
    concat.withUnsafeBufferPointer { buf in
        keccak256_hash_neon(buf.baseAddress, 64, &concatOut)
    }

    if pairOut == concatOut {
        print("  [pass] hash_pair matches hash of concatenated input")
    } else {
        print("  [FAIL] hash_pair mismatch")
        print("    pair:   \(pairOut.map { String(format: "%02x", $0) }.joined())")
        print("    concat: \(concatOut.map { String(format: "%02x", $0) }.joined())")
        return
    }

    // --- Correctness: compare with existing Swift keccak256 ---
    let swiftRef = keccak256(concat)
    if pairOut == swiftRef {
        print("  [pass] NEON output matches Swift CPU reference")
    } else {
        print("  [FAIL] NEON vs Swift mismatch")
        print("    NEON:  \(pairOut.map { String(format: "%02x", $0) }.joined())")
        print("    Swift: \(swiftRef.map { String(format: "%02x", $0) }.joined())")
        return
    }

    // --- Correctness: batch hash_pairs ---
    let batchN = 100
    var batchInputs = [UInt8](repeating: 0, count: batchN * 64)
    for i in 0..<batchInputs.count {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        batchInputs[i] = UInt8(truncatingIfNeeded: rng >> 32)
    }
    var batchOutputs = [UInt8](repeating: 0, count: batchN * 32)
    batchInputs.withUnsafeBufferPointer { inBuf in
        batchOutputs.withUnsafeMutableBufferPointer { outBuf in
            keccak256_batch_hash_pairs_neon(inBuf.baseAddress, outBuf.baseAddress, batchN)
        }
    }

    // Verify each pair
    var batchCorrect = true
    for i in 0..<batchN {
        let ref = keccak256(Array(batchInputs[i*64..<(i+1)*64]))
        let got = Array(batchOutputs[i*32..<(i+1)*32])
        if ref != got {
            print("  [FAIL] Batch pair \(i) mismatch")
            batchCorrect = false
            break
        }
    }
    if batchCorrect {
        print("  [pass] Batch hash_pairs (\(batchN) pairs) matches Swift reference")
    } else {
        return
    }

    // --- Benchmark: single hash_pair ---
    print("\n  --- Single hash_pair benchmark ---")
    let warmup = 10000
    var dummy = [UInt8](repeating: 0, count: 32)
    for _ in 0..<warmup {
        keccak256_hash_pair_neon(testA, testB, &dummy)
    }

    let singleIters = 100000
    let singleStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<singleIters {
        keccak256_hash_pair_neon(testA, testB, &dummy)
    }
    let singleElapsed = (CFAbsoluteTimeGetCurrent() - singleStart) * 1000
    let usPerHash = singleElapsed / Double(singleIters) * 1000
    let hashesPerSec = Double(singleIters) / (singleElapsed / 1000)
    print(String(format: "  NEON hash_pair: %.3f us/hash (%.0f hash/s)", usPerHash, hashesPerSec))

    // Compare with Swift CPU reference
    for _ in 0..<warmup {
        let _ = keccak256(concat)
    }
    let swiftIters = 10000
    let swiftStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<swiftIters {
        let _ = keccak256(concat)
    }
    let swiftElapsed = (CFAbsoluteTimeGetCurrent() - swiftStart) * 1000
    let swiftUs = swiftElapsed / Double(swiftIters) * 1000
    print(String(format: "  Swift CPU ref:  %.3f us/hash (%.0f hash/s)", swiftUs, Double(swiftIters) / (swiftElapsed / 1000)))
    print(String(format: "  Speedup: %.1fx", swiftUs / usPerHash))

    // --- Benchmark: batch hash_pairs ---
    print("\n  --- Batch hash_pairs benchmark ---")
    for logN in [10, 12, 14, 16] {
        let n = 1 << logN
        var inputs = [UInt8](repeating: 0, count: n * 64)
        for i in 0..<inputs.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            inputs[i] = UInt8(truncatingIfNeeded: rng >> 32)
        }
        var outputs = [UInt8](repeating: 0, count: n * 32)

        // Warmup
        inputs.withUnsafeBufferPointer { inBuf in
            outputs.withUnsafeMutableBufferPointer { outBuf in
                keccak256_batch_hash_pairs_neon(inBuf.baseAddress, outBuf.baseAddress, n)
            }
        }

        let iters = max(1, 10000 / n)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            inputs.withUnsafeBufferPointer { inBuf in
                outputs.withUnsafeMutableBufferPointer { outBuf in
                    keccak256_batch_hash_pairs_neon(inBuf.baseAddress, outBuf.baseAddress, n)
                }
            }
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iters)
        let perHash = elapsed / Double(n) * 1000
        print(String(format: "  2^%d (%6d pairs): %8.2f ms  (%.3f us/hash)", logN, n, elapsed, perHash))
    }

    // --- Compare with GPU ---
    do {
        let engine = try Keccak256Engine()
        print("\n  --- GPU comparison ---")
        for logN in [10, 12, 14, 16] {
            let n = 1 << logN
            var inputs = [UInt8](repeating: 0, count: n * 64)
            for i in 0..<inputs.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                inputs[i] = UInt8(truncatingIfNeeded: rng >> 32)
            }

            // GPU warmup
            let _ = try engine.hash64(inputs)

            let gpuIters = max(1, 1000 / n)
            let gpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<gpuIters {
                let _ = try engine.hash64(inputs)
            }
            let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000 / Double(gpuIters)

            // NEON
            var outputs = [UInt8](repeating: 0, count: n * 32)
            let cpuIters = max(1, 10000 / n)
            let cpuStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<cpuIters {
                inputs.withUnsafeBufferPointer { inBuf in
                    outputs.withUnsafeMutableBufferPointer { outBuf in
                        keccak256_batch_hash_pairs_neon(inBuf.baseAddress, outBuf.baseAddress, n)
                    }
                }
            }
            let cpuMs = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000 / Double(cpuIters)

            print(String(format: "  2^%d: NEON %.2f ms  GPU %.2f ms  (GPU %.1fx)", logN, cpuMs, gpuMs, cpuMs / gpuMs))
        }
    } catch {
        print("  GPU unavailable: \(error)")
    }
}
