// GPU Radix Sort Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runSortBench() {
    fputs("\n=== GPU Radix Sort ===\n", stderr)

    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        let engine = try RadixSortEngine()

        // Test 1: Small sorted input
        let sorted = try engine.sort([1, 2, 3, 4, 5, 6, 7, 8])
        fputs("  Already sorted: \(sorted == [1,2,3,4,5,6,7,8] ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Reverse sorted
        let reversed = try engine.sort([8, 7, 6, 5, 4, 3, 2, 1])
        fputs("  Reverse sorted: \(reversed == [1,2,3,4,5,6,7,8] ? "PASS" : "FAIL")\n", stderr)

        // Test 3: Duplicates
        let dups = try engine.sort([3, 1, 4, 1, 5, 9, 2, 6])
        fputs("  With duplicates: \(dups == [1,1,2,3,4,5,6,9] ? "PASS" : "FAIL")\n", stderr)

        // Test 4a: Boundary test (exactly at GPU threshold)
        let boundary = 4096
        var boundaryKeys = [UInt32]()
        for i in (0..<boundary).reversed() {
            boundaryKeys.append(UInt32(i))
        }
        let boundSorted = try engine.sort(boundaryKeys)
        let boundExpected = boundaryKeys.sorted()
        let boundPass = boundSorted == boundExpected
        if !boundPass {
            // Show first mismatches
            for i in 0..<min(10, boundary) {
                if boundSorted[i] != boundExpected[i] {
                    fputs("    mismatch at \(i): got \(boundSorted[i]) expected \(boundExpected[i])\n", stderr)
                }
            }
        }
        fputs("  Boundary 4096: \(boundPass ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Random array — compare with CPU sort
        var rng: UInt64 = 0xDEAD_BEEF_CAFE
        let n = 10000
        var randomKeys = [UInt32]()
        randomKeys.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            randomKeys.append(UInt32(truncatingIfNeeded: rng >> 32))
        }
        let gpuSorted = try engine.sort(randomKeys)
        let cpuSorted = randomKeys.sorted()
        fputs("  Random 10K: \(gpuSorted == cpuSorted ? "PASS" : "FAIL")\n", stderr)

        // Test 5: Key-value sort
        let kvKeys: [UInt32] = [30, 10, 20, 40]
        let kvVals: [UInt32] = [300, 100, 200, 400]
        let (sortedK, sortedV) = try engine.sortKV(keys: kvKeys, values: kvVals)
        let kvPass = sortedK == [10, 20, 30, 40] && sortedV == [100, 200, 300, 400]
        fputs("  Key-value sort: \(kvPass ? "PASS" : "FAIL")\n", stderr)

        // Test 6: Larger random KV sort
        var kvKeys2 = [UInt32]()
        var kvVals2 = [UInt32]()
        for i in 0..<8192 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            kvKeys2.append(UInt32(truncatingIfNeeded: rng >> 32))
            kvVals2.append(UInt32(i))
        }
        let (sk2, sv2) = try engine.sortKV(keys: kvKeys2, values: kvVals2)
        // Verify keys are sorted and values track correctly
        var kvCorrect = true
        for i in 1..<sk2.count {
            if sk2[i] < sk2[i-1] { kvCorrect = false; break }
        }
        // Verify values map back to original keys
        if kvCorrect {
            for i in 0..<sk2.count {
                let origIdx = Int(sv2[i])
                if kvKeys2[origIdx] != sk2[i] { kvCorrect = false; break }
            }
        }
        fputs("  KV sort 8K: \(kvCorrect ? "PASS" : "FAIL")\n", stderr)

        // Test 7: Edge cases
        let empty = try engine.sort([UInt32]())
        fputs("  Empty: \(empty.isEmpty ? "PASS" : "FAIL")\n", stderr)
        let single = try engine.sort([42])
        fputs("  Single: \(single == [42] ? "PASS" : "FAIL")\n", stderr)

        // Test 8: Full range (keys spanning all 32 bits)
        let fullRange: [UInt32] = [0, UInt32.max, UInt32.max / 2, 1, UInt32.max - 1]
        let frSorted = try engine.sort(fullRange)
        fputs("  Full 32-bit range: \(frSorted == fullRange.sorted() ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Performance ---
    if !skipCPU {
        fputs("\n--- Performance ---\n", stderr)
        do {
            let engine = try RadixSortEngine()

            for logN in [14, 16, 18, 20, 22] {
                let n = 1 << logN
                var rng: UInt64 = UInt64(logN) &* 0xCAFE_BABE
                var keys = [UInt32]()
                keys.reserveCapacity(n)
                for _ in 0..<n {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    keys.append(UInt32(truncatingIfNeeded: rng >> 32))
                }

                // Warmup
                let _ = try engine.sort(keys)

                // GPU timed
                let runs = 3
                var gpuTimes = [Double]()
                for _ in 0..<runs {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.sort(keys)
                    gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                gpuTimes.sort()
                let gpuMedian = gpuTimes[runs / 2]

                // CPU timed
                let t1 = CFAbsoluteTimeGetCurrent()
                let _ = keys.sorted()
                let cpuTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

                let speedup = cpuTime / max(gpuMedian, 0.001)
                fputs("  2^\(logN) = \(n) keys: GPU \(String(format: "%.1f", gpuMedian))ms, CPU \(String(format: "%.1f", cpuTime))ms, \(String(format: "%.1f", speedup))×\n", stderr)
            }
        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }
}
