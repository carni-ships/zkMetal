// GPU Radix Sort Benchmark
import zkMetal
import Foundation

public func runSortBench() {
    print("=== GPU Radix Sort Benchmark ===")

    do {
        let engine = try RadixSortEngine()

        let sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]

        for n in sizes {
            // Generate random keys
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
            var keys = [UInt32](repeating: 0, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                keys[i] = UInt32(truncatingIfNeeded: rng >> 32)
            }

            // Warmup
            let _ = try engine.sortKeys(keys)

            // GPU timed
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.sortKeys(keys)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                gpuTimes.append(elapsed)
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[2]

            // CPU reference for smaller sizes
            var cpuMedian: Double = 0
            if n <= (1 << 20) {
                var cpuTimes = [Double]()
                for _ in 0..<3 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = RadixSortEngine.cpuRadixSort(keys)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    cpuTimes.append(elapsed)
                }
                cpuTimes.sort()
                cpuMedian = cpuTimes[1]
            }

            let keysPerSec = Double(n) / (gpuMedian / 1000)
            if cpuMedian > 0 {
                print(String(format: "  n=%7d | GPU: %7.2fms | CPU: %7.2fms | %.1fx | %.0fM keys/s",
                            n, gpuMedian, cpuMedian, cpuMedian / gpuMedian, keysPerSec / 1e6))
            } else {
                print(String(format: "  n=%7d | GPU: %7.2fms | %.0fM keys/s",
                            n, gpuMedian, keysPerSec / 1e6))
            }
        }

        // Correctness check
        print("\n--- Correctness verification ---")
        var rng: UInt64 = 0x12345678
        let testN = 10000
        var testKeys = [UInt32](repeating: 0, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testKeys[i] = UInt32(truncatingIfNeeded: rng >> 32)
        }

        let gpuSorted = try engine.sortKeys(testKeys)
        let cpuSorted = RadixSortEngine.cpuRadixSort(testKeys)

        var correct = true
        for i in 0..<testN {
            if gpuSorted[i] != cpuSorted[i] {
                print("  MISMATCH at \(i): gpu=\(gpuSorted[i]), cpu=\(cpuSorted[i])")
                correct = false
                break
            }
        }
        print("  Sort correctness: \(correct ? "PASS" : "FAIL")")

        // Also verify sorted order
        var ordered = true
        for i in 1..<testN {
            if gpuSorted[i] < gpuSorted[i - 1] {
                print("  NOT SORTED at \(i): \(gpuSorted[i-1]) > \(gpuSorted[i])")
                ordered = false
                break
            }
        }
        print("  Sorted order: \(ordered ? "PASS" : "FAIL")")

        // Key-value sort test
        let kvN = 1000
        var kvKeys = [UInt32](repeating: 0, count: kvN)
        var kvVals = [UInt32](repeating: 0, count: kvN)
        for i in 0..<kvN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            kvKeys[i] = UInt32(truncatingIfNeeded: rng >> 32)
            kvVals[i] = UInt32(i)  // value = original index
        }

        let (sortedKeys, sortedVals) = try engine.sortKeyValue(keys: kvKeys, values: kvVals)
        var kvCorrect = true
        for i in 0..<kvN {
            if sortedKeys[i] != kvKeys[Int(sortedVals[i])] {
                print("  KV MISMATCH at \(i)")
                kvCorrect = false
                break
            }
        }
        print("  Key-value sort: \(kvCorrect ? "PASS" : "FAIL")")

    } catch {
        print("Error: \(error)")
    }
}
