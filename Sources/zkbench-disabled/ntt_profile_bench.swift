// BN254 NTT Detailed Profiling Benchmark
// Measures time spent in each stage of the NTT operation

import Foundation
import Metal
import NeonFieldOps

@main
struct NTTProfilingBench {
    static func main() {
        print("=== BN254 NTT Detailed Profiling ===\n")

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No GPU found")
            return
        }

        print("Device: \(device.name)")
        print("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
        print()

        let n = 65536  // 2^16
        let logN = 16
        var data = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            data[i] = frFromInt(UInt64(i + 1))
        }

        guard let dataBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared) else {
            print("Failed to allocate buffer")
            return
        }

        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
        }

        // Test 1: Twiddle factor generation time
        print("--- Test 1: Twiddle Factor Generation ---")

        let engine = try! NTTEngine()

        // First call (includes computation)
        let t0 = CFAbsoluteTimeGetCurrent()
        let twiddles = engine.getTwiddles(logN: logN)
        let t1 = CFAbsoluteTimeGetCurrent()
        let firstCall = (t1 - t0) * 1000

        // Second call (cached)
        let t2 = CFAbsoluteTimeGetCurrent()
        _ = engine.getTwiddles(logN: logN)
        let t3 = CFAbsoluteTimeGetCurrent()
        let cachedCall = (t3 - t2) * 1000

        print("First call (compute):  \(String(format: "%.3f", firstCall))ms")
        print("Cached call (lookup): \(String(format: "%.3f", cachedCall))ms")
        print("Twiddle buffer size: \(n * MemoryLayout<Fr>.stride) bytes = \(n * MemoryLayout<Fr>.stride / 1024)KB")
        print()

        // Test 2: Memory copy time
        print("--- Test 2: Memory Copy Overhead ---")

        var copyTimes = [Double]()
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            copyTimes.append((t1 - t0) * 1000)
        }

        copyTimes.sort()
        print("memcpy \(n * 32) bytes (median): \(String(format: "%.3f", copyTimes[5]))ms")
        print("Throughput: \(String(format: "%.1f", Double(n * 32) / (copyTimes[5] / 1000) / 1_000_000)) MB/s")
        print()

        // Test 3: Buffer allocation
        print("--- Test 3: Buffer Allocation ---")

        var allocTimes = [Double]()
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let buf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)
            let t1 = CFAbsoluteTimeGetCurrent()
            allocTimes.append((t1 - t0) * 1000)
            _ = buf  // Keep alive
        }

        allocTimes.sort()
        print("Buffer allocation (median): \(String(format: "%.3f", allocTimes[5]))ms")
        print()

        // Test 4: Full NTT breakdown
        print("--- Test 4: Full NTT Operation Breakdown ---")

        // Warmup
        _ = try! engine.ntt(data, logN: logN)

        var nttTimes = [Double]()
        for i in 0..<5 {
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try! engine.ntt(data, logN: logN)
            let t1 = CFAbsoluteTimeGetCurrent()
            nttTimes.append((t1 - t0) * 1000)
        }

        nttTimes.sort()
        let nttMedian = nttTimes[2]
        print("Full NTT 2^16 (median): \(String(format: "%.3f", nttMedian))ms")
        print()

        // Test 5: Bit-reversal time
        print("--- Test 5: Bit-Reversal Operation ---")

        // Create test data for bit reversal
        var testData = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            testData[i] = frFromInt(UInt64(i))
        }

        // Time just the bit reversal
        var bitrevTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let bitrev = NTTEngine.bitReverse(testData)
            let t1 = CFAbsoluteTimeGetCurrent()
            bitrevTimes.append((t1 - t0) * 1000)
            _ = bitrev  // Use result
        }

        bitrevTimes.sort()
        print("CPU bit-reversal 2^16 (median): \(String(format: "%.3f", bitrevTimes[2]))ms")
        print()

        // Test 6: Field operation time
        print("--- Test 6: Field Operation Performance ---")

        let a = frFromInt(12345)
        let b = frFromInt(67890)

        var addTimes = [Double]()
        var mulTimes = [Double]()

        for _ in 0..<10000 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = fr_add(a, b)
            let t1 = CFAbsoluteTimeGetCurrent()
            addTimes.append((t1 - t0) * 1_000_000)  // microseconds

            let t2 = CFAbsoluteTimeGetCurrent()
            let _ = fr_mul(a, b)
            let t3 = CFAbsoluteTimeGetCurrent()
            mulTimes.append((t3 - t2) * 1_000_000)
        }

        addTimes.sort()
        mulTimes.sort()

        print("Fr add (median of 10000): \(String(format: "%.3f", addTimes[5000]))μs")
        print("Fr mul (median of 10000): \(String(format: "%.3f", mulTimes[5000]))μs")
        print("Estimated NTT has ~\(n * logN * 2) field operations")
        print("  Adds: \(n * logN) × 2 = \(n * logN * 2)")
        print("  Muls: \(n * logN) = \(n * logN)")
        print()

        // Test 7: Memory bandwidth estimate
        print("--- Test 7: Memory Bandwidth Analysis ---")

        // Each NTT stage reads and writes the entire array
        let memoryPerStage = n * MemoryLayout<Fr>.stride  // bytes
        let numStages = logN
        let totalMemoryAccessed = memoryPerStage * numStages * 2  // read + write

        print("Per stage: \(memoryPerStage / 1024)KB")
        print("Total accessed: \(totalMemoryAccessed / 1024)KB")
        print()

        if nttMedian > 0 {
            let bandwidth = Double(totalMemoryAccessed) / (nttMedian / 1000) / 1_000_000
            print("Achieved bandwidth: \(String(format: "%.1f", bandwidth)) MB/s")
            print()
        }

        // Test 8: Different sizes
        print("--- Test 8: NTT Performance by Size ---")

        for testLogN in [10, 12, 14, 16, 18, 20] {
            let testN = 1 << testLogN
            var testData = [Fr](repeating: Fr.zero, count: testN)
            for i in 0..<testN {
                testData[i] = frFromInt(UInt64(i))
            }

            // Warmup
            _ = try! engine.ntt(testData, logN: testLogN)

            var times = [Double]()
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try! engine.ntt(testData, logN: testLogN)
                let t1 = CFAbsoluteTimeGetCurrent()
                times.append((t1 - t0) * 1000)
            }

            times.sort()
            let median = times[1]
            let throughput = Double(testN) / (median / 1000) / 1_000_000

            print("  2^\(testLogN) = \(testN) | \(String(format: "%7.2f", median))ms | \(String(format: "%.1f", throughput))M elem/s")
        }
        print()

        // Summary
        print("--- Analysis ---")
        print("Fixed overhead components:")
        print("  Twiddle generation: \(String(format: "%.3f", firstCall))ms (first call)")
        print("  Memory copy: \(String(format: "%.3f", copyTimes[5]))ms")
        print("  Buffer allocation: \(String(format: "%.3f", allocTimes[5]))ms")
        print()
        print("Variable cost components:")
        print("  Field arithmetic: scales with O(n log n)")
        print("  Memory bandwidth: \(totalMemoryAccessed / 1024)KB total accessed")
        print()
        print("Performance scaling:")
        print("  Time per element: \(String(format: "%.3f", nttMedian / Double(n) * 1000))μs")
        print("  Time per logN stage: \(String(format: "%.3f", nttMedian / Double(logN)))ms")
    }
}
