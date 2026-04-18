// Batch Operation Benchmark - Measures overhead amortization
import Foundation
import Metal
import zkMetal

@main
struct BatchingBench {
    static func main() {
        print("=== Batch Operation Benchmark ===\n")

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No GPU found")
            return
        }

        let engine = try! NTTEngine()
        let n = 1024
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

        // Test 1: Sequential operations (current approach)
        print("--- Test 1: Sequential Operations ---")
        var sequentialTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                _ = try! engine.ntt(data, logN: 10)
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            sequentialTimes.append((t1 - t0) * 1000)
        }

        sequentialTimes.sort()
        let seqMedian = sequentialTimes[2]
        print("3 sequential NTTs: \(String(format: "%.3f", seqMedian))ms")
        print("Per operation: \(String(format: "%.3f", seqMedian / 3))ms")
        print()

        // Test 2: Measure twiddle factor generation overhead
        print("--- Test 2: Twiddle Factor Generation Overhead ---")

        var twiddleTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = engine.getTwiddles(logN: 10)
            let t1 = CFAbsoluteTimeGetCurrent()
            twiddleTimes.append((t1 - t0) * 1000)
        }

        twiddleTimes.sort()
        print("Twiddle factor generation (median): \(String(format: "%.3f", twiddleTimes[2]))ms")
        print()

        // Test 3: Memory allocation overhead
        print("--- Test 3: Memory Allocation Overhead ---")

        var allocTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let buf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)
            let t1 = CFAbsoluteTimeGetCurrent()
            allocTimes.append((t1 - t0) * 1000)
            // Keep buffer alive
            _ = buf
        }

        allocTimes.sort()
        print("Buffer allocation (median): \(String(format: "%.3f", allocTimes[2]))ms")
        print()

        // Test 4: Memory copy overhead
        print("--- Test 4: Memory Copy Overhead ---")

        var copyTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            copyTimes.append((t1 - t0) * 1000)
        }

        copyTimes.sort()
        print("Memory copy \(n * 32) bytes (median): \(String(format: "%.3f", copyTimes[2]))ms")
        print()

        // Test 5: Operation breakdown
        print("--- Test 5: Operation Breakdown ---")

        let totalOverhead = twiddleTimes[2] + allocTimes[2] + copyTimes[2]
        let nttOnlyTime = seqMedian - totalOverhead

        print("Total sequential time: \(String(format: "%.3f", seqMedian))ms")
        print("  Twiddle generation: \(String(format: "%.3f", twiddleTimes[2]))ms (\(Int(twiddleTimes[2]/seqMedian*100))%)")
        print("  Buffer allocation: \(String(format: "%.3f", allocTimes[2]))ms (\(Int(allocTimes[2]/seqMedian*100))%)")
        print("  Memory copy: \(String(format: "%.3f", copyTimes[2]))ms (\(Int(copyTimes[2]/seqMedian*100))%)")
        print("  Actual NTT computation: \(String(format: "%.3f", max(0, nttOnlyTime)))ms (\(Int(max(0,nttOnlyTime)/seqMedian*100))%)")
        print()

        // Test 6: Cached twiddle impact
        print("--- Test 6: Cached Twiddle Performance ---")

        // First call caches twiddles
        _ = engine.getTwiddles(logN: 10)

        var cachedTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try! engine.ntt(data, logN: 10)
            let t1 = CFAbsoluteTimeGetCurrent()
            cachedTimes.append((t1 - t0) * 1000)
        }

        cachedTimes.sort()
        print("NTT with cached twiddles (median): \(String(format: "%.3f", cachedTimes[2]))ms")
        print("Improvement: \(String(format: "%.1f", (seqMedian - cachedTimes[2]) / seqMedian * 100))%")
        print()

        print("--- Recommendations ---")
        if twiddleTimes[2] > 1.0 {
            print("✓ Twiddle generation is expensive - precompute and cache")
        }
        if allocTimes[2] > 1.0 {
            print("✓ Buffer allocation is expensive - reuse buffers")
        }
        if copyTimes[2] > 1.0 {
            print("✓ Memory copy is expensive - consider zero-copy approaches")
        }
        if nttOnlyTime < seqMedian * 0.5 {
            print("✓ Actual computation is fast - focus on reducing overhead")
        }
    }
}
