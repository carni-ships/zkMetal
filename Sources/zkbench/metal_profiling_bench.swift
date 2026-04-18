// Metal Command Buffer Profiling Benchmark
// Measures encoding overhead vs actual GPU execution time

import Foundation
import Metal
import zkMetal

@main
struct MetalProfilingBench {
    static func main() {
        print("=== Metal Command Buffer Profiling ===\n")

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No GPU found")
            return
        }

        print("Device: \(device.name)")
        print("Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
        print("Recommended max working set size: \(device.recommendedMaxWorkingSetSize / 1024) KB")
        print("Current allocator: \(device.currentAllocator)")
        print()

        // Test 1: Measure command buffer encoding overhead
        print("--- Test 1: Command Buffer Encoding Overhead ---")
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

        // Measure encoding time (before commit)
        var encodingTimes = [Double]()
        var executionTimes = [Double]()

        for i in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()

            // Create and encode command buffer (don't commit yet)
            guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else {
                print("Failed to create encoder")
                return
            }

            // Minimal encoding
            enc.endEncoding()

            let t1 = CFAbsoluteTimeGetCurrent()
            encodingTimes.append((t1 - t0) * 1000)

            // Now commit and measure execution
            let t2 = CFAbsoluteTimeGetCurrent()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            let t3 = CFAbsoluteTimeGetCurrent()
            executionTimes.append((t3 - t2) * 1000)
        }

        encodingTimes.sort()
        executionTimes.sort()

        print("Encoding time (median): \(String(format: "%.3f", encodingTimes[5]))ms")
        print("Execution time (median): \(String(format: "%.3f", executionTimes[5]))ms")
        print("Total overhead: \(String(format: "%.3f", encodingTimes[5] + executionTimes[5]))ms")
        print()

        // Test 2: Measure actual NTT operation breakdown
        print("--- Test 2: NTT Operation Breakdown ---")

        // Warmup
        _ = try? engine.ntt(data, logN: 10)

        var nttTimings = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try! engine.ntt(data, logN: 10)
            let t1 = CFAbsoluteTimeGetCurrent()
            nttTimings.append((t1 - t0) * 1000)
        }

        nttTimings.sort()
        print("NTT 2^10 total time (median): \(String(format: "%.3f", nttTimings[2]))ms")
        print()

        // Test 3: Check for command buffer reuse
        print("--- Test 3: Multiple Operations in Single Command Buffer ---")

        var multiOpTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()

            // Try to batch operations
            guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
                print("Failed to create command buffer")
                return
            }

            // Encode multiple NTTs in one command buffer
            for _ in 0..<3 {
                // This would require modifying NTTEngine to support encoding into existing buffer
                // For now, just measure sequential calls
                _ = try! engine.ntt(data, logN: 10)
            }

            let t1 = CFAbsoluteTimeGetCurrent()
            multiOpTimes.append((t1 - t0) * 1000)
        }

        multiOpTimes.sort()
        print("3 sequential NTTs (median): \(String(format: "%.3f", multiOpTimes[2]))ms")
        print("Per-operation: \(String(format: "%.3f", multiOpTimes[2] / 3))ms")
        print()

        // Test 4: Check driver info
        print("--- Test 4: Metal Driver Information ---")
        print("Device location: \(device.location)")
        print("Device is headless: \(device.isHeadless)")
        print("Device is low power: \(device.isLowPower)")
        print("Device is removable: \(device.isRemovable)")
        print("Has unified memory: \(device.hasUnifiedMemory)")
        print()

        // Test 5: Memory transfer overhead
        print("--- Test 5: Memory Transfer Overhead ---")

        var memcpyTimes = [Double]()
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            let t1 = CFAbsoluteTimeGetCurrent()
            memcpyTimes.append((t1 - t0) * 1000)
        }

        memcpyTimes.sort()
        print("memcpy \(n * MemoryLayout<Fr>.stride) bytes (median): \(String(format: "%.3f", memcpyTimes[5]))ms")
        print()

        print("--- Analysis ---")
        let fixedOverhead = encodingTimes[5] + executionTimes[5]
        let nttOverhead = nttTimings[2] - fixedOverhead

        print("Estimated fixed overhead per operation: \(String(format: "%.3f", fixedOverhead))ms")
        print("Actual NTT computation time: \(String(format: "%.3f", max(0, nttOverhead)))ms")
        print()
        print("If fixed overhead > 10ms, consider:")
        print("  - Using async command buffers")
        print("  - Batching multiple operations")
        print("  - Checking for Metal driver updates")
    }
}
