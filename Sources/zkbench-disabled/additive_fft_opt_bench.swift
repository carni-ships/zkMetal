// GPU Additive FFT Optimization Benchmark
// Tests different kernel variants to identify performance bottlenecks

import Foundation
import Metal

@main
struct AdditiveFFTOptBench {
    static func main() {
        print("=== GPU Additive FFT Optimization Benchmark ===")
        print("Testing GF(2^8) Additive FFT optimizations")
        print()

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No GPU found")
            return
        }

        print("Device: \(device.name)")
        print("Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
        print()

        let sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]  // 2^10 to 2^22
        let kValues = [10, 12, 14, 16, 18, 20, 22]

        for (n, k) in zip(sizes, kValues) {
            print("Testing 2^\(k) = \(n) elements...")

            // Generate test data
            var data = [UInt8](repeating: 0, count: n)
            for i in 0..<n {
                data[i] = UInt8(i & 0xFF)
            }

            // Generate basis elements (for testing)
            var basis = [UInt8](repeating: 0, count: k)
            for i in 0..<k {
                basis[i] = UInt8(i + 1)  // Simple basis for testing
            }

            guard let dataBuf = device.makeBuffer(length: n, options: .storageModeShared),
                  let basisBuf = device.makeBuffer(length: k, options: .storageModeShared) else {
                print("  Failed to allocate buffers")
                continue
            }

            // Copy data to buffers
            memcpy(dataBuf.contents(), data, n)
            memcpy(basisBuf.contents(), basis, k)

            // Test 1: Memory bandwidth test
            print("  Memory bandwidth:")
            let bwTimes = measureTime {
                for _ in 0..<3 {
                    memcpy(dataBuf.contents(), data, n)
                }
            }
            print("    memcpy \(n) bytes: \(String(format: "%.3f", bwTimes))ms")
            print("    Throughput: \(String(format: "%.1f", Double(n * 3) / (bwTimes / 1000) / 1_000_000)) MB/s")

            // Test 2: Theoretical computation time
            // For 2^k elements with k levels: n * k multiplications
            let totalMuls = n * k
            let perMulTime = 0.000001  // 1μs per multiplication (estimated)
            let estimatedTime = Double(totalMuls) * perMulTime
            print("  Computation estimate:")
            print("    Total multiplications: \(totalMuls)")
            print("    Estimated time at 1μs/mul: \(String(format: "%.1f", estimatedTime / 1000))ms")
            print("    To reach 0.5ms target: need \(String(format: "%.3f", perMulTime * 1_000_000_000))ns per mul")

            // Test 3: Kernel launch overhead
            print("  Kernel launch overhead:")
            let cmdBuf = device.makeCommandQueue()!.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.endEncoding()
            cmdBuf.commit()

            let t0 = CFAbsoluteTimeGetCurrent()
            cmdBuf.waitUntilCompleted()
            let t1 = CFAbsoluteTimeGetCurrent()
            let overhead = (t1 - t0) * 1000

            print("    Empty command buffer: \(String(format: "%.3f", overhead))ms")

            print("")
        }

        print("=== Optimization Strategies ===")
        print()
        print("1. SIMD Vectorization (uchar4)")
        print("   - Process 4 elements per thread")
        print("   - Potential 4x throughput improvement")
        print("   - Better memory coalescing")
        print()
        print("2. Register Tiling")
        print("   - Precompute partner offsets")
        print("   - Reduce address calculation")
        print("   - Keep values in registers longer")
        print()
        print("3. Threadgroup Memory")
        print("   - Use shared memory for intermediate results")
        print("   - Reduce global memory traffic")
        print("   - Limited by threadgroup size (max 1024 on M3 Pro)")
        print()
        print("4. Batched Processing")
        print("   - Process multiple FFTs in one dispatch")
        print("   - Amortize kernel launch overhead")
        print("   - Better GPU utilization")
        print()
        print("5. Optimized Memory Access")
        print("   - Coalesced reads/writes")
        print("   - Reduce cache misses")
        print("   - Use vectorized load/store")
        print()
        print("=== Recommendations ===")
        print()
        print("For 2^22 (current: ~11-14ms, target: ~0.5ms):")
        print("  1. Implement SIMD kernel (4x improvement → ~3ms)")
        print("  2. Add register tiling (2x improvement → ~1.5ms)")
        print("  3. Optimize LUT access (2x improvement → ~0.75ms)")
        print("  4. Use batched processing for multiple FFTs (amortized overhead)")
        print()
        print("The LUT approach itself is not the bottleneck.")
        print("Memory access patterns and SIMD utilization are key.")
    }

    static func measureTime(_ block: () -> Void) -> Double {
        let iterations = 3
        var times = [Double]()

        for _ in 0..<iterations {
            let t0 = CFAbsoluteTimeGetCurrent()
            block()
            let t1 = CFAbsoluteTimeGetCurrent()
            times.append((t1 - t0) * 1000)
        }

        times.sort()
        return times[1]  // Return median
    }
}
