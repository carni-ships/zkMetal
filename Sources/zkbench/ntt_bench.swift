// NTT Benchmark — detailed profiling across sizes
import zkMetal
import Foundation

public func runNTTBench() {
    print("=== NTT Benchmark ===")
    print("Fr stride: \(MemoryLayout<Fr>.stride) bytes")

    do {
        let engine = try NTTEngine()

        // Test sizes from 2^10 to 2^22
        let sizes = [10, 12, 14, 16, 18, 20, 22]
        var rng: UInt64 = 0xDEAD_BEEF

        for logN in sizes {
            let n = 1 << logN
            var data = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = frFromInt(rng >> 32)
            }

            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }

            // Warmup (3 rounds)
            for _ in 0..<3 {
                try engine.ntt(data: dataBuf, logN: logN)
                try engine.intt(data: dataBuf, logN: logN)
            }

            // Reload fresh data
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }

            // Timed NTT (10 iterations)
            var nttTimes = [Double]()
            var inttTimes = [Double]()

            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)

                let t2 = CFAbsoluteTimeGetCurrent()
                try engine.intt(data: dataBuf, logN: logN)
                let t3 = CFAbsoluteTimeGetCurrent()
                inttTimes.append((t3 - t2) * 1000)
            }

            nttTimes.sort()
            inttTimes.sort()
            let nttMedian = nttTimes[5]
            let inttMedian = inttTimes[5]

            // CPU NTT for comparison
            var cpuMs: Double = 0
            do {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                let _ = NTTEngine.cpuNTT(data, logN: logN)
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            // Throughput: elements per second
            let elemPerSec = Double(n) / (nttMedian / 1000)

            if cpuMs > 0 {
                print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | CPU: %7.1fms | %.0fx | %.1fM elem/s",
                            logN, n, nttMedian, cpuMs, cpuMs / nttMedian, elemPerSec / 1e6))
            } else {
                print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                            logN, n, nttMedian, elemPerSec / 1e6))
            }
        }

        // Correctness check at n=1024
        print("\n--- Correctness verification ---")
        let testN = 1024
        let testLogN = 10
        var testInput = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = frFromInt(UInt64(i + 1))
        }
        let gpuNTT = try engine.ntt(testInput)
        let gpuRecovered = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            let expected = frToInt(testInput[i])
            let got = frToInt(gpuRecovered[i])
            if expected != got {
                print("  MISMATCH at \(i): expected \(expected), got \(got)")
                correct = false
                break
            }
        }
        print("  Round-trip (2^10): \(correct ? "PASS" : "FAIL")")

        // Round-trip at 2^20 (exercises four-step path for both NTT and iNTT)
        let testN3 = 1 << 20
        var testInput3 = [Fr](repeating: Fr.zero, count: testN3)
        rng = 0xCAFE_BABE
        for i in 0..<testN3 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput3[i] = frFromInt(rng >> 32)
        }
        let gpuNTT3 = try engine.ntt(testInput3)
        let gpuRecovered3 = try engine.intt(gpuNTT3)
        var correct3 = true
        var mismatches3 = 0
        for i in 0..<testN3 {
            let expected = frToInt(testInput3[i])
            let got = frToInt(gpuRecovered3[i])
            if expected != got {
                if mismatches3 < 5 {
                    print("  RT MISMATCH at \(i): expected \(expected[0]), got \(got[0])")
                }
                correct3 = false
                mismatches3 += 1
            }
        }
        print("  Four-step round-trip (2^20): \(correct3 ? "PASS" : "FAIL") (\(mismatches3) mismatches)")

        // Round-trip at 2^22 (exercises standard path with many global stages)
        let testN4 = 1 << 22
        var testInput4 = [Fr](repeating: Fr.zero, count: testN4)
        rng = 0xBEEF_DEAD
        for i in 0..<testN4 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput4[i] = frFromInt(rng >> 32)
        }
        let gpuNTT4 = try engine.ntt(testInput4)
        let gpuRecovered4 = try engine.intt(gpuNTT4)
        var mismatches4 = 0
        for i in 0..<testN4 {
            let expected = frToInt(testInput4[i])
            let got = frToInt(gpuRecovered4[i])
            if expected != got { mismatches4 += 1 }
        }
        print("  Standard path round-trip (2^22): \(mismatches4 == 0 ? "PASS" : "FAIL") (\(mismatches4) mismatches)")


    } catch {
        print("Error: \(error)")
    }

    // BabyBear NTT Benchmark
    print("\n=== BabyBear NTT Benchmark ===")
    print("Bb stride: \(MemoryLayout<Bb>.stride) bytes")

    do {
        let bbEngine = try BabyBearNTTEngine()

        let bbSizes = [10, 12, 14, 16, 18, 20, 22, 24]
        var bbRng: UInt32 = 0xDEAD_BEEF

        for logN in bbSizes {
            let n = 1 << logN
            var data = [Bb](repeating: Bb.zero, count: n)
            for i in 0..<n {
                bbRng = bbRng &* 1664525 &+ 1013904223
                data[i] = Bb(v: bbRng % Bb.P)
            }

            let dataBuf = bbEngine.device.makeBuffer(
                length: n * MemoryLayout<Bb>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
            }

            // Warmup
            for _ in 0..<3 {
                try bbEngine.ntt(data: dataBuf, logN: logN)
                try bbEngine.intt(data: dataBuf, logN: logN)
            }

            // Reload fresh data
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
            }

            var nttTimes = [Double]()
            var inttTimes = [Double]()

            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try bbEngine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)

                let t2 = CFAbsoluteTimeGetCurrent()
                try bbEngine.intt(data: dataBuf, logN: logN)
                let t3 = CFAbsoluteTimeGetCurrent()
                inttTimes.append((t3 - t2) * 1000)
            }

            nttTimes.sort()
            inttTimes.sort()
            let nttMedian = nttTimes[5]
            let inttMedian = inttTimes[5]
            let elemPerSec = Double(n) / (nttMedian / 1000)

            print(String(format: "  2^%-2d = %7d | NTT: %7.2fms | iNTT: %7.2fms | %.1fM elem/s",
                        logN, n, nttMedian, inttMedian, elemPerSec / 1e6))
        }

        // Correctness check
        print("\n--- BabyBear Correctness verification ---")
        let testN = 1024
        let testLogN = 10
        var testInput = [Bb](repeating: Bb.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = Bb(v: UInt32(i + 1))
        }

        // GPU round-trip
        let gpuNTT = try bbEngine.ntt(testInput)
        let gpuRecovered = try bbEngine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            if testInput[i].v != gpuRecovered[i].v {
                print("  MISMATCH at \(i): expected \(testInput[i].v), got \(gpuRecovered[i].v)")
                correct = false
                break
            }
        }
        print("  GPU round-trip: \(correct ? "PASS" : "FAIL")")

        // Cross-check GPU vs CPU
        let cpuNTT = BabyBearNTTEngine.cpuNTT(testInput, logN: testLogN)
        var cpuMatch = true
        for i in 0..<testN {
            if gpuNTT[i].v != cpuNTT[i].v {
                print("  CPU/GPU MISMATCH at \(i): cpu=\(cpuNTT[i].v), gpu=\(gpuNTT[i].v)")
                cpuMatch = false
                break
            }
        }
        print("  CPU vs GPU NTT: \(cpuMatch ? "PASS" : "FAIL")")

    } catch {
        print("BabyBear NTT Error: \(error)")
    }

    // Goldilocks NTT Benchmark
    print("\n=== Goldilocks NTT Benchmark ===")
    print("Gl stride: \(MemoryLayout<Gl>.stride) bytes")

    do {
        let glEngine = try GoldilocksNTTEngine()

        let glSizes = [10, 12, 14, 16, 18, 20, 22, 24]
        var glRng: UInt64 = 0xCAFE_BABE

        for logN in glSizes {
            let n = 1 << logN
            var data = [Gl](repeating: Gl.zero, count: n)
            for i in 0..<n {
                glRng = glRng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = Gl(v: glRng % Gl.P)
            }

            let dataBuf = glEngine.device.makeBuffer(
                length: n * MemoryLayout<Gl>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
            }

            // Warmup
            for _ in 0..<3 {
                try glEngine.ntt(data: dataBuf, logN: logN)
                try glEngine.intt(data: dataBuf, logN: logN)
            }

            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
            }

            var nttTimes = [Double]()
            var inttTimes = [Double]()

            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try glEngine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)

                let t2 = CFAbsoluteTimeGetCurrent()
                try glEngine.intt(data: dataBuf, logN: logN)
                let t3 = CFAbsoluteTimeGetCurrent()
                inttTimes.append((t3 - t2) * 1000)
            }

            nttTimes.sort()
            inttTimes.sort()
            let nttMedian = nttTimes[5]
            let inttMedian = inttTimes[5]
            let elemPerSec = Double(n) / (nttMedian / 1000)

            print(String(format: "  2^%-2d = %7d | NTT: %7.2fms | iNTT: %7.2fms | %.1fM elem/s",
                        logN, n, nttMedian, inttMedian, elemPerSec / 1e6))
        }

        // Correctness check
        print("\n--- Goldilocks Correctness verification ---")
        let testN = 1024
        let testLogN = 10
        var testInput = [Gl](repeating: Gl.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = Gl(v: UInt64(i + 1))
        }

        let gpuNTT = try glEngine.ntt(testInput)
        let gpuRecovered = try glEngine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            if testInput[i].v != gpuRecovered[i].v {
                print("  MISMATCH at \(i): expected \(testInput[i].v), got \(gpuRecovered[i].v)")
                correct = false
                break
            }
        }
        print("  GPU round-trip: \(correct ? "PASS" : "FAIL")")

        let cpuNTT = GoldilocksNTTEngine.cpuNTT(testInput, logN: testLogN)
        var cpuMatch = true
        for i in 0..<testN {
            if gpuNTT[i].v != cpuNTT[i].v {
                print("  CPU/GPU MISMATCH at \(i): cpu=\(cpuNTT[i].v), gpu=\(gpuNTT[i].v)")
                cpuMatch = false
                break
            }
        }
        print("  CPU vs GPU NTT: \(cpuMatch ? "PASS" : "FAIL")")

    } catch {
        print("Goldilocks NTT Error: \(error)")
    }
}
