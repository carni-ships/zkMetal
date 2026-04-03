// BLS12-377 NTT Benchmark
import zkMetal
import Foundation

public func runBLS12377NTTBench() {
    print("\n=== BLS12-377 NTT Benchmark ===")
    print("Fr377 stride: \(MemoryLayout<Fr377>.stride) bytes")

    do {
        let engine = try BLS12377NTTEngine()

        // Correctness: round-trip at 2^10
        let testN = 1024
        var testInput = [Fr377](repeating: Fr377.zero, count: testN)
        for i in 0..<testN {
            testInput[i] = fr377FromInt(UInt64(i + 1))
        }
        let gpuNTT = try engine.ntt(testInput)
        let gpuRecovered = try engine.intt(gpuNTT)
        var correct = true
        for i in 0..<testN {
            let expected = fr377ToInt(testInput[i])
            let got = fr377ToInt(gpuRecovered[i])
            if expected != got {
                print("  MISMATCH at \(i): expected \(expected), got \(got)")
                correct = false
                break
            }
        }
        print("  Round-trip (2^10): \(correct ? "PASS" : "FAIL")")

        // Round-trip at 2^20 (four-step path)
        let testN2 = 1 << 20
        var testInput2 = [Fr377](repeating: Fr377.zero, count: testN2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testInput2[i] = fr377FromInt(rng >> 32)
        }
        let gpuNTT2 = try engine.ntt(testInput2)
        let gpuRecovered2 = try engine.intt(gpuNTT2)
        var correct2 = true
        var mismatches2 = 0
        for i in 0..<testN2 {
            let expected = fr377ToInt(testInput2[i])
            let got = fr377ToInt(gpuRecovered2[i])
            if expected != got {
                if mismatches2 < 3 {
                    print("  RT MISMATCH at \(i): expected \(expected[0]), got \(got[0])")
                }
                correct2 = false
                mismatches2 += 1
            }
        }
        print("  Four-step round-trip (2^20): \(correct2 ? "PASS" : "FAIL") (\(mismatches2) mismatches)")

        // Performance benchmark
        print("\n--- BLS12-377 NTT Performance ---")
        let sizes = [10, 12, 14, 16, 18, 20, 22]

        for logN in sizes {
            let n = 1 << logN
            var data = [Fr377](repeating: Fr377.zero, count: n)
            rng = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = fr377FromInt(rng >> 32)
            }

            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Fr377>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
            }

            // Warmup
            for _ in 0..<3 {
                try engine.ntt(data: dataBuf, logN: logN)
                try engine.intt(data: dataBuf, logN: logN)
            }

            // Reload fresh data
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
            }

            var nttTimes = [Double]()
            for _ in 0..<10 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr377>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                let t1 = CFAbsoluteTimeGetCurrent()
                nttTimes.append((t1 - t0) * 1000)
            }

            nttTimes.sort()
            let nttMedian = nttTimes[5]
            let elemPerSec = Double(n) / (nttMedian / 1000)
            print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                        logN, n, nttMedian, elemPerSec / 1e6))
        }

    } catch {
        print("  [FAIL] BLS12-377 NTT error: \(error)")
    }
}
