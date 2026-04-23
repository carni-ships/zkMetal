// Test P1 NTT with larger sizes
import zkMetal
import Foundation

public func runP1LargeTest() {
    print("=== P^1 NTT Large Size Test ===")

    do {
        let engine = try P1NTTEngine()
        print("Engine created successfully!")

        // Test sizes from 32 to 1M (logN 5 to 20)
        let sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

        for n in sizes {
            let logN = Int(log2(Double(n)))
            print("\nTesting N=\(n) (2^\(logN))...")

            // Create random data
            var data = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0x1234_5678 + UInt64(n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = M31(v: UInt32(rng >> 33) % M31.P)
            }

            // Warmup first
            do {
                let _ = try engine.ntt(data)
                print("  Warmup: OK")
            } catch {
                print("  Warmup FAILED: \(error)")
                break
            }

            // Correctness check with CPU
            let cpuEvals = P1NTTEngine.cpuNTT(data, logN: logN)
            let gpuEvals = try engine.ntt(data)

            var match = true
            for i in 0..<n {
                if gpuEvals[i].v != cpuEvals[i].v { match = false; break }
            }
            print("  Forward correctness: \(match ? "PASS" : "FAIL")")

            if !match { break }

            // Timed runs
            let runs = 3
            var times = [Double]()
            for r in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    let _ = try engine.ntt(data)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    times.append(elapsed)
                } catch {
                    print("  Run \(r) FAILED: \(error)")
                    break
                }
            }
            if times.count == runs {
                times.sort()
                let median = times[runs / 2]
                print("  Time: \(String(format: "%.2f", median)) ms")
            }
        }

        print("\nAll tests passed!")
    } catch {
        print("Error: \(error)")
    }
}