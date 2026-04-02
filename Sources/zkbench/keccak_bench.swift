// Keccak-256 Benchmark and correctness test
import zkMetal
import Foundation

public func runKeccakBench() {
    print("=== Keccak-256 Benchmark ===")

    // CPU reference: hash of empty string should be known value
    let emptyHash = keccak256([])
    let emptyHex = emptyHash.map { String(format: "%02x", $0) }.joined()
    // Keccak-256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    let expected = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    print("  keccak256(\"\") = \(emptyHex)")
    if emptyHex == expected {
        print("  [pass] Empty hash matches known value")
    } else {
        print("  [FAIL] Expected \(expected)")
        return
    }

    // Hash of "abc"
    let abcHash = keccak256([0x61, 0x62, 0x63])
    let abcHex = abcHash.map { String(format: "%02x", $0) }.joined()
    // Keccak-256("abc") = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    let expectedAbc = "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    print("  keccak256(\"abc\") = \(abcHex)")
    if abcHex == expectedAbc {
        print("  [pass] abc hash matches known value")
    } else {
        print("  [FAIL] Expected \(expectedAbc)")
        return
    }

    // CPU benchmark: hash 64-byte inputs
    if !skipCPU {
        let cpuInput = [UInt8](repeating: 0x42, count: 64)
        let warmup = 5000
        for _ in 0..<warmup { let _ = keccak256(cpuInput) }
        let cpuIters = 10000
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<cpuIters { let _ = keccak256(cpuInput) }
        let cpuElapsed = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000
        let cpuPerHash = cpuElapsed / Double(cpuIters) * 1000
        print(String(format: "\n  CPU: %.2f µs/hash (%.0f hash/s)", cpuPerHash, Double(cpuIters) / (cpuElapsed / 1000)))
    }

    // GPU tests
    do {
        let engine = try Keccak256Engine()

        // GPU correctness: compare 64-byte hash against CPU
        var testInput = [UInt8](repeating: 0, count: 64 * 4)
        for i in 0..<4 {
            for j in 0..<64 { testInput[i * 64 + j] = UInt8((i * 64 + j) & 0xFF) }
        }
        let gpuResults = try engine.hash64(testInput)

        var gpuCorrect = true
        for i in 0..<4 {
            let cpuRef = keccak256(Array(testInput[i*64..<(i+1)*64]))
            let gpuSlice = Array(gpuResults[i*32..<(i+1)*32])
            if cpuRef != gpuSlice {
                print("  [FAIL] GPU hash \(i) mismatch")
                print("    CPU: \(cpuRef.map { String(format: "%02x", $0) }.joined())")
                print("    GPU: \(gpuSlice.map { String(format: "%02x", $0) }.joined())")
                gpuCorrect = false
            }
        }
        if gpuCorrect {
            print("  [pass] GPU matches CPU for 4 test inputs")
        } else {
            return
        }

        // GPU batch benchmarks
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var input = [UInt8](repeating: 0, count: n * 64)
            var rng: UInt64 = 0xCAFEBABE
            for i in stride(from: 0, to: input.count, by: 8) {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                for b in 0..<min(8, input.count - i) {
                    input[i + b] = UInt8((rng >> (b * 8)) & 0xFF)
                }
            }

            // Warmup
            let _ = try engine.hash64(input)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hash64(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let hashPerSec = Double(n) / (median / 1000)
            print(String(format: "  GPU batch 2^%-2d = %7d: %8.2f ms (%10.0f hash/s, %.3f µs/hash)",
                        logN, n, median, hashPerSec, median / Double(n) * 1000))
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nKeccak-256 benchmark complete.")
}
