// Blake3 Benchmark
import zkMetal
import Foundation

public func runBlake3Bench() {
    print("=== Blake3 Benchmark ===")

    // CPU correctness: known test vector
    // Blake3("") = af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
    let emptyHash = blake3([])
    let emptyHex = emptyHash.map { String(format: "%02x", $0) }.joined()
    let expectedEmpty = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    print("  Blake3('') = \(emptyHex)")
    print("  Expected    = \(expectedEmpty)")
    print("  CPU empty: \(emptyHex == expectedEmpty ? "PASS" : "FAIL")")

    do {
        let engine = try Blake3Engine()

        // GPU correctness: hash 4 inputs and compare with CPU
        var testInput = [UInt8](repeating: 0, count: 64 * 4)
        for i in 0..<(64 * 4) { testInput[i] = UInt8(i & 0xFF) }
        let gpuResults = try engine.hash64(testInput)

        var gpuCorrect = true
        for i in 0..<4 {
            let block = Array(testInput[(i * 64)..<((i + 1) * 64)])
            let cpuHash = blake3(block)
            let gpuHash = Array(gpuResults[(i * 32)..<((i + 1) * 32)])
            if cpuHash != gpuHash {
                print("  MISMATCH at block \(i)")
                print("    CPU: \(cpuHash.prefix(8).map { String(format: "%02x", $0) }.joined())")
                print("    GPU: \(gpuHash.prefix(8).map { String(format: "%02x", $0) }.joined())")
                gpuCorrect = false
            }
        }
        print("  GPU vs CPU (4 blocks): \(gpuCorrect ? "PASS" : "FAIL")")

        // CPU benchmark
        if !skipCPU {
            let cpuInput = [UInt8](repeating: 0x42, count: 64)
            let warmup = 5000
            for _ in 0..<warmup { let _ = blake3(cpuInput) }
            let cpuIters = 10000
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<cpuIters { let _ = blake3(cpuInput) }
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuT0
            let cpuPerHash = cpuTime / Double(cpuIters) * 1e6
            print(String(format: "  CPU: %.1f µs/hash", cpuPerHash))
        }

        // GPU batch benchmark
        print("\n--- Blake3 GPU Batch Benchmark ---")
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var input = [UInt8](repeating: 0, count: n * 64)
            var rng: UInt64 = 0xB1A3ECAFE
            for i in stride(from: 0, to: input.count, by: 8) {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let bytes = withUnsafeBytes(of: rng) { Array($0) }
                for j in 0..<min(8, input.count - i) {
                    input[i + j] = bytes[j]
                }
            }

            // Warmup
            for _ in 0..<3 { let _ = try engine.hash64(input) }

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hash64(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let usPerHash = median * 1000 / Double(n)
            print(String(format: "  2^%-2d = %7d | %7.2f ms | %.2f µs/hash",
                        logN, n, median, usPerHash))
        }

    } catch {
        print("  [FAIL] Blake3 error: \(error)")
    }
}
