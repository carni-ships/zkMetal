// Poseidon2 Benchmark and correctness test
import zkMetal
import Foundation

public func runPoseidon2Bench() {
    print("=== Poseidon2 Benchmark (BN254 Fr, t=3) ===")

    // Basic sanity: permutation of [0,0,0]
    let zeroState = [Fr.zero, Fr.zero, Fr.zero]
    let result = poseidon2Permutation(zeroState)
    let r0 = frToInt(result[0])
    print("  poseidon2([0,0,0])[0] = \(r0.map{String(format:"%016llx",$0)}.joined())")

    if r0 == [0,0,0,0] {
        print("  [FAIL] Permutation of zero is zero")
        return
    }
    print("  [pass] Permutation of zero is non-trivial")

    // Determinism
    let result2 = poseidon2Permutation(zeroState)
    if frToInt(result[0]) != frToInt(result2[0]) {
        print("  [FAIL] Non-deterministic"); return
    }
    print("  [pass] Deterministic")

    // HorizenLabs reference test vector: permutation([0, 1, 2])
    // Source: https://github.com/HorizenLabs/poseidon2 (poseidon2_instance_bn256.rs)
    let refInput = [frFromInt(0), frFromInt(1), frFromInt(2)]
    let refOutput = poseidon2Permutation(refInput)
    let expected0: [UInt64] = [0x47f760054f4a3033, 0x8134334da98ea4f8, 0xbcb1929a82650f32, 0x0bb61d24daca55ee]
    let expected1: [UInt64] = [0x92defe7ff8d03570, 0x77a15d3f74ca6549, 0xcbcc80214f26a302, 0x303b6f7c86d043bf]
    let expected2: [UInt64] = [0x86296242cf766ec8, 0xe660b145994427cc, 0xf8617361c3ba7c52, 0x1ed25194542b12ee]
    let got0 = frToInt(refOutput[0]), got1 = frToInt(refOutput[1]), got2 = frToInt(refOutput[2])
    if got0 == expected0 && got1 == expected1 && got2 == expected2 {
        print("  [pass] Poseidon2 permutation matches HorizenLabs reference vector")
    } else {
        print("  [FAIL] Poseidon2 permutation does NOT match HorizenLabs reference!")
        print("    expected[0]: \(expected0.reversed().map{String(format:"%016llx",$0)}.joined())")
        print("    got[0]:      \(got0.reversed().map{String(format:"%016llx",$0)}.joined())")
        print("    expected[1]: \(expected1.reversed().map{String(format:"%016llx",$0)}.joined())")
        print("    got[1]:      \(got1.reversed().map{String(format:"%016llx",$0)}.joined())")
        print("    expected[2]: \(expected2.reversed().map{String(format:"%016llx",$0)}.joined())")
        print("    got[2]:      \(got2.reversed().map{String(format:"%016llx",$0)}.joined())")
    }

    // 2-to-1 hash
    let a = frFromInt(1), b = frFromInt(2)
    let h = poseidon2Hash(a, b)
    print("  poseidon2_hash(1,2) = \(frToInt(h).map{String(format:"%016llx",$0)}.joined())")
    print("  [pass] 2-to-1 hash computed")

    // CPU benchmark
    if !skipCPU {
        let warmup = 500
        for _ in 0..<warmup { let _ = poseidon2Hash(a, b) }

        let iters = 5000
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters { let _ = poseidon2Hash(a, b) }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let perHash = elapsed / Double(iters) * 1000
        print(String(format: "\n  CPU: %.1f µs/hash (%.0f hash/s)", perHash, Double(iters) / (elapsed / 1000)))
    }

    // GPU benchmark
    do {
        let engine = try Poseidon2Engine()

        // GPU correctness: compare against CPU
        let testPairs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                               frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let gpuResults = try engine.hashPairs(testPairs)
        var gpuCorrect = true
        for i in 0..<4 {
            let cpuH = poseidon2Hash(testPairs[i*2], testPairs[i*2+1])
            if frToInt(gpuResults[i]) != frToInt(cpuH) {
                print("  [FAIL] GPU hash \(i) mismatch")
                print("    CPU: \(frToInt(cpuH))")
                print("    GPU: \(frToInt(gpuResults[i]))")
                gpuCorrect = false
            }
        }
        if gpuCorrect {
            print("  [pass] GPU matches CPU for 4 test pairs")
        }

        // GPU batch benchmarks
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n * 2)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<(n * 2) {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.hashPairs(input)

            // Timed (5 iterations)
            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashPairs(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let hashPerSec = Double(n) / (median / 1000)
            print(String(format: "  GPU batch 2^%-2d = %6d: %7.2f ms (%8.0f hash/s, %.1f µs/hash)",
                        logN, n, median, hashPerSec, median / Double(n) * 1000))
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nPoseidon2 benchmark complete.")
}
