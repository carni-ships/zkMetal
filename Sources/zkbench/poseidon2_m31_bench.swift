// Poseidon2 M31 Benchmark and correctness test
import zkMetal
import Foundation

public func runPoseidon2M31Bench() {
    print("=== Poseidon2 M31 Benchmark (Mersenne31, t=16) ===")

    // Basic sanity: permutation of all zeros
    let zeroState = [M31](repeating: M31.zero, count: 16)
    let result = poseidon2M31Permutation(zeroState)
    print("  poseidon2_m31([0]*16)[0] = \(result[0].v)")

    if result[0].v == 0 && result[1].v == 0 {
        print("  [FAIL] Permutation of zero is zero")
        return
    }
    print("  [pass] Permutation of zero is non-trivial")

    // Determinism
    let result2 = poseidon2M31Permutation(zeroState)
    if result[0].v != result2[0].v || result[7].v != result2[7].v {
        print("  [FAIL] Non-deterministic"); return
    }
    print("  [pass] Deterministic")

    // Permutation of [1, 2, ..., 16]
    var testInput = [M31](repeating: M31.zero, count: 16)
    for i in 0..<16 { testInput[i] = M31(v: UInt32(i + 1)) }
    let testResult = poseidon2M31Permutation(testInput)
    print("  poseidon2_m31([1..16])[0..3] = [\(testResult[0].v), \(testResult[1].v), \(testResult[2].v), \(testResult[3].v)]")
    print("  [pass] Permutation computed")

    // 2-to-1 hash
    let left = [M31](repeating: M31(v: 1), count: 8)
    let right = [M31](repeating: M31(v: 2), count: 8)
    let h = poseidon2M31Hash(left: left, right: right)
    print("  poseidon2_m31_hash([1]*8, [2]*8)[0..3] = [\(h[0].v), \(h[1].v), \(h[2].v), \(h[3].v)]")
    print("  [pass] 2-to-1 hash computed")

    // CPU benchmark: single permutation
    if !skipCPU {
        let warmup = 2000
        var state = zeroState
        for _ in 0..<warmup { state = poseidon2M31Permutation(state) }

        let iters = 10000
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters { state = poseidon2M31Permutation(state) }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let perPerm = elapsed / Double(iters) * 1000
        _ = state  // prevent optimization
        print(String(format: "\n  CPU permutation: %.2f µs/perm (%.0f perm/s)", perPerm, Double(iters) / (elapsed / 1000)))

        // CPU 2-to-1 hash benchmark
        let hashIters = 10000
        var acc = left
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<hashIters { acc = poseidon2M31Hash(left: acc, right: right) }
        let hElapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let perHash = hElapsed / Double(hashIters) * 1000
        _ = acc
        print(String(format: "  CPU 2-to-1 hash: %.2f µs/hash (%.0f hash/s)", perHash, Double(hashIters) / (hElapsed / 1000)))
    }

    // GPU benchmark
    do {
        let engine = try Poseidon2M31Engine()

        // GPU correctness: compare against CPU for hash pairs
        let nodeSize = 8
        let numTestPairs = 4
        var testPairs = [M31](repeating: M31.zero, count: numTestPairs * 2 * nodeSize)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testPairs.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testPairs[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
        }

        let gpuResults = try engine.hashPairs(testPairs)
        var gpuCorrect = true
        for i in 0..<numTestPairs {
            let l = Array(testPairs[(i*2*nodeSize)..<(i*2*nodeSize + nodeSize)])
            let r = Array(testPairs[(i*2*nodeSize + nodeSize)..<(i*2*nodeSize + 2*nodeSize)])
            let cpuH = poseidon2M31Hash(left: l, right: r)
            let gpuH = Array(gpuResults[(i*nodeSize)..<(i*nodeSize + nodeSize)])
            for j in 0..<nodeSize {
                if cpuH[j].v != gpuH[j].v {
                    print("  [FAIL] GPU hash pair \(i), element \(j): CPU=\(cpuH[j].v) GPU=\(gpuH[j].v)")
                    gpuCorrect = false
                }
            }
        }
        if gpuCorrect {
            print("  [pass] GPU matches CPU for \(numTestPairs) test pairs")
        }

        // GPU Merkle tree correctness
        let merkleLeaves = 16  // 16 leaves of 8 M31 each
        var mLeaves = [M31](repeating: M31.zero, count: merkleLeaves * nodeSize)
        rng = 0xDEAD_BEEF
        for i in 0..<mLeaves.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            mLeaves[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
        }
        let gpuRoot = try engine.merkleCommit(leaves: mLeaves)

        // CPU Merkle for comparison
        var level = [[M31]]()
        for i in 0..<merkleLeaves {
            level.append(Array(mLeaves[(i*nodeSize)..<(i*nodeSize + nodeSize)]))
        }
        while level.count > 1 {
            var next = [[M31]]()
            for i in stride(from: 0, to: level.count, by: 2) {
                next.append(poseidon2M31Hash(left: level[i], right: level[i+1]))
            }
            level = next
        }
        let cpuRoot = level[0]

        var merkleCorrect = true
        for j in 0..<nodeSize {
            if cpuRoot[j].v != gpuRoot[j].v {
                print("  [FAIL] Merkle root element \(j): CPU=\(cpuRoot[j].v) GPU=\(gpuRoot[j].v)")
                merkleCorrect = false
            }
        }
        if merkleCorrect {
            print("  [pass] GPU Merkle root matches CPU for \(merkleLeaves) leaves")
        }

        // GPU batch hash pairs benchmark
        print("\n  --- GPU Hash Pairs ---")
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            // n pairs, each is 16 M31 elements
            var input = [M31](repeating: M31.zero, count: n * 2 * nodeSize)
            rng = 0xDEAD_BEEF
            for i in 0..<input.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
            }

            // Warmup
            let _ = try engine.hashPairs(input)

            // Timed
            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashPairs(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let hashPerSec = Double(n) / (median / 1000)
            print(String(format: "  GPU hash 2^%-2d = %6d pairs: %7.2f ms (%8.0f hash/s, %.2f µs/hash)",
                        logN, n, median, hashPerSec, median / Double(n) * 1000))
        }

        // GPU Merkle tree benchmark
        print("\n  --- GPU Merkle Tree ---")
        for logN in [10, 12, 14, 16, 18] {
            let n = 1 << logN
            var mInput = [M31](repeating: M31.zero, count: n * nodeSize)
            rng = 0xBEEF_CAFE
            for i in 0..<mInput.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                mInput[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
            }

            // Warmup
            let _ = try engine.merkleCommit(leaves: mInput)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleCommit(leaves: mInput)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  Merkle 2^%-2d = %6d leaves: %7.2f ms", logN, n, median))
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nPoseidon2 M31 benchmark complete.")
}

/// Batched hash pairs benchmark with batch size sweep
public func runPoseidon2M31BatchedBench() {
    print("=== Poseidon2 M31 Batched Hash Pairs ===")

    do {
        let engine = try Poseidon2M31Engine()
        print("GPU: \(engine.device.name)")
        print("hashPairs maxTG: \(engine.hashPairsMaxTG)")

        // Test different pair counts and batch sizes
        let configs: [(logN: Int, batchSizes: [Int])] = [
            (10, [1, 2, 4, 8, 16]),
            (12, [1, 4, 16, 64]),
            (14, [1, 4, 16, 64]),
            (16, [1, 4, 16, 64]),
            (18, [1, 4, 16, 64]),
        ]

        for (logN, batchSizes) in configs {
            let n = 1 << logN
            let nodeSize = 8

            // Generate input data once
            var input = [M31](repeating: M31.zero, count: n * 2 * nodeSize)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<input.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
            }

            print("\n  --- 2^\(logN) = \(n) pairs ---")
            print("  BS | Time(ms) | Hash/s | vs baseline")
            print("  ---|----------|--------|------------")

            // Baseline (non-batched)
            let _ = try engine.hashPairs(input)
            var baselineTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashPairs(input)
                baselineTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            baselineTimes.sort()
            let baseline = baselineTimes[2]
            let baselineHps = Double(n) / (baseline / 1000)
            print(String(format: "  baseline: %7.2f ms (%8.0f hash/s)", baseline, baselineHps))

            // Batched variants
            for bs in batchSizes {
                if bs == 1 { continue }  // skip bs=1 (same as baseline)

                // Warmup
                let _ = try engine.hashPairsBatched(input, batchSize: bs)

                var times = [Double]()
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.hashPairsBatched(input, batchSize: bs)
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                times.sort()
                let median = times[2]
                let hashPerSec = Double(n) / (median / 1000)
                let speedup = baseline / median
                let marker = speedup > 1.0 ? " ↑\(String(format: "%.2f", speedup))x" : (speedup < 1.0 ? " ↓\(String(format: "%.2f", speedup))x" : "")
                print(String(format: "  %3d |  %6.2f  | %8.0f%@", bs, median, hashPerSec, marker))
            }
        }

        // Correctness check for batched kernel
        print("\n  --- Correctness Check ---")
        let n = 1024
        var input = [M31](repeating: M31.zero, count: n * 2 * 8)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<input.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            input[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
        }

        let baselineResult = try engine.hashPairs(input)
        var allCorrect = true
        for bs in [2, 4, 8, 16] {
            let batchedResult = try engine.hashPairsBatched(input, batchSize: bs)
            var match = true
            for i in 0..<baselineResult.count {
                if baselineResult[i].v != batchedResult[i].v {
                    match = false
                    break
                }
            }
            print("  batchSize=\(bs): \(match ? "[pass]" : "[FAIL]")")
            if !match { allCorrect = false }
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nPoseidon2 M31 batched benchmark complete.")
}

/// Threadgroup size sweep for Poseidon2-M31 hash pairs kernel
public func runPoseidon2M31TGSweep() {
    print("=== Poseidon2 M31 Threadgroup Size Sweep ===")

    do {
        let engine = try Poseidon2M31Engine()
        print("GPU: \(engine.device.name)")
        print("hashPairs maxTG: \(engine.hashPairsMaxTG)")
        print("merkleFused maxTG: \(engine.merkleFusedMaxTG)")

        // Test at different scales and threadgroup sizes
        for logN in [14, 16, 18] {
            let n = 1 << logN
            let nodeSize = 8

            var input = [M31](repeating: M31.zero, count: n * 2 * nodeSize)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<input.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = M31(v: UInt32(truncatingIfNeeded: rng >> 33) % M31.P)
            }

            print("\n  --- 2^\(logN) = \(n) pairs ---")
            print("  TG | Time(ms) | Hash/s")
            print("  ---|----------|---------")

            // Get valid TG sizes to test (must be <= maxTG)
            let maxTG = engine.hashPairsMaxTG
            let tgSizes: [Int]
            if maxTG >= 1024 {
                tgSizes = [32, 64, 128, 256, 512, 1024]
            } else if maxTG >= 512 {
                tgSizes = [32, 64, 128, 256, 512]
            } else {
                tgSizes = [32, 64, 128, 256]
            }

            for tgSize in tgSizes {
                // Warmup
                let _ = try engine.hashPairsCustomTG(input, customTG: tgSize)

                // Timed (3 runs)
                var times = [Double]()
                for _ in 0..<3 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.hashPairsCustomTG(input, customTG: tgSize)
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                times.sort()
                let median = times[1]
                let hashPerSec = Double(n) / (median / 1000)
                let marker = tgSize == 256 ? " ← current" : (median == times.min() ? " ← best" : "")
                print(String(format: "  %3d |  %6.2f  | %8.0f%@", tgSize, median, hashPerSec, marker))
            }
        }
    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nPoseidon2 M31 TG sweep complete.")
}
