// Poseidon2 BabyBear Benchmark and correctness test
// t=16, d=7, SP1/Plonky3 compatible
import zkMetal
import Foundation

public func runPoseidon2BabyBearBench() {
    print("=== Poseidon2 BabyBear Benchmark (t=16, x^7, SP1/Plonky3) ===")

    // Basic sanity: permutation of all zeros
    let zeroState = [Bb](repeating: Bb.zero, count: 16)
    let result = poseidon2BbPermutation(zeroState)
    print("  poseidon2_bb([0]*16)[0] = 0x\(String(result[0].v, radix: 16))")

    if result[0].v == 0 && result[1].v == 0 {
        print("  [FAIL] Permutation of zero is zero")
        return
    }
    print("  [pass] Permutation of zero is non-trivial")

    // Determinism
    let result2 = poseidon2BbPermutation(zeroState)
    if result[0].v != result2[0].v || result[7].v != result2[7].v {
        print("  [FAIL] Non-deterministic"); return
    }
    print("  [pass] Deterministic")

    // Permutation of [1, 2, ..., 16]
    var testInput = [Bb](repeating: Bb.zero, count: 16)
    for i in 0..<16 { testInput[i] = Bb(v: UInt32(i + 1)) }
    let testResult = poseidon2BbPermutation(testInput)
    print("  poseidon2_bb([1..16])[0..3] = [0x\(String(testResult[0].v, radix: 16)), 0x\(String(testResult[1].v, radix: 16)), 0x\(String(testResult[2].v, radix: 16)), 0x\(String(testResult[3].v, radix: 16))]")
    print("  [pass] Permutation computed")

    // 2-to-1 hash
    let left = [Bb](repeating: Bb(v: 1), count: 8)
    let right = [Bb](repeating: Bb(v: 2), count: 8)
    let h = poseidon2BbHash(left: left, right: right)
    print("  poseidon2_bb_hash([1]*8, [2]*8)[0..3] = [0x\(String(h[0].v, radix: 16)), 0x\(String(h[1].v, radix: 16)), 0x\(String(h[2].v, radix: 16)), 0x\(String(h[3].v, radix: 16))]")
    print("  [pass] 2-to-1 hash computed")

    // CPU benchmark
    if !skipCPU {
        let warmup = 2000
        var state = zeroState
        for _ in 0..<warmup { state = poseidon2BbPermutation(state) }

        let iters = 10000
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters { state = poseidon2BbPermutation(state) }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let perPerm = elapsed / Double(iters) * 1000
        _ = state
        print(String(format: "\n  CPU permutation: %.2f us/perm (%.0f perm/s)", perPerm, Double(iters) / (elapsed / 1000)))

        let hashIters = 10000
        var acc = left
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<hashIters { acc = poseidon2BbHash(left: acc, right: right) }
        let hElapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let perHash = hElapsed / Double(hashIters) * 1000
        _ = acc
        print(String(format: "  CPU 2-to-1 hash: %.2f us/hash (%.0f hash/s)", perHash, Double(hashIters) / (hElapsed / 1000)))
    }

    // GPU benchmark
    do {
        let engine = try Poseidon2BabyBearEngine()

        // GPU correctness: compare against CPU for hash pairs
        let nodeSize = 8
        let numTestPairs = 4
        var testPairs = [Bb](repeating: Bb.zero, count: numTestPairs * 2 * nodeSize)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<testPairs.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testPairs[i] = Bb(v: UInt32(truncatingIfNeeded: rng >> 33) % Bb.P)
        }

        let gpuResults = try engine.hashPairs(testPairs)
        var gpuCorrect = true
        for i in 0..<numTestPairs {
            let l = Array(testPairs[(i*2*nodeSize)..<(i*2*nodeSize + nodeSize)])
            let r = Array(testPairs[(i*2*nodeSize + nodeSize)..<(i*2*nodeSize + 2*nodeSize)])
            let cpuH = poseidon2BbHash(left: l, right: r)
            let gpuH = Array(gpuResults[(i*nodeSize)..<(i*nodeSize + nodeSize)])
            for j in 0..<nodeSize {
                if cpuH[j].v != gpuH[j].v {
                    print("  [FAIL] GPU hash pair \(i), element \(j): CPU=0x\(String(cpuH[j].v, radix: 16)) GPU=0x\(String(gpuH[j].v, radix: 16))")
                    gpuCorrect = false
                }
            }
        }
        if gpuCorrect {
            print("  [pass] GPU matches CPU for \(numTestPairs) test pairs")
        }

        // GPU Merkle tree correctness
        let merkleLeaves = 16
        var mLeaves = [Bb](repeating: Bb.zero, count: merkleLeaves * nodeSize)
        rng = 0xDEAD_BEEF
        for i in 0..<mLeaves.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            mLeaves[i] = Bb(v: UInt32(truncatingIfNeeded: rng >> 33) % Bb.P)
        }
        let gpuRoot = try engine.merkleCommit(leaves: mLeaves)

        // CPU Merkle for comparison
        var level = [[Bb]]()
        for i in 0..<merkleLeaves {
            level.append(Array(mLeaves[(i*nodeSize)..<(i*nodeSize + nodeSize)]))
        }
        while level.count > 1 {
            var next = [[Bb]]()
            for i in stride(from: 0, to: level.count, by: 2) {
                next.append(poseidon2BbHash(left: level[i], right: level[i+1]))
            }
            level = next
        }
        let cpuRoot = level[0]

        var merkleCorrect = true
        for j in 0..<nodeSize {
            if cpuRoot[j].v != gpuRoot[j].v {
                print("  [FAIL] Merkle root element \(j): CPU=0x\(String(cpuRoot[j].v, radix: 16)) GPU=0x\(String(gpuRoot[j].v, radix: 16))")
                merkleCorrect = false
            }
        }
        if merkleCorrect {
            print("  [pass] GPU Merkle root matches CPU for \(merkleLeaves) leaves")
        }

        // GPU batch hash pairs benchmark
        print("\n  --- GPU Hash Pairs ---")
        for logN in [12, 14, 16, 18, 20] {
            let n = 1 << logN
            var input = [Bb](repeating: Bb.zero, count: n * 2 * nodeSize)
            rng = 0xDEAD_BEEF
            for i in 0..<input.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = Bb(v: UInt32(truncatingIfNeeded: rng >> 33) % Bb.P)
            }

            // Warmup
            let _ = try engine.hashPairs(input)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashPairs(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let hashPerSec = Double(n) / (median / 1000)
            print(String(format: "  GPU hash 2^%-2d = %7d pairs: %7.2f ms (%8.0f hash/s, %.2f us/hash)",
                        logN, n, median, hashPerSec, median / Double(n) * 1000))
        }

        // GPU Merkle tree benchmark
        print("\n  --- GPU Merkle Tree ---")
        for logN in [12, 14, 16, 18, 20] {
            let n = 1 << logN
            var mInput = [Bb](repeating: Bb.zero, count: n * nodeSize)
            rng = 0xBEEF_CAFE
            for i in 0..<mInput.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                mInput[i] = Bb(v: UInt32(truncatingIfNeeded: rng >> 33) % Bb.P)
            }

            let _ = try engine.merkleCommit(leaves: mInput)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleCommit(leaves: mInput)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  Merkle 2^%-2d = %7d leaves: %7.2f ms", logN, n, median))
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nPoseidon2 BabyBear benchmark complete.")
}
