// SHA-256 Benchmark and correctness test
import zkMetal
import Foundation

public func runSHA256Bench() {
    print("=== SHA-256 Benchmark ===")

    // --- NIST Test Vectors ---

    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    let emptyHash = sha256([])
    let emptyHex = emptyHash.map { String(format: "%02x", $0) }.joined()
    let expectedEmpty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    print("  sha256(\"\") = \(emptyHex)")
    if emptyHex == expectedEmpty {
        print("  [pass] Empty string hash matches NIST")
    } else {
        print("  [FAIL] Expected \(expectedEmpty)")
        return
    }

    // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    let abcHash = sha256([0x61, 0x62, 0x63])
    let abcHex = abcHash.map { String(format: "%02x", $0) }.joined()
    let expectedAbc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    print("  sha256(\"abc\") = \(abcHex)")
    if abcHex == expectedAbc {
        print("  [pass] 'abc' hash matches NIST")
    } else {
        print("  [FAIL] Expected \(expectedAbc)")
        return
    }

    // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
    // = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
    let longInput: [UInt8] = Array("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".utf8)
    let longHash = sha256(longInput)
    let longHex = longHash.map { String(format: "%02x", $0) }.joined()
    let expectedLong = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    print("  sha256(\"abcdbcde...\") = \(longHex)")
    if longHex == expectedLong {
        print("  [pass] 448-bit message hash matches NIST")
    } else {
        print("  [FAIL] Expected \(expectedLong)")
        return
    }

    // CPU benchmark: hash 64-byte inputs
    if !skipCPU {
        let cpuInput = [UInt8](repeating: 0x42, count: 64)
        let warmup = 5000
        for _ in 0..<warmup { let _ = sha256(cpuInput) }
        let cpuIters = 10000
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<cpuIters { let _ = sha256(cpuInput) }
        let cpuElapsed = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000
        let cpuPerHash = cpuElapsed / Double(cpuIters) * 1000
        print(String(format: "\n  CPU: %.2f us/hash (%.0f hash/s)", cpuPerHash, Double(cpuIters) / (cpuElapsed / 1000)))
    }

    // GPU tests
    do {
        let engine = try SHA256Engine()

        // GPU correctness: compare 64-byte hash against CPU
        var testInput = [UInt8](repeating: 0, count: 64 * 4)
        for i in 0..<4 {
            for j in 0..<64 { testInput[i * 64 + j] = UInt8((i * 64 + j) & 0xFF) }
        }
        let gpuResults = try engine.hashBatch(testInput)

        var gpuCorrect = true
        for i in 0..<4 {
            let cpuRef = sha256(Array(testInput[i*64..<(i+1)*64]))
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

        // GPU hashPairs correctness
        var pairInput = [UInt8](repeating: 0, count: 64 * 2)
        for i in 0..<128 { pairInput[i] = UInt8(i & 0xFF) }
        let gpuPairResults = try engine.hashPairs(pairInput)
        var pairCorrect = true
        for i in 0..<2 {
            let cpuRef = sha256(Array(pairInput[i*64..<(i+1)*64]))
            let gpuSlice = Array(gpuPairResults[i*32..<(i+1)*32])
            if cpuRef != gpuSlice {
                print("  [FAIL] GPU hashPairs \(i) mismatch")
                pairCorrect = false
            }
        }
        if pairCorrect {
            print("  [pass] GPU hashPairs matches CPU")
        }

        // GPU batch benchmarks
        print("\n  --- Batch Hash (64-byte inputs) ---")
        for logN in [12, 14, 16, 18, 20] {
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
            let _ = try engine.hashBatch(input)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashBatch(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            let hashPerSec = Double(n) / (median / 1000)
            print(String(format: "  GPU batch 2^%-2d = %7d: %8.2f ms (%10.0f hash/s, %.3f us/hash)",
                        logN, n, median, hashPerSec, median / Double(n) * 1000))
        }

        // Merkle tree benchmarks
        print("\n  --- Merkle Tree (SHA-256) ---")
        let merkle = try SHA256MerkleEngine()

        for logN in [12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [[UInt8]]()
            leaves.reserveCapacity(n)
            var rng: UInt64 = 0xDEADBEEF
            for _ in 0..<n {
                var leaf = [UInt8](repeating: 0, count: 32)
                for j in 0..<32 {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    leaf[j] = UInt8((rng >> 32) & 0xFF)
                }
                leaves.append(leaf)
            }

            // Warmup
            let _ = try merkle.merkleRoot(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try merkle.merkleRoot(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  Merkle 2^%-2d = %7d leaves: %8.2f ms",
                        logN, n, median))
        }

        // Merkle correctness: build tree and verify root matches manual computation
        print("\n  --- Merkle Correctness ---")
        let testLeaves: [[UInt8]] = (0..<4).map { i in
            var leaf = [UInt8](repeating: 0, count: 32)
            leaf[0] = UInt8(i)
            return leaf
        }
        let tree = try merkle.buildTree(testLeaves)
        let gpuRoot = SHA256MerkleEngine.node(tree, at: 6) // 2*4-2 = 6

        // Manual: hash pairs bottom-up
        let h01 = sha256(testLeaves[0] + testLeaves[1])
        let h23 = sha256(testLeaves[2] + testLeaves[3])
        let cpuRoot = sha256(h01 + h23)

        if gpuRoot == cpuRoot {
            print("  [pass] Merkle root matches manual CPU computation")
        } else {
            print("  [FAIL] Merkle root mismatch")
            print("    GPU: \(gpuRoot.map { String(format: "%02x", $0) }.joined())")
            print("    CPU: \(cpuRoot.map { String(format: "%02x", $0) }.joined())")
        }

    } catch {
        print("  [FAIL] GPU error: \(error)")
    }

    print("\nSHA-256 benchmark complete.")
}
