// Merkle Tree Benchmark
import zkMetal
import Foundation

public func runMerkleBench() {
    print("=== Merkle Tree Benchmark ===")

    // Poseidon2 Merkle
    do {
        let engine = try Poseidon2MerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Warmup
            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            // CPU Merkle tree
            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                var level = leaves
                while level.count > 1 {
                    var next = [Fr]()
                    next.reserveCapacity(level.count / 2)
                    for i in stride(from: 0, to: level.count, by: 2) {
                        next.append(poseidon2Hash(level[i], level[i+1]))
                    }
                    level = next
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  Poseidon2 Merkle 2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.0f ms | %.0fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Poseidon2 Merkle 2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }
        }

        // Correctness: tree[2n-2] should be deterministic
        let testLeaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree = try engine.buildTree(testLeaves)
        let root = frToInt(tree.last!)
        print("  Root(1,2,3,4) = \(root.map{String(format:"%016llx",$0)}.joined())")
        print("  [pass] Poseidon2 Merkle tree")

        // Fused merkleRoot benchmark (root-only, uses fused subtree kernel)
        print("")
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Warmup
            let _ = try engine.merkleRoot(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleRoot(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  P2 MerkleRoot  2^%-2d = %6d leaves: %7.2f ms (fused subtrees)",
                        logN, n, median))
        }

        // Correctness: merkleRoot must match buildTree root
        let testLeaves2 = (0..<2048).map { frFromInt(UInt64($0 + 1)) }
        let fusedRoot = try engine.merkleRoot(testLeaves2)
        let fullTree = try engine.buildTree(testLeaves2)
        if frToInt(fusedRoot) == frToInt(fullTree.last!) {
            print("  [pass] Fused merkleRoot matches buildTree root (2048 leaves)")
        } else {
            print("  [FAIL] Fused merkleRoot mismatch!")
            print("    fused: \(frToInt(fusedRoot).map{String(format:"%016llx",$0)}.joined())")
            print("    tree:  \(frToInt(fullTree.last!).map{String(format:"%016llx",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Poseidon2 Merkle: \(error)")
    }

    // Keccak Merkle
    do {
        let engine = try KeccakMerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [[UInt8]]()
            for i in 0..<n {
                var leaf = [UInt8](repeating: 0, count: 32)
                let val = UInt64(i)
                for b in 0..<8 { leaf[b] = UInt8((val >> (b * 8)) & 0xFF) }
                leaves.append(leaf)
            }

            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            // CPU Keccak Merkle tree
            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                var level = leaves
                while level.count > 1 {
                    var next = [[UInt8]]()
                    next.reserveCapacity(level.count / 2)
                    for i in stride(from: 0, to: level.count, by: 2) {
                        next.append(keccak256(level[i] + level[i+1]))
                    }
                    level = next
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  Keccak Merkle   2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.0f ms | %.0fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Keccak Merkle   2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }

            // merkleRoot timing (avoids output copy)
            let _ = try engine.merkleRoot(leaves)
            var rootTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleRoot(leaves)
                rootTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            rootTimes.sort()
            let rootMedian = rootTimes[2]
            let copyOverhead = gpuMedian - rootMedian
            print(String(format: "    merkleRoot:  %7.2f ms  (output copy overhead: %.1f ms)", rootMedian, copyOverhead))
        }
        // Correctness: verify GPU Merkle root matches CPU level-by-level
        let testN = 1024
        var testLeaves = [[UInt8]]()
        for i in 0..<testN {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            testLeaves.append(leaf)
        }
        let gpuTree = try engine.buildTree(testLeaves)
        let gpuRoot = KeccakMerkleEngine.node(gpuTree, at: 2 * testN - 2)
        var cpuNodes = testLeaves
        while cpuNodes.count > 1 {
            var next = [[UInt8]]()
            for i in stride(from: 0, to: cpuNodes.count, by: 2) {
                next.append(keccak256(cpuNodes[i] + cpuNodes[i+1]))
            }
            cpuNodes = next
        }
        if cpuNodes[0] == gpuRoot {
            print("  [pass] Keccak fused Merkle root matches CPU (\(testN) leaves)")
        } else {
            print("  [FAIL] Keccak fused root mismatch!")
            print("    CPU: \(cpuNodes[0].map{String(format:"%02x",$0)}.joined())")
            print("    GPU: \(gpuRoot.map{String(format:"%02x",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Keccak Merkle: \(error)")
    }

    // Blake3 Merkle
    do {
        let engine = try Blake3MerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [[UInt8]]()
            for i in 0..<n {
                var leaf = [UInt8](repeating: 0, count: 32)
                let val = UInt64(i)
                for b in 0..<8 { leaf[b] = UInt8((val >> (b * 8)) & 0xFF) }
                leaves.append(leaf)
            }

            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                var level = leaves
                while level.count > 1 {
                    var next = [[UInt8]]()
                    next.reserveCapacity(level.count / 2)
                    for i in stride(from: 0, to: level.count, by: 2) {
                        next.append(blake3Parent(level[i] + level[i+1]))
                    }
                    level = next
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  Blake3 Merkle   2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.0f ms | %.0fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Blake3 Merkle   2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }
        }

        // Correctness: verify GPU Merkle root matches CPU level-by-level
        let testN = 1024
        var testLeaves = [[UInt8]]()
        for i in 0..<testN {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            testLeaves.append(leaf)
        }
        let gpuTree = try engine.buildTree(testLeaves)
        let gpuRoot = Blake3MerkleEngine.node(gpuTree, at: 2 * testN - 2)
        var cpuNodes = testLeaves
        while cpuNodes.count > 1 {
            var next = [[UInt8]]()
            for i in stride(from: 0, to: cpuNodes.count, by: 2) {
                next.append(blake3Parent(cpuNodes[i] + cpuNodes[i+1]))
            }
            cpuNodes = next
        }
        if cpuNodes[0] == gpuRoot {
            print("  [pass] Blake3 Merkle root matches CPU (\(testN) leaves)")
        } else {
            print("  [FAIL] Blake3 root mismatch!")
            print("    CPU: \(cpuNodes[0].map{String(format:"%02x",$0)}.joined())")
            print("    GPU: \(gpuRoot.map{String(format:"%02x",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Blake3 Merkle: \(error)")
    }

    print("\nMerkle benchmark complete.")
}
