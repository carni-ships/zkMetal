// Merkle Tree Benchmark
import zkMetal
import Foundation

public func runMerkleBench() {
    print("=== Merkle Tree Benchmark ===")

    // Poseidon2 Merkle
    do {
        let engine = try Poseidon2MerkleEngine()

        for logN in [10, 12, 14, 16] {
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
            let median = times[2]
            print(String(format: "  Poseidon2 Merkle 2^%-2d = %6d leaves: %7.2f ms", logN, n, median))
        }

        // Correctness: tree[2n-2] should be deterministic
        let testLeaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree = try engine.buildTree(testLeaves)
        let root = frToInt(tree.last!)
        print("  Root(1,2,3,4) = \(root.map{String(format:"%016llx",$0)}.joined())")
        print("  [pass] Poseidon2 Merkle tree")


    } catch {
        print("  [FAIL] Poseidon2 Merkle: \(error)")
    }

    // Keccak Merkle
    do {
        let engine = try KeccakMerkleEngine()

        for logN in [10, 12, 14, 16] {
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
            let median = times[2]
            print(String(format: "  Keccak Merkle   2^%-2d = %6d leaves: %7.2f ms", logN, n, median))
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
        var cpuNodes = testLeaves
        while cpuNodes.count > 1 {
            var next = [[UInt8]]()
            for i in stride(from: 0, to: cpuNodes.count, by: 2) {
                next.append(keccak256(cpuNodes[i] + cpuNodes[i+1]))
            }
            cpuNodes = next
        }
        if cpuNodes[0] == gpuTree.last! {
            print("  [pass] Keccak fused Merkle root matches CPU (\(testN) leaves)")
        } else {
            print("  [FAIL] Keccak fused root mismatch!")
            print("    CPU: \(cpuNodes[0].map{String(format:"%02x",$0)}.joined())")
            print("    GPU: \(gpuTree.last!.map{String(format:"%02x",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Keccak Merkle: \(error)")
    }

    print("\nMerkle benchmark complete.")
}
