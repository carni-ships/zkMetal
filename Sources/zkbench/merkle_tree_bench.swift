// GPU Poseidon2 Merkle Tree Engine Benchmark — GPU vs CPU
// Benchmarks GPUMerkleTreeEngine (structured API) at various tree sizes,
// measuring tree build time and proof verification path time.

import zkMetal
import NeonFieldOps
import Foundation

public func runMerkleTreeBench() {
    print("=== GPU Poseidon2 Merkle Tree Engine Benchmark ===")
    print("    (GPUMerkleTreeEngine: buildTree + proof + verify)")
    print("")

    do {
        let engine = try GPUMerkleTreeEngine()

        // Header
        print(String(format: "  %-12s  %10s  %10s  %10s  %8s  %10s",
                     "Size", "GPU Build", "CPU Build", "Speedup", "Proof", "Verify"))
        print(String(format: "  %-12s  %10s  %10s  %10s  %8s  %10s",
                     "----", "---------", "---------", "-------", "-----", "------"))

        for logN in [10, 14, 18, 20] {
            let n = 1 << logN

            // Generate leaves
            var leaves = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                leaves[i] = frFromInt(rng >> 32)
            }

            // --- GPU buildTree benchmark ---
            // Warmup
            let _ = try engine.buildTree(leaves: leaves)

            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves: leaves)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[2]

            // --- CPU buildTree benchmark (C Poseidon2 Merkle) ---
            var cpuMedian: Double = 0
            if logN <= 18 && !skipCPU {
                let treeSize = 2 * n - 1
                var cpuTree = [Fr](repeating: Fr.zero, count: treeSize)

                // Warmup
                leaves.withUnsafeBytes { lPtr in
                    cpuTree.withUnsafeMutableBytes { tPtr in
                        poseidon2_merkle_tree_cpu(
                            lPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n),
                            tPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }

                var cpuTimes = [Double]()
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    leaves.withUnsafeBytes { lPtr in
                        cpuTree.withUnsafeMutableBytes { tPtr in
                            poseidon2_merkle_tree_cpu(
                                lPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(n),
                                tPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            )
                        }
                    }
                    cpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                cpuTimes.sort()
                cpuMedian = cpuTimes[2]
            }

            // --- Proof extraction benchmark ---
            let tree = try engine.buildTree(leaves: leaves)
            let proofIndex = n / 3  // arbitrary non-trivial index
            var proofTimes = [Double]()
            for _ in 0..<100 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = tree.proof(forLeafAt: proofIndex)
                proofTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000_000) // microseconds
            }
            proofTimes.sort()
            let proofMedianUs = proofTimes[50]

            // --- Verify path benchmark ---
            let proof = tree.proof(forLeafAt: proofIndex)
            let leafVal = tree.leaf(at: proofIndex)
            let root = tree.root
            var verifyTimes = [Double]()
            for _ in 0..<100 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = MerkleTree.verifyPath(root: root, leaf: leafVal,
                                               path: proof.siblings, index: proofIndex)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000_000) // microseconds
            }
            verifyTimes.sort()
            let verifyMedianUs = verifyTimes[50]

            // Format output row
            let sizeStr = String(format: "2^%-2d =%7d", logN, n)
            let gpuStr = String(format: "%7.2f ms", gpuMedian)
            let cpuStr: String
            let speedupStr: String
            if cpuMedian > 0 {
                cpuStr = String(format: "%7.2f ms", cpuMedian)
                speedupStr = String(format: "%5.1fx", cpuMedian / gpuMedian)
            } else {
                cpuStr = "       --"
                speedupStr = "    --"
            }
            let proofStr = String(format: "%5.1f us", proofMedianUs)
            let verifyStr = String(format: "%6.1f us", verifyMedianUs)

            print(String(format: "  %-12s  %10s  %10s  %10s  %8s  %10s",
                         sizeStr, gpuStr, cpuStr, speedupStr, proofStr, verifyStr))
        }

        // --- Correctness checks ---
        print("")
        print("  Correctness:")

        // 1. GPU tree root matches CPU tree root
        let testN = 1024
        let testLeaves = (0..<testN).map { frFromInt(UInt64($0 + 1)) }
        let gpuTree = try engine.buildTree(leaves: testLeaves)
        let gpuRoot = gpuTree.root

        var cpuTreeArr = [Fr](repeating: Fr.zero, count: 2 * testN - 1)
        testLeaves.withUnsafeBytes { lPtr in
            cpuTreeArr.withUnsafeMutableBytes { tPtr in
                poseidon2_merkle_tree_cpu(
                    lPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(testN),
                    tPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        let cpuRoot = cpuTreeArr[2 * testN - 2]
        if frToInt(gpuRoot) == frToInt(cpuRoot) {
            print("  [pass] GPU root matches CPU root (\(testN) leaves)")
        } else {
            print("  [FAIL] GPU root != CPU root!")
            print("    GPU: \(frToInt(gpuRoot).map{String(format:"%016llx",$0)}.joined())")
            print("    CPU: \(frToInt(cpuRoot).map{String(format:"%016llx",$0)}.joined())")
        }

        // 2. Proof verification round-trip
        for idx in [0, testN / 2, testN - 1] {
            let proof = gpuTree.proof(forLeafAt: idx)
            let leaf = gpuTree.leaf(at: idx)
            if proof.verify(root: gpuRoot, leaf: leaf) {
                print("  [pass] Proof verifies for leaf \(idx)")
            } else {
                print("  [FAIL] Proof fails for leaf \(idx)!")
            }
        }

        // 3. Proof fails for wrong leaf
        let badLeaf = frFromInt(0xDEAD)
        let badProof = gpuTree.proof(forLeafAt: 0)
        if !badProof.verify(root: gpuRoot, leaf: badLeaf) {
            print("  [pass] Proof correctly rejects wrong leaf")
        } else {
            print("  [FAIL] Proof accepted wrong leaf!")
        }

        // 4. merkleRoot matches buildTree root
        let rootOnly = try engine.merkleRoot(leaves: testLeaves)
        if frToInt(rootOnly) == frToInt(gpuRoot) {
            print("  [pass] merkleRoot matches buildTree root")
        } else {
            print("  [FAIL] merkleRoot != buildTree root!")
        }

    } catch {
        print("  [FAIL] GPUMerkleTreeEngine: \(error)")
    }

    print("\nMerkle tree engine benchmark complete.")
}
