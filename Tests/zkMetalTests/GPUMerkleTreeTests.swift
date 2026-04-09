// GPU Merkle Tree Engine Tests
import zkMetal
import Foundation

func runGPUMerkleTreeTests() {
    suite("GPUMerkleTreeEngine")

    do {
        let engine = try GPUMerkleTreeEngine()

        // Test 1: Small tree (4 leaves)
        let leaves4 = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree4 = try engine.buildTree(leaves: leaves4)
        expectEqual(tree4.leafCount, 4, "leafCount == 4")
        expectEqual(tree4.nodes.count, 7, "4-leaf tree has 7 nodes")
        expectEqual(tree4.depth, 2, "depth == 2")

        // Verify leaves are preserved
        for i in 0..<4 {
            expect(frToInt(tree4.leaf(at: i)) == frToInt(leaves4[i]),
                   "leaf[\(i)] preserved")
        }

        // Verify root matches CPU computation
        let h01 = poseidon2Hash(leaves4[0], leaves4[1])
        let h23 = poseidon2Hash(leaves4[2], leaves4[3])
        let expectedRoot = poseidon2Hash(h01, h23)
        expect(frToInt(tree4.root) == frToInt(expectedRoot), "root matches CPU")

        // Test 2: Proof generation and verification (4 leaves)
        for i in 0..<4 {
            let proof = tree4.proof(forLeafAt: i)
            expectEqual(proof.depth, 2, "proof depth == 2 for leaf \(i)")
            let valid = proof.verify(root: tree4.root, leaf: leaves4[i])
            expect(valid, "proof valid for leaf \(i)")
        }

        // Test 3: verifyPath static method
        let proof0 = tree4.proof(forLeafAt: 0)
        expect(MerkleTree.verifyPath(root: tree4.root, leaf: leaves4[0],
                                     path: proof0.siblings, index: 0),
               "static verifyPath works")

        // Test 4: Wrong leaf should fail verification
        let wrongLeaf = frFromInt(999)
        expect(!proof0.verify(root: tree4.root, leaf: wrongLeaf),
               "wrong leaf fails verification")

        // Test 5: Wrong root should fail verification
        let wrongRoot = frFromInt(888)
        expect(!proof0.verify(root: wrongRoot, leaf: leaves4[0]),
               "wrong root fails verification")

        // Test 6: Medium tree (64 leaves)
        let n64 = 64
        let leaves64 = (0..<n64).map { frFromInt(UInt64($0 + 1)) }
        let tree64 = try engine.buildTree(leaves: leaves64)
        expectEqual(tree64.leafCount, n64, "leafCount == 64")
        expectEqual(tree64.nodes.count, 2 * n64 - 1, "64-leaf tree has 127 nodes")
        expectEqual(tree64.depth, 6, "depth == 6")

        // Verify all proofs for medium tree
        var allProofsValid = true
        for i in 0..<n64 {
            let proof = tree64.proof(forLeafAt: i)
            if !proof.verify(root: tree64.root, leaf: leaves64[i]) {
                allProofsValid = false
                break
            }
        }
        expect(allProofsValid, "all 64 proofs valid")

        // Test 7: merkleRoot matches buildTree root
        let rootOnly = try engine.merkleRoot(leaves: leaves64)
        expect(frToInt(rootOnly) == frToInt(tree64.root),
               "merkleRoot == buildTree root")

        // Test 8: Large tree (1024 leaves — triggers fused subtree kernel)
        let n1024 = 1024
        let leaves1024 = (0..<n1024).map { frFromInt(UInt64($0 + 1)) }
        let tree1024 = try engine.buildTree(leaves: leaves1024)
        expectEqual(tree1024.leafCount, n1024, "leafCount == 1024")
        expectEqual(tree1024.depth, 10, "depth == 10")

        // Spot-check a few proofs in the large tree
        for i in [0, 1, 512, 1023] {
            let proof = tree1024.proof(forLeafAt: i)
            expect(proof.verify(root: tree1024.root, leaf: leaves1024[i]),
                   "1024-leaf proof valid for leaf \(i)")
        }

        // Test 9: buildTreeLevel (one level)
        let children = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let parents = try engine.buildTreeLevel(children: children)
        expectEqual(parents.count, 2, "2 parents from 4 children")
        let expP0 = poseidon2Hash(children[0], children[1])
        let expP1 = poseidon2Hash(children[2], children[3])
        expect(frToInt(parents[0]) == frToInt(expP0), "parent[0] matches CPU")
        expect(frToInt(parents[1]) == frToInt(expP1), "parent[1] matches CPU")

        // Test 10: Batch tree building (multiple small trees)
        let batchLeaves: [[Fr]] = [
            (0..<4).map { frFromInt(UInt64($0 + 100)) },
            (0..<8).map { frFromInt(UInt64($0 + 200)) },
            (0..<16).map { frFromInt(UInt64($0 + 300)) },
        ]
        let batchRoots = try engine.buildTreesBatch(treesLeaves: batchLeaves)
        expectEqual(batchRoots.count, 3, "3 batch roots")

        // Verify batch roots match individual builds
        for (i, leaves) in batchLeaves.enumerated() {
            let singleTree = try engine.buildTree(leaves: leaves)
            expect(frToInt(batchRoots[i]) == frToInt(singleTree.root),
                   "batch root[\(i)] matches single build")
        }

        // Test 11: Determinism — building same tree twice gives same root
        let tree1024b = try engine.buildTree(leaves: leaves1024)
        expect(frToInt(tree1024.root) == frToInt(tree1024b.root),
               "deterministic: same leaves -> same root")

        // Test 12: 4096 leaves (triggers fused subtrees + level-by-level upper)
        let n4096 = 4096
        let leaves4096 = (0..<n4096).map { frFromInt(UInt64($0 + 1)) }
        let tree4096 = try engine.buildTree(leaves: leaves4096)
        expectEqual(tree4096.leafCount, n4096, "leafCount == 4096")
        let root4096 = try engine.merkleRoot(leaves: leaves4096)
        expect(frToInt(tree4096.root) == frToInt(root4096),
               "4096-leaf: buildTree root == merkleRoot")

        // Spot-check proof at boundary
        let proofLast = tree4096.proof(forLeafAt: n4096 - 1)
        expect(proofLast.verify(root: tree4096.root, leaf: leaves4096[n4096 - 1]),
               "4096-leaf: last leaf proof valid")

    } catch {
        expect(false, "GPUMerkleTreeEngine error: \(error)")
    }
}

func runKeccak4aryMerkleTests() {
    suite("Keccak4aryMerkleEngine")

    do {
        let engine4 = try Keccak4aryMerkleEngine()
        let engine2 = try KeccakMerkleEngine()

        // Helper: CPU Keccak hash of 4 children (128 bytes)
        func keccak4ary(_ a: [UInt8], _ b: [UInt8], _ c: [UInt8], _ d: [UInt8]) -> [UInt8] {
            return keccak256(a + b + c + d)
        }

        // Test 1: 4 leaves — single 4-ary hash
        var leaves4 = [[UInt8]]()
        for i in 0..<4 {
            var leaf = [UInt8](repeating: 0, count: 32)
            leaf[0] = UInt8(i + 1)
            leaves4.append(leaf)
        }
        let tree4 = try engine4.buildTree(leaves4)
        let root4 = try engine4.merkleRoot(leaves4)
        let expectedRoot4 = keccak4ary(leaves4[0], leaves4[1], leaves4[2], leaves4[3])
        let treeNodes4 = tree4.count / 32
        expectEqual(treeNodes4, 5, "4 leaves: 5 total nodes (4 leaves + 1 root)")
        let gpuRoot4 = Keccak4aryMerkleEngine.node(tree4, at: treeNodes4 - 1)
        expect(gpuRoot4 == expectedRoot4, "4-leaf 4-ary root matches CPU keccak256(4 children)")
        expect(root4 == expectedRoot4, "4-leaf merkleRoot matches CPU")

        // Test 2: 16 leaves — 2 levels of 4-ary
        var leaves16 = [[UInt8]]()
        for i in 0..<16 {
            var leaf = [UInt8](repeating: 0, count: 32)
            leaf[0] = UInt8(i)
            leaves16.append(leaf)
        }
        let tree16 = try engine4.buildTree(leaves16)
        let treeNodes16 = tree16.count / 32
        // 16 leaves + 4 internal + 1 root = 21
        expectEqual(treeNodes16, 21, "16 leaves: 21 total nodes")

        // CPU reference for 16 leaves, 4-ary
        var level1 = [[UInt8]]()
        for i in stride(from: 0, to: 16, by: 4) {
            level1.append(keccak4ary(leaves16[i], leaves16[i+1], leaves16[i+2], leaves16[i+3]))
        }
        let cpuRoot16 = keccak4ary(level1[0], level1[1], level1[2], level1[3])
        let gpuRoot16 = Keccak4aryMerkleEngine.node(tree16, at: treeNodes16 - 1)
        expect(gpuRoot16 == cpuRoot16, "16-leaf 4-ary root matches CPU")

        // Test 3: 2 leaves — binary fallback
        var leaves2 = [[UInt8]]()
        for i in 0..<2 {
            var leaf = [UInt8](repeating: 0, count: 32)
            leaf[0] = UInt8(i + 10)
            leaves2.append(leaf)
        }
        let tree2 = try engine4.buildTree(leaves2)
        let treeNodes2 = tree2.count / 32
        expectEqual(treeNodes2, 3, "2 leaves: 3 total nodes")
        let gpuRoot2 = Keccak4aryMerkleEngine.node(tree2, at: 2)
        let cpuRoot2 = keccak256(leaves2[0] + leaves2[1])
        expect(gpuRoot2 == cpuRoot2, "2-leaf falls back to binary, root matches CPU")

        // Test 4: 8 leaves — first level is 4-ary (8->2), second is binary (2->1)
        var leaves8 = [[UInt8]]()
        for i in 0..<8 {
            var leaf = [UInt8](repeating: 0, count: 32)
            leaf[0] = UInt8(i + 20)
            leaves8.append(leaf)
        }
        let tree8 = try engine4.buildTree(leaves8)
        let treeNodes8 = tree8.count / 32
        // 8 + 2 + 1 = 11
        expectEqual(treeNodes8, 11, "8 leaves: 11 total nodes")
        let h01 = keccak4ary(leaves8[0], leaves8[1], leaves8[2], leaves8[3])
        let h23 = keccak4ary(leaves8[4], leaves8[5], leaves8[6], leaves8[7])
        let cpuRoot8 = keccak256(h01 + h23)
        let gpuRoot8 = Keccak4aryMerkleEngine.node(tree8, at: treeNodes8 - 1)
        expect(gpuRoot8 == cpuRoot8, "8-leaf 4-ary root matches CPU (mixed 4-ary + binary)")

        // Test 5: 256 leaves — larger tree
        var leaves256 = [[UInt8]]()
        for i in 0..<256 {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            leaves256.append(leaf)
        }
        let root256 = try engine4.merkleRoot(leaves256)

        // CPU reference: 4-ary tree
        var cpuLevel = leaves256
        while cpuLevel.count > 1 {
            if cpuLevel.count >= 4 {
                var next = [[UInt8]]()
                next.reserveCapacity(cpuLevel.count / 4)
                for i in stride(from: 0, to: cpuLevel.count, by: 4) {
                    next.append(keccak4ary(cpuLevel[i], cpuLevel[i+1], cpuLevel[i+2], cpuLevel[i+3]))
                }
                cpuLevel = next
            } else {
                // cpuLevel.count == 2
                cpuLevel = [keccak256(cpuLevel[0] + cpuLevel[1])]
            }
        }
        expect(root256 == cpuLevel[0], "256-leaf 4-ary root matches CPU")

        // Test 6: 1024 leaves
        var leaves1024 = [[UInt8]]()
        for i in 0..<1024 {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            leaves1024.append(leaf)
        }
        let root1024 = try engine4.merkleRoot(leaves1024)

        // CPU reference for 1024 leaves
        cpuLevel = leaves1024
        while cpuLevel.count > 1 {
            if cpuLevel.count >= 4 {
                var next = [[UInt8]]()
                next.reserveCapacity(cpuLevel.count / 4)
                for i in stride(from: 0, to: cpuLevel.count, by: 4) {
                    next.append(keccak4ary(cpuLevel[i], cpuLevel[i+1], cpuLevel[i+2], cpuLevel[i+3]))
                }
                cpuLevel = next
            } else {
                cpuLevel = [keccak256(cpuLevel[0] + cpuLevel[1])]
            }
        }
        expect(root1024 == cpuLevel[0], "1024-leaf 4-ary root matches CPU")

        // Test 7: Determinism
        let root1024b = try engine4.merkleRoot(leaves1024)
        expect(root1024 == root1024b, "deterministic: same leaves -> same root")

        // Test 8: 4-ary root differs from binary root (they are different tree structures)
        let binRoot = try engine2.merkleRoot(leaves16)
        let aryRoot = try engine4.merkleRoot(leaves16)
        expect(binRoot != aryRoot, "4-ary root differs from binary root (different structure)")

    } catch {
        expect(false, "Keccak4aryMerkleEngine error: \(error)")
    }
}
