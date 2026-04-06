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
