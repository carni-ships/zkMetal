// GPU Merkle Batch Proof Engine Tests
import zkMetal
import Foundation
import Metal

public func runGPUMerkleBatchProofTests() {
    suite("GPUMerkleBatchProofEngine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUMerkleBatchProofEngine() else {
        print("  [SKIP] Failed to create GPUMerkleBatchProofEngine")
        return
    }

    do {
        let merkle = engine.merkleEngine

        // ================================================================
        // Test 1: Single-leaf batch proof (degenerate case)
        // ================================================================
        suite("GPUMerkleBatchProof — Single Leaf")

        let leaves4 = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree4 = try merkle.buildTree(leaves: leaves4)

        let singleProof = engine.generateBatchProof(tree: tree4, leafIndices: [2])
        expectEqual(singleProof.leafIndices.count, 1, "single leaf proof has 1 index")
        expectEqual(singleProof.leafIndices[0], 2, "leaf index == 2")
        expectEqual(singleProof.leafCount, 4, "leafCount == 4")
        expectEqual(singleProof.depth, 2, "depth == 2")

        let singleValid = engine.verifyBatchProof(singleProof)
        expect(singleValid, "single leaf batch proof verifies")

        // ================================================================
        // Test 2: Two-leaf batch proof (adjacent leaves share a parent)
        // ================================================================
        suite("GPUMerkleBatchProof — Adjacent Leaves")

        let adjProof = engine.generateBatchProof(tree: tree4, leafIndices: [0, 1])
        expectEqual(adjProof.leafIndices.count, 2, "adjacent proof has 2 indices")

        // Adjacent leaves (0,1) share parent — their sibling is each other,
        // so we only need the sibling of the parent at level 1.
        // auxiliaryCount should be 1 (just h(leaf[2], leaf[3]))
        expectEqual(adjProof.auxiliaryCount, 1, "adjacent leaves share parent: 1 aux node")

        let adjValid = engine.verifyBatchProof(adjProof)
        expect(adjValid, "adjacent leaf batch proof verifies")

        // ================================================================
        // Test 3: Two-leaf batch proof (non-adjacent, different subtrees)
        // ================================================================
        suite("GPUMerkleBatchProof — Non-Adjacent Leaves")

        let naProof = engine.generateBatchProof(tree: tree4, leafIndices: [0, 3])
        expectEqual(naProof.leafIndices.count, 2, "non-adjacent proof has 2 indices")
        // leaf 0 needs sibling leaf 1, leaf 3 needs sibling leaf 2 => 2 aux at level 0
        // At level 1, both parents are known => 0 aux
        expectEqual(naProof.auxiliaryCount, 2, "non-adjacent leaves: 2 aux nodes")

        let naValid = engine.verifyBatchProof(naProof)
        expect(naValid, "non-adjacent leaf batch proof verifies")

        // ================================================================
        // Test 4: All-leaves batch proof (minimal auxiliary)
        // ================================================================
        suite("GPUMerkleBatchProof — All Leaves")

        let allProof = engine.generateBatchProof(tree: tree4, leafIndices: [0, 1, 2, 3])
        expectEqual(allProof.leafIndices.count, 4, "all-leaf proof has 4 indices")
        expectEqual(allProof.auxiliaryCount, 0, "all leaves known: 0 aux nodes")

        let allValid = engine.verifyBatchProof(allProof)
        expect(allValid, "all-leaf batch proof verifies")

        // ================================================================
        // Test 5: Medium tree (64 leaves, batch of 8)
        // ================================================================
        suite("GPUMerkleBatchProof — Medium Tree")

        let leaves64 = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let tree64 = try merkle.buildTree(leaves: leaves64)

        let indices8 = [0, 7, 15, 16, 31, 32, 48, 63]
        let med8Proof = engine.generateBatchProof(tree: tree64, leafIndices: indices8)
        expectEqual(med8Proof.leafIndices.count, 8, "8-leaf batch has 8 indices")
        expect(med8Proof.auxiliaryCount < 8 * 6, "compressed: fewer aux than 8 * depth(6)")

        let med8Valid = engine.verifyBatchProof(med8Proof)
        expect(med8Valid, "medium tree 8-leaf batch proof verifies")

        // ================================================================
        // Test 6: Duplicate indices handled correctly
        // ================================================================
        suite("GPUMerkleBatchProof — Duplicate Indices")

        let dupProof = engine.generateBatchProof(tree: tree4, leafIndices: [1, 1, 1])
        expectEqual(dupProof.leafIndices.count, 1, "duplicates deduplicated to 1")
        let dupValid = engine.verifyBatchProof(dupProof)
        expect(dupValid, "deduplicated proof verifies")

        // ================================================================
        // Test 7: Tampered leaf fails verification
        // ================================================================
        suite("GPUMerkleBatchProof — Tampered Leaf")

        var tamperedProof = singleProof
        var badLeafValues = tamperedProof.leafValues
        // Flip a byte in the leaf value
        if !badLeafValues.isEmpty {
            badLeafValues[0] ^= 0xFF
        }
        tamperedProof = MerkleBatchProof(
            backend: tamperedProof.backend,
            leafCount: tamperedProof.leafCount,
            leafIndices: tamperedProof.leafIndices,
            leafValues: badLeafValues,
            auxiliaryNodes: tamperedProof.auxiliaryNodes,
            auxiliaryCount: tamperedProof.auxiliaryCount,
            root: tamperedProof.root
        )
        let tamperedValid = engine.verifyBatchProof(tamperedProof)
        expect(!tamperedValid, "tampered leaf fails verification")

        // ================================================================
        // Test 8: Wrong root fails verification
        // ================================================================
        suite("GPUMerkleBatchProof — Wrong Root")

        var badRoot = singleProof.root
        if !badRoot.isEmpty { badRoot[0] ^= 0xFF }
        let wrongRootProof = MerkleBatchProof(
            backend: singleProof.backend,
            leafCount: singleProof.leafCount,
            leafIndices: singleProof.leafIndices,
            leafValues: singleProof.leafValues,
            auxiliaryNodes: singleProof.auxiliaryNodes,
            auxiliaryCount: singleProof.auxiliaryCount,
            root: badRoot
        )
        expect(!engine.verifyBatchProof(wrongRootProof), "wrong root fails verification")

        // ================================================================
        // Test 9: Serialization / Deserialization roundtrip
        // ================================================================
        suite("GPUMerkleBatchProof — Serialization")

        let serialized = med8Proof.serialize()
        expect(serialized.count > 0, "serialized bytes non-empty")

        guard let deserialized = MerkleBatchProof.deserialize(serialized) else {
            expect(false, "deserialization returned nil")
            return
        }

        expectEqual(deserialized.leafCount, med8Proof.leafCount, "roundtrip leafCount")
        expectEqual(deserialized.leafIndices, med8Proof.leafIndices, "roundtrip leafIndices")
        expectEqual(deserialized.auxiliaryCount, med8Proof.auxiliaryCount, "roundtrip auxCount")
        expectEqual(deserialized.leafValues, med8Proof.leafValues, "roundtrip leafValues")
        expectEqual(deserialized.auxiliaryNodes, med8Proof.auxiliaryNodes, "roundtrip auxNodes")
        expectEqual(deserialized.root, med8Proof.root, "roundtrip root")

        // Deserialized proof should still verify
        let deserValid = engine.verifyBatchProof(deserialized)
        expect(deserValid, "deserialized proof verifies")

        // ================================================================
        // Test 10: Batch verification of multiple proofs
        // ================================================================
        suite("GPUMerkleBatchProof — Batch Verification")

        let proof0 = engine.generateBatchProof(tree: tree64, leafIndices: [0, 1])
        let proof1 = engine.generateBatchProof(tree: tree64, leafIndices: [32, 33, 62, 63])
        let proof2 = engine.generateBatchProof(tree: tree64, leafIndices: [15])

        let results = engine.verifyBatchProofs([proof0, proof1, proof2])
        expectEqual(results.count, 3, "3 verification results")
        expect(results[0], "batch verify proof 0")
        expect(results[1], "batch verify proof 1")
        expect(results[2], "batch verify proof 2")

        // ================================================================
        // Test 11: Large tree (1024 leaves, batch of 32)
        // ================================================================
        suite("GPUMerkleBatchProof — Large Tree")

        let leaves1024 = (0..<1024).map { frFromInt(UInt64($0 + 1)) }
        let tree1024 = try merkle.buildTree(leaves: leaves1024)

        // Pick 32 spread-out indices
        let indices32 = stride(from: 0, to: 1024, by: 32).map { $0 }
        let largeProof = engine.generateBatchProof(tree: tree1024, leafIndices: indices32)
        expectEqual(largeProof.leafIndices.count, 32, "32 leaves in large proof")
        expect(largeProof.auxiliaryCount < 32 * 10, "compressed: fewer aux than 32*depth(10)")

        let largeValid = engine.verifyBatchProof(largeProof)
        expect(largeValid, "large tree 32-leaf batch proof verifies")

        // ================================================================
        // Test 12: Keccak-256 backend batch proof
        // ================================================================
        suite("GPUMerkleBatchProof — Keccak Backend")

        if let kEngine = try? KeccakMerkleEngine() {
            let kLeaves = (0..<8).map { i -> [UInt8] in
                var h = keccak256([UInt8(i)])
                return Array(h[0..<32])
            }
            let kTree = try kEngine.buildTree(kLeaves)
            let kProof = engine.generateByteBatchProof(
                flatTree: kTree, leafCount: 8,
                leafIndices: [0, 3, 7], backend: .keccak256
            )
            expectEqual(kProof.leafIndices.count, 3, "keccak proof has 3 indices")
            let kValid = engine.verifyBatchProof(kProof)
            expect(kValid, "keccak batch proof verifies")

            // Serialization roundtrip for keccak
            let kSer = kProof.serialize()
            if let kDeser = MerkleBatchProof.deserialize(kSer) {
                expect(engine.verifyBatchProof(kDeser), "keccak deserialized proof verifies")
            } else {
                expect(false, "keccak deserialization failed")
            }
        } else {
            print("  [SKIP] KeccakMerkleEngine unavailable")
        }

        // ================================================================
        // Test 13: Blake3 backend batch proof
        // ================================================================
        suite("GPUMerkleBatchProof — Blake3 Backend")

        if let bEngine = try? Blake3MerkleEngine() {
            let bLeaves = (0..<8).map { i -> [UInt8] in
                var h = blake3([UInt8(i)])
                return Array(h[0..<32])
            }
            let bTree = try bEngine.buildTree(bLeaves)
            let bProof = engine.generateByteBatchProof(
                flatTree: bTree, leafCount: 8,
                leafIndices: [1, 4, 6], backend: .blake3
            )
            expectEqual(bProof.leafIndices.count, 3, "blake3 proof has 3 indices")
            let bValid = engine.verifyBatchProof(bProof)
            expect(bValid, "blake3 batch proof verifies")
        } else {
            print("  [SKIP] Blake3MerkleEngine unavailable")
        }

        // ================================================================
        // Test 14: Compression ratio — batch vs individual proofs
        // ================================================================
        suite("GPUMerkleBatchProof — Compression")

        // 16 adjacent pairs in a 64-leaf tree
        let pairIndices = Array(0..<16)
        let compProof = engine.generateBatchProof(tree: tree64, leafIndices: pairIndices)

        // Individual proofs would need 16 * depth(6) = 96 siblings total
        let individualTotal = 16 * tree64.depth
        expect(compProof.auxiliaryCount < individualTotal,
               "batch proof compressed: \(compProof.auxiliaryCount) < \(individualTotal) individual siblings")

        // ================================================================
        // Test 15: Empty deserialization returns nil
        // ================================================================
        suite("GPUMerkleBatchProof — Edge Cases")

        expect(MerkleBatchProof.deserialize([]) == nil, "empty data returns nil")
        expect(MerkleBatchProof.deserialize([0, 0]) == nil, "truncated data returns nil")

        print("  GPUMerkleBatchProof: all tests passed")

    } catch {
        expect(false, "GPUMerkleBatchProofEngine error: \(error)")
    }
}
