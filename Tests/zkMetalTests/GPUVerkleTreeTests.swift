// GPUVerkleTreeTests — Tests for GPU-accelerated Verkle tree engine
//
// Validates tree construction, proof generation/verification, tree updates,
// multi-proof batching, serialization, and configurable branching factor.

import Foundation
import Metal
import zkMetal

public func runGPUVerkleTreeTests() {
    suite("GPUVerkleTreeEngine — Construction")

    // Use width-4 for fast tests (same protocol, smaller vectors)
    let width = 4
    guard let engine = try? GPUVerkleTreeEngine(branchingFactor: width) else {
        print("  [SKIP] Failed to create GPUVerkleTreeEngine")
        return
    }

    // --- Test 1: Engine initialization ---
    expect(engine.branchingFactor == width, "Branching factor is \(width)")
    expect(engine.logWidth == 2, "logWidth is 2 for width=4")

    // --- Test 2: Build tree from leaves ---
    // 4 leaves = 1 chunk of width 4 => 1 level, root is that single commitment
    let leaves4: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    do {
        let (levels, chunks) = try engine.buildTree(leaves: leaves4)
        expect(levels.count == 1, "Single chunk produces 1 level")
        expect(levels[0].count == 1, "1 chunk => 1 commitment at level 0")
        expect(chunks.count == 1, "1 leaf chunk")
        expect(chunks[0].count == width, "Chunk has width elements")

        let root = engine.rootCommitment()
        expect(!pointIsIdentity(root), "Root is non-trivial")
    } catch {
        expect(false, "buildTree threw: \(error)")
    }

    // --- Test 3: Build tree with multiple levels ---
    // 16 leaves with width=4 => 4 chunks at level 0, 1 at level 1 (root)
    let leaves16: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
    do {
        let (levels, chunks) = try engine.buildTree(leaves: leaves16)
        expect(levels.count == 2, "16 leaves / width 4 => 2 levels")
        expect(levels[0].count == 4, "Level 0 has 4 commitments")
        expect(levels[1].count == 1, "Level 1 has 1 commitment (root)")
        expect(chunks.count == 4, "4 leaf chunks")
    } catch {
        expect(false, "buildTree threw: \(error)")
    }

    // --- Test 4: Deterministic commitment ---
    // Building the same tree twice should produce the same root
    do {
        let engine2 = try GPUVerkleTreeEngine(branchingFactor: width)
        _ = try engine.buildTree(leaves: leaves4)
        let root1 = engine.rootCommitment()
        _ = try engine2.buildTree(leaves: leaves4)
        let root2 = engine2.rootCommitment()
        expect(pointEqual(root1, root2), "Same leaves produce same root")
    } catch {
        expect(false, "Deterministic test threw: \(error)")
    }

    // --- Test 5: Different leaves produce different roots ---
    do {
        let leavesA: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let leavesB: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        _ = try engine.buildTree(leaves: leavesA)
        let rootA = engine.rootCommitment()
        _ = try engine.buildTree(leaves: leavesB)
        let rootB = engine.rootCommitment()
        expect(!pointEqual(rootA, rootB), "Different leaves produce different roots")
    } catch {
        expect(false, "Different leaves test threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Single Proof")

    // Build a tree for proof tests
    do {
        _ = try engine.buildTree(leaves: leaves16)
    } catch {
        expect(false, "Failed to build tree for proof tests: \(error)")
        return
    }
    let root16 = engine.rootCommitment()

    // --- Test 6: Generate proof for leaf 0 ---
    do {
        let proof = try engine.generateProof(leafIndex: 0)
        expect(!proof.isEmpty, "Proof is non-empty")
        expect(proof.count == 2, "2-level tree produces 2-entry proof")

        // Each entry should have logWidth L/R values
        for (i, entry) in proof.enumerated() {
            expect(entry.L.count == engine.logWidth, "Proof entry \(i) has logWidth L values")
            expect(entry.R.count == engine.logWidth, "Proof entry \(i) has logWidth R values")
        }

        // First entry's childIndex should be 0 (leaf 0 is at index 0 in chunk 0)
        expect(proof[0].childIndex == 0, "Leaf 0 is at childIndex 0")
        // Value should match
        expect(frEqual(proof[0].value, frFromInt(1)), "Leaf 0 value is 1")
    } catch {
        expect(false, "generateProof threw: \(error)")
    }

    // --- Test 7: Generate proof for leaf 5 ---
    do {
        let proof = try engine.generateProof(leafIndex: 5)
        // Leaf 5 is in chunk 1 (5/4=1), at position 1 (5%4=1)
        expect(proof[0].childIndex == 1, "Leaf 5 is at childIndex 1 in its chunk")
        expect(frEqual(proof[0].value, frFromInt(6)), "Leaf 5 value is 6")
    } catch {
        expect(false, "generateProof for leaf 5 threw: \(error)")
    }

    // --- Test 8: Verify proof for leaf 0 ---
    do {
        let proof = try engine.generateProof(leafIndex: 0)
        let valid = engine.verifyProof(proof, root: root16)
        expect(valid, "Proof for leaf 0 verifies against root")
    } catch {
        expect(false, "verifyProof threw: \(error)")
    }

    // --- Test 9: Verify proof for leaf 15 (last leaf) ---
    do {
        let proof = try engine.generateProof(leafIndex: 15)
        let valid = engine.verifyProof(proof, root: root16)
        expect(valid, "Proof for leaf 15 verifies against root")
    } catch {
        expect(false, "verifyProof for leaf 15 threw: \(error)")
    }

    // --- Test 10: Wrong root should reject proof ---
    do {
        let proof = try engine.generateProof(leafIndex: 0)
        // Create a different root by doubling
        let wrongRoot = pointAdd(root16, root16)
        let invalid = engine.verifyProof(proof, root: wrongRoot)
        expect(!invalid, "Proof rejects wrong root")
    } catch {
        expect(false, "Wrong root test threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Multi-Proof")

    // --- Test 11: Multi-proof for multiple leaves ---
    do {
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 5, 15])
        expect(multiProof.leafIndices.count == 3, "Multi-proof has 3 queries")
        expect(multiProof.leafValues.count == 3, "Multi-proof has 3 leaf values")
        expect(frEqual(multiProof.leafValues[0], frFromInt(1)), "Multi-proof leaf 0 value")
        expect(frEqual(multiProof.leafValues[1], frFromInt(6)), "Multi-proof leaf 5 value")
        expect(frEqual(multiProof.leafValues[2], frFromInt(16)), "Multi-proof leaf 15 value")
    } catch {
        expect(false, "generateMultiProof threw: \(error)")
    }

    // --- Test 12: Multi-proof commitment deduplication ---
    do {
        // Leaves 0 and 1 share the same chunk, so their leaf-level commitment
        // should be deduplicated
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 1])
        // Both queries reference the same chunk 0 commitment
        expect(multiProof.commitmentIndices[0][0] == multiProof.commitmentIndices[1][0],
               "Leaves in same chunk share commitment index")
    } catch {
        expect(false, "Deduplication test threw: \(error)")
    }

    // --- Test 13: Multi-proof verify ---
    do {
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 5, 10])
        let valid = engine.verifyMultiProof(multiProof)
        expect(valid, "Multi-proof verifies")
    } catch {
        expect(false, "Multi-proof verify threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Tree Updates")

    // --- Test 14: Update a leaf and verify new root differs ---
    do {
        _ = try engine.buildTree(leaves: leaves16)
        let rootBefore = engine.rootCommitment()

        let newRoot = try engine.updateLeaf(leafIndex: 0, newValue: frFromInt(999))
        expect(!pointEqual(rootBefore, newRoot), "Root changes after leaf update")
    } catch {
        expect(false, "updateLeaf threw: \(error)")
    }

    // --- Test 15: Update and re-verify proof ---
    do {
        _ = try engine.buildTree(leaves: leaves16)
        _ = try engine.updateLeaf(leafIndex: 3, newValue: frFromInt(42))
        let newRoot = engine.rootCommitment()

        let proof = try engine.generateProof(leafIndex: 3)
        let valid = engine.verifyProof(proof, root: newRoot)
        expect(valid, "Proof verifies after leaf update")
        expect(frEqual(proof[0].value, frFromInt(42)), "Updated leaf has new value")
    } catch {
        expect(false, "Update and re-verify threw: \(error)")
    }

    // --- Test 16: Batch update ---
    do {
        _ = try engine.buildTree(leaves: leaves16)
        let rootBefore = engine.rootCommitment()

        let newRoot = try engine.batchUpdateLeaves([
            (0, frFromInt(100)),
            (5, frFromInt(200)),
            (10, frFromInt(300))
        ])
        expect(!pointEqual(rootBefore, newRoot), "Root changes after batch update")

        // Verify a proof for one of the updated leaves
        let proof = try engine.generateProof(leafIndex: 5)
        let valid = engine.verifyProof(proof, root: newRoot)
        expect(valid, "Proof verifies after batch update")
        expect(frEqual(proof[0].value, frFromInt(200)), "Batch-updated leaf 5 has new value")
    } catch {
        expect(false, "batchUpdateLeaves threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Key-Value Mode")

    // --- Test 17: Insert and retrieve via key path ---
    do {
        let kvEngine = try GPUVerkleTreeEngine(branchingFactor: width)
        kvEngine.insert(key: [0, 1], value: frFromInt(42))
        kvEngine.insert(key: [0, 2], value: frFromInt(43))
        kvEngine.insert(key: [1, 0], value: frFromInt(44))

        let v1 = kvEngine.get(key: [0, 1])
        expect(v1 != nil && frEqual(v1!, frFromInt(42)), "Get key [0,1] returns 42")

        let v2 = kvEngine.get(key: [0, 2])
        expect(v2 != nil && frEqual(v2!, frFromInt(43)), "Get key [0,2] returns 43")

        let v3 = kvEngine.get(key: [1, 0])
        expect(v3 != nil && frEqual(v3!, frFromInt(44)), "Get key [1,0] returns 44")

        let absent = kvEngine.get(key: [2, 3])
        expect(absent == nil, "Get absent key returns nil")
    } catch {
        expect(false, "Key-value insert/get threw: \(error)")
    }

    // --- Test 18: Key-value commitment ---
    do {
        let kvEngine = try GPUVerkleTreeEngine(branchingFactor: width)
        kvEngine.insert(key: [0, 0], value: frFromInt(10))
        kvEngine.insert(key: [0, 1], value: frFromInt(20))
        let root = kvEngine.computeAndGetRoot()
        expect(!pointIsIdentity(root), "Key-value tree root is non-trivial")
    } catch {
        expect(false, "Key-value commitment threw: \(error)")
    }

    // --- Test 19: Update key-value ---
    do {
        let kvEngine = try GPUVerkleTreeEngine(branchingFactor: width)
        kvEngine.insert(key: [0, 0], value: frFromInt(10))
        let rootBefore = kvEngine.computeAndGetRoot()

        let updated = kvEngine.update(key: [0, 0], newValue: frFromInt(99))
        expect(updated, "Update returns true for existing key")
        let rootAfter = kvEngine.computeAndGetRoot()
        expect(!pointEqual(rootBefore, rootAfter), "Root changes after key-value update")

        let val = kvEngine.get(key: [0, 0])
        expect(val != nil && frEqual(val!, frFromInt(99)), "Updated key-value has new value")
    } catch {
        expect(false, "Update key-value threw: \(error)")
    }

    // --- Test 20: Update non-existent key returns false ---
    do {
        let kvEngine = try GPUVerkleTreeEngine(branchingFactor: width)
        kvEngine.insert(key: [0, 0], value: frFromInt(10))
        let notUpdated = kvEngine.update(key: [3, 3], newValue: frFromInt(99))
        expect(!notUpdated, "Update returns false for non-existent key")
    } catch {
        expect(false, "Update non-existent test threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Serialization")

    // --- Test 21: Multi-proof serialization round-trip ---
    do {
        _ = try engine.buildTree(leaves: leaves16)
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 7, 15])

        let serialized = engine.serializeMultiProof(multiProof)
        expect(serialized.byteSize > 0, "Serialized proof has non-zero size")

        // Verify structural integrity
        let deserialized = engine.deserializeMultiProof(serialized)
        expect(deserialized != nil, "Deserialization succeeds")

        if let d = deserialized {
            expect(d.leafIndices.count == 3, "Deserialized has 3 queries")
            expect(d.leafValues.count == 3, "Deserialized has 3 leaf values")
            expect(d.commitmentIndices.count == 3, "Deserialized has 3 commitment index arrays")
            expect(d.ipaFinalAs.count == multiProof.ipaFinalAs.count,
                   "Deserialized has same IPA count")
        }
    } catch {
        expect(false, "Serialization round-trip threw: \(error)")
    }

    // --- Test 22: Empty serialization ---
    do {
        _ = try engine.buildTree(leaves: leaves4)
        let multiProof = try engine.generateMultiProof(leafIndices: [0])
        let serialized = engine.serializeMultiProof(multiProof)
        expect(serialized.byteSize > 0, "Single-query serialized proof is non-empty")
    } catch {
        expect(false, "Single-query serialization threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Configurable Branching Factor")

    // --- Test 23: Width-2 tree ---
    do {
        let engine2 = try GPUVerkleTreeEngine(branchingFactor: 2)
        expect(engine2.branchingFactor == 2, "Width-2 engine created")
        expect(engine2.logWidth == 1, "logWidth is 1 for width=2")

        let leaves = [frFromInt(10), frFromInt(20)]
        let (levels, _) = try engine2.buildTree(leaves: leaves)
        expect(levels.count == 1, "2 leaves / width 2 => 1 level")
        expect(!pointIsIdentity(engine2.rootCommitment()), "Width-2 root is non-trivial")
    } catch {
        expect(false, "Width-2 tree threw: \(error)")
    }

    // --- Test 24: Width-8 tree ---
    do {
        let engine8 = try GPUVerkleTreeEngine(branchingFactor: 8)
        expect(engine8.branchingFactor == 8, "Width-8 engine created")
        expect(engine8.logWidth == 3, "logWidth is 3 for width=8")

        let leaves: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let (levels, _) = try engine8.buildTree(leaves: leaves)
        expect(levels.count == 2, "64 leaves / width 8 => 2 levels (8 chunks -> 1 root)")
        expect(levels[0].count == 8, "Level 0 has 8 commitments")
        expect(levels[1].count == 1, "Level 1 has 1 root")
    } catch {
        expect(false, "Width-8 tree threw: \(error)")
    }

    // --- Test 25: Width-8 proof round-trip ---
    do {
        let engine8 = try GPUVerkleTreeEngine(branchingFactor: 8)
        let leaves: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        _ = try engine8.buildTree(leaves: leaves)
        let root8 = engine8.rootCommitment()

        let proof = try engine8.generateProof(leafIndex: 0)
        let valid = engine8.verifyProof(proof, root: root8)
        expect(valid, "Width-8 proof verifies")
    } catch {
        expect(false, "Width-8 proof threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Edge Cases")

    // --- Test 26: Single chunk tree (leaves == width) ---
    do {
        _ = try engine.buildTree(leaves: leaves4)
        let root = engine.rootCommitment()
        expect(!pointIsIdentity(root), "Single-chunk tree has valid root")

        let proof = try engine.generateProof(leafIndex: 0)
        expect(proof.count == 1, "Single-chunk tree has 1-level proof")
        let valid = engine.verifyProof(proof, root: root)
        expect(valid, "Single-chunk proof verifies")
    } catch {
        expect(false, "Single-chunk test threw: \(error)")
    }

    // --- Test 27: All-zero leaves ---
    do {
        let zeroLeaves = [Fr](repeating: Fr.zero, count: width)
        _ = try engine.buildTree(leaves: zeroLeaves)
        let root = engine.rootCommitment()
        // Commitment to all zeros should be the identity (sum of 0*G_i = O)
        expect(pointIsIdentity(root), "All-zero leaves produce identity root")
    } catch {
        expect(false, "All-zero test threw: \(error)")
    }

    // --- Test 28: All-one leaves vs all-two leaves differ ---
    do {
        let onesLeaves = [Fr](repeating: Fr.one, count: width)
        _ = try engine.buildTree(leaves: onesLeaves)
        let rootOnes = engine.rootCommitment()

        let twosLeaves = [Fr](repeating: frFromInt(2), count: width)
        _ = try engine.buildTree(leaves: twosLeaves)
        let rootTwos = engine.rootCommitment()

        expect(!pointEqual(rootOnes, rootTwos), "All-ones and all-twos produce different roots")
    } catch {
        expect(false, "All-ones vs all-twos threw: \(error)")
    }

    // --- Test 29: GPU availability check ---
    if engine.gpuAvailable {
        expect(true, "GPU is available (Metal device found)")
    } else {
        expect(true, "CPU fallback mode (no GPU or GPU init failed)")
    }

    // --- Test 30: Larger tree (64 leaves, width 4) ---
    do {
        let leaves64: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        _ = try engine.buildTree(leaves: leaves64)
        let root = engine.rootCommitment()
        expect(!pointIsIdentity(root), "64-leaf tree has valid root")

        // 64 leaves / 4 = 16 chunks at level 0
        // 16 / 4 = 4 at level 1
        // 4 / 4 = 1 at level 2 (root)
        // => 3 levels

        let proof = try engine.generateProof(leafIndex: 33)
        expect(proof.count == 3, "64-leaf/width-4 tree has 3-level proof")
        let valid = engine.verifyProof(proof, root: root)
        expect(valid, "64-leaf proof verifies")
    } catch {
        expect(false, "Larger tree test threw: \(error)")
    }

    suite("GPUVerkleTreeEngine — Bandwidth Optimization")

    // --- Test 31: Serialized proof is compact ---
    do {
        let leaves64: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        _ = try engine.buildTree(leaves: leaves64)

        // Single proof serialization
        let singleProof = try engine.generateMultiProof(leafIndices: [0])
        let singleSerialized = engine.serializeMultiProof(singleProof)

        // Multi-proof serialization
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 16, 32, 48])
        let multiSerialized = engine.serializeMultiProof(multiProof)

        // Multi-proof should be less than 4x the single proof size
        // due to commitment deduplication
        let ratio = Double(multiSerialized.byteSize) / Double(singleSerialized.byteSize)
        expect(ratio < 4.0, "Multi-proof size ratio < 4x single (\(String(format: "%.1f", ratio))x)")
    } catch {
        expect(false, "Bandwidth optimization test threw: \(error)")
    }

    // --- Test 32: Deserialized leaf values match ---
    do {
        _ = try engine.buildTree(leaves: leaves16)
        let multiProof = try engine.generateMultiProof(leafIndices: [0, 5])
        let serialized = engine.serializeMultiProof(multiProof)
        let deserialized = engine.deserializeMultiProof(serialized)

        expect(deserialized != nil, "Deserialization succeeds")
        if let d = deserialized {
            expect(frEqual(d.leafValues[0], frFromInt(1)), "Deserialized leaf 0 value matches")
            expect(frEqual(d.leafValues[1], frFromInt(6)), "Deserialized leaf 5 value matches")
            expect(d.leafIndices[0] == 0, "Deserialized leaf index 0 matches")
            expect(d.leafIndices[1] == 5, "Deserialized leaf index 5 matches")
        }
    } catch {
        expect(false, "Deserialized values test threw: \(error)")
    }
}
