// GPU Merkle Aggregate Engine Tests
import zkMetal
import Foundation
import Metal

public func runGPUMerkleAggregateTests() {
    suite("GPU Merkle Aggregate Engine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUMerkleAggregateEngine() else {
        print("  [SKIP] Failed to create GPUMerkleAggregateEngine")
        return
    }

    do {
        let treeEngine = engine.treeEngine

        testSingleProofAggregate(engine: engine, treeEngine: treeEngine)
        try testTwoAdjacentProofsAggregate(engine: engine, treeEngine: treeEngine)
        try testTwoNonAdjacentProofsAggregate(engine: engine, treeEngine: treeEngine)
        try testAllLeavesAggregate(engine: engine, treeEngine: treeEngine)
        try testMediumTreeAggregate(engine: engine, treeEngine: treeEngine)
        try testDuplicateLeafIndices(engine: engine, treeEngine: treeEngine)
        try testTamperedLeafFails(engine: engine, treeEngine: treeEngine)
        try testWrongRootFails(engine: engine, treeEngine: treeEngine)
        try testSerializationRoundtrip(engine: engine, treeEngine: treeEngine)
        try testBatchAggregateVerification(engine: engine, treeEngine: treeEngine)
        try testLargeTreeAggregate(engine: engine, treeEngine: treeEngine)
        try testCompressionRatio(engine: engine, treeEngine: treeEngine)
        try testMultiTreeAggregation(engine: engine, treeEngine: treeEngine)
        try testIncrementalAggregation(engine: engine, treeEngine: treeEngine)
        try testRootConsistency(engine: engine, treeEngine: treeEngine)
        try testAggregateRootConsistency(engine: engine, treeEngine: treeEngine)
        try testSingleProofVerification(engine: engine, treeEngine: treeEngine)
        try testProofBatchVerification(engine: engine, treeEngine: treeEngine)
        try testMergeAggregates(engine: engine, treeEngine: treeEngine)
        try testAggregateStats(engine: engine, treeEngine: treeEngine)
        try testEmptyAggregate(engine: engine)
        try testReconstructProofs(engine: engine, treeEngine: treeEngine)
        try testMultiTreeDifferentDepths(engine: engine, treeEngine: treeEngine)
        try testDeserializationEdgeCases()
        try testThreeTreeAggregation(engine: engine, treeEngine: treeEngine)
        try testIncrementalMultiTree(engine: engine, treeEngine: treeEngine)
        try testSiblingCompression16Leaves(engine: engine, treeEngine: treeEngine)
        try testMergeMultiTreeAggregates(engine: engine, treeEngine: treeEngine)
        try testConsecutiveLeavesMaxCompression(engine: engine, treeEngine: treeEngine)
        try testSpreadLeavesMinimalCompression(engine: engine, treeEngine: treeEngine)

        print("  GPUMerkleAggregateEngine: all tests passed")
    } catch {
        expect(false, "GPUMerkleAggregateEngine error: \(error)")
    }
}

// MARK: - Test: Single proof aggregate

private func testSingleProofAggregate(engine: GPUMerkleAggregateEngine,
                                       treeEngine: GPUMerkleTreeEngine) {
    suite("MerkleAggregate -- Single Proof")

    do {
        let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree = try treeEngine.buildTree(leaves: leaves)

        let proofs = engine.buildProofs(tree: tree, leafIndices: [2])
        let aggregate = engine.aggregate(proofs: proofs)

        expectEqual(aggregate.proofCount, 1, "single proof aggregate has 1 proof")
        expectEqual(aggregate.treeCount, 1, "single proof aggregate has 1 tree")
        expectEqual(aggregate.leafEntries[0].leafIndex, 2, "leaf index is 2")

        let valid = engine.verifyAggregate(aggregate)
        expect(valid, "single proof aggregate verifies")
    } catch {
        expect(false, "single proof aggregate error: \(error)")
    }
}

// MARK: - Test: Two adjacent proofs

private func testTwoAdjacentProofsAggregate(engine: GPUMerkleAggregateEngine,
                                              treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Adjacent Proofs")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 1])
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.proofCount, 2, "adjacent aggregate has 2 proofs")
    // Leaves 0 and 1 are siblings — their sibling at level 0 is each other,
    // so only 1 auxiliary sibling needed (the hash of leaves 2,3 at level 1)
    let sibCount = aggregate.siblings.count
    expect(sibCount == 1, "adjacent leaves: \(sibCount) siblings (expected 1)")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "adjacent proof aggregate verifies")
}

// MARK: - Test: Two non-adjacent proofs

private func testTwoNonAdjacentProofsAggregate(engine: GPUMerkleAggregateEngine,
                                                 treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Non-Adjacent Proofs")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 3])
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.proofCount, 2, "non-adjacent aggregate has 2 proofs")
    // Leaf 0 needs sibling leaf 1, leaf 3 needs sibling leaf 2
    // At level 1, both parents are known => no more siblings
    let sibCount = aggregate.siblings.count
    expectEqual(sibCount, 2, "non-adjacent leaves: 2 siblings")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "non-adjacent proof aggregate verifies")
}

// MARK: - Test: All leaves aggregate

private func testAllLeavesAggregate(engine: GPUMerkleAggregateEngine,
                                     treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- All Leaves")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 1, 2, 3])
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.proofCount, 4, "all-leaf aggregate has 4 proofs")
    expectEqual(aggregate.siblings.count, 0, "all leaves known: 0 siblings")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "all-leaf aggregate verifies")
}

// MARK: - Test: Medium tree aggregate

private func testMediumTreeAggregate(engine: GPUMerkleAggregateEngine,
                                      treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Medium Tree 64 Leaves")

    let leaves = (0..<64).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let indices = [0, 7, 15, 16, 31, 32, 48, 63]
    let proofs = engine.buildProofs(tree: tree, leafIndices: indices)
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.proofCount, 8, "medium aggregate has 8 proofs")
    // 8 proofs in depth-6 tree: naive = 48 siblings, compressed < 48
    let naiveTotal = 8 * 6
    expect(aggregate.siblings.count < naiveTotal,
           "compressed: \(aggregate.siblings.count) < \(naiveTotal) naive siblings")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "medium tree 8-leaf aggregate verifies")
}

// MARK: - Test: Duplicate leaf indices

private func testDuplicateLeafIndices(engine: GPUMerkleAggregateEngine,
                                       treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Duplicate Indices")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // Provide same leaf index multiple times
    let proofs = engine.buildProofs(tree: tree, leafIndices: [1, 1, 1])
    let aggregate = engine.aggregate(proofs: proofs)

    // Should deduplicate to 1 unique leaf
    expectEqual(aggregate.proofCount, 1, "duplicates deduplicated to 1")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "deduplicated aggregate verifies")
}

// MARK: - Test: Tampered leaf fails verification

private func testTamperedLeafFails(engine: GPUMerkleAggregateEngine,
                                    treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Tampered Leaf")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [2])
    let goodAggregate = engine.aggregate(proofs: proofs)

    // Tamper with the leaf value
    var tamperedEntries = goodAggregate.leafEntries
    let badLeaf = AggregateLeafEntry(
        leafIndex: tamperedEntries[0].leafIndex,
        leaf: frFromInt(999),
        treeInfoIndex: tamperedEntries[0].treeInfoIndex
    )
    tamperedEntries[0] = badLeaf

    let tampered = AggregateMerkleProof(
        trees: goodAggregate.trees,
        leafEntries: tamperedEntries,
        siblings: goodAggregate.siblings,
        siblingTreeIndices: goodAggregate.siblingTreeIndices
    )

    let valid = engine.verifyAggregate(tampered)
    expect(!valid, "tampered leaf fails verification")
}

// MARK: - Test: Wrong root fails verification

private func testWrongRootFails(engine: GPUMerkleAggregateEngine,
                                 treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Wrong Root")

    let leaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0])
    let goodAggregate = engine.aggregate(proofs: proofs)

    // Tamper with the root
    var tamperedTrees = goodAggregate.trees
    let wrongRoot = frFromInt(888)
    tamperedTrees[0] = TreeAggregateInfo(
        treeId: tamperedTrees[0].treeId,
        root: wrongRoot,
        leafCount: tamperedTrees[0].leafCount,
        proofIndices: tamperedTrees[0].proofIndices
    )

    let tampered = AggregateMerkleProof(
        trees: tamperedTrees,
        leafEntries: goodAggregate.leafEntries,
        siblings: goodAggregate.siblings,
        siblingTreeIndices: goodAggregate.siblingTreeIndices
    )

    let valid = engine.verifyAggregate(tampered)
    expect(!valid, "wrong root fails verification")
}

// MARK: - Test: Serialization roundtrip

private func testSerializationRoundtrip(engine: GPUMerkleAggregateEngine,
                                          treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Serialization")

    let leaves = (0..<64).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 7, 15, 16, 31, 32, 48, 63])
    let aggregate = engine.aggregate(proofs: proofs)

    let serialized = aggregate.serialize()
    expect(serialized.count > 0, "serialized bytes non-empty")

    guard let deserialized = AggregateMerkleProof.deserialize(serialized) else {
        expect(false, "deserialization returned nil")
        return
    }

    expectEqual(deserialized.proofCount, aggregate.proofCount, "roundtrip proofCount")
    expectEqual(deserialized.treeCount, aggregate.treeCount, "roundtrip treeCount")
    expectEqual(deserialized.siblings.count, aggregate.siblings.count, "roundtrip siblingCount")
    expectEqual(deserialized.leafEntries.count, aggregate.leafEntries.count, "roundtrip entryCount")

    // Verify leaf indices match
    for i in 0..<aggregate.leafEntries.count {
        expectEqual(deserialized.leafEntries[i].leafIndex,
                    aggregate.leafEntries[i].leafIndex,
                    "roundtrip leaf index \(i)")
    }

    // Deserialized proof should still verify
    let valid = engine.verifyAggregate(deserialized)
    expect(valid, "deserialized aggregate verifies")
}

// MARK: - Test: Batch aggregate verification

private func testBatchAggregateVerification(engine: GPUMerkleAggregateEngine,
                                              treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Batch Verification")

    let leaves = (0..<64).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let agg0 = engine.aggregate(proofs: engine.buildProofs(tree: tree, leafIndices: [0, 1]))
    let agg1 = engine.aggregate(proofs: engine.buildProofs(tree: tree, leafIndices: [32, 33, 62, 63]))
    let agg2 = engine.aggregate(proofs: engine.buildProofs(tree: tree, leafIndices: [15]))

    let results = engine.verifyAggregates([agg0, agg1, agg2])
    expectEqual(results.count, 3, "3 verification results")
    expect(results[0], "batch verify aggregate 0")
    expect(results[1], "batch verify aggregate 1")
    expect(results[2], "batch verify aggregate 2")
}

// MARK: - Test: Large tree aggregate

private func testLargeTreeAggregate(engine: GPUMerkleAggregateEngine,
                                     treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Large Tree 1024")

    let leaves = (0..<1024).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let indices = stride(from: 0, to: 1024, by: 32).map { $0 }
    let proofs = engine.buildProofs(tree: tree, leafIndices: indices)
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.proofCount, 32, "32 proofs in large aggregate")

    // Naive total: 32 * depth(10) = 320
    let naiveTotal = 32 * 10
    expect(aggregate.siblings.count < naiveTotal,
           "compressed: \(aggregate.siblings.count) < \(naiveTotal)")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "large tree 32-proof aggregate verifies")
}

// MARK: - Test: Compression ratio

private func testCompressionRatio(engine: GPUMerkleAggregateEngine,
                                   treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Compression Ratio")

    let leaves = (0..<64).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // 16 consecutive leaves in the first quarter
    let proofs = engine.buildProofs(tree: tree, leafIndices: Array(0..<16))
    let aggregate = engine.aggregate(proofs: proofs)

    let ratio = aggregate.compressionRatio
    expect(ratio < 1.0, "compression ratio < 1.0: \(ratio)")
    expect(ratio > 0.0, "compression ratio > 0.0: \(ratio)")

    let stats = engine.aggregateStats(aggregate)
    expect(stats.siblingsSaved > 0, "siblings saved: \(stats.siblingsSaved)")
    expectEqual(stats.proofCount, 16, "stats proofCount == 16")
    expectEqual(stats.treeCount, 1, "stats treeCount == 1")
}

// MARK: - Test: Multi-tree aggregation

private func testMultiTreeAggregation(engine: GPUMerkleAggregateEngine,
                                       treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Multi-Tree")

    let leavesA = (0..<8).map { frFromInt(UInt64($0 + 100)) }
    let treeA = try treeEngine.buildTree(leaves: leavesA)

    let leavesB = (0..<8).map { frFromInt(UInt64($0 + 200)) }
    let treeB = try treeEngine.buildTree(leaves: leavesB)

    let proofsA = engine.buildProofs(tree: treeA, leafIndices: [0, 3], treeId: 1)
    let proofsB = engine.buildProofs(tree: treeB, leafIndices: [1, 5], treeId: 2)

    let allProofs = proofsA + proofsB
    let aggregate = engine.aggregate(proofs: allProofs)

    expectEqual(aggregate.treeCount, 2, "multi-tree has 2 trees")
    expectEqual(aggregate.proofCount, 4, "multi-tree has 4 proofs")

    // Check trees have correct roots
    let treeInfoA = aggregate.trees.first { $0.treeId == 1 }
    let treeInfoB = aggregate.trees.first { $0.treeId == 2 }
    expect(treeInfoA != nil, "tree A found in aggregate")
    expect(treeInfoB != nil, "tree B found in aggregate")

    if let tiA = treeInfoA {
        expect(frEqual(tiA.root, treeA.root), "tree A root matches")
        expectEqual(tiA.proofIndices.count, 2, "tree A has 2 proofs")
    }
    if let tiB = treeInfoB {
        expect(frEqual(tiB.root, treeB.root), "tree B root matches")
        expectEqual(tiB.proofIndices.count, 2, "tree B has 2 proofs")
    }

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "multi-tree aggregate verifies")
}

// MARK: - Test: Incremental aggregation

private func testIncrementalAggregation(engine: GPUMerkleAggregateEngine,
                                         treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Incremental")

    let leaves = (0..<16).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // Start with 2 proofs
    let initialProofs = engine.buildProofs(tree: tree, leafIndices: [0, 3])
    let initialAgg = engine.aggregate(proofs: initialProofs)
    expect(engine.verifyAggregate(initialAgg), "initial aggregate verifies")

    // Add 2 more proofs incrementally
    let newProofs = engine.buildProofs(tree: tree, leafIndices: [7, 15])
    let expandedAgg = engine.incrementalAggregate(existing: initialAgg, newProofs: newProofs)

    expectEqual(expandedAgg.proofCount, 4, "incremental aggregate has 4 proofs")

    let valid = engine.verifyAggregate(expandedAgg)
    expect(valid, "incremental aggregate verifies")

    // Verify all leaf indices are present
    let leafIndices = Set(expandedAgg.leafEntries.map { $0.leafIndex })
    expect(leafIndices.contains(0), "incremental contains leaf 0")
    expect(leafIndices.contains(3), "incremental contains leaf 3")
    expect(leafIndices.contains(7), "incremental contains leaf 7")
    expect(leafIndices.contains(15), "incremental contains leaf 15")
}

// MARK: - Test: Root consistency check

private func testRootConsistency(engine: GPUMerkleAggregateEngine,
                                  treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Root Consistency")

    let leavesA = (0..<4).map { frFromInt(UInt64($0 + 1)) }
    let treeA = try treeEngine.buildTree(leaves: leavesA)

    let leavesB = (0..<4).map { frFromInt(UInt64($0 + 100)) }
    let treeB = try treeEngine.buildTree(leaves: leavesB)

    // Consistent: same tree
    let proofsConsistent = engine.buildProofs(tree: treeA, leafIndices: [0, 1, 2], treeId: 1)
    expect(engine.checkRootConsistency(proofsConsistent), "same-tree proofs are consistent")

    // Inconsistent: different trees, same treeId
    let proofA = engine.buildProofs(tree: treeA, leafIndices: [0], treeId: 1)
    let proofB = engine.buildProofs(tree: treeB, leafIndices: [0], treeId: 1)
    let mixedProofs = proofA + proofB
    expect(!engine.checkRootConsistency(mixedProofs), "different-tree same-id proofs are inconsistent")

    // Different trees, different treeIds — consistent
    let proofA2 = engine.buildProofs(tree: treeA, leafIndices: [0], treeId: 1)
    let proofB2 = engine.buildProofs(tree: treeB, leafIndices: [0], treeId: 2)
    expect(engine.checkRootConsistency(proofA2 + proofB2), "different-tree different-id proofs are consistent")
}

// MARK: - Test: Aggregate root consistency

private func testAggregateRootConsistency(engine: GPUMerkleAggregateEngine,
                                            treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Aggregate Root Consistency")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 3, 7])
    let aggregate = engine.aggregate(proofs: proofs)

    expect(engine.checkAggregateRootConsistency(aggregate),
           "valid aggregate has consistent roots")
}

// MARK: - Test: Single proof verification

private func testSingleProofVerification(engine: GPUMerkleAggregateEngine,
                                           treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Single Proof Verify")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    for i in 0..<8 {
        let proof = engine.buildProofs(tree: tree, leafIndices: [i])[0]
        let valid = engine.verifySingleProof(proof)
        expect(valid, "single proof verifies for leaf \(i)")
    }

    // Wrong leaf should fail
    let wrongProof = MerkleInclusionProof(
        root: tree.root,
        leaf: frFromInt(999),
        leafIndex: 0,
        leafCount: 8,
        path: tree.proof(forLeafAt: 0).siblings
    )
    expect(!engine.verifySingleProof(wrongProof), "wrong leaf single proof fails")
}

// MARK: - Test: Proof batch verification

private func testProofBatchVerification(engine: GPUMerkleAggregateEngine,
                                          treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Proof Batch Verify")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 3, 7])
    let results = engine.verifyProofsBatch(proofs)

    expectEqual(results.count, 3, "3 batch results")
    expect(results[0], "batch verify proof 0")
    expect(results[1], "batch verify proof 1")
    expect(results[2], "batch verify proof 2")
}

// MARK: - Test: Merge aggregates

private func testMergeAggregates(engine: GPUMerkleAggregateEngine,
                                   treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Merge Aggregates")

    let leaves = (0..<16).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let aggA = engine.aggregate(proofs: engine.buildProofs(tree: tree, leafIndices: [0, 1, 2]))
    let aggB = engine.aggregate(proofs: engine.buildProofs(tree: tree, leafIndices: [8, 9, 10]))

    let merged = engine.mergeAggregates(aggA, aggB)

    expectEqual(merged.proofCount, 6, "merged has 6 proofs")

    let valid = engine.verifyAggregate(merged)
    expect(valid, "merged aggregate verifies")

    // Check all leaf indices present
    let mergedIndices = Set(merged.leafEntries.map { $0.leafIndex })
    for idx in [0, 1, 2, 8, 9, 10] {
        expect(mergedIndices.contains(idx), "merged contains leaf \(idx)")
    }
}

// MARK: - Test: Aggregate stats

private func testAggregateStats(engine: GPUMerkleAggregateEngine,
                                  treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Stats")

    let leaves = (0..<32).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let proofs = engine.buildProofs(tree: tree, leafIndices: Array(0..<8))
    let aggregate = engine.aggregate(proofs: proofs)

    let stats = engine.aggregateStats(aggregate)
    expectEqual(stats.proofCount, 8, "stats proofCount == 8")
    expectEqual(stats.treeCount, 1, "stats treeCount == 1")
    // 8 proofs * depth 5 = 40 naive siblings
    expectEqual(stats.totalNaiveSiblings, 40, "naive siblings == 40")
    expect(stats.actualSiblings < stats.totalNaiveSiblings,
           "actual < naive: \(stats.actualSiblings) < \(stats.totalNaiveSiblings)")
    expectEqual(stats.siblingsSaved, stats.totalNaiveSiblings - stats.actualSiblings,
                "saved = naive - actual")
    expect(stats.compressionRatio < 1.0, "ratio < 1.0: \(stats.compressionRatio)")
}

// MARK: - Test: Empty aggregate

private func testEmptyAggregate(engine: GPUMerkleAggregateEngine) throws {
    suite("MerkleAggregate -- Empty")

    let aggregate = engine.aggregate(proofs: [])
    expectEqual(aggregate.proofCount, 0, "empty aggregate has 0 proofs")
    expectEqual(aggregate.treeCount, 0, "empty aggregate has 0 trees")
    expectEqual(aggregate.siblings.count, 0, "empty aggregate has 0 siblings")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "empty aggregate verifies vacuously")
}

// MARK: - Test: Reconstruct proofs

private func testReconstructProofs(engine: GPUMerkleAggregateEngine,
                                    treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Reconstruct Proofs")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    let originalProofs = engine.buildProofs(tree: tree, leafIndices: [1, 4, 6])
    let aggregate = engine.aggregate(proofs: originalProofs)

    let reconstructed = engine.reconstructProofs(from: aggregate)
    expectEqual(reconstructed.count, 3, "reconstructed 3 proofs")

    // Each reconstructed proof should individually verify
    for (i, rp) in reconstructed.enumerated() {
        let valid = engine.verifySingleProof(rp)
        expect(valid, "reconstructed proof \(i) verifies (leaf \(rp.leafIndex))")
    }

    // Leaf indices should match
    let reconIndices = Set(reconstructed.map { $0.leafIndex })
    for idx in [1, 4, 6] {
        expect(reconIndices.contains(idx), "reconstructed contains leaf \(idx)")
    }
}

// MARK: - Test: Multi-tree with different depths

private func testMultiTreeDifferentDepths(engine: GPUMerkleAggregateEngine,
                                            treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Multi-Tree Different Depths")

    let leavesSmall = (0..<4).map { frFromInt(UInt64($0 + 10)) }   // depth 2
    let treeSmall = try treeEngine.buildTree(leaves: leavesSmall)

    let leavesBig = (0..<32).map { frFromInt(UInt64($0 + 200)) }   // depth 5
    let treeBig = try treeEngine.buildTree(leaves: leavesBig)

    let proofsSmall = engine.buildProofs(tree: treeSmall, leafIndices: [0, 3], treeId: 10)
    let proofsBig = engine.buildProofs(tree: treeBig, leafIndices: [0, 15, 31], treeId: 20)

    let aggregate = engine.aggregate(proofs: proofsSmall + proofsBig)

    expectEqual(aggregate.treeCount, 2, "2 trees with different depths")
    expectEqual(aggregate.proofCount, 5, "5 proofs total")

    // Check tree leaf counts
    let smallInfo = aggregate.trees.first { $0.treeId == 10 }
    let bigInfo = aggregate.trees.first { $0.treeId == 20 }
    expect(smallInfo != nil, "small tree found")
    expect(bigInfo != nil, "big tree found")

    if let si = smallInfo {
        expectEqual(si.leafCount, 4, "small tree leafCount == 4")
    }
    if let bi = bigInfo {
        expectEqual(bi.leafCount, 32, "big tree leafCount == 32")
    }

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "multi-depth aggregate verifies")
}

// MARK: - Test: Deserialization edge cases

private func testDeserializationEdgeCases() throws {
    suite("MerkleAggregate -- Deserialization Edge Cases")

    expect(AggregateMerkleProof.deserialize([]) == nil, "empty data returns nil")
    expect(AggregateMerkleProof.deserialize([0, 0]) == nil, "truncated data returns nil")
    expect(AggregateMerkleProof.deserialize([0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0]) == nil,
           "huge tree count returns nil (not enough data)")

    // Valid empty aggregate serialization should roundtrip
    let empty = AggregateMerkleProof(trees: [], leafEntries: [], siblings: [],
                                      siblingTreeIndices: [])
    let ser = empty.serialize()
    let deser = AggregateMerkleProof.deserialize(ser)
    expect(deser != nil, "empty aggregate deserializes")
    if let d = deser {
        expectEqual(d.proofCount, 0, "empty deser proofCount == 0")
        expectEqual(d.treeCount, 0, "empty deser treeCount == 0")
    }
}

// MARK: - Test: Three-tree aggregation

private func testThreeTreeAggregation(engine: GPUMerkleAggregateEngine,
                                       treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Three Trees")

    let trees: [(MerkleTree, UInt64)] = try [
        (treeEngine.buildTree(leaves: (0..<8).map { frFromInt(UInt64($0 + 1)) }), 100),
        (treeEngine.buildTree(leaves: (0..<16).map { frFromInt(UInt64($0 + 50)) }), 200),
        (treeEngine.buildTree(leaves: (0..<4).map { frFromInt(UInt64($0 + 300)) }), 300),
    ]

    var allProofs = [MerkleInclusionProof]()
    allProofs += engine.buildProofs(tree: trees[0].0, leafIndices: [0, 7], treeId: trees[0].1)
    allProofs += engine.buildProofs(tree: trees[1].0, leafIndices: [4, 10], treeId: trees[1].1)
    allProofs += engine.buildProofs(tree: trees[2].0, leafIndices: [1, 2], treeId: trees[2].1)

    let aggregate = engine.aggregate(proofs: allProofs)

    expectEqual(aggregate.treeCount, 3, "three-tree aggregate has 3 trees")
    expectEqual(aggregate.proofCount, 6, "three-tree aggregate has 6 proofs")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "three-tree aggregate verifies")

    // Serialization roundtrip for multi-tree
    let ser = aggregate.serialize()
    guard let deser = AggregateMerkleProof.deserialize(ser) else {
        expect(false, "three-tree deserialization failed")
        return
    }
    expect(engine.verifyAggregate(deser), "three-tree deserialized aggregate verifies")
}

// MARK: - Test: Incremental multi-tree

private func testIncrementalMultiTree(engine: GPUMerkleAggregateEngine,
                                       treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Incremental Multi-Tree")

    let treeA = try treeEngine.buildTree(leaves: (0..<8).map { frFromInt(UInt64($0 + 1)) })
    let treeB = try treeEngine.buildTree(leaves: (0..<8).map { frFromInt(UInt64($0 + 50)) })

    // Start with tree A proofs
    let initialProofs = engine.buildProofs(tree: treeA, leafIndices: [0, 3], treeId: 1)
    let initialAgg = engine.aggregate(proofs: initialProofs)

    // Add tree B proofs incrementally
    let newProofs = engine.buildProofs(tree: treeB, leafIndices: [2, 5], treeId: 2)
    let expanded = engine.incrementalAggregate(existing: initialAgg, newProofs: newProofs)

    expectEqual(expanded.treeCount, 2, "incremental multi-tree has 2 trees")
    expectEqual(expanded.proofCount, 4, "incremental multi-tree has 4 proofs")

    let valid = engine.verifyAggregate(expanded)
    expect(valid, "incremental multi-tree aggregate verifies")
}

// MARK: - Test: Sibling compression 16 leaves

private func testSiblingCompression16Leaves(engine: GPUMerkleAggregateEngine,
                                              treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Sibling Compression 16")

    let leaves = (0..<16).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // Pairs that share parents: (0,1), (2,3), (4,5), (6,7)
    let proofs = engine.buildProofs(tree: tree, leafIndices: [0, 1, 2, 3, 4, 5, 6, 7])
    let aggregate = engine.aggregate(proofs: proofs)

    // 8 leaves cover the left half of a 16-leaf tree.
    // At level 0: all siblings are other leaves in the set => 0 aux at level 0
    // At level 1: left half parents are all known => right half parent H(8,9), H(10,11), H(12,13), H(14,15)
    // At level 2: left half grandparent known => right half grandparents
    // etc.
    // Naive: 8 * 4 = 32 siblings
    let naiveTotal = 8 * tree.depth
    expect(aggregate.siblings.count < naiveTotal,
           "left-half compression: \(aggregate.siblings.count) < \(naiveTotal)")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "16-leaf left-half aggregate verifies")
}

// MARK: - Test: Merge multi-tree aggregates

private func testMergeMultiTreeAggregates(engine: GPUMerkleAggregateEngine,
                                            treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Merge Multi-Tree Aggregates")

    let treeA = try treeEngine.buildTree(leaves: (0..<8).map { frFromInt(UInt64($0 + 1)) })
    let treeB = try treeEngine.buildTree(leaves: (0..<8).map { frFromInt(UInt64($0 + 100)) })

    let aggA = engine.aggregate(proofs: engine.buildProofs(tree: treeA, leafIndices: [0, 1], treeId: 1))
    let aggB = engine.aggregate(proofs: engine.buildProofs(tree: treeB, leafIndices: [6, 7], treeId: 2))

    let merged = engine.mergeAggregates(aggA, aggB)

    expectEqual(merged.treeCount, 2, "merged multi-tree has 2 trees")
    expectEqual(merged.proofCount, 4, "merged multi-tree has 4 proofs")

    let valid = engine.verifyAggregate(merged)
    expect(valid, "merged multi-tree aggregate verifies")
}

// MARK: - Test: Consecutive leaves max compression

private func testConsecutiveLeavesMaxCompression(engine: GPUMerkleAggregateEngine,
                                                   treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Max Compression")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // All leaves -> 0 siblings needed
    let proofs = engine.buildProofs(tree: tree, leafIndices: Array(0..<8))
    let aggregate = engine.aggregate(proofs: proofs)

    expectEqual(aggregate.siblings.count, 0, "all leaves: 0 siblings")
    expectEqual(aggregate.compressionRatio, 0.0, "all leaves: ratio == 0")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "all-leaves max compression verifies")
}

// MARK: - Test: Spread leaves minimal compression

private func testSpreadLeavesMinimalCompression(engine: GPUMerkleAggregateEngine,
                                                  treeEngine: GPUMerkleTreeEngine) throws {
    suite("MerkleAggregate -- Minimal Compression")

    let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let tree = try treeEngine.buildTree(leaves: leaves)

    // Single leaf -> no compression possible
    let proofs = engine.buildProofs(tree: tree, leafIndices: [4])
    let aggregate = engine.aggregate(proofs: proofs)

    // depth == 3, 1 proof, should have 3 siblings (full path)
    let depth = tree.depth
    expectEqual(aggregate.siblings.count, depth,
                "single leaf: \(depth) siblings (full path)")

    let stats = engine.aggregateStats(aggregate)
    expectEqual(stats.siblingsSaved, 0, "single leaf: 0 siblings saved")
    expect(stats.compressionRatio == 1.0,
           "single leaf: ratio == 1.0, got \(stats.compressionRatio)")

    let valid = engine.verifyAggregate(aggregate)
    expect(valid, "single leaf aggregate verifies")
}
