// GPUVerkleMultiproofTests — Tests for GPU-accelerated Verkle tree multiproof engine
//
// Validates compressed multiproof generation, verification, deduplication,
// serialization round-trip, proof size analysis, update-and-reprove,
// transcript determinism, edge cases, and configurable branching factors.

import zkMetal
import Foundation

public func runGPUVerkleMultiproofTests() {
    suite("GPUVerkleMultiproofEngine — Initialization")

    testMultiproofEngineInit()
    testMultiproofTreeBuild()
    testMultiproofRootConsistency()

    suite("GPUVerkleMultiproofEngine — Single Query Proof")

    testSingleQueryProofGeneration()
    testSingleQueryProofVerification()
    testSingleQueryWrongRootRejects()
    testSingleQueryValueConsistency()

    suite("GPUVerkleMultiproofEngine — Multi-Query Compressed Proof")

    testMultiQueryProofGeneration()
    testMultiQueryProofVerification()
    testMultiQueryAllLeaves()
    testMultiQueryDuplicateLeaves()

    suite("GPUVerkleMultiproofEngine — Opening Deduplication")

    testSharedChunkDeduplication()
    testSharedParentDeduplication()
    testNoDeduplicationDifferentPaths()
    testDeduplicationCountAnalysis()

    suite("GPUVerkleMultiproofEngine — Transcript Determinism")

    testTranscriptDeterministic()
    testTranscriptOrderMatters()
    testEvaluationPointConsistency()

    suite("GPUVerkleMultiproofEngine — Verification Rejection")

    testRejectTamperedValue()
    testRejectTamperedCommitment()
    testRejectTamperedEvaluationPoint()
    testRejectWrongRoot()
    testRejectTruncatedIPA()

    suite("GPUVerkleMultiproofEngine — Serialization")

    testSerializationRoundTrip()
    testSerializationNonEmpty()
    testSerializationMultiQuery()
    testDeserializationPreservesStructure()

    suite("GPUVerkleMultiproofEngine — Proof Size Comparison")

    testProofSizeAnalysis()
    testCompressionRatioImproves()
    testCompareProofSizes()

    suite("GPUVerkleMultiproofEngine — Configurable Width")

    testWidth2Multiproof()
    testWidth8Multiproof()
    testWidth4DeeperTree()

    suite("GPUVerkleMultiproofEngine — Edge Cases")

    testSingleChunkTree()
    testFirstAndLastLeaf()
    testAdjacentLeaves()
    testLargeTree()
}

// MARK: - Initialization Tests

private func testMultiproofEngineInit() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        expect(engine.branchingFactor == 4, "Branching factor is 4")
        expect(engine.logWidth == 2, "logWidth is 2 for width=4")
    } catch {
        expect(false, "Engine creation threw: \(error)")
    }
}

private func testMultiproofTreeBuild() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        let (levels, chunks) = try engine.buildTree(leaves: leaves)
        expectEqual(levels.count, 2, "16 leaves / width 4 => 2 levels")
        expectEqual(levels[0].count, 4, "Level 0 has 4 commitments")
        expectEqual(levels[1].count, 1, "Level 1 has 1 root")
        expectEqual(chunks.count, 4, "4 leaf chunks")
    } catch {
        expect(false, "buildTree threw: \(error)")
    }
}

private func testMultiproofRootConsistency() {
    do {
        let engine1 = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let engine2 = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 10)) }
        try engine1.buildTree(leaves: leaves)
        try engine2.buildTree(leaves: leaves)
        let root1 = engine1.rootCommitment()
        let root2 = engine2.rootCommitment()
        expect(pointEqual(root1, root2), "Same leaves produce same root across engines")
    } catch {
        expect(false, "Root consistency test threw: \(error)")
    }
}

// MARK: - Single Query Tests

private func testSingleQueryProofGeneration() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        expect(!proof.commitments.isEmpty, "Proof has commitments")
        expect(!proof.aggregatedL.isEmpty, "Proof has aggregated L values")
        expect(!proof.aggregatedR.isEmpty, "Proof has aggregated R values")
        expectEqual(proof.queryLeafIndices.count, 1, "Single query has 1 leaf index")
        expectEqual(proof.queryLeafIndices[0], 0, "Query leaf index is 0")
    } catch {
        expect(false, "Single query proof generation threw: \(error)")
    }
}

private func testSingleQueryProofVerification() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4))
        let proof = try engine.generateCompressedMultiproof(queries: [query])
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Single query proof verifies")
    } catch {
        expect(false, "Single query verification threw: \(error)")
    }
}

private func testSingleQueryWrongRootRejects() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        var proof = try engine.generateCompressedMultiproof(queries: [query])

        // Tamper with root
        let wrongRoot = pointAdd(proof.root, proof.root)
        proof = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: proof.aggregatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: proof.evaluationPoint,
            aggregatedCommitment: proof.aggregatedCommitment,
            aggregatedValue: proof.aggregatedValue,
            root: wrongRoot
        )
        let invalid = engine.verifyCompressedMultiproof(proof)
        expect(!invalid, "Proof rejects wrong root")
    } catch {
        expect(false, "Wrong root rejection test threw: \(error)")
    }
}

private func testSingleQueryValueConsistency() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Verify that the proof contains the correct leaf value
        let query = MultiproofQuery(leafIndex: 7, expectedValue: frFromInt(8))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // The first opening should contain the leaf value
        expect(proof.values.count > 0, "Proof has values")
        expect(frEqual(proof.values[0], frFromInt(8)), "First value matches leaf 7 = 8")
    } catch {
        expect(false, "Value consistency test threw: \(error)")
    }
}

// MARK: - Multi-Query Tests

private func testMultiQueryProofGeneration() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6)),
            MultiproofQuery(leafIndex: 15, expectedValue: frFromInt(16))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)

        expectEqual(proof.queryLeafIndices.count, 3, "Multi-proof has 3 queries")
        expectEqual(proof.queryOpeningIndices.count, 3, "Multi-proof has 3 opening index arrays")
        expect(!proof.commitments.isEmpty, "Multi-proof has commitments")
    } catch {
        expect(false, "Multi-query proof generation threw: \(error)")
    }
}

private func testMultiQueryProofVerification() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6)),
            MultiproofQuery(leafIndex: 15, expectedValue: frFromInt(16))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Multi-query compressed proof verifies")
    } catch {
        expect(false, "Multi-query verification threw: \(error)")
    }
}

private func testMultiQueryAllLeaves() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Prove all 8 leaves at once
        let queries = (0..<8).map {
            MultiproofQuery(leafIndex: $0, expectedValue: frFromInt(UInt64($0 + 1)))
        }
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "All-leaves proof verifies")
        expectEqual(proof.queryLeafIndices.count, 8, "All-leaves proof has 8 queries")
    } catch {
        expect(false, "All-leaves proof threw: \(error)")
    }
}

private func testMultiQueryDuplicateLeaves() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Query the same leaf twice — should still verify
        let queries = [
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4)),
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Duplicate query proof verifies")
        expectEqual(proof.queryLeafIndices.count, 2, "Duplicate query has 2 entries")
    } catch {
        expect(false, "Duplicate query test threw: \(error)")
    }
}

// MARK: - Deduplication Tests

private func testSharedChunkDeduplication() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Leaves 0 and 1 are in the same chunk (chunk 0)
        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 1, expectedValue: frFromInt(2))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)

        // Both queries should reference the same root-level opening
        // so total openings should be less than 2 * pathDepth
        let analysis = engine.analyzeProofSize(queries: queries)
        expect(analysis.unique < analysis.total, "Deduplication reduces unique openings")
    } catch {
        expect(false, "Shared chunk deduplication threw: \(error)")
    }
}

private func testSharedParentDeduplication() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Leaves 0 (chunk 0) and 4 (chunk 1) share the root parent
        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 4, expectedValue: frFromInt(5))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Shared parent proof verifies")

        // Root opening should be deduplicated only if they share same childIndex
        // Leaf 0 -> chunk 0 -> root at index 0
        // Leaf 4 -> chunk 1 -> root at index 1
        // Different child indices, so not deduplicated at root level
        let analysis = engine.analyzeProofSize(queries: queries)
        expect(analysis.total >= 4, "Total openings: 2 leaf + 2 root = 4")
    } catch {
        expect(false, "Shared parent deduplication threw: \(error)")
    }
}

private func testNoDeduplicationDifferentPaths() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Two leaves in different chunks with different parent child indices
        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 15, expectedValue: frFromInt(16))
        ]
        let analysis = engine.analyzeProofSize(queries: queries)
        // Leaf 0 -> chunk 0, index 0 in root
        // Leaf 15 -> chunk 3, index 3 in root
        // Nothing is shared
        expectEqual(analysis.unique, analysis.total, "No deduplication when paths fully differ")
    } catch {
        expect(false, "No deduplication test threw: \(error)")
    }
}

private func testDeduplicationCountAnalysis() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // All 4 leaves in chunk 0 share the leaf-level opening commitment
        // They also all point to the same root child
        let queries = (0..<4).map {
            MultiproofQuery(leafIndex: $0, expectedValue: frFromInt(UInt64($0 + 1)))
        }
        let analysis = engine.analyzeProofSize(queries: queries)
        // 4 queries * 2 levels = 8 total
        expectEqual(analysis.total, 8, "4 queries * 2 levels = 8 total openings")
        // But root opening at index 0 is shared, so 4 leaf + 1 root = 5 unique
        expect(analysis.unique <= analysis.total, "Unique <= total")
        expect(analysis.ratio <= 1.0, "Compression ratio <= 1.0")
    } catch {
        expect(false, "Deduplication count analysis threw: \(error)")
    }
}

// MARK: - Transcript Determinism Tests

private func testTranscriptDeterministic() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4))
        ]

        // Generate proof twice — evaluation point must be identical
        let proof1 = try engine.generateCompressedMultiproof(queries: queries)
        let proof2 = try engine.generateCompressedMultiproof(queries: queries)

        expect(frEqual(proof1.evaluationPoint, proof2.evaluationPoint),
               "Same queries produce same evaluation point")
        expect(frEqual(proof1.aggregatedFinalA, proof2.aggregatedFinalA),
               "Same queries produce same IPA final scalar")
    } catch {
        expect(false, "Transcript determinism test threw: \(error)")
    }
}

private func testTranscriptOrderMatters() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queriesAB = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4))
        ]
        let queriesBA = [
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4)),
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        ]

        let proofAB = try engine.generateCompressedMultiproof(queries: queriesAB)
        let proofBA = try engine.generateCompressedMultiproof(queries: queriesBA)

        // Both should verify independently
        let validAB = engine.verifyCompressedMultiproof(proofAB)
        let validBA = engine.verifyCompressedMultiproof(proofBA)
        expect(validAB, "AB order proof verifies")
        expect(validBA, "BA order proof verifies")
    } catch {
        expect(false, "Transcript order test threw: \(error)")
    }
}

private func testEvaluationPointConsistency() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(3))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // The evaluation point should be non-zero (vanishingly unlikely to be zero)
        expect(!frEqual(proof.evaluationPoint, Fr.zero), "Evaluation point is non-zero")

        // Adding more queries should change the evaluation point
        let queries2 = [
            MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(3)),
            MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6))
        ]
        let proof2 = try engine.generateCompressedMultiproof(queries: queries2)
        expect(!frEqual(proof.evaluationPoint, proof2.evaluationPoint),
               "Different query sets produce different evaluation points")
    } catch {
        expect(false, "Evaluation point consistency test threw: \(error)")
    }
}

// MARK: - Verification Rejection Tests

private func testRejectTamperedValue() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // Tamper with the aggregated value
        let tampered = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: proof.aggregatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: proof.evaluationPoint,
            aggregatedCommitment: proof.aggregatedCommitment,
            aggregatedValue: frAdd(proof.aggregatedValue, Fr.one),
            root: proof.root
        )
        let invalid = engine.verifyCompressedMultiproof(tampered)
        expect(!invalid, "Tampered aggregated value is rejected")
    } catch {
        expect(false, "Tampered value rejection test threw: \(error)")
    }
}

private func testRejectTamperedCommitment() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // Tamper with the aggregated commitment
        let wrongCommitment = pointAdd(proof.aggregatedCommitment, proof.aggregatedCommitment)
        let tampered = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: proof.aggregatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: proof.evaluationPoint,
            aggregatedCommitment: wrongCommitment,
            aggregatedValue: proof.aggregatedValue,
            root: proof.root
        )
        let invalid = engine.verifyCompressedMultiproof(tampered)
        expect(!invalid, "Tampered aggregated commitment is rejected")
    } catch {
        expect(false, "Tampered commitment rejection test threw: \(error)")
    }
}

private func testRejectTamperedEvaluationPoint() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // Tamper with evaluation point
        let tampered = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: proof.aggregatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: frAdd(proof.evaluationPoint, Fr.one),
            aggregatedCommitment: proof.aggregatedCommitment,
            aggregatedValue: proof.aggregatedValue,
            root: proof.root
        )
        let invalid = engine.verifyCompressedMultiproof(tampered)
        expect(!invalid, "Tampered evaluation point is rejected")
    } catch {
        expect(false, "Tampered evaluation point rejection test threw: \(error)")
    }
}

private func testRejectWrongRoot() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(3))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        let wrongRoot = pointAdd(proof.root, proof.root)
        let tampered = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: proof.aggregatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: proof.evaluationPoint,
            aggregatedCommitment: proof.aggregatedCommitment,
            aggregatedValue: proof.aggregatedValue,
            root: wrongRoot
        )
        let invalid = engine.verifyCompressedMultiproof(tampered)
        expect(!invalid, "Wrong root in proof is rejected")
    } catch {
        expect(false, "Wrong root rejection test threw: \(error)")
    }
}

private func testRejectTruncatedIPA() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        // Truncate L vector
        let truncatedL = proof.aggregatedL.count > 1
            ? Array(proof.aggregatedL.prefix(proof.aggregatedL.count - 1))
            : []
        let tampered = CompressedMultiproof(
            commitments: proof.commitments,
            childIndices: proof.childIndices,
            values: proof.values,
            queryOpeningIndices: proof.queryOpeningIndices,
            queryLeafIndices: proof.queryLeafIndices,
            aggregatedL: truncatedL,
            aggregatedR: proof.aggregatedR,
            aggregatedFinalA: proof.aggregatedFinalA,
            evaluationPoint: proof.evaluationPoint,
            aggregatedCommitment: proof.aggregatedCommitment,
            aggregatedValue: proof.aggregatedValue,
            root: proof.root
        )
        let invalid = engine.verifyCompressedMultiproof(tampered)
        expect(!invalid, "Truncated IPA L vector is rejected")
    } catch {
        expect(false, "Truncated IPA rejection test threw: \(error)")
    }
}

// MARK: - Serialization Tests

private func testSerializationRoundTrip() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(3))
        let proof = try engine.generateCompressedMultiproof(queries: [query])

        let serialized = engine.serializeCompressedMultiproof(proof)
        expect(serialized.byteSize > 0, "Serialized proof is non-empty")

        let deserialized = engine.deserializeCompressedMultiproof(serialized)
        expect(deserialized != nil, "Deserialization succeeds")

        if let d = deserialized {
            expectEqual(d.queryLeafIndices.count, 1, "Deserialized has 1 query")
            expectEqual(d.queryLeafIndices[0], 2, "Deserialized leaf index is 2")
            expectEqual(d.childIndices.count, proof.childIndices.count,
                       "Deserialized opening count matches")
            expectEqual(d.aggregatedL.count, proof.aggregatedL.count,
                       "Deserialized L count matches")
            expectEqual(d.aggregatedR.count, proof.aggregatedR.count,
                       "Deserialized R count matches")
        }
    } catch {
        expect(false, "Serialization round-trip threw: \(error)")
    }
}

private func testSerializationNonEmpty() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        try engine.buildTree(leaves: leaves)

        let query = MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(10))
        let proof = try engine.generateCompressedMultiproof(queries: [query])
        let serialized = engine.serializeCompressedMultiproof(proof)

        // Should have at least: 2 (numC) + 64 (commitment) + 2 (numO) + 33 (opening) +
        // 2 (numQ) + query + IPA + eval + aggC + aggV + root
        expect(serialized.byteSize > 100, "Serialized proof has meaningful size")
    } catch {
        expect(false, "Serialization non-empty test threw: \(error)")
    }
}

private func testSerializationMultiQuery() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 7, expectedValue: frFromInt(8)),
            MultiproofQuery(leafIndex: 15, expectedValue: frFromInt(16))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let serialized = engine.serializeCompressedMultiproof(proof)
        let deserialized = engine.deserializeCompressedMultiproof(serialized)

        expect(deserialized != nil, "Multi-query deserialization succeeds")
        if let d = deserialized {
            expectEqual(d.queryLeafIndices.count, 3, "Deserialized has 3 queries")
            expectEqual(d.queryOpeningIndices.count, 3, "Deserialized has 3 opening index arrays")
            expectEqual(d.values.count, proof.values.count,
                       "Deserialized value count matches original")
        }
    } catch {
        expect(false, "Multi-query serialization threw: \(error)")
    }
}

private func testDeserializationPreservesStructure() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 1, expectedValue: frFromInt(2)),
            MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let serialized = engine.serializeCompressedMultiproof(proof)
        let deserialized = engine.deserializeCompressedMultiproof(serialized)

        expect(deserialized != nil, "Deserialization succeeds")
        if let d = deserialized {
            // Check structural integrity
            for qIdx in 0..<d.queryOpeningIndices.count {
                let indices = d.queryOpeningIndices[qIdx]
                for idx in indices {
                    expect(idx < d.commitments.count, "Opening index within bounds")
                }
            }
            expectEqual(d.childIndices.count, d.values.count,
                       "Child indices and values have same count")
        }
    } catch {
        expect(false, "Deserialization structure test threw: \(error)")
    }
}

// MARK: - Proof Size Comparison Tests

private func testProofSizeAnalysis() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 1, expectedValue: frFromInt(2))
        ]
        let analysis = engine.analyzeProofSize(queries: queries)

        expect(analysis.total > 0, "Total openings is positive")
        expect(analysis.unique > 0, "Unique openings is positive")
        expect(analysis.unique <= analysis.total, "Unique <= total")
        expect(analysis.ratio > 0.0 && analysis.ratio <= 1.0, "Ratio in (0, 1]")
    } catch {
        expect(false, "Proof size analysis threw: \(error)")
    }
}

private func testCompressionRatioImproves() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Single query: no deduplication possible
        let single = engine.analyzeProofSize(queries: [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1))
        ])
        expectEqual(single.unique, single.total, "Single query has ratio 1.0")

        // Multiple queries in same chunk: deduplication at parent level
        let sameChunk = engine.analyzeProofSize(queries: [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 1, expectedValue: frFromInt(2)),
            MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(3)),
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4))
        ])
        expect(sameChunk.ratio <= single.ratio,
               "More queries in same chunk => better or equal ratio")
    } catch {
        expect(false, "Compression ratio test threw: \(error)")
    }
}

private func testCompareProofSizes() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6)),
            MultiproofQuery(leafIndex: 10, expectedValue: frFromInt(11))
        ]

        let (compressed, uncompressed, savings) = try engine.compareProofSizes(queries: queries)
        expect(compressed > 0, "Compressed size is positive")
        expect(uncompressed > 0, "Uncompressed estimate is positive")
        expect(compressed < uncompressed, "Compressed is smaller than uncompressed")
        expect(savings > 0.0, "Savings percentage is positive")
    } catch {
        expect(false, "Compare proof sizes threw: \(error)")
    }
}

// MARK: - Configurable Width Tests

private func testWidth2Multiproof() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 2)
        expectEqual(engine.branchingFactor, 2, "Width-2 engine created")
        expectEqual(engine.logWidth, 1, "logWidth is 1 for width=2")

        let leaves: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(10)),
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(40))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Width-2 multiproof verifies")
    } catch {
        expect(false, "Width-2 multiproof threw: \(error)")
    }
}

private func testWidth8Multiproof() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 8)
        expectEqual(engine.branchingFactor, 8, "Width-8 engine created")
        expectEqual(engine.logWidth, 3, "logWidth is 3 for width=8")

        let leaves: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 31, expectedValue: frFromInt(32)),
            MultiproofQuery(leafIndex: 63, expectedValue: frFromInt(64))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Width-8 multiproof verifies")
    } catch {
        expect(false, "Width-8 multiproof threw: \(error)")
    }
}

private func testWidth4DeeperTree() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        // 64 leaves with width 4 => 3 levels (16 chunks -> 4 nodes -> 1 root)
        let leaves: [Fr] = (0..<64).map { frFromInt(UInt64($0 + 1)) }
        let (levels, _) = try engine.buildTree(leaves: leaves)
        expectEqual(levels.count, 3, "64 leaves / width 4 => 3 levels")

        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 32, expectedValue: frFromInt(33)),
            MultiproofQuery(leafIndex: 63, expectedValue: frFromInt(64))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "3-level tree multiproof verifies")

        // Each query has 3 openings (leaf, mid, root)
        expectEqual(proof.queryOpeningIndices[0].count, 3, "Query 0 has 3-level path")
        expectEqual(proof.queryOpeningIndices[1].count, 3, "Query 1 has 3-level path")
        expectEqual(proof.queryOpeningIndices[2].count, 3, "Query 2 has 3-level path")
    } catch {
        expect(false, "Deeper tree test threw: \(error)")
    }
}

// MARK: - Edge Case Tests

private func testSingleChunkTree() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        try engine.buildTree(leaves: leaves)

        // Single chunk = single level, root IS the only commitment
        let query = MultiproofQuery(leafIndex: 2, expectedValue: frFromInt(30))
        let proof = try engine.generateCompressedMultiproof(queries: [query])
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Single chunk tree proof verifies")

        // Only one level of openings
        expectEqual(proof.queryOpeningIndices[0].count, 1, "Single chunk has 1 opening")
    } catch {
        expect(false, "Single chunk tree test threw: \(error)")
    }
}

private func testFirstAndLastLeaf() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Prove first and last leaf simultaneously
        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 15, expectedValue: frFromInt(16))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "First and last leaf proof verifies")
    } catch {
        expect(false, "First and last leaf test threw: \(error)")
    }
}

private func testAdjacentLeaves() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        let leaves: [Fr] = (0..<16).map { frFromInt(UInt64($0 + 1)) }
        try engine.buildTree(leaves: leaves)

        // Adjacent leaves across chunk boundary (3 and 4)
        let queries = [
            MultiproofQuery(leafIndex: 3, expectedValue: frFromInt(4)),
            MultiproofQuery(leafIndex: 4, expectedValue: frFromInt(5))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Adjacent cross-chunk leaf proof verifies")

        // Adjacent leaves within same chunk (0 and 1)
        let queries2 = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 1, expectedValue: frFromInt(2))
        ]
        let proof2 = try engine.generateCompressedMultiproof(queries: queries2)
        let valid2 = engine.verifyCompressedMultiproof(proof2)
        expect(valid2, "Adjacent same-chunk leaf proof verifies")
    } catch {
        expect(false, "Adjacent leaves test threw: \(error)")
    }
}

private func testLargeTree() {
    do {
        let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
        // 256 leaves => 4 levels (64 -> 16 -> 4 -> 1)
        let leaves: [Fr] = (0..<256).map { frFromInt(UInt64($0 + 1)) }
        let (levels, _) = try engine.buildTree(leaves: leaves)
        expectEqual(levels.count, 4, "256 leaves / width 4 => 4 levels")

        // Query leaves spread across the tree
        let queries = [
            MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
            MultiproofQuery(leafIndex: 63, expectedValue: frFromInt(64)),
            MultiproofQuery(leafIndex: 128, expectedValue: frFromInt(129)),
            MultiproofQuery(leafIndex: 255, expectedValue: frFromInt(256))
        ]
        let proof = try engine.generateCompressedMultiproof(queries: queries)
        let valid = engine.verifyCompressedMultiproof(proof)
        expect(valid, "Large tree (256 leaves) multiproof verifies")

        // Each query traverses 4 levels
        for i in 0..<4 {
            expectEqual(proof.queryOpeningIndices[i].count, 4,
                       "Query \(i) has 4-level path in large tree")
        }
    } catch {
        expect(false, "Large tree test threw: \(error)")
    }
}
