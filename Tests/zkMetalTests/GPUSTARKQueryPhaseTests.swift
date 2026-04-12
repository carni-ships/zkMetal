// GPU STARK Query Phase Tests — validates GPU-accelerated STARK query/decommitment engine
//
// Tests: query sampling, trace/composition/FRI decommitments, deep eval at queries,
// Merkle path extraction/verification, FRI index folding, deduplication, invalid inputs

import zkMetal
import Foundation

public func runGPUSTARKQueryPhaseTests() {
    suite("GPU STARK Query Phase — Query Sampling")
    testQuerySampling()

    suite("GPU STARK Query Phase — Query Sampling Determinism")
    testQuerySamplingDeterminism()

    suite("GPU STARK Query Phase — Trace Decommitment")
    testTraceDecommitment()

    suite("GPU STARK Query Phase — Composition Decommitment")
    testCompositionDecommitment()

    suite("GPU STARK Query Phase — FRI Query Decommitment")
    testFRIQueryDecommitment()

    suite("GPU STARK Query Phase — GPU Batch FRI Decommitment")
    testGPUBatchFRIDecommitment()

    suite("GPU STARK Query Phase — Deep Composition At Queries")
    testDeepCompositionAtQueries()

    suite("GPU STARK Query Phase — Batch Decommitment Pipeline")
    testBatchDecommitmentPipeline()

    suite("GPU STARK Query Phase — Merkle Path Extraction")
    testMerklePathExtraction()

    suite("GPU STARK Query Phase — Merkle Path Verification")
    testMerklePathVerification()

    suite("GPU STARK Query Phase — Batch Merkle Verify")
    testBatchMerkleVerify()

    suite("GPU STARK Query Phase — FRI Index Folding")
    testFRIIndexFolding()

    suite("GPU STARK Query Phase — Query Deduplication")
    testQueryDeduplication()

    suite("GPU STARK Query Phase — Sibling Index")
    testSiblingIndex()

    suite("GPU STARK Query Phase — Domain Points")
    testDomainPoints()

    suite("GPU STARK Query Phase — Invalid Input Rejection")
    testSTARKQueryInvalidInputRejection()

    suite("GPU STARK Query Phase — Consistency Verification")
    testConsistencyVerification()

    suite("GPU STARK Query Phase — Build Merkle Tree + Security")
    testBuildTraceMerkleTreeAndSecurity()
}

// MARK: - Helpers

/// Build a simple test config.
private func makeConfig(
    logTraceLen: Int = 3, blowupFactor: Int = 4, numTraceColumns: Int = 2,
    numQueries: Int = 4, numFRIRounds: Int = 2, numCompositionSegments: Int = 1
) -> STARKQueryPhaseConfig {
    return STARKQueryPhaseConfig(
        logTraceLen: logTraceLen, blowupFactor: blowupFactor,
        numTraceColumns: numTraceColumns, numQueries: numQueries,
        numFRIRounds: numFRIRounds, numCompositionSegments: numCompositionSegments)
}

/// Build simple LDE columns: col_c[i] = frFromInt(c * domainSize + i + 1).
private func buildSimpleLDE(config: STARKQueryPhaseConfig) -> [[Fr]] {
    let m = config.ldeDomainSize
    var cols = [[Fr]]()
    for c in 0..<config.numTraceColumns {
        var col = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            col[i] = frFromInt(UInt64(c * m + i + 1))
        }
        cols.append(col)
    }
    return cols
}

/// Build simple composition segments: seg_s[i] = frFromInt(100 + s * domainSize + i).
private func buildSimpleComposition(config: STARKQueryPhaseConfig) -> [[Fr]] {
    let m = config.ldeDomainSize
    var segs = [[Fr]]()
    for s in 0..<config.numCompositionSegments {
        var seg = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<m {
            seg[i] = frFromInt(UInt64(100 + s * m + i))
        }
        segs.append(seg)
    }
    return segs
}

/// Build a Merkle tree from LDE columns using the engine.
private func buildTree(
    engine: GPUSTARKQueryPhaseEngine, ldeColumns: [[Fr]], domainSize: Int
) -> (root: Fr, leaves: [Fr], nodes: [Fr]) {
    return engine.buildTraceMerkleTree(ldeColumns: ldeColumns, domainSize: domainSize)
}

/// Build FRI layer data: each layer halves in size, values = frFromInt(layerIdx * size + i).
private func buildFRILayers(
    engine: GPUSTARKQueryPhaseEngine, config: STARKQueryPhaseConfig
) -> (layers: [[Fr]], merkleData: [(leaves: [Fr], nodes: [Fr], root: Fr)]) {
    var layers = [[Fr]]()
    var merkleData = [(leaves: [Fr], nodes: [Fr], root: Fr)]()
    var layerSize = config.ldeDomainSize

    for round in 0..<config.numFRIRounds {
        layerSize = layerSize / 2
        let size = max(layerSize, 2)
        var layerVals = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<size {
            layerVals[i] = frFromInt(UInt64(1000 + round * 100 + i))
        }
        layers.append(layerVals)

        let (root, leaves, nodes) = engine.buildMerkleTree(leaves: layerVals)
        merkleData.append((leaves: leaves, nodes: nodes, root: root))
    }

    return (layers, merkleData)
}

// MARK: - Query Sampling

private func testQuerySampling() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numQueries: 8)
        let seed = frFromInt(42)

        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        // Should produce the requested number of queries
        expectEqual(querySet.indices.count, 8, "Should sample 8 query indices")

        // All indices should be within domain
        let domainSize = config.ldeDomainSize
        for idx in querySet.indices {
            expect(idx >= 0 && idx < domainSize,
                   "Query index \(idx) should be in [0, \(domainSize))")
        }

        // Indices should be unique
        let uniqueSet = Set(querySet.indices)
        expectEqual(uniqueSet.count, querySet.indices.count,
                    "All query indices should be unique")

        // Indices should be sorted
        let sorted = querySet.indices.sorted()
        for i in 0..<querySet.indices.count {
            expectEqual(querySet.indices[i], sorted[i],
                        "Query indices should be sorted")
        }

        // Domain size should match config
        expectEqual(querySet.domainSize, domainSize, "Domain size should match config")

    } catch {
        expect(false, "Query sampling test failed: \(error)")
    }
}

// MARK: - Query Sampling Determinism

private func testQuerySamplingDeterminism() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numQueries: 6)

        // Same seed should produce same indices
        let seed = frFromInt(99)
        let qs1 = engine.sampleQueryIndices(seed: seed, config: config)
        let qs2 = engine.sampleQueryIndices(seed: seed, config: config)

        for i in 0..<qs1.indices.count {
            expectEqual(qs1.indices[i], qs2.indices[i],
                        "Same seed should produce same index at position \(i)")
        }

        // Different seeds should produce different indices
        let seed2 = frFromInt(100)
        let qs3 = engine.sampleQueryIndices(seed: seed2, config: config)
        var anyDifferent = false
        for i in 0..<qs1.indices.count {
            if qs1.indices[i] != qs3.indices[i] {
                anyDifferent = true
                break
            }
        }
        expect(anyDifferent, "Different seeds should produce different indices")

    } catch {
        expect(false, "Query sampling determinism test failed: \(error)")
    }
}

// MARK: - Trace Decommitment

private func testTraceDecommitment() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig()
        let ldeColumns = buildSimpleLDE(config: config)
        let (root, leaves, nodes) = buildTree(
            engine: engine, ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)

        let seed = frFromInt(7)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let traceDecomm = try engine.generateTraceDecommitments(
            querySet: querySet, ldeColumns: ldeColumns,
            merkleLeaves: leaves, merkleNodes: nodes,
            commitment: root, config: config)

        expectEqual(traceDecomm.count, querySet.indices.count,
                    "Should have one decommitment per query")

        // Check that decommitment values match LDE columns
        for (qi, td) in traceDecomm.enumerated() {
            let idx = querySet.indices[qi]
            expectEqual(td.queryIndex, idx, "Query index should match")
            expectEqual(td.values.count, config.numTraceColumns,
                        "Should have one value per trace column")

            for c in 0..<config.numTraceColumns {
                expect(frEqual(td.values[c], ldeColumns[c][idx]),
                       "Trace value at col \(c), idx \(idx) should match LDE")
            }

            // Merkle path should verify
            expect(engine.verifyMerklePath(td.merklePath, expectedRoot: root),
                   "Trace Merkle path should verify for query \(qi)")
        }

    } catch {
        expect(false, "Trace decommitment test failed: \(error)")
    }
}

// MARK: - Composition Decommitment

private func testCompositionDecommitment() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig()
        let compSegs = buildSimpleComposition(config: config)

        // Build Merkle tree for composition (single-column)
        let (root, leaves, nodes) = engine.buildMerkleTree(leaves: compSegs[0])

        let seed = frFromInt(11)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let compDecomm = try engine.generateCompositionDecommitments(
            querySet: querySet, compositionSegments: compSegs,
            merkleLeaves: leaves, merkleNodes: nodes,
            commitment: root, config: config)

        expectEqual(compDecomm.count, querySet.indices.count,
                    "Should have one composition decommitment per query")

        for (qi, cd) in compDecomm.enumerated() {
            let idx = querySet.indices[qi]
            expectEqual(cd.queryIndex, idx, "Query index should match")
            expectEqual(cd.segments.count, config.numCompositionSegments,
                        "Should have one segment value per composition segment")

            for s in 0..<config.numCompositionSegments {
                expect(frEqual(cd.segments[s], compSegs[s][idx]),
                       "Composition segment \(s) at idx \(idx) should match")
            }

            // Merkle path should verify
            expect(engine.verifyMerklePath(cd.merklePath, expectedRoot: root),
                   "Composition Merkle path should verify for query \(qi)")
        }

    } catch {
        expect(false, "Composition decommitment test failed: \(error)")
    }
}

// MARK: - FRI Query Decommitment

private func testFRIQueryDecommitment() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numFRIRounds: 3)
        let seed = frFromInt(13)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let (friLayers, friMerkle) = buildFRILayers(engine: engine, config: config)

        let friDecomm = try engine.generateFRIDecommitments(
            querySet: querySet, friLayers: friLayers,
            friMerkleData: friMerkle, config: config)

        expectEqual(friDecomm.count, querySet.indices.count,
                    "Should have one FRI decommitment per query")

        for fd in friDecomm {
            expectEqual(fd.layers.count, config.numFRIRounds,
                        "Should have one layer decommitment per FRI round")
            expectEqual(fd.foldedIndices.count, config.numFRIRounds,
                        "Should have one folded index per FRI round")

            // Each layer's value should be from the FRI layer data
            for (round, layer) in fd.layers.enumerated() {
                let fIdx = fd.foldedIndices[round]
                expect(fIdx >= 0 && fIdx < friLayers[round].count,
                       "Folded index should be within layer bounds")
                expect(frEqual(layer.value, friLayers[round][fIdx]),
                       "FRI layer value should match at round \(round)")
            }
        }

    } catch {
        expect(false, "FRI query decommitment test failed: \(error)")
    }
}

// MARK: - GPU Batch FRI Decommitment

/// Test the GPU-accelerated batch FRI query decommitment.
private func testGPUBatchFRIDecommitment() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numQueries: 32, numFRIRounds: 3)  // 32 >= gpuThreshold of 16
        let seed = frFromInt(13)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let (friLayers, friMerkle) = buildFRILayers(engine: engine, config: config)

        // Build domain inverses (needed for GPU batch path)
        var domainInvs = [[Fr]]()
        for round in 0..<config.numFRIRounds {
            let size = friLayers[round].count
            var invs = [Fr](repeating: .zero, count: size)
            for i in 0..<size {
                // Simple domain inverse: 1/(i+1)
                let val = frFromInt(UInt64(i + 1))
                invs[i] = frInverse(val)
            }
            domainInvs.append(invs)
        }

        // Fake challenges (one per FRI round)
        let challenges = [
            frFromInt(42), frFromInt(43), frFromInt(44)
        ]

        // GPU batch path — produce FRI decommitments
        let gpuDecomm = try engine.gpuBatchFRIDecommit(
            querySet: querySet,
            friLayers: friLayers,
            friMerkleData: friMerkle,
            domainInvs: domainInvs,
            challenges: challenges,
            config: config)

        // Verify structure: one decommitment per query, one layer per FRI round
        expectEqual(gpuDecomm.count, querySet.indices.count,
                    "Should have one FRI decommitment per query")

        for fd in gpuDecomm {
            expectEqual(fd.layers.count, config.numFRIRounds,
                        "Should have one layer decommitment per FRI round")
            expectEqual(fd.foldedIndices.count, config.numFRIRounds,
                        "Should have one folded index per FRI round")

            // Each layer's value should be from the FRI layer data
            for (round, layer) in fd.layers.enumerated() {
                let fIdx = fd.foldedIndices[round]
                expect(fIdx >= 0 && fIdx < friLayers[round].count,
                       "Folded index should be within layer bounds at round \(round)")
                expect(frEqual(layer.value, friLayers[round][fIdx]),
                       "FRI layer value should match at round \(round)")
            }
        }

        // Also verify that generateDecommitments with domainInvs uses GPU path
        let ldeColumns = buildSimpleLDE(config: config)
        let compSegs = buildSimpleComposition(config: config)
        let (traceRoot, traceLeaves, traceNodes) = buildTree(
            engine: engine, ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)
        let (compRoot, compLeaves, compNodes) = engine.buildMerkleTree(leaves: compSegs[0])
        let zeta = frFromInt(7)
        let oodFrame = OODEvaluationFrame(
            zeta: zeta,
            traceEvals: [frFromInt(10), frFromInt(20)],
            traceNextEvals: [frFromInt(30), frFromInt(40)],
            compositionEvals: [frFromInt(50)])
        let alpha = frFromInt(5)

        let result = try engine.generateDecommitments(
            querySet: querySet,
            ldeColumns: ldeColumns,
            traceMerkleLeaves: traceLeaves,
            traceMerkleNodes: traceNodes,
            traceCommitment: traceRoot,
            compositionSegments: compSegs,
            compMerkleLeaves: compLeaves,
            compMerkleNodes: compNodes,
            compCommitment: compRoot,
            friLayers: friLayers,
            friMerkleData: friMerkle,
            domainInvs: domainInvs,
            challenges: challenges,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config)

        expectEqual(result.friDecommitments.count, querySet.indices.count,
                    "FRI decommitment count should match")

        expect(true, "GPU batch FRI decommitment")
    } catch {
        expect(false, "GPU batch FRI decommitment test failed: \(error)")
    }
}

// MARK: - Deep Composition At Queries

private func testDeepCompositionAtQueries() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig()
        let ldeColumns = buildSimpleLDE(config: config)
        let compSegs = buildSimpleComposition(config: config)
        let (traceRoot, traceLeaves, traceNodes) = buildTree(
            engine: engine, ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)
        let (compRoot, compLeaves, compNodes) = engine.buildMerkleTree(leaves: compSegs[0])

        let seed = frFromInt(17)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let traceDecomm = try engine.generateTraceDecommitments(
            querySet: querySet, ldeColumns: ldeColumns,
            merkleLeaves: traceLeaves, merkleNodes: traceNodes,
            commitment: traceRoot, config: config)

        let compDecomm = try engine.generateCompositionDecommitments(
            querySet: querySet, compositionSegments: compSegs,
            merkleLeaves: compLeaves, merkleNodes: compNodes,
            commitment: compRoot, config: config)

        // Build OOD frame with known values
        let zeta = frFromInt(7)
        let omega = frRootOfUnity(logN: config.logTraceLen)
        let traceEvals = [frFromInt(10), frFromInt(20)]
        let traceNextEvals = [frFromInt(30), frFromInt(40)]
        let compositionEvals = [frFromInt(50)]

        let oodFrame = OODEvaluationFrame(
            zeta: zeta, traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEvals: compositionEvals)

        let alpha = frFromInt(3)

        let deepEvals = try engine.evaluateDeepAtQueries(
            querySet: querySet,
            traceDecommitments: traceDecomm,
            compositionDecommitments: compDecomm,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config)

        expectEqual(deepEvals.count, querySet.indices.count,
                    "Should have one deep eval per query")

        // Deep evals should be non-zero (since trace values differ from OOD evals)
        for (i, eval) in deepEvals.enumerated() {
            expect(!frEqual(eval, Fr.zero),
                   "Deep eval at query \(i) should be non-zero")
        }

        // Different query positions should generally produce different deep evals
        if deepEvals.count >= 2 {
            var anyDifferent = false
            for i in 1..<deepEvals.count {
                if !frEqual(deepEvals[i], deepEvals[0]) {
                    anyDifferent = true
                    break
                }
            }
            expect(anyDifferent,
                   "Different query positions should produce different deep evals")
        }

    } catch {
        expect(false, "Deep composition at queries test failed: \(error)")
    }
}

// MARK: - Batch Decommitment Pipeline

private func testBatchDecommitmentPipeline() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numFRIRounds: 2)
        let ldeColumns = buildSimpleLDE(config: config)
        let compSegs = buildSimpleComposition(config: config)

        let (traceRoot, traceLeaves, traceNodes) = buildTree(
            engine: engine, ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)
        let (compRoot, compLeaves, compNodes) = engine.buildMerkleTree(leaves: compSegs[0])
        let (friLayers, friMerkle) = buildFRILayers(engine: engine, config: config)

        let zeta = frFromInt(7)
        let traceEvals = [frFromInt(10), frFromInt(20)]
        let traceNextEvals = [frFromInt(30), frFromInt(40)]
        let compositionEvals = [frFromInt(50)]
        let oodFrame = OODEvaluationFrame(
            zeta: zeta, traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEvals: compositionEvals)
        let alpha = frFromInt(5)

        let seed = frFromInt(23)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let result = try engine.generateDecommitments(
            querySet: querySet,
            ldeColumns: ldeColumns,
            traceMerkleLeaves: traceLeaves,
            traceMerkleNodes: traceNodes,
            traceCommitment: traceRoot,
            compositionSegments: compSegs,
            compMerkleLeaves: compLeaves,
            compMerkleNodes: compNodes,
            compCommitment: compRoot,
            friLayers: friLayers,
            friMerkleData: friMerkle,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config)

        // Verify structure
        expectEqual(result.numQueries, querySet.indices.count,
                    "Result should have correct query count")
        expectEqual(result.traceDecommitments.count, querySet.indices.count,
                    "Trace decommitments count should match")
        expectEqual(result.compositionDecommitments.count, querySet.indices.count,
                    "Composition decommitments count should match")
        expectEqual(result.friDecommitments.count, querySet.indices.count,
                    "FRI decommitments count should match")
        expectEqual(result.deepEvals.count, querySet.indices.count,
                    "Deep evals count should match")

        // Verify consistency
        let consistent = engine.verifyDecommitmentConsistency(
            result: result, traceCommitment: traceRoot,
            compCommitment: compRoot)
        expect(consistent, "Batch decommitment should be internally consistent")

    } catch {
        expect(false, "Batch decommitment pipeline test failed: \(error)")
    }
}

// MARK: - Merkle Path Extraction

private func testMerklePathExtraction() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // Build a small tree: 8 leaves
        let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let (root, leafVals, nodes) = engine.buildMerkleTree(leaves: leaves)

        // Extract paths for each leaf
        for i in 0..<8 {
            let path = engine.extractMerklePath(
                leafIndex: i, leaves: leafVals, nodes: nodes, root: root)

            expectEqual(path.leafIndex, i, "Leaf index should match")
            expect(frEqual(path.leaf, leaves[i]), "Leaf value should match")
            expect(frEqual(path.root, root), "Root should match")

            // Path should have log2(8) = 3 siblings
            expectEqual(path.siblings.count, 3,
                        "Path should have 3 siblings for 8 leaves")
        }

    } catch {
        expect(false, "Merkle path extraction test failed: \(error)")
    }
}

// MARK: - Merkle Path Verification

private func testMerklePathVerification() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // Build a small tree
        let leaves = (0..<4).map { frFromInt(UInt64($0 + 10)) }
        let (root, leafVals, nodes) = engine.buildMerkleTree(leaves: leaves)

        // Valid paths should verify
        for i in 0..<4 {
            let path = engine.extractMerklePath(
                leafIndex: i, leaves: leafVals, nodes: nodes, root: root)
            expect(engine.verifyMerklePath(path, expectedRoot: root),
                   "Valid path should verify at index \(i)")
        }

        // Wrong root should fail
        let wrongRoot = frFromInt(99999)
        let path0 = engine.extractMerklePath(
            leafIndex: 0, leaves: leafVals, nodes: nodes, root: root)
        expect(!engine.verifyMerklePath(path0, expectedRoot: wrongRoot),
               "Path with wrong root should fail verification")

        // Tampered leaf should fail
        var tamperedPath = FrMerklePath(
            leafIndex: 0, leaf: frFromInt(99999),
            siblings: path0.siblings, root: root)
        expect(!engine.verifyMerklePath(tamperedPath, expectedRoot: root),
               "Tampered leaf should fail verification")

    } catch {
        expect(false, "Merkle path verification test failed: \(error)")
    }
}

// MARK: - Batch Merkle Verify

private func testBatchMerkleVerify() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        let leaves = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let (root, leafVals, nodes) = engine.buildMerkleTree(leaves: leaves)

        // Build some valid paths
        var paths = [FrMerklePath]()
        for i in 0..<4 {
            let path = engine.extractMerklePath(
                leafIndex: i, leaves: leafVals, nodes: nodes, root: root)
            paths.append(path)
        }

        let results = engine.batchVerifyMerklePaths(paths, expectedRoot: root)
        expectEqual(results.count, 4, "Should have 4 verification results")

        for (i, valid) in results.enumerated() {
            expect(valid, "Path \(i) should be valid")
        }

    } catch {
        expect(false, "Batch Merkle verify test failed: \(error)")
    }
}

// MARK: - FRI Index Folding

private func testFRIIndexFolding() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // Layer sizes: 32, 16, 8
        let layerSizes = [32, 16, 8]

        // Query index 25 through 3 layers
        let folded = engine.computeFRIFoldedIndices(
            queryIndex: 25, layerSizes: layerSizes)

        expectEqual(folded.count, 3, "Should have 3 folded indices")

        // 25 % 32 = 25
        expectEqual(folded[0], 25, "Layer 0: 25 % 32 = 25")
        // 25 >> 1 = 12, then 12 % 16 = 12
        expectEqual(folded[1], 12, "Layer 1: 12 % 16 = 12")
        // 12 >> 1 = 6, then 6 % 8 = 6
        expectEqual(folded[2], 6, "Layer 2: 6 % 8 = 6")

        // Edge case: query index 0
        let folded0 = engine.computeFRIFoldedIndices(
            queryIndex: 0, layerSizes: layerSizes)
        expectEqual(folded0[0], 0, "Layer 0: 0 % 32 = 0")
        expectEqual(folded0[1], 0, "Layer 1: 0 % 16 = 0")
        expectEqual(folded0[2], 0, "Layer 2: 0 % 8 = 0")

    } catch {
        expect(false, "FRI index folding test failed: \(error)")
    }
}

// MARK: - Query Deduplication

private func testQueryDeduplication() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // Indices with duplicates when modded by domainSize=16
        let indices = [3, 19, 7, 3, 35, 7]  // 19%16=3, 35%16=3
        let (unique, mapping) = engine.deduplicateQueries(
            indices: indices, domainSize: 16)

        // unique should have [3, 7] (order of first appearance)
        expectEqual(unique.count, 2, "Should have 2 unique indices")
        expectEqual(unique[0], 3, "First unique index = 3")
        expectEqual(unique[1], 7, "Second unique index = 7")

        // mapping: [0, 0, 1, 0, 0, 1]
        expectEqual(mapping.count, 6, "Mapping should have 6 entries")
        expectEqual(mapping[0], 0, "Index 0 maps to unique[0]")
        expectEqual(mapping[1], 0, "Index 1 (19%16=3) maps to unique[0]")
        expectEqual(mapping[2], 1, "Index 2 maps to unique[1]")
        expectEqual(mapping[3], 0, "Index 3 (3) maps to unique[0]")
        expectEqual(mapping[4], 0, "Index 4 (35%16=3) maps to unique[0]")
        expectEqual(mapping[5], 1, "Index 5 (7) maps to unique[1]")

    } catch {
        expect(false, "Query deduplication test failed: \(error)")
    }
}

// MARK: - Sibling Index

private func testSiblingIndex() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // XOR with 1: even <-> odd pairs
        expectEqual(engine.computeSiblingIndex(0), 1, "Sibling of 0 is 1")
        expectEqual(engine.computeSiblingIndex(1), 0, "Sibling of 1 is 0")
        expectEqual(engine.computeSiblingIndex(2), 3, "Sibling of 2 is 3")
        expectEqual(engine.computeSiblingIndex(3), 2, "Sibling of 3 is 2")
        expectEqual(engine.computeSiblingIndex(10), 11, "Sibling of 10 is 11")
        expectEqual(engine.computeSiblingIndex(11), 10, "Sibling of 11 is 10")

        // Double sibling = identity
        let idx = 42
        let sib = engine.computeSiblingIndex(idx)
        let sibSib = engine.computeSiblingIndex(sib)
        expectEqual(sibSib, idx, "Double sibling should return original index")

    } catch {
        expect(false, "Sibling index test failed: \(error)")
    }
}

// MARK: - Domain Points

private func testDomainPoints() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(logTraceLen: 2, blowupFactor: 2, numTraceColumns: 1)

        let points = engine.computeDomainPoints(config: config)
        let m = config.ldeDomainSize  // 4 * 2 = 8

        expectEqual(points.count, m, "Should have \(m) domain points")

        // First point should be cosetShift * omega^0 = cosetShift
        let logM = config.logLDEDomainSize
        let omegaM = frRootOfUnity(logN: logM)

        // Verify single-point computation matches
        for i in 0..<min(4, m) {
            let singlePt = engine.computeDomainPoint(
                index: i, omegaM: omegaM, cosetShift: config.cosetShift)
            expect(frEqual(points[i], singlePt),
                   "Domain point \(i) should match single computation")
        }

        // All points should be distinct (in a proper domain)
        var allDistinct = true
        for i in 0..<m {
            for j in (i+1)..<m {
                if frEqual(points[i], points[j]) {
                    allDistinct = false
                }
            }
        }
        expect(allDistinct, "All domain points should be distinct")

    } catch {
        expect(false, "Domain points test failed: \(error)")
    }
}

// MARK: - Invalid Input Rejection

private func testSTARKQueryInvalidInputRejection() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig()
        let m = config.ldeDomainSize

        // Wrong number of trace columns
        let seed = frFromInt(1)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)
        let wrongCols = [[Fr](repeating: Fr.zero, count: m)]  // 1 col, expected 2

        let emptyLeaves = [Fr](repeating: Fr.zero, count: m)
        let emptyNodes = [Fr](repeating: Fr.zero, count: 2 * m - 1)

        var threw = false
        do {
            _ = try engine.generateTraceDecommitments(
                querySet: querySet, ldeColumns: wrongCols,
                merkleLeaves: emptyLeaves, merkleNodes: emptyNodes,
                commitment: Fr.zero, config: config)
        } catch {
            threw = true
        }
        expect(threw, "Wrong column count should throw")

        // Wrong number of composition segments
        threw = false
        do {
            _ = try engine.generateCompositionDecommitments(
                querySet: querySet,
                compositionSegments: [],  // empty, expected 1
                merkleLeaves: emptyLeaves, merkleNodes: emptyNodes,
                commitment: Fr.zero, config: config)
        } catch {
            threw = true
        }
        expect(threw, "Wrong composition segment count should throw")

        // Wrong number of FRI layers
        threw = false
        do {
            _ = try engine.generateFRIDecommitments(
                querySet: querySet,
                friLayers: [],  // empty, expected numFRIRounds
                friMerkleData: [],
                config: config)
        } catch {
            threw = true
        }
        expect(threw, "Wrong FRI layer count should throw")

        // Mismatched Merkle leaves count
        threw = false
        do {
            let wrongLeaves = [Fr](repeating: Fr.zero, count: 3)  // wrong size
            _ = try engine.generateTraceDecommitments(
                querySet: querySet,
                ldeColumns: buildSimpleLDE(config: config),
                merkleLeaves: wrongLeaves, merkleNodes: emptyNodes,
                commitment: Fr.zero, config: config)
        } catch {
            threw = true
        }
        expect(threw, "Wrong Merkle leaves count should throw")

        // Mismatched deep eval inputs
        threw = false
        do {
            let oodFrame = OODEvaluationFrame(
                zeta: frFromInt(1),
                traceEvals: [Fr.zero, Fr.zero],
                traceNextEvals: [Fr.zero, Fr.zero],
                compositionEvals: [Fr.zero])
            _ = try engine.evaluateDeepAtQueries(
                querySet: querySet,
                traceDecommitments: [],  // empty, expected 4
                compositionDecommitments: [],
                oodFrame: oodFrame,
                alpha: Fr.one,
                config: config)
        } catch {
            threw = true
        }
        expect(threw, "Mismatched deep eval input should throw")

    } catch {
        expect(false, "Invalid input rejection test failed: \(error)")
    }
}

// MARK: - Consistency Verification

private func testConsistencyVerification() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()
        let config = makeConfig(numFRIRounds: 2)
        let ldeColumns = buildSimpleLDE(config: config)
        let compSegs = buildSimpleComposition(config: config)

        let (traceRoot, traceLeaves, traceNodes) = buildTree(
            engine: engine, ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)
        let (compRoot, compLeaves, compNodes) = engine.buildMerkleTree(leaves: compSegs[0])
        let (friLayers, friMerkle) = buildFRILayers(engine: engine, config: config)

        let zeta = frFromInt(7)
        let oodFrame = OODEvaluationFrame(
            zeta: zeta,
            traceEvals: [frFromInt(10), frFromInt(20)],
            traceNextEvals: [frFromInt(30), frFromInt(40)],
            compositionEvals: [frFromInt(50)])
        let alpha = frFromInt(5)
        let seed = frFromInt(29)
        let querySet = engine.sampleQueryIndices(seed: seed, config: config)

        let result = try engine.generateDecommitments(
            querySet: querySet,
            ldeColumns: ldeColumns,
            traceMerkleLeaves: traceLeaves,
            traceMerkleNodes: traceNodes,
            traceCommitment: traceRoot,
            compositionSegments: compSegs,
            compMerkleLeaves: compLeaves,
            compMerkleNodes: compNodes,
            compCommitment: compRoot,
            friLayers: friLayers,
            friMerkleData: friMerkle,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config)

        // Should pass with correct roots
        let valid = engine.verifyDecommitmentConsistency(
            result: result, traceCommitment: traceRoot,
            compCommitment: compRoot)
        expect(valid, "Consistency check should pass with correct roots")

        // Should fail with wrong trace root
        let invalidTrace = engine.verifyDecommitmentConsistency(
            result: result, traceCommitment: frFromInt(99999),
            compCommitment: compRoot)
        expect(!invalidTrace, "Consistency check should fail with wrong trace root")

        // Should fail with wrong composition root
        let invalidComp = engine.verifyDecommitmentConsistency(
            result: result, traceCommitment: traceRoot,
            compCommitment: frFromInt(88888))
        expect(!invalidComp, "Consistency check should fail with wrong composition root")

    } catch {
        expect(false, "Consistency verification test failed: \(error)")
    }
}

// MARK: - Build Trace Merkle Tree + Security Bits

private func testBuildTraceMerkleTreeAndSecurity() {
    do {
        let engine = try GPUSTARKQueryPhaseEngine()

        // Single column, 4 rows
        let col = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let (root, leaves, nodes) = engine.buildTraceMerkleTree(
            ldeColumns: [col], domainSize: 4)

        expectEqual(leaves.count, 4, "Should have 4 leaves")
        expectEqual(nodes.count, 7, "Should have 2*4-1 = 7 nodes")
        expect(!frEqual(root, Fr.zero), "Root should be non-zero")

        // Two columns should produce different tree
        let col2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let (root2, _, _) = engine.buildTraceMerkleTree(
            ldeColumns: [col, col2], domainSize: 4)
        expect(!frEqual(root, root2),
               "Different column sets should produce different roots")

        // Same data should produce same root
        let (root3, _, _) = engine.buildTraceMerkleTree(
            ldeColumns: [col], domainSize: 4)
        expect(frEqual(root, root3), "Same data should produce same root")

    } catch {
        expect(false, "Build trace Merkle tree test failed: \(error)")
    }

    // Security bit calculations
    let config1 = makeConfig(blowupFactor: 4, numQueries: 32)
    expectEqual(config1.securityBits, 64, "32 queries * log2(4)=2 = 64 bits")

    let config2 = makeConfig(blowupFactor: 8, numQueries: 16)
    expectEqual(config2.securityBits, 48, "16 queries * log2(8)=3 = 48 bits")

    let config = makeConfig(logTraceLen: 4, blowupFactor: 4)
    expectEqual(config.traceLen, 16, "2^4 = 16")
    expectEqual(config.ldeDomainSize, 64, "16 * 4 = 64")
    expectEqual(config.logLDEDomainSize, 6, "log2(64) = 6")
}
