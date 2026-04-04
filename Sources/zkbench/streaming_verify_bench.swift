// Streaming Verification Benchmark
// Compares sequential vs pipelined FRI proof verification
// to measure speedup from CPU/GPU overlap on unified memory.

import Foundation
import Metal
import zkMetal

public func runStreamingVerifyBench() {
    print("=== Streaming Verification Benchmark ===")
    print("  Apple Silicon unified memory: zero-copy CPU<->GPU verification\n")

    do {
        let friEngine = try FRIEngine()
        let streamVerifier = try StreamingVerifier()
        let pipelinedVerifier = try PipelinedFRIVerifier()

        // Test sizes from 2^10 to 2^18
        let logSizes = CommandLine.arguments.contains("--quick") ? [10, 14] : [10, 12, 14, 16, 18]

        for logN in logSizes {
            let n = 1 << logN
            print("--- 2^\(logN) = \(n) evaluations ---")

            // Generate random evaluations
            var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE &+ UInt64(logN)
            var evals = [Fr]()
            evals.reserveCapacity(n)
            for _ in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals.append(frFromInt(rng >> 32))
            }

            // Generate FRI challenges
            let numFolds = min(logN, logN)  // fold all the way down
            var betas = [Fr]()
            for i in 0..<numFolds {
                betas.append(frFromInt(UInt64(i + 1) * 13 + 7))
            }

            // Commit phase
            let commitT0 = CFAbsoluteTimeGetCurrent()
            let commitment = try friEngine.commitPhase(evals: evals, betas: betas)
            let commitTime = (CFAbsoluteTimeGetCurrent() - commitT0) * 1000
            print("  Commit: \(String(format: "%.1f", commitTime)) ms")

            // Query phase
            let numQueries = min(16, n / 4)
            var queryIndices = [UInt32]()
            for i in 0..<numQueries {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                queryIndices.append(UInt32(rng >> 32) % UInt32(n))
            }

            let queryT0 = CFAbsoluteTimeGetCurrent()
            let queries = try friEngine.queryPhase(commitment: commitment, queryIndices: queryIndices)
            let queryTime = (CFAbsoluteTimeGetCurrent() - queryT0) * 1000
            print("  Query:  \(String(format: "%.1f", queryTime)) ms (\(numQueries) queries)")

            // Benchmark: Sequential verification (CPU-only, baseline)
            let seqRuns = 3
            var seqTimes = [Double]()
            for _ in 0..<seqRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                let ok = streamVerifier.verifyFRISequential(commitment: commitment, queries: queries)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                seqTimes.append(elapsed)
                if !ok {
                    print("  WARNING: Sequential verification FAILED")
                }
            }
            seqTimes.sort()
            let seqMedian = seqTimes[seqRuns / 2]

            // Benchmark: Streaming pipelined verification (GPU Merkle + CPU fold overlap)
            let pipeRuns = 3
            var pipeTimes = [Double]()
            for _ in 0..<pipeRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                let ok = try streamVerifier.verifyFRIProofPipelined(
                    commitment: commitment, queries: queries)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                pipeTimes.append(elapsed)
                if !ok {
                    print("  WARNING: Pipelined verification FAILED")
                }
            }
            pipeTimes.sort()
            let pipeMedian = pipeTimes[pipeRuns / 2]

            // Benchmark: Double-buffered pipelined verification
            let dbRuns = 3
            var dbTimes = [Double]()
            for _ in 0..<dbRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                let ok = try pipelinedVerifier.verify(
                    commitment: commitment, queries: queries)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                dbTimes.append(elapsed)
                if !ok {
                    print("  WARNING: Double-buffered verification FAILED")
                }
            }
            dbTimes.sort()
            let dbMedian = dbTimes[dbRuns / 2]

            // Benchmark: GPU batch Merkle verification only
            let merkleVerifier = streamVerifier.merkleVerifier
            var maxDepth = 0
            var totalPaths = 0
            for query in queries {
                for layerPath in query.merklePaths {
                    for siblings in layerPath {
                        maxDepth = max(maxDepth, siblings.count)
                    }
                    totalPaths += layerPath.count
                }
            }

            let merkleRuns = 3
            var merkleTimes = [Double]()
            if totalPaths > 0 && maxDepth > 0 {
                // Build layer trees
                let merkleEngine = try Poseidon2MerkleEngine()
                let numLayers = commitment.layers.count - 1
                var layerRoots = [Fr]()
                for layer in 0..<numLayers {
                    let root = try merkleEngine.merkleRoot(commitment.layers[layer])
                    layerRoots.append(root)
                }

                // Collect all paths
                var allLeaves = [Fr]()
                var allIndices = [UInt32]()
                var allPaths = [[Fr]]()
                var allRoots = [Fr]()

                for query in queries {
                    var idx = query.initialIndex
                    for layer in 0..<numLayers {
                        let nLayer = commitment.layers[layer].count
                        let halfN = UInt32(nLayer / 2)
                        let leaf = commitment.layers[layer][Int(idx)]
                        if layer < query.merklePaths.count && !query.merklePaths[layer].isEmpty {
                            let path = query.merklePaths[layer][0]
                            allLeaves.append(leaf)
                            allIndices.append(idx)
                            allPaths.append(path)
                            allRoots.append(layerRoots[layer])
                        }
                        idx = idx < halfN ? idx : idx - halfN
                    }
                }

                for _ in 0..<merkleRuns {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try merkleVerifier.batchVerify(
                        leaves: allLeaves, indices: allIndices,
                        paths: allPaths, roots: allRoots, maxDepth: maxDepth)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    merkleTimes.append(elapsed)
                }
                merkleTimes.sort()
            }

            let merkleMedian = merkleTimes.isEmpty ? 0.0 : merkleTimes[merkleRuns / 2]

            // Report
            let speedup1 = seqMedian > 0 ? seqMedian / pipeMedian : 0
            let speedup2 = seqMedian > 0 ? seqMedian / dbMedian : 0
            print(String(format: "  Sequential:       %7.2f ms", seqMedian))
            print(String(format: "  Pipelined:        %7.2f ms (%.2f\u{00d7})", pipeMedian, speedup1))
            print(String(format: "  Double-buffered:  %7.2f ms (%.2f\u{00d7})", dbMedian, speedup2))
            if merkleMedian > 0 {
                print(String(format: "  GPU Merkle batch: %7.2f ms (%d paths)", merkleMedian, totalPaths))
            }
            print()
        }

        // Correctness check
        print("--- Correctness Verification ---")
        let testLogN = 12
        let testN = 1 << testLogN
        var rng2: UInt64 = 0xCAFE_BABE
        var testEvals = [Fr]()
        testEvals.reserveCapacity(testN)
        for _ in 0..<testN {
            rng2 = rng2 &* 6364136223846793005 &+ 1442695040888963407
            testEvals.append(frFromInt(rng2 >> 32))
        }
        var testBetas = [Fr]()
        for i in 0..<testLogN {
            testBetas.append(frFromInt(UInt64(i + 1) * 11))
        }

        let testCommitment = try friEngine.commitPhase(evals: testEvals, betas: testBetas)
        var testQueryIndices = [UInt32]()
        for i in 0..<8 {
            rng2 = rng2 &* 6364136223846793005 &+ 1442695040888963407
            testQueryIndices.append(UInt32(rng2 >> 32) % UInt32(testN))
        }
        let testQueries = try friEngine.queryPhase(commitment: testCommitment, queryIndices: testQueryIndices)

        let seqOk = streamVerifier.verifyFRISequential(commitment: testCommitment, queries: testQueries)
        let pipeOk = try streamVerifier.verifyFRIProofPipelined(commitment: testCommitment, queries: testQueries)
        let dbOk = try pipelinedVerifier.verify(commitment: testCommitment, queries: testQueries)

        print("  Sequential:      \(seqOk ? "PASS" : "FAIL")")
        print("  Pipelined:       \(pipeOk ? "PASS" : "FAIL")")
        print("  Double-buffered: \(dbOk ? "PASS" : "FAIL")")

        // Test GPU batch Merkle correctness
        let batchMerkle = try PipelinedMerkleVerifier()
        let merkleEngine = try Poseidon2MerkleEngine()

        let testLeaves: [Fr] = (0..<8).map { frFromInt(UInt64($0) + 1) }
        let tree = try merkleEngine.buildTree(testLeaves)
        let root = tree.last!

        // Extract path for leaf 3
        var path = [Fr]()
        var treeIdx = 3
        var levelStart = 0
        var levelSize = 8
        while levelSize > 1 {
            let sibIdx = treeIdx ^ 1
            path.append(tree[levelStart + sibIdx])
            treeIdx /= 2
            levelStart += levelSize
            levelSize /= 2
        }

        let batchResult = try batchMerkle.batchVerify(
            leaves: [testLeaves[3]], indices: [3], paths: [path],
            roots: [root], maxDepth: path.count)
        print("  Batch Merkle:    \(batchResult[0] ? "PASS" : "FAIL")")

        // Test with wrong root
        let wrongRoot = frFromInt(999999)
        let wrongResult = try batchMerkle.batchVerify(
            leaves: [testLeaves[3]], indices: [3], paths: [path],
            roots: [wrongRoot], maxDepth: path.count)
        print("  Wrong root:      \(!wrongResult[0] ? "PASS (correctly rejected)" : "FAIL")")

    } catch {
        print("ERROR: \(error)")
    }
}
