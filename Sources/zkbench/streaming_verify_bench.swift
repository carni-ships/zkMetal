// Streaming Verification Benchmark
// Compares sequential vs pipelined FRI proof verification and exercises
// the unified-memory task-queue API (Merkle checks, EC on-curve checks).
// Measures speedup from CPU/GPU overlap on unified memory.

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

        // ========================================
        // Part 1: FRI Proof Verification Pipeline
        // ========================================
        print("--- FRI Proof Verification Pipeline ---\n")

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

            // Benchmark: Task-queue API (beginVerification/submit/finalize)
            let tqRuns = 3
            var tqTimes = [Double]()
            let merkleEngine = try Poseidon2MerkleEngine()
            let numLayers = commitment.layers.count - 1
            var layerRoots = [Fr]()
            for layer in 0..<numLayers {
                let root = try merkleEngine.merkleRoot(commitment.layers[layer])
                layerRoots.append(root)
            }

            for _ in 0..<tqRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                streamVerifier.beginVerification()

                // Submit Merkle checks via task-queue API
                for query in queries {
                    var idx = query.initialIndex
                    for layer in 0..<numLayers {
                        let nLayer = commitment.layers[layer].count
                        let halfN = UInt32(nLayer / 2)
                        if layer < query.merklePaths.count && !query.merklePaths[layer].isEmpty {
                            let leaf = commitment.layers[layer][Int(idx)]
                            let path = query.merklePaths[layer][0]
                            streamVerifier.submitMerkleCheck(
                                root: layerRoots[layer], leaf: leaf,
                                path: path, index: idx)
                        }
                        idx = idx < halfN ? idx : idx - halfN
                    }
                }

                let ok = try streamVerifier.finalize()
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                tqTimes.append(elapsed)
                if !ok {
                    print("  WARNING: Task-queue verification FAILED")
                }
            }
            tqTimes.sort()
            let tqMedian = tqTimes[tqRuns / 2]

            // Report
            let speedup1 = seqMedian > 0 ? seqMedian / pipeMedian : 0
            let speedup2 = seqMedian > 0 ? seqMedian / dbMedian : 0
            let speedup3 = seqMedian > 0 ? seqMedian / tqMedian : 0
            print(String(format: "  Sequential:       %7.2f ms", seqMedian))
            print(String(format: "  Pipelined:        %7.2f ms (%.2f\u{00d7})", pipeMedian, speedup1))
            print(String(format: "  Double-buffered:  %7.2f ms (%.2f\u{00d7})", dbMedian, speedup2))
            print(String(format: "  Task-queue:       %7.2f ms (%.2f\u{00d7})", tqMedian, speedup3))
            print()
        }

        // ============================================
        // Part 2: GPU EC On-Curve Batch Verification
        // ============================================
        print("--- GPU EC On-Curve Batch Verification ---\n")

        let ecSizes = CommandLine.arguments.contains("--quick") ? [100, 1000] : [100, 1000, 10_000, 100_000]

        for count in ecSizes {
            // Generate random valid BN254 points by scalar multiplication
            var rng2: UInt64 = 0xCAFE_BABE_1234 &+ UInt64(count)
            var points = [PointAffine]()
            points.reserveCapacity(count)

            // Generate points: use generator * random_scalar to get valid curve points
            let g = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
            let gProj = pointFromAffine(g)
            for _ in 0..<count {
                rng2 = rng2 &* 6364136223846793005 &+ 1442695040888963407
                let scalar = frFromInt(rng2 >> 32)
                let pt = cPointScalarMul(gProj, scalar)
                let ptAff = batchToAffine([pt])
                points.append(ptAff[0])
            }

            // Benchmark: CPU on-curve check (baseline)
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            var cpuOk = true
            for pt in points {
                if !StreamingVerifier.cpuCheckOnCurve(point: pt) {
                    cpuOk = false
                }
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // Benchmark: GPU batch on-curve check
            // Warmup
            let _ = try streamVerifier.batchCheckOnCurve(points: Array(points.prefix(min(10, count))))

            let gpuRuns = 3
            var gpuTimes = [Double]()
            var gpuOk = true
            for _ in 0..<gpuRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                let results = try streamVerifier.batchCheckOnCurve(points: points)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                gpuTimes.append(elapsed)
                if results.contains(false) { gpuOk = false }
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[gpuRuns / 2]

            // Benchmark: Task-queue API for EC checks
            let tqRuns2 = 3
            var tqTimes2 = [Double]()
            for _ in 0..<tqRuns2 {
                let t0 = CFAbsoluteTimeGetCurrent()
                streamVerifier.beginVerification()
                for pt in points {
                    streamVerifier.submitPointCheck(point: pt, expectedOnCurve: true)
                }
                let ok = try streamVerifier.finalize()
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                tqTimes2.append(elapsed)
                if !ok {
                    print("  WARNING: Task-queue EC check FAILED")
                }
            }
            tqTimes2.sort()
            let tqMedian2 = tqTimes2[tqRuns2 / 2]

            let speedup = cpuTime / max(gpuMedian, 0.001)
            print(String(format: "  N=%-6d  CPU: %7.2f ms  GPU: %7.2f ms  TQ: %7.2f ms  (%.1f\u{00d7})",
                         count, cpuTime, gpuMedian, tqMedian2, speedup))
        }

        // ============================================
        // Part 3: Correctness Checks
        // ============================================
        print("\n--- Correctness Verification ---")

        // FRI correctness
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

        // Task-queue Merkle correctness
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

        // EC on-curve correctness
        let g = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let gProj = pointFromAffine(g)
        let validPt = cPointScalarMul(gProj, frFromInt(42))
        let validAff = batchToAffine([validPt])

        let ecResults = try streamVerifier.batchCheckOnCurve(points: validAff)
        print("  EC on-curve:     \(ecResults[0] ? "PASS" : "FAIL")")

        // Test CPU vs GPU consistency for on-curve
        let cpuResult = StreamingVerifier.cpuCheckOnCurve(point: validAff[0])
        print("  CPU on-curve:    \(cpuResult ? "PASS" : "FAIL")")

        // Task-queue API correctness: mixed Merkle + EC checks
        streamVerifier.beginVerification()
        streamVerifier.submitMerkleCheck(root: root, leaf: testLeaves[3],
                                          path: path, index: 3)
        streamVerifier.submitPointCheck(point: validAff[0], expectedOnCurve: true)
        let mixedOk = try streamVerifier.finalize()
        print("  Mixed task-queue: \(mixedOk ? "PASS" : "FAIL")")

        // Task-queue with invalid point should fail
        let badPoint = PointAffine(x: fpFromInt(99), y: fpFromInt(99))
        streamVerifier.beginVerification()
        streamVerifier.submitPointCheck(point: badPoint, expectedOnCurve: true)
        let badOk = try streamVerifier.finalize()
        print("  Bad point reject: \(!badOk ? "PASS (correctly rejected)" : "FAIL")")

    } catch {
        print("ERROR: \(error)")
    }
}
