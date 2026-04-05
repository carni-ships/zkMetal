// Verkle Tree Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runVerkleBench() {
    fputs("\n=== Verkle Tree (IPA-based) ===\n", stderr)

    // Use small width for fast testing
    let testWidth = 4  // 4-ary tree for quick tests

    fputs("\n--- Correctness Tests (width=\(testWidth)) ---\n", stderr)

    do {
        let engine = try VerkleEngine(width: testWidth)

        // Test basic commitment
        let values = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let C = try engine.commit(values)
        let notIdentity = !pointIsIdentity(C)
        fputs("  Commit non-trivial: \(notIdentity ? "PASS" : "FAIL")\n", stderr)

        // Test opening proof at each index
        var allOpeningsPass = true
        for idx in 0..<testWidth {
            let proof = try engine.createOpeningProof(values: values, index: idx)
            let valid = engine.verifyOpeningProof(proof)
            if !valid {
                fputs("  Opening at index \(idx): FAIL\n", stderr)
                allOpeningsPass = false
            }
        }
        fputs("  All single openings verify: \(allOpeningsPass ? "PASS" : "FAIL")\n", stderr)

        // Test tree building (2 levels: 2 leaf chunks → 1 root)
        let numLeaves = testWidth * 2  // 8 leaves for 4-ary → 2 leaf nodes → 1 root
        var leaves = [Fr]()
        for i in 0..<numLeaves {
            leaves.append(frFromInt(UInt64(i + 1)))
        }

        let (levels, _) = try engine.buildTree(leaves: leaves)
        fputs("  Tree levels: \(levels.count) (expected: 2)\n", stderr)
        fputs("  Level 0: \(levels[0].count) nodes, Level 1: \(levels[1].count) node\n", stderr)

        // Test path proof
        let pathProof = try engine.createPathProof(leaves: leaves, leafIndex: 0)
        let root = levels.last![0]
        let pathValid = engine.verifyPathProof(pathProof, root: root)
        fputs("  Path proof (leaf 0 → root): \(pathValid ? "PASS" : "FAIL")\n", stderr)

        // Test path proof for different leaf
        let pathProof2 = try engine.createPathProof(leaves: leaves, leafIndex: numLeaves - 1)
        let pathValid2 = engine.verifyPathProof(pathProof2, root: root)
        fputs("  Path proof (leaf \(numLeaves-1) → root): \(pathValid2 ? "PASS" : "FAIL")\n", stderr)

        // Test rejection: wrong root
        let wrongRoot = pointDouble(root)
        let rejectedRoot = !engine.verifyPathProof(pathProof, root: wrongRoot)
        fputs("  Reject wrong root: \(rejectedRoot ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // Performance test with larger width
    if !skipCPU {
        fputs("\n--- Performance (width=16) ---\n", stderr)
        do {
            let w = 16
            let engine = try VerkleEngine(width: w)

            let numLeaves = w * w  // 256 leaves → 16 leaf nodes → 1 root
            var leaves = [Fr]()
            for i in 0..<numLeaves {
                leaves.append(frFromInt(UInt64(i + 1)))
            }

            // Warmup run
            let (wLevels, wChunks) = try engine.buildTree(leaves: leaves)
            let wProof = try engine.createPathProof(levels: wLevels, leafChunks: wChunks, leafIndex: 0)
            _ = engine.verifyPathProof(wProof, root: wLevels.last![0])

            // Benchmark build (median of 5)
            let iters = 5
            var buildTimes = [Double]()
            var lastLevels = wLevels
            var lastChunks = wChunks
            for _ in 0..<iters {
                let t0 = CFAbsoluteTimeGetCurrent()
                let (levels, leafChunks) = try engine.buildTree(leaves: leaves)
                buildTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                lastLevels = levels
                lastChunks = leafChunks
            }
            buildTimes.sort()
            fputs(String(format: "  Build tree (%d leaves, %d levels): %.1f ms (median of %d)\n",
                  numLeaves, lastLevels.count, buildTimes[iters/2], iters), stderr)

            // Benchmark path proof (median of 5, reusing pre-built tree)
            var proveTimes = [Double]()
            var lastProof = wProof
            for _ in 0..<iters {
                let t1 = CFAbsoluteTimeGetCurrent()
                let proof = try engine.createPathProof(levels: lastLevels, leafChunks: lastChunks, leafIndex: 0)
                proveTimes.append((CFAbsoluteTimeGetCurrent() - t1) * 1000)
                lastProof = proof
            }
            proveTimes.sort()
            fputs(String(format: "  Path proof (%d openings): %.1f ms (median of %d)\n",
                  lastProof.openings.count, proveTimes[iters/2], iters), stderr)

            // Benchmark verify (median of 5)
            let root = lastLevels.last![0]
            var verifyTimes = [Double]()
            var lastValid = false
            for _ in 0..<iters {
                let t2 = CFAbsoluteTimeGetCurrent()
                let valid = engine.verifyPathProof(lastProof, root: root)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - t2) * 1000)
                lastValid = valid
            }
            verifyTimes.sort()
            fputs("  Verify path: \(String(format: "%.1f", verifyTimes[iters/2])) ms (median of \(iters)) — \(lastValid ? "PASS" : "FAIL")\n", stderr)

        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }
}
