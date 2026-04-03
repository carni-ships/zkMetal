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

            let t0 = CFAbsoluteTimeGetCurrent()
            let (levels, _) = try engine.buildTree(leaves: leaves)
            let buildTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            fputs(String(format: "  Build tree (%d leaves, %d levels): %.1f ms\n",
                  numLeaves, levels.count, buildTime), stderr)

            let t1 = CFAbsoluteTimeGetCurrent()
            let proof = try engine.createPathProof(leaves: leaves, leafIndex: 0)
            let proveTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
            fputs(String(format: "  Path proof (%d openings): %.1f ms\n",
                  proof.openings.count, proveTime), stderr)

            let root = levels.last![0]
            let t2 = CFAbsoluteTimeGetCurrent()
            let valid = engine.verifyPathProof(proof, root: root)
            let verifyTime = (CFAbsoluteTimeGetCurrent() - t2) * 1000
            fputs("  Verify path: \(String(format: "%.1f", verifyTime)) ms — \(valid ? "PASS" : "FAIL")\n", stderr)

        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }
}
