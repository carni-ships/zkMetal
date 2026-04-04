// Incremental Merkle Tree Benchmark
import zkMetal
import Foundation

public func runIncrementalMerkleBench() {
    print("=== Incremental Merkle Tree Benchmark ===")
    print("  (Poseidon2 BN254 Fr, unified memory, GPU hashing)\n")

    do {
        // --- 0. Quick smoke test ---
        print("--- Smoke Test ---")
        do {
            let tree = try IncrementalMerkleTree(depth: 4)
            for i in 0..<8 {
                try tree.append(leaf: frFromInt(UInt64(i + 1)))
            }
            let fullRoot = try tree.fullRebuildRoot()
            let match = frEqual(tree.root, fullRoot)
            print("  Depth-4, 8 leaves: root match = \(match ? "PASS" : "FAIL")")

            // Batch append
            let tree2 = try IncrementalMerkleTree(depth: 4)
            var leaves4 = [Fr]()
            for i in 0..<8 { leaves4.append(frFromInt(UInt64(i + 1))) }
            try tree2.appendBatchFused(leaves: leaves4)
            let match2 = frEqual(tree2.root, fullRoot)
            print("  Depth-4, batch 8: root match = \(match2 ? "PASS" : "FAIL")")

            // Single append matches batch append
            let match3 = frEqual(tree.root, tree2.root)
            print("  Single vs batch root match = \(match3 ? "PASS" : "FAIL")")

            // Proof verification using GPU-consistent method
            let prf = tree.proof(index: 3)
            let leaf3 = frFromInt(UInt64(4))
            let proofOk = try tree.verifyProof(leaf: leaf3, proof: prf)
            print("  Proof for leaf 3: \(proofOk ? "PASS" : "FAIL")")
        }

        // --- 1. Single append to existing tree ---
        print("\n--- Single Append ---")
        for logN in [16, 20] {
            let depth = logN
            let prefill = (1 << logN) / 2  // half full
            let tree = try IncrementalMerkleTree(depth: depth)

            // Prefill with batch
            var leaves = [Fr](repeating: Fr.zero, count: prefill)
            for i in 0..<prefill { leaves[i] = frFromInt(UInt64(i + 1)) }
            try tree.appendBatchFused(leaves: leaves)

            let newLeaf = frFromInt(UInt64(prefill + 1))

            // Warmup
            try tree.append(leaf: newLeaf)

            // Benchmark single append
            var times = [Double]()
            for trial in 0..<20 {
                let t0 = CFAbsoluteTimeGetCurrent()
                try tree.append(leaf: frFromInt(UInt64(prefill + 2 + trial)))
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1_000_000)  // microseconds
            }
            times.sort()
            let median = times[times.count / 2]
            print(String(format: "  Append 1 leaf (tree 2^%d, %d filled): %.1f us (GPU path, %d hashes)",
                        logN, prefill, median, depth))
        }

        // --- 2. Batch append ---
        print("\n--- Batch Append ---")
        for logN in [16, 20] {
            let depth = logN
            for batchSize in [256, 1024, 4096] {
                if batchSize > (1 << logN) / 2 { continue }

                let tree = try IncrementalMerkleTree(depth: depth)
                // Prefill half
                let prefill = (1 << logN) / 2
                var prefillLeaves = [Fr](repeating: Fr.zero, count: prefill)
                for i in 0..<prefill { prefillLeaves[i] = frFromInt(UInt64(i + 1)) }
                try tree.appendBatchFused(leaves: prefillLeaves)

                // Prepare batch
                var batch = [Fr](repeating: Fr.zero, count: batchSize)
                for i in 0..<batchSize { batch[i] = frFromInt(UInt64(prefill + i + 1)) }

                // Warmup
                let warmTree = try IncrementalMerkleTree(depth: depth)
                try warmTree.appendBatchFused(leaves: prefillLeaves)
                try warmTree.appendBatchFused(leaves: batch)

                // Benchmark: incremental batch append
                var incTimes = [Double]()
                for _ in 0..<5 {
                    let t = try IncrementalMerkleTree(depth: depth)
                    try t.appendBatchFused(leaves: prefillLeaves)
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try t.appendBatchFused(leaves: batch)
                    incTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                incTimes.sort()
                let incMedian = incTimes[2]

                // Full rebuild for comparison
                var fullLeaves = prefillLeaves
                fullLeaves.append(contentsOf: batch)
                while fullLeaves.count < (1 << depth) { fullLeaves.append(Fr.zero) }

                let merkle = try Poseidon2MerkleEngine()
                let _ = try merkle.merkleRoot(fullLeaves)  // warmup
                var fullTimes = [Double]()
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try merkle.merkleRoot(fullLeaves)
                    fullTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                fullTimes.sort()
                let fullMedian = fullTimes[2]

                let speedup = fullMedian / incMedian
                print(String(format: "  Batch %4d to tree 2^%d: inc %.2f ms | full %.2f ms | %.1fx",
                            batchSize, logN, incMedian, fullMedian, speedup))
            }
        }

        // --- 3. Random updates ---
        print("\n--- Random Updates ---")
        for updateCount in [1, 10, 100] {
            let depth = 16
            let tree = try IncrementalMerkleTree(depth: depth)
            let n = 1 << depth

            // Fill tree
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }
            try tree.appendBatchFused(leaves: leaves)

            // Generate random update indices
            var updates = [(index: Int, leaf: Fr)]()
            for i in 0..<updateCount {
                let idx = (i * 7919 + 13) % n
                updates.append((index: idx, leaf: frFromInt(UInt64(n + i + 1))))
            }

            // Warmup
            try tree.batchUpdate(updates: updates)

            // Benchmark incremental
            var incTimes = [Double]()
            for trial in 0..<10 {
                var trialUpdates = [(index: Int, leaf: Fr)]()
                for i in 0..<updateCount {
                    let idx = (i * 7919 + 13 + trial * 101) % n
                    trialUpdates.append((index: idx, leaf: frFromInt(UInt64(n + updateCount + trial * updateCount + i + 1))))
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try tree.batchUpdate(updates: trialUpdates)
                incTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            incTimes.sort()
            let incMedian = incTimes[incTimes.count / 2]

            // Full rebuild for comparison
            let merkle = try Poseidon2MerkleEngine()
            let _ = try merkle.merkleRoot(leaves)  // warmup
            var fullTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try merkle.merkleRoot(leaves)
                fullTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            fullTimes.sort()
            let fullMedian = fullTimes[2]

            let speedup = fullMedian / incMedian
            print(String(format: "  Update %3d leaves in 2^16: inc %.2f ms | full %.2f ms | %.1fx",
                        updateCount, incMedian, fullMedian, speedup))
        }

        // --- 4. Sequential vs batch build (correctness + timing) ---
        print("\n--- Sequential vs Batch Build ---")
        for logN in [10, 12] {
            let depth = logN
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Sequential (one at a time)
            let seqTree = try IncrementalMerkleTree(depth: depth)
            let t0 = CFAbsoluteTimeGetCurrent()
            for i in 0..<n {
                try seqTree.append(leaf: leaves[i])
            }
            let seqMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Batch (single call)
            let batchTree = try IncrementalMerkleTree(depth: depth)
            let t1 = CFAbsoluteTimeGetCurrent()
            try batchTree.appendBatchFused(leaves: leaves)
            let batchMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            // Full rebuild
            let merkle = try Poseidon2MerkleEngine()
            let fullRoot = try merkle.merkleRoot(leaves)
            let seqMatch = frEqual(seqTree.root, fullRoot)
            let batchMatch = frEqual(batchTree.root, fullRoot)

            print(String(format: "  2^%d: seq %.0f ms | batch %.1f ms | seq=%@ batch=%@",
                        logN, seqMs, batchMs, seqMatch ? "OK" : "FAIL", batchMatch ? "OK" : "FAIL"))
        }

        // --- 5. Correctness verification ---
        print("\n--- Correctness ---")
        let depth = 14
        let tree = try IncrementalMerkleTree(depth: depth)
        let n = 1 << depth

        // Append leaves in batches
        let batchSize = 1024
        var allLeaves = [Fr]()
        for batch in 0..<(n / batchSize) {
            var batchLeaves = [Fr](repeating: Fr.zero, count: batchSize)
            for i in 0..<batchSize {
                batchLeaves[i] = frFromInt(UInt64(batch * batchSize + i + 1))
            }
            try tree.appendBatchFused(leaves: batchLeaves)
            allLeaves.append(contentsOf: batchLeaves)
        }

        // Verify root matches full rebuild
        let fullRoot = try tree.fullRebuildRoot()
        let rootMatch = frEqual(tree.root, fullRoot)
        print("  Root matches full rebuild: \(rootMatch ? "PASS" : "FAIL")")

        // Verify internal node consistency using GPU hashing
        // Check that each node equals GPU hash of its children
        let p2e = try Poseidon2Engine()
        var totalBad = 0
        for level in 0..<depth {
            let lo = n >> (level + 1)
            let hi = n >> level
            let count = hi - lo
            if count == 0 { continue }
            // Build pairs for this level: children of nodes [lo, hi)
            var pairs = [Fr]()
            pairs.reserveCapacity(count * 2)
            let bufPtr = tree.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 2 * n)
            for k in lo..<hi {
                pairs.append(bufPtr[k * 2])
                pairs.append(bufPtr[k * 2 + 1])
            }
            let hashes = try p2e.hashPairs(pairs)
            var badAtLevel = 0
            for (i, k) in (lo..<hi).enumerated() {
                if !frEqual(hashes[i], bufPtr[k]) {
                    badAtLevel += 1
                }
            }
            totalBad += badAtLevel
        }
        print("  Internal node consistency (GPU): \(totalBad == 0 ? "PASS" : "FAIL (\(totalBad) bad)")")

        // Verify proof for a few leaves using GPU-consistent verification
        var proofOk = true
        for idx in [0, n/2, n-1, 42, n/4] {
            let prf = tree.proof(index: idx)
            let leaf = allLeaves[idx]
            if !(try tree.verifyProof(leaf: leaf, proof: prf)) {
                print("  Proof verification FAILED for index \(idx)")
                proofOk = false
            }
        }
        print("  Merkle proofs (GPU verify): \(proofOk ? "PASS" : "FAIL")")

        // Verify after random updates
        var updatedLeaves = allLeaves
        var updates = [(index: Int, leaf: Fr)]()
        for i in 0..<50 {
            let idx = (i * 311 + 7) % n
            let newVal = frFromInt(UInt64(n + i + 1))
            updates.append((index: idx, leaf: newVal))
            updatedLeaves[idx] = newVal
        }
        try tree.batchUpdate(updates: updates)

        // Rebuild from updated leaves
        let merkle = try Poseidon2MerkleEngine()
        let expectedRoot = try merkle.merkleRoot(updatedLeaves)
        let updateMatch = frEqual(tree.root, expectedRoot)
        print("  Root after 50 updates matches rebuild: \(updateMatch ? "PASS" : "FAIL")")

        // Verify proof after updates using GPU
        var postUpdateProofOk = true
        for (idx, leaf) in updates.prefix(5) {
            let prf = tree.proof(index: idx)
            if !(try tree.verifyProof(leaf: leaf, proof: prf)) {
                print("  Post-update proof FAILED for index \(idx)")
                postUpdateProofOk = false
            }
        }
        print("  Post-update proofs (GPU verify): \(postUpdateProofOk ? "PASS" : "FAIL")")

    } catch {
        print("ERROR: \(error)")
    }
}
