// Incremental Merkle Tree Benchmark
import zkMetal
import Foundation

public func runIncrementalMerkleBench() {
    print("=== Incremental Merkle Tree Benchmark ===")
    print("  (Poseidon2 BN254 Fr, unified memory)\n")

    do {
        // --- Quick correctness debug ---
        print("--- Quick Debug ---")
        do {
            // Build a small tree (depth 4) using CPU append and verify all nodes
            let t = try IncrementalMerkleTree(depth: 4)  // capacity 16
            for i in 0..<16 {
                try t.append(leaf: frFromInt(UInt64(i + 1)))
            }
            let bufPtr = t.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 32)
            var bad4 = 0
            for k in stride(from: 15, through: 1, by: -1) {
                let left = bufPtr[k * 2]
                let right = bufPtr[k * 2 + 1]
                let expected = poseidon2Hash(left, right)
                let actual = bufPtr[k]
                if !frEqual(actual, expected) {
                    print("  Depth-4 BAD node[\(k)]")
                    bad4 += 1
                }
            }
            print("  Depth-4 full tree (append 1-by-1): \(bad4) bad nodes")

            // Now depth 4 using batch append
            let t2 = try IncrementalMerkleTree(depth: 4)
            var leaves4 = [Fr]()
            for i in 0..<16 { leaves4.append(frFromInt(UInt64(i + 1))) }
            try t2.appendBatch(leaves: leaves4)
            let bufPtr2 = t2.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 32)
            var bad4b = 0
            for k in stride(from: 15, through: 1, by: -1) {
                let left = bufPtr2[k * 2]
                let right = bufPtr2[k * 2 + 1]
                let expected = poseidon2Hash(left, right)
                let actual = bufPtr2[k]
                if !frEqual(actual, expected) {
                    print("  Depth-4 batch BAD node[\(k)]")
                    bad4b += 1
                }
            }
            print("  Depth-4 batch 16 leaves: \(bad4b) bad nodes")

            // Depth 4 using 2 batches of 8
            let t3 = try IncrementalMerkleTree(depth: 4)
            var b1 = [Fr]()
            var b2 = [Fr]()
            for i in 0..<8 { b1.append(frFromInt(UInt64(i + 1))) }
            for i in 8..<16 { b2.append(frFromInt(UInt64(i + 1))) }
            try t3.appendBatch(leaves: b1)
            try t3.appendBatch(leaves: b2)
            let bufPtr3 = t3.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 32)
            var bad4c = 0
            for k in stride(from: 15, through: 1, by: -1) {
                let left = bufPtr3[k * 2]
                let right = bufPtr3[k * 2 + 1]
                let expected = poseidon2Hash(left, right)
                let actual = bufPtr3[k]
                if !frEqual(actual, expected) {
                    print("  Depth-4 2-batch BAD node[\(k)]")
                    bad4c += 1
                }
            }
            print("  Depth-4 two batches of 8: \(bad4c) bad nodes")

            // Compare CPU vs GPU hash of same inputs
            let a = frFromInt(1)
            let b = frFromInt(2)
            let cpuHash = poseidon2Hash(a, b)
            // GPU hash via Poseidon2MerkleEngine
            let gpuLeaves = [a, b]
            let gpuMerkle = try Poseidon2MerkleEngine()
            let gpuRoot = try gpuMerkle.merkleRoot(gpuLeaves)
            print("  CPU hash(1,2) == GPU merkle_root([1,2]): \(frEqual(cpuHash, gpuRoot) ? "MATCH" : "MISMATCH")")

            // Test with specific leaves that fail at depth 14
            let l34 = frFromInt(35)
            let l35 = frFromInt(36)
            let cpuH = poseidon2Hash(l34, l35)
            // GPU hash via merkleRoot (2 leaves -> root is hash(l, r))
            let gpuRoot2 = try gpuMerkle.merkleRoot([l34, l35])
            let match35 = frEqual(cpuH, gpuRoot2)
            print("  CPU hash(35,36) == GPU merkle_root([35,36]): \(match35 ? "MATCH" : "MISMATCH")")
            if !match35 {
                let cl = frToInt(cpuH)
                let gl = frToInt(gpuRoot2)
                print("    CPU: \(cl.map { String(format: "%016llx", $0) }.joined(separator: " "))")
                print("    GPU: \(gl.map { String(format: "%016llx", $0) }.joined(separator: " "))")
                // Also check that frFromInt(35) is the same on both sides
                let l34limbs = l34.to64()
                print("    frFromInt(35) limbs: \(l34limbs.map { String(format: "%016llx", $0) }.joined(separator: " "))")
            }

            // Find the SMALLEST input that causes a mismatch
            var firstBad: UInt64 = 0
            for testVal: UInt64 in 1...100 {
                let la = frFromInt(testVal)
                let lb = frFromInt(testVal + 1)
                let ch = poseidon2Hash(la, lb)
                let gh = try gpuMerkle.merkleRoot([la, lb])
                if !frEqual(ch, gh) {
                    firstBad = testVal
                    break
                }
            }
            print("  First mismatching pair: hash(\(firstBad), \(firstBad+1))")
            // Direct GPU hashPairs test
            let in2 = frFromInt(2)
            let in3 = frFromInt(3)
            let p2e = try Poseidon2Engine()

            // Verify round constants in GPU buffer match CPU
            let rcBuf = p2e.rcBuffer
            let gpuRCPtr = rcBuf.contents().bindMemory(to: Fr.self, capacity: 192)
            let cpuRC = POSEIDON2_ROUND_CONSTANTS
            var rcMismatch = 0
            for r in 0..<64 {
                for e in 0..<3 {
                    let gpuVal = gpuRCPtr[r * 3 + e]
                    let cpuVal = cpuRC[r][e]
                    if !frEqual(gpuVal, cpuVal) {
                        if rcMismatch < 3 {
                            print("  RC mismatch at r=\(r) e=\(e)")
                        }
                        rcMismatch += 1
                    }
                }
            }
            print("  Round constant mismatches: \(rcMismatch)/192")

            let gpuH23 = try p2e.hashPairs([in2, in3])
            let cpuH23 = poseidon2Hash(in2, in3)
            print("  Direct GPU hashPairs(2,3) == CPU: \(frEqual(gpuH23[0], cpuH23) ? "MATCH" : "MISMATCH")")

            // Also test: hash(2,3) called TWICE on GPU to check determinism
            let gpuH23b = try p2e.hashPairs([in2, in3])
            print("  GPU hashPairs(2,3) deterministic: \(frEqual(gpuH23[0], gpuH23b[0]) ? "YES" : "NO")")

            // Test hash(1,2) via same engine
            let in1 = frFromInt(1)
            let gpuH12 = try p2e.hashPairs([in1, in2])
            let cpuH12 = poseidon2Hash(in1, in2)
            print("  Direct GPU hashPairs(1,2) == CPU: \(frEqual(gpuH12[0], cpuH12) ? "MATCH" : "MISMATCH")")
            // Test hash(0,0) - the empty tree hash
            let gpuH00 = try p2e.hashPairs([Fr.zero, Fr.zero])
            let cpuH00 = poseidon2Hash(Fr.zero, Fr.zero)
            print("  Direct GPU hashPairs(0,0) == CPU: \(frEqual(gpuH00[0], cpuH00) ? "MATCH" : "MISMATCH")")

            // Test range: hash(n, n+1) for n in 0..20
            var matchCount = 0
            var mismatchCount = 0
            var failedNs = [UInt64]()
            for n: UInt64 in 0...20 {
                let la = frFromInt(n)
                let lb = frFromInt(n + 1)
                let ch = poseidon2Hash(la, lb)
                let gh = try p2e.hashPairs([la, lb])
                if frEqual(ch, gh[0]) { matchCount += 1 } else { mismatchCount += 1; failedNs.append(n) }
            }
            print("  hash(n,n+1) for n=0..20: \(matchCount) match, \(mismatchCount) mismatch, failed: \(failedNs)")

            // Also test hash(n, 0) for various n - single non-zero input
            var failedN2 = [UInt64]()
            for n: UInt64 in 0...20 {
                let la = frFromInt(n)
                let lb = Fr.zero
                let ch = poseidon2Hash(la, lb)
                let gh = try p2e.hashPairs([la, lb])
                if !frEqual(ch, gh[0]) { failedN2.append(n) }
            }
            print("  hash(n,0) mismatches for n=0..20: \(failedN2)")

            // Direct GPU hash_pairs test using encodeHashPairs
            let p2Engine = try Poseidon2Engine()
            let frStride = MemoryLayout<Fr>.stride
            // Buffer layout: [input pairs at offset 0 (4 Fr)][output at offset 4*stride (2 Fr)]
            guard let testBuf = p2Engine.device.makeBuffer(length: 6 * frStride, options: .storageModeShared) else {
                print("  Failed to allocate test buffer")
                return
            }
            let tPtr = testBuf.contents().bindMemory(to: Fr.self, capacity: 6)
            tPtr[0] = l34; tPtr[1] = l35  // pair 0
            tPtr[2] = frFromInt(1); tPtr[3] = frFromInt(2)  // pair 1
            guard let cmdBuf = p2Engine.commandQueue.makeCommandBuffer() else { return }
            let enc = cmdBuf.makeComputeCommandEncoder()!
            p2Engine.encodeHashPairs(encoder: enc, buffer: testBuf, inputOffset: 0,
                                      outputOffset: 4 * frStride, count: 2)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            let gpuDirect0 = tPtr[4]
            let gpuDirect1 = tPtr[5]
            print("  GPU hash_pairs(35,36) == CPU: \(frEqual(gpuDirect0, cpuH) ? "MATCH" : "MISMATCH")")
            print("  GPU hash_pairs(1,2) == CPU: \(frEqual(gpuDirect1, cpuHash) ? "MATCH" : "MISMATCH")")

            // Now test what the incremental tree stores after GPU rehash
            // Build a depth-6 tree (capacity=64) and fill with 64 leaves using batch
            let t6 = try IncrementalMerkleTree(depth: 6)
            var leaves6 = [Fr]()
            for i in 0..<64 { leaves6.append(frFromInt(UInt64(i + 1))) }
            try t6.appendBatchFused(leaves: leaves6)
            let bp6 = t6.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 128)
            // Parent of leaves 34,35 is node 64+34/2... no, 1-indexed: leaf34 is at 64+34=98
            // Parent is 98/2 = 49. Children of 49 are 98 and 99.
            let stored49 = bp6[49]
            let child98 = bp6[98]
            let child99 = bp6[99]
            let cpuCheck49 = poseidon2Hash(child98, child99)
            print("  Depth-6 node[49]: stored matches CPU recomputed = \(frEqual(stored49, cpuCheck49) ? "PASS" : "FAIL")")
            var bad6 = 0
            for k in stride(from: 63, through: 1, by: -1) {
                let left = bp6[k * 2]
                let right = bp6[k * 2 + 1]
                let expected = poseidon2Hash(left, right)
                let actual = bp6[k]
                if !frEqual(actual, expected) { bad6 += 1 }
            }
            print("  Depth-6 batch 64: \(bad6) bad nodes")
        }

        // --- 0. Quick smoke test ---
        print("\n--- Smoke Test ---")
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

            // Proof
            let prf = tree.proof(index: 3)
            let leaf3 = frFromInt(UInt64(4))
            let proofOk = IncrementalMerkleTree.verify(leaf: leaf3, proof: prf, root: tree.root)
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
            print(String(format: "  Append 1 leaf (tree 2^%d, %d filled): %.1f us (CPU path, %d hashes)",
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
                // Pad to power of 2
                let totalCount = prefill + batchSize
                let nextPow2 = 1 << depth
                while fullLeaves.count < nextPow2 { fullLeaves.append(Fr.zero) }

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
                let idx = (i * 7919 + 13) % n  // deterministic pseudo-random
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

        // --- 4. Sequential appends (build from empty) ---
        print("\n--- Sequential Build (append 1 at a time) ---")
        for logN in [10, 14] {
            let depth = logN
            let n = 1 << logN
            let tree = try IncrementalMerkleTree(depth: depth)

            let t0 = CFAbsoluteTimeGetCurrent()
            for i in 0..<n {
                try tree.append(leaf: frFromInt(UInt64(i + 1)))
            }
            let seqMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Full rebuild
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }
            let merkle = try Poseidon2MerkleEngine()
            let _ = try merkle.merkleRoot(leaves)  // warmup
            let t1 = CFAbsoluteTimeGetCurrent()
            let _ = try merkle.merkleRoot(leaves)
            let fullMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            print(String(format: "  Build 2^%d sequential: %.1f ms | single rebuild: %.2f ms",
                        logN, seqMs, fullMs))
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

        // Debug: check internal nodes by rebuilding path manually
        do {
            // Verify CPU hash determinism
            let a = frFromInt(1)
            let b = frFromInt(2)
            let h1 = poseidon2Hash(a, b)
            let h2 = poseidon2Hash(a, b)
            print("  CPU hash determinism: \(frEqual(h1, h2) ? "PASS" : "FAIL")")

            // Check specific bad node 8209: parent of leaves 34,35
            let bufPtr2 = tree.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 2 * n)
            let leaf34 = bufPtr2[n + 34]   // leaf 34 stored in buffer
            let leaf35 = bufPtr2[n + 35]   // leaf 35
            let expected34 = frFromInt(UInt64(34 + 1))  // what we wrote
            let expected35 = frFromInt(UInt64(35 + 1))
            print("  Leaf 34 correct: \(frEqual(leaf34, expected34) ? "PASS" : "FAIL")")
            print("  Leaf 35 correct: \(frEqual(leaf35, expected35) ? "PASS" : "FAIL")")
            let recomputed = poseidon2Hash(leaf34, leaf35)
            let stored8209 = bufPtr2[8209]
            print("  Node 8209: stored matches recomputed = \(frEqual(stored8209, recomputed) ? "PASS" : "FAIL")")

            let merkle2 = try Poseidon2MerkleEngine()
            let fullTree = try merkle2.buildTree(allLeaves)
            // fullTree layout: [leaves 0..<n, internal n..<2n-1] root at 2n-2
            // incremental layout: 1-indexed heap, leaves at [cap, 2*cap), root at 1
            // Compare a few internal nodes
            let bufPtr = tree.nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 2 * n)
            var badByLevel = [Int](repeating: 0, count: depth)
            var totalByLevel = [Int](repeating: 0, count: depth)
            var firstBadPerLevel = [Int](repeating: -1, count: depth)
            // Check bottom-up: level 0 = parents of leaves (indices [n/2, n))
            // level L = indices [n >> (L+1), n >> L)
            for level in 0..<depth {
                let lo = n >> (level + 1)
                let hi = n >> level
                for k in lo..<hi {
                    totalByLevel[level] += 1
                    let left = bufPtr[k * 2]
                    let right = bufPtr[k * 2 + 1]
                    let expected = poseidon2Hash(left, right)
                    let actual = bufPtr[k]
                    if !frEqual(actual, expected) {
                        badByLevel[level] += 1
                        if firstBadPerLevel[level] == -1 { firstBadPerLevel[level] = k }
                    }
                }
            }
            var totalBad = 0
            for level in 0..<depth {
                totalBad += badByLevel[level]
                if badByLevel[level] > 0 {
                    let lo = n >> (level + 1)
                    print("  Level \(level) [\(lo)..\(lo + totalByLevel[level])): \(badByLevel[level])/\(totalByLevel[level]) bad, first bad=\(firstBadPerLevel[level])")
                }
            }
            print("  Internal node check: \(totalBad) bad nodes out of \(n - 1)")
        }

        // Verify proof for a few leaves
        var proofOk = true
        for idx in [0, n/2, n-1, 42, n/4] {
            let prf = tree.proof(index: idx)
            let leaf = allLeaves[idx]
            if !IncrementalMerkleTree.verify(leaf: leaf, proof: prf, root: tree.root) {
                print("  Proof verification FAILED for index \(idx)")
                proofOk = false
            }
        }
        print("  Merkle proofs: \(proofOk ? "PASS" : "FAIL")")

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

        // Verify proof after updates
        var postUpdateProofOk = true
        for (idx, leaf) in updates.prefix(5) {
            let prf = tree.proof(index: idx)
            if !IncrementalMerkleTree.verify(leaf: leaf, proof: prf, root: tree.root) {
                print("  Post-update proof FAILED for index \(idx)")
                postUpdateProofOk = false
            }
        }
        print("  Post-update proofs: \(postUpdateProofOk ? "PASS" : "FAIL")")

    } catch {
        print("ERROR: \(error)")
    }
}
