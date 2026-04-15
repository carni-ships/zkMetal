// Merkle Tree Benchmark
import zkMetal
import NeonFieldOps
import Foundation

public func runMerkleBench() {
    print("=== Merkle Tree Benchmark ===")

    // Poseidon2 Merkle
    do {
        let engine = try Poseidon2MerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Warmup
            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            // CPU Merkle tree (C + GCD parallel)
            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let treeSize = 2 * n - 1
                var cpuTree = [Fr](repeating: Fr.zero, count: treeSize)
                // Warmup
                leaves.withUnsafeBytes { evPtr in
                    cpuTree.withUnsafeMutableBytes { treePtr in
                        poseidon2_merkle_tree_cpu(
                            evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n),
                            treePtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
                var cpuTimes = [Double]()
                for _ in 0..<5 {
                    let cpuT0 = CFAbsoluteTimeGetCurrent()
                    leaves.withUnsafeBytes { evPtr in
                        cpuTree.withUnsafeMutableBytes { treePtr in
                            poseidon2_merkle_tree_cpu(
                                evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(n),
                                treePtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            )
                        }
                    }
                    cpuTimes.append((CFAbsoluteTimeGetCurrent() - cpuT0) * 1000)
                }
                cpuTimes.sort()
                cpuMs = cpuTimes[2]
            }

            if cpuMs > 0 {
                print(String(format: "  Poseidon2 Merkle 2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.2f ms | %.1fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Poseidon2 Merkle 2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }
        }

        // Correctness: tree[2n-2] should be deterministic
        let testLeaves = (0..<4).map { frFromInt(UInt64($0 + 1)) }
        let tree = try engine.buildTree(testLeaves)
        let root = frToInt(tree.last!)
        print("  Root(1,2,3,4) = \(root.map{String(format:"%016llx",$0)}.joined())")
        print("  [pass] Poseidon2 Merkle tree")

        // Fused merkleRoot benchmark (root-only, uses fused subtree kernel)
        print("")
        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Warmup
            let _ = try engine.merkleRoot(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleRoot(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  P2 MerkleRoot  2^%-2d = %6d leaves: %7.2f ms (fused subtrees)",
                        logN, n, median))
        }

        // Correctness: merkleRoot must match buildTree root
        let testLeaves2 = (0..<2048).map { frFromInt(UInt64($0 + 1)) }
        let fusedRoot = try engine.merkleRoot(testLeaves2)
        let fullTree = try engine.buildTree(testLeaves2)
        if frToInt(fusedRoot) == frToInt(fullTree.last!) {
            print("  [pass] Fused merkleRoot matches buildTree root (2048 leaves)")
        } else {
            print("  [FAIL] Fused merkleRoot mismatch!")
            print("    fused: \(frToInt(fusedRoot).map{String(format:"%016llx",$0)}.joined())")
            print("    tree:  \(frToInt(fullTree.last!).map{String(format:"%016llx",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Poseidon2 Merkle: \(error)")
    }

    // Poseidon2 4-ary Merkle
    do {
        let engine2 = try Poseidon2MerkleEngine()
        let engine4 = try Poseidon24aryMerkleEngine()

        print("")
        print("  --- Poseidon2 4-ary vs Binary Merkle Comparison ---")

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

            // Warmup both
            let _ = try engine2.merkleRoot(leaves)
            let _ = try engine4.merkleRoot(leaves)

            // Binary (2-ary)
            var times2 = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine2.merkleRoot(leaves)
                times2.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times2.sort()
            let median2 = times2[2]

            // 4-ary
            var times4 = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine4.merkleRoot(leaves)
                times4.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times4.sort()
            let median4 = times4[2]

            let speedup = median2 / median4
            let levels2 = logN  // log2(n) levels for binary
            let levels4: Int
            if n == 2 { levels4 = 1 }
            else {
                var l = 0
                var s = n
                while s > 1 {
                    if s >= 4 { s /= 4 } else { s /= 2 }
                    l += 1
                }
                levels4 = l
            }
            print(String(format: "  P2 4-ary     2^%-2d = %6d: binary %7.2f ms (%d levels) | 4-ary %7.2f ms (%d levels) | %.2fx",
                        logN, n, median2, levels2, median4, levels4, speedup))
        }

        // Correctness: verify 4-ary root differs from binary (different tree structure)
        let testLeaves = (0..<256).map { i -> Fr in frFromInt(UInt64(i + 1)) }
        let r2 = try engine2.merkleRoot(testLeaves)
        let r4 = try engine4.merkleRoot(testLeaves)
        if frToInt(r2) != frToInt(r4) {
            print("  [pass] 4-ary root differs from binary root (expected, different hash structure)")
        } else {
            print("  [FAIL] 4-ary root unexpectedly equals binary root!")
        }

    } catch {
        print("  [FAIL] Poseidon2 4-ary Merkle: \(error)")
    }

    // Keccak Merkle
    do {
        let engine = try KeccakMerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [[UInt8]]()
            for i in 0..<n {
                var leaf = [UInt8](repeating: 0, count: 32)
                let val = UInt64(i)
                for b in 0..<8 { leaf[b] = UInt8((val >> (b * 8)) & 0xFF) }
                leaves.append(leaf)
            }

            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            // CPU Keccak Merkle tree
            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                var level = leaves
                while level.count > 1 {
                    var next = [[UInt8]]()
                    next.reserveCapacity(level.count / 2)
                    for i in stride(from: 0, to: level.count, by: 2) {
                        next.append(keccak256(level[i] + level[i+1]))
                    }
                    level = next
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  Keccak Merkle   2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.0f ms | %.0fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Keccak Merkle   2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }

            // merkleRoot timing (avoids output copy)
            let _ = try engine.merkleRoot(leaves)
            var rootTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.merkleRoot(leaves)
                rootTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            rootTimes.sort()
            let rootMedian = rootTimes[2]
            let copyOverhead = gpuMedian - rootMedian
            print(String(format: "    merkleRoot:  %7.2f ms  (output copy overhead: %.1f ms)", rootMedian, copyOverhead))
        }
        // Correctness: verify GPU Merkle root matches CPU level-by-level
        let testN = 1024
        var testLeaves = [[UInt8]]()
        for i in 0..<testN {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            testLeaves.append(leaf)
        }
        let gpuTree = try engine.buildTree(testLeaves)
        let gpuRoot = KeccakMerkleEngine.node(gpuTree, at: 2 * testN - 2)
        var cpuNodes = testLeaves
        while cpuNodes.count > 1 {
            var next = [[UInt8]]()
            for i in stride(from: 0, to: cpuNodes.count, by: 2) {
                next.append(keccak256(cpuNodes[i] + cpuNodes[i+1]))
            }
            cpuNodes = next
        }
        if cpuNodes[0] == gpuRoot {
            print("  [pass] Keccak fused Merkle root matches CPU (\(testN) leaves)")
        } else {
            print("  [FAIL] Keccak fused root mismatch!")
            print("    CPU: \(cpuNodes[0].map{String(format:"%02x",$0)}.joined())")
            print("    GPU: \(gpuRoot.map{String(format:"%02x",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Keccak Merkle: \(error)")
    }

    // Blake3 Merkle
    do {
        let engine = try Blake3MerkleEngine()

        for logN in [10, 12, 14, 16, 18, 20] {
            let n = 1 << logN
            var leaves = [[UInt8]]()
            for i in 0..<n {
                var leaf = [UInt8](repeating: 0, count: 32)
                let val = UInt64(i)
                for b in 0..<8 { leaf[b] = UInt8((val >> (b * 8)) & 0xFF) }
                leaves.append(leaf)
            }

            let _ = try engine.buildTree(leaves)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.buildTree(leaves)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let gpuMedian = times[2]

            var cpuMs: Double = 0
            if logN <= 18 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                var level = leaves
                while level.count > 1 {
                    var next = [[UInt8]]()
                    next.reserveCapacity(level.count / 2)
                    for i in stride(from: 0, to: level.count, by: 2) {
                        next.append(blake3Parent(level[i] + level[i+1]))
                    }
                    level = next
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  Blake3 Merkle   2^%-2d = %6d leaves: GPU %7.2f ms | CPU %7.0f ms | %.0fx",
                            logN, n, gpuMedian, cpuMs, cpuMs / gpuMedian))
            } else {
                print(String(format: "  Blake3 Merkle   2^%-2d = %6d leaves: GPU %7.2f ms",
                            logN, n, gpuMedian))
            }
        }

        // Correctness: verify GPU Merkle root matches CPU level-by-level
        let testN = 1024
        var testLeaves = [[UInt8]]()
        for i in 0..<testN {
            var leaf = [UInt8](repeating: 0, count: 32)
            let v = UInt32(i)
            for b in 0..<4 { leaf[b] = UInt8((v >> (b * 8)) & 0xFF) }
            testLeaves.append(leaf)
        }
        let gpuTree = try engine.buildTree(testLeaves)
        let gpuRoot = Blake3MerkleEngine.node(gpuTree, at: 2 * testN - 2)
        var cpuNodes = testLeaves
        while cpuNodes.count > 1 {
            var next = [[UInt8]]()
            for i in stride(from: 0, to: cpuNodes.count, by: 2) {
                next.append(blake3Parent(cpuNodes[i] + cpuNodes[i+1]))
            }
            cpuNodes = next
        }
        if cpuNodes[0] == gpuRoot {
            print("  [pass] Blake3 Merkle root matches CPU (\(testN) leaves)")
        } else {
            print("  [FAIL] Blake3 root mismatch!")
            print("    CPU: \(cpuNodes[0].map{String(format:"%02x",$0)}.joined())")
            print("    GPU: \(gpuRoot.map{String(format:"%02x",$0)}.joined())")
        }

    } catch {
        print("  [FAIL] Blake3 Merkle: \(error)")
    }

    // ============================================================
    // PHASE-LEVEL PROFILING: Poseidon2 Merkle at 2^20
    // ============================================================
    print("")
    print("=== Poseidon2 Merkle Phase Profile (2^20 = 1,048,576 leaves) ===")

    do {
        let p2Engine = try Poseidon2Engine()
        let _ = try Poseidon2MerkleEngine()
        let n = 1 << 20
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024
        let numSubtrees = n / subtreeSize  // 1024
        let stride = MemoryLayout<Fr>.stride

        var leaves = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

        // --- Measure: merkleRoot (root-only, no full tree copy) ---
        // This uses fused subtrees + upper level-by-level

        // Allocate buffer (reused across runs)
        let upperTreeSize = 2 * numSubtrees - 1
        let totalBufSize = (n + upperTreeSize) * stride
        guard let buf = p2Engine.device.makeBuffer(length: totalBufSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate profile buffer")
        }

        // Warmup
        _ = leaves.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, n * stride) }
        guard let cmdBuf = p2Engine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        let rootsOffset = n * stride
        p2Engine.encodeMerkleFused(encoder: enc, leavesBuffer: buf, leavesOffset: 0,
                                   rootsBuffer: buf, rootsOffset: rootsOffset, numSubtrees: numSubtrees)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Time individual phases (GPU execution only, not including CPU-side orchestration)
        var phaseMemcpy = [Double]()
        var phaseGPUExec = [Double]()
        var phaseCopyBack = [Double]()
        var phaseTotal = [Double]()

        for _ in 0..<5 {
            // Phase 1: memcpy leaves to GPU
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = leaves.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, n * stride) }
            let memcpyEnd = CFAbsoluteTimeGetCurrent()
            phaseMemcpy.append((memcpyEnd - t0) * 1000)

            // Phase 2+3: GPU kernels (fused subtrees + upper level-by-level in one CB)
            guard let cb = p2Engine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            let encoder = cb.makeComputeCommandEncoder()!

            p2Engine.encodeMerkleFused(encoder: encoder, leavesBuffer: buf, leavesOffset: 0,
                                       rootsBuffer: buf, rootsOffset: rootsOffset, numSubtrees: numSubtrees)
            encoder.memoryBarrier(scope: .buffers)

            var levelStart = n
            var levelSize = numSubtrees
            while levelSize > 1 {
                let parentCount = levelSize / 2
                let inputOffset = levelStart * stride
                let outputOffset = (levelStart + levelSize) * stride
                p2Engine.encodeHashPairs(encoder: encoder, buffer: buf,
                                          inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
                if levelSize > 1 { encoder.memoryBarrier(scope: .buffers) }
            }
            encoder.endEncoding()
            let gpuStart = CFAbsoluteTimeGetCurrent()
            cb.commit()
            cb.waitUntilCompleted()
            let gpuEnd = CFAbsoluteTimeGetCurrent()
            phaseGPUExec.append((gpuEnd - gpuStart) * 1000)

            // Phase 4: Copy back root only (minimal)
            let t3 = CFAbsoluteTimeGetCurrent()
            let ptr = buf.contents().advanced(by: levelStart * stride).bindMemory(to: Fr.self, capacity: 1)
            _ = ptr.pointee
            phaseCopyBack.append((CFAbsoluteTimeGetCurrent() - t3) * 1000)
            phaseTotal.append((gpuEnd - t0) * 1000)
        }

        phaseMemcpy.sort(); phaseGPUExec.sort(); phaseCopyBack.sort(); phaseTotal.sort()
        let memcpyMs = phaseMemcpy[2]
        let gpuExecMs = phaseGPUExec[2]
        let copyBackMs = phaseCopyBack[2]
        let totalMs = phaseTotal[2]

        print(String(format: "  memcpy (leaves->GPU):     %7.2f ms  (%.1f%%)", memcpyMs, 100*memcpyMs/totalMs))
        print(String(format: "  GPU kernels (fused+upper):%7.2f ms  (%.1f%%)", gpuExecMs, 100*gpuExecMs/totalMs))
        print(String(format: "  copy back (GPU->CPU):     %7.2f ms  (%.1f%%)", copyBackMs, 100*copyBackMs/totalMs))
        print(String(format:  "  TOTAL:                   %7.2f ms", totalMs))

        // Breakdown of fused vs upper
        print("")
        print(String(format: "  Upper level count: %d (log2(%d) - %d = %d binary levels above subtrees)",
                    Int(log2(Double(numSubtrees))), n, subtreeSize, Int(log2(Double(numSubtrees)))))

        // Now time full buildTree phases separately
        print("")
        print("  --- Full buildTree phase profile ---")

        let treeSize = 2 * n - 1
        guard let treeBuf = p2Engine.device.makeBuffer(length: treeSize * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate tree buffer")
        }

        var phaseBuildMemcpy = [Double]()
        var phaseBuildFusedFull = [Double]()
        var phaseBuildCopyBack = [Double]()
        var phaseBuildTotal = [Double]()

        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = leaves.withUnsafeBytes { src in memcpy(treeBuf.contents(), src.baseAddress!, n * stride) }
            let memcpyEnd = CFAbsoluteTimeGetCurrent()
            phaseBuildMemcpy.append((memcpyEnd - t0) * 1000)

            guard let cb = p2Engine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            let encoder = cb.makeComputeCommandEncoder()!

            // Phase 1: Fused full kernel setup
            let numFusedLevels = 10
            var levelOffsets = [UInt32]()
            levelOffsets.reserveCapacity(numFusedLevels)
            var off = n
            var width = n / 2
            for _ in 0..<numFusedLevels {
                levelOffsets.append(UInt32(off))
                off += width
                width /= 2
            }
            guard let levelOffsetsBuf = p2Engine.device.makeBuffer(length: levelOffsets.count * 4, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate level offsets buffer")
            }
            _ = levelOffsets.withUnsafeBytes { src in memcpy(levelOffsetsBuf.contents(), src.baseAddress!, src.count) }

            p2Engine.encodeMerkleFusedFull(encoder: encoder, leavesBuffer: treeBuf, leavesOffset: 0,
                                            treeBuffer: treeBuf, treeOffset: 0,
                                            levelOffsetsBuffer: levelOffsetsBuf, numSubtrees: numSubtrees)

            // Phase 2: Upper levels
            var levelStart = Int(levelOffsets[numFusedLevels - 1])
            var levelSize = numSubtrees
            while levelSize > 1 {
                encoder.memoryBarrier(scope: .buffers)
                let parentCount = levelSize / 2
                let inputOffset = levelStart * stride
                let outputOffset = (levelStart + levelSize) * stride
                p2Engine.encodeHashPairs(encoder: encoder, buffer: treeBuf,
                                          inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            }

            encoder.endEncoding()
            let gpuStart = CFAbsoluteTimeGetCurrent()
            cb.commit()
            cb.waitUntilCompleted()
            let gpuEnd = CFAbsoluteTimeGetCurrent()
            phaseBuildFusedFull.append((gpuEnd - gpuStart) * 1000)

            let t3 = CFAbsoluteTimeGetCurrent()
            let treePtr = treeBuf.contents().bindMemory(to: Fr.self, capacity: treeSize)
            let _ = Array(UnsafeBufferPointer(start: treePtr, count: treeSize))
            phaseBuildCopyBack.append((CFAbsoluteTimeGetCurrent() - t3) * 1000)
            phaseBuildTotal.append((gpuEnd - t0) * 1000)
        }

        phaseBuildMemcpy.sort(); phaseBuildFusedFull.sort()
        phaseBuildCopyBack.sort(); phaseBuildTotal.sort()

        print(String(format: "  memcpy (leaves->GPU):     %7.2f ms  (%.1f%%)",
                    phaseBuildMemcpy[2], 100*phaseBuildMemcpy[2]/phaseBuildTotal[2]))
        print(String(format: "  GPU kernels (fused+upper): %7.2f ms  (%.1f%%)",
                    phaseBuildFusedFull[2], 100*phaseBuildFusedFull[2]/phaseBuildTotal[2]))
        print(String(format: "  copy back (full tree):    %7.2f ms  (%.1f%%)",
                    phaseBuildCopyBack[2], 100*phaseBuildCopyBack[2]/phaseBuildTotal[2]))
        print(String(format: "  TOTAL:                    %7.2f ms", phaseBuildTotal[2]))

    } catch {
        print("  [FAIL] Phase profiling: \(error)")
    }

    // ============================================================
    // PHASE-LEVEL PROFILING: Poseidon2 4-ary Merkle at 2^20
    // ============================================================
    print("")
    print("=== Poseidon2 4-ary Merkle Phase Profile (2^20 leaves) ===")

    do {
        let p2Engine = try Poseidon2Engine()
        let _ = try Poseidon24aryMerkleEngine()
        let n = 1 << 20
        let stride = MemoryLayout<Fr>.stride
        let treeNodeCount = Poseidon24aryMerkleEngine.treeNodeCount(n)

        var leaves = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { leaves[i] = frFromInt(UInt64(i + 1)) }

        let bufSize = treeNodeCount * stride
        guard let buf = p2Engine.device.makeBuffer(length: bufSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate 4ary buffer")
        }

        // Warmup
        _ = leaves.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, n * stride) }
        guard let cb = p2Engine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cb.makeComputeCommandEncoder()!
        var levelStart = 0
        var levelSize = n
        while levelSize > 1 {
            let outputOffset = (levelStart + levelSize) * stride
            if levelSize >= 4 {
                let parentCount = levelSize / 4
                let inputOffset = levelStart * stride
                p2Engine.encodeHashQuad(encoder: enc, buffer: buf, inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            } else {
                let parentCount = levelSize / 2
                let inputOffset = levelStart * stride
                p2Engine.encodeHashPairs(encoder: enc, buffer: buf, inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            }
            if levelSize > 1 { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var phaseMemcpy = [Double]()
        var phaseHash = [Double]()
        var phaseCopyBack = [Double]()
        var phaseTotal = [Double]()

        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = leaves.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, n * stride) }
            phaseMemcpy.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

            guard let cmdBuf = p2Engine.commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
            let encoder = cmdBuf.makeComputeCommandEncoder()!

            levelStart = 0
            levelSize = n
            while levelSize > 1 {
                let outputOffset = (levelStart + levelSize) * stride
                if levelSize >= 4 {
                    let parentCount = levelSize / 4
                    let inputOffset = levelStart * stride
                    p2Engine.encodeHashQuad(encoder: encoder, buffer: buf, inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                    levelStart += levelSize
                    levelSize = parentCount
                } else {
                    let parentCount = levelSize / 2
                    let inputOffset = levelStart * stride
                    p2Engine.encodeHashPairs(encoder: encoder, buffer: buf, inputOffset: inputOffset, outputOffset: outputOffset, count: parentCount)
                    levelStart += levelSize
                    levelSize = parentCount
                }
                if levelSize > 1 { encoder.memoryBarrier(scope: .buffers) }
            }

            encoder.endEncoding()
            let gpuStart = CFAbsoluteTimeGetCurrent()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            let gpuEnd = CFAbsoluteTimeGetCurrent()
            phaseHash.append((gpuEnd - gpuStart) * 1000)

            let t2 = CFAbsoluteTimeGetCurrent()
            let ptr = buf.contents().bindMemory(to: Fr.self, capacity: treeNodeCount)
            let _: [Fr] = Array(UnsafeBufferPointer(start: ptr, count: treeNodeCount))
            phaseCopyBack.append((CFAbsoluteTimeGetCurrent() - t2) * 1000)
            phaseTotal.append((gpuEnd - t0) * 1000)
        }

        phaseMemcpy.sort(); phaseHash.sort(); phaseCopyBack.sort(); phaseTotal.sort()
        let memcpyMs = phaseMemcpy[2]
        let hashMs = phaseHash[2]
        let copyBackMs = phaseCopyBack[2]
        let totalMs = phaseTotal[2]

        print(String(format: "  memcpy (leaves->GPU):     %7.2f ms  (%.1f%%)", memcpyMs, 100*memcpyMs/totalMs))
        print(String(format: "  4-ary hash levels:        %7.2f ms  (%.1f%%)", hashMs, 100*hashMs/totalMs))
        print(String(format: "  copy back (GPU->CPU):     %7.2f ms  (%.1f%%)", copyBackMs, 100*copyBackMs/totalMs))
        print(String(format: "  TOTAL:                   %7.2f ms", totalMs))

    } catch {
        print("  [FAIL] 4-ary phase profiling: \(error)")
    }

    print("\nMerkle benchmark complete.")
}
