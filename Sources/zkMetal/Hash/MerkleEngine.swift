// GPU-accelerated Merkle tree construction
// Supports both Poseidon2 (field-native) and Keccak-256 hash functions.
import Foundation
import Metal

// MARK: - Poseidon2 Merkle Tree

public class Poseidon2MerkleEngine {
    private let engine: Poseidon2Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0
    private var cachedLevelOffsetsBuf: MTLBuffer?
    private var cachedRootBuf: MTLBuffer?
    private var treeSize2: Int = 0

    public init() throws {
        self.engine = try Poseidon2Engine()
    }

    /// Build a Merkle tree from leaf Fr elements using Poseidon2 2-to-1 hashing.
    /// Returns all tree nodes: [leaves..., internal nodes..., root].
    /// Tree layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal (bottom up), nodes[2n-2] = root.
    /// For n >= 1024: fused subtree kernel processes bottom 10 levels in shared memory
    /// (writes intermediate nodes via global memory), then level-by-level for remaining levels.
    public func buildTree(_ leaves: [Fr]) throws -> [Fr] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")

        let treeSize = 2 * n - 1
        let stride = MemoryLayout<Fr>.stride
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024

        // Reuse cached tree buffer when possible
        if treeSize > cachedTreeBufNodes {
            guard let buf = engine.device.makeBuffer(length: treeSize * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate Merkle tree buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufNodes = treeSize
        }
        let treeBuf = cachedTreeBuf!

        // Copy leaves to GPU buffer once
        leaves.withUnsafeBytes { src in
            memcpy(treeBuf.contents(), src.baseAddress!, n * stride)
        }

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Use fused subtrees only when n <= 65536. At larger sizes, the thread waste
        // from idle threads in upper subtree levels exceeds dispatch overhead savings.
        let useFused = n >= subtreeSize && n <= 65536

        if useFused {
            // Phase 1: Fused subtrees for bottom 10 levels.
            let numSubtrees = n / subtreeSize
            let numFusedLevels = 10

            // Compute level offsets for the tree layout
            var levelOffsets = [UInt32]()
            levelOffsets.reserveCapacity(numFusedLevels)
            var off = n
            var width = n / 2
            for _ in 0..<numFusedLevels {
                levelOffsets.append(UInt32(off))
                off += width
                width /= 2
            }

            if cachedLevelOffsetsBuf == nil || cachedLevelOffsetsBuf!.length < levelOffsets.count * 4 {
                cachedLevelOffsetsBuf = engine.device.makeBuffer(length: levelOffsets.count * 4, options: .storageModeShared)
            }
            let levelOffsetsBuf = cachedLevelOffsetsBuf!
            levelOffsets.withUnsafeBytes { src in
                memcpy(levelOffsetsBuf.contents(), src.baseAddress!, src.count)
            }

            engine.encodeMerkleFusedFull(encoder: enc,
                                          leavesBuffer: treeBuf, leavesOffset: 0,
                                          treeBuffer: treeBuf, treeOffset: 0,
                                          levelOffsetsBuffer: levelOffsetsBuf,
                                          numSubtrees: numSubtrees)

            // Phase 2: Level-by-level for remaining levels above the fused subtree roots
            var levelStart = Int(levelOffsets[numFusedLevels - 1])
            var levelSize = numSubtrees

            while levelSize > 1 {
                enc.memoryBarrier(scope: .buffers)
                let parentCount = levelSize / 2
                let inputOffset = levelStart * stride
                let outputOffset = (levelStart + levelSize) * stride

                engine.encodeHashPairs(encoder: enc, buffer: treeBuf,
                                       inputOffset: inputOffset,
                                       outputOffset: outputOffset,
                                       count: parentCount)

                levelStart += levelSize
                levelSize = parentCount
            }
        } else {
            // Small tree: level-by-level
            var levelStart = 0
            var levelSize = n

            while levelSize > 1 {
                let parentCount = levelSize / 2
                let inputOffset = levelStart * stride
                let outputOffset = (levelStart + levelSize) * stride

                engine.encodeHashPairs(encoder: enc, buffer: treeBuf,
                                       inputOffset: inputOffset,
                                       outputOffset: outputOffset,
                                       count: parentCount)

                levelStart += levelSize
                levelSize = parentCount

                if levelSize > 1 {
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Single copy back
        let ptr = treeBuf.contents().bindMemory(to: Fr.self, capacity: treeSize)
        return Array(UnsafeBufferPointer(start: ptr, count: treeSize))
    }

    /// Get the Merkle root from leaves.
    /// Optimized: uses fused subtree kernel for bottom 10 levels (64 subtrees of 1024)
    /// to eliminate dispatch overhead, then level-by-level for remaining levels.
    public func merkleRoot(_ leaves: [Fr]) throws -> Fr {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024
        let stride = MemoryLayout<Fr>.stride

        // For small trees or large trees (where thread waste exceeds dispatch overhead),
        // use buildTree which has the appropriate strategy for each size range.
        if n < subtreeSize || n > 65536 {
            let tree = try buildTree(leaves)
            return tree.last!
        }

        let numSubtrees = n / subtreeSize
        // We need: input buffer (leaves), intermediate buffer (for subtree roots + upper levels)
        let upperTreeSize = 2 * numSubtrees - 1  // upper tree nodes
        let totalBufSize = (n + upperTreeSize) * stride

        if treeSize2 < totalBufSize {
            guard let buf = engine.device.makeBuffer(length: totalBufSize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate Merkle root buffer")
            }
            cachedRootBuf = buf
            treeSize2 = totalBufSize
        }
        let buf = cachedRootBuf!

        // Copy leaves
        leaves.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * stride)
        }

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Phase 1: Fused subtrees — 1024-leaf subtrees in shared memory
        let rootsOffset = n * stride
        engine.encodeMerkleFused(encoder: enc,
                                  leavesBuffer: buf, leavesOffset: 0,
                                  rootsBuffer: buf, rootsOffset: rootsOffset,
                                  numSubtrees: numSubtrees)

        // Phase 2: Level-by-level for upper tree
        var levelStart = n
        var levelSize = numSubtrees

        while levelSize > 1 {
            enc.memoryBarrier(scope: .buffers)
            let parentCount = levelSize / 2
            let inputOffset = levelStart * stride
            let outputOffset = (levelStart + levelSize) * stride

            engine.encodeHashPairs(encoder: enc, buffer: buf,
                                   inputOffset: inputOffset,
                                   outputOffset: outputOffset,
                                   count: parentCount)

            levelStart += levelSize
            levelSize = parentCount
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = buf.contents().advanced(by: levelStart * stride).bindMemory(to: Fr.self, capacity: 1)
        return ptr.pointee
    }
}

// MARK: - Keccak Merkle Tree

public class KeccakMerkleEngine {
    private let engine: Keccak256Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0

    public init() throws {
        self.engine = try Keccak256Engine()
    }

    /// Build a Merkle tree from 32-byte leaf hashes using Keccak-256.
    /// Returns all tree nodes as 32-byte hashes: [leaves..., internal nodes..., root].
    /// Optimized: single GPU buffer, single command buffer, all levels encoded with barriers.
    public func buildTree(_ leaves: [[UInt8]]) throws -> [[UInt8]] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")
        precondition(leaves.allSatisfy { $0.count == 32 }, "Leaves must be 32 bytes each")

        let treeSize = 2 * n - 1

        if treeSize > cachedTreeBufNodes {
            guard let buf = engine.device.makeBuffer(length: treeSize * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate Keccak Merkle buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufNodes = treeSize
        }
        let treeBuf = cachedTreeBuf!

        // Copy leaves to GPU buffer once (contiguous 32-byte hashes)
        let ptr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<n {
            leaves[i].withUnsafeBytes { src in
                memcpy(ptr + i * 32, src.baseAddress!, 32)
            }
        }

        // Single command buffer with single encoder + memoryBarrier between levels
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        let subtreeSize = Keccak256Engine.merkleSubtreeSize  // 1024
        let useFused = n >= subtreeSize

        if useFused {
            // Phase 1: Fused subtrees — process bottom 10 levels in shared memory
            let numSubtrees = n / subtreeSize
            let fusedOutputOffset = (2 * n - 2 * numSubtrees) * 32

            engine.encodeMerkleFused(encoder: enc,
                                      leavesBuffer: treeBuf, leavesOffset: 0,
                                      rootsBuffer: treeBuf, rootsOffset: fusedOutputOffset,
                                      numSubtrees: numSubtrees)

            // Phase 2: Continue level-by-level from the subtree roots
            var levelStart = 2 * n - 2 * numSubtrees
            var levelSize = numSubtrees

            while levelSize > 1 {
                enc.memoryBarrier(scope: .buffers)
                let parentCount = levelSize / 2
                let inputOffset = levelStart * 32
                let outputOffset = (levelStart + levelSize) * 32

                engine.encodeHash64(encoder: enc, buffer: treeBuf,
                                    inputOffset: inputOffset,
                                    outputOffset: outputOffset,
                                    count: parentCount)

                levelStart += levelSize
                levelSize = parentCount
            }
        } else {
            // Small tree: level-by-level
            var levelStart = 0
            var levelSize = n

            while levelSize > 1 {
                let parentCount = levelSize / 2
                let inputOffset = levelStart * 32
                let outputOffset = (levelStart + levelSize) * 32

                engine.encodeHash64(encoder: enc, buffer: treeBuf,
                                    inputOffset: inputOffset,
                                    outputOffset: outputOffset,
                                    count: parentCount)

                levelStart += levelSize
                levelSize = parentCount

                if levelSize > 1 {
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Single flat copy, then slice into per-node arrays
        let readPtr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        let flat = Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
        var tree = [[UInt8]]()
        tree.reserveCapacity(treeSize)
        for i in 0..<treeSize {
            let start = i * 32
            tree.append(Array(flat[start..<start + 32]))
        }
        return tree
    }

    /// Get the Merkle root from 32-byte leaf hashes.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let tree = try buildTree(leaves)
        return tree.last!
    }
}

// MARK: - Blake3 Merkle Tree

public class Blake3MerkleEngine {
    private let engine: Blake3Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0

    public init() throws {
        self.engine = try Blake3Engine()
    }

    /// Build a Merkle tree from 32-byte leaf hashes using Blake3 parent compression.
    /// Returns all tree nodes as 32-byte hashes: [leaves..., internal nodes..., root].
    /// For n >= 1024, uses fused subtree kernel for bottom 10 levels, then level-by-level.
    public func buildTree(_ leaves: [[UInt8]]) throws -> [[UInt8]] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")
        precondition(leaves.allSatisfy { $0.count == 32 }, "Leaves must be 32 bytes each")

        let treeSize = 2 * n - 1

        if treeSize > cachedTreeBufNodes {
            guard let buf = engine.device.makeBuffer(length: treeSize * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate Blake3 Merkle buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufNodes = treeSize
        }
        let treeBuf = cachedTreeBuf!

        // Copy leaves to GPU buffer
        let ptr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<n {
            leaves[i].withUnsafeBytes { src in
                memcpy(ptr + i * 32, src.baseAddress!, 32)
            }
        }

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Level-by-level (fused subtrees tested but regressed — Blake3 parent compression
        // is too lightweight for shared memory barrier overhead to pay off)
        var levelStart = 0
        var levelSize = n

        while levelSize > 1 {
            let parentCount = levelSize / 2
            let inputOffset = levelStart * 32
            let outputOffset = (levelStart + levelSize) * 32

            engine.encodeHashPairs(encoder: enc, buffer: treeBuf,
                                   inputOffset: inputOffset,
                                   outputOffset: outputOffset,
                                   count: parentCount)

            levelStart += levelSize
            levelSize = parentCount

            if levelSize > 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let readPtr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        let flat = Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
        var tree = [[UInt8]]()
        tree.reserveCapacity(treeSize)
        for i in 0..<treeSize {
            let start = i * 32
            tree.append(Array(flat[start..<start + 32]))
        }
        return tree
    }

    /// Get the Merkle root from 32-byte leaf hashes.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let tree = try buildTree(leaves)
        return tree.last!
    }
}
