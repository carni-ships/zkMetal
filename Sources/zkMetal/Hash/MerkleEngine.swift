// GPU-accelerated Merkle tree construction
// Supports both Poseidon2 (field-native) and Keccak-256 hash functions.
import Foundation
import Metal

// MARK: - Poseidon2 Merkle Tree

public class Poseidon2MerkleEngine {
    private let engine: Poseidon2Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0

    public init() throws {
        self.engine = try Poseidon2Engine()
    }

    /// Build a Merkle tree from leaf Fr elements using Poseidon2 2-to-1 hashing.
    /// Returns all tree nodes: [leaves..., internal nodes..., root].
    /// Tree layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal (bottom up), nodes[2n-2] = root.
    /// For n >= 1024: uses fused subtree kernel for bottom 10 levels (eliminates 9 barriers per subtree).
    public func buildTree(_ leaves: [Fr]) throws -> [Fr] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")

        let treeSize = 2 * n - 1
        let stride = MemoryLayout<Fr>.stride

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

        // Level-by-level for Poseidon2: each hash is compute-heavy, so dispatch overhead
        // is negligible relative to hash time, while fused subtrees waste GPU occupancy
        // on idle threads at upper levels within each subtree.
        let enc = cmdBuf.makeComputeCommandEncoder()!

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
    public func merkleRoot(_ leaves: [Fr]) throws -> Fr {
        let tree = try buildTree(leaves)
        return tree.last!
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
