// GPU-accelerated batch Poseidon2 Merkle tree builder
// Computes entire tree levels in parallel on GPU — each level is embarrassingly parallel.
// Uses fused subtree kernels (shared memory) for bottom levels, level-by-level for upper levels.
//
// Tree layout (flat array, 0-indexed):
//   nodes[0..<n]       = leaves
//   nodes[n..<2n-1]    = internal nodes (bottom-up, level by level)
//   nodes[2n-2]        = root
//
// This matches Poseidon2MerkleEngine's buildTree layout.

import Foundation
import Metal

// MARK: - MerkleTree

/// Complete binary Merkle tree built from Fr leaves using Poseidon2 2-to-1 hashing.
/// Stores all nodes in a flat array for zero-copy proof extraction.
public struct MerkleTree {
    /// All tree nodes: [leaves..., internal nodes..., root].
    /// nodes[0..<leafCount] = leaves, nodes[leafCount..<2*leafCount-1] = internal, nodes[2*leafCount-2] = root.
    public let nodes: [Fr]
    /// Number of leaves (power of 2).
    public let leafCount: Int

    /// The Merkle root.
    public var root: Fr {
        nodes[2 * leafCount - 2]
    }

    /// Depth of the tree (number of levels from leaf to root).
    public var depth: Int {
        leafCount.trailingZeroBitCount
    }

    /// Get the leaf at a given index.
    public func leaf(at index: Int) -> Fr {
        precondition(index >= 0 && index < leafCount, "Leaf index out of range")
        return nodes[index]
    }

    /// Extract authentication path (sibling hashes from leaf to root, bottom-up).
    /// path[0] = sibling of leaf, path[depth-1] = sibling of root's child.
    public func proof(forLeafAt index: Int) -> MerkleAuthPath {
        precondition(index >= 0 && index < leafCount, "Leaf index out of range")

        var siblings = [Fr]()
        siblings.reserveCapacity(depth)
        var levelStart = 0
        var levelSize = leafCount
        var idx = index

        while levelSize > 1 {
            let sibling = idx ^ 1
            siblings.append(nodes[levelStart + sibling])
            levelStart += levelSize
            levelSize /= 2
            idx /= 2
        }

        return MerkleAuthPath(siblings: siblings, leafIndex: index, leafCount: leafCount)
    }

    /// Verify that a leaf at the given index produces the expected root.
    /// Recomputes the path from leaf to root on CPU using Poseidon2.
    public static func verifyPath(root: Fr, leaf: Fr, path: [Fr], index: Int) -> Bool {
        var current = leaf
        var idx = index

        for sibling in path {
            if idx & 1 == 0 {
                // current is left child
                current = poseidon2Hash(current, sibling)
            } else {
                // current is right child
                current = poseidon2Hash(sibling, current)
            }
            idx >>= 1
        }

        return frToInt(current) == frToInt(root)
    }
}

// MARK: - MerkleAuthPath

/// Authentication path for a Merkle proof.
public struct MerkleAuthPath {
    /// Sibling hashes from leaf to root (bottom-up).
    public let siblings: [Fr]
    /// Leaf index in [0, leafCount).
    public let leafIndex: Int
    /// Total number of leaves in the tree.
    public let leafCount: Int

    /// Verify this path against a root and leaf value.
    public func verify(root: Fr, leaf: Fr) -> Bool {
        MerkleTree.verifyPath(root: root, leaf: leaf, path: siblings, index: leafIndex)
    }

    /// Depth of the path (should equal log2(leafCount)).
    public var depth: Int { siblings.count }
}

// MARK: - GPUMerkleTreeEngine

/// High-level GPU Merkle tree engine using Poseidon2 hashing.
/// Wraps Poseidon2MerkleEngine with a structured MerkleTree API.
///
/// Usage:
///   let engine = try GPUMerkleTreeEngine()
///   let tree = try engine.buildTree(leaves: myLeaves)
///   let root = tree.root
///   let proof = tree.proof(forLeafAt: 0)
///   assert(proof.verify(root: root, leaf: myLeaves[0]))
public class GPUMerkleTreeEngine {
    public static let version = Versions.gpuMerkleTree

    private let inner: Poseidon2MerkleEngine

    /// Access the underlying Poseidon2Engine for advanced use.
    public var poseidon2Engine: Poseidon2Engine { inner.p2Engine }

    /// Access the underlying Poseidon2MerkleEngine for low-level buffer operations.
    public var merkleEngine: Poseidon2MerkleEngine { inner }

    public init() throws {
        self.inner = try Poseidon2MerkleEngine()
    }

    /// Build a complete binary Merkle tree on GPU from Fr leaves.
    /// Leaves count must be a power of 2.
    /// Returns a MerkleTree with all nodes (leaves + internal + root).
    ///
    /// GPU strategy:
    /// - n <= 65536: fused subtree kernel (shared memory, minimal barriers)
    /// - n > 65536: level-by-level parallel dispatch with memory barriers
    public func buildTree(leaves: [Fr]) throws -> MerkleTree {
        let nodes = try inner.buildTree(leaves)
        return MerkleTree(nodes: nodes, leafCount: leaves.count)
    }

    /// Compute only the Merkle root on GPU (avoids copying full tree back).
    /// More efficient than buildTree when you only need the root.
    public func merkleRoot(leaves: [Fr]) throws -> Fr {
        try inner.merkleRoot(leaves)
    }

    /// Build one tree level on GPU: given n children, compute n/2 parents.
    /// Children buffer contains pairs of Fr elements at childrenOffset.
    /// Results written to parentsBuffer at parentsOffset.
    /// Uses a single GPU dispatch — all pairs computed in parallel.
    public func buildTreeLevel(childrenBuffer: MTLBuffer, childrenOffset: Int,
                               parentsBuffer: MTLBuffer, parentsOffset: Int,
                               count: Int) throws {
        guard let cmdBuf = inner.p2Engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        inner.p2Engine.encodeHashPairs(encoder: enc,
                                        buffer: childrenBuffer,
                                        inputOffset: childrenOffset,
                                        outputOffset: parentsOffset,
                                        count: count)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Build one tree level from a Swift array: given 2*count children, compute count parents.
    /// Returns count Fr parent hashes.
    public func buildTreeLevel(children: [Fr]) throws -> [Fr] {
        precondition(children.count % 2 == 0, "Children count must be even")
        return try inner.p2Engine.hashPairs(children)
    }

    /// Verify a Merkle authentication path on CPU.
    /// Returns true if hashing leaf through the sibling path produces the expected root.
    public func verifyPath(root: Fr, leaf: Fr, path: [Fr], index: Int) -> Bool {
        MerkleTree.verifyPath(root: root, leaf: leaf, path: path, index: index)
    }

    /// Build multiple independent Merkle trees in a single GPU dispatch.
    /// Each tree's leaves must be a power of 2 and <= 1024.
    /// Returns an array of roots, one per tree.
    public func buildTreesBatch(treesLeaves: [[Fr]]) throws -> [Fr] {
        let numTrees = treesLeaves.count
        guard numTrees > 0 else { return [] }

        let stride = MemoryLayout<Fr>.stride
        let device = inner.p2Engine.device

        // Compute total leaves and tree params
        var totalLeaves = 0
        var treeParams = [UInt32]()
        treeParams.reserveCapacity(numTrees * 2)
        for leaves in treesLeaves {
            let n = leaves.count
            precondition(n > 0 && n <= 1024 && (n & (n - 1)) == 0,
                         "Each tree must have 1..1024 leaves (power of 2)")
            treeParams.append(UInt32(totalLeaves))  // leaf_offset
            treeParams.append(UInt32(n.trailingZeroBitCount))  // num_levels
            totalLeaves += n
        }

        // Allocate and fill leaves buffer
        guard let leavesBuf = device.makeBuffer(length: totalLeaves * stride, options: .storageModeShared),
              let rootsBuf = device.makeBuffer(length: numTrees * stride, options: .storageModeShared),
              let paramsBuf = device.makeBuffer(length: treeParams.count * 4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate batch Merkle buffers")
        }

        var offset = 0
        for leaves in treesLeaves {
            _ = leaves.withUnsafeBytes { src in
                memcpy(leavesBuf.contents().advanced(by: offset), src.baseAddress!, src.count)
            }
            offset += leaves.count * stride
        }

        _ = treeParams.withUnsafeBytes { src in
            memcpy(paramsBuf.contents(), src.baseAddress!, src.count)
        }

        // Single dispatch: one threadgroup per tree
        guard let cmdBuf = inner.p2Engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        inner.p2Engine.encodeMerkleFusedBatch(encoder: enc,
                                               leavesBuffer: leavesBuf,
                                               rootsBuffer: rootsBuf, rootsOffset: 0,
                                               rcBuffer: inner.p2Engine.rcBuffer,
                                               treeParamsBuffer: paramsBuf,
                                               numTrees: numTrees)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = rootsBuf.contents().bindMemory(to: Fr.self, capacity: numTrees)
        return Array(UnsafeBufferPointer(start: ptr, count: numTrees))
    }
}
