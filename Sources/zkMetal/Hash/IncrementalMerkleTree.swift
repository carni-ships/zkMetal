// Incremental Merkle tree with GPU-accelerated rehashing via unified memory
// CPU appends leaves directly into a shared MTLBuffer; GPU re-hashes only dirty subtrees.
// Tree layout: 1-indexed heap. nodes[1] = root, nodes[2i] = left child, nodes[2i+1] = right child.
// Leaves occupy indices [capacity, 2*capacity). Internal nodes occupy [1, capacity).
// Zero (Fr.zero) is the default empty-leaf value.

import Foundation
import Metal

// MARK: - Merkle Proof

public struct MerkleProof {
    /// Sibling hashes from leaf to root (bottom-up). siblings[0] is the sibling of the leaf.
    public let siblings: [Fr]
    /// Path bits: false = leaf is left child, true = leaf is right child
    public let pathBits: [Bool]
    /// Leaf index in [0, capacity)
    public let leafIndex: Int
}

// MARK: - Dirty Tracker

/// Tracks which internal nodes need rehashing after leaf mutations.
/// Organized by level (0 = leaves, depth = root). Each level stores deduplicated node indices.
struct DirtyTracker {
    let depth: Int
    /// dirtyByLevel[level] contains 1-indexed node indices that need rehashing.
    /// Level 0 = parent of leaves (capacity/2 .. capacity-1), level depth-1 = root (index 1).
    var dirtyByLevel: [Set<Int>]

    init(depth: Int) {
        self.depth = depth
        self.dirtyByLevel = Array(repeating: Set<Int>(), count: depth)
    }

    /// Mark a single leaf index (0-based) as dirty, propagating up all ancestors.
    mutating func markDirty(leafIndex: Int, capacity: Int) {
        // Leaf is at 1-indexed position (capacity + leafIndex).
        // Its parent is at (capacity + leafIndex) / 2.
        var nodeIdx = (capacity + leafIndex) >> 1
        for level in 0..<depth {
            dirtyByLevel[level].insert(nodeIdx)
            nodeIdx >>= 1
        }
    }

    /// Mark a contiguous range of leaves [start, start+count) as dirty.
    mutating func markRange(start: Int, count: Int, capacity: Int) {
        // For each level, compute the range of affected parents.
        var lo = (capacity + start) >> 1
        var hi = (capacity + start + count - 1) >> 1
        for level in 0..<depth {
            for idx in lo...hi {
                dirtyByLevel[level].insert(idx)
            }
            lo >>= 1
            hi >>= 1
        }
    }

    /// Total dirty nodes across all levels.
    var totalDirty: Int {
        dirtyByLevel.reduce(0) { $0 + $1.count }
    }

    mutating func clear() {
        for i in 0..<dirtyByLevel.count {
            dirtyByLevel[i].removeAll(keepingCapacity: true)
        }
    }
}

// MARK: - Incremental Merkle Tree

/// Persistent Merkle tree that supports incremental leaf updates.
/// Uses Apple Silicon unified memory: tree nodes live in an MTLBuffer accessible by both CPU and GPU.
/// Poseidon2 2-to-1 hashing (BN254 Fr).
public class IncrementalMerkleTree {
    public static let version = Versions.incrementalMerkle

    /// Tree depth (number of levels from leaf to root).
    public let depth: Int
    /// Maximum number of leaves = 2^depth
    public let capacity: Int
    /// Current number of appended leaves
    public private(set) var count: Int = 0

    /// Tree stored as 1-indexed heap in unified memory.
    /// Total allocation: (2 * capacity) Fr elements. Index 0 is unused.
    /// nodes[1] = root, nodes[capacity..2*capacity-1] = leaves.
    public let nodeBuffer: MTLBuffer

    private let engine: Poseidon2Engine
    private var dirtyTracker: DirtyTracker
    private var dirtyIndicesBuf: MTLBuffer?
    private var dirtyIndicesBufCapacity: Int = 0

    /// GPU rehash threshold: use GPU when dirty count at a level exceeds this.
    /// Below this, CPU hashing is faster due to dispatch overhead.
    private let gpuThreshold: Int = 16

    public init(depth: Int) throws {
        precondition(depth > 0 && depth <= 26, "Depth must be in [1, 26]")
        self.depth = depth
        self.capacity = 1 << depth
        self.engine = try Poseidon2Engine()
        self.dirtyTracker = DirtyTracker(depth: depth)

        let totalNodes = 2 * capacity  // index 0 unused, 1..2*capacity-1
        let stride = MemoryLayout<Fr>.stride
        guard let buf = engine.device.makeBuffer(length: totalNodes * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate incremental Merkle buffer (\(totalNodes) nodes)")
        }
        self.nodeBuffer = buf

        // Initialize all nodes to zero (default empty leaf)
        memset(buf.contents(), 0, totalNodes * stride)

        // Pre-hash the empty tree: hash(0,0) at each level
        // This gives correct roots for partially-filled trees.
        precomputeEmptyTree()
    }

    /// Initialize with an existing Poseidon2Engine (shares GPU resources).
    public init(depth: Int, engine: Poseidon2Engine) throws {
        precondition(depth > 0 && depth <= 26, "Depth must be in [1, 26]")
        self.depth = depth
        self.capacity = 1 << depth
        self.engine = engine
        self.dirtyTracker = DirtyTracker(depth: depth)

        let totalNodes = 2 * capacity
        let stride = MemoryLayout<Fr>.stride
        guard let buf = engine.device.makeBuffer(length: totalNodes * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate incremental Merkle buffer (\(totalNodes) nodes)")
        }
        self.nodeBuffer = buf
        memset(buf.contents(), 0, totalNodes * stride)
        precomputeEmptyTree()
    }

    // MARK: - Node Access (unified memory — zero copy)

    private var nodePtr: UnsafeMutablePointer<Fr> {
        nodeBuffer.contents().bindMemory(to: Fr.self, capacity: 2 * capacity)
    }

    @inline(__always)
    private func getNode(_ index: Int) -> Fr {
        nodePtr[index]
    }

    @inline(__always)
    private func setNode(_ index: Int, _ value: Fr) {
        nodePtr[index] = value
    }

    // MARK: - Empty Tree

    /// Pre-compute internal nodes for an all-zero tree so that root is valid from the start.
    private func precomputeEmptyTree() {
        // Leaves are already zero. Hash bottom-up: each parent = hash(left, right) = hash(0, 0).
        // Since all leaves are identical, each level has the same hash value.
        var levelHash = Fr.zero
        // Level 0 parents: hash(leaf, leaf) for all
        for level in 0..<depth {
            levelHash = poseidon2Hash(levelHash, levelHash)
            let levelStart = capacity >> (level + 1)  // first node index at this internal level
            let levelEnd = capacity >> level           // exclusive
            // All nodes at this level have the same value
            for i in levelStart..<levelEnd {
                setNode(i, levelHash)
            }
        }
    }

    // MARK: - Append

    /// Append a single leaf. CPU hashes the O(log n) path.
    public func append(leaf: Fr) throws {
        guard count < capacity else {
            throw MSMError.gpuError("Incremental Merkle tree full (capacity \(capacity))")
        }
        let leafIdx = count
        setNode(capacity + leafIdx, leaf)
        count += 1

        // Rehash the path from this leaf to root (CPU, O(depth) hashes)
        rehashPathCPU(leafIndex: leafIdx)
    }

    /// Batch append multiple leaves. GPU accelerated for large batches.
    public func appendBatch(leaves: [Fr]) throws {
        guard count + leaves.count <= capacity else {
            throw MSMError.gpuError("Batch append would exceed capacity")
        }
        let startIdx = count
        let batchCount = leaves.count

        // Write leaves into the unified buffer (CPU writes, GPU will read)
        leaves.withUnsafeBytes { src in
            let dst = nodeBuffer.contents().advanced(by: (capacity + startIdx) * MemoryLayout<Fr>.stride)
            memcpy(dst, src.baseAddress!, batchCount * MemoryLayout<Fr>.stride)
        }
        count += batchCount

        // Mark dirty and rehash
        dirtyTracker.markRange(start: startIdx, count: batchCount, capacity: capacity)
        try rehashDirty()
        dirtyTracker.clear()
    }

    // MARK: - Update

    /// Update a single leaf at index. CPU rehashes the O(log n) path.
    public func update(index: Int, newLeaf: Fr) throws {
        guard index < count else {
            throw MSMError.gpuError("Leaf index \(index) out of range [0, \(count))")
        }
        setNode(capacity + index, newLeaf)
        rehashPathCPU(leafIndex: index)
    }

    /// Batch update multiple leaves. GPU re-hashes affected subtrees.
    public func batchUpdate(updates: [(index: Int, leaf: Fr)]) throws {
        guard !updates.isEmpty else { return }

        // Write new leaf values
        for (index, leaf) in updates {
            guard index < count else {
                throw MSMError.gpuError("Leaf index \(index) out of range [0, \(count))")
            }
            setNode(capacity + index, leaf)
            dirtyTracker.markDirty(leafIndex: index, capacity: capacity)
        }

        try rehashDirty()
        dirtyTracker.clear()
    }

    // MARK: - Root

    /// Current Merkle root (read from unified memory — zero copy).
    public var root: Fr {
        getNode(1)
    }

    // MARK: - Merkle Proof

    /// Generate a Merkle proof for the leaf at the given index.
    public func proof(index: Int) -> MerkleProof {
        precondition(index < count, "Leaf index out of range")
        var siblings = [Fr]()
        siblings.reserveCapacity(depth)
        var pathBits = [Bool]()
        pathBits.reserveCapacity(depth)

        var nodeIdx = capacity + index
        for _ in 0..<depth {
            let isRight = (nodeIdx & 1) == 1
            let siblingIdx = isRight ? nodeIdx - 1 : nodeIdx + 1
            siblings.append(getNode(siblingIdx))
            pathBits.append(isRight)
            nodeIdx >>= 1
        }

        return MerkleProof(siblings: siblings, pathBits: pathBits, leafIndex: index)
    }

    /// Verify a Merkle proof against a root.
    public static func verify(leaf: Fr, proof: MerkleProof, root: Fr) -> Bool {
        var current = leaf
        for i in 0..<proof.siblings.count {
            if proof.pathBits[i] {
                current = poseidon2Hash(proof.siblings[i], current)
            } else {
                current = poseidon2Hash(current, proof.siblings[i])
            }
        }
        return frEqual(current, root)
    }

    // MARK: - CPU Rehash (single path)

    /// Rehash a single path from leaf to root on CPU. O(depth) Poseidon2 hashes.
    private func rehashPathCPU(leafIndex: Int) {
        var nodeIdx = (capacity + leafIndex) >> 1
        for _ in 0..<depth {
            let left = getNode(nodeIdx * 2)
            let right = getNode(nodeIdx * 2 + 1)
            setNode(nodeIdx, poseidon2Hash(left, right))
            nodeIdx >>= 1
        }
    }

    // MARK: - GPU Rehash (dirty subtrees)

    /// Rehash all dirty nodes, level by level bottom-up.
    /// Uses GPU for levels with many dirty nodes, CPU for sparse levels.
    private func rehashDirty() throws {
        let stride = MemoryLayout<Fr>.stride

        // Process level by level, bottom up
        for level in 0..<depth {
            let dirtySet = dirtyTracker.dirtyByLevel[level]
            let dirtyCount = dirtySet.count
            if dirtyCount == 0 { continue }

            if dirtyCount < gpuThreshold {
                // CPU path for small number of dirty nodes
                for nodeIdx in dirtySet {
                    let left = getNode(nodeIdx * 2)
                    let right = getNode(nodeIdx * 2 + 1)
                    setNode(nodeIdx, poseidon2Hash(left, right))
                }
            } else {
                // GPU path: dispatch hash_pairs kernel on dirty nodes.
                // We need to set up input/output in the nodeBuffer.
                // Strategy: use encodeHashPairs for contiguous ranges,
                // or fall back to per-pair dispatch for scattered nodes.
                try rehashLevelGPU(level: level, dirtyNodes: dirtySet)
            }
        }
    }

    /// GPU rehash for a single level with many dirty nodes.
    /// Collects children into a temp buffer, hashes them, writes results back.
    private func rehashLevelGPU(level: Int, dirtyNodes: Set<Int>) throws {
        let dirtyCount = dirtyNodes.count
        let stride = MemoryLayout<Fr>.stride
        let sortedDirty = dirtyNodes.sorted()

        // Check if the dirty range is contiguous (common for batch appends)
        let isContiguous = sortedDirty.count > 1 &&
            sortedDirty.last! - sortedDirty.first! + 1 == sortedDirty.count

        if isContiguous {
            // Contiguous range: children are at [2*first, 2*first + 2*count)
            // This is a contiguous block in nodeBuffer — use encodeHashPairs directly.
            let firstParent = sortedDirty.first!
            let inputOffset = firstParent * 2 * stride  // children start
            let outputOffset = firstParent * stride      // parents start

            try rehashContiguousGPU(inputOffset: inputOffset,
                                     outputOffset: outputOffset,
                                     count: dirtyCount)
        } else {
            // Scattered dirty nodes: gather children into temp buffer, hash, scatter back.
            try rehashScatteredGPU(sortedDirty: sortedDirty)
        }
    }

    /// GPU rehash for contiguous parent range — zero-copy, no temp buffers.
    /// Children at nodeBuffer[2*firstParent..] are hashed to nodeBuffer[firstParent..].
    private func rehashContiguousGPU(inputOffset: Int, outputOffset: Int, count: Int) throws {
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        engine.encodeHashPairs(encoder: enc, buffer: nodeBuffer,
                                inputOffset: inputOffset,
                                outputOffset: outputOffset,
                                count: count)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// GPU rehash for scattered dirty nodes: gather children, hash, scatter results.
    private func rehashScatteredGPU(sortedDirty: [Int]) throws {
        let count = sortedDirty.count
        let stride = MemoryLayout<Fr>.stride

        // Allocate temp buffers for input pairs and output hashes
        let inputSize = count * 2 * stride
        let outputSize = count * stride

        guard let inputBuf = engine.device.makeBuffer(length: inputSize, options: .storageModeShared),
              let outputBuf = engine.device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate scatter buffers")
        }

        // Gather: copy children from nodeBuffer into contiguous inputBuf
        let srcPtr = nodePtr
        let dstPtr = inputBuf.contents().bindMemory(to: Fr.self, capacity: count * 2)
        for (i, parentIdx) in sortedDirty.enumerated() {
            dstPtr[i * 2] = srcPtr[parentIdx * 2]
            dstPtr[i * 2 + 1] = srcPtr[parentIdx * 2 + 1]
        }

        // GPU hash: use two-buffer variant
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(engine.hashPairsFunction)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(engine.rcBuffer, offset: 0, index: 2)
        var n = UInt32(count)
        enc.setBytes(&n, length: 4, index: 3)
        let tg = min(256, Int(engine.hashPairsFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Scatter: copy results back to nodeBuffer
        let resultPtr = outputBuf.contents().bindMemory(to: Fr.self, capacity: count)
        for (i, parentIdx) in sortedDirty.enumerated() {
            srcPtr[parentIdx] = resultPtr[i]
        }
    }

    // MARK: - Full Rebuild (for verification)

    /// Rebuild the entire tree from current leaves using the existing MerkleEngine.
    /// Returns the root. Used for correctness verification.
    public func fullRebuildRoot() throws -> Fr {
        let merkle = try Poseidon2MerkleEngine()
        // Collect current leaves (pad to power of 2 with zeros)
        let ptr = nodePtr
        var leaves = [Fr](repeating: Fr.zero, count: capacity)
        for i in 0..<capacity {
            leaves[i] = ptr[capacity + i]
        }
        return try merkle.merkleRoot(leaves)
    }

    // MARK: - Batch Append with Single Command Buffer

    /// Optimized batch append: uses a single GPU command buffer with memory barriers
    /// between levels. Avoids per-level command buffer overhead.
    public func appendBatchFused(leaves: [Fr]) throws {
        guard count + leaves.count <= capacity else {
            throw MSMError.gpuError("Batch append would exceed capacity")
        }
        let startIdx = count
        let batchCount = leaves.count

        // Write leaves into unified buffer
        leaves.withUnsafeBytes { src in
            let dst = nodeBuffer.contents().advanced(by: (capacity + startIdx) * MemoryLayout<Fr>.stride)
            memcpy(dst, src.baseAddress!, batchCount * MemoryLayout<Fr>.stride)
        }
        count += batchCount

        // Mark dirty
        dirtyTracker.markRange(start: startIdx, count: batchCount, capacity: capacity)

        // Check if all levels have contiguous ranges (true for appends)
        let allContiguous = (0..<depth).allSatisfy { level in
            let s = dirtyTracker.dirtyByLevel[level]
            guard s.count > 1 else { return true }
            let sorted = s.sorted()
            return sorted.last! - sorted.first! + 1 == sorted.count
        }

        if allContiguous && dirtyTracker.totalDirty >= gpuThreshold {
            try rehashDirtyFused()
        } else {
            try rehashDirty()
        }
        dirtyTracker.clear()
    }

    /// Single command buffer rehash for contiguous dirty ranges at each level.
    private func rehashDirtyFused() throws {
        let stride = MemoryLayout<Fr>.stride
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for level in 0..<depth {
            let dirtySet = dirtyTracker.dirtyByLevel[level]
            let dirtyCount = dirtySet.count
            if dirtyCount == 0 { continue }

            if level > 0 {
                enc.memoryBarrier(scope: .buffers)
            }

            if dirtyCount < gpuThreshold {
                // For very small levels near the root, CPU is faster.
                // But we're in a fused GPU path — just dispatch small kernels.
                // The overhead is minimal within an existing encoder.
            }

            let sorted = dirtySet.sorted()
            let firstParent = sorted.first!

            if sorted.last! - firstParent + 1 == dirtyCount {
                // Contiguous: direct hash in nodeBuffer
                let inputOffset = firstParent * 2 * stride
                let outputOffset = firstParent * stride
                engine.encodeHashPairs(encoder: enc, buffer: nodeBuffer,
                                        inputOffset: inputOffset,
                                        outputOffset: outputOffset,
                                        count: dirtyCount)
            } else {
                // Shouldn't happen for appends, but fall back to per-node CPU after GPU
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                // CPU fallback for remaining levels
                for remainingLevel in level..<depth {
                    for nodeIdx in dirtyTracker.dirtyByLevel[remainingLevel] {
                        let left = getNode(nodeIdx * 2)
                        let right = getNode(nodeIdx * 2 + 1)
                        setNode(nodeIdx, poseidon2Hash(left, right))
                    }
                }
                return
            }
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }
}
