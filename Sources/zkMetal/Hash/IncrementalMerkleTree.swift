// Incremental Merkle tree with GPU-accelerated rehashing via unified memory
// CPU appends leaves directly into a shared MTLBuffer; GPU re-hashes only dirty subtrees.
// Tree layout: 1-indexed heap. nodes[1] = root, nodes[2i] = left child, nodes[2i+1] = right child.
// Leaves occupy indices [capacity, 2*capacity). Internal nodes occupy [1, capacity).
// Zero (Fr.zero) is the default empty-leaf value.
//
// IMPORTANT: All hashing uses GPU Poseidon2 (encodeHashPairs) to ensure consistency.
// CPU poseidon2Hash differs from GPU for some inputs due to lazy vs. strict reduction
// in the external linear layer. Mixing CPU and GPU hashing produces inconsistent trees.

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

// MARK: - Dirty Level Info

/// Represents dirty nodes at one level: either a contiguous range or scattered indices.
enum DirtyLevel {
    case empty
    case contiguous(first: Int, count: Int)  // [first, first+count)
    case scattered(Set<Int>)

    var nodeCount: Int {
        switch self {
        case .empty: return 0
        case .contiguous(_, let count): return count
        case .scattered(let s): return s.count
        }
    }

    var isContiguous: Bool {
        switch self {
        case .empty, .contiguous: return true
        case .scattered: return false
        }
    }

    /// First node index (for contiguous ranges).
    var first: Int {
        switch self {
        case .empty: return 0
        case .contiguous(let f, _): return f
        case .scattered(let s): return s.min() ?? 0
        }
    }

    /// Sorted array of dirty indices (materializes for scattered case).
    var sortedIndices: [Int] {
        switch self {
        case .empty: return []
        case .contiguous(let first, let count):
            return Array(first..<(first + count))
        case .scattered(let s):
            return s.sorted()
        }
    }
}

// MARK: - Dirty Tracker

/// Tracks which internal nodes need rehashing after leaf mutations.
/// Organized by level (0 = parents of leaves, depth-1 = root).
/// Optimized: contiguous ranges (common for appends) avoid Set overhead.
struct DirtyTracker {
    let depth: Int
    var dirtyByLevel: [DirtyLevel]

    init(depth: Int) {
        self.depth = depth
        self.dirtyByLevel = Array(repeating: .empty, count: depth)
    }

    /// Mark a single leaf index (0-based) as dirty, propagating up all ancestors.
    mutating func markDirty(leafIndex: Int, capacity: Int) {
        var nodeIdx = (capacity + leafIndex) >> 1
        for level in 0..<depth {
            switch dirtyByLevel[level] {
            case .empty:
                dirtyByLevel[level] = .contiguous(first: nodeIdx, count: 1)
            case .contiguous(let first, let count):
                if nodeIdx >= first && nodeIdx < first + count {
                    // Already covered
                } else if nodeIdx == first + count {
                    dirtyByLevel[level] = .contiguous(first: first, count: count + 1)
                } else if nodeIdx == first - 1 {
                    dirtyByLevel[level] = .contiguous(first: nodeIdx, count: count + 1)
                } else {
                    // Can't extend contiguous -- switch to scattered
                    var s = Set<Int>(minimumCapacity: count + 1)
                    for i in first..<(first + count) { s.insert(i) }
                    s.insert(nodeIdx)
                    dirtyByLevel[level] = .scattered(s)
                }
            case .scattered(var s):
                s.insert(nodeIdx)
                dirtyByLevel[level] = .scattered(s)
            }
            nodeIdx >>= 1
        }
    }

    /// Mark a contiguous range of leaves [start, start+count) as dirty.
    /// Optimized: stores ranges directly without materializing all indices.
    mutating func markRange(start: Int, count: Int, capacity: Int) {
        var lo = (capacity + start) >> 1
        var hi = (capacity + start + count - 1) >> 1
        for level in 0..<depth {
            let rangeCount = hi - lo + 1
            switch dirtyByLevel[level] {
            case .empty:
                dirtyByLevel[level] = .contiguous(first: lo, count: rangeCount)
            case .contiguous(let first, let existCount):
                // Try to merge
                let newLo = min(first, lo)
                let newHi = max(first + existCount - 1, hi)
                let gap = newHi - newLo + 1
                // Check if merged range is at most 2x the sum of counts (allow some gaps)
                if gap <= existCount + rangeCount + 16 {
                    dirtyByLevel[level] = .contiguous(first: newLo, count: gap)
                } else {
                    var s = Set<Int>(minimumCapacity: existCount + rangeCount)
                    for i in first..<(first + existCount) { s.insert(i) }
                    for i in lo...hi { s.insert(i) }
                    dirtyByLevel[level] = .scattered(s)
                }
            case .scattered(var s):
                for i in lo...hi { s.insert(i) }
                dirtyByLevel[level] = .scattered(s)
            }
            lo >>= 1
            hi >>= 1
        }
    }

    var totalDirty: Int {
        dirtyByLevel.reduce(0) { $0 + $1.nodeCount }
    }

    mutating func clear() {
        for i in 0..<dirtyByLevel.count {
            dirtyByLevel[i] = .empty
        }
    }
}

// MARK: - Incremental Merkle Tree

/// Persistent Merkle tree that supports incremental leaf updates.
/// Uses Apple Silicon unified memory: tree nodes live in an MTLBuffer accessible by both CPU and GPU.
/// Poseidon2 2-to-1 hashing (BN254 Fr). All hashing uses GPU for consistency.
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
    /// Cached buffer for dirty node indices (GPU scattered update kernel)
    private var cachedDirtyIndicesBuf: MTLBuffer?
    private var cachedDirtyIndicesCap: Int = 0

    public init(depth: Int) throws {
        precondition(depth > 0 && depth <= 26, "Depth must be in [1, 26]")
        self.depth = depth
        self.capacity = 1 << depth
        self.engine = try Poseidon2Engine()
        self.dirtyTracker = DirtyTracker(depth: depth)

        let totalNodes = 2 * capacity  // index 0 unused, 1..2*capacity-1
        let frStride = MemoryLayout<Fr>.stride
        guard let buf = engine.device.makeBuffer(length: totalNodes * frStride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate incremental Merkle buffer (\(totalNodes) nodes)")
        }
        self.nodeBuffer = buf

        // Initialize all nodes to zero (default empty leaf)
        memset(buf.contents(), 0, totalNodes * frStride)

        // Pre-hash the empty tree using GPU
        try precomputeEmptyTree()
    }

    /// Initialize with an existing Poseidon2Engine (shares GPU resources).
    public init(depth: Int, engine: Poseidon2Engine) throws {
        precondition(depth > 0 && depth <= 26, "Depth must be in [1, 26]")
        self.depth = depth
        self.capacity = 1 << depth
        self.engine = engine
        self.dirtyTracker = DirtyTracker(depth: depth)

        let totalNodes = 2 * capacity
        let frStride = MemoryLayout<Fr>.stride
        guard let buf = engine.device.makeBuffer(length: totalNodes * frStride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate incremental Merkle buffer (\(totalNodes) nodes)")
        }
        self.nodeBuffer = buf
        memset(buf.contents(), 0, totalNodes * frStride)
        try precomputeEmptyTree()
    }

    // MARK: - Node Access (unified memory -- zero copy)

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

    // MARK: - Empty Tree (GPU)

    /// Pre-compute internal nodes for an all-zero tree so that root is valid from the start.
    /// Uses GPU hashing for consistency with all other hash operations.
    private func precomputeEmptyTree() throws {
        // All leaves are zero. Hash bottom-up level by level.
        // Each level has the same hash value since all siblings are identical.
        let frStride = MemoryLayout<Fr>.stride

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Level 0: hash pairs of leaves. All are (0, 0).
        // We only need to hash ONE pair, then fill the level.
        // But to use GPU consistently, hash all pairs at level 0.
        for level in 0..<depth {
            let levelStart = capacity >> (level + 1)
            let levelCount = capacity >> (level + 1)  // number of nodes at this level = capacity / 2^(level+1)
            if levelCount == 0 { break }

            if level > 0 {
                enc.memoryBarrier(scope: .buffers)
            }

            let inputOffset = levelStart * 2 * frStride  // children start at 2*levelStart
            let outputOffset = levelStart * frStride
            engine.encodeHashPairs(encoder: enc, buffer: nodeBuffer,
                                    inputOffset: inputOffset,
                                    outputOffset: outputOffset,
                                    count: levelCount)
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Append

    /// Append a single leaf. Uses GPU fused rehash (same path as batch append).
    public func append(leaf: Fr) throws {
        try appendBatchFused(leaves: [leaf])
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

    /// Update a single leaf at index. GPU rehashes the affected path.
    public func update(index: Int, newLeaf: Fr) throws {
        try batchUpdate(updates: [(index: index, leaf: newLeaf)])
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

    /// Current Merkle root (read from unified memory -- zero copy).
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

    /// Verify a Merkle proof against a root using GPU Poseidon2 hashing.
    /// Uses GPU hashing to be consistent with how the tree was built.
    public func verifyProof(leaf: Fr, proof: MerkleProof) throws -> Bool {
        let pairs = proof.siblings.count
        guard pairs > 0 else { return false }

        // Build input pairs for GPU hashing, one level at a time
        var current = leaf
        for i in 0..<pairs {
            let left: Fr
            let right: Fr
            if proof.pathBits[i] {
                left = proof.siblings[i]
                right = current
            } else {
                left = current
                right = proof.siblings[i]
            }
            let result = try engine.hashPairs([left, right])
            current = result[0]
        }
        return frEqual(current, root)
    }

    /// Verify a Merkle proof against a root (static, uses CPU hash).
    /// NOTE: This may not match GPU-hashed trees due to CPU/GPU hash divergence.
    /// Prefer the instance method verifyProof() for trees built with this class.
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

    // MARK: - GPU Rehash (dirty subtrees)

    /// Rehash all dirty nodes, level by level bottom-up.
    /// All hashing uses GPU for consistency. Both contiguous and scattered cases
    /// stay in a single command buffer using the scattered update kernel.
    private func rehashDirty() throws {
        let frStride = MemoryLayout<Fr>.stride

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for level in 0..<depth {
            let dirty = dirtyTracker.dirtyByLevel[level]
            let dirtyCount = dirty.nodeCount
            if dirtyCount == 0 { continue }

            if level > 0 { enc.memoryBarrier(scope: .buffers) }

            if dirty.isContiguous {
                let firstParent = dirty.first
                let inputOffset = firstParent * 2 * frStride
                let outputOffset = firstParent * frStride
                engine.encodeHashPairs(encoder: enc, buffer: nodeBuffer,
                                        inputOffset: inputOffset,
                                        outputOffset: outputOffset,
                                        count: dirtyCount)
            } else {
                // Scattered: use GPU kernel that reads/writes the heap in-place
                encodeScatteredUpdate(encoder: enc, sortedDirty: dirty.sortedIndices)
            }
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Ensure the cached dirty indices buffer can hold `count` UInt32 values.
    private func ensureDirtyIndicesBuf(count: Int) -> MTLBuffer {
        if count > cachedDirtyIndicesCap {
            let newCap = max(count, 256)  // minimum 256 to avoid frequent realloc
            cachedDirtyIndicesBuf = engine.device.makeBuffer(length: newCap * MemoryLayout<UInt32>.stride,
                                                              options: .storageModeShared)
            cachedDirtyIndicesCap = newCap
        }
        return cachedDirtyIndicesBuf!
    }

    /// Encode scattered update into an existing compute encoder.
    /// Writes dirty indices into a GPU buffer, then dispatches the scattered update kernel.
    private func encodeScatteredUpdate(encoder: MTLComputeCommandEncoder, sortedDirty: [Int]) {
        let count = sortedDirty.count
        guard count > 0 else { return }

        let indicesBuf = ensureDirtyIndicesBuf(count: count)
        let ptr = indicesBuf.contents().bindMemory(to: UInt32.self, capacity: count)
        for (i, idx) in sortedDirty.enumerated() {
            ptr[i] = UInt32(idx)
        }

        engine.encodeScatteredUpdate(encoder: encoder,
                                      treeBuf: nodeBuffer,
                                      dirtyIndicesBuf: indicesBuf,
                                      count: count)
    }

    /// GPU rehash for scattered dirty nodes using the dedicated scattered update kernel.
    /// Each GPU thread reads children from the 1-indexed heap and writes the hash back in-place.
    private func rehashScatteredGPU(sortedDirty: [Int]) throws {
        let count = sortedDirty.count
        guard count > 0 else { return }

        let indicesBuf = ensureDirtyIndicesBuf(count: count)
        let ptr = indicesBuf.contents().bindMemory(to: UInt32.self, capacity: count)
        for (i, idx) in sortedDirty.enumerated() {
            ptr[i] = UInt32(idx)
        }

        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        engine.encodeScatteredUpdate(encoder: enc,
                                      treeBuf: nodeBuffer,
                                      dirtyIndicesBuf: indicesBuf,
                                      count: count)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
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

        // Always use GPU fused path for consistency
        try rehashDirtyFused()
        dirtyTracker.clear()
    }

    /// Single command buffer rehash for all dirty ranges at each level.
    /// Both contiguous and scattered cases stay in one command buffer using the
    /// scattered update kernel (reads/writes the 1-indexed heap directly on GPU).
    private func rehashDirtyFused() throws {
        let frStride = MemoryLayout<Fr>.stride
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for level in 0..<depth {
            let dirty = dirtyTracker.dirtyByLevel[level]
            let dirtyCount = dirty.nodeCount
            if dirtyCount == 0 { continue }

            if level > 0 {
                enc.memoryBarrier(scope: .buffers)
            }

            if dirty.isContiguous {
                // Contiguous: use encodeHashPairs (reads consecutive pairs, writes consecutive outputs)
                let firstParent = dirty.first
                let inputOffset = firstParent * 2 * frStride
                let outputOffset = firstParent * frStride
                engine.encodeHashPairs(encoder: enc, buffer: nodeBuffer,
                                        inputOffset: inputOffset,
                                        outputOffset: outputOffset,
                                        count: dirtyCount)
            } else {
                // Scattered: use the dedicated GPU kernel that reads/writes the heap in-place.
                // Each GPU thread independently hashes children of one dirty parent node.
                encodeScatteredUpdate(encoder: enc, sortedDirty: dirty.sortedIndices)
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
