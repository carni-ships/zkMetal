// GPU-accelerated Merkle tree construction
// Supports both Poseidon2 (field-native) and Keccak-256 hash functions.
import Foundation
import Metal

// MARK: - Poseidon2 Merkle Tree

public class Poseidon2MerkleEngine {
    public static let version = Versions.poseidon2Merkle
    private let engine: Poseidon2Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0
    private var cachedLevelOffsetsBuf: MTLBuffer?
    private var cachedRootBuf: MTLBuffer?
    private var treeSize2: Int = 0

    /// Expose engine for advanced encoding
    public var p2Engine: Poseidon2Engine { engine }

    public init() throws {
        self.engine = try Poseidon2Engine()
    }

    /// Encode Merkle root computation into an existing encoder.
    /// treeBuf layout: leaves at [treeOffset, treeOffset + n*stride),
    /// internal nodes at [treeOffset + n*stride, treeOffset + (2n-1)*stride).
    /// After CB completes, root is at treeBuf offset treeOffset + (2n-2)*stride.
    /// Caller must add memoryBarrier before/after if needed.
    public func encodeMerkleRoot(encoder: MTLComputeCommandEncoder,
                                  treeBuf: MTLBuffer, treeOffset: Int,
                                  n: Int) {
        let stride = MemoryLayout<Fr>.stride
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024

        if n >= 2 && n <= subtreeSize && (n & (n - 1)) == 0 {
            // Small tree: single fused dispatch covers all levels
            let rootOffset = treeOffset + (2 * n - 2) * stride
            engine.encodeMerkleFused(encoder: encoder,
                                      leavesBuffer: treeBuf, leavesOffset: treeOffset,
                                      rootsBuffer: treeBuf, rootsOffset: rootOffset,
                                      numSubtrees: 1, subtreeSize: n)
        } else if n >= subtreeSize && n <= 65536 {
            let numSubtrees = n / subtreeSize
            let rootsOffset = treeOffset + n * stride
            engine.encodeMerkleFused(encoder: encoder,
                                      leavesBuffer: treeBuf, leavesOffset: treeOffset,
                                      rootsBuffer: treeBuf, rootsOffset: rootsOffset,
                                      numSubtrees: numSubtrees)

            var levelStart = n
            var levelSize = numSubtrees
            while levelSize > 1 {
                encoder.memoryBarrier(scope: .buffers)
                // Fuse remaining upper levels if they fit in one subtree
                if levelSize >= 2 && levelSize <= subtreeSize && (levelSize & (levelSize - 1)) == 0 {
                    let rootOffset = treeOffset + (2 * n - 2) * stride
                    engine.encodeMerkleFused(encoder: encoder,
                                              leavesBuffer: treeBuf, leavesOffset: treeOffset + levelStart * stride,
                                              rootsBuffer: treeBuf, rootsOffset: rootOffset,
                                              numSubtrees: 1, subtreeSize: levelSize)
                    break
                }
                let parentCount = levelSize / 2
                let inputOffset = treeOffset + levelStart * stride
                let outputOffset = treeOffset + (levelStart + levelSize) * stride
                engine.encodeHashPairs(encoder: encoder, buffer: treeBuf,
                                       inputOffset: inputOffset,
                                       outputOffset: outputOffset,
                                       count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            }
        } else {
            // Large tree: level-by-level
            var levelStart = 0
            var levelSize = n
            while levelSize > 1 {
                let parentCount = levelSize / 2
                let inputOffset = treeOffset + levelStart * stride
                let outputOffset = treeOffset + (levelStart + levelSize) * stride
                engine.encodeHashPairs(encoder: encoder, buffer: treeBuf,
                                       inputOffset: inputOffset,
                                       outputOffset: outputOffset,
                                       count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
                if levelSize > 1 { encoder.memoryBarrier(scope: .buffers) }
            }
        }
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
        // Tested: fused is 1.7x faster than level-by-level at 2^16 — shared memory
        // saves enough global memory round-trips to outweigh 80% thread waste.
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
    public static let version = Versions.keccakMerkle
    public let engine: Keccak256Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0
    private var cachedM31Buf: MTLBuffer?
    private var cachedM31BufCount: Int = 0

    public init() throws {
        self.engine = try Keccak256Engine()
    }

    /// Build a Merkle tree from 32-byte leaf hashes using Keccak-256.
    /// Returns flat buffer: treeSize * 32 bytes. Node i is at bytes [i*32..<(i+1)*32].
    /// Layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal, node[2n-2] = root.
    public func buildTree(_ leaves: [[UInt8]]) throws -> [UInt8] {
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

        // Level-by-level construction (preserves ALL internal nodes for proof extraction).
        // Note: fused subtrees only output roots, not intermediate nodes, so they
        // cannot be used here. Fused subtrees ARE used in encodeMerkleRoot.
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
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Single flat copy — no per-node slicing
        let readPtr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
    }

    /// Access a specific node from a flat tree buffer returned by buildTree.
    /// Node index 0..n-1 = leaves, n..2n-2 = internal nodes, 2n-2 = root.
    public static func node(_ tree: [UInt8], at index: Int) -> [UInt8] {
        let start = index * 32
        return Array(tree[start..<start + 32])
    }

    /// Get the Merkle root from 32-byte leaf hashes.
    /// Optimized: avoids copying the full tree back from GPU — only reads the 32-byte root.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
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
        let subtreeSize = Keccak256Engine.merkleSubtreeSize

        if n >= subtreeSize {
            let numSubtrees = n / subtreeSize
            let fusedOutputOffset = (2 * n - 2 * numSubtrees) * 32

            engine.encodeMerkleFused(encoder: enc,
                                      leavesBuffer: treeBuf, leavesOffset: 0,
                                      rootsBuffer: treeBuf, rootsOffset: fusedOutputOffset,
                                      numSubtrees: numSubtrees)

            var levelStart = 2 * n - 2 * numSubtrees
            var levelSize = numSubtrees

            while levelSize > 1 {
                enc.memoryBarrier(scope: .buffers)
                if levelSize >= 4 && levelSize <= 1024 && (levelSize & (levelSize - 1)) == 0 {
                    let outputOffset = (treeSize - 1) * 32
                    engine.encodeMerkleFused(encoder: enc,
                                              leavesBuffer: treeBuf, leavesOffset: levelStart * 32,
                                              rootsBuffer: treeBuf, rootsOffset: outputOffset,
                                              numSubtrees: 1, subtreeSize: levelSize)
                    break
                }
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
                if levelSize > 1 { enc.memoryBarrier(scope: .buffers) }
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Only read the 32-byte root
        let rootPtr = treeBuf.contents().advanced(by: (treeSize - 1) * 32).assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: rootPtr, count: 32))
    }
    /// Build a Merkle tree from M31 values: GPU hash (4-byte → 32-byte) + GPU tree.
    /// Returns flat tree bytes. Layout: nodes[0..<n] = leaf hashes, nodes[n..<2n-1] = internal, root at 2n-2.
    /// Use `node(_:at:)` to extract individual nodes, `merkleProofFlat` for auth paths.
    public func buildTreeFromM31(_ values: [UInt32], count n: Int) throws -> [UInt8] {
        precondition(n > 0 && (n & (n - 1)) == 0)
        let treeSize = 2 * n - 1

        // Ensure M31 input buffer
        if n > cachedM31BufCount {
            guard let buf = engine.device.makeBuffer(length: n * 4, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate M31 input buffer")
            }
            cachedM31Buf = buf
            cachedM31BufCount = n
        }
        memcpy(cachedM31Buf!.contents(), values, n * 4)

        // Ensure tree buffer
        if treeSize > cachedTreeBufNodes {
            guard let buf = engine.device.makeBuffer(length: treeSize * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate Keccak Merkle buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufNodes = treeSize
        }
        let treeBuf = cachedTreeBuf!

        // Single command buffer: hash M31 leaves → build Merkle tree
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Phase 0: Hash M31 values → 32-byte leaf hashes at tree[0..<n]
        engine.encodeHashM31(encoder: enc,
                              inputBuffer: cachedM31Buf!, inputOffset: 0,
                              outputBuffer: treeBuf, outputOffset: 0,
                              count: n)
        enc.memoryBarrier(scope: .buffers)

        // Build Merkle tree level-by-level (preserves ALL internal nodes for proof extraction)
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
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let readPtr = treeBuf.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
    }

    /// Extract Merkle authentication path from flat GPU tree layout.
    /// Layout: nodes[0..<n] = leaves, nodes[n..<2n-1] = internal (level-by-level bottom-up).
    public static func merkleProofFlat(_ tree: [UInt8], n: Int, index: Int) -> [[UInt8]] {
        var path = [[UInt8]]()
        var levelStart = 0
        var levelSize = n
        var idx = index

        while levelSize > 1 {
            let sibling = idx ^ 1
            let start = (levelStart + sibling) * 32
            path.append(Array(tree[start..<start + 32]))
            levelStart += levelSize
            levelSize /= 2
            idx /= 2
        }
        return path
    }

    /// Get root from flat tree. Root is at node index 2n-2.
    public static func rootFromFlat(_ tree: [UInt8], n: Int) -> [UInt8] {
        let rootIdx = 2 * n - 2
        let start = rootIdx * 32
        return Array(tree[start..<start + 32])
    }
}

// MARK: - Keccak 4-ary Merkle Tree

/// GPU-accelerated 4-ary Merkle tree using Keccak-256.
/// Each internal node hashes 4 children (128 bytes) instead of 2 (64 bytes).
/// This halves the tree depth: log4(n) = log2(n)/2 levels, halving GPU dispatches.
///
/// Tree layout (flat array, 0-indexed):
///   nodes[0..<n]       = leaves (32 bytes each)
///   Level k (k=0 is leaves): nodes at offsets computed from 4-ary fan-in
///   Root is the final single node.
///
/// Node count for n leaves (power of 4):
///   n + n/4 + n/16 + ... + 1 = n * (4/3) + rounding = (4*n - 1) / 3
///
/// For n that is a power of 2 but not power of 4 (e.g. n=2^odd):
///   The first level uses binary hashing (pairs), subsequent levels use 4-ary.
///   This handles all power-of-2 leaf counts.
public class Keccak4aryMerkleEngine {
    public static let version = "keccak4ary-merkle-v1"
    private let engine: Keccak256Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufSize: Int = 0

    public init() throws {
        self.engine = try Keccak256Engine()
    }

    /// Compute the total number of nodes in a 4-ary Merkle tree.
    /// For n leaves (power of 2), returns the total node count including leaves.
    private static func treeNodeCount(_ n: Int) -> Int {
        var total = n
        var levelSize = n
        while levelSize > 1 {
            if levelSize >= 4 {
                levelSize /= 4
            } else {
                // levelSize == 2: binary pair at top
                levelSize /= 2
            }
            total += levelSize
        }
        return total
    }

    /// Build a 4-ary Merkle tree from 32-byte leaf hashes using Keccak-256.
    /// Leaves count must be a power of 2 (>= 4 for pure 4-ary, == 2 uses one binary level).
    /// Returns flat byte buffer with all nodes. Root is at the end.
    ///
    /// Layout: nodes are stored level by level (leaves first, then each subsequent level).
    public func buildTree(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")
        precondition(leaves.allSatisfy { $0.count == 32 }, "Leaves must be 32 bytes each")

        let treeSize = Keccak4aryMerkleEngine.treeNodeCount(n)
        let bufSize = treeSize * 32

        if bufSize > cachedTreeBufSize {
            guard let buf = engine.device.makeBuffer(length: bufSize, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate 4-ary Merkle buffer")
            }
            cachedTreeBuf = buf
            cachedTreeBufSize = bufSize
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

        var levelStart = 0
        var levelSize = n

        while levelSize > 1 {
            let outputOffset = (levelStart + levelSize) * 32

            if levelSize >= 4 {
                // 4-ary: hash groups of 4 children into 1 parent
                let parentCount = levelSize / 4
                let inputOffset = levelStart * 32
                engine.encodeHash128(encoder: enc, buffer: treeBuf,
                                      inputOffset: inputOffset,
                                      outputOffset: outputOffset,
                                      count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            } else {
                // levelSize == 2: binary hash (fallback for non-power-of-4)
                let parentCount = levelSize / 2
                let inputOffset = levelStart * 32
                engine.encodeHash64(encoder: enc, buffer: treeBuf,
                                     inputOffset: inputOffset,
                                     outputOffset: outputOffset,
                                     count: parentCount)
                levelStart += levelSize
                levelSize = parentCount
            }

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
        return Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
    }

    /// Get the Merkle root from 32-byte leaf hashes using 4-ary tree.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let tree = try buildTree(leaves)
        let treeSize = tree.count / 32
        let start = (treeSize - 1) * 32
        return Array(tree[start..<start + 32])
    }

    /// Access a specific node from a flat tree buffer.
    public static func node(_ tree: [UInt8], at index: Int) -> [UInt8] {
        let start = index * 32
        return Array(tree[start..<start + 32])
    }

    /// Compute the root node index in the flat tree layout.
    public static func rootIndex(leafCount n: Int) -> Int {
        return treeNodeCount(n) - 1
    }
}

// MARK: - Blake3 Merkle Tree

public class Blake3MerkleEngine {
    public static let version = Versions.blake3Merkle
    private let engine: Blake3Engine
    private var cachedTreeBuf: MTLBuffer?
    private var cachedTreeBufNodes: Int = 0

    public init() throws {
        self.engine = try Blake3Engine()
    }

    /// Build a Merkle tree from 32-byte leaf hashes using Blake3 parent compression.
    /// Returns flat buffer: treeSize * 32 bytes. Node i is at bytes [i*32..<(i+1)*32].
    public func buildTree(_ leaves: [[UInt8]]) throws -> [UInt8] {
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
        return Array(UnsafeBufferPointer(start: readPtr, count: treeSize * 32))
    }

    /// Access a specific node from a flat tree buffer returned by buildTree.
    public static func node(_ tree: [UInt8], at index: Int) -> [UInt8] {
        let start = index * 32
        return Array(tree[start..<start + 32])
    }

    /// Get the Merkle root from 32-byte leaf hashes.
    public func merkleRoot(_ leaves: [[UInt8]]) throws -> [UInt8] {
        let tree = try buildTree(leaves)
        let treeSize = tree.count / 32
        return Blake3MerkleEngine.node(tree, at: treeSize - 1)
    }
}
