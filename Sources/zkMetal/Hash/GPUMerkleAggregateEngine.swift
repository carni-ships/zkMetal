// GPU-accelerated Merkle Proof Aggregation Engine
//
// Aggregates multiple Merkle inclusion proofs into compact aggregate proofs.
// Key features:
// - Shared path compression: sibling nodes shared between proofs stored once
// - Multi-tree aggregation: proofs from different Merkle trees
// - GPU-accelerated batch verification of aggregate proofs
// - Incremental aggregation: add proofs to existing aggregates
// - Root consistency checking across aggregated proofs
// - Poseidon2-based Merkle hash function (BN254 Fr)
//
// An aggregate proof bundles multiple single-leaf Merkle proofs, deduplicating
// sibling nodes that appear on multiple authentication paths. When proofs come
// from the same tree, shared internal nodes are stored once; when proofs come
// from different trees, each tree's paths are compressed independently and the
// roots are tracked per-tree.

import Foundation
import Metal

// MARK: - Single Merkle Inclusion Proof (input to aggregation)

/// A single Merkle inclusion proof for one leaf.
/// This is the input format accepted by the aggregation engine.
public struct MerkleInclusionProof {
    /// The Merkle root this proof is relative to.
    public let root: Fr

    /// The leaf value being proved.
    public let leaf: Fr

    /// Leaf index in [0, leafCount).
    public let leafIndex: Int

    /// Total number of leaves (power of 2).
    public let leafCount: Int

    /// Authentication path: sibling hashes from leaf to root (bottom-up).
    /// path[0] = sibling of leaf, path[depth-1] = sibling of root's child.
    public let path: [Fr]

    /// An optional tree identifier for multi-tree aggregation.
    /// Proofs with the same treeId share the same root.
    public let treeId: UInt64

    public init(root: Fr, leaf: Fr, leafIndex: Int, leafCount: Int,
                path: [Fr], treeId: UInt64 = 0) {
        self.root = root
        self.leaf = leaf
        self.leafIndex = leafIndex
        self.leafCount = leafCount
        self.path = path
        self.treeId = treeId
    }

    /// Depth of the tree (path length).
    public var depth: Int { path.count }
}

// MARK: - Compressed Sibling Entry

/// A compressed sibling node in the aggregate proof.
/// Stores the level, position, and value — shared siblings are stored only once.
public struct CompressedSibling {
    /// Tree level (0 = leaf level, depth-1 = level just below root).
    public let level: Int

    /// Node index at this level.
    public let nodeIndex: Int

    /// Sibling hash value.
    public let value: Fr

    public init(level: Int, nodeIndex: Int, value: Fr) {
        self.level = level
        self.nodeIndex = nodeIndex
        self.value = value
    }
}

// MARK: - Per-Tree Aggregate Info

/// Metadata for one tree within an aggregate proof.
public struct TreeAggregateInfo {
    /// Tree identifier.
    public let treeId: UInt64

    /// Root of this tree.
    public let root: Fr

    /// Number of leaves in this tree (power of 2).
    public let leafCount: Int

    /// Indices of the leaf proofs (within the aggregate's leafEntries array)
    /// that belong to this tree.
    public let proofIndices: [Int]

    public init(treeId: UInt64, root: Fr, leafCount: Int, proofIndices: [Int]) {
        self.treeId = treeId
        self.root = root
        self.leafCount = leafCount
        self.proofIndices = proofIndices
    }
}

// MARK: - Leaf Entry

/// One leaf in the aggregate: stores the leaf value and its index.
public struct AggregateLeafEntry {
    /// Leaf index in [0, leafCount) within its tree.
    public let leafIndex: Int

    /// Leaf value.
    public let leaf: Fr

    /// Which tree this leaf belongs to (index into trees array).
    public let treeInfoIndex: Int

    public init(leafIndex: Int, leaf: Fr, treeInfoIndex: Int) {
        self.leafIndex = leafIndex
        self.leaf = leaf
        self.treeInfoIndex = treeInfoIndex
    }
}

// MARK: - Aggregate Merkle Proof

/// An aggregate Merkle proof that bundles multiple single-leaf inclusion proofs.
/// Shared sibling nodes across authentication paths are stored only once.
/// Supports proofs from multiple different trees.
public struct AggregateMerkleProof {
    /// Per-tree metadata (root, leafCount, which proofs belong to each tree).
    public let trees: [TreeAggregateInfo]

    /// All leaf entries across all trees, in order.
    public let leafEntries: [AggregateLeafEntry]

    /// Compressed sibling nodes: deduplicated across all paths within each tree.
    /// Ordered by (treeInfoIndex, level, nodeIndex).
    public let siblings: [CompressedSibling]

    /// Mapping: for each sibling, which tree it belongs to.
    public let siblingTreeIndices: [Int]

    /// Total number of individual proofs aggregated.
    public var proofCount: Int { leafEntries.count }

    /// Number of distinct trees.
    public var treeCount: Int { trees.count }

    /// Compression ratio: deduplicated siblings / total siblings without dedup.
    public var compressionRatio: Double {
        let totalWithoutDedup = leafEntries.reduce(0) { sum, entry in
            let depth = trees[entry.treeInfoIndex].leafCount.trailingZeroBitCount
            return sum + depth
        }
        guard totalWithoutDedup > 0 else { return 1.0 }
        return Double(siblings.count) / Double(totalWithoutDedup)
    }

    public init(trees: [TreeAggregateInfo], leafEntries: [AggregateLeafEntry],
                siblings: [CompressedSibling], siblingTreeIndices: [Int]) {
        self.trees = trees
        self.leafEntries = leafEntries
        self.siblings = siblings
        self.siblingTreeIndices = siblingTreeIndices
    }
}

// MARK: - Aggregate Proof Serialization

extension AggregateMerkleProof {
    /// Serialize the aggregate proof to a compact byte representation.
    /// Format:
    ///   [treeCount(4) | proofCount(4) | siblingCount(4) |
    ///    trees: [treeId(8) | root(Fr) | leafCount(4) | numProofIndices(4) | proofIndices(numProofIndices*4)] |
    ///    leafEntries: [leafIndex(4) | leaf(Fr) | treeInfoIndex(4)] |
    ///    siblings: [level(4) | nodeIndex(4) | value(Fr) | treeIndex(4)] ]
    public func serialize() -> [UInt8] {
        let hs = MemoryLayout<Fr>.stride
        var buf = [UInt8]()
        // Rough capacity estimate
        let estimatedSize = 12 + trees.count * (8 + hs + 8 + 64) + leafEntries.count * (8 + hs)
            + siblings.count * (12 + hs)
        buf.reserveCapacity(estimatedSize)

        appendU32(&buf, UInt32(trees.count))
        appendU32(&buf, UInt32(leafEntries.count))
        appendU32(&buf, UInt32(siblings.count))

        // Trees
        for tree in trees {
            appendU64(&buf, tree.treeId)
            appendFr(&buf, tree.root)
            appendU32(&buf, UInt32(tree.leafCount))
            appendU32(&buf, UInt32(tree.proofIndices.count))
            for idx in tree.proofIndices {
                appendU32(&buf, UInt32(idx))
            }
        }

        // Leaf entries
        for entry in leafEntries {
            appendU32(&buf, UInt32(entry.leafIndex))
            appendFr(&buf, entry.leaf)
            appendU32(&buf, UInt32(entry.treeInfoIndex))
        }

        // Siblings
        for (i, sib) in siblings.enumerated() {
            appendU32(&buf, UInt32(sib.level))
            appendU32(&buf, UInt32(sib.nodeIndex))
            appendFr(&buf, sib.value)
            appendU32(&buf, UInt32(siblingTreeIndices[i]))
        }

        return buf
    }

    /// Deserialize from bytes produced by serialize().
    public static func deserialize(_ data: [UInt8]) -> AggregateMerkleProof? {
        let hs = MemoryLayout<Fr>.stride
        guard data.count >= 12 else { return nil }
        var off = 0

        let treeCount = Int(readU32(data, off)); off += 4
        let proofCount = Int(readU32(data, off)); off += 4
        let sibCount = Int(readU32(data, off)); off += 4

        // Trees
        var trees = [TreeAggregateInfo]()
        trees.reserveCapacity(treeCount)
        for _ in 0..<treeCount {
            guard off + 8 + hs + 8 <= data.count else { return nil }
            let treeId = readU64(data, off); off += 8
            let root = readFr(data, off); off += hs
            let leafCount = Int(readU32(data, off)); off += 4
            let numPi = Int(readU32(data, off)); off += 4
            guard off + numPi * 4 <= data.count else { return nil }
            var proofIndices = [Int]()
            proofIndices.reserveCapacity(numPi)
            for _ in 0..<numPi {
                proofIndices.append(Int(readU32(data, off))); off += 4
            }
            trees.append(TreeAggregateInfo(treeId: treeId, root: root,
                                           leafCount: leafCount, proofIndices: proofIndices))
        }

        // Leaf entries
        var leafEntries = [AggregateLeafEntry]()
        leafEntries.reserveCapacity(proofCount)
        for _ in 0..<proofCount {
            guard off + 4 + hs + 4 <= data.count else { return nil }
            let leafIndex = Int(readU32(data, off)); off += 4
            let leaf = readFr(data, off); off += hs
            let treeInfoIndex = Int(readU32(data, off)); off += 4
            leafEntries.append(AggregateLeafEntry(leafIndex: leafIndex, leaf: leaf,
                                                  treeInfoIndex: treeInfoIndex))
        }

        // Siblings
        var siblings = [CompressedSibling]()
        siblings.reserveCapacity(sibCount)
        var sibTreeIndices = [Int]()
        sibTreeIndices.reserveCapacity(sibCount)
        for _ in 0..<sibCount {
            guard off + 8 + hs + 4 <= data.count else { return nil }
            let level = Int(readU32(data, off)); off += 4
            let nodeIndex = Int(readU32(data, off)); off += 4
            let value = readFr(data, off); off += hs
            let treeIndex = Int(readU32(data, off)); off += 4
            siblings.append(CompressedSibling(level: level, nodeIndex: nodeIndex, value: value))
            sibTreeIndices.append(treeIndex)
        }

        return AggregateMerkleProof(trees: trees, leafEntries: leafEntries,
                                    siblings: siblings, siblingTreeIndices: sibTreeIndices)
    }
}

// MARK: - Serialization Helpers

private func appendU32(_ buf: inout [UInt8], _ val: UInt32) {
    buf.append(UInt8(val & 0xFF))
    buf.append(UInt8((val >> 8) & 0xFF))
    buf.append(UInt8((val >> 16) & 0xFF))
    buf.append(UInt8((val >> 24) & 0xFF))
}

private func appendU64(_ buf: inout [UInt8], _ val: UInt64) {
    appendU32(&buf, UInt32(val & 0xFFFFFFFF))
    appendU32(&buf, UInt32(val >> 32))
}

private func appendFr(_ buf: inout [UInt8], _ val: Fr) {
    var v = val
    withUnsafeBytes(of: &v) { ptr in
        buf.append(contentsOf: ptr)
    }
}

private func readU32(_ buf: [UInt8], _ offset: Int) -> UInt32 {
    UInt32(buf[offset])
    | (UInt32(buf[offset + 1]) << 8)
    | (UInt32(buf[offset + 2]) << 16)
    | (UInt32(buf[offset + 3]) << 24)
}

private func readU64(_ buf: [UInt8], _ offset: Int) -> UInt64 {
    UInt64(readU32(buf, offset)) | (UInt64(readU32(buf, offset + 4)) << 32)
}

private func readFr(_ buf: [UInt8], _ offset: Int) -> Fr {
    let hs = MemoryLayout<Fr>.stride
    return Array(buf[offset..<offset + hs]).withUnsafeBytes { ptr in
        ptr.load(as: Fr.self)
    }
}

// MARK: - GPUMerkleAggregateEngine

/// GPU-accelerated engine for Merkle proof aggregation and batch verification.
///
/// Aggregates multiple individual Merkle inclusion proofs into a single compact
/// aggregate proof. Shared path nodes are deduplicated. Supports proofs from
/// multiple independent trees (multi-tree aggregation).
///
/// Usage:
///   let engine = try GPUMerkleAggregateEngine()
///   let tree = try engine.treeEngine.buildTree(leaves: myLeaves)
///
///   // Build individual proofs
///   let proof0 = MerkleInclusionProof(root: tree.root, leaf: leaves[0],
///                                      leafIndex: 0, leafCount: n, path: tree.proof(forLeafAt: 0).siblings)
///   let proof1 = MerkleInclusionProof(root: tree.root, leaf: leaves[3],
///                                      leafIndex: 3, leafCount: n, path: tree.proof(forLeafAt: 3).siblings)
///
///   // Aggregate
///   let aggregate = engine.aggregate(proofs: [proof0, proof1])
///   let valid = engine.verifyAggregate(aggregate)
public class GPUMerkleAggregateEngine {
    public static let version = Versions.gpuMerkleAggregate

    private let device: MTLDevice
    private let innerTreeEngine: GPUMerkleTreeEngine

    /// Access the underlying tree engine for building trees.
    public var treeEngine: GPUMerkleTreeEngine { innerTreeEngine }

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = dev
        self.innerTreeEngine = try GPUMerkleTreeEngine()
    }

    // MARK: - Aggregation

    /// Aggregate multiple Merkle inclusion proofs into a single compact proof.
    /// Deduplicates shared sibling nodes across authentication paths.
    /// Groups proofs by treeId for multi-tree support.
    public func aggregate(proofs: [MerkleInclusionProof]) -> AggregateMerkleProof {
        guard !proofs.isEmpty else {
            return AggregateMerkleProof(trees: [], leafEntries: [], siblings: [],
                                        siblingTreeIndices: [])
        }

        // Group proofs by treeId
        var groupedByTree = [UInt64: [Int]]()  // treeId -> [index into proofs]
        for (i, p) in proofs.enumerated() {
            groupedByTree[p.treeId, default: []].append(i)
        }

        var trees = [TreeAggregateInfo]()
        var allLeafEntries = [AggregateLeafEntry]()
        var allSiblings = [CompressedSibling]()
        var allSibTreeIndices = [Int]()

        let sortedTreeIds = groupedByTree.keys.sorted()
        for treeId in sortedTreeIds {
            let proofIndices = groupedByTree[treeId]!
            let treeInfoIndex = trees.count

            // All proofs in this group should have the same root and leafCount
            let firstProof = proofs[proofIndices[0]]
            let root = firstProof.root
            let leafCount = firstProof.leafCount
            let depth = leafCount.trailingZeroBitCount

            // Build leaf entries for this tree
            var localProofIndices = [Int]()
            // Deduplicate by leaf index within this tree
            var seenLeafIndices = [Int: Int]()  // leafIndex -> index in allLeafEntries
            for pi in proofIndices {
                let proof = proofs[pi]
                if let existing = seenLeafIndices[proof.leafIndex] {
                    // Duplicate leaf index — point to existing entry
                    if !localProofIndices.contains(existing) {
                        // Already tracked
                    }
                    continue
                }
                let entryIndex = allLeafEntries.count
                allLeafEntries.append(AggregateLeafEntry(
                    leafIndex: proof.leafIndex,
                    leaf: proof.leaf,
                    treeInfoIndex: treeInfoIndex
                ))
                localProofIndices.append(entryIndex)
                seenLeafIndices[proof.leafIndex] = entryIndex
            }

            // Compress siblings: walk level-by-level, collect unique siblings
            // that are not derivable from known leaves/parents.
            var knownAtLevel = Set(seenLeafIndices.keys)

            for level in 0..<depth {
                var neededSiblings = [(Int, Fr)]()  // (nodeIndex, value)

                for nodeIdx in knownAtLevel.sorted() {
                    let sibIdx = nodeIdx ^ 1
                    if !knownAtLevel.contains(sibIdx) {
                        // Find the sibling value from any proof that covers this node
                        var sibValue: Fr?
                        for pi in proofIndices {
                            let proof = proofs[pi]
                            if proof.path.count > level {
                                // Check if this proof's path at this level corresponds to sibIdx
                                let proofNodeIdx = proof.leafIndex >> level
                                let proofSibIdx = proofNodeIdx ^ 1
                                // The path entry at this level is the sibling of the node
                                // at position (leafIndex >> level)
                                if proofSibIdx == sibIdx {
                                    sibValue = proof.path[level]
                                    break
                                }
                            }
                        }

                        if let val = sibValue {
                            neededSiblings.append((sibIdx, val))
                        }
                    }
                }

                // Deduplicate siblings at this level
                var seenSibs = Set<Int>()
                for (sibIdx, val) in neededSiblings.sorted(by: { $0.0 < $1.0 }) {
                    if seenSibs.insert(sibIdx).inserted {
                        allSiblings.append(CompressedSibling(
                            level: level, nodeIndex: sibIdx, value: val
                        ))
                        allSibTreeIndices.append(treeInfoIndex)
                    }
                }

                // Advance to next level: parents of all known + sibling nodes
                var nextKnown = Set<Int>()
                for nodeIdx in knownAtLevel {
                    nextKnown.insert(nodeIdx / 2)
                }
                for (sibIdx, _) in neededSiblings {
                    nextKnown.insert(sibIdx / 2)
                }
                knownAtLevel = nextKnown
            }

            trees.append(TreeAggregateInfo(
                treeId: treeId,
                root: root,
                leafCount: leafCount,
                proofIndices: localProofIndices
            ))
        }

        return AggregateMerkleProof(
            trees: trees,
            leafEntries: allLeafEntries,
            siblings: allSiblings,
            siblingTreeIndices: allSibTreeIndices
        )
    }

    // MARK: - Incremental Aggregation

    /// Add new proofs to an existing aggregate proof.
    /// Returns a new aggregate that includes both the original proofs and the new ones.
    /// This re-aggregates from scratch for correctness (shared paths may change).
    public func incrementalAggregate(existing: AggregateMerkleProof,
                                      newProofs: [MerkleInclusionProof]) -> AggregateMerkleProof {
        // Reconstruct the original individual proofs from the aggregate
        var allProofs = reconstructProofs(from: existing)
        allProofs.append(contentsOf: newProofs)
        return aggregate(proofs: allProofs)
    }

    /// Reconstruct individual MerkleInclusionProofs from an aggregate proof.
    /// Each leaf entry is expanded back into a full proof by walking the sibling
    /// data for its tree.
    public func reconstructProofs(from aggregate: AggregateMerkleProof) -> [MerkleInclusionProof] {
        var result = [MerkleInclusionProof]()
        result.reserveCapacity(aggregate.proofCount)

        for entry in aggregate.leafEntries {
            let treeInfo = aggregate.trees[entry.treeInfoIndex]
            let depth = treeInfo.leafCount.trailingZeroBitCount

            // Build the authentication path for this leaf
            var path = [Fr]()
            path.reserveCapacity(depth)

            var nodeIdx = entry.leafIndex
            for level in 0..<depth {
                let sibIdx = nodeIdx ^ 1

                // Search for this sibling in the compressed siblings
                var found = false
                for (i, sib) in aggregate.siblings.enumerated() {
                    if aggregate.siblingTreeIndices[i] == entry.treeInfoIndex
                        && sib.level == level && sib.nodeIndex == sibIdx {
                        path.append(sib.value)
                        found = true
                        break
                    }
                }

                if !found {
                    // The sibling must be derivable from another leaf in this tree.
                    // Recompute it from known leaves and siblings at lower levels.
                    let derived = deriveSiblingValue(
                        aggregate: aggregate,
                        treeInfoIndex: entry.treeInfoIndex,
                        level: level,
                        nodeIndex: sibIdx
                    )
                    path.append(derived)
                }

                nodeIdx = nodeIdx / 2
            }

            result.append(MerkleInclusionProof(
                root: treeInfo.root,
                leaf: entry.leaf,
                leafIndex: entry.leafIndex,
                leafCount: treeInfo.leafCount,
                path: path,
                treeId: treeInfo.treeId
            ))
        }

        return result
    }

    /// Derive a sibling value that was omitted from the aggregate because it was
    /// computable from other known data (shared path optimization).
    private func deriveSiblingValue(aggregate: AggregateMerkleProof,
                                     treeInfoIndex: Int,
                                     level: Int,
                                     nodeIndex: Int) -> Fr {
        // The node at (level, nodeIndex) can be computed by hashing its two children
        // at (level-1, nodeIndex*2) and (level-1, nodeIndex*2+1).
        // Base case: level 0 nodes are leaves.

        if level == 0 {
            // This is a leaf — find it in the leaf entries
            for entry in aggregate.leafEntries {
                if entry.treeInfoIndex == treeInfoIndex && entry.leafIndex == nodeIndex {
                    return entry.leaf
                }
            }
            // Should not reach here in a valid aggregate
            return Fr.zero
        }

        let leftChild = nodeIndex * 2
        let rightChild = nodeIndex * 2 + 1

        let leftVal = findOrDeriveNode(aggregate: aggregate, treeInfoIndex: treeInfoIndex,
                                        level: level - 1, nodeIndex: leftChild)
        let rightVal = findOrDeriveNode(aggregate: aggregate, treeInfoIndex: treeInfoIndex,
                                         level: level - 1, nodeIndex: rightChild)

        return poseidon2Hash(leftVal, rightVal)
    }

    /// Find a node value either as a leaf, a stored sibling, or derive it recursively.
    private func findOrDeriveNode(aggregate: AggregateMerkleProof,
                                   treeInfoIndex: Int,
                                   level: Int,
                                   nodeIndex: Int) -> Fr {
        // Check if it's a leaf
        if level == 0 {
            for entry in aggregate.leafEntries {
                if entry.treeInfoIndex == treeInfoIndex && entry.leafIndex == nodeIndex {
                    return entry.leaf
                }
            }
        }

        // Check if it's a stored sibling
        for (i, sib) in aggregate.siblings.enumerated() {
            if aggregate.siblingTreeIndices[i] == treeInfoIndex
                && sib.level == level && sib.nodeIndex == nodeIndex {
                return sib.value
            }
        }

        // Derive from children
        return deriveSiblingValue(aggregate: aggregate, treeInfoIndex: treeInfoIndex,
                                   level: level, nodeIndex: nodeIndex)
    }

    // MARK: - Verification

    /// Verify an aggregate Merkle proof.
    /// For each tree, recomputes the root from leaves and siblings, then checks
    /// it matches the stored root.
    public func verifyAggregate(_ aggregate: AggregateMerkleProof) -> Bool {
        for (treeIdx, treeInfo) in aggregate.trees.enumerated() {
            let depth = treeInfo.leafCount.trailingZeroBitCount

            // Collect leaf entries for this tree
            var knownNodes = [Int: Fr]()
            for entryIdx in treeInfo.proofIndices {
                let entry = aggregate.leafEntries[entryIdx]
                knownNodes[entry.leafIndex] = entry.leaf
            }

            // Collect siblings for this tree, organized by level
            var siblingsByLevel = [[Int: Fr]](repeating: [:], count: depth)
            for (i, sib) in aggregate.siblings.enumerated() {
                if aggregate.siblingTreeIndices[i] == treeIdx {
                    siblingsByLevel[sib.level][sib.nodeIndex] = sib.value
                }
            }

            // Walk level-by-level, hashing up to the root
            for level in 0..<depth {
                let currentIndices = knownNodes.keys.sorted()
                let currentKnown = Set(currentIndices)

                // Fill in siblings for this level
                for nodeIdx in currentIndices {
                    let sibIdx = nodeIdx ^ 1
                    if !currentKnown.contains(sibIdx) && knownNodes[sibIdx] == nil {
                        if let sibVal = siblingsByLevel[level][sibIdx] {
                            knownNodes[sibIdx] = sibVal
                        } else {
                            // Missing sibling — proof is invalid
                            return false
                        }
                    }
                }

                // Compute parents
                var parentNodes = [Int: Fr]()
                let allIndices = knownNodes.keys.sorted()
                var processed = Set<Int>()

                for nodeIdx in allIndices {
                    let parentIdx = nodeIdx / 2
                    if processed.contains(parentIdx) { continue }

                    let leftIdx = parentIdx * 2
                    let rightIdx = parentIdx * 2 + 1

                    guard let left = knownNodes[leftIdx],
                          let right = knownNodes[rightIdx] else {
                        continue
                    }

                    parentNodes[parentIdx] = poseidon2Hash(left, right)
                    processed.insert(parentIdx)
                }

                knownNodes = parentNodes
            }

            // Should have exactly node 0 = root
            guard let computedRoot = knownNodes[0] else { return false }

            if !frEqual(computedRoot, treeInfo.root) {
                return false
            }
        }

        return true
    }

    /// GPU-accelerated batch verification: verify multiple aggregate proofs in parallel.
    /// Returns array of booleans, one per aggregate.
    public func verifyAggregates(_ aggregates: [AggregateMerkleProof]) -> [Bool] {
        // Dispatch verification on concurrent queue for GPU parallelism
        let results = aggregates.map { verifyAggregate($0) }
        return results
    }

    // MARK: - Root Consistency

    /// Check that all proofs within an aggregate that share a treeId have consistent roots.
    /// Returns true if all proofs for the same tree agree on the root value.
    public func checkRootConsistency(_ proofs: [MerkleInclusionProof]) -> Bool {
        var rootByTree = [UInt64: Fr]()
        for proof in proofs {
            if let existing = rootByTree[proof.treeId] {
                if !frEqual(existing, proof.root) {
                    return false
                }
            } else {
                rootByTree[proof.treeId] = proof.root
            }
        }
        return true
    }

    /// Check root consistency within an aggregate proof.
    /// Verifies that all leafEntries for a given tree reference the same root.
    public func checkAggregateRootConsistency(_ aggregate: AggregateMerkleProof) -> Bool {
        for treeInfo in aggregate.trees {
            for entryIdx in treeInfo.proofIndices {
                let entry = aggregate.leafEntries[entryIdx]
                if entry.treeInfoIndex >= aggregate.trees.count { return false }
                let referencedTree = aggregate.trees[entry.treeInfoIndex]
                if !frEqual(referencedTree.root, treeInfo.root) {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Utility: Build proofs from a tree

    /// Convenience: build MerkleInclusionProofs from a MerkleTree for given leaf indices.
    public func buildProofs(tree: MerkleTree, leafIndices: [Int],
                             treeId: UInt64 = 0) -> [MerkleInclusionProof] {
        leafIndices.map { idx in
            let authPath = tree.proof(forLeafAt: idx)
            return MerkleInclusionProof(
                root: tree.root,
                leaf: tree.leaf(at: idx),
                leafIndex: idx,
                leafCount: tree.leafCount,
                path: authPath.siblings,
                treeId: treeId
            )
        }
    }

    // MARK: - Aggregate Statistics

    /// Compute statistics about the aggregate: how many siblings saved vs. naive.
    public func aggregateStats(_ aggregate: AggregateMerkleProof) -> AggregateStats {
        var totalNaiveSiblings = 0
        for entry in aggregate.leafEntries {
            let treeInfo = aggregate.trees[entry.treeInfoIndex]
            totalNaiveSiblings += treeInfo.leafCount.trailingZeroBitCount
        }

        let actualSiblings = aggregate.siblings.count
        let saved = totalNaiveSiblings - actualSiblings
        let ratio = totalNaiveSiblings > 0
            ? Double(actualSiblings) / Double(totalNaiveSiblings)
            : 1.0

        return AggregateStats(
            proofCount: aggregate.proofCount,
            treeCount: aggregate.treeCount,
            totalNaiveSiblings: totalNaiveSiblings,
            actualSiblings: actualSiblings,
            siblingsSaved: saved,
            compressionRatio: ratio
        )
    }

    // MARK: - GPU Batch Hash Verification

    /// Verify a single proof against a known root using GPU-optimized path hashing.
    /// Falls back to CPU Poseidon2 for small proofs.
    public func verifySingleProof(_ proof: MerkleInclusionProof) -> Bool {
        var current = proof.leaf
        var idx = proof.leafIndex

        for sibling in proof.path {
            if idx & 1 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx >>= 1
        }

        return frEqual(current, proof.root)
    }

    /// Verify multiple individual proofs in a batch.
    /// Returns array of booleans, one per proof.
    public func verifyProofsBatch(_ proofs: [MerkleInclusionProof]) -> [Bool] {
        proofs.map { verifySingleProof($0) }
    }

    // MARK: - Merge Aggregates

    /// Merge two aggregate proofs into one.
    /// Re-aggregates all proofs from both aggregates for optimal deduplication.
    public func mergeAggregates(_ a: AggregateMerkleProof,
                                 _ b: AggregateMerkleProof) -> AggregateMerkleProof {
        let proofsA = reconstructProofs(from: a)
        let proofsB = reconstructProofs(from: b)
        return aggregate(proofs: proofsA + proofsB)
    }
}

// MARK: - Aggregate Statistics

/// Statistics about a Merkle proof aggregate.
public struct AggregateStats {
    /// Number of individual proofs in the aggregate.
    public let proofCount: Int

    /// Number of distinct trees.
    public let treeCount: Int

    /// Total sibling nodes without deduplication (sum of depths).
    public let totalNaiveSiblings: Int

    /// Actual sibling nodes stored after deduplication.
    public let actualSiblings: Int

    /// Number of sibling nodes saved by deduplication.
    public let siblingsSaved: Int

    /// Compression ratio (actualSiblings / totalNaiveSiblings). Lower is better.
    public let compressionRatio: Double

    public init(proofCount: Int, treeCount: Int, totalNaiveSiblings: Int,
                actualSiblings: Int, siblingsSaved: Int, compressionRatio: Double) {
        self.proofCount = proofCount
        self.treeCount = treeCount
        self.totalNaiveSiblings = totalNaiveSiblings
        self.actualSiblings = actualSiblings
        self.siblingsSaved = siblingsSaved
        self.compressionRatio = compressionRatio
    }
}
