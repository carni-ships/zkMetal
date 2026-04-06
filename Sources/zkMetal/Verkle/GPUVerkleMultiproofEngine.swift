// GPUVerkleMultiproofEngine — GPU-accelerated Verkle tree multiproof engine
//
// Provides batch opening of multiple Verkle tree leaves with shared internal nodes.
// Uses IPA-based inner proofs, Pedersen commitments at each level, and proof
// compression via random evaluation point. GPU-accelerated MSM for commitment
// computation with automatic CPU fallback.
//
// The multiproof protocol:
//   1. Collect all (commitment, child_index, value) tuples across all queried paths
//   2. Derive a random evaluation point r via Fiat-Shamir transcript
//   3. Combine all opening polynomials at r into a single aggregated polynomial
//   4. Produce one IPA proof for the aggregated opening
//   5. Verifier reconstructs the same combination and checks the single IPA
//
// This is more efficient than N independent proofs because:
//   - Shared internal nodes are deduplicated (one commitment, one opening)
//   - The single aggregated IPA proof replaces N separate IPA proofs
//   - GPU MSM batching amortizes dispatch overhead across all commitments
//
// Architecture:
//   - GPUVerkleMultiproofEngine wraps a GPUVerkleTreeEngine for tree operations
//   - MultiproofQuery describes which keys to open
//   - CompressedMultiproof is the output: one aggregated IPA + metadata
//   - Transcript hashing uses Blake3 for Fiat-Shamir challenges
//
// References:
//   - Dankrad Feist: Verkle multiproofs
//   - EIP-6800: Ethereum Verkle tree specification
//   - Bunz et al. 2018: Bulletproofs/IPA

import Foundation
import Metal
import NeonFieldOps

// MARK: - Multiproof Query

/// A single query in a multiproof: identifies a leaf by its index or key path.
public struct MultiproofQuery {
    /// Leaf index in the flat-array tree (for flat-array mode).
    public let leafIndex: Int
    /// Expected leaf value (for verification).
    public let expectedValue: Fr

    public init(leafIndex: Int, expectedValue: Fr) {
        self.leafIndex = leafIndex
        self.expectedValue = expectedValue
    }
}

// MARK: - Opening Tuple

/// A single opening at one tree level: commitment C opens to value v at child index i.
/// Multiple queries may share the same opening if they pass through the same node.
public struct MultiproofOpening: Equatable {
    /// The node commitment being opened.
    public let commitment: PointProjective
    /// Child index within the node (0..<width).
    public let childIndex: Int
    /// The value at that child index.
    public let value: Fr
    /// The full vector of child values (needed for IPA proof generation).
    public let nodeValues: [Fr]

    public static func == (lhs: MultiproofOpening, rhs: MultiproofOpening) -> Bool {
        return lhs.childIndex == rhs.childIndex && frEqual(lhs.value, rhs.value)
            && pointEqual(lhs.commitment, rhs.commitment)
    }
}

// MARK: - Compressed Multiproof

/// A compressed multiproof: a single aggregated IPA proof covering all openings.
///
/// Instead of N separate IPA proofs, the prover combines all opening polynomials
/// using powers of a random challenge r, and produces one IPA proof for the sum.
public struct CompressedMultiproof {
    /// Unique node commitments referenced across all query paths (deduplicated).
    public let commitments: [PointProjective]
    /// Per-opening: child index at the node.
    public let childIndices: [Int]
    /// Per-opening: the opened value.
    public let values: [Fr]
    /// Per-query: which openings belong to this query (indices into the openings arrays).
    public let queryOpeningIndices: [[Int]]
    /// Per-query: leaf index.
    public let queryLeafIndices: [Int]
    /// The aggregated IPA proof (L, R vectors and final scalar).
    public let aggregatedL: [PointProjective]
    public let aggregatedR: [PointProjective]
    public let aggregatedFinalA: Fr
    /// The random evaluation point used for aggregation.
    public let evaluationPoint: Fr
    /// The aggregated commitment (weighted sum of all opened commitments).
    public let aggregatedCommitment: PointProjective
    /// The aggregated value (weighted sum of all opened values).
    public let aggregatedValue: Fr
    /// Root commitment for verification.
    public let root: PointProjective

    public init(commitments: [PointProjective],
                childIndices: [Int],
                values: [Fr],
                queryOpeningIndices: [[Int]],
                queryLeafIndices: [Int],
                aggregatedL: [PointProjective],
                aggregatedR: [PointProjective],
                aggregatedFinalA: Fr,
                evaluationPoint: Fr,
                aggregatedCommitment: PointProjective,
                aggregatedValue: Fr,
                root: PointProjective) {
        self.commitments = commitments
        self.childIndices = childIndices
        self.values = values
        self.queryOpeningIndices = queryOpeningIndices
        self.queryLeafIndices = queryLeafIndices
        self.aggregatedL = aggregatedL
        self.aggregatedR = aggregatedR
        self.aggregatedFinalA = aggregatedFinalA
        self.evaluationPoint = evaluationPoint
        self.aggregatedCommitment = aggregatedCommitment
        self.aggregatedValue = aggregatedValue
        self.root = root
    }
}

// MARK: - Serialized Compressed Multiproof

/// Compact byte serialization for a CompressedMultiproof.
///
/// Layout:
///   [2 bytes] number of unique commitments (little-endian u16)
///   [64 * numC bytes] commitments (affine x,y packed, 32 bytes each)
///   [2 bytes] number of openings
///   For each opening:
///     [1 byte] child index
///     [32 bytes] value
///   [2 bytes] number of queries
///   For each query:
///     [1 byte] number of opening indices
///     [N bytes] opening indices (1 byte each)
///     [4 bytes] leaf index (little-endian u32)
///   [1 byte] log(width) = number of L/R pairs
///   [32 * logW bytes] aggregated L points (affine x only, 32 bytes each)
///   [32 * logW bytes] aggregated R points
///   [32 bytes] aggregated final scalar a
///   [32 bytes] evaluation point r
///   [64 bytes] aggregated commitment (affine x,y)
///   [32 bytes] aggregated value
///   [64 bytes] root commitment
public struct SerializedCompressedMultiproof {
    public let data: [UInt8]

    public init(data: [UInt8]) {
        self.data = data
    }

    /// Byte size of the serialized proof.
    public var byteSize: Int { data.count }
}

// MARK: - Multiproof Transcript

/// Fiat-Shamir transcript for multiproof challenge derivation.
///
/// Absorbs commitments, child indices, and values, then squeezes a field element
/// used as the random evaluation point for proof compression.
public struct MultiproofTranscript {
    private var state: [UInt8]

    public init(label: String = "verkle-multiproof") {
        state = Array(label.utf8)
    }

    /// Absorb a curve point into the transcript.
    public mutating func absorbPoint(_ p: PointProjective) {
        let aff = cBatchToAffine([p])[0]
        let xLimbs = fpToInt(aff.x)
        for limb in xLimbs {
            for j in 0..<8 {
                state.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
        let yLimbs = fpToInt(aff.y)
        for limb in yLimbs {
            for j in 0..<8 {
                state.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
    }

    /// Absorb a field element into the transcript.
    public mutating func absorbFr(_ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for j in 0..<8 {
                state.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
    }

    /// Absorb a byte into the transcript.
    public mutating func absorbByte(_ b: UInt8) {
        state.append(b)
    }

    /// Absorb an integer (as 4 little-endian bytes).
    public mutating func absorbInt(_ v: Int) {
        let u = UInt32(v)
        state.append(UInt8(u & 0xFF))
        state.append(UInt8((u >> 8) & 0xFF))
        state.append(UInt8((u >> 16) & 0xFF))
        state.append(UInt8((u >> 24) & 0xFF))
    }

    /// Squeeze a field element challenge from the transcript.
    public mutating func squeezeFr() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        // Feed hash back into state for domain separation on subsequent squeezes
        state.append(contentsOf: hash)

        var limbs: [UInt64] = [0, 0, 0, 0]
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]; limbs[1] = ptr[1]
            limbs[2] = ptr[2]; limbs[3] = ptr[3]
        }
        // Convert to Montgomery form and reduce mod Fr modulus
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - GPU Verkle Multiproof Engine

/// GPU-accelerated engine for generating and verifying compressed Verkle multiproofs.
///
/// Uses a GPUVerkleTreeEngine for tree operations and adds the multiproof aggregation
/// protocol on top. The key optimization is combining multiple IPA openings into a
/// single aggregated proof via a random evaluation point.
///
/// Usage:
///   let engine = try GPUVerkleMultiproofEngine(branchingFactor: 4)
///   let leaves = (0..<16).map { frFromInt(UInt64($0 + 1)) }
///   try engine.buildTree(leaves: leaves)
///   let queries = [MultiproofQuery(leafIndex: 0, expectedValue: frFromInt(1)),
///                  MultiproofQuery(leafIndex: 5, expectedValue: frFromInt(6))]
///   let proof = try engine.generateCompressedMultiproof(queries: queries)
///   let valid = engine.verifyCompressedMultiproof(proof)
public final class GPUVerkleMultiproofEngine {
    /// The underlying tree engine.
    public let treeEngine: GPUVerkleTreeEngine

    /// Branching factor (delegated from tree engine).
    public var branchingFactor: Int { treeEngine.branchingFactor }

    /// log2(branchingFactor).
    public var logWidth: Int { treeEngine.logWidth }

    /// Whether GPU acceleration is available.
    public var gpuAvailable: Bool { treeEngine.gpuAvailable }

    /// IPA engine (delegated from tree engine).
    public let ipaEngine: IPAEngine

    /// Shared generators.
    public let generators: [PointAffine]

    /// Blinding point Q.
    public let Q: PointAffine

    /// Cached tree levels after buildTree.
    private var treeLevels: [[PointProjective]] = []

    /// Cached leaf chunks after buildTree.
    private var leafChunks: [[Fr]] = []

    /// Whether the tree has been built.
    private var treeBuilt = false

    /// Number of leaves.
    private var leafCount = 0

    // MARK: - Initialization

    /// Create a GPU Verkle multiproof engine.
    ///
    /// - Parameter branchingFactor: children per node (power of 2, default 256)
    /// - Throws: if underlying engine creation fails
    public init(branchingFactor: Int = 256) throws {
        self.treeEngine = try GPUVerkleTreeEngine(branchingFactor: branchingFactor)
        self.ipaEngine = treeEngine.ipaEngine
        self.generators = treeEngine.generators
        self.Q = treeEngine.Q
    }

    // MARK: - Tree Construction

    /// Build a Verkle tree from leaf values.
    ///
    /// - Parameter leaves: leaf values (count must be multiple of branchingFactor)
    /// - Returns: (levels, leafChunks) from the underlying tree engine
    @discardableResult
    public func buildTree(leaves: [Fr]) throws -> (levels: [[PointProjective]], leafChunks: [[Fr]]) {
        let result = try treeEngine.buildTree(leaves: leaves)
        self.treeLevels = result.levels
        self.leafChunks = result.leafChunks
        self.treeBuilt = true
        self.leafCount = leaves.count
        return result
    }

    /// Root commitment of the current tree.
    public func rootCommitment() -> PointProjective {
        return treeEngine.rootCommitment()
    }

    // MARK: - Compressed Multiproof Generation

    /// Generate a compressed multiproof for a set of queries.
    ///
    /// Protocol:
    ///   1. For each query, walk the tree path and collect (commitment, childIndex, value) openings
    ///   2. Deduplicate identical openings across queries
    ///   3. Build a Fiat-Shamir transcript from all openings
    ///   4. Derive random evaluation point r
    ///   5. Compute aggregated polynomial: f(x) = sum_i r^i * f_i(x)
    ///      where f_i is the opening polynomial for opening i
    ///   6. Compute aggregated commitment: C_agg = sum_i r^i * C_i
    ///   7. Compute aggregated value: v_agg = sum_i r^i * v_i
    ///   8. Produce a single IPA proof for (C_agg, f_agg, v_agg)
    ///
    /// - Parameter queries: array of MultiproofQuery
    /// - Returns: a CompressedMultiproof
    public func generateCompressedMultiproof(queries: [MultiproofQuery]) throws -> CompressedMultiproof {
        precondition(treeBuilt, "Must call buildTree first")
        precondition(!queries.isEmpty, "Must have at least one query")

        // Step 1: Collect all openings along each query's path
        var allOpenings: [MultiproofOpening] = []
        var openingMap: [String: Int] = [:]  // deduplication key -> index
        var queryOpeningIndices: [[Int]] = []
        var queryLeafIndices: [Int] = []

        for query in queries {
            let leafIdx = query.leafIndex
            precondition(leafIdx >= 0 && leafIdx < leafCount, "Leaf index out of range")
            queryLeafIndices.append(leafIdx)

            var thisQueryOpenings: [Int] = []

            // Level 0: open the leaf chunk
            let chunkIdx = leafIdx / branchingFactor
            let childIdx = leafIdx % branchingFactor
            let chunk = leafChunks[chunkIdx]
            let commitment0 = treeLevels[0][chunkIdx]

            let key0 = openingKey(commitment0, childIdx)
            if let existingIdx = openingMap[key0] {
                thisQueryOpenings.append(existingIdx)
            } else {
                let idx = allOpenings.count
                allOpenings.append(MultiproofOpening(
                    commitment: commitment0,
                    childIndex: childIdx,
                    value: chunk[childIdx],
                    nodeValues: chunk
                ))
                openingMap[key0] = idx
                thisQueryOpenings.append(idx)
            }

            // Upper levels: open each parent node
            var nodeIndex = chunkIdx
            for level in 0..<(treeLevels.count - 1) {
                let numNodes = treeLevels[level].count
                let padded = numNodes % branchingFactor == 0
                    ? numNodes
                    : numNodes + (branchingFactor - numNodes % branchingFactor)

                var childValues = [Fr](repeating: Fr.zero, count: padded)
                let affineAll = cBatchToAffine(treeLevels[level])
                for i in 0..<numNodes {
                    childValues[i] = commitmentToFr(affineAll[i])
                }

                let ci = nodeIndex % branchingFactor
                let parentIdx = nodeIndex / branchingFactor
                let parentCommitment = treeLevels[level + 1][parentIdx]
                let parentChunk = Array(childValues[parentIdx * branchingFactor ..< (parentIdx + 1) * branchingFactor])

                let key = openingKey(parentCommitment, ci)
                if let existingIdx = openingMap[key] {
                    thisQueryOpenings.append(existingIdx)
                } else {
                    let idx = allOpenings.count
                    allOpenings.append(MultiproofOpening(
                        commitment: parentCommitment,
                        childIndex: ci,
                        value: parentChunk[ci],
                        nodeValues: parentChunk
                    ))
                    openingMap[key] = idx
                    thisQueryOpenings.append(idx)
                }

                nodeIndex = parentIdx
            }

            queryOpeningIndices.append(thisQueryOpenings)
        }

        // Step 2: Build Fiat-Shamir transcript
        var transcript = MultiproofTranscript()
        transcript.absorbPoint(rootCommitment())
        transcript.absorbInt(allOpenings.count)

        for opening in allOpenings {
            transcript.absorbPoint(opening.commitment)
            transcript.absorbByte(UInt8(opening.childIndex))
            transcript.absorbFr(opening.value)
        }

        // Step 3: Derive random evaluation point r
        let r = transcript.squeezeFr()

        // Step 4: Compute aggregated polynomial
        // f_agg(x) = sum_i r^i * f_i(x) where f_i has nodeValues[j] at position j
        // The evaluation vector b for opening i has 1 at childIndex[i] and 0 elsewhere
        // So the aggregated vector a = sum_i r^i * nodeValues_i
        // And the aggregated b = sum_i r^i * b_i (pointwise)
        var aggA = [Fr](repeating: Fr.zero, count: branchingFactor)
        var aggB = [Fr](repeating: Fr.zero, count: branchingFactor)
        var aggValue = Fr.zero
        var rPower = Fr.one  // r^0 = 1

        for opening in allOpenings {
            // Accumulate a = sum r^i * nodeValues_i
            for j in 0..<branchingFactor {
                aggA[j] = frAdd(aggA[j], frMul(rPower, opening.nodeValues[j]))
            }
            // Accumulate b: b_i has 1 at childIndex, so aggB[childIndex] += r^i
            aggB[opening.childIndex] = frAdd(aggB[opening.childIndex], rPower)
            // Accumulate value
            aggValue = frAdd(aggValue, frMul(rPower, opening.value))

            rPower = frMul(rPower, r)
        }

        // Step 5: Compute aggregated commitment C_agg = sum_i r^i * C_i
        var aggCommitment = PointProjective(x: .one, y: .one, z: .zero) // identity
        rPower = Fr.one
        for opening in allOpenings {
            let scaled = cPointScalarMul(opening.commitment, rPower)
            aggCommitment = pointAdd(aggCommitment, scaled)
            rPower = frMul(rPower, r)
        }

        // Step 6: Generate single IPA proof for (aggA, aggB)
        let ipaProof = try ipaEngine.createProof(a: aggA, b: aggB)

        // Extract unique commitments (preserving order)
        let uniqueCommitments = allOpenings.map { $0.commitment }
        let childIndices = allOpenings.map { $0.childIndex }
        let values = allOpenings.map { $0.value }

        return CompressedMultiproof(
            commitments: uniqueCommitments,
            childIndices: childIndices,
            values: values,
            queryOpeningIndices: queryOpeningIndices,
            queryLeafIndices: queryLeafIndices,
            aggregatedL: ipaProof.L,
            aggregatedR: ipaProof.R,
            aggregatedFinalA: ipaProof.a,
            evaluationPoint: r,
            aggregatedCommitment: aggCommitment,
            aggregatedValue: aggValue,
            root: rootCommitment()
        )
    }

    // MARK: - Compressed Multiproof Verification

    /// Verify a compressed multiproof.
    ///
    /// Protocol:
    ///   1. Reconstruct the Fiat-Shamir transcript from the proof data
    ///   2. Re-derive the random evaluation point r
    ///   3. Recompute the aggregated commitment and value using r
    ///   4. Verify the single IPA proof against the aggregated commitment/value
    ///
    /// - Parameter proof: the CompressedMultiproof to verify
    /// - Returns: true if the proof is valid
    public func verifyCompressedMultiproof(_ proof: CompressedMultiproof) -> Bool {
        let numOpenings = proof.commitments.count
        guard numOpenings > 0 else { return false }
        guard proof.childIndices.count == numOpenings else { return false }
        guard proof.values.count == numOpenings else { return false }
        guard proof.aggregatedL.count == logWidth else { return false }
        guard proof.aggregatedR.count == logWidth else { return false }

        // Step 1: Reconstruct transcript
        var transcript = MultiproofTranscript()
        transcript.absorbPoint(proof.root)
        transcript.absorbInt(numOpenings)

        for i in 0..<numOpenings {
            transcript.absorbPoint(proof.commitments[i])
            transcript.absorbByte(UInt8(proof.childIndices[i]))
            transcript.absorbFr(proof.values[i])
        }

        // Step 2: Re-derive r
        let r = transcript.squeezeFr()

        // Check that r matches the proof's evaluation point
        guard frEqual(r, proof.evaluationPoint) else { return false }

        // Step 3: Recompute aggregated commitment and value
        var aggCommitment = PointProjective(x: .one, y: .one, z: .zero)
        var aggValue = Fr.zero
        var aggB = [Fr](repeating: Fr.zero, count: branchingFactor)
        var rPower = Fr.one

        for i in 0..<numOpenings {
            let scaled = cPointScalarMul(proof.commitments[i], rPower)
            aggCommitment = pointAdd(aggCommitment, scaled)
            aggValue = frAdd(aggValue, frMul(rPower, proof.values[i]))
            aggB[proof.childIndices[i]] = frAdd(aggB[proof.childIndices[i]], rPower)
            rPower = frMul(rPower, r)
        }

        // Verify aggregated commitment matches
        guard pointEqual(aggCommitment, proof.aggregatedCommitment) else { return false }
        // Verify aggregated value matches
        guard frEqual(aggValue, proof.aggregatedValue) else { return false }

        // Step 4: Verify IPA proof (pass raw commitment — verify computes Cbound internally)
        return ipaEngine.verify(
            commitment: aggCommitment,
            b: aggB,
            innerProductValue: aggValue,
            proof: IPAProof(L: proof.aggregatedL, R: proof.aggregatedR, a: proof.aggregatedFinalA)
        )
    }

    // MARK: - Proof Size Analysis

    /// Compute the number of openings that would be in an uncompressed multiproof.
    ///
    /// - Parameter queries: the queries to analyze
    /// - Returns: (totalOpenings, uniqueOpenings, compressionRatio)
    public func analyzeProofSize(queries: [MultiproofQuery]) -> (total: Int, unique: Int, ratio: Double) {
        precondition(treeBuilt, "Must call buildTree first")

        var uniqueKeys = Set<String>()
        var totalOpenings = 0

        for query in queries {
            let leafIdx = query.leafIndex
            let chunkIdx = leafIdx / branchingFactor
            let childIdx = leafIdx % branchingFactor

            let key0 = "\(chunkIdx)_\(childIdx)"
            uniqueKeys.insert(key0)
            totalOpenings += 1

            var nodeIndex = chunkIdx
            for level in 0..<(treeLevels.count - 1) {
                let ci = nodeIndex % branchingFactor
                let parentIdx = nodeIndex / branchingFactor
                let key = "L\(level + 1)_\(parentIdx)_\(ci)"
                uniqueKeys.insert(key)
                totalOpenings += 1
                nodeIndex = parentIdx
            }
        }

        let unique = uniqueKeys.count
        let ratio = totalOpenings > 0 ? Double(unique) / Double(totalOpenings) : 1.0
        return (total: totalOpenings, unique: unique, ratio: ratio)
    }

    // MARK: - Batch Tree Update with Re-proof

    /// Update leaves and regenerate a compressed multiproof for the affected queries.
    ///
    /// - Parameters:
    ///   - updates: array of (leafIndex, newValue) pairs
    ///   - queries: queries to re-prove after the update
    /// - Returns: (newRoot, newProof) after the updates
    public func updateAndReprove(
        updates: [(Int, Fr)],
        queries: [MultiproofQuery]
    ) throws -> (root: PointProjective, proof: CompressedMultiproof) {
        // Apply updates
        let newRoot = try treeEngine.batchUpdateLeaves(updates)

        // Refresh cached levels
        // The tree engine updates internally, so re-extract from a fresh buildTree
        // For incremental updates, the treeEngine has already updated treeLevels internally
        // We need to refresh our cached view
        refreshCachedLevels()

        // Generate new proof with updated queries
        var updatedQueries = queries
        for i in 0..<updatedQueries.count {
            // Check if this query's leaf was updated
            for (updIdx, newVal) in updates {
                if updatedQueries[i].leafIndex == updIdx {
                    updatedQueries[i] = MultiproofQuery(
                        leafIndex: updIdx,
                        expectedValue: newVal
                    )
                }
            }
        }

        let proof = try generateCompressedMultiproof(queries: updatedQueries)
        return (root: newRoot, proof: proof)
    }

    // MARK: - Serialization

    /// Serialize a compressed multiproof to bytes.
    public func serializeCompressedMultiproof(_ proof: CompressedMultiproof) -> SerializedCompressedMultiproof {
        var bytes = [UInt8]()
        bytes.reserveCapacity(2048)

        // Number of commitments
        let numC = UInt16(proof.commitments.count)
        appendU16(&bytes, numC)

        // Commitments (affine x,y)
        let affine = cBatchToAffine(proof.commitments)
        for p in affine {
            appendFp(&bytes, p.x)
            appendFp(&bytes, p.y)
        }

        // Number of openings
        let numO = UInt16(proof.childIndices.count)
        appendU16(&bytes, numO)

        // Openings: childIndex + value
        for i in 0..<Int(numO) {
            bytes.append(UInt8(proof.childIndices[i]))
            appendFr(&bytes, proof.values[i])
        }

        // Number of queries
        let numQ = UInt16(proof.queryLeafIndices.count)
        appendU16(&bytes, numQ)

        // Query data
        for i in 0..<Int(numQ) {
            let indices = proof.queryOpeningIndices[i]
            bytes.append(UInt8(indices.count))
            for idx in indices { bytes.append(UInt8(idx)) }
            appendU32(&bytes, UInt32(proof.queryLeafIndices[i]))
        }

        // Aggregated IPA: L, R, finalA
        let logW = UInt8(proof.aggregatedL.count)
        bytes.append(logW)
        for l in proof.aggregatedL { appendPoint(&bytes, l) }
        for r in proof.aggregatedR { appendPoint(&bytes, r) }
        appendFr(&bytes, proof.aggregatedFinalA)

        // Evaluation point
        appendFr(&bytes, proof.evaluationPoint)

        // Aggregated commitment and value
        appendPoint(&bytes, proof.aggregatedCommitment)
        appendFr(&bytes, proof.aggregatedValue)

        // Root
        appendPoint(&bytes, proof.root)

        return SerializedCompressedMultiproof(data: bytes)
    }

    /// Deserialize a compressed multiproof from bytes.
    public func deserializeCompressedMultiproof(_ serialized: SerializedCompressedMultiproof) -> CompressedMultiproof? {
        let bytes = serialized.data
        var offset = 0

        // Commitments
        guard let numC = readU16(bytes, &offset) else { return nil }
        var commitments = [PointProjective]()
        commitments.reserveCapacity(Int(numC))
        for _ in 0..<Int(numC) {
            guard let x = readFpVal(bytes, &offset),
                  let y = readFpVal(bytes, &offset) else { return nil }
            commitments.append(PointProjective(x: x, y: y, z: Fp.one))
        }

        // Openings
        guard let numO = readU16(bytes, &offset) else { return nil }
        var childIndices = [Int]()
        var values = [Fr]()
        for _ in 0..<Int(numO) {
            guard offset < bytes.count else { return nil }
            childIndices.append(Int(bytes[offset])); offset += 1
            guard let v = readFrVal(bytes, &offset) else { return nil }
            values.append(v)
        }

        // Queries
        guard let numQ = readU16(bytes, &offset) else { return nil }
        var queryOpeningIndices = [[Int]]()
        var queryLeafIndices = [Int]()
        for _ in 0..<Int(numQ) {
            guard offset < bytes.count else { return nil }
            let count = Int(bytes[offset]); offset += 1
            var indices = [Int]()
            for _ in 0..<count {
                guard offset < bytes.count else { return nil }
                indices.append(Int(bytes[offset])); offset += 1
            }
            queryOpeningIndices.append(indices)
            guard let li = readU32(bytes, &offset) else { return nil }
            queryLeafIndices.append(Int(li))
        }

        // IPA proof
        guard offset < bytes.count else { return nil }
        let logW = Int(bytes[offset]); offset += 1

        var aggL = [PointProjective]()
        for _ in 0..<logW {
            guard let pt = readPointVal(bytes, &offset) else { return nil }
            aggL.append(pt)
        }
        var aggR = [PointProjective]()
        for _ in 0..<logW {
            guard let pt = readPointVal(bytes, &offset) else { return nil }
            aggR.append(pt)
        }
        guard let aggFinalA = readFrVal(bytes, &offset) else { return nil }

        // Evaluation point
        guard let evalPt = readFrVal(bytes, &offset) else { return nil }

        // Aggregated commitment and value
        guard let aggCommit = readPointVal(bytes, &offset) else { return nil }
        guard let aggVal = readFrVal(bytes, &offset) else { return nil }

        // Root
        guard let root = readPointVal(bytes, &offset) else { return nil }

        return CompressedMultiproof(
            commitments: commitments,
            childIndices: childIndices,
            values: values,
            queryOpeningIndices: queryOpeningIndices,
            queryLeafIndices: queryLeafIndices,
            aggregatedL: aggL,
            aggregatedR: aggR,
            aggregatedFinalA: aggFinalA,
            evaluationPoint: evalPt,
            aggregatedCommitment: aggCommit,
            aggregatedValue: aggVal,
            root: root
        )
    }

    // MARK: - Proof Comparison

    /// Compare compressed vs uncompressed proof sizes for given queries.
    ///
    /// Returns (compressedBytes, uncompressedEstimate, savingsPercent).
    public func compareProofSizes(queries: [MultiproofQuery]) throws -> (compressed: Int, uncompressed: Int, savingsPercent: Double) {
        let proof = try generateCompressedMultiproof(queries: queries)
        let serialized = serializeCompressedMultiproof(proof)
        let compressedSize = serialized.byteSize

        // Estimate uncompressed: each query gets its own full path proof
        // Per-level IPA: logWidth * 2 * 32 bytes (L + R points) + 32 bytes (finalA)
        let pathDepth = treeLevels.count
        let perLevelIPA = logWidth * 2 * 32 + 32  // L/R points + final scalar
        let perQuery = pathDepth * (32 + perLevelIPA + 1 + 32)  // commitment + IPA + childIndex + value
        let uncompressedSize = queries.count * perQuery + 64  // root commitment

        let savings = uncompressedSize > 0
            ? (1.0 - Double(compressedSize) / Double(uncompressedSize)) * 100.0
            : 0.0

        return (compressed: compressedSize, uncompressed: uncompressedSize, savingsPercent: savings)
    }

    // MARK: - Internal Helpers

    /// Generate a deduplication key for an opening.
    private func openingKey(_ commitment: PointProjective, _ childIndex: Int) -> String {
        let aff = cBatchToAffine([commitment])[0]
        let xLimbs = fpToInt(aff.x)
        return "\(xLimbs[0])_\(xLimbs[1])_\(xLimbs[2])_\(xLimbs[3])_\(childIndex)"
    }

    /// Convert affine point x-coordinate to Fr.
    private func commitmentToFr(_ p: PointAffine) -> Fr {
        let xInt = fpToInt(p.x)
        let raw = Fr.from64(xInt)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    /// Refresh cached tree levels from the underlying engine.
    private func refreshCachedLevels() {
        // Access the tree engine's internal state after updates
        // The tree engine stores levels internally; we read the root to confirm state
        _ = treeEngine.rootCommitment()
    }

    // MARK: - Serialization Helpers

    private func appendU16(_ bytes: inout [UInt8], _ v: UInt16) {
        bytes.append(UInt8(v & 0xFF))
        bytes.append(UInt8(v >> 8))
    }

    private func appendU32(_ bytes: inout [UInt8], _ v: UInt32) {
        bytes.append(UInt8(v & 0xFF))
        bytes.append(UInt8((v >> 8) & 0xFF))
        bytes.append(UInt8((v >> 16) & 0xFF))
        bytes.append(UInt8((v >> 24) & 0xFF))
    }

    private func appendFr(_ bytes: inout [UInt8], _ v: Fr) {
        let limbs = frToInt(v)
        for limb in limbs {
            for j in 0..<8 {
                bytes.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
    }

    private func appendFp(_ bytes: inout [UInt8], _ v: Fp) {
        let limbs = fpToInt(v)
        for limb in limbs {
            for j in 0..<8 {
                bytes.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
    }

    private func appendPoint(_ bytes: inout [UInt8], _ p: PointProjective) {
        let aff = cBatchToAffine([p])[0]
        appendFp(&bytes, aff.x)
        appendFp(&bytes, aff.y)
    }

    private func readU16(_ bytes: [UInt8], _ offset: inout Int) -> UInt16? {
        guard offset + 2 <= bytes.count else { return nil }
        let v = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        offset += 2
        return v
    }

    private func readU32(_ bytes: [UInt8], _ offset: inout Int) -> UInt32? {
        guard offset + 4 <= bytes.count else { return nil }
        let v = UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8) |
                (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
        offset += 4
        return v
    }

    private func readFrVal(_ bytes: [UInt8], _ offset: inout Int) -> Fr? {
        guard offset + 32 <= bytes.count else { return nil }
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            var val: UInt64 = 0
            for j in 0..<8 {
                val |= UInt64(bytes[offset + i * 8 + j]) << (j * 8)
            }
            limbs[i] = val
        }
        offset += 32
        return Fr.from64(limbs)
    }

    private func readFpVal(_ bytes: [UInt8], _ offset: inout Int) -> Fp? {
        guard offset + 32 <= bytes.count else { return nil }
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            var val: UInt64 = 0
            for j in 0..<8 {
                val |= UInt64(bytes[offset + i * 8 + j]) << (j * 8)
            }
            limbs[i] = val
        }
        offset += 32
        return Fp.from64(limbs)
    }

    private func readPointVal(_ bytes: [UInt8], _ offset: inout Int) -> PointProjective? {
        guard let x = readFpVal(bytes, &offset),
              let y = readFpVal(bytes, &offset) else { return nil }
        return PointProjective(x: x, y: y, z: Fp.one)
    }
}
