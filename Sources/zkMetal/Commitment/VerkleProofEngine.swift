// VerkleProofEngine — Complete Verkle tree proof engine with IPA-based polynomial commitments
//
// Implements the Ethereum Verkle tree proof protocol (EIP-6800 compatible) using:
//   - Pedersen vector commitments over the Banderwagon curve
//   - IPA (Inner Product Argument) opening proofs
//   - Width-256 branching factor (Ethereum spec)
//   - Multi-proof batching for efficient multi-key proofs
//   - Proof serialization for Ethereum compatibility
//
// Architecture:
//   VerkleProofEngine orchestrates the full proof lifecycle:
//     1. Tree construction with Pedersen commitments at each node
//     2. Single-key proof generation (path proof with IPA openings)
//     3. Multi-key batch proof (combined IPA with random evaluation)
//     4. Proof verification (single and batch)
//     5. Proof-of-absence for non-existent keys
//     6. Tree updates with commitment maintenance
//     7. Serialization/deserialization for Ethereum wire format
//
// References:
//   - EIP-6800: Ethereum state Verkle trees
//   - Verkle trees (Kuszmaul 2019)
//   - IPA/Bulletproofs (Bunz et al. 2018)

import Foundation
import NeonFieldOps

// MARK: - Verkle Commitment (Pedersen-based)

/// A Pedersen vector commitment at a Verkle tree node.
/// Wraps a Banderwagon extended point with metadata for proof construction.
public struct VerkleNodeCommitment {
    /// The commitment point (Pedersen vector commitment over Banderwagon).
    public let point: BanderwagonExtended
    /// The child values that were committed (width-many field elements).
    public let childValues: [Fr381]

    public init(point: BanderwagonExtended, childValues: [Fr381]) {
        self.point = point
        self.childValues = childValues
    }

    /// Serialize the commitment to 32 bytes (Banderwagon y-coordinate).
    public func serialize() -> [UInt8] {
        return bwSerialize(point)
    }

    /// Deserialize a commitment from 32 bytes.
    public static func deserialize(_ bytes: [UInt8]) -> VerkleNodeCommitment? {
        guard let pt = bwDeserialize(bytes) else { return nil }
        return VerkleNodeCommitment(point: pt, childValues: [])
    }
}

// MARK: - Serialized Proof Format (Ethereum-compatible)

/// Serialized Verkle proof suitable for Ethereum wire format.
/// Layout:
///   - 1 byte: number of path commitments (depth)
///   - For each commitment: 32 bytes (serialized Banderwagon point)
///   - For each IPA proof: L count (1 byte) + L points + R points + final scalar (32 bytes)
///   - 1 byte: extension status
///   - 31 bytes: stem
///   - 1 byte: suffix index
///   - 1 byte: depth
public struct SerializedVerkleProof {
    public let data: [UInt8]

    public init(data: [UInt8]) {
        self.data = data
    }

    /// Serialize a VerkleProof to bytes.
    public static func serialize(_ proof: VerkleProof) -> SerializedVerkleProof {
        var bytes = [UInt8]()
        bytes.reserveCapacity(2048) // typical proof size

        // Number of commitments
        bytes.append(UInt8(proof.commitments.count))

        // Commitments
        for c in proof.commitments {
            bytes.append(contentsOf: bwSerialize(c))
        }

        // IPA proofs
        bytes.append(UInt8(proof.ipaProofs.count))
        for ipa in proof.ipaProofs {
            let logN = ipa.L.count
            bytes.append(UInt8(logN))
            for l in ipa.L { bytes.append(contentsOf: bwSerialize(l)) }
            for r in ipa.R { bytes.append(contentsOf: bwSerialize(r)) }
            // Final scalar a (32 bytes, little-endian)
            let aLimbs = [ipa.a.limbs.0, ipa.a.limbs.1, ipa.a.limbs.2, ipa.a.limbs.3]
            for limb in aLimbs {
                for j in 0..<8 {
                    bytes.append(UInt8((limb >> (j * 8)) & 0xFF))
                }
            }
        }

        // Extension status
        bytes.append(proof.extensionStatus.rawValue)

        // Stem (31 bytes)
        precondition(proof.stem.count == 31)
        bytes.append(contentsOf: proof.stem)

        // Suffix index
        bytes.append(proof.suffixIndex)

        // Depth
        bytes.append(UInt8(proof.depth))

        return SerializedVerkleProof(data: bytes)
    }

    /// Deserialize a VerkleProof from bytes.
    public static func deserialize(_ serialized: SerializedVerkleProof) -> VerkleProof? {
        let bytes = serialized.data
        var offset = 0

        guard offset < bytes.count else { return nil }
        let numCommitments = Int(bytes[offset]); offset += 1

        var commitments = [BanderwagonExtended]()
        for _ in 0..<numCommitments {
            guard offset + 32 <= bytes.count else { return nil }
            let ptBytes = Array(bytes[offset..<offset+32]); offset += 32
            guard let pt = bwDeserialize(ptBytes) else { return nil }
            commitments.append(pt)
        }

        guard offset < bytes.count else { return nil }
        let numIPAProofs = Int(bytes[offset]); offset += 1

        var ipaProofs = [BanderwagonIPAProof]()
        for _ in 0..<numIPAProofs {
            guard offset < bytes.count else { return nil }
            let logN = Int(bytes[offset]); offset += 1

            var Ls = [BanderwagonExtended]()
            for _ in 0..<logN {
                guard offset + 32 <= bytes.count else { return nil }
                let lBytes = Array(bytes[offset..<offset+32]); offset += 32
                guard let l = bwDeserialize(lBytes) else { return nil }
                Ls.append(l)
            }

            var Rs = [BanderwagonExtended]()
            for _ in 0..<logN {
                guard offset + 32 <= bytes.count else { return nil }
                let rBytes = Array(bytes[offset..<offset+32]); offset += 32
                guard let r = bwDeserialize(rBytes) else { return nil }
                Rs.append(r)
            }

            // Final scalar a
            guard offset + 32 <= bytes.count else { return nil }
            var aLimbs = [UInt64](repeating: 0, count: 4)
            for i in 0..<4 {
                var word: UInt64 = 0
                for j in 0..<8 {
                    word |= UInt64(bytes[offset]) << (j * 8)
                    offset += 1
                }
                aLimbs[i] = word
            }
            let a = BwScalar(aLimbs[0], aLimbs[1], aLimbs[2], aLimbs[3])

            ipaProofs.append(BanderwagonIPAProof(L: Ls, R: Rs, a: a))
        }

        guard offset < bytes.count else { return nil }
        guard let status = VerkleExtensionStatus(rawValue: bytes[offset]) else { return nil }
        offset += 1

        guard offset + 31 <= bytes.count else { return nil }
        let stem = Array(bytes[offset..<offset+31]); offset += 31

        guard offset < bytes.count else { return nil }
        let suffixIndex = bytes[offset]; offset += 1

        guard offset < bytes.count else { return nil }
        let depth = Int(bytes[offset]); offset += 1

        return VerkleProof(
            commitments: commitments,
            ipaProofs: ipaProofs,
            extensionStatus: status,
            stem: stem,
            suffixIndex: suffixIndex,
            depth: depth)
    }
}

/// Serialized multi-proof format.
public struct SerializedVerkleMultiProof {
    public let data: [UInt8]

    public init(data: [UInt8]) {
        self.data = data
    }

    /// Serialize a VerkleMultiProof.
    public static func serialize(_ proof: VerkleMultiProof) -> SerializedVerkleMultiProof {
        var bytes = [UInt8]()
        bytes.reserveCapacity(4096)

        // Number of unique commitments
        let numC = UInt16(proof.commitments.count)
        bytes.append(UInt8(numC & 0xFF))
        bytes.append(UInt8(numC >> 8))

        for c in proof.commitments {
            bytes.append(contentsOf: bwSerialize(c))
        }

        // IPA proof
        let logN = proof.ipaProof.L.count
        bytes.append(UInt8(logN))
        for l in proof.ipaProof.L { bytes.append(contentsOf: bwSerialize(l)) }
        for r in proof.ipaProof.R { bytes.append(contentsOf: bwSerialize(r)) }
        let aLimbs = [proof.ipaProof.a.limbs.0, proof.ipaProof.a.limbs.1,
                      proof.ipaProof.a.limbs.2, proof.ipaProof.a.limbs.3]
        for limb in aLimbs {
            for j in 0..<8 { bytes.append(UInt8((limb >> (j * 8)) & 0xFF)) }
        }

        // Number of keys
        let numKeys = UInt16(proof.extensionStatuses.count)
        bytes.append(UInt8(numKeys & 0xFF))
        bytes.append(UInt8(numKeys >> 8))

        // Per-key data
        for i in 0..<Int(numKeys) {
            bytes.append(proof.extensionStatuses[i].rawValue)
            precondition(proof.stems[i].count == 31)
            bytes.append(contentsOf: proof.stems[i])
            bytes.append(proof.suffixIndices[i])
            bytes.append(UInt8(proof.depths[i]))
        }

        // Evaluation point (Fr381 in Montgomery form, serialize as 32 LE bytes)
        let evalLimbs = fr381ToInt(proof.evaluationPoint)
        for limb in evalLimbs {
            for j in 0..<8 { bytes.append(UInt8((limb >> (j * 8)) & 0xFF)) }
        }

        return SerializedVerkleMultiProof(data: bytes)
    }

    /// Deserialize a VerkleMultiProof.
    public static func deserialize(_ serialized: SerializedVerkleMultiProof) -> VerkleMultiProof? {
        let bytes = serialized.data
        var offset = 0

        guard offset + 2 <= bytes.count else { return nil }
        let numC = Int(UInt16(bytes[offset]) | (UInt16(bytes[offset+1]) << 8)); offset += 2

        var commitments = [BanderwagonExtended]()
        for _ in 0..<numC {
            guard offset + 32 <= bytes.count else { return nil }
            guard let pt = bwDeserialize(Array(bytes[offset..<offset+32])) else { return nil }
            commitments.append(pt); offset += 32
        }

        guard offset < bytes.count else { return nil }
        let logN = Int(bytes[offset]); offset += 1

        var Ls = [BanderwagonExtended]()
        for _ in 0..<logN {
            guard offset + 32 <= bytes.count else { return nil }
            guard let l = bwDeserialize(Array(bytes[offset..<offset+32])) else { return nil }
            Ls.append(l); offset += 32
        }
        var Rs = [BanderwagonExtended]()
        for _ in 0..<logN {
            guard offset + 32 <= bytes.count else { return nil }
            guard let r = bwDeserialize(Array(bytes[offset..<offset+32])) else { return nil }
            Rs.append(r); offset += 32
        }

        guard offset + 32 <= bytes.count else { return nil }
        var aLimbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            var word: UInt64 = 0
            for j in 0..<8 { word |= UInt64(bytes[offset]) << (j * 8); offset += 1 }
            aLimbs[i] = word
        }
        let ipaProof = BanderwagonIPAProof(
            L: Ls, R: Rs,
            a: BwScalar(aLimbs[0], aLimbs[1], aLimbs[2], aLimbs[3]))

        guard offset + 2 <= bytes.count else { return nil }
        let numKeys = Int(UInt16(bytes[offset]) | (UInt16(bytes[offset+1]) << 8)); offset += 2

        var statuses = [VerkleExtensionStatus]()
        var stems = [[UInt8]]()
        var suffixes = [UInt8]()
        var depths = [Int]()

        for _ in 0..<numKeys {
            guard offset + 34 <= bytes.count else { return nil }
            guard let s = VerkleExtensionStatus(rawValue: bytes[offset]) else { return nil }
            statuses.append(s); offset += 1
            stems.append(Array(bytes[offset..<offset+31])); offset += 31
            suffixes.append(bytes[offset]); offset += 1
            depths.append(Int(bytes[offset])); offset += 1
        }

        guard offset + 32 <= bytes.count else { return nil }
        var evalLimbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            var word: UInt64 = 0
            for j in 0..<8 { word |= UInt64(bytes[offset]) << (j * 8); offset += 1 }
            evalLimbs[i] = word
        }
        let evalRaw = Fr381.from64(evalLimbs)
        let evalPoint = fr381Mul(evalRaw, Fr381.from64(Fr381.R2_MOD_R))

        return VerkleMultiProof(
            commitments: commitments,
            ipaProof: ipaProof,
            extensionStatuses: statuses,
            stems: stems,
            suffixIndices: suffixes,
            depths: depths,
            evaluationPoint: evalPoint)
    }
}

// MARK: - Verkle Proof Engine

/// Complete Verkle proof engine with IPA-based polynomial commitments.
///
/// Provides the full lifecycle for Ethereum-compatible Verkle tree proofs:
///   - Tree construction and updates
///   - Single-key and multi-key proof generation
///   - Proof verification (inclusion and absence)
///   - Serialization for wire format
///
/// Uses width-256 branching factor per the Ethereum Verkle specification.
public class VerkleProofEngine {
    /// The underlying IPA engine over Banderwagon.
    public let ipaEngine: BanderwagonIPAEngine

    /// The branching factor (256 for Ethereum).
    public let width: Int

    /// Domain separator for transcript.
    private static let domainSep = "VerkleProofEngine-v1"

    /// Create a Verkle proof engine with the specified width.
    /// - Parameter width: branching factor (must be power of 2, default 256 for Ethereum)
    public init(width: Int = 256) {
        precondition(width > 0 && (width & (width - 1)) == 0, "Width must be power of 2")
        self.width = width
        self.ipaEngine = BanderwagonIPAEngine(width: width)
    }

    /// Create a Verkle proof engine with an existing IPA engine.
    public init(ipaEngine: BanderwagonIPAEngine) {
        self.width = ipaEngine.n
        self.ipaEngine = ipaEngine
    }

    // MARK: - Tree Construction

    /// Build a new Verkle tree (convenience wrapper).
    public func createTree() -> VerkleTree {
        return VerkleTree(ipaEngine: ipaEngine)
    }

    /// Commit to a vector of field elements at a single node.
    /// Returns a VerkleNodeCommitment wrapping the Pedersen commitment.
    public func commitNode(values: [Fr381]) -> VerkleNodeCommitment {
        precondition(values.count == width)
        let point = ipaEngine.commitFr381(values)
        return VerkleNodeCommitment(point: point, childValues: values)
    }

    // MARK: - Single Key Proof

    /// Generate a proof for a single key in a tree.
    /// Computes commitments if needed, then generates path proof with IPA openings.
    ///
    /// - Parameters:
    ///   - tree: the Verkle tree
    ///   - key: 32-byte key (31-byte stem + 1-byte suffix)
    /// - Returns: VerkleProof with path commitments and IPA proofs
    public func generateProof(tree: VerkleTree, key: [UInt8]) -> VerkleProof {
        return tree.generateProof(key: key)
    }

    /// Verify a single-key proof against a root commitment.
    ///
    /// - Parameters:
    ///   - root: the tree root commitment
    ///   - key: the 32-byte key
    ///   - value: expected value (nil for absence proof)
    ///   - proof: the proof to verify
    /// - Returns: true if the proof is valid
    public func verifyProof(
        root: BanderwagonExtended,
        key: [UInt8],
        value: Fr381?,
        proof: VerkleProof
    ) -> Bool {
        return VerkleTree.verifyProof(
            root: root, key: key, value: value,
            proof: proof, ipaEngine: ipaEngine)
    }

    // MARK: - Multi-Key Batch Proof

    /// Generate a batch proof for multiple keys.
    /// All openings are combined into a single IPA proof using a random evaluation point.
    ///
    /// - Parameters:
    ///   - tree: the Verkle tree
    ///   - keys: array of 32-byte keys
    /// - Returns: VerkleMultiProof with batched IPA
    public func generateMultiProof(tree: VerkleTree, keys: [[UInt8]]) -> VerkleMultiProof {
        return tree.generateMultiProof(keys: keys)
    }

    /// Verify a multi-key batch proof against a root commitment.
    ///
    /// - Parameters:
    ///   - root: the tree root commitment
    ///   - keys: the keys included in the proof
    ///   - values: expected values (nil entries for absent keys)
    ///   - proof: the multi-proof to verify
    /// - Returns: true if the proof is valid
    public func verifyMultiProof(
        root: BanderwagonExtended,
        keys: [[UInt8]],
        values: [Fr381?],
        proof: VerkleMultiProof
    ) -> Bool {
        return VerkleTree.verifyMultiProof(
            root: root, keys: keys, values: values,
            proof: proof, ipaEngine: ipaEngine)
    }

    // MARK: - IPA Opening Proof for Internal Nodes

    /// Create an IPA opening proof for a specific child index at a given node.
    /// This proves that the node commitment opens to a particular value at the given index.
    ///
    /// - Parameters:
    ///   - nodeCommitment: the committed node
    ///   - childIndex: the index to open (0..<width)
    /// - Returns: the IPA proof and the opened value
    public func createNodeOpeningProof(
        nodeCommitment: VerkleNodeCommitment,
        childIndex: Int
    ) -> (proof: BanderwagonIPAProof, value: BwScalar) {
        precondition(childIndex >= 0 && childIndex < width)

        let aQ = nodeCommitment.childValues.map { bwScalarFromFr381($0) }
        var bQ = [BwScalar](repeating: .zero, count: width)
        bQ[childIndex] = .one

        let v = aQ[childIndex]
        let proof = ipaEngine.createProof(a: aQ, b: bQ)
        return (proof: proof, value: v)
    }

    /// Verify an IPA opening proof for a node at a given child index.
    ///
    /// - Parameters:
    ///   - commitment: the node's Banderwagon commitment point
    ///   - childIndex: the index being opened
    ///   - value: the claimed opened value
    ///   - proof: the IPA proof
    /// - Returns: true if the proof verifies
    public func verifyNodeOpeningProof(
        commitment: BanderwagonExtended,
        childIndex: Int,
        value: BwScalar,
        proof: BanderwagonIPAProof
    ) -> Bool {
        var bQ = [BwScalar](repeating: .zero, count: width)
        bQ[childIndex] = .one

        let Cbound = bwAdd(commitment, bwScalarMulQ(ipaEngine.Q, value))
        return ipaEngine.verify(
            commitment: Cbound, b: bQ,
            innerProductValue: value, proof: proof)
    }

    // MARK: - Tree Update with Proof Maintenance

    /// Update a key-value pair in the tree and recompute affected commitments.
    /// Returns the new root commitment after the update.
    ///
    /// - Parameters:
    ///   - tree: the tree to update
    ///   - key: 32-byte key
    ///   - newValue: new value for the key
    /// - Returns: new root commitment
    @discardableResult
    public func updateAndRecommit(
        tree: VerkleTree,
        key: [UInt8],
        newValue: Fr381
    ) -> BanderwagonExtended {
        tree.insert(key: key, value: newValue)
        tree.computeCommitments()
        return tree.rootCommitment()
    }

    /// Batch update multiple key-value pairs and recompute commitments.
    ///
    /// - Parameters:
    ///   - tree: the tree to update
    ///   - updates: array of (key, value) pairs
    /// - Returns: new root commitment
    @discardableResult
    public func batchUpdateAndRecommit(
        tree: VerkleTree,
        updates: [([UInt8], Fr381)]
    ) -> BanderwagonExtended {
        for (key, value) in updates {
            tree.insert(key: key, value: value)
        }
        tree.computeCommitments()
        return tree.rootCommitment()
    }

    // MARK: - Proof Serialization

    /// Serialize a single-key proof to bytes (Ethereum wire format).
    public func serializeProof(_ proof: VerkleProof) -> [UInt8] {
        return SerializedVerkleProof.serialize(proof).data
    }

    /// Deserialize a single-key proof from bytes.
    public func deserializeProof(_ bytes: [UInt8]) -> VerkleProof? {
        return SerializedVerkleProof.deserialize(SerializedVerkleProof(data: bytes))
    }

    /// Serialize a multi-key proof to bytes.
    public func serializeMultiProof(_ proof: VerkleMultiProof) -> [UInt8] {
        return SerializedVerkleMultiProof.serialize(proof).data
    }

    /// Deserialize a multi-key proof from bytes.
    public func deserializeMultiProof(_ bytes: [UInt8]) -> VerkleMultiProof? {
        return SerializedVerkleMultiProof.deserialize(SerializedVerkleMultiProof(data: bytes))
    }

    // MARK: - Proof of Absence

    /// Generate a proof that a key does NOT exist in the tree.
    /// The proof shows either an empty slot or a different stem at the expected position.
    ///
    /// - Parameters:
    ///   - tree: the Verkle tree
    ///   - key: the key to prove absent
    /// - Returns: VerkleProof with extensionStatus of .absent or .otherStem
    public func generateAbsenceProof(tree: VerkleTree, key: [UInt8]) -> VerkleProof {
        let proof = tree.generateProof(key: key)
        precondition(proof.extensionStatus == .absent || proof.extensionStatus == .otherStem,
                     "Key exists in tree; cannot generate absence proof")
        return proof
    }

    /// Verify that a proof demonstrates absence of a key.
    ///
    /// - Parameters:
    ///   - root: the tree root commitment
    ///   - key: the key claimed to be absent
    ///   - proof: the absence proof
    /// - Returns: true if the proof validly demonstrates absence
    public func verifyAbsenceProof(
        root: BanderwagonExtended,
        key: [UInt8],
        proof: VerkleProof
    ) -> Bool {
        guard proof.extensionStatus == .absent || proof.extensionStatus == .otherStem else {
            return false
        }
        return VerkleTree.verifyProof(
            root: root, key: key, value: nil,
            proof: proof, ipaEngine: ipaEngine)
    }

    // MARK: - Utility

    /// Make a 32-byte key from a stem and suffix.
    public static func makeKey(stem: [UInt8], suffix: UInt8) -> [UInt8] {
        precondition(stem.count == 31)
        return stem + [suffix]
    }

    /// Extract the stem (first 31 bytes) from a 32-byte key.
    public static func stemFromKey(_ key: [UInt8]) -> [UInt8] {
        precondition(key.count == 32)
        return Array(key.prefix(31))
    }

    /// Extract the suffix (last byte) from a 32-byte key.
    public static func suffixFromKey(_ key: [UInt8]) -> UInt8 {
        precondition(key.count == 32)
        return key[31]
    }
}
