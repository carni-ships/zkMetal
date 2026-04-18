// BinaryMerkleTree — Merkle tree for binary field elements
//
// Implements Merkle tree construction and verification for binary FRI
// using a simple XOR-based hash (placeholder for Poseidon2 over GF(2^8)).
//
// For production, this would use Poseidon2 with the BinaryField domain,
// but this implementation provides the structural integration needed
// for the Binary-FRI system.

import Foundation

// MARK: - Binary Merkle Tree Parameters

/// Parameters for binary Merkle tree construction.
public struct BinaryMerkleParams {
    /// Log of the number of leaves
    public let logLeaves: Int

    /// Hash output size in bytes
    public let hashSize: Int

    public init(logLeaves: Int, hashSize: Int = 32) {
        self.logLeaves = logLeaves
        self.hashSize = hashSize
    }

    /// Total number of leaves
    public var numLeaves: Int { 1 << logLeaves }

    /// Total number of nodes in the tree
    public var numNodes: Int { 2 * numLeaves - 1 }
}

// MARK: - Binary Merkle Node

/// A node in the binary Merkle tree.
public enum BinaryMerkleNode: Equatable {
    /// Leaf node containing raw data
    case leaf(Data)

    /// Internal node containing hash
    case hash(Data)

    /// The hash value of this node
    public var hash: Data {
        switch self {
        case .leaf(let data): return data
        case .hash(let data): return data
        }
    }
}

// MARK: - Binary Merkle Tree

/// Merkle tree for binary field elements.
///
/// Uses a simple XOR-based hash for demonstration.
/// In production, would use Poseidon2 over the binary field.
public struct BinaryMerkleTree {

    /// The tree nodes in breadth-first order
    public let nodes: [BinaryMerkleNode]

    /// The root hash
    public var root: Data { nodes.last?.hash ?? Data() }

    /// Tree parameters
    public let params: BinaryMerkleParams

    /// Create a Merkle tree from binary field evaluations.
    public init(evaluations: [UInt8], params: BinaryMerkleParams) {
        self.params = params
        self.nodes = BinaryMerkleTree.buildTree(evaluations: evaluations, params: params)
    }

    /// Build the Merkle tree recursively.
    private static func buildTree(evaluations: [UInt8], params: BinaryMerkleParams) -> [BinaryMerkleNode] {
        let numLeaves = params.numLeaves
        var currentLevel = [BinaryMerkleNode]()

        // Pad evaluations to numLeaves if needed
        var paddedEvals = evaluations
        while paddedEvals.count < numLeaves {
            paddedEvals.append(0)
        }

        // Create leaf nodes
        for i in 0..<numLeaves {
            let leafData = Data([paddedEvals[i]])
            currentLevel.append(.leaf(leafData))
        }

        var allNodes = currentLevel

        // Build internal nodes
        while currentLevel.count > 1 {
            var nextLevel = [BinaryMerkleNode]()
            for i in stride(from: 0, to: currentLevel.count, by: 2) {
                let left = currentLevel[i].hash
                let right = i + 1 < currentLevel.count ? currentLevel[i + 1].hash : left
                let parentHash = BinaryMerkleTree.combineHash(left: left, right: right)
                nextLevel.append(.hash(parentHash))
            }
            allNodes.append(contentsOf: nextLevel)
            currentLevel = nextLevel
        }

        return allNodes
    }

    /// Combine two hashes to form a parent node.
    /// Uses XOR-based combination for binary fields.
    private static func combineHash(left: Data, right: Data) -> Data {
        var result = Data(count: max(left.count, right.count))
        for i in 0..<result.count {
            let l = i < left.count ? left[i] : 0
            let r = i < right.count ? right[i] : 0
            result[i] = l ^ r
        }
        return result
    }

    /// Get the authentication path for a leaf.
    public func getAuthPath(leafIndex: Int) -> [Data] {
        var path = [Data]()
        var idx = leafIndex
        var levelSize = params.numLeaves

        while levelSize > 1 {
            let siblingIdx = idx ^ 1  // XOR to get sibling
            let nodeIndex = params.numNodes - levelSize + siblingIdx
            if nodeIndex < nodes.count {
                path.append(nodes[nodeIndex].hash)
            }
            idx = idx / 2
            levelSize = levelSize / 2
        }

        return path
    }

    /// Verify an authentication path against the root.
    public static func verifyAuthPath(
        leafHash: Data,
        leafIndex: Int,
        authPath: [Data],
        root: Data
    ) -> Bool {
        var currentHash = leafHash
        var idx = leafIndex

        for siblingHash in authPath {
            let left = idx % 2 == 0 ? currentHash : siblingHash
            let right = idx % 2 == 0 ? siblingHash : currentHash
            currentHash = combineHash(left: left, right: right)
            idx = idx / 2
        }

        return currentHash == root
    }
}

// MARK: - Binary FRI Merkle Commitment

/// Merkle commitment for binary FRI.
public struct BinaryFRIMerkleCommitment {
    /// The Merkle root
    public let root: Data

    /// Number of leaves (domain size)
    public let numLeaves: Int

    /// Log of the number of leaves
    public let logLeaves: Int

    public init(root: Data, numLeaves: Int) {
        self.root = root
        self.numLeaves = numLeaves
        self.logLeaves = Int(log2(Double(numLeaves)))
    }
}

/// Authentication path for binary FRI verification.
public struct BinaryFRIMerkleProof {
    /// The leaf hash being proven
    public let leafHash: Data

    /// The index of the leaf
    public let leafIndex: Int

    /// Authentication path hashes
    public let authPath: [Data]
}

// MARK: - Binary FRI Merkle Verifier

/// Verifier for binary FRI Merkle proofs.
public struct BinaryFRIMerkleVerifier {

    /// Verify a Merkle proof.
    public static func verify(
        proof: BinaryFRIMerkleProof,
        root: Data
    ) -> Bool {
        return BinaryMerkleTree.verifyAuthPath(
            leafHash: proof.leafHash,
            leafIndex: proof.leafIndex,
            authPath: proof.authPath,
            root: root
        )
    }

    /// Verify multiple proofs (batch verification).
    public static func verifyBatch(
        proofs: [BinaryFRIMerkleProof],
        root: Data
    ) -> Bool {
        for proof in proofs {
            if !verify(proof: proof, root: root) {
                return false
            }
        }
        return true
    }
}
