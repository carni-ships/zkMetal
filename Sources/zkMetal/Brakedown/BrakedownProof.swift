// Brakedown Proof and Commitment data structures
// Separated for clean API boundaries.
//
// Proof structure:
//   - Merkle root (commitment)
//   - Column openings at queried indices + Merkle authentication paths
//   - t-vector: linear combination of matrix rows for evaluation binding
//
// Proof size: O(sqrt(n)) field elements + O(sqrt(n) * log(n)) hash digests

import Foundation

// MARK: - Commitment

/// Commitment to a multilinear polynomial via Brakedown.
/// The polynomial evaluations are arranged as a sqrt(n) x sqrt(n) matrix,
/// each row is encoded with an expander code, and columns are Merkle-committed.
public struct BrakedownCommitment {
    /// Merkle root of the column hashes — this is the binding commitment
    public let merkleRoot: Fr
    /// Number of rows in the evaluation matrix
    public let numRows: Int
    /// Number of columns before encoding
    public let numCols: Int
    /// Number of columns after encoding (numCols * rateInverse)
    public let numEncodedCols: Int
    /// Full Merkle tree nodes for proof extraction (prover-side only)
    public let tree: [Fr]
    /// Encoded matrix (numRows x numEncodedCols) — prover keeps for opening
    public let encodedMatrix: [Fr]

    /// Commitment size: just the Merkle root (1 field element / 32 bytes)
    public var commitmentSize: Int { MemoryLayout<Fr>.stride }

    /// Create a commitment (typically called by the engine, not directly)
    public init(merkleRoot: Fr, numRows: Int, numCols: Int,
                numEncodedCols: Int, tree: [Fr], encodedMatrix: [Fr]) {
        self.merkleRoot = merkleRoot
        self.numRows = numRows
        self.numCols = numCols
        self.numEncodedCols = numEncodedCols
        self.tree = tree
        self.encodedMatrix = encodedMatrix
    }
}

// MARK: - Opening Proof

/// Brakedown opening proof for a multilinear evaluation.
///
/// Given commitment C to polynomial p, and evaluation p(z) = v,
/// the proof convinces a verifier that v is correct by:
/// 1. Revealing t = M^T * tensor_left (bound to the evaluation point)
/// 2. Opening random columns with Merkle proofs (proximity test)
/// 3. Verifier checks: value = <tensor_right, t> and column consistency
///
/// Proof size: O(sqrt(n)) field elements + O(numQueries * log(n)) hashes
public struct BrakedownProof {
    /// Column openings: for each queried column, the full column vector (numRows elements)
    public let columnOpenings: [[Fr]]
    /// Merkle authentication paths for each queried column
    public let merkleProofs: [[Fr]]
    /// Which columns were queried (derived deterministically from commitment)
    public let queryIndices: [Int]
    /// t = M^T * tensor_left: a vector of length numCols
    /// Binds the evaluation to the committed matrix rows.
    /// Verifier checks: value = <tensor_right, t> and encode(t)[j] = <tensor_left, col_j>
    public let tVector: [Fr]

    /// Approximate proof size in bytes
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        let numQueries = queryIndices.count
        let numRows = columnOpenings.isEmpty ? 0 : columnOpenings[0].count
        let tSize = tVector.count * frSize
        let columnSize = numQueries * numRows * frSize
        let merkleSize = merkleProofs.reduce(0) { $0 + $1.count } * frSize
        return tSize + columnSize + merkleSize
    }

    /// Create a proof (typically called by the engine, not directly)
    public init(columnOpenings: [[Fr]], merkleProofs: [[Fr]],
                queryIndices: [Int], tVector: [Fr]) {
        self.columnOpenings = columnOpenings
        self.merkleProofs = merkleProofs
        self.queryIndices = queryIndices
        self.tVector = tVector
    }
}

// MARK: - Brakedown Parameters

/// Configuration for Brakedown PCS instances.
/// Controls the tradeoff between proof size, prover time, and soundness.
public struct BrakedownParameters {
    /// Rate inverse (blowup factor). Codeword = message * rateInverse. Default: 4
    public let rateInverse: Int
    /// Number of random column queries for soundness. Default: 30
    /// Security ≈ numQueries * log2(1/rate) bits
    public let numQueries: Int
    /// Expander graph degree (edges per right vertex). Default: 10
    /// Higher = better distance but slower encoding
    public let expanderDegree: Int
    /// Seed for deterministic code generation
    public let codeSeed: UInt32

    /// Default parameters: 4x blowup, 30 queries, degree-10 expander
    /// Gives ~60 bits of soundness (30 * log2(4) = 60)
    public static let `default` = BrakedownParameters(
        rateInverse: 4, numQueries: 30, expanderDegree: 10, codeSeed: 0xBEEF
    )

    /// High security: 8x blowup, 50 queries, degree-16 expander
    /// Gives ~150 bits of soundness (50 * log2(8) = 150)
    public static let highSecurity = BrakedownParameters(
        rateInverse: 8, numQueries: 50, expanderDegree: 16, codeSeed: 0xBEEF
    )

    /// Fast prover: 4x blowup, 20 queries, degree-8 expander
    /// Lower security (~40 bits) but faster. Suitable for testing.
    public static let fast = BrakedownParameters(
        rateInverse: 4, numQueries: 20, expanderDegree: 8, codeSeed: 0xBEEF
    )

    public init(rateInverse: Int, numQueries: Int, expanderDegree: Int, codeSeed: UInt32) {
        self.rateInverse = rateInverse
        self.numQueries = numQueries
        self.expanderDegree = expanderDegree
        self.codeSeed = codeSeed
    }

    /// Estimated soundness in bits
    public var soundnessBits: Double {
        return Double(numQueries) * log2(Double(rateInverse))
    }
}
