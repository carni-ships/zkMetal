// WHIR Proof — Data structures for Weighted Hashing IOP proofs
//
// WHIR (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/1586) replaces FRI's
// proximity testing with a sumcheck + hashing approach:
//   - O(log^2 n) query complexity (vs FRI's O(lambda * log n))
//   - Smaller proofs due to fewer queries
//   - Core soundness from weighted hash equations (sumcheck instances)
//
// Proof structure: per-round Merkle commitments, query openings with
// Merkle paths, weighted hash claims, and final polynomial.

import Foundation

// MARK: - Query Opening

/// A single query opening: leaf value + Merkle authentication path.
public struct WHIRQueryOpening {
    /// Leaf index in the evaluation domain
    public let index: UInt32
    /// Evaluation value f(omega^index)
    public let value: Fr
    /// Merkle authentication path from leaf to root
    public let merklePath: [Fr]

    public init(index: UInt32, value: Fr, merklePath: [Fr]) {
        self.index = index
        self.value = value
        self.merklePath = merklePath
    }
}

// MARK: - Round Commitment

/// Commitment to polynomial evaluations at one round of the protocol.
public struct WHIRRoundCommitment {
    /// Poseidon2 Merkle root binding the evaluations
    public let root: Fr
    /// Full Merkle tree (leaves + internal nodes)
    public let tree: [Fr]
    /// Polynomial evaluations on the domain
    public let evaluations: [Fr]

    public init(root: Fr, tree: [Fr], evaluations: [Fr]) {
        self.root = root
        self.tree = tree
        self.evaluations = evaluations
    }
}

// MARK: - Weighted Hash Claim

/// A weighted hash claim from the sumcheck reduction.
/// Asserts: sum_{i in S} w_i * f(x_i) = claimed_sum
/// where S is the query set, w_i are verifier-chosen weights,
/// and f(x_i) are the opened evaluations.
public struct WHIRWeightedHashClaim {
    /// Verifier-chosen random weights for each query position
    public let weights: [Fr]
    /// Claimed weighted sum
    public let claimedSum: Fr
    /// Query indices this claim covers
    public let queryIndices: [UInt32]

    public init(weights: [Fr], claimedSum: Fr, queryIndices: [UInt32]) {
        self.weights = weights
        self.claimedSum = claimedSum
        self.queryIndices = queryIndices
    }
}

// MARK: - WHIR Proof

/// Complete WHIR proof for proximity testing.
///
/// The proof demonstrates that a committed polynomial is close to one of
/// degree < d, using log(d/d_final) folding rounds with weighted hash
/// equations providing extra soundness per query.
public struct WHIRProofData {
    /// Merkle root for each committed layer (layer 0 = original polynomial)
    public let roots: [Fr]

    /// Folding challenges (beta) used at each round
    public let betas: [Fr]

    /// Per-round query openings.
    /// For round i: openings of layer i at positions needed to verify fold into layer i+1.
    /// Each opening contains (foldedIndex, [reductionFactor values], [reductionFactor Merkle paths]).
    public let layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]]

    /// Per-round weighted hash claims (the sumcheck component of WHIR).
    /// Each claim binds the opened values via a random linear combination,
    /// providing extra soundness bits beyond what Merkle opening alone gives.
    public let weightedHashClaims: [WHIRWeightedHashClaim]

    /// Final polynomial evaluations (small degree, sent in the clear)
    public let finalPoly: [Fr]

    /// Number of folding rounds performed
    public let numRounds: Int

    /// Proof size in bytes (for benchmarking / comparison with FRI).
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        var size = roots.count * frSize
        size += betas.count * frSize
        for round in layerOpenings {
            for opening in round {
                size += 4  // index (UInt32)
                size += opening.values.count * frSize
                for path in opening.merklePaths {
                    size += path.count * frSize
                }
            }
        }
        for claim in weightedHashClaims {
            size += claim.weights.count * frSize
            size += frSize  // claimedSum
            size += claim.queryIndices.count * 4
        }
        size += finalPoly.count * frSize
        return size
    }

    /// Number of field elements in the proof (alternative size metric).
    public var proofFieldElements: Int {
        var count = roots.count + betas.count + finalPoly.count
        for round in layerOpenings {
            for opening in round {
                count += opening.values.count
                for path in opening.merklePaths {
                    count += path.count
                }
            }
        }
        for claim in weightedHashClaims {
            count += claim.weights.count + 1  // weights + claimedSum
        }
        return count
    }
}

/// Backward-compatible typealias for the old WHIRProof name.
public typealias WHIRProof = WHIRProofData
