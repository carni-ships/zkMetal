// STIR Proof — Data structures for Shift To Improve Rate proofs
//
// STIR (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/390) is a proximity
// testing protocol that improves on FRI by applying algebraic domain shifts
// after each folding round. The shift decorrelates evaluation errors across
// rounds, achieving better soundness per query:
//
//   FRI:  error ~ rho       per query  -> O(lambda * log n) queries
//   STIR: error ~ rho^1.5   per query  -> O(log^2 n) queries
//
// Proof structure: per-round Merkle commitments, folding challenges (betas),
// shift challenges (alphas), query openings with Merkle paths, and final
// polynomial sent in the clear.

import Foundation

// MARK: - Query Opening

/// A single query opening: leaf value + Merkle authentication path.
public struct STIRQueryOpening2 {
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
public struct STIRRoundCommitment {
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

// MARK: - STIR Proof

/// Complete STIR proof for proximity testing.
///
/// The proof demonstrates that a committed polynomial is close to one of
/// degree < d, using iterative folding + domain shifting. The domain shift
/// after each fold is what distinguishes STIR from FRI — it causes the
/// proximity gap to improve multiplicatively (rho^1.5) rather than
/// additively (rho), reducing total query complexity.
public struct STIRProofData {
    /// Merkle root for each committed layer (layer 0 = original polynomial)
    public let roots: [Fr]

    /// Folding challenges (beta) used at each round
    public let betas: [Fr]

    /// Domain shift challenges (alpha) used at each round.
    /// This is the key STIR-specific component — not present in FRI or WHIR.
    public let alphas: [Fr]

    /// Per-round query openings.
    /// For round i: openings of layer i at positions needed to verify fold into layer i+1.
    /// Each opening contains (foldedIndex, [reductionFactor values], [reductionFactor Merkle paths]).
    public let layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]]

    /// Final polynomial evaluations (small degree, sent in the clear)
    public let finalPoly: [Fr]

    /// Number of folding rounds performed
    public let numRounds: Int

    /// Proof size in bytes (for benchmarking / comparison with FRI and WHIR).
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        var size = roots.count * frSize
        size += betas.count * frSize
        size += alphas.count * frSize
        for round in layerOpenings {
            for opening in round {
                size += 4  // index (UInt32)
                size += opening.values.count * frSize
                for path in opening.merklePaths {
                    size += path.count * frSize
                }
            }
        }
        size += finalPoly.count * frSize
        return size
    }

    /// Number of field elements in the proof (alternative size metric).
    public var proofFieldElements: Int {
        var count = roots.count + betas.count + alphas.count + finalPoly.count
        for round in layerOpenings {
            for opening in round {
                count += opening.values.count
                for path in opening.merklePaths {
                    count += path.count
                }
            }
        }
        return count
    }
}
