// WHIR Engine — Weighted Hashing IOP for Reed-Solomon proximity testing
//
// WHIR (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/1586) is a proximity
// testing protocol that replaces FRI with a sumcheck + hashing approach.
//
// Key differences from FRI:
//   - After folding, the prover computes a *weighted hash* of opened values
//   - The weighted hash is a random linear combination (sumcheck instance)
//   - This provides extra soundness per query: ~2 bits vs FRI's ~1 bit
//   - Result: O(log^2 n) queries vs FRI's O(lambda * log n)
//   - Smaller proofs for the same security level
//
// Protocol (each round):
//   1. Commit to polynomial evaluations via Poseidon2 Merkle tree
//   2. Verifier sends folding challenge beta (Fiat-Shamir)
//   3. Prover folds polynomial: degree halved by reductionFactor
//   4. Verifier sends random weights w_1, ..., w_q
//   5. Prover computes weighted hash: h = sum_i w_i * f(x_i)
//   6. Repeat until polynomial is small; send final poly in the clear
//   7. Verifier checks Merkle proofs + fold consistency + weighted hash
//
// GPU acceleration: Merkle tree construction uses Metal Poseidon2 for
// large domains (>1024 leaves). Polynomial folding uses C CIOS arithmetic.

import Foundation
import NeonFieldOps

public class WHIRProver {
    public static let version = "2.0.0"

    /// Number of queries per round
    public let numQueries: Int
    /// Folding factor (must be power of 2)
    public let reductionFactor: Int
    /// log2(reductionFactor)
    public let logReduction: Int

    private let merkleEngine: Poseidon2MerkleEngine

    /// CPU Merkle threshold: use CPU Poseidon2 for small trees to avoid
    /// GPU command buffer overhead (~5-9ms per dispatch).
    /// GCD dispatch_apply gives near-zero threading overhead, so CPU is
    /// competitive up to ~4096 leaves on Apple Silicon.
    private static let cpuMerkleThreshold = 4096

    /// Initialize WHIR prover.
    /// - Parameters:
    ///   - numQueries: queries per round (default 4; more = higher security)
    ///   - reductionFactor: degree reduction per round (default 4; must be power of 2)
    public init(numQueries: Int = 4, reductionFactor: Int = 4) throws {
        precondition(reductionFactor >= 2 && (reductionFactor & (reductionFactor - 1)) == 0,
                     "reductionFactor must be a power of 2")
        self.numQueries = numQueries
        self.reductionFactor = reductionFactor
        self.logReduction = Int(log2(Double(reductionFactor)))
        self.merkleEngine = try Poseidon2MerkleEngine()
    }

    // MARK: - Commit

    /// Commit to polynomial evaluations via Poseidon2 Merkle tree.
    /// Uses CPU path for small trees, GPU for large ones.
    public func commit(evaluations: [Fr]) throws -> WHIRRoundCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")

        let tree: [Fr]
        if n <= WHIRProver.cpuMerkleThreshold {
            let treeSize = 2 * n - 1
            var treeArr = [Fr](repeating: Fr.zero, count: treeSize)
            evaluations.withUnsafeBytes { evPtr in
                treeArr.withUnsafeMutableBytes { treePtr in
                    poseidon2_merkle_tree_cpu(
                        evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        treePtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
            tree = treeArr
        } else {
            tree = try merkleEngine.buildTree(evaluations)
        }

        let root = tree[2 * n - 2]
        return WHIRRoundCommitment(root: root, tree: tree, evaluations: evaluations)
    }

    // MARK: - Prove

    /// Generate a WHIR proof for polynomial evaluations.
    ///
    /// The proof demonstrates that the committed polynomial is close to
    /// degree < n, using iterative folding with weighted hash equations.
    ///
    /// - Parameters:
    ///   - evaluations: polynomial evaluations on the domain (length must be power of 2)
    ///   - transcript: optional Fiat-Shamir transcript (created internally if nil)
    /// - Returns: WHIR proof with Merkle commitments, query openings, and weighted hashes
    public func prove(evaluations: [Fr], transcript: Transcript? = nil) throws -> WHIRProofData {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Fold until <= 16 elements remain
        let rounds = max(1, (logN - 4) / logReduction)

        let ts = transcript ?? Transcript(label: "whir-v2")

        // Phase 1: Build all layers (commit -> derive beta -> fold)
        var layers: [WHIRRoundCommitment] = []
        var betas: [Fr] = []
        var currentEvals = evaluations

        for round in 0..<rounds {
            let currentN = currentEvals.count
            if currentN <= reductionFactor { break }

            let commitment = try commit(evaluations: currentEvals)
            layers.append(commitment)

            // Transcript: absorb root, squeeze folding challenge
            ts.absorb(commitment.root)
            ts.absorbLabel("whir-r\(round)")
            let beta = ts.squeeze()
            betas.append(beta)

            // Fold polynomial using C CIOS arithmetic (Horner's method)
            currentEvals = WHIRProver.cpuFold(
                evals: currentEvals, challenge: beta,
                reductionFactor: reductionFactor)
        }

        let finalPoly = currentEvals
        let actualRounds = betas.count

        // Transcript: absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in finalPoly { ts.absorb(v) }

        // Phase 2: Query phase with weighted hash equations
        var layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]] = []
        var weightedHashClaims: [WHIRWeightedHashClaim] = []

        for round in 0..<actualRounds {
            let layer = layers[round]
            let layerN = layer.evaluations.count
            let foldedN = layerN / reductionFactor

            // Derive query positions (in folded domain)
            let effectiveQ = min(numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToUInt64(c) % UInt64(foldedN))
                while used.contains(idx) {
                    idx = (idx + 1) % UInt32(foldedN)
                }
                queryIndices.append(idx)
                used.insert(idx)
            }

            // Open reductionFactor positions per query in the current layer
            let layerTree = layer.tree
            let layerEvals = layer.evaluations
            var roundOpenings: [(index: UInt32, values: [Fr], merklePaths: [[Fr]])] = []
            roundOpenings.reserveCapacity(effectiveQ)

            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                var values = [Fr]()
                values.reserveCapacity(reductionFactor)
                var paths = [[Fr]]()
                paths.reserveCapacity(reductionFactor)
                for k in 0..<reductionFactor {
                    let origIdx = foldedIdx * reductionFactor + k
                    values.append(layerEvals[origIdx])
                    paths.append(extractMerklePath(tree: layerTree,
                                                    leafCount: layerN,
                                                    index: origIdx))
                }
                roundOpenings.append((index: queryIndices[qi], values: values, merklePaths: paths))
            }
            layerOpenings.append(roundOpenings)

            // WHIR weighted hash: derive random weights and compute claim
            // This is the key difference from FRI — a sumcheck instance that
            // binds the opened values via a random linear combination.
            ts.absorbLabel("whir-weights-r\(round)")
            var weights = [Fr]()
            weights.reserveCapacity(effectiveQ * reductionFactor)
            for _ in 0..<(effectiveQ * reductionFactor) {
                weights.append(ts.squeeze())
            }

            // Compute weighted sum: h = sum_i w_i * v_i
            var claimedSum = Fr.zero
            var wIdx = 0
            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                for k in 0..<reductionFactor {
                    claimedSum = frAdd(claimedSum, frMul(weights[wIdx], opening.values[k]))
                    wIdx += 1
                }
            }

            weightedHashClaims.append(WHIRWeightedHashClaim(
                weights: weights,
                claimedSum: claimedSum,
                queryIndices: queryIndices))
        }

        return WHIRProofData(
            roots: layers.map { $0.root },
            betas: betas,
            layerOpenings: layerOpenings,
            weightedHashClaims: weightedHashClaims,
            finalPoly: finalPoly,
            numRounds: actualRounds
        )
    }

    // MARK: - Merkle Helpers

    func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        var path = [Fr]()
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1
            if levelStart + siblingIdx < tree.count {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }

    // MARK: - CPU Fold

    /// Fold polynomial evaluations by reductionFactor using C CIOS Montgomery arithmetic.
    /// result[j] = sum_{k=0}^{r-1} beta^k * evals[j*r + k]  (computed via Horner's method)
    public static func cpuFold(evals: [Fr], challenge: Fr, reductionFactor: Int) -> [Fr] {
        let n = evals.count
        let newN = n / reductionFactor
        var result = [Fr](repeating: Fr.zero, count: newN)
        evals.withUnsafeBytes { evalsPtr in
            result.withUnsafeMutableBytes { resPtr in
                var betaLimbs = challenge.to64()
                betaLimbs.withUnsafeBufferPointer { betaPtr in
                    bn254_fr_whir_fold(
                        evalsPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        betaPtr.baseAddress!,
                        Int32(reductionFactor),
                        resPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: - Verify (convenience, delegates to WHIRVerifier)

    /// Succinct verify without original evaluations.
    public func verify(proof: WHIRProofData) -> Bool {
        let verifier = WHIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verify(proof: proof)
    }

    /// Succinct verify with known domain size.
    public func verify(proof: WHIRProofData, evaluations: [Fr]) -> Bool {
        let verifier = WHIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verify(proof: proof, domainSize: evaluations.count)
    }

    /// Full verify with original evaluations (checks every value).
    public func verifyFull(proof: WHIRProofData, evaluations: [Fr]) -> Bool {
        let verifier = WHIRVerifier(numQueries: numQueries, reductionFactor: reductionFactor)
        return verifier.verifyFull(proof: proof, evaluations: evaluations)
    }
}

/// Backward-compatible typealias for the old WHIREngine name.
public typealias WHIREngine = WHIRProver
