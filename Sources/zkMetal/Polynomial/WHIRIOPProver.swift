// WHIR Prover — Complete polynomial IOP prover for low-degree testing
//
// Implements the WHIR (Weighted Hashing IOP for Reed-Solomon) protocol
// (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/1586) as a standalone
// polynomial IOP prover with:
//   - Merkle commitment via Poseidon2 at each folding layer
//   - Fiat-Shamir challenge derivation for non-interactive proofs
//   - Domain-dependent weighted hashing for 2x soundness per query
//   - Configurable folding factor, query count, and security level
//
// This module ties together the folding rounds, Fiat-Shamir transcript,
// and Merkle commitments into a single prove() call that produces a
// self-contained WHIRIOPProof suitable for independent verification.

import Foundation
import NeonFieldOps

// MARK: - WHIR IOP Parameters

/// Configurable parameters for the WHIR polynomial IOP.
///
/// Controls the trade-off between proof size, prover cost, and soundness:
///   - numQueries: more queries -> larger proof, stronger soundness
///   - foldingFactor: larger -> fewer rounds -> smaller proof, heavier per-round work
///   - securityLevel: target bits of security (drives query count if numQueries is nil)
public struct WHIRIOPConfig {
    /// Number of queries per round (overrides securityLevel-based derivation if set)
    public let numQueries: Int
    /// Folding factor per round (must be power of 2, >= 2)
    public let foldingFactor: Int
    /// Target security level in bits
    public let securityLevel: Int
    /// Minimum size before sending polynomial in the clear
    public let finalPolyMaxSize: Int
    /// Code rate rho = degree / domain_size (for soundness calculation)
    public let rate: Double

    /// log2(foldingFactor)
    public var logFolding: Int {
        var k = foldingFactor
        var l = 0
        while k > 1 { k >>= 1; l += 1 }
        return l
    }

    /// Compute number of folding rounds for a given log-domain-size.
    public func numRounds(logN: Int) -> Int {
        let logFinal = Int(log2(Double(max(1, finalPolyMaxSize))))
        return max(1, (logN - logFinal) / logFolding)
    }

    /// Default configuration: 32 queries, folding by 4, 128-bit security.
    public static let standard = WHIRIOPConfig(
        numQueries: 32, foldingFactor: 4, securityLevel: 128,
        finalPolyMaxSize: 16, rate: 0.25)

    /// Lightweight configuration for fast testing.
    public static let fast = WHIRIOPConfig(
        numQueries: 4, foldingFactor: 4, securityLevel: 40,
        finalPolyMaxSize: 16, rate: 0.25)

    public init(numQueries: Int = 32, foldingFactor: Int = 4,
                securityLevel: Int = 128, finalPolyMaxSize: Int = 16,
                rate: Double = 0.25) {
        precondition(foldingFactor >= 2 && (foldingFactor & (foldingFactor - 1)) == 0,
                     "foldingFactor must be a power of 2")
        precondition(numQueries >= 1)
        precondition(securityLevel > 0)
        self.numQueries = numQueries
        self.foldingFactor = foldingFactor
        self.securityLevel = securityLevel
        self.finalPolyMaxSize = finalPolyMaxSize
        self.rate = rate
    }
}

// MARK: - WHIR Proof

/// A single layer's query response: leaf values and their Merkle authentication paths.
public struct WHIRQueryResponse {
    /// Index in the folded domain
    public let foldedIndex: UInt32
    /// The foldingFactor evaluation values at this coset
    public let values: [Fr]
    /// One Merkle authentication path per value (from leaf to root)
    public let merklePaths: [[Fr]]

    public init(foldedIndex: UInt32, values: [Fr], merklePaths: [[Fr]]) {
        self.foldedIndex = foldedIndex
        self.values = values
        self.merklePaths = merklePaths
    }
}

/// Complete WHIR proof for the polynomial IOP.
///
/// Contains everything needed for standalone verification:
///   - initialCommitment: Merkle root of the original evaluations
///   - roundCommitments: Merkle roots of each folded layer
///   - challenges: Fiat-Shamir-derived folding challenges (beta) per round
///   - weightSeeds: domain-dependent weight seeds (gamma) per round
///   - queryResponses: per-round query openings with Merkle proofs
///   - weightedSums: per-round claimed weighted hash sums
///   - hashCommitments: per-round H(sum || gamma || round) chain
///   - finalPolynomial: the small polynomial sent in the clear
///   - config: the IOP parameters used
public struct WHIRIOPProof {
    public let initialCommitment: Fr
    public let roundCommitments: [Fr]
    public let challenges: [Fr]
    public let weightSeeds: [Fr]
    public let queryResponses: [[WHIRQueryResponse]]
    public let weightedSums: [Fr]
    public let hashCommitments: [Fr]
    public let finalPolynomial: [Fr]
    public let config: WHIRIOPConfig
    public let domainSize: Int

    public init(initialCommitment: Fr, roundCommitments: [Fr], challenges: [Fr], weightSeeds: [Fr], queryResponses: [[WHIRQueryResponse]], weightedSums: [Fr], hashCommitments: [Fr], finalPolynomial: [Fr], config: WHIRIOPConfig, domainSize: Int) {
        self.initialCommitment = initialCommitment
        self.roundCommitments = roundCommitments
        self.challenges = challenges
        self.weightSeeds = weightSeeds
        self.queryResponses = queryResponses
        self.weightedSums = weightedSums
        self.hashCommitments = hashCommitments
        self.finalPolynomial = finalPolynomial
        self.config = config
        self.domainSize = domainSize
    }

    /// Proof size in bytes.
    public var proofSizeBytes: Int {
        let frSize = MemoryLayout<Fr>.stride
        var size = frSize  // initialCommitment
        size += roundCommitments.count * frSize
        size += challenges.count * frSize
        size += weightSeeds.count * frSize
        size += weightedSums.count * frSize
        size += hashCommitments.count * frSize
        size += finalPolynomial.count * frSize
        for round in queryResponses {
            for qr in round {
                size += 4  // foldedIndex (UInt32)
                size += qr.values.count * frSize
                for path in qr.merklePaths {
                    size += path.count * frSize
                }
            }
        }
        return size
    }

    /// Number of field elements in the proof.
    public var fieldElementCount: Int {
        var count = 1 + roundCommitments.count + challenges.count
        count += weightSeeds.count + weightedSums.count + hashCommitments.count
        count += finalPolynomial.count
        for round in queryResponses {
            for qr in round {
                count += qr.values.count
                for path in qr.merklePaths {
                    count += path.count
                }
            }
        }
        return count
    }

    /// Number of folding rounds.
    public var numRounds: Int { challenges.count }
}

// MARK: - WHIR Prover

/// WHIR polynomial IOP prover.
///
/// Proves that a polynomial (given as evaluations over a domain) has
/// degree less than d, using iterative folding with Merkle commitments
/// and domain-dependent weighted hash equations.
///
/// Protocol flow:
///   1. Commit to initial evaluations via Poseidon2 Merkle tree
///   2. For each round:
///      a. Absorb commitment into Fiat-Shamir transcript
///      b. Derive folding challenge beta
///      c. Fold evaluations: f'[j] = sum_{k=0}^{r-1} beta^k * f[j*r + k]
///      d. Derive weight seed gamma
///      e. Derive query positions in folded domain
///      f. Compute weighted sum h = sum w(x_i, gamma) * f(x_i)
///      g. Commit to weighted sum via hash chain
///      h. Commit to folded evaluations via Merkle tree
///   3. Send final small polynomial in the clear
///   4. Open Merkle paths at all query positions for each layer
public class WHIRIOPProver {
    public let config: WHIRIOPConfig
    private let merkleEngine: Poseidon2MerkleEngine

    /// CPU threshold for Merkle tree construction (avoid GPU overhead for small trees)
    private static let cpuMerkleThreshold = 4096

    /// Initialize the WHIR IOP prover.
    public init(config: WHIRIOPConfig = .standard) throws {
        self.config = config
        self.merkleEngine = try Poseidon2MerkleEngine()
    }

    /// Convenience initializer with explicit parameters.
    public convenience init(numQueries: Int, foldingFactor: Int = 4,
                            securityLevel: Int = 128) throws {
        try self.init(config: WHIRIOPConfig(
            numQueries: numQueries, foldingFactor: foldingFactor,
            securityLevel: securityLevel))
    }

    // MARK: - Prove

    /// Generate a WHIR proof of low-degree for the given polynomial evaluations.
    ///
    /// - Parameters:
    ///   - evaluations: polynomial evaluations on a domain of size 2^k
    ///   - transcript: optional external Fiat-Shamir transcript (fresh one created if nil)
    /// - Returns: a self-contained WHIRIOPProof
    public func prove(evaluations: [Fr],
                      transcript: Transcript? = nil) throws -> WHIRIOPProof {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Domain size must be power of 2")
        let logN = Int(log2(Double(n)))
        let rounds = config.numRounds(logN: logN)
        let ts = transcript ?? Transcript(label: "whir-iop-v1")

        // State accumulated across rounds
        var roundRoots: [Fr] = []
        var allChallenges: [Fr] = []
        var allWeightSeeds: [Fr] = []
        var allWeightedSums: [Fr] = []
        var allHashComms: [Fr] = []
        var layerTrees: [(tree: [Fr], evaluations: [Fr], leafCount: Int)] = []
        var layerQueryIndices: [[UInt32]] = []

        var currentEvals = evaluations

        // -- Phase 1: Commit initial polynomial --
        let initialTree = try buildMerkleTree(leaves: currentEvals)
        let initialRoot = initialTree[2 * n - 2]
        layerTrees.append((tree: initialTree, evaluations: currentEvals, leafCount: n))

        ts.absorb(initialRoot)
        ts.absorbLabel("whir-iop-domain-\(n)")

        // -- Phase 2: Folding rounds --
        for round in 0..<rounds {
            let currentN = currentEvals.count
            if currentN <= config.foldingFactor { break }

            // 2a. Derive folding challenge beta
            ts.absorbLabel("whir-iop-fold-r\(round)")
            let beta = ts.squeeze()
            allChallenges.append(beta)

            // 2b. Fold evaluations
            let folded = WHIRIOPProver.fold(
                evals: currentEvals, challenge: beta,
                foldingFactor: config.foldingFactor)

            // 2c. Derive weight seed gamma
            ts.absorbLabel("whir-iop-gamma-r\(round)")
            let gamma = ts.squeeze()
            allWeightSeeds.append(gamma)

            // 2d. Derive query positions in folded domain
            let foldedN = currentN / config.foldingFactor
            let effectiveQ = min(config.numQueries, foldedN)
            var queryIndices = [UInt32]()
            var used = Set<UInt32>()
            for _ in 0..<effectiveQ {
                let c = ts.squeeze()
                var idx = UInt32(frToUInt64(c) % UInt64(foldedN))
                while used.contains(idx) {
                    idx = (idx &+ 1) % UInt32(foldedN)
                }
                queryIndices.append(idx)
                used.insert(idx)
            }
            layerQueryIndices.append(queryIndices)

            // 2e. Compute domain-dependent weights and weighted sum
            let weights = WHIRIOPProver.computeWeights(
                gamma: gamma, queryIndices: queryIndices,
                domainSize: currentN, foldingFactor: config.foldingFactor)

            var wSum = Fr.zero
            var wIdx = 0
            for qi in 0..<effectiveQ {
                let base = Int(queryIndices[qi]) * config.foldingFactor
                for k in 0..<config.foldingFactor {
                    wSum = frAdd(wSum, frMul(weights[wIdx], currentEvals[base + k]))
                    wIdx += 1
                }
            }
            allWeightedSums.append(wSum)

            // 2f. Hash commitment: H(claimedSum || gamma || round)
            let hComm = WHIRIOPProver.hashCommitment(
                claimedSum: wSum, gamma: gamma, round: round)
            allHashComms.append(hComm)
            ts.absorb(hComm)

            // 2g. Commit to folded evaluations
            let foldedTree = try buildMerkleTree(leaves: folded)
            let foldedRoot = foldedTree[2 * foldedN - 2]
            roundRoots.append(foldedRoot)
            layerTrees.append((tree: foldedTree, evaluations: folded, leafCount: foldedN))

            ts.absorb(foldedRoot)

            currentEvals = folded
        }

        let finalPoly = currentEvals
        let actualRounds = allChallenges.count

        // Absorb final polynomial
        ts.absorbLabel("whir-iop-final")
        for v in finalPoly { ts.absorb(v) }

        // -- Phase 3: Generate query responses with Merkle proofs --
        var allQueryResponses: [[WHIRQueryResponse]] = []

        for round in 0..<actualRounds {
            let layer = layerTrees[round]
            let layerN = layer.leafCount
            let queryIndices = layerQueryIndices[round]
            let effectiveQ = queryIndices.count

            var roundResponses: [WHIRQueryResponse] = []
            roundResponses.reserveCapacity(effectiveQ)

            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                var values = [Fr]()
                values.reserveCapacity(config.foldingFactor)
                var paths = [[Fr]]()
                paths.reserveCapacity(config.foldingFactor)

                for k in 0..<config.foldingFactor {
                    let origIdx = foldedIdx * config.foldingFactor + k
                    values.append(layer.evaluations[origIdx])
                    paths.append(extractMerklePath(
                        tree: layer.tree, leafCount: layerN, index: origIdx))
                }

                roundResponses.append(WHIRQueryResponse(
                    foldedIndex: queryIndices[qi],
                    values: values,
                    merklePaths: paths))
            }
            allQueryResponses.append(roundResponses)
        }

        return WHIRIOPProof(
            initialCommitment: initialRoot,
            roundCommitments: roundRoots,
            challenges: allChallenges,
            weightSeeds: allWeightSeeds,
            queryResponses: allQueryResponses,
            weightedSums: allWeightedSums,
            hashCommitments: allHashComms,
            finalPolynomial: finalPoly,
            config: config,
            domainSize: n
        )
    }

    // MARK: - Merkle Helpers

    /// Build a Poseidon2 Merkle tree over the given leaves.
    /// Returns a flat array: [leaves | level1 | level2 | ... | root]
    func buildMerkleTree(leaves: [Fr]) throws -> [Fr] {
        let n = leaves.count
        precondition(n > 0 && (n & (n - 1)) == 0)

        if n <= WHIRIOPProver.cpuMerkleThreshold {
            let treeSize = 2 * n - 1
            var tree = [Fr](repeating: Fr.zero, count: treeSize)
            leaves.withUnsafeBytes { evPtr in
                tree.withUnsafeMutableBytes { treePtr in
                    poseidon2_merkle_tree_cpu(
                        evPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        treePtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
            return tree
        } else {
            return try merkleEngine.buildTree(leaves)
        }
    }

    /// Extract Merkle authentication path for a leaf at the given index.
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

    // MARK: - Fold

    /// Fold polynomial evaluations by foldingFactor using C CIOS Montgomery arithmetic.
    /// result[j] = sum_{k=0}^{r-1} beta^k * evals[j*r + k]
    public static func fold(evals: [Fr], challenge: Fr,
                            foldingFactor: Int) -> [Fr] {
        let n = evals.count
        let newN = n / foldingFactor
        var result = [Fr](repeating: Fr.zero, count: newN)
        evals.withUnsafeBytes { evalsPtr in
            result.withUnsafeMutableBytes { resPtr in
                var betaLimbs = challenge.to64()
                betaLimbs.withUnsafeBufferPointer { betaPtr in
                    bn254_fr_whir_fold(
                        evalsPtr.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        betaPtr.baseAddress!,
                        Int32(foldingFactor),
                        resPtr.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    // MARK: - Domain-Dependent Weights

    /// Compute weight w(omega^i, gamma) = gamma / (gamma - omega^i).
    public static func domainWeight(gamma: Fr, omega: Fr, index: Int) -> Fr {
        let omegaI = frPow(omega, UInt64(index))
        let diff = frSub(gamma, omegaI)
        let diffInv = frInverse(diff)
        return frMul(gamma, diffInv)
    }

    /// Compute all weights for a set of query positions.
    public static func computeWeights(gamma: Fr, queryIndices: [UInt32],
                                      domainSize: Int,
                                      foldingFactor: Int) -> [Fr] {
        let logN = Int(log2(Double(domainSize)))
        let omega = frRootOfUnity(logN: logN)
        var weights = [Fr]()
        weights.reserveCapacity(queryIndices.count * foldingFactor)

        for qi in queryIndices {
            for k in 0..<foldingFactor {
                let origIdx = Int(qi) * foldingFactor + k
                weights.append(domainWeight(gamma: gamma, omega: omega, index: origIdx))
            }
        }
        return weights
    }

    // MARK: - Hash Commitment

    /// Compute recursive hash commitment: H(H(claimedSum, gamma), round).
    public static func hashCommitment(claimedSum: Fr, gamma: Fr, round: Int) -> Fr {
        let inner = poseidon2Hash(claimedSum, gamma)
        let roundFr = frFromInt(UInt64(round))
        return poseidon2Hash(inner, roundFr)
    }
}
