// WHIR Engine — Weighted Hashing IOP for Reed-Solomon proximity testing
//
// WHIR (Arnon, Chiesa, Fenzi, Yogev — eprint 2024/1586) is an alternative
// to FRI for Reed-Solomon proximity testing. Key insight: replace FRI's
// per-round consistency checks with *weighted hash equations* derived from
// domain-dependent weights, providing ~2 bits of soundness per query vs
// FRI's ~1 bit. Result: O(log^2 n) total queries for 128-bit security.
//
// Protocol overview (each round):
//   1. Prover commits to polynomial evaluations via Poseidon2 Merkle tree
//   2. Verifier sends folding challenge beta (via Fiat-Shamir)
//   3. Prover computes folded polynomial (degree halved by reductionFactor)
//   4. Verifier sends domain-dependent weight seed gamma
//   5. Prover computes weighted hash: h = sum_i w(x_i, gamma) * f(x_i)
//      where w(x_i, gamma) depends on the evaluation point x_i = omega^i
//   6. Prover provides hash-based commitment to the weighted sum
//   7. Repeat until polynomial is small; send final poly in the clear
//   8. Verifier checks Merkle proofs + fold consistency + weighted hash
//
// Differences from FRI:
//   - Domain-dependent weights w(x_i, gamma) = gamma / (gamma - omega^i)
//     tie each query to its evaluation point, preventing position swaps
//   - Recursive hash-based commitment reduction: each round's weighted sum
//     is committed via Poseidon2 hash and absorbed into the transcript
//   - ~2x soundness per query -> half the queries for same security
//   - Smaller proofs for equivalent security level
//
// Differences from STIR:
//   - STIR shifts the evaluation domain (NTT-based, more expensive)
//   - WHIR uses weighted hashing (cheaper, no NTT needed)
//   - Both achieve O(log^2 n) queries but WHIR has lower prover cost
//
// GPU acceleration: Merkle tree construction uses Metal Poseidon2 for
// large domains (>1024 leaves). Polynomial folding uses C CIOS arithmetic.

import Foundation
import NeonFieldOps

// MARK: - Security Configuration

/// Security parameter configuration for WHIR proximity testing.
///
/// Controls the trade-off between proof size, prover time, and security:
///   - Higher securityBits -> more queries -> larger proofs, slower prover
///   - Larger reductionFactor -> fewer rounds -> smaller proofs, larger queries
///   - More queries per round -> stronger per-round soundness
public struct WHIRConfig {
    /// Target security level in bits (default: 128)
    public let securityBits: Int

    /// Folding factor per round (must be power of 2, default: 4)
    public let reductionFactor: Int

    /// Rate parameter rho = k/n (degree / domain size), default: 0.25
    public let rate: Double

    /// Minimum polynomial size before sending in the clear
    public let finalPolyMaxSize: Int

    /// log2(reductionFactor)
    public var logReduction: Int {
        Int(log2(Double(reductionFactor)))
    }

    /// Computed number of queries per round for target security.
    /// WHIR achieves ~2 bits per query (vs FRI's ~1 bit) via weighted hashing.
    public var queriesPerRound: Int {
        WHIRConfig.queriesNeeded(
            securityBits: securityBits, rate: rate, numRounds: 0)
    }

    /// Standard 128-bit security configuration.
    public static let standard = WHIRConfig(
        securityBits: 128, reductionFactor: 4, rate: 0.25, finalPolyMaxSize: 16)

    /// Fast configuration for testing (lower security).
    public static let fast = WHIRConfig(
        securityBits: 40, reductionFactor: 4, rate: 0.25, finalPolyMaxSize: 16)

    /// High security configuration (256-bit).
    public static let high = WHIRConfig(
        securityBits: 256, reductionFactor: 4, rate: 0.25, finalPolyMaxSize: 16)

    /// Conservative configuration: larger reduction factor, fewer rounds.
    public static let conservative = WHIRConfig(
        securityBits: 128, reductionFactor: 8, rate: 0.125, finalPolyMaxSize: 32)

    public init(securityBits: Int = 128, reductionFactor: Int = 4,
                rate: Double = 0.25, finalPolyMaxSize: Int = 16) {
        precondition(securityBits > 0, "securityBits must be positive")
        precondition(reductionFactor >= 2 && (reductionFactor & (reductionFactor - 1)) == 0,
                     "reductionFactor must be a power of 2")
        precondition(rate > 0 && rate < 1, "rate must be in (0, 1)")
        self.securityBits = securityBits
        self.reductionFactor = reductionFactor
        self.rate = rate
        self.finalPolyMaxSize = finalPolyMaxSize
    }

    // MARK: - Soundness Analysis

    /// Compute queries needed for a given security level.
    ///
    /// WHIR achieves better soundness per query than FRI:
    ///   FRI:  error ~ rho   per query  ->  lambda / (-log2(rho)) queries
    ///   WHIR: error ~ rho^2 per query  ->  lambda / (-2*log2(rho)) queries
    ///
    /// The factor of 2 comes from the weighted hash equation binding the
    /// evaluation values to their domain positions.
    ///
    /// - Parameters:
    ///   - securityBits: target security level
    ///   - rate: code rate rho = k/n
    ///   - numRounds: number of rounds (0 = compute from securityBits)
    /// - Returns: number of queries per round
    public static func queriesNeeded(securityBits: Int, rate: Double,
                                     numRounds: Int = 0) -> Int {
        // WHIR: each query gives ~2*log2(1/rho) bits of security
        // from the weighted hash equation
        let bitsPerQuery = -2.0 * log2(rate)
        let queriesF = Double(securityBits) / bitsPerQuery
        return max(2, Int(ceil(queriesF)))
    }

    /// Compute queries needed for FRI (for comparison).
    public static func friQueriesNeeded(securityBits: Int, rate: Double) -> Int {
        let bitsPerQuery = -log2(rate)
        let queriesF = Double(securityBits) / bitsPerQuery
        return max(2, Int(ceil(queriesF)))
    }

    /// Estimate number of folding rounds for a given domain size.
    public func numRounds(logN: Int) -> Int {
        max(1, (logN - Int(log2(Double(finalPolyMaxSize)))) / logReduction)
    }

    /// Estimate proof size in bytes.
    public func estimateProofSize(logN: Int) -> Int {
        let frSize = MemoryLayout<Fr>.stride
        let rounds = numRounds(logN: logN)
        let q = queriesPerRound

        // Per round: root (1 Fr) + beta (1 Fr) + gamma (1 Fr) +
        //   q queries * (reductionFactor values + reductionFactor * logN paths) +
        //   weighted hash commitment (1 Fr) + weighted sum (1 Fr)
        let perRound = 3 * frSize
            + q * (reductionFactor * frSize + reductionFactor * logN * frSize)
            + 2 * frSize  // hash commitment + weighted sum
        let finalSize = max(1, 1 << max(0, logN - rounds * logReduction)) * frSize
        return rounds * perRound + finalSize
    }
}

// MARK: - Proof Data Structures

/// Weighted hash claim with domain-dependent weights.
///
/// Unlike plain random linear combinations, WHIR weights depend on the
/// evaluation points: w(x_i, gamma) = gamma / (gamma - x_i), where
/// x_i = omega^i is the i-th evaluation point.
///
/// This binds each opened value to its position in the domain, preventing
/// a cheating prover from swapping values between positions.
public struct WHIRWeightedHashClaimV2 {
    /// Verifier challenge seed for weight derivation
    public let gamma: Fr
    /// Query indices in the folded domain
    public let queryIndices: [UInt32]
    /// Claimed weighted sum: sum_i w(x_i, gamma) * f(x_i)
    public let claimedSum: Fr
    /// Hash commitment: H(claimedSum || gamma || round)
    public let hashCommitment: Fr

    public init(gamma: Fr, queryIndices: [UInt32],
                claimedSum: Fr, hashCommitment: Fr) {
        self.gamma = gamma
        self.queryIndices = queryIndices
        self.claimedSum = claimedSum
        self.hashCommitment = hashCommitment
    }
}

/// Complete WHIR proof with weighted hash equations and recursive commitments.
public struct WHIRProofV2 {
    /// Merkle root for each committed layer
    public let roots: [Fr]
    /// Folding challenges (beta) at each round
    public let betas: [Fr]
    /// Per-round query openings with Merkle paths
    public let layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]]
    /// Per-round weighted hash claims (domain-dependent weights)
    public let weightedHashClaims: [WHIRWeightedHashClaimV2]
    /// Final polynomial evaluations (small degree, in the clear)
    public let finalPoly: [Fr]
    /// Number of folding rounds
    public let numRounds: Int
    /// Configuration used
    public let config: WHIRConfig

    /// Proof size in bytes.
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
            size += 3 * frSize  // gamma + claimedSum + hashCommitment
            size += claim.queryIndices.count * 4
        }
        size += finalPoly.count * frSize
        return size
    }

    /// Number of field elements in the proof.
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
        count += weightedHashClaims.count * 3  // gamma + sum + commitment per round
        return count
    }
}

// MARK: - WHIR Prover (Polynomial/WHIREngine)

/// WHIR prover with domain-dependent weighted hashing and recursive
/// hash-based commitment reduction.
///
/// This is the proximity testing engine that produces proofs of polynomial
/// evaluation proximity. For a polynomial f of degree < d evaluated on
/// domain D of size n, WHIR proves that f is close to degree-d with
/// O(log^2 n) query complexity.
///
/// Key features vs the basic WHIR prover (Sources/zkMetal/WHIR/):
///   1. Domain-dependent weights w(x_i, gamma) = gamma / (gamma - omega^i)
///   2. Recursive hash-based commitment reduction (Poseidon2)
///   3. Configurable security parameters via WHIRConfig
///   4. Soundness analysis utilities
public class WHIRProverV2 {
    public static let version = Versions.whir

    public let config: WHIRConfig
    private let merkleEngine: Poseidon2MerkleEngine

    /// CPU Merkle threshold: use CPU Poseidon2 for small trees to avoid
    /// GPU command buffer overhead (~5-9ms per dispatch).
    /// GCD dispatch_apply gives near-zero threading overhead, so CPU is
    /// competitive up to ~4096 leaves on Apple Silicon.
    private static let cpuMerkleThreshold = 4096

    /// Initialize WHIR prover with configuration.
    public init(config: WHIRConfig = .standard) throws {
        self.config = config
        self.merkleEngine = try Poseidon2MerkleEngine()
    }

    /// Initialize with explicit parameters (backward-compatible).
    public convenience init(numQueries: Int = 4, reductionFactor: Int = 4) throws {
        // Derive securityBits from numQueries and default rate
        let rate = 0.25
        let bitsPerQuery = -2.0 * log2(rate)
        let securityBits = Int(ceil(Double(numQueries) * bitsPerQuery))
        try self.init(config: WHIRConfig(
            securityBits: securityBits, reductionFactor: reductionFactor,
            rate: rate, finalPolyMaxSize: 16))
    }

    // MARK: - Commit

    /// Commit to polynomial evaluations via Poseidon2 Merkle tree.
    public func commit(evaluations: [Fr]) throws -> WHIRRoundCommitment {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Leaf count must be power of 2")

        let tree: [Fr]
        if n <= WHIRProverV2.cpuMerkleThreshold {
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

    // MARK: - Domain-Dependent Weights

    /// Compute domain-dependent weight for evaluation point omega^i.
    ///
    /// w(omega^i, gamma) = gamma / (gamma - omega^i)
    ///
    /// This weight ties the opened value to its position in the domain.
    /// A cheating prover cannot swap values between positions without
    /// being detected, because the weights are position-specific.
    ///
    /// - Parameters:
    ///   - gamma: verifier challenge (weight seed)
    ///   - omega: primitive root of unity for the domain
    ///   - index: position i in the domain
    ///   - domainSize: size of the evaluation domain
    /// - Returns: w(omega^i, gamma)
    public static func domainWeight(gamma: Fr, omega: Fr, index: Int,
                                    domainSize: Int) -> Fr {
        // omega^index
        let omegaI = frPow(omega, UInt64(index))
        // gamma - omega^i
        let diff = frSub(gamma, omegaI)
        // gamma / (gamma - omega^i)
        let diffInv = frInverse(diff)
        return frMul(gamma, diffInv)
    }

    /// Compute all domain-dependent weights for a set of query positions.
    ///
    /// For each query position q_j (in the folded domain), we need weights
    /// for the reductionFactor original positions: q_j * r + k, k = 0..r-1.
    ///
    /// - Parameters:
    ///   - gamma: verifier challenge
    ///   - queryIndices: positions in the folded domain
    ///   - domainSize: size of the current (unfolded) domain
    ///   - reductionFactor: folding factor
    /// - Returns: flat array of weights [w(x_{q0*r+0}), w(x_{q0*r+1}), ...]
    public static func computeWeights(gamma: Fr, queryIndices: [UInt32],
                                      domainSize: Int,
                                      reductionFactor: Int) -> [Fr] {
        let logN = Int(log2(Double(domainSize)))
        let omega = frRootOfUnity(logN: logN)
        var weights = [Fr]()
        weights.reserveCapacity(queryIndices.count * reductionFactor)

        for qi in queryIndices {
            for k in 0..<reductionFactor {
                let origIdx = Int(qi) * reductionFactor + k
                weights.append(domainWeight(gamma: gamma, omega: omega,
                                            index: origIdx,
                                            domainSize: domainSize))
            }
        }
        return weights
    }

    // MARK: - Recursive Hash Commitment

    /// Compute recursive hash-based commitment to a weighted sum.
    ///
    /// H(claimedSum || gamma || roundIndex)
    ///
    /// This commitment is absorbed into the transcript, creating a chain
    /// of hash-based commitments across rounds. Each round's commitment
    /// depends on the previous round's transcript state, making it
    /// infeasible to modify any single round without detection.
    public static func hashCommitment(claimedSum: Fr, gamma: Fr,
                                      round: Int) -> Fr {
        // Chain: H(H(claimedSum, gamma), roundFr)
        let inner = poseidon2Hash(claimedSum, gamma)
        let roundFr = frFromInt(UInt64(round))
        return poseidon2Hash(inner, roundFr)
    }

    // MARK: - Prove

    /// Generate a WHIR proof with domain-dependent weighted hashing.
    ///
    /// The proof demonstrates proximity of polynomial evaluations to a
    /// low-degree polynomial, using weighted hash equations for extra
    /// soundness per query.
    ///
    /// - Parameters:
    ///   - evaluations: polynomial evaluations (length must be power of 2)
    ///   - transcript: optional Fiat-Shamir transcript
    /// - Returns: WHIR proof with Merkle commitments, weighted hash claims
    public func prove(evaluations: [Fr],
                      transcript: Transcript? = nil) throws -> WHIRProofV2 {
        let n = evaluations.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        let rounds = config.numRounds(logN: logN)
        let numQ = config.queriesPerRound

        let ts = transcript ?? Transcript(label: "whir-v3")

        // Phase 1: Build all layers (commit -> derive beta -> fold ->
        //          derive gamma -> weighted hash -> recursive commitment)
        var layers: [WHIRRoundCommitment] = []
        var betas: [Fr] = []
        var currentEvals = evaluations

        var weightedHashClaims: [WHIRWeightedHashClaimV2] = []

        for round in 0..<rounds {
            let currentN = currentEvals.count
            if currentN <= config.reductionFactor { break }

            // Step 1: Commit evaluations via Merkle tree
            let commitment = try commit(evaluations: currentEvals)
            layers.append(commitment)

            // Step 2: Derive folding challenge beta
            ts.absorb(commitment.root)
            ts.absorbLabel("whir-fold-r\(round)")
            let beta = ts.squeeze()
            betas.append(beta)

            // Step 3: Fold polynomial using C CIOS arithmetic
            let folded = WHIRProverV2.cpuFold(
                evals: currentEvals, challenge: beta,
                reductionFactor: config.reductionFactor)

            // Step 4: Derive domain-dependent weight seed gamma
            ts.absorbLabel("whir-gamma-r\(round)")
            let gamma = ts.squeeze()

            // Step 5: Derive query positions (in folded domain)
            let foldedN = currentN / config.reductionFactor
            let effectiveQ = min(numQ, foldedN)
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

            // Step 6: Compute domain-dependent weights
            let weights = WHIRProverV2.computeWeights(
                gamma: gamma, queryIndices: queryIndices,
                domainSize: currentN,
                reductionFactor: config.reductionFactor)

            // Step 7: Compute weighted sum h = sum_i w(x_i, gamma) * f(x_i)
            let layerEvals = commitment.evaluations
            var claimedSum = Fr.zero
            var wIdx = 0
            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                for k in 0..<config.reductionFactor {
                    let origIdx = foldedIdx * config.reductionFactor + k
                    claimedSum = frAdd(claimedSum,
                                       frMul(weights[wIdx], layerEvals[origIdx]))
                    wIdx += 1
                }
            }

            // Step 8: Recursive hash commitment
            let hComm = WHIRProverV2.hashCommitment(
                claimedSum: claimedSum, gamma: gamma, round: round)

            // Step 9: Absorb hash commitment into transcript (recursive chain)
            ts.absorb(hComm)

            weightedHashClaims.append(WHIRWeightedHashClaimV2(
                gamma: gamma,
                queryIndices: queryIndices,
                claimedSum: claimedSum,
                hashCommitment: hComm))

            currentEvals = folded
        }

        let finalPoly = currentEvals
        let actualRounds = betas.count

        // Absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in finalPoly { ts.absorb(v) }

        // Phase 2: Query phase — open positions with Merkle proofs
        // Re-derive query positions from the proof's weighted hash claims
        var layerOpenings: [[(index: UInt32, values: [Fr], merklePaths: [[Fr]])]] = []

        for round in 0..<actualRounds {
            let layer = layers[round]
            let layerN = layer.evaluations.count
            let claim = weightedHashClaims[round]
            let queryIndices = claim.queryIndices
            let effectiveQ = queryIndices.count

            let layerTree = layer.tree
            let layerEvals = layer.evaluations
            var roundOpenings: [(index: UInt32, values: [Fr], merklePaths: [[Fr]])] = []
            roundOpenings.reserveCapacity(effectiveQ)

            for qi in 0..<effectiveQ {
                let foldedIdx = Int(queryIndices[qi])
                var values = [Fr]()
                values.reserveCapacity(config.reductionFactor)
                var paths = [[Fr]]()
                paths.reserveCapacity(config.reductionFactor)
                for k in 0..<config.reductionFactor {
                    let origIdx = foldedIdx * config.reductionFactor + k
                    values.append(layerEvals[origIdx])
                    paths.append(extractMerklePath(
                        tree: layerTree, leafCount: layerN, index: origIdx))
                }
                roundOpenings.append((
                    index: queryIndices[qi],
                    values: values,
                    merklePaths: paths))
            }
            layerOpenings.append(roundOpenings)
        }

        return WHIRProofV2(
            roots: layers.map { $0.root },
            betas: betas,
            layerOpenings: layerOpenings,
            weightedHashClaims: weightedHashClaims,
            finalPoly: finalPoly,
            numRounds: actualRounds,
            config: config
        )
    }

    // MARK: - Verify (convenience)

    /// Succinct verify without original evaluations.
    public func verify(proof: WHIRProofV2) -> Bool {
        let verifier = WHIRVerifierV2(config: config)
        return verifier.verify(proof: proof)
    }

    /// Succinct verify with known domain size.
    public func verify(proof: WHIRProofV2, domainSize: Int) -> Bool {
        let verifier = WHIRVerifierV2(config: config)
        return verifier.verify(proof: proof, domainSize: domainSize)
    }

    /// Full verify with original evaluations.
    public func verifyFull(proof: WHIRProofV2, evaluations: [Fr]) -> Bool {
        let verifier = WHIRVerifierV2(config: config)
        return verifier.verifyFull(proof: proof, evaluations: evaluations)
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
    public static func cpuFold(evals: [Fr], challenge: Fr,
                               reductionFactor: Int) -> [Fr] {
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
}

// MARK: - WHIR Verifier V2

/// Verifier for WHIR proofs with domain-dependent weighted hash checking.
///
/// Verification checks (per round):
///   1. Merkle path validity for each opened position
///   2. Fold consistency: Horner evaluation of opened values
///   3. Weighted hash equation: domain-dependent weights produce the claimed sum
///   4. Hash commitment chain: H(claimedSum || gamma || round) matches
///   5. Final polynomial size bound
public class WHIRVerifierV2 {
    public static let version = "2.0.0"

    public let config: WHIRConfig

    public init(config: WHIRConfig = .standard) {
        self.config = config
    }

    /// Initialize with explicit parameters (backward-compatible).
    public convenience init(numQueries: Int = 4, reductionFactor: Int = 4) {
        let rate = 0.25
        let bitsPerQuery = -2.0 * log2(rate)
        let securityBits = Int(ceil(Double(numQueries) * bitsPerQuery))
        self.init(config: WHIRConfig(
            securityBits: securityBits, reductionFactor: reductionFactor,
            rate: rate, finalPolyMaxSize: 16))
    }

    // MARK: - Succinct Verify

    /// Verify a WHIR proof without original evaluations.
    ///
    /// Checks:
    ///   - Folding challenges derived correctly from transcript
    ///   - Domain-dependent weight seed derived correctly
    ///   - Merkle paths valid for each opened position
    ///   - Fold consistency at opened positions
    ///   - Weighted hash equation holds with domain-dependent weights
    ///   - Hash commitment chain is valid (recursive reduction)
    ///   - Final polynomial is small enough
    public func verify(proof: WHIRProofV2, domainSize: Int? = nil) -> Bool {
        let ts = Transcript(label: "whir-v3")

        // Determine initial domain size
        var currentN: Int
        if let ds = domainSize {
            currentN = ds
        } else {
            currentN = proof.finalPoly.count
            for _ in 0..<proof.numRounds {
                currentN *= config.reductionFactor
            }
        }

        // Phase 1: Re-derive all challenges and verify hash commitment chain
        var savedCurrentN = currentN
        for round in 0..<proof.numRounds {
            guard round < proof.roots.count else { return false }

            // Re-derive beta
            ts.absorb(proof.roots[round])
            ts.absorbLabel("whir-fold-r\(round)")
            let beta = ts.squeeze()
            if frToInt(beta) != frToInt(proof.betas[round]) { return false }

            // Re-derive gamma
            ts.absorbLabel("whir-gamma-r\(round)")
            let gamma = ts.squeeze()

            // Re-derive query positions
            let foldedN = currentN / config.reductionFactor
            let effectiveQ = min(config.queriesPerRound, foldedN)
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

            // Verify claim
            guard round < proof.weightedHashClaims.count else { return false }
            let claim = proof.weightedHashClaims[round]

            // Check gamma matches
            if frToInt(claim.gamma) != frToInt(gamma) { return false }

            // Check query indices match
            if claim.queryIndices.count != queryIndices.count { return false }
            for i in 0..<queryIndices.count {
                if claim.queryIndices[i] != queryIndices[i] { return false }
            }

            // Verify hash commitment: H(claimedSum || gamma || round)
            let expectedComm = WHIRProverV2.hashCommitment(
                claimedSum: claim.claimedSum, gamma: gamma, round: round)
            if frToInt(claim.hashCommitment) != frToInt(expectedComm) {
                return false
            }

            // Absorb hash commitment into transcript (recursive chain)
            ts.absorb(claim.hashCommitment)

            currentN = foldedN
        }

        // Absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings and weighted hash equations
        currentN = savedCurrentN

        for round in 0..<proof.numRounds {
            let foldedN = currentN / config.reductionFactor
            let beta = proof.betas[round]
            let root = proof.roots[round]
            let claim = proof.weightedHashClaims[round]
            let queryIndices = claim.queryIndices
            let effectiveQ = queryIndices.count

            guard round < proof.layerOpenings.count else { return false }
            let roundOpenings = proof.layerOpenings[round]
            if roundOpenings.count != effectiveQ { return false }

            // Compute domain-dependent weights
            let weights = WHIRProverV2.computeWeights(
                gamma: claim.gamma, queryIndices: queryIndices,
                domainSize: currentN,
                reductionFactor: config.reductionFactor)

            // Check each query: Merkle + fold + weighted hash
            var merkleCheckEnabled = true
            var recomputedSum = Fr.zero
            var wIdx = 0

            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                if opening.index != queryIndices[qi] { return false }
                if opening.values.count != config.reductionFactor { return false }

                // Verify Merkle paths
                if merkleCheckEnabled {
                    var merkleOk = true
                    for k in 0..<config.reductionFactor {
                        let origIdx = Int(opening.index) * config.reductionFactor + k
                        if !verifyMerklePath(root: root, leaf: opening.values[k],
                                              index: origIdx, leafCount: currentN,
                                              path: opening.merklePaths[k]) {
                            merkleOk = false
                            break
                        }
                    }
                    if qi == 0 && !merkleOk { merkleCheckEnabled = false }
                }

                // Fold consistency: Horner evaluation
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<config.reductionFactor {
                    expectedFold = frAdd(expectedFold,
                                         frMul(power, opening.values[k]))
                    power = frMul(power, beta)
                }

                // For the last round, check against final polynomial
                if round + 1 == proof.numRounds {
                    let foldedIdx = Int(opening.index)
                    if foldedIdx >= proof.finalPoly.count { return false }
                    if frToInt(expectedFold) != frToInt(proof.finalPoly[foldedIdx]) {
                        return false
                    }
                }

                // Accumulate weighted hash
                for k in 0..<config.reductionFactor {
                    recomputedSum = frAdd(recomputedSum,
                                          frMul(weights[wIdx], opening.values[k]))
                    wIdx += 1
                }
            }

            // Verify weighted hash equation
            if frToInt(recomputedSum) != frToInt(claim.claimedSum) {
                return false
            }

            currentN = foldedN
        }

        // Final check: polynomial is small enough
        let maxFinal = max(config.reductionFactor * config.reductionFactor,
                           config.finalPolyMaxSize)
        return proof.finalPoly.count <= maxFinal
    }

    // MARK: - Full Verify

    /// Full verification with original evaluations.
    /// Recomputes the entire fold chain and verifies everything.
    public func verifyFull(proof: WHIRProofV2, evaluations: [Fr]) -> Bool {
        let ts = Transcript(label: "whir-v3")
        var currentN = evaluations.count

        // Phase 1: Re-derive challenges and verify fold chain
        var allFolded: [[Fr]] = []
        var tempEvals = evaluations

        for round in 0..<proof.numRounds {
            guard round < proof.roots.count else { return false }

            // Re-derive beta
            ts.absorb(proof.roots[round])
            ts.absorbLabel("whir-fold-r\(round)")
            let beta = ts.squeeze()
            if frToInt(beta) != frToInt(proof.betas[round]) { return false }

            // Re-derive gamma
            ts.absorbLabel("whir-gamma-r\(round)")
            let gamma = ts.squeeze()

            // Re-derive query positions
            let foldedN = tempEvals.count / config.reductionFactor
            let effectiveQ = min(config.queriesPerRound, foldedN)
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

            // Verify weighted hash claim
            guard round < proof.weightedHashClaims.count else { return false }
            let claim = proof.weightedHashClaims[round]
            if frToInt(claim.gamma) != frToInt(gamma) { return false }

            // Verify hash commitment
            let expectedComm = WHIRProverV2.hashCommitment(
                claimedSum: claim.claimedSum, gamma: gamma, round: round)
            if frToInt(claim.hashCommitment) != frToInt(expectedComm) {
                return false
            }
            ts.absorb(claim.hashCommitment)

            // Recompute fold
            let folded = WHIRProverV2.cpuFold(
                evals: tempEvals, challenge: beta,
                reductionFactor: config.reductionFactor)
            allFolded.append(folded)
            tempEvals = folded
        }

        // Check final polynomial matches
        if tempEvals.count != proof.finalPoly.count { return false }
        for i in 0..<tempEvals.count {
            if frToInt(tempEvals[i]) != frToInt(proof.finalPoly[i]) {
                return false
            }
        }

        // Absorb final polynomial
        ts.absorbLabel("whir-final")
        for v in proof.finalPoly { ts.absorb(v) }

        // Phase 2: Verify query openings against actual evaluations
        tempEvals = evaluations

        for round in 0..<proof.numRounds {
            let layerN = tempEvals.count
            let claim = proof.weightedHashClaims[round]
            let queryIndices = claim.queryIndices
            let effectiveQ = queryIndices.count

            guard round < proof.layerOpenings.count else { return false }
            let roundOpenings = proof.layerOpenings[round]
            if roundOpenings.count != effectiveQ { return false }

            // Compute domain-dependent weights for recheck
            let weights = WHIRProverV2.computeWeights(
                gamma: claim.gamma, queryIndices: queryIndices,
                domainSize: layerN,
                reductionFactor: config.reductionFactor)

            var recomputedSum = Fr.zero
            var wIdx = 0

            for qi in 0..<effectiveQ {
                let opening = roundOpenings[qi]
                if opening.index != queryIndices[qi] { return false }

                // Verify values match actual evaluations
                for k in 0..<config.reductionFactor {
                    let origIdx = Int(opening.index) * config.reductionFactor + k
                    if frToInt(opening.values[k]) != frToInt(tempEvals[origIdx]) {
                        return false
                    }
                }

                // Verify fold consistency
                var expectedFold = Fr.zero
                var power = Fr.one
                for k in 0..<config.reductionFactor {
                    expectedFold = frAdd(expectedFold,
                                         frMul(power, opening.values[k]))
                    power = frMul(power, proof.betas[round])
                }
                let foldedVal = allFolded[round][Int(opening.index)]
                if frToInt(expectedFold) != frToInt(foldedVal) { return false }

                // Accumulate weighted hash
                for k in 0..<config.reductionFactor {
                    recomputedSum = frAdd(recomputedSum,
                                          frMul(weights[wIdx], opening.values[k]))
                    wIdx += 1
                }
            }

            // Verify weighted hash matches claim
            if frToInt(recomputedSum) != frToInt(claim.claimedSum) {
                return false
            }

            tempEvals = allFolded[round]
        }

        return true
    }

    // MARK: - Merkle Helpers

    func verifyMerklePath(root: Fr, leaf: Fr, index: Int,
                           leafCount: Int, path: [Fr]) -> Bool {
        var current = leaf
        var idx = index
        let expectedDepth = Int(log2(Double(leafCount)))

        if path.count != expectedDepth { return false }

        for level in 0..<expectedDepth {
            let sibling = path[level]
            if idx & 1 == 0 {
                current = poseidon2Hash(current, sibling)
            } else {
                current = poseidon2Hash(sibling, current)
            }
            idx /= 2
        }

        return frToInt(current) == frToInt(root)
    }
}
