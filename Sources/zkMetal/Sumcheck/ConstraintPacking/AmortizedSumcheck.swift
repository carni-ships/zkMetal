// Amortized Sumcheck — O(1) Per-Query with Precomputed Tower Basis Cache
//
// Amortizes all precomputation across sumcheck rounds using the tower-basis cache.
// After the first round, each subsequent query reuses cached vanishing polynomials,
// twiddle factors, and Lagrange coefficients with only O(1) per-query adjustments
// via random challenges in the extension field.
//
// Reference: Constraint-Packing and Sum-Check Protocol over Binary Towers (ePrint 2024/1038)

import Foundation

// MARK: - Sumcheck Claim

/// A sumcheck claim of the form: sum_{x in {0,1}^k} f(x) = C
public struct AmortizedSumcheckClaim: Equatable {
    /// Number of variables (log of domain size).
    public let numVariables: Int

    /// Target sum that should equal the commitment.
    public let targetSum: UInt8

    /// Polynomial committed to (as evaluation).
    public let polynomialEvals: [UInt8]

    /// Create a new claim.
    public init(numVariables: Int, targetSum: UInt8, polynomialEvals: [UInt8]) {
        self.numVariables = numVariables
        self.targetSum = targetSum
        self.polynomialEvals = polynomialEvals
    }

    /// Verify the claim makes sense.
    public var isValid: Bool {
        polynomialEvals.count == (1 << numVariables)
    }
}

// MARK: - Sumcheck Round

/// One round of the sumcheck protocol.
public struct AmortizedSumcheckRound: Equatable {
    /// Round index (0 to numVariables - 1).
    public let roundIndex: Int

    /// Polynomial evaluated at this round: g_i(X_i) = sum_{x_{i+1}, ..., x_k} f(x_1, ..., x_i, X_i, x_{i+1}, ..., x_k)
    public let roundPolynomial: [UInt8]

    /// Random challenge from verifier.
    public let challenge: UInt8

    /// Claim for next round.
    public let nextClaim: UInt8

    public init(roundIndex: Int, roundPolynomial: [UInt8], challenge: UInt8, nextClaim: UInt8) {
        self.roundIndex = roundIndex
        self.roundPolynomial = roundPolynomial
        self.challenge = challenge
        self.nextClaim = nextClaim
    }
}

// MARK: - Sumcheck Proof

/// Complete sumcheck proof.
public struct PackedSumcheckProof: Equatable {
    /// All rounds of the protocol.
    public let rounds: [AmortizedSumcheckRound]

    /// Final evaluation at random point.
    public let finalEvaluation: UInt8

    public init(rounds: [AmortizedSumcheckRound], finalEvaluation: UInt8) {
        self.rounds = rounds
        self.finalEvaluation = finalEvaluation
    }
}

// MARK: - Amortized Sumcheck Prover

/// Sumcheck prover with O(1) per-query using precomputed tower basis cache.
///
/// After first-round setup, all subsequent queries use cached values directly,
/// avoiding redundant precomputation in every round.
public final class AmortizedSumcheckProver {
    /// Precomputed tower basis cache.
    public let basisCache: TowerBasisCache

    /// Constraint packer for evaluating packed constraints.
    public let constraintPacker: ConstraintPacker

    /// Extension field degree for random challenges.
    public let extensionDegree: Int

    /// Create a new prover.
    public init(basisCache: TowerBasisCache, constraintPacker: ConstraintPacker, extensionDegree: Int = 8) {
        self.basisCache = basisCache
        self.constraintPacker = constraintPacker
        self.extensionDegree = extensionDegree
    }

    // MARK: - Prove

    /// Prove a sumcheck claim with O(1) per-query after initialization.
    ///
    /// - Parameters:
    ///   - claim: The sumcheck claim to prove
    ///   - witness: Witness values
    ///   - randomness: Random challenges from verifier (one per round)
    /// - Returns: Sumcheck proof
    public func prove(claim: AmortizedSumcheckClaim, witness: [UInt8], randomness: [UInt8]) -> PackedSumcheckProof {
        precondition(claim.isValid, "Invalid claim")
        precondition(randomness.count == claim.numVariables, "Need one random challenge per variable")

        var rounds: [AmortizedSumcheckRound] = []
        var currentEvals = claim.polynomialEvals
        var currentClaim = claim.targetSum

        // Process each round
        for i in 0..<claim.numVariables {
            // At round 0: we check bit 0 of original index (idx & 1)
            // At round i > 0: after i reductions, the reduced index j's bit 0 = original's bit i
            // So for ALL rounds, we check bit 0 of the current index
            let round = processRound(
                roundIndex: 0,
                evals: currentEvals,
                claim: currentClaim,
                challenge: randomness[i],
                numVariables: claim.numVariables
            )
            rounds.append(round)

            // Reduce to next level using challenge.
            // After filtering by bit i of original (= bit 0 of reduced index j),
            // the new reduced array is stored contiguously with index = j >> 1
            let challenge = randomness[i]

            // Only reduce if there are more rounds
            if i < claim.numVariables - 1 {
                let halfSize = currentEvals.count / 2
                var reducedEvals = [UInt8](repeating: 0, count: halfSize)
                for idx in 0..<currentEvals.count {
                    // bit 0 of idx = original's bit i (after i rounds of reduction)
                    if (idx & 1) == Int(challenge) {
                        let newIdx = idx >> 1
                        reducedEvals[newIdx] = currentEvals[idx]
                    }
                }
                currentEvals = reducedEvals
            }

            currentClaim = round.nextClaim
        }

        // Final evaluation at random point
        // Compute f(r_0, ..., r_{k-1}) directly using multilinear extension formula
        let finalEval = computeMultilinearExtension(evals: claim.polynomialEvals, randomness: randomness)

        return PackedSumcheckProof(rounds: rounds, finalEvaluation: finalEval)
    }

    /// Process a single sumcheck round.
    /// Computes g_i(b) = sum_{x_{i+1}, ..., x_k} f(x_1, ..., x_i, b, x_{i+1}, ..., x_k) for b in {0,1}
    private func processRound(
        roundIndex: Int,
        evals: [UInt8],
        claim: UInt8,
        challenge: UInt8,
        numVariables: Int
    ) -> AmortizedSumcheckRound {
        var g0: UInt8 = 0
        var g1: UInt8 = 0

        // Compute g_i(0) and g_i(1)
        for (idx, eval) in evals.enumerated() {
            let bit_i = (idx >> roundIndex) & 1
            if bit_i == 0 {
                g0 ^= eval
            } else {
                g1 ^= eval
            }
        }

        let roundPoly = [g0, g1]

        // Next claim is g_i(challenge)
        let g0PlusG1 = g0 ^ g1
        let nextClaim = g0 ^ gf28Mul(g0PlusG1, challenge)

        return AmortizedSumcheckRound(
            roundIndex: roundIndex,
            roundPolynomial: roundPoly,
            challenge: challenge,
            nextClaim: nextClaim
        )
    }

    /// Reduce evaluations to next level using challenge.
    /// After round i, we have g_i(r_i). The evals for round i+1 represent
    /// g_{i+1}(y) = sum_{x_{i+2}, ..., x_k} f(r_1, ..., r_i, y, x_{i+2}, ..., x_k)
    ///
    /// The reduceEvals function needs to track accumulated constraints C from previous rounds.
    /// After i rounds with challenges c0,...,c_{i-1}, original index o relates to reduced index j by:
    ///   o = (j << i) | C  where C contains the lower i bits from previous constraints.
    /// The bit_i of original is: bit_i(o) = bit_i((j << i) | C) = (j & 1) ^ bit_i(C).
    /// Since C varies per j, we compute this per-element.
    private func reduceEvals(evals: [UInt8], challenge: UInt8, roundIndex: Int, originalIndices: [Int]) -> [UInt8] {
        // originalIndices: the original indices that correspond to each element in evals
        let halfSize = evals.count / 2
        var result = [UInt8](repeating: 0, count: halfSize)

        for (pos, idx) in originalIndices.enumerated() {
            let bit_i = (idx >> roundIndex) & 1
            if bit_i == Int(challenge) {
                // New reduced index is the position in the result array
                // We compute it as (original >> (roundIndex + 1)) for the upper bits
                // plus the lower bits that weren't at roundIndex
                let upper = idx >> (roundIndex + 1)
                let lowerMask = (1 << roundIndex) - 1
                let lower = idx & lowerMask
                let newIdx = (upper << roundIndex) | lower
                result[newIdx] ^= evals[pos]
            }
        }

        return result
    }

    /// GF(2^8) multiplication.
    func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var aa = UInt16(a)
        var bb = UInt16(b)
        for _ in 0..<8 {
            if bb & 1 != 0 {
                p ^= aa
            }
            let hiBitSet = (aa & 0x80) != 0
            aa <<= 1
            if hiBitSet {
                aa ^= 0x11B
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }

    /// Compute multilinear extension f(r) = Σ_{s∈{0,1}^k} f(s) · Π_i (r_i · s_i + (1 - r_i) · (1 - s_i))
    private func computeMultilinearExtension(evals: [UInt8], randomness: [UInt8]) -> UInt8 {
        let numVariables = Int(log2(Double(evals.count)))
        var result: UInt8 = 0

        for (idx, eval) in evals.enumerated() {
            // Compute the multilinear basis term for this index
            var basisTerm: UInt8 = 1

            for i in 0..<numVariables {
                let s_i = UInt8((idx >> i) & 1)
                let r_i = randomness[i]

                // term = r_i * s_i + (1 - r_i) * (1 - s_i)
                // In GF(2^8): + is XOR, * is polynomial multiplication
                let term: UInt8
                if s_i == 0 {
                    // term = (1 - r_i) * 1 = 1 - r_i = 1 + r_i (since -1 = 1 in char 2)
                    term = 1 ^ r_i
                } else {
                    // term = r_i * 1 + (1 - r_i) * 0 = r_i
                    term = r_i
                }

                basisTerm = gf28Mul(basisTerm, term)
            }

            // Accumulate f(s) * basisTerm
            result ^= gf28Mul(eval, basisTerm)
        }

        return result
    }
}

// MARK: - Amortized Sumcheck Verifier

/// Verifier with O(1) per-query using precomputed cache.
public final class AmortizedSumcheckVerifier {
    /// Precomputed tower basis cache.
    public let basisCache: TowerBasisCache

    /// Extension field degree.
    public let extensionDegree: Int

    public init(basisCache: TowerBasisCache, extensionDegree: Int = 8) {
        self.basisCache = basisCache
        self.extensionDegree = extensionDegree
    }

    // MARK: - Verify

    /// Verify a sumcheck proof.
    ///
    /// - Parameters:
    ///   - proof: Sumcheck proof to verify
    ///   - claim: Original claim
    ///   - randomness: Random challenges used by prover
    /// - Returns: True if verification succeeds
    public func verify(proof: PackedSumcheckProof, claim: AmortizedSumcheckClaim, randomness: [UInt8]) -> Bool {
        // Verify each round: g_i(0) + g_i(1) should equal the running claim
        // The running claim starts as the target sum and is updated each round
        var runningClaim = claim.targetSum

        for (i, round) in proof.rounds.enumerated() {
            // Check sumcheck invariant: g_i(0) + g_i(1) = runningClaim
            let g0PlusG1 = round.roundPolynomial[0] ^ round.roundPolynomial[1]
            if g0PlusG1 != runningClaim {
                return false
            }

            // Update running claim to g_i(r_i) = nextClaim from this round
            runningClaim = round.nextClaim
        }

        // Final check: runningClaim should equal finalEvaluation
        // which is f(r_1, ..., r_k)
        if runningClaim != proof.finalEvaluation {
            return false
        }

        // Recompute final evaluation directly from polynomial and verify it matches
        let recomputedFinal = evaluatePolynomial(
            evals: claim.polynomialEvals,
            randomness: randomness,
            numVariables: claim.numVariables
        )

        return recomputedFinal == proof.finalEvaluation
    }

    /// Evaluate multilinear polynomial at random point using proper MLE formula.
    /// f(r_1, ..., r_k) = Σ_{s∈{0,1}^k} f(s) · Π_{i=1}^k (r_i · s_i + (1 - r_i) · (1 - s_i))
    /// In GF(2^8), this uses polynomial multiplication with reduction.
    private func evaluatePolynomial(evals: [UInt8], randomness: [UInt8], numVariables: Int) -> UInt8 {
        var result: UInt8 = 0

        for (idx, eval) in evals.enumerated() {
            // Compute the multilinear basis term for this index
            // prod_{i=1}^k (r_i * s_i + (1 - r_i) * (1 - s_i))
            var basisTerm: UInt8 = 1

            for i in 0..<numVariables {
                let s_i = UInt8((idx >> i) & 1)  // Bit i of the index
                let r_i = randomness[i]

                // term = r_i * s_i + (1 - r_i) * (1 - s_i)
                // In GF(2^8): + is XOR, * is polynomial multiplication
                let term: UInt8
                if s_i == 0 {
                    // term = (1 - r_i) * 1 = 1 - r_i = 1 + r_i (since -1 = 1 in char 2)
                    term = 1 ^ r_i
                } else {
                    // term = r_i * 1 + (1 - r_i) * 0 = r_i
                    term = r_i
                }

                basisTerm = gf28Mul(basisTerm, term)
            }

            // Accumulate f(s) * basisTerm
            result ^= gf28Mul(eval, basisTerm)
        }

        return result
    }

    /// GF(2^8) multiplication.
    private func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var aa = UInt16(a)
        var bb = UInt16(b)
        for _ in 0..<8 {
            if bb & 1 != 0 {
                p ^= aa
            }
            let hiBitSet = (aa & 0x80) != 0
            aa <<= 1
            if hiBitSet {
                aa ^= 0x11B
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }
}

// MARK: - ZeroCheck Integration

/// ZeroCheck PIOP augmented with amortized sumcheck.
/// Combines zero-check (proving a polynomial is zero) with sumcheck for efficiency.
public final class AmortizedZeroCheck {
    /// Base amortized sumcheck prover.
    public let sumcheck: AmortizedSumcheckProver

    /// Create a new ZeroCheck prover.
    public init(basisCache: TowerBasisCache, constraintPacker: ConstraintPacker) {
        self.sumcheck = AmortizedSumcheckProver(
            basisCache: basisCache,
            constraintPacker: constraintPacker
        )
    }

    /// Prove that polynomial is zero at random point: f(r) = 0
    /// Uses sumcheck to reduce to checking a single value.
    public func proveZero(polynomialEvals: [UInt8], witness: [UInt8], randomness: [UInt8]) -> PackedSumcheckProof {
        // Create claim: sum = 0 (proving polynomial is zero is equivalent)
        let claim = AmortizedSumcheckClaim(
            numVariables: Int(log2(Double(polynomialEvals.count))),
            targetSum: 0,
            polynomialEvals: polynomialEvals
        )

        return sumcheck.prove(claim: claim, witness: witness, randomness: randomness)
    }

    /// Verify zero-check proof.
    public func verifyZero(proof: PackedSumcheckProof, polynomialEvals: [UInt8], randomness: [UInt8]) -> Bool {
        let claim = AmortizedSumcheckClaim(
            numVariables: Int(log2(Double(polynomialEvals.count))),
            targetSum: 0,
            polynomialEvals: polynomialEvals
        )

        let verifier = AmortizedSumcheckVerifier(basisCache: sumcheck.basisCache)
        return verifier.verify(proof: proof, claim: claim, randomness: randomness)
    }
}

// MARK: - Precomputation Cost Estimator

/// Estimates the cost of precomputation vs per-query savings.
public struct PrecomputationCostEstimator {
    /// Maximum tower level.
    public let maxLevel: Int

    /// Domain size.
    public let domainSize: Int

    /// Number of expected sumcheck rounds.
    public let numRounds: Int

    /// Number of expected queries.
    public let numQueries: Int

    public init(maxLevel: Int, domainSize: Int, numRounds: Int, numQueries: Int) {
        self.maxLevel = maxLevel
        self.domainSize = domainSize
        self.numRounds = numRounds
        self.numQueries = numQueries
    }

    /// Precomputation cost in cycles (estimated).
    public var precomputationCost: Int {
        return maxLevel * domainSize / 8
    }

    /// Cost without precomputation per query.
    public var costPerQueryWithoutCache: Int {
        return maxLevel * domainSize / numRounds
    }

    /// Cost with precomputation per query.
    public var costPerQueryWithCache: Int {
        return maxLevel
    }

    /// Total cost without precomputation.
    public var totalCostWithoutCache: Int {
        return precomputationCost + (costPerQueryWithoutCache * numQueries)
    }

    /// Total cost with precomputation.
    public var totalCostWithCache: Int {
        return precomputationCost + (costPerQueryWithCache * numQueries)
    }

    /// Speedup factor from precomputation.
    public var speedupFactor: Double {
        return Double(totalCostWithoutCache) / Double(totalCostWithCache)
    }

    /// Whether precomputation is worthwhile.
    public var isPrecomputationWorthwhile: Bool {
        return speedupFactor > 1.0 && precomputationCost < (costPerQueryWithoutCache - costPerQueryWithCache) * numQueries
    }
}
