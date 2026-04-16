// BinaryProximityGap — Proximity gap framework for binary FRI
//
// Implements the proximity gap between honest provers and malicious provers
// in binary FRI. The gap ensures that:
//   - An honest prover can always convince the verifier (within soundness error)
//   - A malicious prover must deviate significantly to fool the verifier
//
// Key concepts:
//   - Proximity: How close a word is to a valid codeword
//   - Soundness error: Probability of false acceptance
//   - Gap: Ratio between malicious prover success and honest prover success

import Foundation

// MARK: - Proximity Gap Parameters

/// Parameters controlling the proximity gap in binary FRI.
public struct ProximityGapParams {
    /// Security parameter (soundness bits)
    public let securityBits: Int

    /// Number of queries to the commitment
    public let numQueries: Int

    /// Extension degree of the binary field
    public let extensionDegree: Int

    /// Folding factor per round
    public let foldingFactor: Int

    /// Create proximity gap parameters.
    public init(
        securityBits: Int = 40,
        numQueries: Int = 32,
        extensionDegree: Int = 128,
        foldingFactor: Int = 2
    ) {
        self.securityBits = securityBits
        self.numQueries = numQueries
        self.extensionDegree = extensionDegree
        self.foldingFactor = foldingFactor
    }

    /// The soundness error probability (should be <= 2^{-securityBits}).
    public var soundnessError: Double {
        // For binary FRI with q queries, soundness error is roughly q / 2^m
        // where m is the extension degree
        return Double(numQueries) / pow(2.0, Double(extensionDegree))
    }

    /// Check if parameters achieve the desired security level.
    public var achievesSecurity: Bool {
        return -log2(soundnessError) >= Double(securityBits)
    }
}

// MARK: - Proximity Gap Framework

/// Framework for analyzing and ensuring the proximity gap in binary FRI.
///
/// The proximity gap ensures that for a word w and a code C:
///   - If w is far from C (distance > threshold), verifier rejects
///   - If w is close to C (distance < threshold), verifier accepts with high probability
///
/// This creates a "gap" between the accept region and reject region.
public struct BinaryProximityGap<B: BinaryTowerProtocol> {

    /// Configuration parameters
    public let params: ProximityGapParams

    /// The distance threshold for rejection (Johnson radius)
    public let rejectionThreshold: Int

    /// Create a proximity gap framework.
    public init(params: ProximityGapParams, rejectionThreshold: Int) {
        self.params = params
        self.rejectionThreshold = rejectionThreshold
    }

    // MARK: - Gap Analysis

    /// Compute the proximity gap for given code and parameters.
    ///
    /// The gap is defined as:
    ///   gap = rejection_threshold / acceptance_threshold
    ///
    /// A larger gap means better security (easier to detect malicious provers).
    public func computeGap(codeDistance: Int) -> Double {
        // Johnson bound gives the acceptance radius for list decoding
        // Standard unique decoding has radius (d-1)/2
        let acceptanceRadius = codeDistance / 2
        let rejectionRadius = rejectionThreshold

        return Double(rejectionRadius) / Double(acceptanceRadius)
    }

    /// Check if a word is within the accept region.
    public func isWithinAcceptRegion(distance: Int) -> Bool {
        return distance <= params.numQueries
    }

    /// Check if a word is within the reject region.
    public func isWithinRejectRegion(distance: Int) -> Bool {
        return distance > rejectionThreshold
    }

    /// The gap between accept and reject regions.
    public var gapSize: Int {
        return rejectionThreshold - params.numQueries
    }

    // MARK: - Soundness Analysis

    /// Compute the malicious prover success probability.
    ///
    /// Given a word at distance d from the nearest codeword,
    /// the probability of convincing the verifier is bounded by
    /// the Johnson bound for the code.
    public func maliciousSuccessProbability(distance: Int) -> Double {
        if isWithinRejectRegion(distance: distance) {
            return 0.0  // Far from code - will be rejected
        }

        if isWithinAcceptRegion(distance: distance) {
            // Within Johnson radius - could be accepted
            // The exact probability depends on the number of codewords
            // within the Johnson radius
            let params = JohnsonBoundParams(
                n: 1 << params.extensionDegree,
                d: distance,
                L: gapSize
            )
            return Double(params.L) / pow(2.0, Double(params.n))
        }

        // In the gap region - bounded by gap size
        return Double(gapSize) / pow(2.0, Double(params.extensionDegree))
    }

    /// Compute the honest prover success probability.
    ///
    /// An honest prover always produces a valid proof, so success
    /// probability is 1 minus the soundness error.
    public var honestSuccessProbability: Double {
        return 1.0 - soundnessError
    }

    /// The soundness error for these parameters.
    public var soundnessError: Double {
        return params.soundnessError
    }

    // MARK: - Fiat-Shamir Security

    /// Check that the hash function used for Fiat-Shamir is appropriate.
    public func validateHashFunction(domainSize: Int) -> Bool {
        // The hash function output must be large enough to prevent
        // pre-image attacks. We need at least securityBits bits.
        let hashBits = Int(log2(Double(domainSize)))
        return hashBits >= params.securityBits
    }
}

// MARK: - Interactive Proof Simulation

/// Simulates the interactive proof between prover and verifier
/// for proximity testing.
public struct BinaryProximityProofSimulator<B: BinaryTowerProtocol> {

    /// The proximity gap framework
    public let gap: BinaryProximityGap<B>

    /// The co-curvilinearity tester
    public let coCurvilinear: BinaryCoCurvilinear<B>

    public init(gap: BinaryProximityGap<B>, coCurvilinear: BinaryCoCurvilinear<B>) {
        self.gap = gap
        self.coCurvilinear = coCurvilinear
    }

    /// Simulate the verifier's check for a single query.
    ///
    /// - Parameters:
    ///   - word: The word being tested
    ///   - queryIndex: Index of the query point
    ///   - commitment: Merkle root of the commitment
    /// - Returns: True if the verifier accepts this query
    public func verifyQuery(
        word: [B],
        queryIndex: Int,
        commitment: B
    ) -> Bool {
        // In a real implementation, this would:
        // 1. Request the authentication path from the prover
        // 2. Verify the Merkle proof
        // 3. Check co-curvilinearity of the revealed values

        // Simplified: just check that the word has reasonable structure
        guard !word.isEmpty else { return false }

        // Check proximity via co-curvilinearity
        let lineFit = coCurvilinear.fitLine(points: word)
        return lineFit != nil
    }

    /// Simulate the full proof verification.
    ///
    /// - Parameters:
    ///   - layers: All folded layers
    ///   - alphas: Folding challenges
    ///   - queryIndices: Indices queried by verifier
    /// - Returns: True if all queries pass
    public func verifyProof(
        layers: [[B]],
        alphas: [B],
        queryIndices: [Int]
    ) -> Bool {
        // For each query, verify through all layers
        for queryIdx in queryIndices {
            var currentWord = layers[0]

            for (round, alpha) in alphas.enumerated() {
                // Extract the relevant portion of the word at this round
                let queryPoint = currentWord[queryIdx % currentWord.count]

                // Check co-curvilinearity
                if !verifyQueryAtRound(
                    word: currentWord,
                    queryIndex: queryIdx,
                    round: round,
                    alpha: alpha
                ) {
                    return false
                }

                // Fold for next round
                if round < alphas.count - 1 {
                    currentWord = foldWord(currentWord, by: alphas[round])
                }
            }
        }
        return true
    }

    /// Verify at a specific round.
    private func verifyQueryAtRound(
        word: [B],
        queryIndex: Int,
        round: Int,
        alpha: B
    ) -> Bool {
        // Verify the fold constraint at this round
        let idx1 = queryIndex % word.count
        let idx2 = (queryIndex + (word.count / 2)) % word.count

        let f0 = word[idx1]
        let f1 = word[idx2]

        // The fold equation: f' = f0 + alpha * f1
        // We check this is consistent with the co-curvilinear constraint
        return coCurvilinear.testWithOracle(
            points: [f0, f1],
            randomOracle: alpha
        )
    }

    /// Fold a word by one round.
    private func foldWord(_ word: [B], by alpha: B) -> [B] {
        let halfSize = word.count / 2
        var folded = [B](repeating: .zero, count: halfSize)

        for i in 0..<halfSize {
            folded[i] = word[i] + alpha * word[i + halfSize]
        }
        return folded
    }
}

// MARK: - Gap Amplification

/// Utilities for amplifying the proximity gap through parallel repetition.
public struct GapAmplifier {

    /// Amplify the gap through parallel repetition.
    ///
    /// If a protocol has soundness error ε, running it t times in parallel
    /// gives soundness error ε^t.
    ///
    /// - Parameters:
    ///   - baseError: Base soundness error
    ///   - repetitions: Number of parallel repetitions
    /// - Returns: Amplified soundness error
    public static func amplify(baseError: Double, repetitions: Int) -> Double {
        return pow(baseError, Double(repetitions))
    }

    /// Compute the number of repetitions needed for a target security level.
    ///
    /// - Parameters:
    ///   - baseError: Base soundness error
    ///   - targetSecurityBits: Desired security level
    /// - Returns: Number of repetitions needed
    public static func repetitionsForSecurity(
        baseError: Double,
        targetSecurityBits: Int
    ) -> Int {
        let targetError = pow(2.0, -Double(targetSecurityBits))
        return Int(ceil(log(targetError) / log(baseError)))
    }
}
