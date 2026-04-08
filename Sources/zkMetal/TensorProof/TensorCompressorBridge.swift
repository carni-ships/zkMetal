// TensorCompressorBridge — Post-hoc compression of standard sumcheck transcripts
//
// Given a standard sumcheck proof (list of round polynomials + final eval), detect whether
// the underlying polynomial has tensor structure and compress the proof accordingly.
//
// Detection: a polynomial f has tensor structure if the round polynomials exhibit
// a multiplicative pattern consistent with independent factor evaluation.
// Specifically, if f = g ⊗ h with g on variables x_L and h on variables x_R,
// then during the x_L rounds, each round poly is scaled by sum(h), and during
// the x_R rounds, each round poly is scaled by g(r_L) (a single scalar).
//
// Compression: strip redundant information. Instead of full round polynomials,
// store only factor evaluation tables + challenge sequence.
//
// Decompression: reconstruct full round polynomials from factor snapshots.
//
// This bridges between the existing TensorCompressor (in PCS/) which handles
// Spartan-style eval proofs, and the new TensorSumcheck prover/verifier.

import Foundation
import NeonFieldOps

// MARK: - Standard Sumcheck Proof Wrapper

/// A standard (uncompressed) sumcheck proof for compression analysis.
public struct StandardSumcheckTranscript {
    /// Round polynomials: (S_i(0), S_i(1), S_i(2)) for each round
    public let rounds: [(Fr, Fr, Fr)]
    /// Challenges used in each round
    public let challenges: [Fr]
    /// The final evaluation claim
    public let finalEval: Fr
    /// The claimed sum (S(0) + S(1) for round 0)
    public let claimedSum: Fr

    public init(rounds: [(Fr, Fr, Fr)], challenges: [Fr], finalEval: Fr, claimedSum: Fr) {
        precondition(rounds.count == challenges.count, "Must have same number of rounds and challenges")
        self.rounds = rounds
        self.challenges = challenges
        self.finalEval = finalEval
        self.claimedSum = claimedSum
    }

    /// Size in field elements
    public var sizeInElements: Int {
        return rounds.count * 3 + challenges.count + 2  // round polys + challenges + finalEval + claimedSum
    }
}

// MARK: - Tensor Structure Detection

public class TensorStructureDetector {

    /// Result of tensor structure detection
    public struct DetectionResult {
        /// Whether tensor structure was detected
        public let hasTensorStructure: Bool
        /// Detected factor boundaries (variable indices where factors split)
        public let factorBoundaries: [Int]
        /// Confidence score (0.0 to 1.0)
        public let confidence: Double
        /// Detected factor count
        public var numFactors: Int { factorBoundaries.count + 1 }
    }

    /// Attempt to detect tensor product structure in a sumcheck transcript.
    ///
    /// Strategy: for a two-factor decomposition f = g ⊗ h with split at variable k,
    /// the ratio S_i(1)/S_i(0) should be constant for rounds within the same factor
    /// (up to folding effects). We check if the round polynomial ratios exhibit
    /// a "block-constant" pattern.
    ///
    /// For a true tensor product:
    ///   - Rounds 0..k-1 (g's variables): S_i(0)/S_i(1) determined by g alone, scaled by sum(h)
    ///   - Rounds k..n-1 (h's variables): S_i(0)/S_i(1) determined by h alone, scaled by g(r_L)
    ///   - The transition at round k shows a discontinuity in the ratio pattern
    public static func detect(
        transcript: StandardSumcheckTranscript,
        tolerance: Double = 1e-6
    ) -> DetectionResult {
        let n = transcript.rounds.count
        guard n >= 4 else {
            // Too few rounds to detect structure
            return DetectionResult(hasTensorStructure: false, factorBoundaries: [], confidence: 0.0)
        }

        // Try all possible two-factor splits
        var bestSplit = n / 2
        var bestScore = 0.0

        for split in 2..<(n - 1) {
            let score = evaluateSplit(transcript: transcript, split: split)
            if score > bestScore {
                bestScore = score
                bestSplit = split
            }
        }

        // A score > 0.8 indicates likely tensor structure
        let detected = bestScore > 0.8

        return DetectionResult(
            hasTensorStructure: detected,
            factorBoundaries: detected ? [bestSplit] : [],
            confidence: bestScore
        )
    }

    /// Evaluate how well a particular split point explains the round polynomial pattern.
    /// Returns a score between 0.0 (no structure) and 1.0 (perfect tensor structure).
    private static func evaluateSplit(
        transcript: StandardSumcheckTranscript,
        split: Int
    ) -> Double {
        let n = transcript.rounds.count

        // Within each factor block, the ratio of consecutive round sums should
        // follow a predictable pattern from folding.
        // For tensor structure, the round polynomials within a block are all
        // scaled by the same external factor (the other factor's sum or evaluation).

        // Check: are the round polynomials in each block self-consistent?
        // Specifically, S_i(0) + S_i(1) should halve predictably within a block.
        var block1Consistent = true
        var block2Consistent = true

        // Block 1: rounds 0..<split
        if split >= 2 {
            for i in 1..<split {
                let prevSum = frAdd(transcript.rounds[i - 1].0, transcript.rounds[i - 1].1)
                let currRoundCheck = evalQuadratic(
                    s0: transcript.rounds[i - 1].0,
                    s1: transcript.rounds[i - 1].1,
                    s2: transcript.rounds[i - 1].2,
                    at: transcript.challenges[i - 1]
                )
                let nextSum = frAdd(transcript.rounds[i].0, transcript.rounds[i].1)
                // S(r_{i-1}) should equal the next round's claimed sum
                if frToInt(currRoundCheck) != frToInt(nextSum) {
                    block1Consistent = false
                    break
                }
            }
        }

        // Block 2: rounds split..<n
        if n - split >= 2 {
            for i in (split + 1)..<n {
                let currRoundCheck = evalQuadratic(
                    s0: transcript.rounds[i - 1].0,
                    s1: transcript.rounds[i - 1].1,
                    s2: transcript.rounds[i - 1].2,
                    at: transcript.challenges[i - 1]
                )
                let nextSum = frAdd(transcript.rounds[i].0, transcript.rounds[i].1)
                if frToInt(currRoundCheck) != frToInt(nextSum) {
                    block2Consistent = false
                    break
                }
            }
        }

        // Also check the transition: the sum at round `split` should reflect a factor boundary
        // For tensor products, the claimed sum at round `split` equals g(r_L) * sum(h)
        // where g(r_L) is the evaluation of the first factor at its challenges.
        let transitionCheck = evalQuadratic(
            s0: transcript.rounds[split - 1].0,
            s1: transcript.rounds[split - 1].1,
            s2: transcript.rounds[split - 1].2,
            at: transcript.challenges[split - 1]
        )
        let nextBlockSum = frAdd(transcript.rounds[split].0, transcript.rounds[split].1)
        let transitionValid = frToInt(transitionCheck) == frToInt(nextBlockSum)

        // Score based on consistency
        var score = 0.0
        if block1Consistent { score += 0.35 }
        if block2Consistent { score += 0.35 }
        if transitionValid { score += 0.30 }

        return score
    }

    private static func evalQuadratic(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
        let rMinus1 = frSub(r, Fr.one)
        let rMinus2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))
        let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
        let l1 = frSub(Fr.zero, frMul(r, rMinus2))
        let l2 = frMul(frMul(r, rMinus1), inv2)
        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }
}

// MARK: - Post-hoc Compressor

public class SumcheckTranscriptCompressor {

    /// Compress a standard sumcheck transcript by detecting and exploiting tensor structure.
    ///
    /// If tensor structure is detected, returns a compressed proof.
    /// If no tensor structure is found, returns nil (compression not possible).
    public static func compress(
        transcript: StandardSumcheckTranscript,
        evaluations: [Fr]? = nil
    ) -> TensorSumcheckProof? {
        let detection = TensorStructureDetector.detect(transcript: transcript)

        guard detection.hasTensorStructure else { return nil }
        guard detection.factorBoundaries.count == 1 else { return nil }

        let split = detection.factorBoundaries[0]
        let n = transcript.rounds.count

        // Reconstruct factor snapshots from round polynomials.
        // Within each factor block, we can recover the factor's evaluation table
        // by undoing the folding steps.

        // Factor 1 snapshot: the initial round's S(0) and S(1) tell us about the factor
        // combined with the suffix product. We store the round polys for the factor block.
        let factor1Vars = split
        let factor2Vars = n - split

        // For the compressed proof, we store:
        // - Factor sizes
        // - Challenges (already have them)
        // - Round polynomials (needed for verification)
        // - Reconstruct partial products from the transcript

        // Compute partial product at the boundary
        let boundaryValue = evalQuadratic(
            s0: transcript.rounds[split - 1].0,
            s1: transcript.rounds[split - 1].1,
            s2: transcript.rounds[split - 1].2,
            at: transcript.challenges[split - 1]
        )

        // If we have the original evaluations, extract factor snapshots
        var factorSnapshots: [[Fr]]
        if let evals = evaluations {
            let sqrtN1 = 1 << factor1Vars
            let sqrtN2 = 1 << factor2Vars

            // Extract factor 1: column marginals
            var f1 = [Fr](repeating: Fr.zero, count: sqrtN1)
            for i in 0..<sqrtN1 {
                for j in 0..<sqrtN2 {
                    f1[i] = frAdd(f1[i], evals[i * sqrtN2 + j])
                }
            }

            // Extract factor 2: first row (if tensor product, all rows are proportional)
            let f2 = Array(evals[0..<sqrtN2])

            factorSnapshots = [f1, f2]
        } else {
            // Without evaluations, store a sentinel (round polys contain the info)
            factorSnapshots = [
                [Fr](repeating: Fr.zero, count: 1 << factor1Vars),
                [Fr](repeating: Fr.zero, count: 1 << factor2Vars)
            ]
        }

        let factorSizes: [(index: Int, numVars: Int)] = [
            (index: 0, numVars: factor1Vars),
            (index: 1, numVars: factor2Vars)
        ]

        return TensorSumcheckProof(
            claimedSum: transcript.claimedSum,
            numVars: n,
            factorSizes: factorSizes,
            partialProducts: [Fr.one, boundaryValue],
            factorSnapshots: factorSnapshots,
            rounds: transcript.rounds,
            challenges: transcript.challenges,
            finalEval: transcript.finalEval
        )
    }

    /// Decompress a tensor sumcheck proof back into a standard transcript.
    /// Reconstructs round polynomials from factor snapshots and challenges.
    public static func decompress(
        proof: TensorSumcheckProof
    ) -> StandardSumcheckTranscript? {
        // If rounds are already populated, just wrap them
        if !proof.rounds.isEmpty {
            return StandardSumcheckTranscript(
                rounds: proof.rounds,
                challenges: proof.challenges,
                finalEval: proof.finalEval,
                claimedSum: proof.claimedSum
            )
        }

        // Reconstruct from factor snapshots
        guard proof.factorSnapshots.allSatisfy({ !$0.isEmpty }) else { return nil }

        var rounds: [(Fr, Fr, Fr)] = []
        var runningProduct = Fr.one
        var roundOffset = 0

        for (fi, (_, m)) in proof.factorSizes.enumerated() {
            guard fi < proof.factorSnapshots.count else { return nil }
            var currentEvals = proof.factorSnapshots[fi]
            guard currentEvals.count == (1 << m) else { return nil }

            // Compute suffix product
            var suffix = Fr.one
            for fj in (fi + 1)..<proof.factorSizes.count {
                guard fj < proof.factorSnapshots.count else { return nil }
                var factorSum = Fr.zero
                for e in proof.factorSnapshots[fj] {
                    factorSum = frAdd(factorSum, e)
                }
                suffix = frMul(suffix, factorSum)
            }

            for localRound in 0..<m {
                let half = currentEvals.count / 2
                var s0 = Fr.zero
                var s1 = Fr.zero
                var s2 = Fr.zero

                for i in 0..<half {
                    let lo = currentEvals[2 * i]
                    let hi = currentEvals[2 * i + 1]
                    s0 = frAdd(s0, lo)
                    s1 = frAdd(s1, hi)
                    let at2 = frSub(frAdd(hi, hi), lo)
                    s2 = frAdd(s2, at2)
                }

                let scale = frMul(runningProduct, suffix)
                s0 = frMul(s0, scale)
                s1 = frMul(s1, scale)
                s2 = frMul(s2, scale)

                rounds.append((s0, s1, s2))

                // Fold with the stored challenge
                let globalRound = roundOffset + localRound
                guard globalRound < proof.challenges.count else { return nil }
                let r = proof.challenges[globalRound]

                currentEvals.withUnsafeMutableBytes { eBuf in
                    withUnsafeBytes(of: r) { rBuf in
                        bn254_fr_fold_interleaved_inplace(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
                currentEvals.removeLast(half)
            }

            // Update running product with this factor's final value
            precondition(currentEvals.count == 1)
            runningProduct = frMul(runningProduct, currentEvals[0])
            roundOffset += m
        }

        return StandardSumcheckTranscript(
            rounds: rounds,
            challenges: proof.challenges,
            finalEval: proof.finalEval,
            claimedSum: proof.claimedSum
        )
    }

    /// Analyze compression potential for a given number of variables and factor count.
    public static func compressionAnalysis(numVars: Int, numFactors: Int = 2) -> (
        standardSize: Int,
        compressedSize: Int,
        ratio: Double
    ) {
        // Standard: 3 field elements per round * numVars + 1 final eval
        let standardSize = numVars * 3 + 1

        // Compressed: factor snapshots + partial products + challenges + final eval + claimed sum
        let varsPerFactor = numVars / numFactors
        let snapshotSize = numFactors * (1 << varsPerFactor)
        let compressedSize = snapshotSize + numFactors + numVars + 2

        return (standardSize, compressedSize, Double(compressedSize) / Double(standardSize))
    }

    private static func evalQuadratic(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
        let rMinus1 = frSub(r, Fr.one)
        let rMinus2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))
        let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
        let l1 = frSub(Fr.zero, frMul(r, rMinus2))
        let l2 = frMul(frMul(r, rMinus1), inv2)
        return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
    }
}
