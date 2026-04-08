// GPUProofCompositionEngine — GPU-accelerated proof composition and recursive verification
//
// Building blocks for proof aggregation, recursive/IVC proof composition:
//   - Linear combination of multiple proof polynomials using random challenges
//   - Batch opening computation (combine multiple polynomial openings)
//   - Accumulator state management for recursive/IVC proofs
//   - Fiat-Shamir challenge derivation from proof transcripts
//   - Multi-point evaluation combination
//
// Works with BN254 Fr field type. CPU-only (field arithmetic via NEON C ops);
// designed to be composed with GPU MSM/NTT engines for full recursive pipelines.

import Foundation
import NeonFieldOps

// MARK: - Accumulator State

/// Accumulator for recursive/IVC proof folding.
/// Stores the running commitment (as a polynomial in evaluation form)
/// plus the accumulated error term and challenge history.
public struct ProofAccumulator {
    /// Running accumulated polynomial coefficients
    public var accPoly: [Fr]
    /// Accumulated error scalar (folding slack)
    public var error: Fr
    /// Challenge history for audit/replay
    public var challenges: [Fr]
    /// Number of proofs folded into this accumulator
    public var foldCount: Int

    /// Create a fresh accumulator from an initial polynomial.
    public static func initial(poly: [Fr]) -> ProofAccumulator {
        ProofAccumulator(accPoly: poly, error: Fr.zero, challenges: [], foldCount: 1)
    }

    /// Create a zero accumulator of given size.
    public static func zero(size: Int) -> ProofAccumulator {
        ProofAccumulator(
            accPoly: [Fr](repeating: Fr.zero, count: size),
            error: Fr.zero, challenges: [], foldCount: 0)
    }
}

// MARK: - Transcript for Fiat-Shamir

/// Lightweight Fiat-Shamir transcript for proof composition.
/// Uses Blake3 to derive deterministic challenges from proof data.
public struct CompositionTranscript {
    private var state: [UInt8] = []

    public init() {}

    /// Append a domain separator label.
    public mutating func appendLabel(_ label: String) {
        state.append(contentsOf: Array(label.utf8))
    }

    /// Append a field element to the transcript.
    public mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    /// Append multiple field elements.
    public mutating func appendScalars(_ scalars: [Fr]) {
        for s in scalars { appendScalar(s) }
    }

    /// Append raw bytes.
    public mutating func appendBytes(_ bytes: [UInt8]) {
        state.append(contentsOf: bytes)
    }

    /// Squeeze a challenge from the current transcript state.
    public func challenge() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        // Reduce into Fr field
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    /// Squeeze a challenge and advance the transcript state (append the challenge back).
    public mutating func challengeAndAdvance() -> Fr {
        let c = challenge()
        appendScalar(c)
        return c
    }
}

// MARK: - GPU Proof Composition Engine

/// Engine for proof composition and recursive verification building blocks.
///
/// Provides the core operations needed to aggregate multiple proofs:
///   - Linear combination of polynomial vectors with random challenges
///   - Batch opening: combine evaluations from multiple polynomials
///   - Accumulator fold: merge a new proof into a running accumulator
///   - Fiat-Shamir challenge derivation from proof transcripts
///   - Multi-point evaluation combination for opening proofs
public class GPUProofCompositionEngine {

    public static let version = Versions.gpuProofComposition

    public init() {}

    // MARK: - Linear Combination

    /// Compute a random linear combination of polynomials:
    ///   result[i] = sum_j (challenges[j] * polys[j][i])
    ///
    /// All polynomials must have the same length. The number of challenges
    /// must equal the number of polynomials.
    ///
    /// - Parameters:
    ///   - polys: array of polynomial coefficient vectors (all same length)
    ///   - challenges: random scalars, one per polynomial
    /// - Returns: combined polynomial coefficients
    public func linearCombination(polys: [[Fr]], challenges: [Fr]) -> [Fr] {
        precondition(!polys.isEmpty, "Need at least one polynomial")
        precondition(polys.count == challenges.count,
                     "Number of challenges must match number of polynomials")
        let n = polys[0].count
        for p in polys { precondition(p.count == n, "All polynomials must have same length") }

        var result = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<polys.count {
            let c = challenges[j]
            let poly = polys[j]
            for i in 0..<n {
                result[i] = frAdd(result[i], frMul(c, poly[i]))
            }
        }
        return result
    }

    // MARK: - Batch Opening

    /// Combine multiple polynomial openings at the same point into a single check.
    ///
    /// Given polynomials p_0, ..., p_{k-1} each evaluated at the same point z
    /// with claimed values v_0, ..., v_{k-1}, and random challenges r_0, ..., r_{k-1},
    /// computes the combined polynomial and combined evaluation:
    ///   combinedPoly = sum_j r_j * p_j
    ///   combinedEval = sum_j r_j * v_j
    ///
    /// The verifier then only needs to check one opening: combinedPoly(z) == combinedEval.
    ///
    /// - Parameters:
    ///   - polys: polynomial coefficient vectors
    ///   - evals: claimed evaluations v_j = p_j(z)
    ///   - challenges: random batching scalars
    /// - Returns: (combinedPoly, combinedEval)
    public func batchOpening(
        polys: [[Fr]], evals: [Fr], challenges: [Fr]
    ) -> (combinedPoly: [Fr], combinedEval: Fr) {
        precondition(polys.count == evals.count && polys.count == challenges.count)
        precondition(!polys.isEmpty)

        let combinedPoly = linearCombination(polys: polys, challenges: challenges)

        var combinedEval = Fr.zero
        for j in 0..<evals.count {
            combinedEval = frAdd(combinedEval, frMul(challenges[j], evals[j]))
        }

        return (combinedPoly, combinedEval)
    }

    // MARK: - Accumulator Fold

    /// Fold a new polynomial into an existing accumulator using a random challenge.
    ///
    /// Given accumulator with polynomial acc and a new polynomial new_poly,
    /// and a Fiat-Shamir challenge r:
    ///   acc' = acc + r * new_poly
    ///   error' = error + r^2 * cross_term
    ///
    /// The cross_term is the inner product <acc, new_poly> which captures the
    /// "slack" introduced by the folding (needed for soundness of the IVC scheme).
    ///
    /// - Parameters:
    ///   - accumulator: current accumulator state
    ///   - newPoly: new polynomial to fold in
    ///   - challenge: Fiat-Shamir random challenge
    /// - Returns: updated accumulator
    public func fold(
        accumulator: ProofAccumulator,
        newPoly: [Fr],
        challenge: Fr
    ) -> ProofAccumulator {
        let n = accumulator.accPoly.count
        precondition(newPoly.count == n, "Polynomial sizes must match for folding")

        // Compute cross term: <acc, new>
        var crossTerm = Fr.zero
        for i in 0..<n {
            crossTerm = frAdd(crossTerm, frMul(accumulator.accPoly[i], newPoly[i]))
        }

        // acc' = acc + r * new
        var newAcc = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            newAcc[i] = frAdd(accumulator.accPoly[i], frMul(challenge, newPoly[i]))
        }

        // error' = error + r^2 * cross_term
        let r2 = frMul(challenge, challenge)
        let newError = frAdd(accumulator.error, frMul(r2, crossTerm))

        var updatedChallenges = accumulator.challenges
        updatedChallenges.append(challenge)

        return ProofAccumulator(
            accPoly: newAcc,
            error: newError,
            challenges: updatedChallenges,
            foldCount: accumulator.foldCount + 1)
    }

    // MARK: - Challenge Derivation

    /// Derive a Fiat-Shamir challenge from a proof transcript (list of field elements).
    ///
    /// This is deterministic: the same transcript always produces the same challenge.
    /// Uses Blake3 as the hash function with a domain separator.
    ///
    /// - Parameters:
    ///   - transcript: ordered field elements representing the proof transcript
    ///   - domainSeparator: label to bind the challenge to a specific protocol step
    /// - Returns: a field element challenge
    public func deriveChallenge(transcript: [Fr], domainSeparator: String) -> Fr {
        var t = CompositionTranscript()
        t.appendLabel("zkMetal-proof-composition")
        t.appendLabel(domainSeparator)
        t.appendScalars(transcript)
        return t.challenge()
    }

    // MARK: - Multi-Point Evaluation Combination

    /// Combine evaluations at multiple points into a single polynomial check.
    ///
    /// Given polynomial p evaluated at points z_0, ..., z_{k-1} with claimed
    /// values v_0, ..., v_{k-1}, constructs the combined quotient:
    ///   q(X) = sum_j gamma^j * (p(X) - v_j) / (X - z_j)
    ///
    /// This computes the numerator polynomials and combines them. The caller
    /// is responsible for the actual polynomial division or KZG opening.
    ///
    /// Returns the combined numerator: sum_j gamma^j * (poly - v_j), where
    /// the division by (X - z_j) is left to the commitment scheme.
    ///
    /// - Parameters:
    ///   - poly: polynomial coefficients
    ///   - points: evaluation points z_j
    ///   - values: claimed evaluations v_j
    ///   - gamma: batching challenge
    /// - Returns: (combinedNumerators, gammaPowers) where combinedNumerators[j]
    ///            is gamma^j * (poly - v_j) for each opening point
    public func multiPointCombination(
        poly: [Fr],
        points: [Fr],
        values: [Fr],
        gamma: Fr
    ) -> (combinedNumerators: [[Fr]], gammaFactors: [Fr]) {
        precondition(points.count == values.count)
        let k = points.count
        let n = poly.count

        // Compute gamma powers: gamma^0, gamma^1, ..., gamma^{k-1}
        var gammaFactors = [Fr](repeating: Fr.one, count: k)
        for j in 1..<k {
            gammaFactors[j] = frMul(gammaFactors[j - 1], gamma)
        }

        // For each point, compute gamma^j * (poly - v_j)
        var combinedNumerators = [[Fr]]()
        combinedNumerators.reserveCapacity(k)

        for j in 0..<k {
            var numerator = [Fr](repeating: Fr.zero, count: n)
            // poly - v_j: subtract v_j from constant term
            for i in 0..<n {
                numerator[i] = poly[i]
            }
            numerator[0] = frSub(numerator[0], values[j])

            // Scale by gamma^j
            let gj = gammaFactors[j]
            for i in 0..<n {
                numerator[i] = frMul(gj, numerator[i])
            }
            combinedNumerators.append(numerator)
        }

        return (combinedNumerators, gammaFactors)
    }

    // MARK: - Polynomial Evaluation (Horner's method)

    /// Evaluate polynomial at a point using Horner's method.
    /// poly = c_0 + c_1*x + c_2*x^2 + ...
    public func evaluatePolynomial(_ poly: [Fr], at point: Fr) -> Fr {
        guard !poly.isEmpty else { return Fr.zero }
        var result = Fr.zero
        poly.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: point) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(poly.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }
}
