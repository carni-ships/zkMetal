// Gemini Multi-Linear Opening for KZG
//
// Reduces a multilinear polynomial evaluation claim to univariate KZG openings
// via the split-and-fold technique.
//
// Used by Zeromorph and HyperKZG as the core multilinear-to-univariate reduction.
//
// Given: MLE f(x_1, ..., x_n) and evaluation point u = (u_1, ..., u_n),
// Gemini splits f into even/odd parts iteratively:
//   f_0 = f
//   f_{k+1}(X) = f_k^{even}(X) + u_{n-k} * f_k^{odd}(X)
// producing n fold polynomials A_0, ..., A_{n-1} of decreasing degree.
//
// The prover commits to each fold polynomial and then opens them at a
// random challenge point r and r^{-1} to enable the verifier to check
// the folding relation via:
//   A_{k+1}(r) = (1 - u_{n-k}) * A_k^{even}(r) + u_{n-k} * A_k^{odd}(r)
//   where A_k^{even}(r) = (A_k(r) + A_k(-r)) / 2
//         A_k^{odd}(r)  = (A_k(r) - A_k(-r)) / (2r)
//
// Reference: "Gemini: Elastic SNARKs for Diverse Environments" (Bootle et al., 2022)

import Foundation
import NeonFieldOps

// MARK: - Gemini Proof Structure

/// Proof for a Gemini multilinear-to-univariate reduction.
public struct GeminiProof {
    /// Commitments to fold polynomials A_1, ..., A_{n-1}
    /// (A_0 is the original polynomial commitment, not included)
    public let foldCommitments: [PointProjective]
    /// Evaluations of A_0, ..., A_{n-1} at the challenge point r
    public let evaluationsAtR: [Fr]
    /// Evaluations of A_0, ..., A_{n-1} at -r
    public let evaluationsAtNegR: [Fr]
    /// The Fiat-Shamir challenge point r
    public let challenge: Fr
    /// The claimed multilinear evaluation value
    public let claimedValue: Fr

    public init(foldCommitments: [PointProjective], evaluationsAtR: [Fr],
                evaluationsAtNegR: [Fr], challenge: Fr, claimedValue: Fr) {
        self.foldCommitments = foldCommitments
        self.evaluationsAtR = evaluationsAtR
        self.evaluationsAtNegR = evaluationsAtNegR
        self.challenge = challenge
        self.claimedValue = claimedValue
    }
}

// MARK: - Gemini Opening Engine

public class GeminiOpener {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    // MARK: - Open (Prove)

    /// Perform a Gemini multilinear opening.
    ///
    /// Given multilinear polynomial f (as 2^n evaluations on the boolean hypercube)
    /// and evaluation point u = (u_1, ..., u_n), proves that f(u) = value.
    ///
    /// - Parameters:
    ///   - multilinearPoly: 2^n evaluations [f(0,...,0), f(1,0,...,0), ...]
    ///   - point: evaluation point (u_1, ..., u_n)
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: GeminiProof containing fold commitments and evaluations
    public func geminiOpen(
        multilinearPoly: [Fr],
        point: [Fr],
        transcript: Transcript
    ) throws -> GeminiProof {
        let N = multilinearPoly.count
        let n = point.count
        precondition(N == (1 << n), "multilinearPoly must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree \(N)")

        // 1. Compute the claimed evaluation via multilinear extension
        let value = Self.evaluateMLE(multilinearPoly, at: point)

        // 2. Compute fold polynomials A_0, A_1, ..., A_{n-1}
        //    A_0 = multilinearPoly (interpreted as univariate coefficients)
        //    A_{k+1}(X) = A_k^{even}(X) + u_{n-1-k} * A_k^{odd}(X)
        var foldPolynomials = [[Fr]]()
        foldPolynomials.reserveCapacity(n)
        foldPolynomials.append(multilinearPoly)  // A_0

        var current = multilinearPoly
        for k in 0..<(n - 1) {
            let uk = point[n - 1 - k]
            let halfLen = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: halfLen)
            current.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: uk) { uBuf in
            folded.withUnsafeMutableBytes { fBuf in
                bn254_fr_fold_zm_interleaved(
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    uBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfLen))
            }}}
            foldPolynomials.append(folded)
            current = folded
        }
        // After n-1 folds, current should have 2 elements
        // The final fold with u_0 gives the evaluation value

        // 3. Commit to fold polynomials A_1, ..., A_{n-1}
        //    (A_0 commitment is provided by the caller / already known)
        var foldCommitments = [PointProjective]()
        foldCommitments.reserveCapacity(n - 1)
        for k in 1..<n {
            let c = try kzg.commit(foldPolynomials[k])
            foldCommitments.append(c)
        }

        // 4. Absorb fold commitments and squeeze challenge r
        for c in foldCommitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(value)
        let r = transcript.squeeze()

        // 5. Evaluate each fold polynomial at r_k and -r_k
        //    where r_0 = r, r_{k+1} = r_k^2
        var evalsAtR = [Fr]()
        var evalsAtNegR = [Fr]()
        evalsAtR.reserveCapacity(n)
        evalsAtNegR.reserveCapacity(n)

        var rk = r
        for k in 0..<n {
            let negRk = frNeg(rk)
            evalsAtR.append(cEvaluate(foldPolynomials[k], at: rk))
            evalsAtNegR.append(cEvaluate(foldPolynomials[k], at: negRk))
            rk = frMul(rk, rk)  // r_{k+1} = r_k^2
        }

        return GeminiProof(
            foldCommitments: foldCommitments,
            evaluationsAtR: evalsAtR,
            evaluationsAtNegR: evalsAtNegR,
            challenge: r,
            claimedValue: value
        )
    }

    // MARK: - Verify

    /// Verify a Gemini proof using the SRS secret (testing only).
    ///
    /// Checks the folding relation at the challenge point r:
    ///   A_{k+1}(r^2) = (A_k(r) + A_k(-r))/2 + u_{n-1-k} * (A_k(r) - A_k(-r))/(2r)
    ///
    /// At each fold level, the evaluation point squares:
    ///   Level 0: evaluated at r and -r
    ///   Level 1: evaluated at r^2 and -r^2
    ///   Level k: evaluated at r^{2^k} and -r^{2^k}
    ///
    /// The final single-element fold should equal the claimed evaluation.
    public func geminiVerify(
        commitment: PointProjective,
        point: [Fr],
        evaluation: Fr,
        proof: GeminiProof,
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let n = point.count
        guard proof.foldCommitments.count == n - 1 else { return false }
        guard proof.evaluationsAtR.count == n else { return false }
        guard proof.evaluationsAtNegR.count == n else { return false }

        // Check claimed value matches
        let claimedLimbs = frToInt(proof.claimedValue)
        let evalLimbs = frToInt(evaluation)
        guard claimedLimbs == evalLimbs else { return false }

        // Reconstruct transcript to get the same challenge r
        for c in proof.foldCommitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(proof.claimedValue)
        let r = transcript.squeeze()

        // Verify r matches
        let rLimbs = frToInt(r)
        let proofRLimbs = frToInt(proof.challenge)
        guard rLimbs == proofRLimbs else { return false }

        let two = frFromInt(2)
        let twoInv = frInverse(two)

        // 1. Verify folding relations.
        //    At step k, A_k is evaluated at r_k and -r_k where r_0 = r, r_{k+1} = r_k^2.
        //    The relation: A_{k+1}(r_k^2) = (A_k(r_k) + A_k(-r_k))/2 + u * (A_k(r_k) - A_k(-r_k))/(2*r_k)
        //    Since r_{k+1} = r_k^2, this is: A_{k+1}(r_{k+1}) = ...
        //    So we check that evaluationsAtR[k+1] (which is A_{k+1} at r_{k+1}) matches.
        var rk = r
        for k in 0..<(n - 1) {
            let uk = point[n - 1 - k]
            let akR = proof.evaluationsAtR[k]
            let akNegR = proof.evaluationsAtNegR[k]

            let rkInv = frInverse(rk)
            let twoRkInv = frMul(twoInv, rkInv)

            // even_k = (A_k(r_k) + A_k(-r_k)) / 2
            let evenK = frMul(frAdd(akR, akNegR), twoInv)
            // odd_k = (A_k(r_k) - A_k(-r_k)) / (2 * r_k)
            let oddK = frMul(frSub(akR, akNegR), twoRkInv)

            // expected A_{k+1}(r_k^2) = even_k + u_k * odd_k
            let expected = frAdd(evenK, frMul(uk, oddK))

            // A_{k+1} is evaluated at r_{k+1} = r_k^2
            let actual = proof.evaluationsAtR[k + 1]
            guard frToInt(expected) == frToInt(actual) else { return false }

            rk = frMul(rk, rk)  // r_{k+1} = r_k^2
        }

        // 2. Verify the final fold gives the claimed value.
        //    The last fold polynomial A_{n-1} has 2 coefficients [a, b].
        //    Its evaluation at r_{n-1} is known from proof.evaluationsAtR[n-1].
        //    Its evaluation at -r_{n-1} is known from proof.evaluationsAtNegR[n-1].
        //    The final fold with u_0 should give the claimed value:
        //    value = (A_{n-1}(r_{n-1}) + A_{n-1}(-r_{n-1}))/2
        //          + u_0 * (A_{n-1}(r_{n-1}) - A_{n-1}(-r_{n-1}))/(2*r_{n-1})
        let lastR = proof.evaluationsAtR[n - 1]
        let lastNegR = proof.evaluationsAtNegR[n - 1]
        let rkInv = frInverse(rk)
        let twoRkInv = frMul(twoInv, rkInv)
        let lastEven = frMul(frAdd(lastR, lastNegR), twoInv)
        let lastOdd = frMul(frSub(lastR, lastNegR), twoRkInv)
        let finalEval = frAdd(lastEven, frMul(point[0], lastOdd))
        guard frToInt(finalEval) == frToInt(evaluation) else { return false }

        return true
    }

    // MARK: - Multilinear Extension Evaluation

    /// Evaluate the multilinear extension via even/odd folding (ZMFold convention).
    ///
    /// Uses the same convention as Zeromorph: iteratively fold from variable n-1 down to 0.
    /// At each step k (going n-1, n-2, ..., 0):
    ///   folded[i] = current[2*i] + point[k] * current[2*i + 1]
    ///
    /// This matches the Zeromorph evaluateZMFold convention where bit j of the evaluation
    /// index corresponds to variable point[n-1-j].
    public static func evaluateMLE(_ evaluations: [Fr], at point: [Fr]) -> Fr {
        let n = point.count
        precondition(evaluations.count == (1 << n))

        var current = evaluations
        for k in stride(from: n - 1, through: 0, by: -1) {
            let halfLen = current.count / 2
            let uk = point[k]
            var folded = [Fr](repeating: Fr.zero, count: halfLen)
            current.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: uk) { uBuf in
            folded.withUnsafeMutableBytes { fBuf in
                bn254_fr_fold_zm_interleaved(
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    uBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfLen))
            }}}
            current = folded
        }
        return current[0]
    }

    // MARK: - Helpers

    /// Fr negation
    private func frNeg(_ a: Fr) -> Fr {
        frSub(Fr.zero, a)
    }

    /// Evaluate polynomial using C Horner evaluation.
    private func cEvaluate(_ coeffs: [Fr], at z: Fr) -> Fr {
        if coeffs.isEmpty { return Fr.zero }
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(coeffs.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// Absorb a projective point into the transcript.
    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            let xLimbs = fpToInt(aff[0].x)
            let yLimbs = fpToInt(aff[0].y)
            transcript.absorb(Fr.from64(xLimbs))
            transcript.absorb(Fr.from64(yLimbs))
        }
    }
}
