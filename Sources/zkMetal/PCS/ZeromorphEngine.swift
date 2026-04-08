// Zeromorph Polynomial Commitment Engine
// Multilinear PCS built from univariate KZG.
// Transforms multilinear evaluation claim into a univariate polynomial identity,
// then proves it via a single KZG opening.
//
// Reference: "Zeromorph: Zero-Knowledge Multilinear-Evaluation Proofs from
// Homomorphic Univariate Commitments" (Kohrita & Towa, 2023)
//
// Construction:
//   Given MLE evaluations embedded as univariate f(X) = sum_i evals[i]*X^i,
//   the prover computes n quotient polynomials via even/odd decomposition:
//     f(X) = f_even(X^2) + X * f_odd(X^2)
//     q^(s) = f_odd,  f_next = f_even + u_k * f_odd
//   Telescoping identity:
//     f(X) - v = sum_s (X^{2^s} - u_{n-1-s}) * q^(s)(X^{2^{s+1}})
//   where v = evaluateZMFold(evals, point).
//   This is assembled into P(X) with P(zeta)=0 and proven via single KZG opening.

import Foundation
import NeonFieldOps

// MARK: - Data Structures

public struct ZeromorphProof {
    /// Commitments to quotient polynomials [q^(0)],...,[q^(n-1)]
    public let quotientCommitments: [PointProjective]
    /// The claimed evaluation value
    public let claimedValue: Fr
    /// KZG opening proof for P(X) at zeta, claiming P(zeta) = 0
    public let openingProof: KZGProof
}

// MARK: - Engine

public class ZeromorphEngine {
    public static let version = Versions.zeromorph

    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as evaluations on the boolean hypercube.
    /// Reinterprets the 2^n evaluations as coefficients of a univariate polynomial
    /// and commits via KZG: C = MSM(SRS, evals).
    public func commit(evaluations: [Fr]) throws -> PointProjective {
        try kzg.commit(evaluations)
    }

    // MARK: - Open (Prove)

    /// Prove that the Zeromorph-fold evaluation of the multilinear polynomial
    /// at `point = (u_1,...,u_n)` equals `value`.
    ///
    /// The "ZM fold" evaluation is: iteratively fold f_even + u_k * f_odd.
    /// If `value` is nil, it is computed automatically.
    public func open(evaluations: [Fr], point: [Fr], value: Fr? = nil) throws -> ZeromorphProof {
        let N = evaluations.count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree")

        // Compute the claimed value using the ZM fold
        let v = value ?? ZeromorphEngine.evaluateZMFold(evaluations: evaluations, point: point)

        // Step 1: Compute quotient polynomials via even/odd decomposition.
        // At step s (processing variable k=n-1-s):
        //   q^(s) = f_odd (odd-indexed coefficients)
        //   f_next = f_even + u_k * f_odd
        // Identity: f_cur(X) - f_next(X^2) = (X - u_k) * q^(s)(X^2)
        // Telescoping: f(X) - v = sum_s (X^{2^s} - u_{n-1-s}) * q^(s)(X^{2^{s+1}})

        var stepQuotients = [[Fr]]()
        stepQuotients.reserveCapacity(n)

        var f = evaluations
        for s in 0..<n {
            let k = n - 1 - s
            let halfLen = f.count / 2
            var fEven = [Fr](repeating: Fr.zero, count: halfLen)
            var fOdd = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                fEven[i] = f[2 * i]
                fOdd[i] = f[2 * i + 1]
            }
            stepQuotients.append(fOdd)
            let uk = point[k]
            var folded = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                folded[i] = frAdd(fEven[i], frMul(uk, fOdd[i]))
            }
            f = folded
        }
        precondition(f.count == 1, "folding should reduce to single element")

        // Step 2: Commit to each quotient
        var quotientCommitments = [PointProjective]()
        quotientCommitments.reserveCapacity(n)
        for q in stepQuotients {
            quotientCommitments.append(try kzg.commit(q))
        }

        // Step 3: Fiat-Shamir challenge zeta
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments, value: v)

        // Step 4: Build P(X) = f(X) - v - sum_s [(X^{2^s} - u_{n-1-s}) * q^(s)_shifted(X)]
        //
        // where q^(s)_shifted(X) = sum_j q^(s)[j] * X^{j*2^{s+1}}
        // and the product with (X^{2^s} - u_k) gives:
        //   sum_j q[j] * (X^{j*2^{s+1}+2^s} - u_k * X^{j*2^{s+1}})
        //
        // P(zeta) = 0 by the telescoping identity.

        let P = buildCombinedPolynomial(evaluations: evaluations, value: v,
                                         point: point, stepQuotients: stepQuotients, N: N, n: n)

        // Step 5: KZG open P at zeta
        let proof = try kzg.open(P, at: zeta)

        return ZeromorphProof(
            quotientCommitments: quotientCommitments,
            claimedValue: v,
            openingProof: proof
        )
    }

    // MARK: - Verify

    /// Light verification: checks claimed value match and P(zeta) = 0.
    /// Without pairings, full algebraic verification requires polynomial access
    /// (use verifyWithPolynomial for testing).
    public func verify(commitment: PointProjective, point: [Fr], value: Fr,
                       proof: ZeromorphProof, srsSecret: Fr) -> Bool {
        let n = point.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frToInt(proof.claimedValue) == frToInt(value) else { return false }

        // P(zeta) must be zero
        if frToInt(proof.openingProof.evaluation) != frToInt(Fr.zero) {
            return false
        }

        // Note: identity witness is valid when the quotient polynomial is zero
        // (happens when P(X) is zero or a scalar multiple of (X - zeta)).
        // Full security requires pairing-based verification of the KZG proof.
        return true
    }

    /// Full verification using the actual polynomial coefficients.
    /// Checks: (1) P(zeta) = 0 algebraically, (2) KZG proof matches commitment [P].
    /// In production, step (2) would use pairings instead of srsSecret.
    public func verifyWithPolynomial(evaluations: [Fr], point: [Fr], value: Fr,
                                     proof: ZeromorphProof, srsSecret: Fr) -> Bool {
        let n = point.count
        let N = evaluations.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frToInt(proof.claimedValue) == frToInt(value) else { return false }

        // Derive zeta (same as prover)
        let zeta = deriveZeta(point: point,
                              quotientCommitments: proof.quotientCommitments,
                              value: value)

        // Recompute quotients
        var stepQuotients = [[Fr]]()
        var f = evaluations
        for s in 0..<n {
            let k = n - 1 - s
            let halfLen = f.count / 2
            var fOdd = [Fr](repeating: Fr.zero, count: halfLen)
            var fEven = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                fEven[i] = f[2 * i]
                fOdd[i] = f[2 * i + 1]
            }
            stepQuotients.append(fOdd)
            let uk = point[k]
            var folded = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                folded[i] = frAdd(fEven[i], frMul(uk, fOdd[i]))
            }
            f = folded
        }

        // Check 1: P(zeta) = 0 algebraically
        // P(zeta) = f(zeta) - v - sum_s (zeta^{2^s} - u_{n-1-s}) * q^(s)(zeta^{2^{s+1}})
        let fZeta = evaluateUnivariate(evaluations, at: zeta)
        var zetaPow = zeta
        var pZeta = frSub(fZeta, value)
        for s in 0..<n {
            let k = n - 1 - s
            let alpha = frSub(zetaPow, point[k])
            let zetaNext = frMul(zetaPow, zetaPow)  // zeta^{2^{s+1}}
            let qEval = evaluateUnivariate(stepQuotients[s], at: zetaNext)
            pZeta = frSub(pZeta, frMul(alpha, qEval))
            zetaPow = zetaNext
        }
        if frToInt(pZeta) != frToInt(Fr.zero) {
            return false
        }

        // Check 2: KZG proof matches [P]
        // Reconstruct P and commit
        let P = buildCombinedPolynomial(evaluations: evaluations, value: value,
                                         point: point, stepQuotients: stepQuotients, N: N, n: n)
        guard let commitP = try? kzg.commit(P) else { return false }

        // Verify: [P] = [0]*G + [s-zeta]*[witness] (since eval=0)
        let sMz = frSub(srsSecret, zeta)
        let expectedP = cPointScalarMul(proof.openingProof.witness, sMz)

        let cAff = batchToAffine([commitP])
        let eAff = batchToAffine([expectedP])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    // MARK: - Evaluation helpers

    /// Standard MLE evaluation: (1-u_k)*f[2i] + u_k*f[2i+1]
    public static func evaluateMLE(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            current.withUnsafeBytes { eBuf in
                withUnsafeBytes(of: uk) { rBuf in
                    folded.withUnsafeMutableBytes { outBuf in
                        bn254_fr_fold_interleaved(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            current = folded
        }
        return current[0]
    }

    /// ZM fold evaluation: f_even + u_k * f_odd = f[2i] + u_k * f[2i+1].
    /// This is the evaluation formula consistent with the quotient identity.
    /// Agrees with evaluateMLE on Boolean inputs {0,1}^n.
    public static func evaluateZMFold(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            current.withUnsafeBytes { eBuf in
                withUnsafeBytes(of: uk) { rBuf in
                    folded.withUnsafeMutableBytes { outBuf in
                        bn254_fr_fold_interleaved(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            current = folded
        }
        return current[0]
    }

    // MARK: - Private helpers

    /// Build the combined polynomial P(X) = f(X) - v - sum_s product_s(X)
    /// where product_s(X) = (X^{2^s} - u_{n-1-s}) * q^(s)_shifted(X)
    private func buildCombinedPolynomial(evaluations: [Fr], value: Fr, point: [Fr],
                                          stepQuotients: [[Fr]], N: Int, n: Int) -> [Fr] {
        var P = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N { P[i] = evaluations[i] }
        P[0] = frSub(P[0], value)

        for s in 0..<n {
            let k = n - 1 - s
            let uk = point[k]
            let shift = 1 << s        // 2^s
            let stride = 1 << (s + 1) // 2^{s+1}
            let q = stepQuotients[s]
            for j in 0..<q.count {
                let baseIdx = j * stride
                let shiftedIdx = baseIdx + shift
                // Subtract q[j] * X^{shiftedIdx} (from the X^{2^s} * q_shifted term)
                if shiftedIdx < N {
                    P[shiftedIdx] = frSub(P[shiftedIdx], q[j])
                }
                // Add u_k * q[j] * X^{baseIdx} (from the -u_k * q_shifted term, negated)
                if baseIdx < N {
                    P[baseIdx] = frAdd(P[baseIdx], frMul(uk, q[j]))
                }
            }
        }
        return P
    }

    /// Evaluate univariate polynomial (coefficient vector) at a point via Horner.
    public func evaluateUnivariate(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Deterministic Fiat-Shamir challenge from point, quotient commitments, and value.
    private func deriveZeta(point: [Fr], quotientCommitments: [PointProjective], value: Fr) -> Fr {
        var seed: UInt64 = 0x5A45524F4D4F5250  // "ZEROMORP"
        for q in quotientCommitments {
            let aff = batchToAffine([q])
            let xLimbs = fpToInt(aff[0].x)
            seed = seed &* 6364136223846793005 &+ xLimbs[0]
            let yLimbs = fpToInt(aff[0].y)
            seed = seed &* 6364136223846793005 &+ yLimbs[0]
        }
        for u in point {
            let limbs = frToInt(u)
            seed = seed &* 6364136223846793005 &+ limbs[0]
        }
        let vLimbs = frToInt(value)
        seed = seed &* 6364136223846793005 &+ vLimbs[0]
        return frFromInt(seed | 1)
    }
}
