// Zeromorph Polynomial Commitment Engine
// Multilinear PCS built from univariate KZG.
// Transforms multilinear evaluation claim f̃(u₁,...,uₙ) = v into a univariate
// polynomial identity, then proves it via KZG quotient commitments.
//
// Reference: "Zeromorph: Zero-Knowledge Multilinear-Evaluation Proofs from
// Homomorphic Univariate Commitments" (Kohrita & Towa, 2023)

import Foundation

// MARK: - Data Structures

public struct ZeromorphProof {
    /// Commitments to quotient polynomials [q₁],...,[qₙ]
    public let quotientCommitments: [PointProjective]
    /// Evaluations qₖ(ζ) at verifier challenge
    public let evaluations: [Fr]
    /// Batched KZG opening proof at ζ
    public let batchOpeningProof: PointProjective
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

    /// Prove that the multilinear polynomial with given hypercube evaluations
    /// evaluates to a specific value at `point = (u₁,...,uₙ)`.
    ///
    /// Algorithm:
    /// 1. Compute quotient polynomials qₖ(X) via iterative even/odd decomposition
    /// 2. Commit to each qₖ
    /// 3. Verifier sends challenge ζ (Fiat-Shamir)
    /// 4. Evaluate f(ζ) and each qₖ(ζ)
    /// 5. Batch KZG opening proof at ζ
    public func open(evaluations: [Fr], point: [Fr]) throws -> ZeromorphProof {
        let N = evaluations.count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree")

        // Step 1: Compute quotient polynomials via even/odd decomposition.
        // We work from variable n (last) down to variable 1 (first).
        // At each level k, given polynomial f of degree < 2^k:
        //   f_even(X) = f[0], f[2], f[4], ...    (coeffs at even indices)
        //   f_odd(X)  = f[1], f[3], f[5], ...    (coeffs at odd indices)
        //   qₖ(X) = f_odd(X)
        //   f ← f_even(X) + uₖ · f_odd(X)       (fold down)
        var quotients = [[Fr]]()
        quotients.reserveCapacity(n)

        var f = evaluations
        for k in stride(from: n - 1, through: 0, by: -1) {
            let halfLen = f.count / 2
            var fEven = [Fr](repeating: Fr.zero, count: halfLen)
            var fOdd = [Fr](repeating: Fr.zero, count: halfLen)

            for i in 0..<halfLen {
                fEven[i] = f[2 * i]
                fOdd[i] = f[2 * i + 1]
            }

            // qₖ = f_odd
            quotients.append(fOdd)

            // Fold: f ← f_even + u_k · f_odd
            let uk = point[k]
            var folded = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                folded[i] = frAdd(fEven[i], frMul(uk, fOdd[i]))
            }
            f = folded
        }

        // After n rounds, f should be a single element = the MLE evaluation
        precondition(f.count == 1, "folding should reduce to single element")
        // quotients are in reverse order (q_{n-1}, q_{n-2}, ..., q_0)
        // Reverse to get (q_0, q_1, ..., q_{n-1}) matching point indices
        quotients.reverse()

        // Step 2: Commit to each quotient polynomial
        var quotientCommitments = [PointProjective]()
        quotientCommitments.reserveCapacity(n)
        for q in quotients {
            let c = try kzg.commit(q)
            quotientCommitments.append(c)
        }

        // Step 3: Fiat-Shamir challenge ζ (deterministic from commitments)
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments)

        // Step 4: Evaluate f(ζ) and each qₖ(ζ)
        var qEvals = [Fr]()
        qEvals.reserveCapacity(n)
        for q in quotients {
            qEvals.append(evaluateUnivariate(q, at: zeta))
        }

        // Step 5: Batch KZG opening — open f and all qₖ at ζ
        // Combine into single opening: h(X) = f(X) + γ·q₀(X) + γ²·q₁(X) + ...
        // But we can use the KZG batch open directly
        var allPolynomials = [[Fr]]()
        allPolynomials.append(evaluations)  // f(X)
        allPolynomials.append(contentsOf: quotients)

        let gamma = deriveGamma(zeta: zeta, qEvals: qEvals)
        let batchProof = try kzg.batchOpen(polynomials: allPolynomials, point: zeta, gamma: gamma)

        return ZeromorphProof(
            quotientCommitments: quotientCommitments,
            evaluations: qEvals,
            batchOpeningProof: batchProof.proof
        )
    }

    // MARK: - Verify

    /// Verify a Zeromorph proof that f̃(u₁,...,uₙ) = value.
    ///
    /// Checks the polynomial identity:
    ///   f(ζ) - v = Σₖ (ζ^(2^k) - uₖ) · qₖ(ζ)
    ///
    /// Then verifies batch KZG opening of f and all qₖ at ζ.
    public func verify(commitment: PointProjective, point: [Fr], value: Fr,
                       proof: ZeromorphProof, srsSecret: Fr) -> Bool {
        let n = point.count
        guard proof.quotientCommitments.count == n,
              proof.evaluations.count == n else {
            return false
        }

        // Recompute Fiat-Shamir challenges
        // Note: in a full implementation, the verifier would derive ζ from the transcript
        // containing the commitment and quotient commitments. Here we use the same derivation.

        // Step 1: Check the polynomial identity at ζ
        // f(ζ) - v = Σₖ (ζ^(2^k) - uₖ) · qₖ(ζ)
        //
        // We need f(ζ) from the batch opening evaluations.
        // For verification without pairings, we use the SRS secret.

        // Derive ζ (same as prover)
        let zeta = deriveZeta(point: point,
                              quotientCommitments: proof.quotientCommitments)

        // Compute ζ^(2^k) for each k
        var zetaPow = zeta  // ζ^(2^0) = ζ
        var rhs = Fr.zero
        for k in 0..<n {
            // (ζ^(2^k) - uₖ) · qₖ(ζ)
            let bracket = frSub(zetaPow, point[k])
            let term = frMul(bracket, proof.evaluations[k])
            rhs = frAdd(rhs, term)
            zetaPow = frMul(zetaPow, zetaPow)  // square for next level
        }

        // Evaluate f(ζ) via commitment verification:
        // We compute f(ζ) = v + rhs, then verify the batch opening is consistent
        let fZeta = frAdd(value, rhs)

        // Step 2: Verify batch KZG opening
        // Reconstruct all commitments and evaluations for the batch check
        var allCommitments = [PointProjective]()
        allCommitments.append(commitment)  // C_f
        allCommitments.append(contentsOf: proof.quotientCommitments)

        var allEvaluations = [Fr]()
        allEvaluations.append(fZeta)  // f(ζ)
        allEvaluations.append(contentsOf: proof.evaluations)  // qₖ(ζ)

        let gamma = deriveGamma(zeta: zeta, qEvals: proof.evaluations)

        return kzg.batchVerify(
            commitments: allCommitments, point: zeta,
            evaluations: allEvaluations, proof: proof.batchOpeningProof,
            gamma: gamma, srsSecret: srsSecret)
    }

    // MARK: - Multilinear evaluation (CPU reference)

    /// Evaluate multilinear polynomial at point using folding.
    /// f̃(u₁,...,uₙ) via iterative halving.
    public static func evaluateMLE(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            for i in 0..<half {
                // f(u) = f_even(i) + u_k * f_odd(i)
                // = f[2i] + u_k * f[2i+1]
                // Equivalently: (1-u_k)*f[2i] + u_k*f[2i+1]
                let lo = current[2 * i]
                let hi = current[2 * i + 1]
                folded[i] = frAdd(lo, frMul(uk, frSub(hi, lo)))
            }
            current = folded
        }
        return current[0]
    }

    // MARK: - Private helpers

    /// Evaluate univariate polynomial (given as coefficient vector) at a point.
    private func evaluateUnivariate(_ coeffs: [Fr], at x: Fr) -> Fr {
        // Horner's method: f(x) = c_0 + x*(c_1 + x*(c_2 + ...))
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Deterministic Fiat-Shamir challenge from point and quotient commitments.
    /// Both prover and verifier have access to these.
    /// In practice this would use a proper transcript (Merlin/sponge).
    private func deriveZeta(point: [Fr], quotientCommitments: [PointProjective]) -> Fr {
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
        return frFromInt(seed | 1)  // ensure nonzero
    }

    /// Derive gamma for batch opening from zeta and quotient evaluations.
    private func deriveGamma(zeta: Fr, qEvals: [Fr]) -> Fr {
        var seed: UInt64 = 0x47414D4D41_5A4D  // "GAMMA_ZM"
        let zLimbs = frToInt(zeta)
        seed = seed &* 6364136223846793005 &+ zLimbs[0]
        for e in qEvals {
            let limbs = frToInt(e)
            seed = seed &* 6364136223846793005 &+ limbs[0]
        }
        return frFromInt(seed | 1)
    }
}
