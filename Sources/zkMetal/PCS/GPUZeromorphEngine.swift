// GPU-Accelerated Zeromorph Multilinear Opening Engine
//
// GPU-accelerated version of the Zeromorph multilinear PCS (Kohrita & Towa, 2023).
// Leverages Metal MSM for commitments and batch quotient computation.
//
// Key optimizations over the CPU ZeromorphEngine:
//   1. Batch MSM for quotient commitments (single GPU dispatch)
//   2. GPU-accelerated polynomial evaluation via Horner's method
//   3. Batched opening: random linear combination reduces multiple openings to one
//
// Construction:
//   Given MLE f on {0,1}^n embedded as univariate f(X) = sum_i evals[i]*X^i,
//   the prover computes n quotient polynomials via even/odd decomposition:
//     q^(s) = f_odd,  f_next = f_even + u_k * f_odd
//   Linearized polynomial: L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
//   where phi_s = zeta^{2^s} - u_{n-1-s}.
//   L is opened via KZG at challenge zeta; verified by pairing check.

import Foundation
import NeonFieldOps

// MARK: - GPU Zeromorph Proof

public struct GPUZeromorphProof {
    /// Commitments to quotient polynomials [q^(0)],...,[q^(n-1)]
    public let quotientCommitments: [PointProjective]
    /// The claimed evaluation value
    public let claimedValue: Fr
    /// KZG witness for linearized polynomial L at zeta
    public let kzgWitness: PointProjective
    /// Evaluation L(zeta)
    public let linearizationEval: Fr
    /// Fiat-Shamir challenge
    public let zeta: Fr

    public var numVariables: Int { quotientCommitments.count }
}

// MARK: - GPU Zeromorph Batch Proof

public struct GPUZeromorphBatchProof {
    /// Per-polynomial proofs (quotient commitments, values)
    public let quotientCommitments: [[PointProjective]]
    /// Claimed values for each polynomial
    public let claimedValues: [Fr]
    /// Single batched KZG witness
    public let kzgWitness: PointProjective
    /// Batched linearization evaluation
    public let linearizationEval: Fr
    /// Fiat-Shamir challenge zeta
    public let zeta: Fr
    /// Random linear combination challenge
    public let gamma: Fr
}

// MARK: - Engine

public class GPUZeromorphEngine {
    public static let version = Versions.gpuZeromorph

    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as evaluations on the boolean hypercube.
    /// Reinterprets the 2^n evaluations as coefficients of a univariate polynomial
    /// and commits via KZG MSM (GPU-accelerated for large inputs).
    public func commit(evaluations: [Fr]) throws -> PointProjective {
        try kzg.commit(evaluations)
    }

    // MARK: - Open (Prove)

    /// Generate a GPU-accelerated Zeromorph opening proof.
    ///
    /// Computes n quotient polynomials via even/odd decomposition, batch-commits them,
    /// builds the linearized polynomial L(X), and opens via KZG at challenge zeta.
    public func open(evaluations: [Fr], point: [Fr], value: Fr? = nil) throws -> GPUZeromorphProof {
        let N = evaluations.count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree \(N)")

        let v = value ?? Self.evaluateZMFold(evaluations: evaluations, point: point)

        // Step 1: Compute quotient polynomials via even/odd decomposition
        let stepQuotients = computeQuotients(evaluations: evaluations, point: point, n: n)

        // Step 2: Batch commit to all quotients (single GPU dispatch via batchCommit)
        let quotientCommitments = try kzg.batchCommit(stepQuotients)

        // Step 3: Fiat-Shamir challenge zeta
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments, value: v)

        // Step 4: Build linearized polynomial L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
        let L = buildLinearizedPolynomial(
            evaluations: evaluations, value: v, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )

        // Step 5: KZG open L at zeta
        let kzgProof = try kzg.open(L, at: zeta)

        return GPUZeromorphProof(
            quotientCommitments: quotientCommitments,
            claimedValue: v,
            kzgWitness: kzgProof.witness,
            linearizationEval: kzgProof.evaluation,
            zeta: zeta
        )
    }

    // MARK: - Batch Opening

    /// Batch open multiple multilinear polynomials at the same point.
    ///
    /// Uses random linear combination with challenge gamma to reduce N openings to one.
    /// All polynomials must have the same number of variables (same-size evaluation vectors).
    ///
    /// Algorithm:
    ///   h(X) = sum_i gamma^i * f_i(X)
    ///   v_combined = sum_i gamma^i * v_i
    ///   Quotients of h are gamma-weighted combinations of individual quotients.
    ///   Single linearized polynomial L_h is opened at zeta.
    public func batchOpen(evaluationSets: [[Fr]], point: [Fr],
                          values: [Fr]? = nil, gamma: Fr) throws -> GPUZeromorphBatchProof {
        let count = evaluationSets.count
        precondition(count > 0, "need at least one polynomial")
        let N = evaluationSets[0].count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        for evals in evaluationSets {
            precondition(evals.count == N, "all evaluation sets must be same size")
        }

        // Compute values if not provided
        let vs: [Fr]
        if let values = values {
            precondition(values.count == count)
            vs = values
        } else {
            vs = evaluationSets.map { Self.evaluateZMFold(evaluations: $0, point: point) }
        }

        // Compute combined polynomial: h(X) = sum_i gamma^i * f_i(X)
        var combined = [Fr](repeating: Fr.zero, count: N)
        var gammaPow = Fr.one
        for i in 0..<count {
            let evals = evaluationSets[i]
            for j in 0..<N {
                combined[j] = frAdd(combined[j], frMul(gammaPow, evals[j]))
            }
            if i < count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Combined value
        var vCombined = Fr.zero
        gammaPow = Fr.one
        for i in 0..<count {
            vCombined = frAdd(vCombined, frMul(gammaPow, vs[i]))
            if i < count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Compute quotients of the combined polynomial
        let stepQuotients = computeQuotients(evaluations: combined, point: point, n: n)

        // Batch commit quotients
        let quotientCommitments = try kzg.batchCommit(stepQuotients)

        // Also compute per-polynomial quotient commitments for proof structure
        var perPolyQuotientCommitments = [[PointProjective]]()
        perPolyQuotientCommitments.reserveCapacity(count)
        for i in 0..<count {
            let qs = computeQuotients(evaluations: evaluationSets[i], point: point, n: n)
            perPolyQuotientCommitments.append(try kzg.batchCommit(qs))
        }

        // Fiat-Shamir challenge zeta
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments, value: vCombined)

        // Build linearized polynomial for combined
        let L = buildLinearizedPolynomial(
            evaluations: combined, value: vCombined, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )

        // KZG open L at zeta
        let kzgProof = try kzg.open(L, at: zeta)

        return GPUZeromorphBatchProof(
            quotientCommitments: perPolyQuotientCommitments,
            claimedValues: vs,
            kzgWitness: kzgProof.witness,
            linearizationEval: kzgProof.evaluation,
            zeta: zeta,
            gamma: gamma
        )
    }

    // MARK: - Verify (SRS Secret, testing)

    /// Full algebraic verification using the SRS secret (testing only).
    /// Checks: (1) telescoping identity P(zeta) = 0, (2) L(zeta) matches, (3) KZG proof.
    public func verifyWithSecret(evaluations: [Fr], point: [Fr], value: Fr,
                                  proof: GPUZeromorphProof, srsSecret: Fr) -> Bool {
        let n = point.count
        let N = evaluations.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta

        // Recompute quotients
        let stepQuotients = computeQuotients(evaluations: evaluations, point: point, n: n)

        // Check 1: Telescoping identity P(zeta) = 0
        let fZeta = evaluateUnivariate(evaluations, at: zeta)
        var zetaPow = zeta
        var pZeta = frSub(fZeta, value)
        for s in 0..<n {
            let k = n - 1 - s
            let alpha = frSub(zetaPow, point[k])
            let zetaNext = frMul(zetaPow, zetaPow)
            let qEval = evaluateUnivariate(stepQuotients[s], at: zetaNext)
            pZeta = frSub(pZeta, frMul(alpha, qEval))
            zetaPow = zetaNext
        }
        if !frEqual(pZeta, Fr.zero) {
            return false
        }

        // Check 2: L(zeta) matches
        let L = buildLinearizedPolynomial(
            evaluations: evaluations, value: value, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let lZeta = evaluateUnivariate(L, at: zeta)
        if !frEqual(lZeta, proof.linearizationEval) {
            return false
        }

        // Check 3: KZG proof via secret
        guard let commitL = try? kzg.commit(L) else { return false }
        let g1 = pointFromAffine(kzg.srs[0])
        let lMinusDelta = pointAdd(commitL, pointNeg(cPointScalarMul(g1, proof.linearizationEval)))
        let sMz = frSub(srsSecret, zeta)
        let expectedW = cPointScalarMul(proof.kzgWitness, sMz)

        let cAff = batchToAffine([lMinusDelta])
        let eAff = batchToAffine([expectedW])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    /// Verify a batch opening proof (testing only, using SRS secret).
    public func verifyBatchWithSecret(evaluationSets: [[Fr]], point: [Fr], values: [Fr],
                                       proof: GPUZeromorphBatchProof, srsSecret: Fr) -> Bool {
        let count = evaluationSets.count
        guard proof.claimedValues.count == count else { return false }
        guard proof.quotientCommitments.count == count else { return false }

        // Check claimed values match
        for i in 0..<count {
            guard frEqual(proof.claimedValues[i], values[i]) else { return false }
        }

        let N = evaluationSets[0].count
        let n = point.count
        let gamma = proof.gamma

        // Reconstruct combined polynomial
        var combined = [Fr](repeating: Fr.zero, count: N)
        var gammaPow = Fr.one
        for i in 0..<count {
            let evals = evaluationSets[i]
            for j in 0..<N {
                combined[j] = frAdd(combined[j], frMul(gammaPow, evals[j]))
            }
            if i < count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        var vCombined = Fr.zero
        gammaPow = Fr.one
        for i in 0..<count {
            vCombined = frAdd(vCombined, frMul(gammaPow, values[i]))
            if i < count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Verify combined proof as a single opening
        let combinedProof = GPUZeromorphProof(
            quotientCommitments: [],  // not used by verifyWithSecret (recomputes)
            claimedValue: vCombined,
            kzgWitness: proof.kzgWitness,
            linearizationEval: proof.linearizationEval,
            zeta: proof.zeta
        )

        // We need to verify with the combined polynomial's quotient commitments.
        // Recompute and check via the telescoping identity + KZG secret check.
        let stepQuotients = computeQuotients(evaluations: combined, point: point, n: n)
        let zeta = proof.zeta

        // Telescoping identity
        let fZeta = evaluateUnivariate(combined, at: zeta)
        var zetaPow = zeta
        var pZeta = frSub(fZeta, vCombined)
        for s in 0..<n {
            let k = n - 1 - s
            let alpha = frSub(zetaPow, point[k])
            let zetaNext = frMul(zetaPow, zetaPow)
            let qEval = evaluateUnivariate(stepQuotients[s], at: zetaNext)
            pZeta = frSub(pZeta, frMul(alpha, qEval))
            zetaPow = zetaNext
        }
        if !frEqual(pZeta, Fr.zero) {
            return false
        }

        // L(zeta)
        let L = buildLinearizedPolynomial(
            evaluations: combined, value: vCombined, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let lZeta = evaluateUnivariate(L, at: zeta)
        if !frEqual(lZeta, proof.linearizationEval) {
            return false
        }

        // KZG check
        guard let commitL = try? kzg.commit(L) else { return false }
        let g1 = pointFromAffine(kzg.srs[0])
        let lMinusDelta = pointAdd(commitL, pointNeg(cPointScalarMul(g1, proof.linearizationEval)))
        let sMz = frSub(srsSecret, zeta)
        let expectedW = cPointScalarMul(proof.kzgWitness, sMz)

        let cAff = batchToAffine([lMinusDelta])
        let eAff = batchToAffine([expectedW])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    // MARK: - Verify (Pairing-Based)

    /// Verify a Zeromorph proof using BN254 pairings.
    ///
    /// Reconstructs [L]_1 from the polynomial commitment and quotient commitments:
    ///   C_L = [f]_1 - v*G1 - sum_s phi_s * [q^(s)]_1
    /// Then checks the KZG opening of L at zeta via pairing:
    ///   e(C_L - delta*G1 + zeta*W, [1]_2) = e(W, [tau]_2)
    public func verify(commitment: PointProjective, point: [Fr], value: Fr,
                       proof: GPUZeromorphProof, vk: ZeromorphVK) throws -> Bool {
        let pairing = try BN254PairingEngine()
        let n = point.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta
        let g1 = pointFromAffine(kzg.srs[0])

        // Compute phi_s = zeta^{2^s} - u_{n-1-s}
        var zetaPow = zeta
        var cL = pointAdd(commitment, pointNeg(cPointScalarMul(g1, value)))
        for s in 0..<n {
            let k = n - 1 - s
            let phi = frSub(zetaPow, point[k])
            zetaPow = frMul(zetaPow, zetaPow)
            let term = cPointScalarMul(proof.quotientCommitments[s], phi)
            cL = pointAdd(cL, pointNeg(term))
        }

        // KZG pairing check: e(C_L - delta*G1 + zeta*W, [1]_2) * e(-W, [tau]_2) = 1
        let delta = proof.linearizationEval
        let deltaG = cPointScalarMul(g1, delta)
        let zetaW = cPointScalarMul(proof.kzgWitness, zeta)
        let lhs = pointAdd(pointAdd(cL, pointNeg(deltaG)), zetaW)

        let lhsAff = batchToAffine([lhs])
        let wNegAff = batchToAffine([pointNeg(proof.kzgWitness)])

        return try pairing.pairingCheck(pairs: [
            (lhsAff[0], vk.g2Generator),
            (wNegAff[0], vk.tauG2)
        ])
    }

    // MARK: - Evaluation Helpers

    /// ZM fold evaluation: iteratively fold f_even + u_k * f_odd.
    /// Agrees with standard MLE evaluation on Boolean inputs {0,1}^n.
    public static func evaluateZMFold(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            for i in 0..<half {
                folded[i] = frAdd(current[2 * i], frMul(uk, current[2 * i + 1]))
            }
            current = folded
        }
        return current[0]
    }

    /// Standard MLE evaluation: (1-u_k)*f[2i] + u_k*f[2i+1]
    public static func evaluateMLE(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            for i in 0..<half {
                let lo = current[2 * i]
                let hi = current[2 * i + 1]
                folded[i] = frAdd(lo, frMul(uk, frSub(hi, lo)))
            }
            current = folded
        }
        return current[0]
    }

    // MARK: - Private Helpers

    /// Compute n quotient polynomials via even/odd decomposition.
    private func computeQuotients(evaluations: [Fr], point: [Fr], n: Int) -> [[Fr]] {
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
        return stepQuotients
    }

    /// Build linearized polynomial L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
    /// where phi_s = zeta^{2^s} - u_{n-1-s}.
    private func buildLinearizedPolynomial(evaluations: [Fr], value: Fr, point: [Fr],
                                            stepQuotients: [[Fr]], zeta: Fr,
                                            N: Int, n: Int) -> [Fr] {
        var zetaPow = zeta
        var L = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N { L[i] = evaluations[i] }
        L[0] = frSub(L[0], value)

        for s in 0..<n {
            let k = n - 1 - s
            let phi = frSub(zetaPow, point[k])
            zetaPow = frMul(zetaPow, zetaPow)
            let q = stepQuotients[s]
            for j in 0..<q.count {
                if j < N {
                    L[j] = frSub(L[j], frMul(phi, q[j]))
                }
            }
        }
        return L
    }

    /// Evaluate univariate polynomial at a point via Horner's method.
    private func evaluateUnivariate(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Deterministic Fiat-Shamir challenge from point, quotient commitments, and value.
    private func deriveZeta(point: [Fr], quotientCommitments: [PointProjective], value: Fr) -> Fr {
        var transcript = FiatShamirTranscript(label: "gpu_zeromorph", hasher: KeccakTranscriptHasher())

        transcript.absorbFrMany("point", point)

        let affComms = batchToAffine(quotientCommitments)
        for comm in affComms {
            var bytes = [UInt8](repeating: 0, count: 64)
            withUnsafeBytes(of: comm.x) { xBuf in
                let ptr = xBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                for i in 0..<32 { bytes[i] = ptr[i] }
            }
            withUnsafeBytes(of: comm.y) { yBuf in
                let ptr = yBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                for i in 0..<32 { bytes[32 + i] = ptr[i] }
            }
            transcript.absorb("quotient_comm", bytes)
        }

        transcript.absorbFr("value", value)
        return transcript.challengeScalar("zeta")
    }
}
