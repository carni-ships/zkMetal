// Zeromorph Multilinear Polynomial Commitment Scheme
//
// A multilinear PCS built from univariate KZG (Kohrita & Towa, eprint 2023/917).
// Bridges univariate KZG (MSM + SRS) to multilinear protocols (sumcheck-based).
//
// Given MLE f(x_1,...,x_n) on {0,1}^n, embed as univariate f(X) = sum_i evals[i]*X^i.
// To prove f(u) = v, compute n quotient polynomials via even/odd decomposition.
// Telescoping identity: f(X) - v = sum_s (X^{2^s} - u_{n-1-s}) * q^(s)(X^{2^{s+1}})
//
// The prover opens the linearized polynomial L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
// where phi_s = zeta^{2^s} - u_{n-1-s}. The verifier reconstructs [L]_1 from
// commitments and checks the KZG opening via a single pairing equation.
//
// Commit: 1 MSM of size 2^n
// Open:   n MSMs (quotients) + 1 MSM (KZG witness for L)
// Verify: O(n) G1 scalar muls + 1 pairing check (2 pairings)

import Foundation
import NeonFieldOps

// MARK: - Zeromorph PCS Engine

public class ZeromorphPCS {
    public static let version = Versions.zeromorph

    public let kzg: KZGEngine
    public let pairing: BN254PairingEngine

    public init(kzg: KZGEngine) throws {
        self.kzg = kzg
        self.pairing = try BN254PairingEngine()
    }

    // MARK: - Commit

    /// Commit to a multilinear polynomial (evaluations on {0,1}^n).
    /// Reinterprets 2^n evaluations as univariate coefficients, commits via KZG MSM.
    public func commit(evaluations: [Fr]) throws -> PointProjective {
        try kzg.commit(evaluations)
    }

    // MARK: - Open (Prove)

    /// Generate a Zeromorph opening proof for ZM-fold evaluation at point u.
    ///
    /// Computes n quotient polynomials via even/odd decomposition, commits each,
    /// then opens the linearized polynomial L(X) at the Fiat-Shamir challenge zeta.
    public func open(evaluations: [Fr], point: [Fr], value: Fr? = nil) throws -> ZeromorphOpeningProof {
        let N = evaluations.count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree \(N)")

        let v = value ?? Self.evaluateZMFold(evaluations: evaluations, point: point)

        // Step 1: Compute quotient polynomials via even/odd decomposition
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
        let zeta = deriveZetaFS(point: point, quotientCommitments: quotientCommitments, value: v)

        // Step 4: Build linearized polynomial L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
        // where phi_s = zeta^{2^s} - u_{n-1-s}
        var zetaPow = zeta
        var L = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N { L[i] = evaluations[i] }
        L[0] = frSub(L[0], v)

        for s in 0..<n {
            let k = n - 1 - s
            let phi = frSub(zetaPow, point[k])
            zetaPow = frMul(zetaPow, zetaPow)  // advance to zeta^{2^{s+1}}
            let q = stepQuotients[s]
            for j in 0..<q.count {
                // Subtract phi_s * q^(s)[j] from coefficient j
                if j < N {
                    L[j] = frSub(L[j], frMul(phi, q[j]))
                }
            }
        }

        // Step 5: KZG open L at zeta
        let kzgProof = try kzg.open(L, at: zeta)

        return ZeromorphOpeningProof(
            quotientCommitments: quotientCommitments,
            claimedValue: v,
            kzgWitness: kzgProof.witness,
            linearizationEval: kzgProof.evaluation,
            zeta: zeta
        )
    }

    // MARK: - Verify (Pairing-Based)

    /// Verify a Zeromorph proof using BN254 pairings.
    ///
    /// Reconstructs [L]_1 from the polynomial commitment and quotient commitments:
    ///   C_L = [f]_1 - v*G1 - sum_s phi_s * [q^(s)]_1
    /// Then checks the KZG opening of L at zeta via pairing:
    ///   e(C_L - delta*G1 + zeta*W, [1]_2) = e(W, [tau]_2)
    /// where delta = L(zeta) is the linearization evaluation.
    public func verify(commitment: PointProjective, point: [Fr], value: Fr,
                       proof: ZeromorphOpeningProof, vk: ZeromorphVK) throws -> Bool {
        let n = point.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta
        let g1 = pointFromAffine(kzg.srs[0])

        // Compute phi_s = zeta^{2^s} - u_{n-1-s}
        var zetaPow = zeta
        var phis = [Fr]()
        phis.reserveCapacity(n)
        for s in 0..<n {
            let k = n - 1 - s
            phis.append(frSub(zetaPow, point[k]))
            zetaPow = frMul(zetaPow, zetaPow)
        }

        // C_L = [f]_1 - v*G1 - sum_s phi_s * [q^(s)]_1
        var cL = pointAdd(commitment, pointNeg(cPointScalarMul(g1, value)))
        for s in 0..<n {
            let term = cPointScalarMul(proof.quotientCommitments[s], phis[s])
            cL = pointAdd(cL, pointNeg(term))
        }

        // KZG pairing check for L(zeta) = delta:
        //   e(C_L - delta*G1 + zeta*W, [1]_2) * e(-W, [tau]_2) = 1
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

    // MARK: - Verify (SRS Secret, testing only)

    /// Full algebraic verification using the SRS secret (NOT secure in production).
    /// Checks P(zeta) = 0 algebraically and KZG proof consistency via secret scalar.
    public func verifyWithSecret(evaluations: [Fr], point: [Fr], value: Fr,
                                  proof: ZeromorphOpeningProof, srsSecret: Fr) -> Bool {
        let n = point.count
        let N = evaluations.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta

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

        // Check the telescoping identity: P(zeta) = 0
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

        // Rebuild L(X) and verify KZG proof
        var L = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N { L[i] = evaluations[i] }
        L[0] = frSub(L[0], value)

        zetaPow = zeta
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

        // Verify: L(zeta) matches proof
        let lZeta = evaluateUnivariate(L, at: zeta)
        if !frEqual(lZeta, proof.linearizationEval) {
            return false
        }

        // Verify KZG: [L]_1 - delta*G1 = (s - zeta) * [W]_1
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

    // MARK: - Evaluation Helpers

    /// Standard MLE evaluation: fold with (1-u_k)*lo + u_k*hi
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

    /// ZM fold evaluation: f_even + u_k * f_odd.
    /// Agrees with evaluateMLE on Boolean inputs {0,1}^n.
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

    // MARK: - Private Helpers

    /// Evaluate univariate polynomial at a point via Horner's method.
    public func evaluateUnivariate(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Fiat-Shamir challenge zeta using Keccak transcript.
    private func deriveZetaFS(point: [Fr], quotientCommitments: [PointProjective], value: Fr) -> Fr {
        var transcript = FiatShamirTranscript(label: "zeromorph", hasher: KeccakTranscriptHasher())

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
