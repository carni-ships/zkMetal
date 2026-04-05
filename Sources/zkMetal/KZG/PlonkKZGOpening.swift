// Plonk-Specific KZG Opening
//
// Specialized multi-opening for the Plonk proof system pattern:
//   - Opens a, b, c, z, t_lo, t_mid, t_hi, S_sigma1, S_sigma2 at zeta (9 polynomials)
//   - Opens z at zeta*omega (shifted opening for the permutation argument)
//   - Produces two opening proofs W_zeta and W_zeta_omega
//
// This is more efficient than generic multi-open because:
//   1. Only 2 evaluation points (zeta, zeta*omega)
//   2. Fixed structure allows precomputation
//   3. Batch pairing verification of both proofs
//
// Reference: "PlonK: Permutations over Lagrange-bases for Oecumenical Noninteractive
//             arguments of Knowledge" (Gabizon, Williamson, Ciobotaru, 2019)

import Foundation
import NeonFieldOps

// MARK: - Plonk Opening Structures

/// Plonk opening proof containing two witness points.
public struct PlonkOpeningProof {
    /// Witness for openings at zeta: W_zeta = [q_zeta(s)]_1
    public let wZeta: PointProjective
    /// Witness for opening at zeta*omega: W_zetaOmega = [q_{zeta*omega}(s)]_1
    public let wZetaOmega: PointProjective

    /// Evaluations of the 9 polynomials at zeta
    /// Order: a, b, c, z, t_lo, t_mid, t_hi, sigma1, sigma2
    public let zetaEvaluations: [Fr]
    /// Evaluation of z(X) at zeta*omega
    public let zOmegaEvaluation: Fr

    /// Commitments to all polynomials (for verification)
    /// Order: a, b, c, z, t_lo, t_mid, t_hi, sigma1, sigma2
    public let commitments: [PointProjective]

    public init(wZeta: PointProjective, wZetaOmega: PointProjective,
                zetaEvaluations: [Fr], zOmegaEvaluation: Fr,
                commitments: [PointProjective]) {
        self.wZeta = wZeta
        self.wZetaOmega = wZetaOmega
        self.zetaEvaluations = zetaEvaluations
        self.zOmegaEvaluation = zOmegaEvaluation
        self.commitments = commitments
    }
}

// MARK: - Plonk KZG Opener

public class PlonkKZGOpener {
    public let kzg: KZGEngine

    /// Polynomial index constants for the standard Plonk opening pattern.
    public static let POLY_A = 0
    public static let POLY_B = 1
    public static let POLY_C = 2
    public static let POLY_Z = 3
    public static let POLY_T_LO = 4
    public static let POLY_T_MID = 5
    public static let POLY_T_HI = 6
    public static let POLY_SIGMA1 = 7
    public static let POLY_SIGMA2 = 8

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Open Plonk polynomials at zeta and zeta*omega.
    ///
    /// - Parameters:
    ///   - polynomials: [a, b, c, z, t_lo, t_mid, t_hi, sigma1, sigma2] coefficient arrays
    ///   - zeta: evaluation challenge point
    ///   - omega: root of unity (domain generator) so zeta*omega is the shifted point
    ///   - transcript: Fiat-Shamir transcript for challenge generation
    /// - Returns: PlonkOpeningProof with W_zeta and W_zeta_omega
    public func open(
        polynomials: [[Fr]],
        zeta: Fr,
        omega: Fr,
        transcript: Transcript
    ) throws -> PlonkOpeningProof {
        guard polynomials.count >= 9 else { throw MSMError.invalidInput }

        let zetaOmega = frMul(zeta, omega)

        // 1. Commit to all polynomials
        let commitments = try kzg.batchCommit(polynomials)

        // Absorb commitments into transcript
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(zeta)

        // 2. Compute evaluations at zeta for all 9 polynomials
        var zetaEvals = [Fr]()
        zetaEvals.reserveCapacity(9)
        for i in 0..<9 {
            zetaEvals.append(cEvaluate(polynomials[i], at: zeta))
        }

        // Compute z(zeta*omega)
        let zOmegaEval = cEvaluate(polynomials[Self.POLY_Z], at: zetaOmega)

        // Absorb evaluations
        for eval in zetaEvals {
            transcript.absorb(eval)
        }
        transcript.absorb(zOmegaEval)

        // Squeeze random linear combination challenge v
        let v = transcript.squeeze()

        // 3. Compute W_zeta: linearized opening at zeta
        //    q_zeta(X) = sum_i v^i * (f_i(X) - f_i(zeta)) / (X - zeta)
        let maxDeg = polynomials[0..<9].map { $0.count }.max() ?? 0
        var combinedZeta = [Fr](repeating: Fr.zero, count: maxDeg)
        var vPow = Fr.one

        for i in 0..<9 {
            let poly = polynomials[i]
            for k in 0..<poly.count {
                var coeff = poly[k]
                if k == 0 {
                    coeff = frSub(coeff, zetaEvals[i])
                }
                combinedZeta[k] = frAdd(combinedZeta[k], frMul(vPow, coeff))
            }
            vPow = frMul(vPow, v)
        }

        let qZeta = syntheticDivision(combinedZeta, root: zeta)
        let wZeta: PointProjective
        if qZeta.isEmpty {
            wZeta = pointIdentity()
        } else {
            wZeta = try kzg.commit(qZeta)
        }

        // 4. Compute W_zeta_omega: opening z(X) at zeta*omega
        //    q_{zeta*omega}(X) = (z(X) - z(zeta*omega)) / (X - zeta*omega)
        let zPoly = polynomials[Self.POLY_Z]
        var shiftedZ = zPoly
        shiftedZ[0] = frSub(shiftedZ[0], zOmegaEval)

        let qZetaOmega = syntheticDivision(shiftedZ, root: zetaOmega)
        let wZetaOmega: PointProjective
        if qZetaOmega.isEmpty {
            wZetaOmega = pointIdentity()
        } else {
            wZetaOmega = try kzg.commit(qZetaOmega)
        }

        return PlonkOpeningProof(
            wZeta: wZeta,
            wZetaOmega: wZetaOmega,
            zetaEvaluations: zetaEvals,
            zOmegaEvaluation: zOmegaEval,
            commitments: commitments
        )
    }

    /// Open arbitrary polynomial sets at zeta and zeta*omega.
    /// More flexible variant: specify which polynomial indices open at each point.
    ///
    /// - Parameters:
    ///   - polynomials: all polynomials
    ///   - zetaIndices: indices of polynomials opened at zeta
    ///   - zetaOmegaIndices: indices of polynomials opened at zeta*omega
    ///   - zeta: evaluation challenge
    ///   - omega: domain generator
    ///   - transcript: Fiat-Shamir transcript
    public func openFlexible(
        polynomials: [[Fr]],
        zetaIndices: [Int],
        zetaOmegaIndices: [Int],
        zeta: Fr,
        omega: Fr,
        transcript: Transcript
    ) throws -> PlonkOpeningProof {
        guard !polynomials.isEmpty else { throw MSMError.invalidInput }

        let zetaOmega = frMul(zeta, omega)

        // Commit
        let commitments = try kzg.batchCommit(polynomials)
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(zeta)

        // Evaluate at zeta
        var zetaEvals = [Fr]()
        for i in zetaIndices {
            zetaEvals.append(cEvaluate(polynomials[i], at: zeta))
        }

        // Evaluate at zeta*omega
        var zetaOmegaEvals = [Fr]()
        for i in zetaOmegaIndices {
            zetaOmegaEvals.append(cEvaluate(polynomials[i], at: zetaOmega))
        }

        // Absorb all evaluations
        for eval in zetaEvals { transcript.absorb(eval) }
        for eval in zetaOmegaEvals { transcript.absorb(eval) }

        let v = transcript.squeeze()

        // W_zeta: combine all zeta openings
        let maxDegZeta = zetaIndices.map { polynomials[$0].count }.max() ?? 0
        var combinedZeta = [Fr](repeating: Fr.zero, count: maxDegZeta)
        var vPow = Fr.one
        for (evalIdx, polyIdx) in zetaIndices.enumerated() {
            let poly = polynomials[polyIdx]
            for k in 0..<poly.count {
                var coeff = poly[k]
                if k == 0 { coeff = frSub(coeff, zetaEvals[evalIdx]) }
                combinedZeta[k] = frAdd(combinedZeta[k], frMul(vPow, coeff))
            }
            vPow = frMul(vPow, v)
        }
        let qZeta = syntheticDivision(combinedZeta, root: zeta)
        let wZeta = qZeta.isEmpty ? pointIdentity() : try kzg.commit(qZeta)

        // W_zeta_omega: combine zeta*omega openings
        let maxDegOmega = zetaOmegaIndices.map { polynomials[$0].count }.max() ?? 0
        var combinedOmega = [Fr](repeating: Fr.zero, count: maxDegOmega)
        vPow = Fr.one
        for (evalIdx, polyIdx) in zetaOmegaIndices.enumerated() {
            let poly = polynomials[polyIdx]
            for k in 0..<poly.count {
                var coeff = poly[k]
                if k == 0 { coeff = frSub(coeff, zetaOmegaEvals[evalIdx]) }
                combinedOmega[k] = frAdd(combinedOmega[k], frMul(vPow, coeff))
            }
            vPow = frMul(vPow, v)
        }
        let qOmega = syntheticDivision(combinedOmega, root: zetaOmega)
        let wZetaOmega = qOmega.isEmpty ? pointIdentity() : try kzg.commit(qOmega)

        // Compose evaluations for proof (zeta evals followed by first omega eval)
        let zOmegaEval = zetaOmegaEvals.isEmpty ? Fr.zero : zetaOmegaEvals[0]

        return PlonkOpeningProof(
            wZeta: wZeta,
            wZetaOmega: wZetaOmega,
            zetaEvaluations: zetaEvals,
            zOmegaEvaluation: zOmegaEval,
            commitments: commitments
        )
    }

    // MARK: - Helpers

    private func cEvaluate(_ coeffs: [Fr], at z: Fr) -> Fr {
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

    private func syntheticDivision(_ poly: [Fr], root: Fr) -> [Fr] {
        let n = poly.count
        if n < 2 { return [] }
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        poly.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: root) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    bn254_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return quotient
    }

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

// MARK: - Plonk KZG Verifier

public class PlonkKZGVerifier {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Verify a Plonk opening proof using the SRS secret (testing mode).
    ///
    /// In production, this uses a batch pairing check:
    ///   e(W_zeta + u * W_zetaOmega, [s]_2) =
    ///     e(zeta * W_zeta + u * zetaOmega * W_zetaOmega + F - E, [1]_2)
    /// where:
    ///   F = sum_i v^i * C_i  (combined commitment)
    ///   E = [sum_i v^i * y_i + u * z_omega_eval] * G  (combined evaluation)
    ///
    /// Test verification: reconstruct expected witness from SRS secret.
    public func verify(
        proof: PlonkOpeningProof,
        zeta: Fr,
        omega: Fr,
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let commitments = proof.commitments
        guard proof.zetaEvaluations.count > 0 else { return false }

        let zetaOmega = frMul(zeta, omega)

        // Reconstruct transcript
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(zeta)
        for eval in proof.zetaEvaluations {
            transcript.absorb(eval)
        }
        transcript.absorb(proof.zOmegaEvaluation)
        let v = transcript.squeeze()

        let g1 = pointFromAffine(kzg.srs[0])
        let sMzeta = frSub(srsSecret, zeta)
        let sMzetaInv = frInverse(sMzeta)
        let sMzetaOmega = frSub(srsSecret, zetaOmega)
        let sMzetaOmegaInv = frInverse(sMzetaOmega)

        // Verify W_zeta: reconstruct q_zeta(s)
        // q_zeta(s) = (1/(s-zeta)) * sum_i v^i * (C_i(s) - y_i)
        var zetaPoint = pointIdentity()
        var vPow = Fr.one
        let zetaCount = min(proof.zetaEvaluations.count, commitments.count)
        for i in 0..<zetaCount {
            let yG = cPointScalarMul(g1, proof.zetaEvaluations[i])
            let diff = pointAdd(commitments[i], pointNeg(yG))
            zetaPoint = pointAdd(zetaPoint, cPointScalarMul(diff, vPow))
            vPow = frMul(vPow, v)
        }
        let expectedWZeta = cPointScalarMul(zetaPoint, sMzetaInv)

        if !pointsEqual(proof.wZeta, expectedWZeta) {
            return false
        }

        // Verify W_zeta_omega: reconstruct q_{zeta*omega}(s)
        // For the standard pattern, only z(X) is opened at zeta*omega
        let zIdx = PlonkKZGOpener.POLY_Z
        guard zIdx < commitments.count else { return false }
        let yG = cPointScalarMul(g1, proof.zOmegaEvaluation)
        let diff = pointAdd(commitments[zIdx], pointNeg(yG))
        let expectedWZetaOmega = cPointScalarMul(diff, sMzetaOmegaInv)

        return pointsEqual(proof.wZetaOmega, expectedWZetaOmega)
    }

    /// Verify with flexible polynomial index sets.
    public func verifyFlexible(
        proof: PlonkOpeningProof,
        zetaIndices: [Int],
        zetaOmegaIndices: [Int],
        zeta: Fr,
        omega: Fr,
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let commitments = proof.commitments
        let zetaOmega = frMul(zeta, omega)

        // Reconstruct transcript
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        transcript.absorb(zeta)
        for eval in proof.zetaEvaluations {
            transcript.absorb(eval)
        }
        // Absorb zeta*omega evaluations (just the one in standard proof)
        if !zetaOmegaIndices.isEmpty {
            transcript.absorb(proof.zOmegaEvaluation)
        }
        let v = transcript.squeeze()

        let g1 = pointFromAffine(kzg.srs[0])

        // Verify W_zeta
        let sMzetaInv = frInverse(frSub(srsSecret, zeta))
        var zetaPoint = pointIdentity()
        var vPow = Fr.one
        for (evalIdx, polyIdx) in zetaIndices.enumerated() {
            guard polyIdx < commitments.count, evalIdx < proof.zetaEvaluations.count else { return false }
            let yG = cPointScalarMul(g1, proof.zetaEvaluations[evalIdx])
            let diff = pointAdd(commitments[polyIdx], pointNeg(yG))
            zetaPoint = pointAdd(zetaPoint, cPointScalarMul(diff, vPow))
            vPow = frMul(vPow, v)
        }
        let expectedWZeta = cPointScalarMul(zetaPoint, sMzetaInv)
        if !pointsEqual(proof.wZeta, expectedWZeta) {
            return false
        }

        // Verify W_zeta_omega
        if !zetaOmegaIndices.isEmpty {
            let sMzetaOmegaInv = frInverse(frSub(srsSecret, zetaOmega))
            var omegaPoint = pointIdentity()
            vPow = Fr.one
            for (_, polyIdx) in zetaOmegaIndices.enumerated() {
                guard polyIdx < commitments.count else { return false }
                let yG = cPointScalarMul(g1, proof.zOmegaEvaluation)
                let diff = pointAdd(commitments[polyIdx], pointNeg(yG))
                omegaPoint = pointAdd(omegaPoint, cPointScalarMul(diff, vPow))
                vPow = frMul(vPow, v)
            }
            let expectedWOmega = cPointScalarMul(omegaPoint, sMzetaOmegaInv)
            if !pointsEqual(proof.wZetaOmega, expectedWOmega) {
                return false
            }
        }

        return true
    }

    // MARK: - Helpers

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

    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
