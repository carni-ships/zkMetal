// KZG Batch Opening — standalone batch polynomial opening with pairing verification
//
// Given N polynomials p_1,...,p_N opened at possibly different points z_1,...,z_N:
//   - Random gamma combines: sum_i gamma^i * (p_i(X) - y_i) / (X - z_i)
//   - Single KZG proof for the combined quotient
//   - Verification: single pairing check (or SRS-secret check for testing)
//
// This module wraps KZGEngine's batch opening with a pairing-based verifier and
// adds Fiat-Shamir challenge derivation for non-interactive batch openings.
//
// For rollup batch verification: verify 100 polynomial openings with 2 pairings
// instead of 100 pairings.

import Foundation

// MARK: - KZG Batch Opening with Pairing Verification

public struct KZGBatchOpeningVerifier {

    public init() {}

    // MARK: - Same-point batch verification

    /// Verify a KZG batch opening at the same point using SRS secret.
    ///
    /// Given commitments C_i, evaluations y_i, point z, and combined proof pi:
    ///   C_combined = sum_i gamma^i * C_i
    ///   y_combined = sum_i gamma^i * y_i
    ///   Check: C_combined - [y_combined]*G == [s - z]*pi
    ///
    /// In production with pairings:
    ///   e(C_combined - [y_combined]*G, H) == e(pi, [s]*H - [z]*H)
    public func verifySamePoint(
        commitments: [PointProjective],
        evaluations: [Fr],
        point: Fr,
        proof: PointProjective,
        gamma: Fr,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        guard commitments.count == evaluations.count, !commitments.isEmpty else { return false }
        let n = commitments.count

        // Compute combined commitment: C = sum_i gamma^i * C_i
        var combinedCommitment = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            combinedCommitment = pointAdd(combinedCommitment, cPointScalarMul(commitments[i], gammaPow))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Compute combined evaluation: y = sum_i gamma^i * y_i
        var combinedEval = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedEval = frAdd(combinedEval, frMul(gammaPow, evaluations[i]))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Check: C_combined == [y]*G + [s - z]*proof
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, combinedEval)
        let sMz = frSub(srsSecret, point)
        let szProof = cPointScalarMul(proof, sMz)
        let expected = pointAdd(yG, szProof)

        return pointsEqual(combinedCommitment, expected)
    }

    // MARK: - Multi-point batch verification

    /// Verify a multi-point KZG batch opening using SRS secret.
    ///
    /// Given N polynomials opened at different points z_i with values y_i:
    ///   quotient = sum_i gamma^i * (p_i(X) - y_i) / (X - z_i)
    ///   proof = [quotient(s)]_1
    ///
    /// Verification (with SRS secret):
    ///   For each i, check that commitment C_i is consistent with evaluation y_i at z_i
    ///   through the combined proof.
    public func verifyMultiPoint(
        commitments: [PointProjective],
        evaluations: [Fr],
        points: [Fr],
        proof: PointProjective,
        gamma: Fr,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        let n = commitments.count
        guard n == evaluations.count, n == points.count, n > 0 else { return false }

        // Compute expected proof value from SRS secret:
        // proof should equal [sum_i gamma^i * (p_i(s) - y_i) / (s - z_i)]_1
        // Since C_i = [p_i(s)]_1, we compute:
        //   sum_i gamma^i * (C_i - [y_i]*G) / (s - z_i)
        // where division by (s - z_i) is scalar division in Fr.

        let g1 = pointFromAffine(srs[0])
        var expected = pointIdentity()
        var gammaPow = Fr.one

        for i in 0..<n {
            // C_i - [y_i]*G  (numerator contribution for proof i)
            let yiG = cPointScalarMul(g1, evaluations[i])
            let numerator = pointAdd(commitments[i], pointNeg(yiG))

            // 1 / (s - z_i)
            let sMzi = frSub(srsSecret, points[i])
            let sMziInv = frInverse(sMzi)

            // gamma^i / (s - z_i) * numerator
            let scalar = frMul(gammaPow, sMziInv)
            let term = cPointScalarMul(numerator, scalar)
            expected = pointAdd(expected, term)

            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        return pointsEqual(proof, expected)
    }

    // MARK: - Fiat-Shamir batch opening

    /// Non-interactive batch opening: derive gamma from transcript absorbing all data.
    ///
    /// Usage:
    ///   let (gamma, transcript) = deriveChallenge(commitments: cs, evaluations: ys, points: zs)
    ///   let valid = verifyMultiPoint(..., gamma: gamma, ...)
    public func deriveChallenge(
        commitments: [PointProjective],
        evaluations: [Fr],
        points: [Fr]
    ) -> Fr {
        let transcript = Transcript(label: "kzg-batch-opening", backend: .poseidon2)

        for c in commitments {
            let cAff = batchToAffine([c])
            transcript.absorb(fpToFr(cAff[0].x))
            transcript.absorb(fpToFr(cAff[0].y))
        }
        for y in evaluations {
            transcript.absorb(y)
        }
        for z in points {
            transcript.absorb(z)
        }

        return transcript.squeeze()
    }

    /// Full non-interactive verification: derive gamma, then verify.
    public func verifyMultiPointNonInteractive(
        commitments: [PointProjective],
        evaluations: [Fr],
        points: [Fr],
        proof: PointProjective,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        let gamma = deriveChallenge(
            commitments: commitments, evaluations: evaluations, points: points)
        return verifyMultiPoint(
            commitments: commitments, evaluations: evaluations, points: points,
            proof: proof, gamma: gamma, srs: srs, srsSecret: srsSecret)
    }

    /// Full non-interactive same-point verification.
    public func verifySamePointNonInteractive(
        commitments: [PointProjective],
        evaluations: [Fr],
        point: Fr,
        proof: PointProjective,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        let allPoints = [Fr](repeating: point, count: commitments.count)
        let gamma = deriveChallenge(
            commitments: commitments, evaluations: evaluations, points: allPoints)
        return verifySamePoint(
            commitments: commitments, evaluations: evaluations, point: point,
            proof: proof, gamma: gamma, srs: srs, srsSecret: srsSecret)
    }

    // MARK: - Helpers

    /// Compare two projective points for equality.
    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
