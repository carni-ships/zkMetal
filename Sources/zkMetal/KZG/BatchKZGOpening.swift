// Batch KZG Polynomial Commitment Opening — Shplonk-style
//
// Opens N polynomials at M distinct points in a single combined proof.
// Critical for Plonk (which opens ~20 polynomials at 2 points: zeta and zeta*omega)
// and KZG batch verification in general.
//
// Shplonk batching (Brechenmacher, Gabizon, Khovratovich):
//   Given polynomial-point pairs {(f_i, z_j)} where f_i is opened at z_j,
//   construct a single witness polynomial W(x) such that one pairing check
//   suffices to verify all openings.
//
// Construction:
//   1. Group openings by evaluation point: S_j = {i : f_i opened at z_j}
//   2. For each point z_j, form combined polynomial:
//        h_j(x) = sum_{i in S_j} nu^{idx(i)} * (f_i(x) - f_i(z_j))
//   3. Compute the vanishing polynomial Z_T(x) = prod_j (x - z_j)
//   4. Compute combined quotient:
//        W(x) = sum_j [ h_j(x) / (x - z_j) * (Z_T(x) / (x - z_j))^{-1} ]
//      Simplified: W(x) = sum_j [ h_j(x) / (x - z_j) * Z_{T\j}(z_j)^{-1} ]
//      where Z_{T\j}(x) = Z_T(x) / (x - z_j)
//   5. Single pairing check verifies all openings.
//
// For the common Plonk case (2 points), this reduces from 2*N pairings to 2 pairings.
//
// Reference: "Efficient polynomial commitment schemes for multiple points and polynomials"
//            (Gabizon, Khovratovich, 2024)

import Foundation
import NeonFieldOps

// MARK: - Batch Opening Proof Structures

/// A claim that polynomial with commitment C evaluates to y at point z.
public struct OpeningClaim {
    /// Index into the commitments array
    public let polynomialIndex: Int
    /// Evaluation point
    public let point: Fr
    /// Claimed evaluation value
    public let evaluation: Fr

    public init(polynomialIndex: Int, point: Fr, evaluation: Fr) {
        self.polynomialIndex = polynomialIndex
        self.point = point
        self.evaluation = evaluation
    }
}

/// Batch opening proof: a single witness polynomial proving all claims.
public struct BatchOpeningProof {
    /// Combined witness commitment [W(s)]_1
    public let witness: PointProjective
    /// All evaluation claims (polynomial index, point, value)
    public let claims: [OpeningClaim]
    /// Commitments to the opened polynomials
    public let commitments: [PointProjective]

    public init(witness: PointProjective, claims: [OpeningClaim], commitments: [PointProjective]) {
        self.witness = witness
        self.claims = claims
        self.commitments = commitments
    }
}

// MARK: - Shplonk Batch Opening Engine

public class ShplonkBatchOpener {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    // MARK: - Core batch open

    /// Open N polynomials at M points using Shplonk batching.
    ///
    /// Each entry in `openingSets` maps a point index to the polynomial indices opened at that point.
    /// For Plonk: openingSets = [0: [0,1,...,19], 1: [3,7]] meaning 20 polys at z_0, 2 polys at z_1.
    ///
    /// - Parameters:
    ///   - commitments: pre-computed commitments [C_0, ..., C_{N-1}]
    ///   - polynomials: coefficient arrays [f_0(x), ..., f_{N-1}(x)]
    ///   - points: evaluation points [z_0, ..., z_{M-1}]
    ///   - openingSets: for each point index j, which polynomial indices are opened there
    ///   - transcript: Fiat-Shamir transcript for challenge generation
    /// - Returns: BatchOpeningProof with a single witness polynomial
    public func batchOpen(
        commitments: [PointProjective],
        polynomials: [[Fr]],
        points: [Fr],
        openingSets: [Int: [Int]],
        transcript: Transcript
    ) throws -> BatchOpeningProof {
        let numPolys = polynomials.count
        let numPoints = points.count
        guard numPolys == commitments.count else { throw MSMError.invalidInput }
        guard numPoints > 0 else { throw MSMError.invalidInput }

        // 1. Absorb commitments and points into transcript
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        for z in points {
            transcript.absorb(z)
        }

        // 2. Compute all evaluations f_i(z_j) for the requested openings
        //    Use sorted point indices for deterministic ordering (must match verify)
        var claims = [OpeningClaim]()
        var evaluationsByPoint = [Int: [(Int, Fr)]]()  // pointIdx -> [(polyIdx, eval)]

        let sortedPointIndicesForClaims = openingSets.keys.sorted()
        for pointIdx in sortedPointIndicesForClaims {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            guard pointIdx < numPoints else { throw MSMError.invalidInput }
            var evals = [(Int, Fr)]()
            for polyIdx in polyIndices {
                guard polyIdx < numPolys else { throw MSMError.invalidInput }
                let eval = cEvaluate(polynomials[polyIdx], at: points[pointIdx])
                claims.append(OpeningClaim(
                    polynomialIndex: polyIdx,
                    point: points[pointIdx],
                    evaluation: eval
                ))
                evals.append((polyIdx, eval))
            }
            evaluationsByPoint[pointIdx] = evals
        }

        // 3. Absorb evaluations into transcript, squeeze batching challenge nu
        for claim in claims {
            transcript.absorb(claim.evaluation)
        }
        let nu = transcript.squeeze()

        // 4. For each point z_j, form the combined numerator polynomial:
        //    h_j(x) = sum_{i in S_j} nu^{flatIdx} * (f_i(x) - f_i(z_j))
        //    then quotient q_j(x) = h_j(x) / (x - z_j)
        //
        // The witness is W(x) = sum_j r^j * q_j(x)
        // where r is a second challenge to combine across points.

        let r = transcript.squeeze()

        // Determine maximum polynomial degree for the combined quotient
        var maxDeg = 0
        for (_, polyIndices) in openingSets {
            for polyIdx in polyIndices {
                maxDeg = max(maxDeg, polynomials[polyIdx].count)
            }
        }
        guard maxDeg >= 2 else {
            return BatchOpeningProof(
                witness: pointIdentity(),
                claims: claims,
                commitments: commitments
            )
        }

        // 5. Build combined quotient W(x) = sum_j r^j * q_j(x)
        var combinedQuotient = [Fr](repeating: Fr.zero, count: maxDeg - 1)
        var nuPow = Fr.one  // Tracks nu^{flatIdx} across all openings
        var rPow = Fr.one   // Tracks r^j across points

        // Sort point indices for deterministic ordering
        let sortedPointIndices = openingSets.keys.sorted()

        for pointIdx in sortedPointIndices {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            let z = points[pointIdx]

            // Build h_j(x) = sum_i nu^{flatIdx} * (f_i(x) - y_i)
            var hj = [Fr](repeating: Fr.zero, count: maxDeg)
            for polyIdx in polyIndices {
                let poly = polynomials[polyIdx]
                let eval = evaluationsByPoint[pointIdx]!.first(where: { $0.0 == polyIdx })!.1

                // Accumulate nu^{flatIdx} * (f_i(x) - y_i)
                for k in 0..<poly.count {
                    var coeff = poly[k]
                    if k == 0 {
                        coeff = frSub(coeff, eval)
                    }
                    hj[k] = frAdd(hj[k], frMul(nuPow, coeff))
                }
                nuPow = frMul(nuPow, nu)
            }

            // Divide h_j by (x - z_j) via synthetic division
            let qj = syntheticDivision(hj, root: z)

            // Accumulate r^j * q_j(x) into combinedQuotient
            for k in 0..<qj.count {
                if k < combinedQuotient.count {
                    combinedQuotient[k] = frAdd(combinedQuotient[k], frMul(rPow, qj[k]))
                }
            }
            rPow = frMul(rPow, r)
        }

        // 6. Commit to the combined quotient: W = [combinedQuotient(s)]_1
        let witness = try kzg.commit(combinedQuotient)

        return BatchOpeningProof(
            witness: witness,
            claims: claims,
            commitments: commitments
        )
    }

    // MARK: - Simplified API: all polynomials at all points

    /// Open all polynomials at all points (fully crossed).
    /// Common case: every polynomial is opened at every point.
    public func batchOpen(
        commitments: [PointProjective],
        polynomials: [[Fr]],
        points: [Fr],
        transcript: Transcript
    ) throws -> BatchOpeningProof {
        var openingSets = [Int: [Int]]()
        let polyIndices = Array(0..<polynomials.count)
        for j in 0..<points.count {
            openingSets[j] = polyIndices
        }
        return try batchOpen(
            commitments: commitments,
            polynomials: polynomials,
            points: points,
            openingSets: openingSets,
            transcript: transcript
        )
    }

    // MARK: - Batch verify (SRS secret, for testing)

    /// Verify a batch opening proof using the SRS secret (testing only).
    ///
    /// In production, this would be a pairing check:
    ///   e(W, [s]_2) == e(combinedCommitment, [1]_2)
    /// Since we use the SRS secret for testing, we verify algebraically.
    public func batchVerify(
        proof: BatchOpeningProof,
        points: [Fr],
        openingSets: [Int: [Int]],
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let commitments = proof.commitments
        guard !proof.claims.isEmpty else { return true }

        // Reconstruct transcript state: absorb commitments and points
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        for z in points {
            transcript.absorb(z)
        }
        for claim in proof.claims {
            transcript.absorb(claim.evaluation)
        }
        let nu = transcript.squeeze()
        let r = transcript.squeeze()

        // Reconstruct the combined quotient evaluated at s (the SRS secret):
        // W(s) should equal sum_j r^j * h_j(s) / (s - z_j)
        // where h_j(s) = sum_{i in S_j} nu^{flatIdx} * (C_i_at_s - y_i)
        // and C_i_at_s can be recovered from commitment via the SRS secret.

        // Instead, verify that the proof witness W matches the expected value:
        // [W(s)]_1 == expected point computed from commitments and evaluations.

        let g1 = pointFromAffine(kzg.srs[0])
        var expected = pointIdentity()
        var nuPow = Fr.one
        var rPow = Fr.one

        let sortedPointIndices = openingSets.keys.sorted()
        var claimIdx = 0

        for pointIdx in sortedPointIndices {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            let z = points[pointIdx]
            let sMz = frSub(srsSecret, z)
            let sMzInv = frInverse(sMz)

            // h_j(s) = sum_{i in S_j} nu^{flatIdx} * (C_i evaluated at s - y_i)
            // Since C_i = [f_i(s)]_1, we have C_i = f_i(s) * G, so:
            // [h_j(s)]_1 = sum_{i in S_j} nu^{flatIdx} * (C_i - [y_i]*G)
            var hjPoint = pointIdentity()
            for polyIdx in polyIndices {
                let claim = proof.claims[claimIdx]
                claimIdx += 1

                let yG = cPointScalarMul(g1, claim.evaluation)
                let numerator = pointAdd(commitments[polyIdx], pointNeg(yG))
                hjPoint = pointAdd(hjPoint, cPointScalarMul(numerator, nuPow))
                nuPow = frMul(nuPow, nu)
            }

            // r^j * h_j(s) / (s - z_j) = r^j * sMzInv * [h_j(s)]_1
            let scalar = frMul(rPow, sMzInv)
            expected = pointAdd(expected, cPointScalarMul(hjPoint, scalar))
            rPow = frMul(rPow, r)
        }

        return pointsEqual(proof.witness, expected)
    }

    /// Simplified verify: all polynomials at all points.
    public func batchVerify(
        proof: BatchOpeningProof,
        points: [Fr],
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        var openingSets = [Int: [Int]]()
        let polyIndices = Array(0..<proof.commitments.count)
        for j in 0..<points.count {
            openingSets[j] = polyIndices
        }
        return batchVerify(
            proof: proof,
            points: points,
            openingSets: openingSets,
            transcript: transcript,
            srsSecret: srsSecret
        )
    }

    // MARK: - Helpers

    /// Evaluate polynomial at a point using Horner's method via C CIOS.
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

    /// Synthetic division: divide poly by (x - root), returning quotient.
    /// poly(x) = (x - root) * q(x) + remainder
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

    /// Absorb a projective point into the transcript.
    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            // Convert Fp coordinates to Fr for absorption (truncating is fine for Fiat-Shamir)
            let xLimbs = fpToInt(aff[0].x)
            let yLimbs = fpToInt(aff[0].y)
            transcript.absorb(Fr.from64(xLimbs))
            transcript.absorb(Fr.from64(yLimbs))
        }
    }

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
