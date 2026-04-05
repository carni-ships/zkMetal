// KZG Multi-Opening Engine
//
// Amortizes the cost of opening multiple polynomials at multiple evaluation points.
// Critical for Plonk (opens ~20 polynomials at zeta and zeta*omega) and Marlin.
//
// Protocol (two-round Fiat-Shamir):
//   Round 1: For each evaluation point z_j, compute random-linear-combination quotient
//            h_j(X) = sum_i r^i * (f_i(X) - f_i(z_j)) / (X - z_j)
//            and commit to h_j.
//   Round 2: After verifier challenge gamma, compute combined quotient
//            W(X) = sum_j gamma^j * h_j(X)
//            Final proof is a single G1 point [W(s)]_1.
//
// Verification reconstructs pairing inputs and checks via SRS secret (test mode)
// or a single pairing equation (production).

import Foundation
import NeonFieldOps

// MARK: - Multi-Open Proof Structures

/// Per-point quotient commitment in the multi-open protocol.
public struct PerPointQuotient {
    /// The evaluation point z_j
    public let point: Fr
    /// Commitment to h_j(X) = sum_i r^i * (f_i(X) - f_i(z_j)) / (X - z_j)
    public let commitment: PointProjective
    /// Polynomial indices opened at this point
    public let polynomialIndices: [Int]
    /// Evaluations f_i(z_j) for each polynomial in polynomialIndices
    public let evaluations: [Fr]

    public init(point: Fr, commitment: PointProjective, polynomialIndices: [Int], evaluations: [Fr]) {
        self.point = point
        self.commitment = commitment
        self.polynomialIndices = polynomialIndices
        self.evaluations = evaluations
    }
}

/// Complete multi-opening proof.
public struct KZGMultiOpenProof {
    /// Per-point quotient commitments [H_0], [H_1], ...
    public let perPointQuotients: [PerPointQuotient]
    /// Final combined witness W = sum_j gamma^j * h_j(X) committed
    public let witness: PointProjective
    /// Polynomial commitments (for verification)
    public let commitments: [PointProjective]
    /// Batching challenge r (absorbed from transcript)
    public let r: Fr
    /// Combination challenge gamma (absorbed from transcript)
    public let gamma: Fr

    public init(perPointQuotients: [PerPointQuotient], witness: PointProjective,
                commitments: [PointProjective], r: Fr, gamma: Fr) {
        self.perPointQuotients = perPointQuotients
        self.witness = witness
        self.commitments = commitments
        self.r = r
        self.gamma = gamma
    }
}

// MARK: - Multi-Open Prover

public class KZGMultiOpenProver {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Open N polynomials at M points efficiently.
    ///
    /// - Parameters:
    ///   - polynomials: coefficient arrays [f_0(x), ..., f_{N-1}(x)]
    ///   - points: evaluation points [z_0, ..., z_{M-1}]
    ///   - openingSets: maps point index j -> polynomial indices opened at z_j
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: KZGMultiOpenProof with a single witness point
    public func open(
        polynomials: [[Fr]],
        points: [Fr],
        openingSets: [Int: [Int]],
        transcript: Transcript
    ) throws -> KZGMultiOpenProof {
        let numPolys = polynomials.count
        let numPoints = points.count
        guard numPolys > 0, numPoints > 0 else { throw MSMError.invalidInput }

        // 1. Commit to all polynomials
        let commitments = try kzg.batchCommit(polynomials)

        // Absorb commitments and points
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        for z in points {
            transcript.absorb(z)
        }

        // 2. Compute evaluations and absorb them
        let sortedPointIndices = openingSets.keys.sorted()
        var allEvaluations = [Int: [(Int, Fr)]]() // pointIdx -> [(polyIdx, eval)]
        for pointIdx in sortedPointIndices {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            guard pointIdx < numPoints else { throw MSMError.invalidInput }
            var evals = [(Int, Fr)]()
            for polyIdx in polyIndices {
                guard polyIdx < numPolys else { throw MSMError.invalidInput }
                let eval = cEvaluate(polynomials[polyIdx], at: points[pointIdx])
                evals.append((polyIdx, eval))
                transcript.absorb(eval)
            }
            allEvaluations[pointIdx] = evals
        }

        // Squeeze batching challenge r
        let r = transcript.squeeze()

        // 3. For each point z_j, compute h_j(X) = sum_i r^i * (f_i(X) - f_i(z_j)) / (X - z_j)
        //    and commit to h_j
        var perPointQuotients = [PerPointQuotient]()
        var hjPolynomials = [[Fr]]() // Store for round 2
        var rPow = Fr.one

        for pointIdx in sortedPointIndices {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            let z = points[pointIdx]
            let evals = allEvaluations[pointIdx]!

            // Build the combined numerator: sum_i r^i * (f_i(X) - f_i(z_j))
            let maxDeg = polyIndices.map { polynomials[$0].count }.max() ?? 0
            var combined = [Fr](repeating: Fr.zero, count: maxDeg)

            var evalList = [Fr]()
            for (_, (polyIdx, eval)) in evals.enumerated() {
                let poly = polynomials[polyIdx]
                for k in 0..<poly.count {
                    var coeff = poly[k]
                    if k == 0 {
                        coeff = frSub(coeff, eval)
                    }
                    combined[k] = frAdd(combined[k], frMul(rPow, coeff))
                }
                evalList.append(eval)
                rPow = frMul(rPow, r)
            }

            // Divide by (X - z_j) via synthetic division
            let hj = syntheticDivision(combined, root: z)
            hjPolynomials.append(hj)

            // Commit to h_j
            let hjCommitment: PointProjective
            if hj.isEmpty || hj.allSatisfy({ frIsZero($0) }) {
                hjCommitment = pointIdentity()
            } else {
                hjCommitment = try kzg.commit(hj)
            }

            perPointQuotients.append(PerPointQuotient(
                point: z,
                commitment: hjCommitment,
                polynomialIndices: polyIndices,
                evaluations: evalList
            ))
        }

        // 4. Absorb per-point commitments, squeeze combination challenge gamma
        for pq in perPointQuotients {
            absorbPoint(pq.commitment, into: transcript)
        }
        let gamma = transcript.squeeze()

        // 5. Compute combined quotient W(X) = sum_j gamma^j * h_j(X)
        let maxQuotientDeg = hjPolynomials.map { $0.count }.max() ?? 0
        var combinedW = [Fr](repeating: Fr.zero, count: maxQuotientDeg)
        var gammaPow = Fr.one

        for hj in hjPolynomials {
            for k in 0..<hj.count {
                combinedW[k] = frAdd(combinedW[k], frMul(gammaPow, hj[k]))
            }
            gammaPow = frMul(gammaPow, gamma)
        }

        // 6. Final MSM: W = [combinedW(s)]_1
        let witness: PointProjective
        if combinedW.isEmpty || combinedW.allSatisfy({ frIsZero($0) }) {
            witness = pointIdentity()
        } else {
            witness = try kzg.commit(combinedW)
        }

        return KZGMultiOpenProof(
            perPointQuotients: perPointQuotients,
            witness: witness,
            commitments: commitments,
            r: r,
            gamma: gamma
        )
    }

    /// Convenience: open all polynomials at all points.
    public func open(
        polynomials: [[Fr]],
        points: [Fr],
        transcript: Transcript
    ) throws -> KZGMultiOpenProof {
        var openingSets = [Int: [Int]]()
        let polyIndices = Array(0..<polynomials.count)
        for j in 0..<points.count {
            openingSets[j] = polyIndices
        }
        return try open(polynomials: polynomials, points: points,
                        openingSets: openingSets, transcript: transcript)
    }

    /// Convenience: open N polynomials at a single point.
    public func openSinglePoint(
        polynomials: [[Fr]],
        point: Fr,
        transcript: Transcript
    ) throws -> KZGMultiOpenProof {
        return try open(polynomials: polynomials, points: [point], transcript: transcript)
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

    private func frIsZero(_ a: Fr) -> Bool {
        return frToInt(a) == frToInt(Fr.zero)
    }
}

// MARK: - Multi-Open Verifier

public class KZGMultiOpenVerifier {
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Verify a multi-opening proof using the SRS secret (testing only).
    ///
    /// Production verification would use pairing:
    ///   e(W, [tau]_2) = e(sum_j gamma^j * [h_j(s) * (s - z_j)] / (s - z_j), [1]_2)
    /// which simplifies to checking that W encodes the correct combined quotient.
    ///
    /// Test verification: reconstruct expected W from commitments and evaluations
    /// using the known SRS secret s, then compare.
    public func verify(
        proof: KZGMultiOpenProof,
        points: [Fr],
        openingSets: [Int: [Int]],
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        let commitments = proof.commitments
        guard !proof.perPointQuotients.isEmpty else { return true }

        // Reconstruct transcript state
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        for z in points {
            transcript.absorb(z)
        }

        // Absorb evaluations in the same order as prover (sorted point indices)
        let sortedPointIndices = openingSets.keys.sorted()
        guard proof.perPointQuotients.count == sortedPointIndices.count else { return false }
        for pq in proof.perPointQuotients {
            for eval in pq.evaluations {
                transcript.absorb(eval)
            }
        }

        let r = transcript.squeeze()

        // Absorb per-point commitments
        for pq in proof.perPointQuotients {
            absorbPoint(pq.commitment, into: transcript)
        }
        let gamma = transcript.squeeze()

        // Verify: reconstruct expected witness using SRS secret
        // W(s) = sum_j gamma^j * h_j(s)
        // h_j(s) = sum_i r^i * (f_i(s) - f_i(z_j)) / (s - z_j)
        // f_i(s) is encoded in commitment C_i (since C_i = [f_i(s)]_1)

        let g1 = pointFromAffine(kzg.srs[0])
        var expectedW = pointIdentity()
        var rPow = Fr.one
        var gammaPow = Fr.one

        for (pqIdx, pointIdx) in sortedPointIndices.enumerated() {
            guard let polyIndices = openingSets[pointIdx] else { continue }
            let z = points[pointIdx]
            let sMz = frSub(srsSecret, z)
            let sMzInv = frInverse(sMz)

            let pq = proof.perPointQuotients[pqIdx]

            // h_j(s) = sum_i r^i * (C_i/G - y_i) / (s - z_j)
            // In point form: [h_j(s)]_1 = sMzInv * sum_i r^i * (C_i - [y_i]*G)
            var hjPoint = pointIdentity()
            for (evalIdx, polyIdx) in polyIndices.enumerated() {
                let eval = pq.evaluations[evalIdx]
                let yG = cPointScalarMul(g1, eval)
                let numerator = pointAdd(commitments[polyIdx], pointNeg(yG))
                hjPoint = pointAdd(hjPoint, cPointScalarMul(numerator, rPow))
                rPow = frMul(rPow, r)
            }
            hjPoint = cPointScalarMul(hjPoint, sMzInv)

            // Verify per-point quotient commitment matches
            if !pointsEqual(pq.commitment, hjPoint) {
                return false
            }

            // Accumulate gamma^j * [h_j(s)]_1
            expectedW = pointAdd(expectedW, cPointScalarMul(hjPoint, gammaPow))
            gammaPow = frMul(gammaPow, gamma)
        }

        return pointsEqual(proof.witness, expectedW)
    }

    /// Convenience: verify when all polynomials are opened at all points.
    public func verify(
        proof: KZGMultiOpenProof,
        points: [Fr],
        transcript: Transcript,
        srsSecret: Fr
    ) -> Bool {
        var openingSets = [Int: [Int]]()
        let polyIndices = Array(0..<proof.commitments.count)
        for j in 0..<points.count {
            openingSets[j] = polyIndices
        }
        return verify(proof: proof, points: points, openingSets: openingSets,
                      transcript: transcript, srsSecret: srsSecret)
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
