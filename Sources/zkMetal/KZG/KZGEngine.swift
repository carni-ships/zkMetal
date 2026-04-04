// KZG Polynomial Commitment Engine
// Composes MSM + polynomial operations for commit() and open().
// SRS (Structured Reference String) is provided externally.

import Foundation
import Metal
import NeonFieldOps

public struct KZGProof {
    public let evaluation: Fr      // p(z)
    public let witness: PointProjective  // [q(s)] where q(x) = (p(x) - p(z)) / (x - z)

    public init(evaluation: Fr, witness: PointProjective) {
        self.evaluation = evaluation
        self.witness = witness
    }
}

public class KZGEngine {
    public static let version = Versions.kzg
    public let msmEngine: MetalMSM
    public let polyEngine: PolyEngine

    /// SRS points: [G, sG, s^2 G, ..., s^(d-1) G] in affine form
    public private(set) var srs: [PointAffine]

    public init(srs: [PointAffine]) throws {
        self.msmEngine = try MetalMSM()
        self.polyEngine = try PolyEngine()
        self.srs = srs
    }

    /// Generate a toy SRS for testing (NOT secure — uses known secret).
    /// secret: the toxic waste scalar s
    /// size: number of SRS points (max polynomial degree + 1)
    /// generator: base point G in affine form
    public static func generateTestSRS(secret: [UInt32], size: Int, generator: PointAffine) -> [PointAffine] {
        let gProj = pointFromAffine(generator)
        var points = [PointProjective]()
        points.reserveCapacity(size)
        var sPow = Fr.one  // s^0 = 1
        let sFr = frFromLimbs(secret)

        for _ in 0..<size {
            points.append(cPointScalarMul(gProj, sPow))
            sPow = frMul(sPow, sFr)
        }
        return batchToAffine(points)
    }

    /// Commit to a polynomial: C = MSM(SRS[0..deg], coefficients)
    public func commit(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n <= srs.count else {
            throw MSMError.invalidInput
        }
        let srsSlice = Array(srs.prefix(n))
        let scalars = coeffs.map { frToLimbs($0) }
        return try msmEngine.msm(points: srsSlice, scalars: scalars)
    }

    /// Open a polynomial at point z: compute evaluation and witness proof.
    /// Returns (p(z), proof_point) where proof_point = MSM(SRS, quotient_coeffs)
    /// and quotient = (p(x) - p(z)) / (x - z).
    public func open(_ coeffs: [Fr], at z: Fr) throws -> KZGProof {
        // Evaluate p(z)
        let evals = try polyEngine.evaluate(coeffs, at: [z])
        let pz = evals[0]

        // Compute quotient polynomial q(x) = (p(x) - p(z)) / (x - z)
        // Using GPU parallel synthetic division for large polynomials.
        let n = coeffs.count
        guard n >= 2 else {
            // Constant polynomial: quotient is zero, witness is identity
            return KZGProof(evaluation: pz, witness: pointIdentity())
        }

        let quotient = try polyEngine.divideByLinear(coeffs, z: z)

        // Witness = MSM(SRS[0..n-1], quotient)
        let srsSlice = Array(srs.prefix(n - 1))
        let scalars = quotient.map { frToLimbs($0) }
        let witness = try msmEngine.msm(points: srsSlice, scalars: scalars)

        return KZGProof(evaluation: pz, witness: witness)
    }
    // MARK: - Batch openings (same point)

    /// Batch KZG proof for multiple polynomials
    public struct BatchProof {
        public let commitments: [PointProjective]
        public let proof: PointProjective
        public let evaluations: [Fr]
    }

    /// Batch open N polynomials at the same point z.
    /// Uses random linear combination with challenge gamma to reduce N openings to 1 MSM.
    /// Algorithm:
    ///   h(x) = p_0(x) + gamma * p_1(x) + gamma^2 * p_2(x) + ...
    ///   y = y_0 + gamma * y_1 + gamma^2 * y_2 + ...
    ///   q(x) = (h(x) - y) / (x - z)
    ///   proof = commit(q)
    public func batchOpen(polynomials: [[Fr]], point: Fr, gamma: Fr) throws -> BatchProof {
        guard !polynomials.isEmpty else { throw MSMError.invalidInput }

        let n = polynomials.count

        // 1. Compute individual commitments and evaluations
        var commitments = [PointProjective]()
        commitments.reserveCapacity(n)
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)

        for poly in polynomials {
            commitments.append(try commit(poly))
            let evals = try polyEngine.evaluate(poly, at: [point])
            evaluations.append(evals[0])
        }

        // 2. Compute combined polynomial h(x) = sum_i gamma^i * p_i(x)
        let maxDeg = polynomials.map { $0.count }.max()!
        var combined = [Fr](repeating: Fr.zero, count: maxDeg)
        var gammaPow = Fr.one
        for i in 0..<n {
            let poly = polynomials[i]
            for j in 0..<poly.count {
                combined[j] = frAdd(combined[j], frMul(gammaPow, poly[j]))
            }
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // 3. Compute combined evaluation y = sum_i gamma^i * y_i
        var combinedEval = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedEval = frAdd(combinedEval, frMul(gammaPow, evaluations[i]))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // 4. Compute quotient q(x) = (h(x) - y) / (x - z) via synthetic division
        //    h(x) - y: subtract combinedEval from constant term
        combined[0] = frSub(combined[0], combinedEval)

        let deg = combined.count
        guard deg >= 2 else {
            // Combined polynomial minus eval is zero (or constant), proof is identity
            return BatchProof(commitments: commitments, proof: pointIdentity(), evaluations: evaluations)
        }

        let quotient = try polyEngine.divideByLinear(combined, z: point)

        // 5. Single MSM for the proof
        let srsSlice = Array(srs.prefix(deg - 1))
        let scalars = quotient.map { frToLimbs($0) }
        let proof = try msmEngine.msm(points: srsSlice, scalars: scalars)

        return BatchProof(commitments: commitments, proof: proof, evaluations: evaluations)
    }

    /// Verify a batch opening at the same point using the SRS secret.
    /// In production, this would use a pairing check:
    ///   e(C_combined - [y]G, H2) == e(proof, [s]H2 - [z]H2)
    /// Since we don't have a pairing engine, we verify using the known SRS secret:
    ///   C_combined == [y]*G + [s - z]*proof
    /// where we compute [s-z]*proof via scalar multiplication.
    /// The `srsSecret` parameter is the toxic waste scalar used to generate the SRS.
    public func batchVerify(commitments: [PointProjective], point: Fr, evaluations: [Fr],
                            proof: PointProjective, gamma: Fr, srsSecret: Fr) -> Bool {
        guard commitments.count == evaluations.count, !commitments.isEmpty else { return false }

        let n = commitments.count

        // 1. Compute combined commitment: C = sum_i gamma^i * C_i
        var combinedCommitment = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            combinedCommitment = pointAdd(combinedCommitment, cPointScalarMul(commitments[i], gammaPow))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // 2. Compute combined evaluation: y = sum_i gamma^i * y_i
        var combinedEval = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedEval = frAdd(combinedEval, frMul(gammaPow, evaluations[i]))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // 3. Verify: C_combined == [y]*G + [s - z]*proof
        let g1 = pointFromAffine(srs[0])

        // [y]*G
        let yG = cPointScalarMul(g1, combinedEval)

        // [s - z]*proof
        let sMz = frSub(srsSecret, point)
        let szProof = cPointScalarMul(proof, sMz)

        // expected = [y]*G + [s-z]*proof
        let expected = pointAdd(yG, szProof)

        // Compare: C_combined == expected (compare affine coordinates)
        let cAffine = batchToAffine([combinedCommitment])
        let eAffine = batchToAffine([expected])
        return fpToInt(cAffine[0].x) == fpToInt(eAffine[0].x) &&
               fpToInt(cAffine[0].y) == fpToInt(eAffine[0].y)
    }

    /// Verify batch opening by re-computing the proof from scratch.
    /// This is an O(N*deg) check suitable for testing correctness.
    public func verifyBatchByReopen(polynomials: [[Fr]], point: Fr, evaluations: [Fr],
                                     proof: PointProjective, gamma: Fr) throws -> Bool {
        // Re-open and compare proof points
        let recomputed = try batchOpen(polynomials: polynomials, point: point, gamma: gamma)

        // Check evaluations match
        guard recomputed.evaluations.count == evaluations.count else { return false }
        for i in 0..<evaluations.count {
            let a = frToInt(recomputed.evaluations[i])
            let b = frToInt(evaluations[i])
            if a != b { return false }
        }

        // Check proof point matches (compare affine coordinates)
        let proofAffine = batchToAffine([proof])
        let recomputedAffine = batchToAffine([recomputed.proof])
        let px = fpToInt(proofAffine[0].x)
        let py = fpToInt(proofAffine[0].y)
        let rx = fpToInt(recomputedAffine[0].x)
        let ry = fpToInt(recomputedAffine[0].y)
        return px == rx && py == ry
    }

    // MARK: - Batch openings (multi-point)

    /// Multi-point batch proof structure
    public struct MultiPointBatchProof {
        public let commitments: [PointProjective]
        public let proof: PointProjective
        public let evaluations: [Fr]
        public let points: [Fr]
    }

    /// Batch open N polynomials at potentially different points z_i.
    /// Uses the technique: for each polynomial, compute quotient q_i(x) = (p_i(x) - y_i) / (x - z_i),
    /// then combine: proof = commit(sum_i gamma^i * q_i(x))
    public func batchOpenMultiPoint(polynomials: [[Fr]], points: [Fr], gamma: Fr) throws -> MultiPointBatchProof {
        guard !polynomials.isEmpty, polynomials.count == points.count else { throw MSMError.invalidInput }

        let n = polynomials.count

        // 1. Compute individual commitments and evaluations
        var commitments = [PointProjective]()
        commitments.reserveCapacity(n)
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)

        for i in 0..<n {
            commitments.append(try commit(polynomials[i]))
            let evals = try polyEngine.evaluate(polynomials[i], at: [points[i]])
            evaluations.append(evals[0])
        }

        // 2. For each polynomial, compute quotient q_i(x) = (p_i(x) - y_i) / (x - z_i)
        //    Then combine: h(x) = sum_i gamma^i * q_i(x)
        var maxQuotientDeg = 0
        for poly in polynomials {
            if poly.count - 1 > maxQuotientDeg { maxQuotientDeg = poly.count - 1 }
        }

        var combined = [Fr](repeating: Fr.zero, count: maxQuotientDeg)
        var gammaPow = Fr.one

        for i in 0..<n {
            let poly = polynomials[i]
            let deg = poly.count
            guard deg >= 2 else {
                if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
                continue
            }

            // Subtract evaluation from constant term
            var shifted = poly
            shifted[0] = frSub(shifted[0], evaluations[i])

            // Synthetic division by (x - z_i) using GPU parallel division
            let quotient = try polyEngine.divideByLinear(shifted, z: points[i])

            // Accumulate gamma^i * q_i(x)
            for j in 0..<quotient.count {
                if j < combined.count {
                    combined[j] = frAdd(combined[j], frMul(gammaPow, quotient[j]))
                }
            }

            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // 3. Single MSM: proof = commit(combined quotient)
        let proof: PointProjective
        if combined.isEmpty || combined.allSatisfy({ frToInt($0) == frToInt(Fr.zero) }) {
            proof = pointIdentity()
        } else {
            let srsSlice = Array(srs.prefix(combined.count))
            let scalars = combined.map { frToLimbs($0) }
            proof = try msmEngine.msm(points: srsSlice, scalars: scalars)
        }

        return MultiPointBatchProof(commitments: commitments, proof: proof,
                                     evaluations: evaluations, points: points)
    }

    /// Verify multi-point batch opening by re-computing proof from scratch.
    public func verifyMultiPointByReopen(polynomials: [[Fr]], points: [Fr], evaluations: [Fr],
                                          proof: PointProjective, gamma: Fr) throws -> Bool {
        let recomputed = try batchOpenMultiPoint(polynomials: polynomials, points: points, gamma: gamma)

        // Check evaluations
        guard recomputed.evaluations.count == evaluations.count else { return false }
        for i in 0..<evaluations.count {
            if frToInt(recomputed.evaluations[i]) != frToInt(evaluations[i]) { return false }
        }

        // Check proof point
        let proofAffine = batchToAffine([proof])
        let recomputedAffine = batchToAffine([recomputed.proof])
        return fpToInt(proofAffine[0].x) == fpToInt(recomputedAffine[0].x) &&
               fpToInt(proofAffine[0].y) == fpToInt(recomputedAffine[0].y)
    }
}

// MARK: - Fr <-> [UInt32] limb conversion helpers

/// Convert Fr (Montgomery form) to raw [UInt32] limbs (8 limbs, little-endian)
public func frToLimbs(_ a: Fr) -> [UInt32] {
    let raw = frToInt(a)  // [UInt64] in standard form
    return [
        UInt32(raw[0] & 0xFFFFFFFF), UInt32(raw[0] >> 32),
        UInt32(raw[1] & 0xFFFFFFFF), UInt32(raw[1] >> 32),
        UInt32(raw[2] & 0xFFFFFFFF), UInt32(raw[2] >> 32),
        UInt32(raw[3] & 0xFFFFFFFF), UInt32(raw[3] >> 32),
    ]
}

/// Convert raw [UInt32] limbs (8 limbs, little-endian) to Fr (Montgomery form)
public func frFromLimbs(_ limbs: [UInt32]) -> Fr {
    let raw: [UInt64] = [
        UInt64(limbs[0]) | (UInt64(limbs[1]) << 32),
        UInt64(limbs[2]) | (UInt64(limbs[3]) << 32),
        UInt64(limbs[4]) | (UInt64(limbs[5]) << 32),
        UInt64(limbs[6]) | (UInt64(limbs[7]) << 32),
    ]
    let a = Fr.from64(raw)
    return frMul(a, Fr.from64(Fr.R2_MOD_R))  // Convert to Montgomery form
}
