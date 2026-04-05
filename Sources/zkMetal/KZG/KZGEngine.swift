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

    // MARK: - Cached SRS slices and scalar conversion

    /// Cache for SRS prefix slices to avoid re-allocation per commit/open.
    private var srsSliceCache: [Int: [PointAffine]] = [:]

    /// Get or cache an SRS prefix slice of the given size.
    private func srsPrefix(_ n: Int) -> [PointAffine] {
        if let cached = srsSliceCache[n] { return cached }
        let slice = Array(srs.prefix(n))
        srsSliceCache[n] = slice
        return slice
    }

    /// Evaluate polynomial at a single point using C Horner (avoids GPU dispatch).
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

    /// Synthetic division on CPU using C CIOS (avoids GPU dispatch).
    private func cDivideByLinear(_ coeffs: [Fr], z: Fr) -> [Fr] {
        let n = coeffs.count
        if n < 2 { return [] }
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
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

    /// Convert Fr array to flat UInt32 limbs using C batch conversion.
    /// Returns contiguous [UInt32] with 8 limbs per element, suitable for cPippengerMSMFlat.
    private func batchFrToFlatLimbs(_ coeffs: [Fr]) -> [UInt32] {
        let n = coeffs.count
        var limbs = [UInt32](repeating: 0, count: n * 8)
        coeffs.withUnsafeBytes { src in
            limbs.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        return limbs
    }

    /// Convert Fr array to [[UInt32]] limbs for GPU MSM path.
    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        let flat = batchFrToFlatLimbs(coeffs)
        let n = coeffs.count
        var result = [[UInt32]]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(flat[i*8..<(i+1)*8]))
        }
        return result
    }

    /// Commit to a polynomial: C = MSM(SRS[0..deg], coefficients)
    public func commit(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n <= srs.count else {
            throw MSMError.invalidInput
        }
        let pts = srsPrefix(n)
        // For small sizes (CPU Pippenger path), use flat limbs to avoid [[UInt32]] allocation
        if n <= 2048 {
            let flatLimbs = batchFrToFlatLimbs(coeffs)
            return cPippengerMSMFlat(points: pts, flatScalars: flatLimbs)
        }
        let scalars = batchFrToLimbs(coeffs)
        return try msmEngine.msm(points: pts, scalars: scalars)
    }

    /// Batch commit to multiple polynomials of the same length using multiMSM.
    /// All polynomials must have the same degree (same SRS prefix). Points are
    /// uploaded to the GPU once and reused across all scalar sets.
    ///
    /// Falls back to sequential commit() if polynomials have different lengths.
    public func batchCommit(_ polynomials: [[Fr]]) throws -> [PointProjective] {
        guard !polynomials.isEmpty else { return [] }
        if polynomials.count == 1 { return [try commit(polynomials[0])] }

        // Check all same length for shared-point multiMSM
        let n = polynomials[0].count
        let allSameLength = polynomials.allSatisfy { $0.count == n }

        if !allSameLength {
            // Different lengths: fall back to sequential commits
            return try polynomials.map { try commit($0) }
        }

        guard n <= srs.count else { throw MSMError.invalidInput }
        let pts = srsPrefix(n)

        // For small sizes, use CPU path (flat limbs)
        if n <= 2048 {
            return polynomials.map { coeffs in
                let flatLimbs = batchFrToFlatLimbs(coeffs)
                return cPippengerMSMFlat(points: pts, flatScalars: flatLimbs)
            }
        }

        // GPU multiMSM: shared points, multiple scalar vectors
        let scalarSets = polynomials.map { batchFrToLimbs($0) }
        return try multiMSM(engine: msmEngine, points: pts, scalarSets: scalarSets)
    }

    /// Open a polynomial at point z: compute evaluation and witness proof.
    /// Returns (p(z), proof_point) where proof_point = MSM(SRS, quotient_coeffs)
    /// and quotient = (p(x) - p(z)) / (x - z).
    public func open(_ coeffs: [Fr], at z: Fr) throws -> KZGProof {
        let n = coeffs.count

        guard n >= 2 else {
            // Constant polynomial: evaluate directly, quotient is zero
            let pz = cEvaluate(coeffs, at: z)
            return KZGProof(evaluation: pz, witness: pointIdentity())
        }

        // For small polynomials: fused eval + division in one C pass
        let pz: Fr
        let quotient: [Fr]
        if n <= 131072 {
            var eval = Fr.zero
            var q = [Fr](repeating: Fr.zero, count: n - 1)
            coeffs.withUnsafeBytes { cBuf in
                withUnsafeBytes(of: z) { zBuf in
                    withUnsafeMutableBytes(of: &eval) { eBuf in
                        q.withUnsafeMutableBytes { qBuf in
                            bn254_fr_eval_and_div(
                                cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(n),
                                zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                            )
                        }
                    }
                }
            }
            pz = eval
            quotient = q
        } else {
            pz = cEvaluate(coeffs, at: z)
            quotient = try polyEngine.divideByLinear(coeffs, z: z)
        }

        // Witness = MSM(SRS[0..n-1], quotient)
        let pts = srsPrefix(n - 1)
        let witness: PointProjective
        if (n - 1) <= 2048 {
            let flatLimbs = batchFrToFlatLimbs(quotient)
            witness = cPippengerMSMFlat(points: pts, flatScalars: flatLimbs)
        } else {
            let scalars = batchFrToLimbs(quotient)
            witness = try msmEngine.msm(points: pts, scalars: scalars)
        }

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

        // 1. Compute individual commitments (batched) and evaluations
        let commitments = try batchCommit(polynomials)
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)

        for poly in polynomials {
            evaluations.append(cEvaluate(poly, at: point))
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

        let quotient: [Fr]
        if deg <= 131072 {
            quotient = cDivideByLinear(combined, z: point)
        } else {
            quotient = try polyEngine.divideByLinear(combined, z: point)
        }

        // 5. Single MSM for the proof
        let pts = srsPrefix(deg - 1)
        let scalars = batchFrToLimbs(quotient)
        let proof = try msmEngine.msm(points: pts, scalars: scalars)

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

        // 1. Compute individual commitments (batched) and evaluations
        let commitments = try batchCommit(polynomials)
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)

        for i in 0..<n {
            evaluations.append(cEvaluate(polynomials[i], at: points[i]))
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

            // Synthetic division by (x - z_i) using C path for small, GPU for large
            let quotient: [Fr]
            if deg <= 131072 {
                quotient = cDivideByLinear(shifted, z: points[i])
            } else {
                quotient = try polyEngine.divideByLinear(shifted, z: points[i])
            }

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
            let pts = srsPrefix(combined.count)
            let scalars = batchFrToLimbs(combined)
            proof = try msmEngine.msm(points: pts, scalars: scalars)
        }

        return MultiPointBatchProof(commitments: commitments, proof: proof,
                                     evaluations: evaluations, points: points)
    }

    /// Fused multi-point batch open: single GPU pass for quotient accumulation.
    /// Replaces N sequential divideByLinear + CPU accumulation with one fused kernel.
    /// Produces the same result as batchOpenMultiPoint but with fewer GPU dispatches.
    public func batchOpenMultiPointFused(polynomials: [[Fr]], points: [Fr], gamma: Fr) throws -> MultiPointBatchProof {
        guard !polynomials.isEmpty, polynomials.count == points.count else { throw MSMError.invalidInput }

        let n = polynomials.count

        // 1. Compute individual commitments (batched) and evaluations
        let commitments = try batchCommit(polynomials)
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)

        for i in 0..<n {
            evaluations.append(cEvaluate(polynomials[i], at: points[i]))
        }

        // 2. Fused quotient accumulation: single GPU pass
        let combined = try polyEngine.fusedQuotientAccumulate(
            polynomials: polynomials,
            evaluations: evaluations,
            points: points,
            gamma: gamma
        )

        // 3. Single MSM: proof = commit(combined quotient)
        let proof: PointProjective
        if combined.isEmpty || combined.allSatisfy({ frToInt($0) == frToInt(Fr.zero) }) {
            proof = pointIdentity()
        } else {
            let pts = srsPrefix(combined.count)
            let scalars = batchFrToLimbs(combined)
            proof = try msmEngine.msm(points: pts, scalars: scalars)
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
