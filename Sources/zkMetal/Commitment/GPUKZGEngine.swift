// GPUKZGEngine — GPU-accelerated KZG polynomial commitment scheme
//
// Forces all MSM operations through the Metal GPU pipeline (no CPU Pippenger fallback).
// Provides batch opening via random linear combination and verification.
//
// Key differences from KZGEngine:
//   - commit() always dispatches to GPU MSM regardless of polynomial size
//   - batchOpen() fuses polynomial combination on CPU then does a single GPU MSM
//   - batchVerify() uses algebraic SRS-secret check (pairing-free, for testing)
//
// For production pairing-based verification, see KZGBatchOpeningVerifier.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Batch proof structure

/// Proof for a batch KZG opening of N polynomials at a single point.
public struct GPUBatchProof {
    /// Commitments to each polynomial [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Evaluations [p_0(z), ..., p_{N-1}(z)]
    public let evaluations: [Fr]
    /// Combined witness point: commit(sum_i gamma^i * q_i(x))
    public let witness: PointProjective
    /// The evaluation point z
    public let point: Fr
    /// The batching challenge gamma
    public let gamma: Fr

    public init(commitments: [PointProjective], evaluations: [Fr],
                witness: PointProjective, point: Fr, gamma: Fr) {
        self.commitments = commitments
        self.evaluations = evaluations
        self.witness = witness
        self.point = point
        self.gamma = gamma
    }
}

// MARK: - GPU KZG Engine

public class GPUKZGEngine {
    public static let version = Versions.gpuKZG

    /// The underlying Metal MSM engine (shared, avoids re-creating pipelines)
    public let msmEngine: MetalMSM

    /// SRS points: [G, sG, s^2 G, ..., s^(d-1) G] in affine form
    public let srs: [PointAffine]

    /// Cached SRS prefix slices
    private var srsSliceCache: [Int: [PointAffine]] = [:]

    public init(srs: [PointAffine]) throws {
        self.msmEngine = try MetalMSM()
        self.srs = srs
    }

    /// Re-use an existing MSM engine (avoids GPU pipeline re-creation).
    public init(srs: [PointAffine], msmEngine: MetalMSM) {
        self.msmEngine = msmEngine
        self.srs = srs
    }

    // MARK: - SRS helpers

    private func srsPrefix(_ n: Int) -> [PointAffine] {
        if let cached = srsSliceCache[n] { return cached }
        let slice = Array(srs.prefix(n))
        srsSliceCache[n] = slice
        return slice
    }

    // MARK: - Scalar conversion (batch Fr -> flat UInt32 limbs)

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

    // MARK: - Polynomial helpers (CPU, using C CIOS)

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

    private func cSyntheticDiv(_ poly: [Fr], root: Fr) -> [Fr] {
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

    // MARK: - Commit (always GPU MSM)

    /// Commit to a polynomial using GPU MSM.
    /// C = MSM(SRS[0..n], coefficients)
    ///
    /// Unlike KZGEngine.commit(), this always dispatches to the Metal GPU pipeline
    /// regardless of polynomial size. For very small polynomials (< 64 coefficients),
    /// the GPU dispatch overhead may exceed the computation time; use KZGEngine for
    /// latency-sensitive small-polynomial commits.
    public func commit(_ coeffs: [Fr]) throws -> PointProjective {
        let n = coeffs.count
        guard n > 0 else { return pointIdentity() }
        guard n <= srs.count else { throw MSMError.invalidInput }

        let pts = srsPrefix(n)
        let scalars = batchFrToLimbs(coeffs)
        return try msmEngine.msm(points: pts, scalars: scalars)
    }

    /// Batch commit to multiple polynomials using GPU multiMSM.
    /// All polynomials sharing the same degree use a single GPU point upload.
    public func batchCommit(_ polynomials: [[Fr]]) throws -> [PointProjective] {
        guard !polynomials.isEmpty else { return [] }
        if polynomials.count == 1 { return [try commit(polynomials[0])] }

        // Group by length for shared-point multiMSM
        let n = polynomials[0].count
        let allSameLength = polynomials.allSatisfy { $0.count == n }

        if !allSameLength {
            return try polynomials.map { try commit($0) }
        }

        guard n <= srs.count else { throw MSMError.invalidInput }
        let pts = srsPrefix(n)
        let scalarSets = polynomials.map { batchFrToLimbs($0) }
        return try multiMSM(engine: msmEngine, points: pts, scalarSets: scalarSets)
    }

    // MARK: - Single open (always GPU MSM)

    /// Open polynomial at point z: returns evaluation p(z) and witness proof.
    public func open(_ coeffs: [Fr], at z: Fr) throws -> KZGProof {
        let n = coeffs.count
        guard n >= 1 else { throw MSMError.invalidInput }

        if n == 1 {
            return KZGProof(evaluation: coeffs[0], witness: pointIdentity())
        }

        // Fused eval + division
        var pz = Fr.zero
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &pz) { eBuf in
                    quotient.withUnsafeMutableBytes { qBuf in
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

        // Always GPU MSM for the witness
        let pts = srsPrefix(n - 1)
        let scalars = batchFrToLimbs(quotient)
        let witness = try msmEngine.msm(points: pts, scalars: scalars)

        return KZGProof(evaluation: pz, witness: witness)
    }

    // MARK: - Batch open (random linear combination, single GPU MSM)

    /// Batch open N polynomials at a single point z.
    ///
    /// Algorithm:
    ///   1. Compute evaluations y_i = p_i(z)
    ///   2. Compute combined polynomial h(x) = sum_i gamma^i * p_i(x)
    ///   3. Compute combined evaluation y = sum_i gamma^i * y_i
    ///   4. Compute quotient q(x) = (h(x) - y) / (x - z)
    ///   5. witness = GPU MSM(SRS, q)
    ///
    /// Cost: N evaluations (CPU) + 1 polynomial combination (CPU) + 1 GPU MSM
    public func batchOpen(polys: [[Fr]], point: Fr, gamma: Fr) throws -> GPUBatchProof {
        guard !polys.isEmpty else { throw MSMError.invalidInput }

        let n = polys.count

        // Step 1: Compute commitments via GPU multiMSM
        let commitments = try batchCommit(polys)

        // Step 2: Compute evaluations
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)
        for poly in polys {
            evaluations.append(cEvaluate(poly, at: point))
        }

        // Step 3: Combine polynomials h(x) = sum_i gamma^i * p_i(x)
        let maxDeg = polys.map { $0.count }.max()!
        var combined = [Fr](repeating: Fr.zero, count: maxDeg)
        var gammaPow = Fr.one
        for i in 0..<n {
            let poly = polys[i]
            // Batch MAC: combined[j] += gammaPow * poly[j]
            poly.withUnsafeBytes { polyBuf in
                combined.withUnsafeMutableBytes { combBuf in
                    withUnsafeBytes(of: gammaPow) { gpBuf in
                        bn254_fr_batch_mac_neon(
                            combBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            polyBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(poly.count))
                    }
                }
            }
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Step 4: Combined evaluation y = sum_i gamma^i * y_i
        var combinedEval = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedEval = frAdd(combinedEval, frMul(gammaPow, evaluations[i]))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        // Step 5: Quotient q(x) = (h(x) - y) / (x - z)
        combined[0] = frSub(combined[0], combinedEval)

        let deg = combined.count
        guard deg >= 2 else {
            return GPUBatchProof(
                commitments: commitments, evaluations: evaluations,
                witness: pointIdentity(), point: point, gamma: gamma)
        }

        let quotient = cSyntheticDiv(combined, root: point)

        // Step 6: Single GPU MSM for the witness
        let pts = srsPrefix(quotient.count)
        let scalars = batchFrToLimbs(quotient)
        let witness = try msmEngine.msm(points: pts, scalars: scalars)

        return GPUBatchProof(
            commitments: commitments, evaluations: evaluations,
            witness: witness, point: point, gamma: gamma)
    }

    // MARK: - Batch verify

    /// Verify a batch opening proof using the SRS secret (testing/trusted-setup mode).
    ///
    /// Checks: C_combined == [y]*G + [s - z]*witness
    /// where C_combined = sum_i gamma^i * C_i and y = sum_i gamma^i * y_i.
    ///
    /// In production, this would be replaced by a pairing check:
    ///   e(C_combined - [y]*G, H2) == e(witness, [s]*H2 - [z]*H2)
    public func batchVerify(
        commitments: [PointProjective],
        point: Fr,
        values: [Fr],
        proof: GPUBatchProof,
        srsSecret: Fr
    ) -> Bool {
        guard commitments.count == values.count, !commitments.isEmpty else { return false }
        let n = commitments.count
        let gamma = proof.gamma

        // Combined commitment: C = sum_i gamma^i * C_i
        var combinedC = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            combinedC = pointAdd(combinedC, cPointScalarMul(commitments[i], gammaPow))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Combined evaluation: y = sum_i gamma^i * y_i
        var combinedY = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedY = frAdd(combinedY, frMul(gammaPow, values[i]))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // expected = [y]*G + [s - z]*witness
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, combinedY)
        let sMz = frSub(srsSecret, point)
        let szW = cPointScalarMul(proof.witness, sMz)
        let expected = pointAdd(yG, szW)

        return gpuPointsEqual(combinedC, expected)
    }

    /// Convenience: verify using proof's own commitments and evaluations.
    public func batchVerifyProof(_ proof: GPUBatchProof, srsSecret: Fr) -> Bool {
        return batchVerify(
            commitments: proof.commitments,
            point: proof.point,
            values: proof.evaluations,
            proof: proof,
            srsSecret: srsSecret)
    }

    // MARK: - Multi-point batch open

    /// Batch open N polynomials at N different points z_i.
    ///
    /// For each i, computes q_i(x) = (p_i(x) - y_i) / (x - z_i), then
    /// combines: witness = commit(sum_i gamma^i * q_i(x)).
    public func batchOpenMultiPoint(
        polys: [[Fr]], points: [Fr], gamma: Fr
    ) throws -> GPUBatchProof {
        guard !polys.isEmpty, polys.count == points.count else { throw MSMError.invalidInput }

        let n = polys.count
        let commitments = try batchCommit(polys)

        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)
        for i in 0..<n {
            evaluations.append(cEvaluate(polys[i], at: points[i]))
        }

        // Accumulate combined quotient: sum_i gamma^i * q_i(x)
        var maxQuotientDeg = 0
        for poly in polys {
            if poly.count - 1 > maxQuotientDeg { maxQuotientDeg = poly.count - 1 }
        }

        var combined = [Fr](repeating: Fr.zero, count: maxQuotientDeg)
        var gammaPow = Fr.one

        for i in 0..<n {
            let poly = polys[i]
            let deg = poly.count
            guard deg >= 2 else {
                if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
                continue
            }

            // p_i(x) - y_i
            var shifted = poly
            shifted[0] = frSub(shifted[0], evaluations[i])

            // q_i(x) = shifted / (x - z_i)
            let qi = cSyntheticDiv(shifted, root: points[i])

            // Accumulate gamma^i * q_i(x)
            for j in 0..<qi.count {
                if j < combined.count {
                    combined[j] = frAdd(combined[j], frMul(gammaPow, qi[j]))
                }
            }
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Single GPU MSM for the combined quotient
        let witness: PointProjective
        if combined.isEmpty || combined.allSatisfy({ frToInt($0) == frToInt(Fr.zero) }) {
            witness = pointIdentity()
        } else {
            let pts = srsPrefix(combined.count)
            let scalars = batchFrToLimbs(combined)
            witness = try msmEngine.msm(points: pts, scalars: scalars)
        }

        // Use the first point for same-point proof structure; for multi-point,
        // callers should use the evaluations and points arrays directly.
        return GPUBatchProof(
            commitments: commitments, evaluations: evaluations,
            witness: witness, point: points[0], gamma: gamma)
    }

    // MARK: - Multi-point batch verify

    /// Verify a multi-point batch opening using SRS secret.
    ///
    /// Checks that the witness encodes the correct combined quotient:
    ///   witness == sum_i gamma^i * (C_i - [y_i]*G) / (s - z_i)
    public func batchVerifyMultiPoint(
        commitments: [PointProjective],
        points: [Fr],
        values: [Fr],
        witness: PointProjective,
        gamma: Fr,
        srsSecret: Fr
    ) -> Bool {
        let n = commitments.count
        guard n == values.count, n == points.count, n > 0 else { return false }

        let g1 = pointFromAffine(srs[0])
        var expected = pointIdentity()
        var gammaPow = Fr.one

        for i in 0..<n {
            let yiG = cPointScalarMul(g1, values[i])
            let numerator = pointAdd(commitments[i], pointNeg(yiG))
            let sMzi = frSub(srsSecret, points[i])
            let sMziInv = frInverse(sMzi)
            let scalar = frMul(gammaPow, sMziInv)
            expected = pointAdd(expected, cPointScalarMul(numerator, scalar))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        return gpuPointsEqual(witness, expected)
    }

    // MARK: - Helpers

    private func gpuPointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
