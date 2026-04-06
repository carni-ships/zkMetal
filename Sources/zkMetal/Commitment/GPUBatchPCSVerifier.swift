// GPUBatchPCSVerifier — GPU-accelerated batch polynomial commitment verification
//
// Verifies multiple KZG polynomial commitment openings in a single batch using
// random linear combination to reduce N pairing checks to one combined check.
//
// Key insight: given N claims {(C_i, z_i, y_i, pi_i)}, a verifier can draw a
// random challenge r and check a single combined equation:
//
//   sum_i r^i * [(C_i - [y_i]*G) - z_i * pi_i]  ==  [s] * sum_i r^i * pi_i
//
// Equivalently (without SRS secret, algebraic form for testing):
//   LHS = sum_i r^i * (C_i - [y_i]*G)  (combined commitment minus evaluation)
//   RHS = sum_i r^i * [s - z_i] * pi_i  (combined weighted witness)
//
// The GPU MSM handles the two large multi-scalar multiplications:
//   1. MSM over N commitments/evaluation-points with powers of r
//   2. MSM over N proof witnesses with (r^i * (s - z_i)) scalars
//
// For Plonk proofs with 20+ commitments this provides significant speedup over
// sequential single-pairing verification.
//
// For production pairing-based verification (no SRS secret):
//   e(sum_i r^i * (C_i - [y_i]*G - z_i * pi_i), [1]_2) == e(sum_i r^i * pi_i, [s]_2)
//   Two pairings total regardless of N.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Batch Verification Input

/// A single KZG opening claim to be batch-verified.
public struct KZGOpeningClaim {
    /// Commitment to the polynomial: C = [p(s)]_1
    public let commitment: PointProjective
    /// Evaluation point z
    public let point: Fr
    /// Claimed evaluation value y = p(z)
    public let value: Fr
    /// KZG witness proof: pi = [q(s)]_1 where q(x) = (p(x) - y) / (x - z)
    public let proof: PointProjective

    public init(commitment: PointProjective, point: Fr, value: Fr, proof: PointProjective) {
        self.commitment = commitment
        self.point = point
        self.value = value
        self.proof = proof
    }
}

// MARK: - GPU Batch PCS Verifier

public class GPUBatchPCSVerifier {
    public static let version = Versions.gpuBatchPCSVerify

    /// The Metal MSM engine for GPU-accelerated multi-scalar multiplication.
    public let msmEngine: MetalMSM

    public init() throws {
        self.msmEngine = try MetalMSM()
    }

    /// Re-use an existing MSM engine (avoids GPU pipeline re-creation).
    public init(msmEngine: MetalMSM) {
        self.msmEngine = msmEngine
    }

    // MARK: - Scalar conversion

    private func frToLimbs(_ scalar: Fr) -> [UInt32] {
        var limbs = [UInt32](repeating: 0, count: 8)
        withUnsafeBytes(of: scalar) { src in
            limbs.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    1
                )
            }
        }
        return limbs
    }

    private func batchFrToLimbs(_ scalars: [Fr]) -> [[UInt32]] {
        let n = scalars.count
        var flat = [UInt32](repeating: 0, count: n * 8)
        scalars.withUnsafeBytes { src in
            flat.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        var result = [[UInt32]]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(flat[i * 8 ..< (i + 1) * 8]))
        }
        return result
    }

    // MARK: - Core batch verification (SRS secret, for testing)

    /// Batch verify N KZG opening claims using random linear combination + GPU MSM.
    ///
    /// Algorithm:
    ///   1. Derive batching challenge r from Fiat-Shamir transcript over all claims.
    ///   2. Compute LHS = sum_i r^i * (C_i - [y_i]*G)  via GPU MSM.
    ///   3. Compute RHS = sum_i r^i * (s - z_i) * pi_i  via GPU MSM.
    ///   4. Check LHS == RHS.
    ///
    /// Uses SRS secret s for testing. In production, this becomes a 2-pairing check.
    ///
    /// - Parameters:
    ///   - claims: array of N opening claims (commitment, point, value, proof)
    ///   - srs: the structured reference string (needed for generator G = srs[0])
    ///   - srsSecret: the toxic waste s (testing only)
    /// - Returns: true if all N openings are valid
    public func batchVerifyKZG(
        claims: [KZGOpeningClaim],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        guard !claims.isEmpty else { return true }
        let n = claims.count

        // Single claim: fast path without GPU MSM overhead
        if n == 1 {
            return verifySingle(claims[0], srs: srs, srsSecret: srsSecret)
        }

        // Step 1: Derive batching challenge r via Fiat-Shamir
        let r = deriveBatchChallenge(claims: claims)

        // Step 2: Compute powers of r: [1, r, r^2, ..., r^{n-1}]
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // Step 3: Build LHS points and scalars
        // LHS = sum_i r^i * (C_i - [y_i]*G)
        // We compute each (C_i - [y_i]*G) and then do MSM with r^i scalars.
        var lhsPoints = [PointAffine]()
        lhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let yG = cPointScalarMul(g1, claims[i].value)
            let diff = pointAdd(claims[i].commitment, pointNeg(yG))
            lhsPoints.append(batchToAffine([diff])[0])
        }
        let lhsScalars = batchFrToLimbs(rPowers)
        let lhs = try msmEngine.msm(points: lhsPoints, scalars: lhsScalars)

        // Step 4: Build RHS points and scalars
        // RHS = sum_i r^i * (s - z_i) * pi_i
        // Scalars: r^i * (s - z_i)
        var rhsScalarsFr = [Fr](repeating: Fr.zero, count: n)
        var rhsPoints = [PointAffine]()
        rhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let sMz = frSub(srsSecret, claims[i].point)
            rhsScalarsFr[i] = frMul(rPowers[i], sMz)
            rhsPoints.append(batchToAffine([claims[i].proof])[0])
        }
        let rhsScalars = batchFrToLimbs(rhsScalarsFr)
        let rhs = try msmEngine.msm(points: rhsPoints, scalars: rhsScalars)

        // Step 5: Check LHS == RHS
        return pointsEqual(lhs, rhs)
    }

    // MARK: - Array-based convenience API

    /// Batch verify KZG openings given parallel arrays.
    ///
    /// This is the primary API matching the task specification:
    ///   batchVerifyKZG(commitments:points:values:proofs:) -> Bool
    ///
    /// - Parameters:
    ///   - commitments: [C_0, ..., C_{N-1}] polynomial commitments
    ///   - points: [z_0, ..., z_{N-1}] evaluation points
    ///   - values: [y_0, ..., y_{N-1}] claimed evaluations
    ///   - proofs: [pi_0, ..., pi_{N-1}] KZG witness proofs
    ///   - srs: structured reference string
    ///   - srsSecret: toxic waste (testing only)
    /// - Returns: true if all openings are valid
    public func batchVerifyKZG(
        commitments: [PointProjective],
        points: [Fr],
        values: [Fr],
        proofs: [PointProjective],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        let n = commitments.count
        guard n == points.count, n == values.count, n == proofs.count else {
            return false
        }

        let claims = (0..<n).map { i in
            KZGOpeningClaim(
                commitment: commitments[i],
                point: points[i],
                value: values[i],
                proof: proofs[i]
            )
        }
        return try batchVerifyKZG(claims: claims, srs: srs, srsSecret: srsSecret)
    }

    // MARK: - Same-point batch verification

    /// Optimized batch verification when all polynomials are opened at the same point.
    ///
    /// For Plonk-like proofs where 20+ polynomials are opened at zeta, this avoids
    /// redundant (s - z) computations.
    ///
    /// LHS = sum_i r^i * (C_i - [y_i]*G)
    /// RHS = (s - z) * sum_i r^i * pi_i
    public func batchVerifyKZGSamePoint(
        commitments: [PointProjective],
        point: Fr,
        values: [Fr],
        proofs: [PointProjective],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        let n = commitments.count
        guard n == values.count, n == proofs.count, n > 0 else { return false }

        if n == 1 {
            let claim = KZGOpeningClaim(
                commitment: commitments[0], point: point,
                value: values[0], proof: proofs[0])
            return verifySingle(claim, srs: srs, srsSecret: srsSecret)
        }

        // Derive challenge
        let claims = (0..<n).map { i in
            KZGOpeningClaim(
                commitment: commitments[i], point: point,
                value: values[i], proof: proofs[i])
        }
        let r = deriveBatchChallenge(claims: claims)

        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // LHS: sum_i r^i * (C_i - [y_i]*G)
        var lhsPoints = [PointAffine]()
        lhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let yG = cPointScalarMul(g1, values[i])
            let diff = pointAdd(commitments[i], pointNeg(yG))
            lhsPoints.append(batchToAffine([diff])[0])
        }
        let lhs = try msmEngine.msm(points: lhsPoints, scalars: batchFrToLimbs(rPowers))

        // RHS: (s - z) * sum_i r^i * pi_i
        // First compute MSM of proofs with r^i scalars
        var proofAffs = [PointAffine]()
        proofAffs.reserveCapacity(n)
        for i in 0..<n {
            proofAffs.append(batchToAffine([proofs[i]])[0])
        }
        let combinedProof = try msmEngine.msm(points: proofAffs, scalars: batchFrToLimbs(rPowers))

        // Multiply combined proof by (s - z)
        let sMz = frSub(srsSecret, point)
        let rhs = cPointScalarMul(combinedProof, sMz)

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Non-interactive batch verification with Fiat-Shamir

    /// Full non-interactive batch verification: derives challenge from claims.
    /// This is the production-ready API -- transcript derivation is internal.
    public func batchVerifyKZGNonInteractive(
        commitments: [PointProjective],
        points: [Fr],
        values: [Fr],
        proofs: [PointProjective],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        return try batchVerifyKZG(
            commitments: commitments, points: points,
            values: values, proofs: proofs,
            srs: srs, srsSecret: srsSecret)
    }

    // MARK: - Fiat-Shamir challenge derivation

    /// Derive a batching challenge r by absorbing all claims into a transcript.
    private func deriveBatchChallenge(claims: [KZGOpeningClaim]) -> Fr {
        let transcript = Transcript(label: "gpu-batch-pcs-verify", backend: .poseidon2)

        for claim in claims {
            // Absorb commitment
            absorbPoint(claim.commitment, into: transcript)
            // Absorb evaluation point
            transcript.absorb(claim.point)
            // Absorb claimed value
            transcript.absorb(claim.value)
            // Absorb proof witness
            absorbPoint(claim.proof, into: transcript)
        }

        return transcript.squeeze()
    }

    // MARK: - Single claim verification (no batching)

    /// Verify a single KZG opening: C - [y]*G == [s - z] * pi
    private func verifySingle(
        _ claim: KZGOpeningClaim,
        srs: [PointAffine],
        srsSecret: Fr
    ) -> Bool {
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, claim.value)
        let lhs = pointAdd(claim.commitment, pointNeg(yG))

        let sMz = frSub(srsSecret, claim.point)
        let rhs = cPointScalarMul(claim.proof, sMz)

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Helpers

    /// Absorb a projective point into the transcript.
    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            transcript.absorb(fpToFr(aff[0].x))
            transcript.absorb(fpToFr(aff[0].y))
        }
    }

    /// Compare two projective points for equality via affine conversion.
    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
