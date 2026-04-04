// Batch KZG Proof Verifier
// Reduces N independent KZG opening verifications from N pairings to 2 MSMs + 2 pairings.
// Uses random linear combination for soundness: verifier samples r_1,...,r_N,
// computes aggregated verification equation Sum_i r_i * V_i = 0.
//
// For rollup sequencers verifying hundreds of proofs per block, this gives
// near-linear speedup: 2 MSMs + 2 pairings instead of N pairings.

import Foundation

/// A single KZG opening proof to be batch-verified.
public struct VerificationItem {
    /// KZG commitment C_i to the polynomial
    public let commitment: PointProjective
    /// Evaluation point z_i
    public let point: Fr
    /// Claimed evaluation value v_i = p(z_i)
    public let value: Fr
    /// KZG opening proof pi_i = [(p(x) - v_i) / (x - z_i)]_1
    public let proof: PointProjective

    public init(commitment: PointProjective, point: Fr, value: Fr, proof: PointProjective) {
        self.commitment = commitment
        self.point = point
        self.value = value
        self.proof = proof
    }
}

/// Batch verifier for KZG opening proofs.
///
/// Given N proofs with verification equations:
///   e(C_i - v_i*G, H) = e(pi_i, tau*H - z_i*H)
///
/// Batched check (random linear combination):
///   e(Sum r_i*(C_i - v_i*G), H) = e(Sum r_i*pi_i, tau*H) * e(-Sum r_i*z_i*pi_i, H)
///
/// This reduces N pairing checks to 2 MSMs of size N + 2 pairings.
///
/// For testing without pairings, we support SRS-secret-based verification:
///   Sum r_i*(C_i - v_i*G) == (s)*Sum(r_i*pi_i) - Sum(r_i*z_i*pi_i)
public class BatchVerifier {
    public static let version = Versions.batchVerify
    public let msmEngine: MetalMSM

    public init() throws {
        self.msmEngine = try MetalMSM()
    }

    /// Initialize with an existing MSM engine (shares GPU resources).
    public init(msmEngine: MetalMSM) {
        self.msmEngine = msmEngine
    }

    // MARK: - Batch KZG Verification (SRS-secret based, for testing)

    /// Batch verify N KZG opening proofs using the SRS secret.
    /// This avoids pairings by using the known secret s from the SRS.
    ///
    /// Verification equation (per proof):
    ///   C_i - v_i*G = (s - z_i) * pi_i
    ///
    /// Batched with random scalars r_i:
    ///   Sum r_i*(C_i - v_i*G) = Sum r_i*(s - z_i)*pi_i
    ///
    /// LHS = Sum r_i*C_i - (Sum r_i*v_i)*G
    /// RHS = s * Sum(r_i*pi_i) - Sum(r_i*z_i*pi_i)
    ///
    /// Check: LHS == RHS
    public func batchVerifyKZG(
        items: [VerificationItem],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        guard !items.isEmpty else { return true }

        // Generate random scalars via Fiat-Shamir transcript
        let transcript = Transcript(label: "batch-kzg-verify", backend: .poseidon2)

        // Absorb all commitments, points, values, proofs into transcript
        for item in items {
            let cAff = batchToAffine([item.commitment])
            transcript.absorb(fpToFr(cAff[0].x))
            transcript.absorb(fpToFr(cAff[0].y))
            transcript.absorb(item.point)
            transcript.absorb(item.value)
            let pAff = batchToAffine([item.proof])
            transcript.absorb(fpToFr(pAff[0].x))
            transcript.absorb(fpToFr(pAff[0].y))
        }

        let scalars = transcript.squeezeN(items.count)
        return try batchVerifyKZGWithScalars(
            items: items, scalars: scalars, srs: srs, srsSecret: srsSecret)
    }

    /// Batch verify with pre-computed random scalars (for deterministic testing).
    ///
    /// LHS = Sum r_i*C_i - (Sum r_i*v_i)*G
    /// RHS = s * Sum(r_i*pi_i) - Sum(r_i*z_i*pi_i)
    public func batchVerifyKZGWithScalars(
        items: [VerificationItem],
        scalars: [Fr],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        let n = items.count
        guard n == scalars.count, n > 0, !srs.isEmpty else { return false }

        let g = pointFromAffine(srs[0])

        // Compute LHS: Sum r_i*(C_i - v_i*G)
        var lhs = pointIdentity()
        var aggregatedEval = Fr.zero

        for i in 0..<n {
            let riCi = cPointScalarMul(items[i].commitment, scalars[i])
            lhs = pointAdd(lhs, riCi)
            aggregatedEval = frAdd(aggregatedEval, frMul(scalars[i], items[i].value))
        }

        let evalG = cPointScalarMul(g, aggregatedEval)
        lhs = pointAdd(lhs, pointNeg(evalG))

        // Compute RHS: s * Sum(r_i*pi_i) - Sum(r_i*z_i*pi_i)
        var sumRiPi = pointIdentity()
        var sumRiZiPi = pointIdentity()

        for i in 0..<n {
            let riPi = cPointScalarMul(items[i].proof, scalars[i])
            sumRiPi = pointAdd(sumRiPi, riPi)

            let riZi = frMul(scalars[i], items[i].point)
            let riZiPi = cPointScalarMul(items[i].proof, riZi)
            sumRiZiPi = pointAdd(sumRiZiPi, riZiPi)
        }

        let rhs = pointAdd(cPointScalarMul(sumRiPi, srsSecret), pointNeg(sumRiZiPi))

        // Compare LHS == RHS via affine coordinates
        if pointIsIdentity(lhs) && pointIsIdentity(rhs) { return true }
        if pointIsIdentity(lhs) || pointIsIdentity(rhs) { return false }

        let lhsAff = batchToAffine([lhs])
        let rhsAff = batchToAffine([rhs])
        return fpToInt(lhsAff[0].x) == fpToInt(rhsAff[0].x) &&
               fpToInt(lhsAff[0].y) == fpToInt(rhsAff[0].y)
    }

    // MARK: - GPU-accelerated batch verification

    /// GPU-accelerated batch verify using MSM for the random linear combinations.
    /// For large N, this is significantly faster than sequential scalar multiplications.
    ///
    /// Computes:
    ///   A = MSM(C_1,...,C_N; r_1,...,r_N)   -- aggregated commitment
    ///   B = MSM(pi_1,...,pi_N; r_1,...,r_N)  -- aggregated proof
    ///   D = MSM(pi_1,...,pi_N; r_1*z_1,...,r_N*z_N)  -- weighted proof
    ///
    /// Check: A - (Sum r_i*v_i)*G == s*B - D
    public func batchVerifyKZGWithMSM(
        items: [VerificationItem],
        scalars: [Fr],
        srs: [PointAffine],
        srsSecret: Fr
    ) throws -> Bool {
        let n = items.count
        guard n == scalars.count, n > 0, !srs.isEmpty else { return false }

        // Convert commitments and proofs to affine for MSM
        let commitments = batchToAffine(items.map { $0.commitment })
        let proofs = batchToAffine(items.map { $0.proof })

        let scalarLimbs = scalars.map { frToLimbs($0) }

        // A = MSM(commitments, r_i)
        let aggCommitment = try msmEngine.msm(points: commitments, scalars: scalarLimbs)

        // B = MSM(proofs, r_i)
        let aggProof = try msmEngine.msm(points: proofs, scalars: scalarLimbs)

        // D = MSM(proofs, r_i*z_i)
        var weightedScalars = [[UInt32]]()
        weightedScalars.reserveCapacity(n)
        var aggregatedEval = Fr.zero
        for i in 0..<n {
            let riZi = frMul(scalars[i], items[i].point)
            weightedScalars.append(frToLimbs(riZi))
            aggregatedEval = frAdd(aggregatedEval, frMul(scalars[i], items[i].value))
        }
        let weightedProof = try msmEngine.msm(points: proofs, scalars: weightedScalars)

        // LHS = A - (Sum r_i*v_i)*G
        let g = pointFromAffine(srs[0])
        let evalG = cPointScalarMul(g, aggregatedEval)
        let lhs = pointAdd(aggCommitment, pointNeg(evalG))

        // RHS = s*B - D
        let rhs = pointAdd(cPointScalarMul(aggProof, srsSecret), pointNeg(weightedProof))

        if pointIsIdentity(lhs) && pointIsIdentity(rhs) { return true }
        if pointIsIdentity(lhs) || pointIsIdentity(rhs) { return false }

        let lhsAff = batchToAffine([lhs])
        let rhsAff = batchToAffine([rhs])
        return fpToInt(lhsAff[0].x) == fpToInt(rhsAff[0].x) &&
               fpToInt(lhsAff[0].y) == fpToInt(rhsAff[0].y)
    }
}
