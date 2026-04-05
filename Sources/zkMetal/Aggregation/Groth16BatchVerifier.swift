// Batch Groth16 Verifier — amortize pairing checks across N proofs
//
// Standard Groth16 verification: e(A, B) = e(alpha, beta) * e(L, gamma) * e(C, delta)
//
// Same-circuit batch (shared VK): pick random r_1,...,r_N, combine:
//   e(Sum r_i*A_i, B) * e(Sum r_i*C_i, -delta) * e(Sum r_i*L_i, -gamma) = e(alpha, beta)^(Sum r_i)
//
// This reduces N * 4 pairings to 4 pairings + N G1 scalar muls (MSMs are GPU-accelerated).
// For rollup batch verification: verify 100 proofs at ~4x the cost of 1 proof.
//
// Cross-circuit batch: different VKs have different B_i points, so we group by VK
// and batch within each group.

import Foundation

// MARK: - Groth16 Batch Verification

public struct Groth16BatchVerifier {

    public init() {}

    // MARK: - Same-circuit batch verification

    /// Batch verify N Groth16 proofs for the same circuit (shared verification key).
    ///
    /// Math: pick random r_1,...,r_N. Compute:
    ///   LHS_A = Sum r_i * A_i           (G1 MSM)
    ///   LHS_C = Sum r_i * C_i           (G1 MSM)
    ///   LHS_L = Sum r_i * L_i           (G1 MSM, where L_i = ic[0] + Sum inp_j * ic[j+1])
    ///   rSum  = Sum r_i                  (scalar sum)
    ///
    /// Check: e(-LHS_A, B) * e(rSum * alpha, beta) * e(LHS_L, gamma) * e(LHS_C, delta) = 1
    ///
    /// Since all proofs share the same VK (same B_i for G2 components), we need only
    /// 4 pairings regardless of N.
    ///
    /// Note: B_i (G2 points) differ per proof even for same circuit. We handle this by
    /// computing individual e(r_i*A_i, B_i) terms. For true batching, we use the product
    /// of Miller loops + single final exponentiation.
    ///
    /// - Parameters:
    ///   - proofs: Array of Groth16 proofs
    ///   - vk: Shared verification key
    ///   - publicInputs: Array of public input vectors (one per proof)
    /// - Returns: true if all proofs are valid (with overwhelming probability)
    public func batchVerifyGroth16(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]]
    ) -> Bool {
        let n = proofs.count
        guard n > 0 else { return true }
        guard n == publicInputs.count else { return false }

        // Validate all public input lengths
        for inputs in publicInputs {
            guard inputs.count + 1 == vk.ic.count else { return false }
        }

        // Generate random scalars via Fiat-Shamir transcript
        let transcript = Transcript(label: "batch-groth16-verify", backend: .poseidon2)

        // Absorb all proof elements into transcript for soundness
        for i in 0..<n {
            if let aAff = pointToAffine(proofs[i].a) {
                transcript.absorb(fpToFr(aAff.x))
                transcript.absorb(fpToFr(aAff.y))
            }
            if let cAff = pointToAffine(proofs[i].c) {
                transcript.absorb(fpToFr(cAff.x))
                transcript.absorb(fpToFr(cAff.y))
            }
            for inp in publicInputs[i] {
                transcript.absorb(inp)
            }
        }

        let scalars = transcript.squeezeN(n)

        return batchVerifyGroth16WithScalars(
            proofs: proofs, vk: vk, publicInputs: publicInputs, scalars: scalars)
    }

    /// Batch verify with pre-computed random scalars (for deterministic testing).
    ///
    /// For each proof i, compute:
    ///   L_i = vk.ic[0] + Sum_j publicInputs[i][j] * vk.ic[j+1]
    ///
    /// Then the batched pairing check is:
    ///   prod_i e(r_i*A_i, B_i) = e(alpha, beta)^(Sum r_i) * prod_i e(r_i*L_i, gamma) * prod_i e(r_i*C_i, delta)
    ///
    /// Rearranged as a product-of-pairings = 1 check:
    ///   prod_i e(-r_i*A_i, B_i) * e((Sum r_i)*alpha, beta) * prod_i e(r_i*L_i, gamma) * prod_i e(r_i*C_i, delta) = 1
    ///
    /// We batch the Miller loops and do one final exponentiation.
    public func batchVerifyGroth16WithScalars(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]],
        scalars: [Fr]
    ) -> Bool {
        let n = proofs.count
        guard n == scalars.count, n == publicInputs.count, n > 0 else { return false }

        // Compute L_i for each proof
        var lPoints = [PointProjective]()
        lPoints.reserveCapacity(n)
        for i in 0..<n {
            var li = vk.ic[0]
            for j in 0..<publicInputs[i].count {
                if !publicInputs[i][j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], publicInputs[i][j]))
                }
            }
            lPoints.append(li)
        }

        // Compute aggregated G1 points:
        //   aggA = Sum r_i * A_i
        //   aggC = Sum r_i * C_i
        //   aggL = Sum r_i * L_i
        //   rSum = Sum r_i
        var aggA = pointIdentity()
        var aggC = pointIdentity()
        var aggL = pointIdentity()
        var rSum = Fr.zero

        for i in 0..<n {
            aggA = pointAdd(aggA, cPointScalarMul(proofs[i].a, scalars[i]))
            aggC = pointAdd(aggC, cPointScalarMul(proofs[i].c, scalars[i]))
            aggL = pointAdd(aggL, cPointScalarMul(lPoints[i], scalars[i]))
            rSum = frAdd(rSum, scalars[i])
        }

        // For same-circuit: all B_i may differ (random blinding), so we cannot
        // merge the A-B pairings into a single MSM. Instead we build the full
        // product of Miller loops.
        //
        // However, for the common case where B_i are identical (no proof re-randomization),
        // we can use just 4 pairings. We check if all B_i are the same first.
        let allBSame = proofs.allSatisfy { g2Equal($0.b, proofs[0].b) }

        if allBSame {
            // Optimized path: e(-aggA, B) * e(rSum*alpha, beta) * e(aggL, gamma) * e(aggC, delta) = 1
            let negAggA = pointNeg(aggA)
            let rSumAlpha = cPointScalarMul(vk.alpha_g1, rSum)

            guard let nA = pointToAffine(negAggA),
                  let ra = pointToAffine(rSumAlpha),
                  let aL = pointToAffine(aggL),
                  let aC = pointToAffine(aggC) else { return false }
            guard let b0 = g2ToAffine(proofs[0].b),
                  let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            return cBN254PairingCheck([(nA, b0), (ra, be), (aL, ga), (aC, de)])
        } else {
            // General path: individual A_i-B_i pairings + 3 shared pairings
            // Build pairs: (-r_i*A_i, B_i) for i=0..N-1, plus (rSum*alpha, beta), (aggL, gamma), (aggC, delta)
            var pairs = [(PointAffine, G2AffinePoint)]()
            pairs.reserveCapacity(n + 3)

            for i in 0..<n {
                let negRiAi = pointNeg(cPointScalarMul(proofs[i].a, scalars[i]))
                guard let nA = pointToAffine(negRiAi),
                      let bi = g2ToAffine(proofs[i].b) else { return false }
                pairs.append((nA, bi))
            }

            let rSumAlpha = cPointScalarMul(vk.alpha_g1, rSum)
            guard let ra = pointToAffine(rSumAlpha),
                  let aL = pointToAffine(aggL),
                  let aC = pointToAffine(aggC) else { return false }
            guard let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            pairs.append((ra, be))
            pairs.append((aL, ga))
            pairs.append((aC, de))

            return cBN254PairingCheck(pairs)
        }
    }

    // MARK: - GPU-accelerated batch verification

    /// GPU-accelerated batch Groth16 verification using MSM for the scalar multiplications.
    /// For large batches (N >= 16), the GPU MSM path is significantly faster.
    ///
    /// Computes via MSM:
    ///   aggA = MSM(A_1,...,A_N; r_1,...,r_N)
    ///   aggC = MSM(C_1,...,C_N; r_1,...,r_N)
    ///   aggL = MSM(L_1,...,L_N; r_1,...,r_N)
    public func batchVerifyGroth16WithMSM(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]],
        msmEngine: MetalMSM
    ) throws -> Bool {
        let n = proofs.count
        guard n > 0, n == publicInputs.count else { return n == 0 }

        for inputs in publicInputs {
            guard inputs.count + 1 == vk.ic.count else { return false }
        }

        // Generate random scalars via Fiat-Shamir
        let transcript = Transcript(label: "batch-groth16-msm-verify", backend: .poseidon2)
        for i in 0..<n {
            if let aAff = pointToAffine(proofs[i].a) {
                transcript.absorb(fpToFr(aAff.x))
                transcript.absorb(fpToFr(aAff.y))
            }
            if let cAff = pointToAffine(proofs[i].c) {
                transcript.absorb(fpToFr(cAff.x))
                transcript.absorb(fpToFr(cAff.y))
            }
            for inp in publicInputs[i] {
                transcript.absorb(inp)
            }
        }
        let scalars = transcript.squeezeN(n)

        // Compute L_i for each proof
        var lPoints = [PointProjective]()
        lPoints.reserveCapacity(n)
        for i in 0..<n {
            var li = vk.ic[0]
            for j in 0..<publicInputs[i].count {
                if !publicInputs[i][j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], publicInputs[i][j]))
                }
            }
            lPoints.append(li)
        }

        // Convert to affine for MSM
        let aPoints = batchToAffine(proofs.map { $0.a })
        let cPoints = batchToAffine(proofs.map { $0.c })
        let lAffine = batchToAffine(lPoints)
        let scalarLimbs = scalars.map { frToLimbs($0) }

        // GPU MSM for all three aggregations
        let aggA = try msmEngine.msm(points: aPoints, scalars: scalarLimbs)
        let aggC = try msmEngine.msm(points: cPoints, scalars: scalarLimbs)
        let aggL = try msmEngine.msm(points: lAffine, scalars: scalarLimbs)

        // Compute rSum
        var rSum = Fr.zero
        for s in scalars { rSum = frAdd(rSum, s) }

        // Pairing check: e(-aggA, B) * e(rSum*alpha, beta) * e(aggL, gamma) * e(aggC, delta) = 1
        // (assuming same B for all proofs in the optimized path)
        let allBSame = proofs.allSatisfy { g2Equal($0.b, proofs[0].b) }

        if allBSame {
            let negAggA = pointNeg(aggA)
            let rSumAlpha = cPointScalarMul(vk.alpha_g1, rSum)

            guard let nA = pointToAffine(negAggA),
                  let ra = pointToAffine(rSumAlpha),
                  let aL = pointToAffine(aggL),
                  let aC = pointToAffine(aggC) else { return false }
            guard let b0 = g2ToAffine(proofs[0].b),
                  let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            return cBN254PairingCheck([(nA, b0), (ra, be), (aL, ga), (aC, de)])
        } else {
            // General path with individual A-B pairings
            var pairs = [(PointAffine, G2AffinePoint)]()
            pairs.reserveCapacity(n + 3)
            for i in 0..<n {
                let negRiAi = pointNeg(cPointScalarMul(proofs[i].a, scalars[i]))
                guard let nA = pointToAffine(negRiAi),
                      let bi = g2ToAffine(proofs[i].b) else { return false }
                pairs.append((nA, bi))
            }
            let rSumAlpha = cPointScalarMul(vk.alpha_g1, rSum)
            guard let ra = pointToAffine(rSumAlpha),
                  let aL = pointToAffine(aggL),
                  let aC = pointToAffine(aggC) else { return false }
            guard let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }
            pairs.append((ra, be))
            pairs.append((aL, ga))
            pairs.append((aC, de))
            return cBN254PairingCheck(pairs)
        }
    }

    /// Adaptive batch verification: chooses CPU or GPU path based on batch size.
    public static let gpuMSMThreshold = 16

    public func batchVerifyGroth16Adaptive(
        proofs: [Groth16Proof],
        vk: Groth16VerificationKey,
        publicInputs: [[Fr]],
        msmEngine: MetalMSM? = nil
    ) throws -> Bool {
        if proofs.count >= Groth16BatchVerifier.gpuMSMThreshold, let msm = msmEngine {
            return try batchVerifyGroth16WithMSM(
                proofs: proofs, vk: vk, publicInputs: publicInputs, msmEngine: msm)
        } else {
            return batchVerifyGroth16(proofs: proofs, vk: vk, publicInputs: publicInputs)
        }
    }

    // MARK: - Cross-circuit batch verification

    /// Batch verify proofs from different circuits (different VKs).
    /// Groups proofs by VK, then batch-verifies each group.
    /// Returns true only if ALL groups pass.
    public func batchVerifyCrossCircuit(
        proofs: [(proof: Groth16Proof, vk: Groth16VerificationKey, inputs: [Fr])]
    ) -> Bool {
        guard !proofs.isEmpty else { return true }

        // Group by VK identity (using alpha_g1 + beta_g2 as fingerprint)
        // For simplicity, compare VK pointers via their serialized alpha point
        var groups = [[Int]]()  // indices into proofs array
        var vkFingerprints = [[UInt64]]()

        for (idx, item) in proofs.enumerated() {
            let fp = vkFingerprint(item.vk)
            if let gIdx = vkFingerprints.firstIndex(of: fp) {
                groups[gIdx].append(idx)
            } else {
                vkFingerprints.append(fp)
                groups.append([idx])
            }
        }

        // Batch verify each group
        for group in groups {
            let vk = proofs[group[0]].vk
            let groupProofs = group.map { proofs[$0].proof }
            let groupInputs = group.map { proofs[$0].inputs }

            if !batchVerifyGroth16(proofs: groupProofs, vk: vk, publicInputs: groupInputs) {
                return false
            }
        }

        return true
    }

    // MARK: - Helpers

    /// Fingerprint a VK for grouping (not cryptographic, just for equality check)
    private func vkFingerprint(_ vk: Groth16VerificationKey) -> [UInt64] {
        if let a = pointToAffine(vk.alpha_g1) {
            return a.x.to64() + a.y.to64()
        }
        return [0, 0, 0, 0, 0, 0, 0, 0]
    }
}

// MARK: - G2 point equality helper

/// Check if two G2 projective points are equal (via cross-multiplication).
private func g2Equal(_ a: G2ProjectivePoint, _ b: G2ProjectivePoint) -> Bool {
    if g2IsIdentity(a) && g2IsIdentity(b) { return true }
    if g2IsIdentity(a) || g2IsIdentity(b) { return false }
    // a.x * b.z^2 == b.x * a.z^2
    let az2 = fp2Sqr(a.z)
    let bz2 = fp2Sqr(b.z)
    let lhsX = fp2Mul(a.x, bz2)
    let rhsX = fp2Mul(b.x, az2)
    if !fp2Equal(lhsX, rhsX) { return false }
    // a.y * b.z^3 == b.y * a.z^3
    let az3 = fp2Mul(az2, a.z)
    let bz3 = fp2Mul(bz2, b.z)
    let lhsY = fp2Mul(a.y, bz3)
    let rhsY = fp2Mul(b.y, az3)
    return fp2Equal(lhsY, rhsY)
}

/// Check Fp2 equality
private func fp2Equal(_ a: Fp2, _ b: Fp2) -> Bool {
    let dx = fpSub(a.c0, b.c0)
    let dy = fpSub(a.c1, b.c1)
    return dx.isZero && dy.isZero
}
