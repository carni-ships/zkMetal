// VarunaVerifier — Marlin/Varuna pairing-based verifier with variable-degree support
//
// Marlin (Chiesa et al., EUROCRYPT 2020) verifies AHP proofs using KZG commitments.
// Varuna (Bowe et al.) extends Marlin with variable-degree polynomial support,
// allowing each committed polynomial to have a different degree bound.
//
// This verifier replaces SRS-secret-based KZG checks with proper BN254 pairing checks:
//   e(C - [y]G1, G2) = e(W, [s]G2 - [z]G2)
//
// Batch optimization: 19 polynomial openings at 2 evaluation points (beta, gamma)
// are reduced to 2 MSMs + 1 multi-pairing check via random linear combination.
//
// Variable-degree (Varuna): polynomials with degree < D are "shifted" by
// committing to X^{D_max - D} * p(X), enforcing the degree bound via an
// additional pairing check.

import Foundation
import NeonFieldOps

// MARK: - Varuna Verification Key

/// Extended verification key with G2 SRS elements for pairing-based checks.
public struct VarunaVerifyingKey {
    /// The base Marlin index (R1CS structure)
    public let index: MarlinIndex
    /// KZG commitments to index polynomials: row, col, val, row_col for A, B, C
    public let indexCommitments: [PointProjective]  // 12 commitments
    /// G1 generator (first SRS point)
    public let g1: PointAffine
    /// G2 generator
    public let g2: G2AffinePoint
    /// [s]G2 — G2 element at the SRS secret (for KZG pairing checks)
    public let sG2: G2AffinePoint
    /// Maximum degree supported by the SRS
    public let maxDegree: Int
    /// Degree bounds for each committed polynomial (Varuna extension).
    /// If nil, all polynomials are assumed to have degree < maxDegree (standard Marlin).
    public let degreeBounds: [Int]?
    /// Shifted G2 elements: [s^{D_max - D_i}]G2 for each degree-bound polynomial.
    /// Used to verify degree-bound enforcement via pairing: e(C_shifted, G2) = e(C, [s^shift]G2).
    public let shiftedG2: [G2AffinePoint]?

    public init(index: MarlinIndex, indexCommitments: [PointProjective],
                g1: PointAffine, g2: G2AffinePoint, sG2: G2AffinePoint,
                maxDegree: Int, degreeBounds: [Int]? = nil,
                shiftedG2: [G2AffinePoint]? = nil) {
        self.index = index
        self.indexCommitments = indexCommitments
        self.g1 = g1
        self.g2 = g2
        self.sG2 = sG2
        self.maxDegree = maxDegree
        self.degreeBounds = degreeBounds
        self.shiftedG2 = shiftedG2
    }

    /// Construct from a MarlinVerifyingKey (convenience for upgrade path).
    /// The SRS secret is used only to derive [s]G2; it is NOT stored.
    public static func fromMarlinVK(_ mvk: MarlinVerifyingKey, maxDegree: Int) -> VarunaVerifyingKey {
        let g1Aff = mvk.srs[0]
        let g2Aff = bn254G2Generator()
        let g2Proj = g2FromAffine(g2Aff)
        let sG2Proj = g2ScalarMul(g2Proj, frToInt(mvk.srsSecret))
        let sG2Aff = g2ToAffine(sG2Proj)!
        return VarunaVerifyingKey(
            index: mvk.index, indexCommitments: mvk.indexCommitments,
            g1: g1Aff, g2: g2Aff, sG2: sG2Aff, maxDegree: maxDegree
        )
    }
}

// MARK: - Varuna Proof (extends Marlin proof with degree-bound witnesses)

/// A Varuna proof wraps a Marlin proof with optional shifted commitments
/// for degree-bounded polynomials.
public struct VarunaProof {
    /// The underlying Marlin proof
    public let marlinProof: MarlinProof
    /// Shifted commitments for degree-bounded polynomials.
    /// shiftedCommitments[i] = commit(X^{D_max - D_i} * p_i(X)).
    /// If nil, no degree bounds are enforced (standard Marlin).
    public let shiftedCommitments: [PointProjective]?
    /// Opening proofs for shifted polynomials at the same evaluation points
    public let shiftedBetaProof: PointProjective?
    public let shiftedGammaProof: PointProjective?

    public init(marlinProof: MarlinProof,
                shiftedCommitments: [PointProjective]? = nil,
                shiftedBetaProof: PointProjective? = nil,
                shiftedGammaProof: PointProjective? = nil) {
        self.marlinProof = marlinProof
        self.shiftedCommitments = shiftedCommitments
        self.shiftedBetaProof = shiftedBetaProof
        self.shiftedGammaProof = shiftedGammaProof
    }
}

// MARK: - VarunaVerifier

public class VarunaVerifier {
    public static let version = Versions.varuna
    public let kzg: KZGEngine

    public init(kzg: KZGEngine) {
        self.kzg = kzg
    }

    /// Verify a Marlin proof using BN254 pairing-based KZG checks.
    /// This is the standard Marlin verification with pairings instead of SRS secret.
    ///
    /// Cost: O(1) pairings (2 multi-pairing checks via batching) + O(log n) field ops.
    public func verify(vk: VarunaVerifyingKey, publicInput: [Fr],
                       proof: MarlinProof) -> Bool {
        let idx = vk.index

        // --- Step 1: Reconstruct Fiat-Shamir challenges ---
        let allPoints = collectProofPoints(vk: vk, proof: proof)
        let (identityFlags, affineLookup) = batchConvertToAffine(allPoints)

        func absorbPoint(_ transcript: Transcript, _ idx: Int) {
            if identityFlags[idx] {
                transcript.absorb(Fr.zero)
                transcript.absorb(Fr.zero)
            } else if let aff = affineLookup[idx] {
                transcript.absorb(Fr.from64(aff.x))
                transcript.absorb(Fr.from64(aff.y))
            }
        }

        let transcript = Transcript(label: "marlin", backend: .keccak256)
        transcript.absorb(frFromInt(UInt64(idx.numConstraints)))
        transcript.absorb(frFromInt(UInt64(idx.numVariables)))
        transcript.absorb(frFromInt(UInt64(idx.numNonZero)))

        for i in 0..<vk.indexCommitments.count { absorbPoint(transcript, i) }
        for pi in publicInput { transcript.absorb(pi) }

        // Round 1 commitments
        absorbPoint(transcript, 12) // w
        absorbPoint(transcript, 13) // zA
        absorbPoint(transcript, 14) // zB
        absorbPoint(transcript, 15) // zC

        let _etaA = transcript.squeeze()
        let _etaB = transcript.squeeze()
        let _etaC = transcript.squeeze()

        // Round 2
        absorbPoint(transcript, 16) // t
        let alpha = transcript.squeeze()
        for coeffs in proof.sumcheckPolyCoeffs {
            for c in coeffs { transcript.absorb(c) }
        }
        let beta = transcript.squeeze()

        // Round 3
        absorbPoint(transcript, 17) // g
        absorbPoint(transcript, 18) // h
        let gamma = transcript.squeeze()

        // --- Step 2: Verify outer sumcheck rounds ---
        if !verifySumcheckRounds(proof.sumcheckPolyCoeffs, alpha: alpha) {
            return false
        }

        // --- Step 3: Verify outer relation: zA(beta)*zB(beta) - zC(beta) = t(beta)*v_H(beta) ---
        let evals = proof.evaluations
        var lhsLimbs = [UInt64](repeating: 0, count: 4)
        var rhsLimbs = [UInt64](repeating: 0, count: 4)
        var prod = [UInt64](repeating: 0, count: 4)
        bn254_fr_mul(evals.zABeta.to64(), evals.zBBeta.to64(), &prod)
        bn254_fr_sub(prod, evals.zCBeta.to64(), &lhsLimbs)

        var vH = [UInt64](repeating: 0, count: 4)
        bn254_fr_pow(beta.to64(), UInt64(idx.constraintDomainSize), &vH)
        bn254_fr_sub(vH, Fr.one.to64(), &vH)
        bn254_fr_mul(evals.tBeta.to64(), vH, &rhsLimbs)

        if lhsLimbs != rhsLimbs {
            return false
        }

        // --- Step 4: Verify KZG openings via pairing ---
        let batchChallenge = transcript.squeeze()
        return verifyOpeningsViaPairing(
            vk: vk, proof: proof, beta: beta, gamma: gamma,
            batchChallenge: batchChallenge
        )
    }

    /// Verify a Varuna proof (Marlin + variable-degree bounds).
    /// Checks:
    ///   1. Standard Marlin verification (AHP + batched KZG via pairing)
    ///   2. Degree-bound enforcement via shifted commitment pairing checks
    public func verifyVaruna(vk: VarunaVerifyingKey, publicInput: [Fr],
                             proof: VarunaProof) -> Bool {
        // Step 1: Standard Marlin verification
        guard verify(vk: vk, publicInput: publicInput, proof: proof.marlinProof) else {
            return false
        }

        // Step 2: Degree-bound enforcement (Varuna extension)
        guard let degreeBounds = vk.degreeBounds,
              let shiftedG2 = vk.shiftedG2,
              let shiftedCommitments = proof.shiftedCommitments else {
            // No degree bounds to check -- standard Marlin, already verified
            return true
        }

        return verifyDegreeBounds(
            vk: vk, degreeBounds: degreeBounds,
            shiftedG2: shiftedG2,
            shiftedCommitments: shiftedCommitments,
            proof: proof
        )
    }

    // MARK: - Pairing-based KZG Verification

    /// Verify all KZG opening proofs using BN254 multi-pairing.
    ///
    /// Batch verification reduces 19 polynomial openings (5 at beta, 14 at gamma)
    /// to a single multi-pairing check via random linear combination:
    ///
    ///   e(Σ rho^i * (C_i - [y_i]G1), G2) = e(Σ rho^i * W_i, [s]G2 - [z_i]G2)
    ///
    /// For 2 evaluation points, this becomes:
    ///   e(accumC - [accumY]G1, G2) = e(accumW_beta, [s]G2 - [beta]G2) *
    ///                                 e(accumW_gamma, [s]G2 - [gamma]G2)
    ///
    /// Which is a 3-pairing check (or equivalently, a 3-pair multi-Miller + final exp).
    private func verifyOpeningsViaPairing(vk: VarunaVerifyingKey, proof: MarlinProof,
                                          beta: Fr, gamma: Fr,
                                          batchChallenge: Fr) -> Bool {
        let evals = proof.evaluations
        let g1Proj = pointFromAffine(vk.g1)

        // Collect all (commitment, evaluation, isBeta) tuples
        // Beta group (5 polys): w, zA, zB, zC, t
        let betaCommits = [proof.wCommit, proof.zACommit, proof.zBCommit,
                           proof.zCCommit, proof.tCommit]
        let betaEvals = [evals.wBeta, evals.zABeta, evals.zBBeta,
                         evals.zCBeta, evals.tBeta]

        // Gamma group (14 polys): g, h, 12 index polys
        var gammaCommits = [proof.gCommit, proof.hCommit]
        gammaCommits.append(contentsOf: vk.indexCommitments)
        var gammaEvals: [Fr] = [evals.gGamma, evals.hGamma]
        for m in 0..<3 {
            gammaEvals.append(evals.rowGamma[m])
            gammaEvals.append(evals.colGamma[m])
            gammaEvals.append(evals.valGamma[m])
            gammaEvals.append(evals.rowColGamma[m])
        }

        // If batch proofs are available, verify them via pairing
        if let betaProof = proof.betaBatchProof,
           let gammaProof = proof.gammaBatchProof,
           let batchGamma = proof.batchChallenge {
            return verifyBatchPairing(
                vk: vk, betaCommits: betaCommits, betaEvals: betaEvals, betaProof: betaProof,
                gammaCommits: gammaCommits, gammaEvals: gammaEvals, gammaProof: gammaProof,
                beta: beta, gamma: gamma, batchGamma: batchGamma
            )
        }

        // Legacy individual proofs: accumulate into single multi-pairing
        return verifyIndividualOpeningsViaPairing(
            vk: vk, proof: proof, beta: beta, gamma: gamma,
            betaCommits: betaCommits, betaEvals: betaEvals,
            gammaCommits: gammaCommits, gammaEvals: gammaEvals,
            batchChallenge: batchChallenge
        )
    }

    /// Verify batch KZG proofs (one per evaluation point) via 3-pair pairing check.
    ///
    /// For each group (beta or gamma):
    ///   C_combined = Σ gamma^i * C_i
    ///   y_combined = Σ gamma^i * y_i
    ///
    /// Check: e(C_combined - [y_combined]G1, G2) = e(W, [s]G2 - [z]G2)
    ///
    /// Combining both groups with random rho:
    ///   e(LHS1 + rho*LHS2, G2) = e(W_beta, sG2 - [beta]G2) * e(rho*W_gamma, sG2 - [gamma]G2)
    ///
    /// Rearranged to a single product check == 1:
    ///   e(negLHS, G2) * e(W_beta, sG2-betaG2) * e(W_gamma_scaled, sG2-gammaG2) == 1
    private func verifyBatchPairing(
        vk: VarunaVerifyingKey,
        betaCommits: [PointProjective], betaEvals: [Fr], betaProof: PointProjective,
        gammaCommits: [PointProjective], gammaEvals: [Fr], gammaProof: PointProjective,
        beta: Fr, gamma: Fr, batchGamma: Fr
    ) -> Bool {
        let g1Proj = pointFromAffine(vk.g1)

        // Beta group: C_beta = Σ gamma^i * C_i, y_beta = Σ gamma^i * y_i
        let (cBeta, yBeta) = combinedCommitmentAndEval(
            commitments: betaCommits, evaluations: betaEvals, gamma: batchGamma
        )

        // Gamma group: C_gamma = Σ gamma^i * C_i, y_gamma = Σ gamma^i * y_i
        let (cGamma, yGamma) = combinedCommitmentAndEval(
            commitments: gammaCommits, evaluations: gammaEvals, gamma: batchGamma
        )

        // LHS_beta = C_beta - [y_beta]G1
        let lhsBeta = pointAdd(cBeta, pointNeg(cPointScalarMul(g1Proj, yBeta)))
        // LHS_gamma = C_gamma - [y_gamma]G1
        let lhsGamma = pointAdd(cGamma, pointNeg(cPointScalarMul(g1Proj, yGamma)))

        // Combine with random rho for a single check
        let rho = batchGamma  // reuse as independent randomness
        let combinedLHS = pointAdd(lhsBeta, cPointScalarMul(lhsGamma, rho))
        let scaledGammaProof = cPointScalarMul(gammaProof, rho)

        // Compute [z]G2 for each eval point
        let g2Proj = g2FromAffine(vk.g2)
        let betaG2 = g2ToAffine(g2ScalarMul(g2Proj, frToInt(beta)))!
        let gammaG2 = g2ToAffine(g2ScalarMul(g2Proj, frToInt(gamma)))!

        // sG2 - [beta]G2, sG2 - [gamma]G2
        let sMinusBetaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                              g2Negate(g2FromAffine(betaG2))))!
        let sMinusGammaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                               g2Negate(g2FromAffine(gammaG2))))!

        // Convert LHS to affine (negate for pairing check format: prod e(...) = 1)
        guard let negLHSAff = pointToAffine(pointNeg(combinedLHS)),
              let betaProofAff = pointToAffine(betaProof),
              let gammaProofAff = pointToAffine(scaledGammaProof) else {
            return false
        }

        // Pairing check: e(-combinedLHS, G2) * e(W_beta, sG2-betaG2) * e(W_gamma_scaled, sG2-gammaG2) = 1
        return cBN254PairingCheck([
            (negLHSAff, vk.g2),
            (betaProofAff, sMinusBetaG2),
            (gammaProofAff, sMinusGammaG2)
        ])
    }

    /// Verify individual (legacy) opening proofs via accumulated multi-pairing.
    /// Accumulates all 19 openings into 2 groups (beta, gamma), then performs
    /// a single 3-pair pairing check.
    private func verifyIndividualOpeningsViaPairing(
        vk: VarunaVerifyingKey, proof: MarlinProof,
        beta: Fr, gamma: Fr,
        betaCommits: [PointProjective], betaEvals: [Fr],
        gammaCommits: [PointProjective], gammaEvals: [Fr],
        batchChallenge: Fr
    ) -> Bool {
        let g1Proj = pointFromAffine(vk.g1)
        let evals = proof.evaluations

        // Accumulate: combinedC = Σ rho^i * C_i, combinedY = Σ rho^i * y_i
        // Split by evaluation point into beta-group and gamma-group witnesses
        var accumCBeta = pointIdentity()
        var accumYBeta = Fr.zero
        var accumWBeta = pointIdentity()

        var accumCGamma = pointIdentity()
        var accumYGamma = Fr.zero
        var accumWGamma = pointIdentity()

        var rho = Fr.one

        // Beta group: 5 openings
        for i in 0..<betaCommits.count {
            let c = betaCommits[i]
            let y = betaEvals[i]
            accumCBeta = pointAdd(accumCBeta, cPointScalarMul(c, rho))
            accumYBeta = frAdd(accumYBeta, frMul(rho, y))
            if i < proof.openingProofs.count {
                accumWBeta = pointAdd(accumWBeta, cPointScalarMul(proof.openingProofs[i], rho))
            }
            rho = frMul(rho, batchChallenge)
        }

        // Gamma group: g, h
        let gammaProofStart = 5
        for i in 0..<2 {
            let c = gammaCommits[i]
            let y = gammaEvals[i]
            accumCGamma = pointAdd(accumCGamma, cPointScalarMul(c, rho))
            accumYGamma = frAdd(accumYGamma, frMul(rho, y))
            let proofIdx = gammaProofStart + i
            if proofIdx < proof.openingProofs.count {
                accumWGamma = pointAdd(accumWGamma, cPointScalarMul(proof.openingProofs[proofIdx], rho))
            }
            rho = frMul(rho, batchChallenge)
        }

        // Gamma group: 12 index polynomial openings
        for m in 0..<3 {
            let matEvals = [evals.rowGamma[m], evals.colGamma[m],
                            evals.valGamma[m], evals.rowColGamma[m]]
            for k in 0..<4 {
                let commitIdx = m * 4 + k
                let proofIdx = 7 + m * 4 + k
                if commitIdx < vk.indexCommitments.count {
                    accumCGamma = pointAdd(accumCGamma,
                                           cPointScalarMul(vk.indexCommitments[commitIdx], rho))
                    accumYGamma = frAdd(accumYGamma, frMul(rho, matEvals[k]))
                    if proofIdx < proof.openingProofs.count {
                        accumWGamma = pointAdd(accumWGamma,
                                               cPointScalarMul(proof.openingProofs[proofIdx], rho))
                    }
                }
                rho = frMul(rho, batchChallenge)
            }
        }

        // LHS = accumC - [accumY]G1
        let lhsBeta = pointAdd(accumCBeta, pointNeg(cPointScalarMul(g1Proj, accumYBeta)))
        let lhsGamma = pointAdd(accumCGamma, pointNeg(cPointScalarMul(g1Proj, accumYGamma)))
        let combinedLHS = pointAdd(lhsBeta, lhsGamma)

        // Compute G2 elements
        let g2Proj = g2FromAffine(vk.g2)
        let betaG2 = g2ToAffine(g2ScalarMul(g2Proj, frToInt(beta)))!
        let gammaG2Pt = g2ToAffine(g2ScalarMul(g2Proj, frToInt(gamma)))!
        let sMinusBetaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                              g2Negate(g2FromAffine(betaG2))))!
        let sMinusGammaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                               g2Negate(g2FromAffine(gammaG2Pt))))!

        guard let negLHSAff = pointToAffine(pointNeg(combinedLHS)),
              let wBetaAff = pointToAffine(accumWBeta),
              let wGammaAff = pointToAffine(accumWGamma) else {
            return false
        }

        // 3-pair pairing check
        return cBN254PairingCheck([
            (negLHSAff, vk.g2),
            (wBetaAff, sMinusBetaG2),
            (wGammaAff, sMinusGammaG2)
        ])
    }

    // MARK: - Varuna Degree-Bound Verification

    /// Verify degree bounds via shifted commitment pairing checks.
    ///
    /// For each degree-bounded polynomial p_i with degree bound D_i:
    ///   The prover commits to p_shifted_i(X) = X^{D_max - D_i} * p_i(X)
    ///   Verification: e(C_shifted_i, G2) = e(C_i, [s^{D_max - D_i}]G2)
    ///
    /// This ensures deg(p_i) <= D_i because if deg(p_i) > D_i, then
    /// deg(p_shifted_i) > D_max, and the KZG commitment would be invalid
    /// (not representable with the SRS of size D_max + 1).
    ///
    /// Batched via random linear combination into a single 2-pair pairing check.
    private func verifyDegreeBounds(
        vk: VarunaVerifyingKey,
        degreeBounds: [Int],
        shiftedG2: [G2AffinePoint],
        shiftedCommitments: [PointProjective],
        proof: VarunaProof
    ) -> Bool {
        guard degreeBounds.count == shiftedG2.count,
              degreeBounds.count == shiftedCommitments.count else {
            return false
        }

        let n = degreeBounds.count
        if n == 0 { return true }

        // Generate batching randomness from shifted commitments
        let ts = Transcript(label: "varuna-degree-bounds", backend: .keccak256)
        for sc in shiftedCommitments {
            varunaAbsorbPoint(ts, sc)
        }
        let rho = ts.squeeze()

        // Accumulate: Σ rho^i * C_shifted_i and Σ rho^i * [s^shift_i]G2 pairing with C_i
        // Check: e(Σ rho^i * C_shifted_i, G2) = e(Σ rho^i * C_i, ???)
        //
        // Actually, for different shift amounts, we cannot merge the G2 side.
        // Instead: accumulate LHS - RHS = 0 via random linear combination.
        //
        // e(C_shifted_i, G2) = e(C_i, [s^shift]G2)
        // => e(C_shifted_i, G2) * e(-C_i, [s^shift]G2) = 1
        //
        // With batching: product_i (e(C_shifted_i, G2) * e(-C_i, [s^shift]G2))^{rho^i} = 1
        //
        // This is hard to batch across different G2 elements. Instead, we use
        // the multi-pairing approach: accumulate into 2 groups.
        //
        // Optimization: since all G2 elements on the "shifted" side are the same (G2),
        // we can combine the shifted commitments:
        //   e(Σ rho^i * C_shifted_i, G2)
        // And for the commitment side, each has a different [s^shift]G2.
        // We batch those using the same rho into a multi-pairing.

        // Group 1: accumulate shifted commitments (all paired with G2)
        var accumShifted = pointIdentity()
        var rhoPow = Fr.one
        for i in 0..<n {
            accumShifted = pointAdd(accumShifted, cPointScalarMul(shiftedCommitments[i], rhoPow))
            rhoPow = frMul(rhoPow, rho)
        }

        // Group 2: accumulate original commitments (each paired with different shifted G2)
        // Since we cannot batch different G2 elements trivially, we use the relation:
        //   e(Σ rho^i * C_shifted_i, G2) = Π_i e(C_i, [s^shift_i]G2)^{rho^i}
        //
        // For efficiency, batch this as n+1 pair multi-pairing check:
        //   e(-accumShifted, G2) * Π_i e(rho^i * C_i, [s^shift_i]G2) = 1
        //
        // For small n (typically <= 19), this is fast.

        guard let negAccumAff = pointToAffine(pointNeg(accumShifted)) else {
            return false
        }

        var pairs: [(PointAffine, G2AffinePoint)] = [(negAccumAff, vk.g2)]

        rhoPow = Fr.one
        for i in 0..<n {
            let commitIdx = i  // Map to the appropriate commitment
            // For Varuna: degree-bounded polys map to a subset of all committed polys.
            // Here we assume shiftedCommitments[i] corresponds to indexCommitments[i]
            // (or the appropriate witness polynomial).
            let scaledC = cPointScalarMul(vk.indexCommitments.count > commitIdx
                                          ? vk.indexCommitments[commitIdx]
                                          : pointIdentity(), rhoPow)
            guard let scaledCAff = pointToAffine(scaledC) else {
                return false
            }
            pairs.append((scaledCAff, shiftedG2[i]))
            rhoPow = frMul(rhoPow, rho)
        }

        return cBN254PairingCheck(pairs)
    }

    // MARK: - Batch Verification (multiple proofs)

    /// Batch verify multiple Marlin proofs via accumulated multi-pairing.
    /// All KZG checks across all proofs are combined into a single pairing check.
    ///
    /// Cost: 1 multi-pairing (2*N+1 pairs for N proofs) + N * O(log n) field ops.
    public func batchVerify(vk: VarunaVerifyingKey,
                            proofs: [(publicInput: [Fr], proof: MarlinProof)]) -> Bool {
        guard !proofs.isEmpty else { return false }
        if proofs.count == 1 {
            return verify(vk: vk, publicInput: proofs[0].publicInput, proof: proofs[0].proof)
        }

        let idx = vk.index
        let g1Proj = pointFromAffine(vk.g1)
        let g2Proj = g2FromAffine(vk.g2)

        // Generate batch randomness
        let batchTs = Transcript(label: "varuna-batch", backend: .keccak256)
        batchTs.absorb(frFromInt(UInt64(proofs.count)))
        for (pi, proof) in proofs {
            for p in pi { batchTs.absorb(p) }
            varunaAbsorbPoint(batchTs, proof.wCommit)
        }
        let outerRho = batchTs.squeeze()

        // Accumulate across all proofs
        var allBetaLHS = pointIdentity()
        var allGammaLHS = pointIdentity()
        var allBetaW = pointIdentity()
        var allGammaW = pointIdentity()

        // We need per-proof evaluation points, so collect beta/gamma for each
        // and verify they all come from consistent Fiat-Shamir
        var betaPoints = [Fr]()
        var gammaPoints = [Fr]()

        var outerRhoPow = Fr.one

        for (publicInput, proof) in proofs {
            // Reconstruct challenges via Fiat-Shamir
            let ts = Transcript(label: "marlin", backend: .keccak256)
            ts.absorb(frFromInt(UInt64(idx.numConstraints)))
            ts.absorb(frFromInt(UInt64(idx.numVariables)))
            ts.absorb(frFromInt(UInt64(idx.numNonZero)))
            for c in vk.indexCommitments { varunaAbsorbPoint(ts, c) }
            for pi in publicInput { ts.absorb(pi) }
            varunaAbsorbPoint(ts, proof.wCommit)
            varunaAbsorbPoint(ts, proof.zACommit)
            varunaAbsorbPoint(ts, proof.zBCommit)
            varunaAbsorbPoint(ts, proof.zCCommit)
            let _ = ts.squeeze() // etaA
            let _ = ts.squeeze() // etaB
            let _ = ts.squeeze() // etaC
            varunaAbsorbPoint(ts, proof.tCommit)
            let alpha = ts.squeeze()
            for coeffs in proof.sumcheckPolyCoeffs { for c in coeffs { ts.absorb(c) } }
            let beta = ts.squeeze()
            varunaAbsorbPoint(ts, proof.gCommit)
            varunaAbsorbPoint(ts, proof.hCommit)
            let gamma = ts.squeeze()

            // Verify sumcheck
            guard verifySumcheckRounds(proof.sumcheckPolyCoeffs, alpha: alpha) else {
                return false
            }

            // Verify outer relation
            let evals = proof.evaluations
            let lhs = frSub(frMul(evals.zABeta, evals.zBBeta), evals.zCBeta)
            let vHBeta = frSub(frPow(beta, UInt64(idx.constraintDomainSize)), Fr.one)
            let rhs = frMul(evals.tBeta, vHBeta)
            guard frToInt(lhs) == frToInt(rhs) else { return false }

            betaPoints.append(beta)
            gammaPoints.append(gamma)

            // Accumulate KZG tuples for this proof
            let innerBatchChal = ts.squeeze()

            guard let betaProof = proof.betaBatchProof,
                  let gammaProof = proof.gammaBatchProof,
                  let batchGamma = proof.batchChallenge else {
                // Fall back to single-proof verification for legacy proofs
                if !verifyOpeningsViaPairing(vk: vk, proof: proof, beta: beta,
                                              gamma: gamma, batchChallenge: innerBatchChal) {
                    return false
                }
                outerRhoPow = frMul(outerRhoPow, outerRho)
                continue
            }

            // Batch path: accumulate into combined multi-pairing
            let betaCommits = [proof.wCommit, proof.zACommit, proof.zBCommit,
                               proof.zCCommit, proof.tCommit]
            let betaEvals = [evals.wBeta, evals.zABeta, evals.zBBeta,
                             evals.zCBeta, evals.tBeta]
            var gammaCommits = [proof.gCommit, proof.hCommit]
            gammaCommits.append(contentsOf: vk.indexCommitments)
            var gammaEvals: [Fr] = [evals.gGamma, evals.hGamma]
            for m in 0..<3 {
                gammaEvals.append(evals.rowGamma[m])
                gammaEvals.append(evals.colGamma[m])
                gammaEvals.append(evals.valGamma[m])
                gammaEvals.append(evals.rowColGamma[m])
            }

            let (cBeta, yBeta) = combinedCommitmentAndEval(
                commitments: betaCommits, evaluations: betaEvals, gamma: batchGamma
            )
            let (cGamma, yGamma) = combinedCommitmentAndEval(
                commitments: gammaCommits, evaluations: gammaEvals, gamma: batchGamma
            )

            let lhsBeta = pointAdd(cBeta, pointNeg(cPointScalarMul(g1Proj, yBeta)))
            let lhsGamma = pointAdd(cGamma, pointNeg(cPointScalarMul(g1Proj, yGamma)))

            // Scale by outer rho and accumulate
            allBetaLHS = pointAdd(allBetaLHS, cPointScalarMul(lhsBeta, outerRhoPow))
            allGammaLHS = pointAdd(allGammaLHS, cPointScalarMul(lhsGamma, outerRhoPow))
            allBetaW = pointAdd(allBetaW, cPointScalarMul(betaProof, outerRhoPow))
            allGammaW = pointAdd(allGammaW, cPointScalarMul(gammaProof, outerRhoPow))

            outerRhoPow = frMul(outerRhoPow, outerRho)
        }

        // If we only had legacy proofs (handled individually above), we're done
        if pointIsIdentity(allBetaLHS) && pointIsIdentity(allGammaLHS) &&
           pointIsIdentity(allBetaW) && pointIsIdentity(allGammaW) {
            return true
        }

        // Combined LHS
        let combinedLHS = pointAdd(allBetaLHS, allGammaLHS)

        // For batch with multiple distinct beta/gamma values, we need per-proof G2 elements.
        // When all proofs use the same VK/circuit, beta/gamma differ per proof.
        // We accumulate: Σ outerRho^i * W_beta_i * (s - beta_i) on G2 side.
        // This requires per-proof G2 computation.
        //
        // For the common case (few proofs), we build the multi-pairing explicitly.
        guard let negLHSAff = pointToAffine(pointNeg(combinedLHS)) else {
            return false
        }

        var pairs: [(PointAffine, G2AffinePoint)] = [(negLHSAff, vk.g2)]

        // Add per-proof beta/gamma witnesses
        // For accumulated witnesses with different eval points, we split into per-proof pairs
        outerRhoPow = Fr.one
        for i in 0..<proofs.count {
            let proof = proofs[i].proof
            guard let betaProof = proof.betaBatchProof,
                  let gammaProof = proof.gammaBatchProof else {
                outerRhoPow = frMul(outerRhoPow, outerRho)
                continue
            }

            let beta = betaPoints[i]
            let gamma = gammaPoints[i]

            let betaG2 = g2ToAffine(g2ScalarMul(g2Proj, frToInt(beta)))!
            let gammaG2 = g2ToAffine(g2ScalarMul(g2Proj, frToInt(gamma)))!
            let sMinusBetaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                                  g2Negate(g2FromAffine(betaG2))))!
            let sMinusGammaG2 = g2ToAffine(g2Add(g2FromAffine(vk.sG2),
                                                   g2Negate(g2FromAffine(gammaG2))))!

            let scaledBetaW = cPointScalarMul(betaProof, outerRhoPow)
            let scaledGammaW = cPointScalarMul(gammaProof, outerRhoPow)

            guard let bwAff = pointToAffine(scaledBetaW),
                  let gwAff = pointToAffine(scaledGammaW) else {
                return false
            }

            pairs.append((bwAff, sMinusBetaG2))
            pairs.append((gwAff, sMinusGammaG2))

            outerRhoPow = frMul(outerRhoPow, outerRho)
        }

        return cBN254PairingCheck(pairs)
    }

    // MARK: - Helpers

    /// Combine commitments and evaluations via random linear combination.
    private func combinedCommitmentAndEval(
        commitments: [PointProjective], evaluations: [Fr], gamma: Fr
    ) -> (PointProjective, Fr) {
        var combinedC = pointIdentity()
        var combinedY = Fr.zero
        var gammaPow = Fr.one
        for i in 0..<commitments.count {
            combinedC = pointAdd(combinedC, cPointScalarMul(commitments[i], gammaPow))
            combinedY = frAdd(combinedY, frMul(gammaPow, evaluations[i]))
            if i < commitments.count - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }
        return (combinedC, combinedY)
    }

    /// Collect all proof points for batch affine conversion.
    private func collectProofPoints(vk: VarunaVerifyingKey,
                                    proof: MarlinProof) -> [PointProjective] {
        var pts = [PointProjective]()
        pts.append(contentsOf: vk.indexCommitments)  // 0..11
        pts.append(proof.wCommit)                     // 12
        pts.append(proof.zACommit)                    // 13
        pts.append(proof.zBCommit)                    // 14
        pts.append(proof.zCCommit)                    // 15
        pts.append(proof.tCommit)                     // 16
        pts.append(proof.gCommit)                     // 17
        pts.append(proof.hCommit)                     // 18
        return pts
    }

    /// Batch convert projective points to affine, returning identity flags and lookup table.
    private func batchConvertToAffine(_ points: [PointProjective])
        -> ([Bool], [(x: [UInt64], y: [UInt64])?])
    {
        var identityFlags = [Bool]()
        var nonIdentityPoints = [PointProjective]()
        var nonIdentityMap = [Int]()
        for (i, p) in points.enumerated() {
            let isId = pointIsIdentity(p)
            identityFlags.append(isId)
            if !isId {
                nonIdentityPoints.append(p)
                nonIdentityMap.append(i)
            }
        }

        let affinePoints = batchToAffine(nonIdentityPoints)
        var lookup = [(x: [UInt64], y: [UInt64])?](repeating: nil, count: points.count)
        for (i, origIdx) in nonIdentityMap.enumerated() {
            lookup[origIdx] = (fpToInt(affinePoints[i].x), fpToInt(affinePoints[i].y))
        }
        return (identityFlags, lookup)
    }

    /// Absorb a projective point into transcript.
    private func varunaAbsorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        let aff = batchToAffine([p])
        transcript.absorb(Fr.from64(fpToInt(aff[0].x)))
        transcript.absorb(Fr.from64(fpToInt(aff[0].y)))
    }

    // MARK: - Sumcheck (reused from MarlinVerifier)

    /// Precomputed inverse of 2 in Fr.
    private static let inv2: Fr = {
        var result = [UInt64](repeating: 0, count: 4)
        bn254_fr_inverse(frFromInt(2).to64(), &result)
        return Fr.from64(result)
    }()

    /// Verify sumcheck round polynomial consistency.
    private func verifySumcheckRounds(_ roundPolys: [[Fr]], alpha: Fr) -> Bool {
        guard !roundPolys.isEmpty else { return false }
        var challenges = [Fr]()
        var chalSeed = alpha
        for _ in 0..<roundPolys.count {
            challenges.append(chalSeed)
            var newSeed = [UInt64](repeating: 0, count: 4)
            bn254_fr_mul(chalSeed.to64(), alpha.to64(), &newSeed)
            chalSeed = Fr.from64(newSeed)
        }

        var firstSum = [UInt64](repeating: 0, count: 4)
        bn254_fr_add(roundPolys[0][0].to64(), roundPolys[0][1].to64(), &firstSum)
        if firstSum != [0,0,0,0] { return false }

        for i in 0..<(roundPolys.count - 1) {
            let siRi = evaluateDeg2Poly(roundPolys[i], at: challenges[i])
            var nextSum = [UInt64](repeating: 0, count: 4)
            bn254_fr_add(roundPolys[i + 1][0].to64(), roundPolys[i + 1][1].to64(), &nextSum)
            if siRi.to64() != nextSum.map({ $0 }) { return false }
        }
        return true
    }

    /// Evaluate degree-2 polynomial [f(0), f(1), f(2)] at point r via Lagrange interpolation.
    private func evaluateDeg2Poly(_ coeffs: [Fr], at r: Fr) -> Fr {
        guard coeffs.count >= 3 else { return Fr.zero }
        let f0 = coeffs[0], f1 = coeffs[1], f2 = coeffs[2]
        let inv2 = VarunaVerifier.inv2

        var rL = r.to64()
        var rM1 = [UInt64](repeating: 0, count: 4)
        var rM2 = [UInt64](repeating: 0, count: 4)
        bn254_fr_sub(rL, Fr.one.to64(), &rM1)
        bn254_fr_sub(rL, frFromInt(2).to64(), &rM2)

        var tmp = [UInt64](repeating: 0, count: 4)
        var t0 = [UInt64](repeating: 0, count: 4)
        bn254_fr_mul(rM1, rM2, &tmp)
        bn254_fr_mul(tmp, inv2.to64(), &tmp)
        bn254_fr_mul(f0.to64(), tmp, &t0)

        var negF1 = [UInt64](repeating: 0, count: 4)
        bn254_fr_neg(f1.to64(), &negF1)
        var t1 = [UInt64](repeating: 0, count: 4)
        bn254_fr_mul(rL, rM2, &tmp)
        bn254_fr_mul(negF1, tmp, &t1)

        var t2 = [UInt64](repeating: 0, count: 4)
        bn254_fr_mul(rL, rM1, &tmp)
        bn254_fr_mul(tmp, inv2.to64(), &tmp)
        bn254_fr_mul(f2.to64(), tmp, &t2)

        var result = [UInt64](repeating: 0, count: 4)
        bn254_fr_add(t0, t1, &result)
        bn254_fr_add(result, t2, &result)
        return Fr.from64(result)
    }
}
