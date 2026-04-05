// Cross-Scheme Batch Verifier — verify mixed Groth16 + Plonk + KZG proofs
// in a single multi-pairing check.
//
// Key insight: Groth16, Plonk, and KZG all use BN254 pairings. Each scheme
// produces one or more pairing equations of the form:
//   product_i e(A_i, B_i) = 1    (rearranged to product-of-pairings = 1)
//
// To batch across schemes, we:
//   1. Assign a random challenge per proof (Fiat-Shamir over all proofs)
//   2. Scale each scheme's G1 points by its challenge
//   3. Accumulate all (G1, G2) pairs into a single list
//   4. One call to cBN254PairingCheck — single product of Miller loops + final exp
//
// Cost savings:
//   - Individual: N_groth16 * 4 pairings + N_plonk * 2 pairings + N_kzg * 2 pairings
//   - Batched:    (N_groth16 * 4 + N_plonk * 2 + N_kzg * 2) Miller loops + 1 final exp
//   - The final exponentiation (~40% of pairing cost) is done once instead of per-proof
//
// Security: random linear combination ensures that with overwhelming probability
// (1 - 1/|Fr|), if any individual proof is invalid, the batch check fails.

import Foundation

// MARK: - Cross-Scheme Proof Wrappers

/// A Plonk proof bundled with its verification data for cross-scheme batching.
/// Includes the two KZG opening pairing equations that Plonk verification reduces to.
///
/// Plonk verification (with pairings) checks two equations:
///   1. e(W_zeta, [s]_2 - zeta*[1]_2) = e(F - y*G, [1]_2)
///      Rearranged: e(W_zeta, [s]_2 - zeta*[1]_2) * e(-(F - y*G), [1]_2) = 1
///   2. e(W_{zeta*omega}, [s]_2 - zeta*omega*[1]_2) = e(Z - z_omega*G, [1]_2)
///
/// We pre-compute the G1/G2 pairs so the batch verifier just scales and accumulates.
public struct PlonkPairingData {
    /// First opening: (W_zeta, [s]_2 - zeta*[1]_2) and (-(F - y*G), [1]_2)
    public let w1: PointProjective       // W_zeta
    public let w1G2: G2ProjectivePoint   // [s]_2 - zeta*[1]_2
    public let f1: PointProjective       // -(F - y*G)  (negated so product = 1)
    /// Second opening: (W_{zeta*omega}, [s]_2 - zeta*omega*[1]_2) and (-(Z - z_omega*G), [1]_2)
    public let w2: PointProjective       // W_{zeta*omega}
    public let w2G2: G2ProjectivePoint   // [s]_2 - zeta*omega*[1]_2
    public let f2: PointProjective       // -(Z - z_omega*G)
    /// G2 generator (shared across openings)
    public let g2Gen: G2ProjectivePoint

    public init(w1: PointProjective, w1G2: G2ProjectivePoint, f1: PointProjective,
                w2: PointProjective, w2G2: G2ProjectivePoint, f2: PointProjective,
                g2Gen: G2ProjectivePoint) {
        self.w1 = w1; self.w1G2 = w1G2; self.f1 = f1
        self.w2 = w2; self.w2G2 = w2G2; self.f2 = f2
        self.g2Gen = g2Gen
    }
}

/// A KZG opening proof bundled for cross-scheme batching.
///
/// KZG verification checks:
///   e(C - [y]*G, [1]_2) = e(pi, [s]_2 - [z]*[1]_2)
///
/// Rearranged: e(C - [y]*G, [1]_2) * e(-pi, [s]_2 - [z]*[1]_2) = 1
public struct KZGPairingData {
    /// C - [y]*G  (commitment minus evaluation-times-generator)
    public let lhs: PointProjective
    /// [1]_2
    public let g2Gen: G2ProjectivePoint
    /// -pi  (negated proof point)
    public let negProof: PointProjective
    /// [s]_2 - [z]*[1]_2
    public let sMinusZG2: G2ProjectivePoint

    public init(lhs: PointProjective, g2Gen: G2ProjectivePoint,
                negProof: PointProjective, sMinusZG2: G2ProjectivePoint) {
        self.lhs = lhs; self.g2Gen = g2Gen
        self.negProof = negProof; self.sMinusZG2 = sMinusZG2
    }
}

// MARK: - Cross-Scheme Batch Verifier

public struct CrossSchemeBatchVerifier {

    public init() {}

    // MARK: - Main batch verification entry point

    /// Batch verify a mixed set of Groth16, Plonk, and KZG proofs in a single
    /// multi-pairing check.
    ///
    /// All proof types must be over BN254. The method:
    /// 1. Derives per-proof random challenges via Fiat-Shamir
    /// 2. For each Groth16 proof: scales its 4 pairing pairs by challenge r_i
    /// 3. For each Plonk proof: scales its 4 pairing pairs (2 openings x 2 sides) by challenge
    /// 4. For each KZG proof: scales its 2 pairing pairs by challenge
    /// 5. Checks product of all scaled Miller loops = 1 via single final exponentiation
    ///
    /// - Parameters:
    ///   - groth16: Array of (proof, verification key, public inputs) tuples
    ///   - plonk: Array of pre-computed Plonk pairing data
    ///   - kzg: Array of pre-computed KZG pairing data
    /// - Returns: true if all proofs are valid (with overwhelming probability)
    public func batchVerify(
        groth16: [(proof: Groth16Proof, vk: Groth16VerificationKey, inputs: [Fr])],
        plonk: [PlonkPairingData],
        kzg: [KZGPairingData]
    ) -> Bool {
        let nG = groth16.count
        let nP = plonk.count
        let nK = kzg.count
        let totalProofs = nG + nP + nK
        guard totalProofs > 0 else { return true }

        // --- Derive random challenges via Fiat-Shamir ---
        let transcript = Transcript(label: "cross-scheme-batch-verify", backend: .poseidon2)

        // Domain-separate each scheme with a tag
        transcript.absorb(frFromInt(UInt64(nG)))
        transcript.absorb(frFromInt(UInt64(nP)))
        transcript.absorb(frFromInt(UInt64(nK)))

        // Absorb Groth16 proof data
        for (proof, _, inputs) in groth16 {
            if let aAff = pointToAffine(proof.a) {
                transcript.absorb(fpToFr(aAff.x))
                transcript.absorb(fpToFr(aAff.y))
            }
            if let cAff = pointToAffine(proof.c) {
                transcript.absorb(fpToFr(cAff.x))
                transcript.absorb(fpToFr(cAff.y))
            }
            for inp in inputs {
                transcript.absorb(inp)
            }
        }

        // Absorb Plonk pairing data
        for pd in plonk {
            if let w1Aff = pointToAffine(pd.w1) {
                transcript.absorb(fpToFr(w1Aff.x))
                transcript.absorb(fpToFr(w1Aff.y))
            }
            if let f1Aff = pointToAffine(pd.f1) {
                transcript.absorb(fpToFr(f1Aff.x))
                transcript.absorb(fpToFr(f1Aff.y))
            }
        }

        // Absorb KZG pairing data
        for kd in kzg {
            if let lhsAff = pointToAffine(kd.lhs) {
                transcript.absorb(fpToFr(lhsAff.x))
                transcript.absorb(fpToFr(lhsAff.y))
            }
        }

        // Squeeze one challenge per proof
        let challenges = transcript.squeezeN(totalProofs)

        // --- Accumulate all pairing pairs ---
        // Each pairing equation of the form product e(A_i, B_i) = 1
        // gets its G1 points scaled by the proof's challenge scalar.
        // G2 points are NOT scaled (scaling G2 is expensive and unnecessary:
        // we scale G1 only, which has the same effect on the pairing product).

        var pairs = [(PointAffine, G2AffinePoint)]()
        // Reserve: Groth16 = 4 pairs each, Plonk = 4 pairs each, KZG = 2 pairs each
        pairs.reserveCapacity(nG * 4 + nP * 4 + nK * 2)

        var challengeIdx = 0

        // --- Groth16 proofs ---
        // Each Groth16 proof checks:
        //   e(-A, B) * e(alpha, beta) * e(L, gamma) * e(C, delta) = 1
        // where L = ic[0] + sum(input_j * ic[j+1])
        //
        // With challenge r_i, we scale all G1 points by r_i:
        //   e(-r_i*A, B) * e(r_i*alpha, beta) * e(r_i*L, gamma) * e(r_i*C, delta)
        for i in 0..<nG {
            let r = challenges[challengeIdx]; challengeIdx += 1
            let proof = groth16[i].proof
            let vk = groth16[i].vk
            let inputs = groth16[i].inputs

            guard inputs.count + 1 == vk.ic.count else { return false }

            // Compute L = ic[0] + sum(input_j * ic[j+1])
            var li = vk.ic[0]
            for j in 0..<inputs.count {
                if !inputs[j].isZero {
                    li = pointAdd(li, pointScalarMul(vk.ic[j + 1], inputs[j]))
                }
            }

            // Scale G1 points by challenge r
            let negRiA = pointNeg(cPointScalarMul(proof.a, r))
            let riAlpha = cPointScalarMul(vk.alpha_g1, r)
            let riL = cPointScalarMul(li, r)
            let riC = cPointScalarMul(proof.c, r)

            // Convert to affine
            guard let nA = pointToAffine(negRiA),
                  let ra = pointToAffine(riAlpha),
                  let rl = pointToAffine(riL),
                  let rc = pointToAffine(riC) else { return false }
            guard let pB = g2ToAffine(proof.b),
                  let be = g2ToAffine(vk.beta_g2),
                  let ga = g2ToAffine(vk.gamma_g2),
                  let de = g2ToAffine(vk.delta_g2) else { return false }

            pairs.append((nA, pB))
            pairs.append((ra, be))
            pairs.append((rl, ga))
            pairs.append((rc, de))
        }

        // --- Plonk proofs ---
        // Each Plonk proof has two KZG opening equations (4 pairing pairs total):
        //   Opening 1: e(W_zeta, [s-zeta]_2) * e(-(F-y*G), [1]_2) = 1
        //   Opening 2: e(W_{zeta*omega}, [s-zeta*omega]_2) * e(-(Z-z_omega*G), [1]_2) = 1
        //
        // With challenge r_i, scale G1 points by r_i.
        for i in 0..<nP {
            let r = challenges[challengeIdx]; challengeIdx += 1
            let pd = plonk[i]

            let rW1 = cPointScalarMul(pd.w1, r)
            let rF1 = cPointScalarMul(pd.f1, r)
            let rW2 = cPointScalarMul(pd.w2, r)
            let rF2 = cPointScalarMul(pd.f2, r)

            guard let w1Aff = pointToAffine(rW1),
                  let f1Aff = pointToAffine(rF1),
                  let w2Aff = pointToAffine(rW2),
                  let f2Aff = pointToAffine(rF2) else { return false }
            guard let w1G2Aff = g2ToAffine(pd.w1G2),
                  let g2GenAff = g2ToAffine(pd.g2Gen),
                  let w2G2Aff = g2ToAffine(pd.w2G2) else { return false }

            pairs.append((w1Aff, w1G2Aff))
            pairs.append((f1Aff, g2GenAff))
            pairs.append((w2Aff, w2G2Aff))
            pairs.append((f2Aff, g2GenAff))
        }

        // --- KZG proofs ---
        // Each KZG opening checks:
        //   e(C - [y]*G, [1]_2) * e(-pi, [s-z]_2) = 1
        //
        // With challenge r_i, scale G1 points by r_i.
        for i in 0..<nK {
            let r = challenges[challengeIdx]; challengeIdx += 1
            let kd = kzg[i]

            let rLhs = cPointScalarMul(kd.lhs, r)
            let rNegProof = cPointScalarMul(kd.negProof, r)

            guard let lhsAff = pointToAffine(rLhs),
                  let npAff = pointToAffine(rNegProof) else { return false }
            guard let g2GenAff = g2ToAffine(kd.g2Gen),
                  let szAff = g2ToAffine(kd.sMinusZG2) else { return false }

            pairs.append((lhsAff, g2GenAff))
            pairs.append((npAff, szAff))
        }

        // --- Single multi-pairing check ---
        return cBN254PairingCheck(pairs)
    }

    // MARK: - Convenience: Groth16-only batch (delegates to existing)

    /// Batch verify Groth16 proofs only (convenience wrapper).
    public func batchVerifyGroth16Only(
        proofs: [(proof: Groth16Proof, vk: Groth16VerificationKey, inputs: [Fr])]
    ) -> Bool {
        return batchVerify(groth16: proofs, plonk: [], kzg: [])
    }

    // MARK: - Plonk pairing data extraction

    /// Extract pairing data from a Plonk proof and its verification setup.
    ///
    /// This re-runs the Plonk verifier logic to reconstruct the commitment F
    /// and evaluation y, then packages the two KZG opening equations as pairing data.
    ///
    /// Requires the SRS G2 points:
    ///   - g2Gen: [1]_2 (G2 generator)
    ///   - sG2:   [s]_2 (SRS secret times G2 generator)
    public static func extractPlonkPairingData(
        proof: PlonkProof,
        setup: PlonkSetup,
        g2Gen: G2ProjectivePoint,
        sG2: G2ProjectivePoint
    ) -> PlonkPairingData? {
        let n = setup.n
        let omega = setup.omega
        let k1 = setup.k1
        let k2 = setup.k2

        // --- Reconstruct transcript challenges (same as PlonkVerifier) ---
        let transcript = Transcript(label: "plonk", backend: .keccak256)

        for c in setup.selectorCommitments {
            absorbPointForCross(transcript, c)
        }
        for c in setup.permutationCommitments {
            absorbPointForCross(transcript, c)
        }

        // Round 1
        absorbPointForCross(transcript, proof.aCommit)
        absorbPointForCross(transcript, proof.bCommit)
        absorbPointForCross(transcript, proof.cCommit)

        // Round 2
        let beta = transcript.squeeze()
        let gamma = transcript.squeeze()

        absorbPointForCross(transcript, proof.zCommit)

        // Round 3
        let alpha = transcript.squeeze()
        let alpha2 = frSqr(alpha)

        absorbPointForCross(transcript, proof.tLoCommit)
        absorbPointForCross(transcript, proof.tMidCommit)
        absorbPointForCross(transcript, proof.tHiCommit)
        for extraCommit in proof.tExtraCommits {
            absorbPointForCross(transcript, extraCommit)
        }

        // Round 4
        let zeta = transcript.squeeze()

        transcript.absorb(proof.aEval)
        transcript.absorb(proof.bEval)
        transcript.absorb(proof.cEval)
        transcript.absorb(proof.sigma1Eval)
        transcript.absorb(proof.sigma2Eval)
        transcript.absorb(proof.zOmegaEval)

        // Round 5
        let v = transcript.squeeze()

        // --- Compute linearization commitment (same as PlonkVerifier) ---
        let zetaN = frPow(zeta, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)
        let nInv = frInverse(frFromInt(UInt64(n)))
        let zetaMinusOne = frSub(zeta, Fr.one)
        guard !zetaMinusOne.isZero else { return nil }
        let l1Zeta = frMul(zhZeta, frMul(nInv, frInverse(zetaMinusOne)))

        let permNum = frMul(
            frMul(frAdd(frAdd(proof.aEval, frMul(beta, zeta)), gamma),
                  frAdd(frAdd(proof.bEval, frMul(beta, frMul(k1, zeta))), gamma)),
            frAdd(frAdd(proof.cEval, frMul(beta, frMul(k2, zeta))), gamma)
        )
        let permDenPartial = frMul(
            frMul(frAdd(frAdd(proof.aEval, frMul(beta, proof.sigma1Eval)), gamma),
                  frAdd(frAdd(proof.bEval, frMul(beta, proof.sigma2Eval)), gamma)),
            frMul(beta, proof.zOmegaEval)
        )

        let abEval = frMul(proof.aEval, proof.bEval)
        var rCommit = pointIdentity()

        // Selector part
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[3], abEval))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[0], proof.aEval))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[1], proof.bEval))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[2], proof.cEval))
        rCommit = pointAdd(rCommit, setup.selectorCommitments[4])

        // Custom gate selectors
        let aEvalSq = frSqr(proof.aEval)
        let rangeScalar = frSub(proof.aEval, aEvalSq)
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[5], rangeScalar))

        var lookupScalar = Fr.zero
        for table in setup.lookupTables {
            if table.values.isEmpty { continue }
            var prod = Fr.one
            for tVal in table.values {
                prod = frMul(prod, frSub(proof.aEval, tVal))
            }
            lookupScalar = frAdd(lookupScalar, prod)
        }
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[6], lookupScalar))

        let bEvalSq = frSqr(proof.bEval)
        let poseidonScalar = frSub(proof.cEval, frMul(proof.aEval, bEvalSq))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[7], poseidonScalar))

        // Permutation part
        let zCoeff = frAdd(frMul(alpha, permNum), frMul(alpha2, l1Zeta))
        rCommit = pointAdd(rCommit, cPointScalarMul(proof.zCommit, zCoeff))

        let sigma3Coeff = frSub(Fr.zero, frMul(alpha, permDenPartial))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.permutationCommitments[2], sigma3Coeff))

        // Quotient part
        var tCommit = proof.tLoCommit
        var zetaNPow = zetaN
        tCommit = pointAdd(tCommit, cPointScalarMul(proof.tMidCommit, zetaNPow))
        zetaNPow = frMul(zetaNPow, zetaN)
        tCommit = pointAdd(tCommit, cPointScalarMul(proof.tHiCommit, zetaNPow))
        for extraCommit in proof.tExtraCommits {
            zetaNPow = frMul(zetaNPow, zetaN)
            tCommit = pointAdd(tCommit, cPointScalarMul(extraCommit, zetaNPow))
        }
        rCommit = pointAdd(rCommit, cPointScalarMul(tCommit, frSub(Fr.zero, zhZeta)))

        // --- Build combined commitment F ---
        var fCommit = rCommit
        var vPow = v
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.aCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.bCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.cCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(setup.permutationCommitments[0], vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(setup.permutationCommitments[1], vPow))

        // Combined evaluation
        let rZeta = computeLinearizationEvalForCross(
            proof: proof, alpha: alpha, beta: beta, gamma: gamma, l1Zeta: l1Zeta)
        var combinedEval = rZeta
        vPow = v
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.aEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.bEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.cEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.sigma1Eval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.sigma2Eval))

        // --- Build pairing data ---
        // Opening 1: e(W_zeta, [s]_2 - zeta*[1]_2) = e(F - y*G, [1]_2)
        // Rearranged: e(W_zeta, [s]_2 - zeta*[1]_2) * e(-(F - y*G), [1]_2) = 1
        let g1 = pointFromAffine(setup.srs[0])
        let zetaG2 = g2ScalarMul(g2Gen, frToInt(zeta))
        let w1G2 = g2Add(sG2, g2Negate(zetaG2))  // [s-zeta]_2

        let fMinusYG = pointAdd(fCommit, cPointScalarMul(g1, frSub(Fr.zero, combinedEval)))
        let negFMinusYG = pointNeg(fMinusYG)

        // Opening 2: e(W_{zeta*omega}, [s]_2 - (zeta*omega)*[1]_2) = e(Z - z_omega*G, [1]_2)
        let zetaOmega = frMul(zeta, omega)
        let zetaOmegaG2 = g2ScalarMul(g2Gen, frToInt(zetaOmega))
        let w2G2 = g2Add(sG2, g2Negate(zetaOmegaG2))  // [s-zeta*omega]_2

        let zMinusEvalG = pointAdd(proof.zCommit, cPointScalarMul(g1, frSub(Fr.zero, proof.zOmegaEval)))
        let negZMinusEvalG = pointNeg(zMinusEvalG)

        return PlonkPairingData(
            w1: proof.openingProof, w1G2: w1G2, f1: negFMinusYG,
            w2: proof.shiftedOpeningProof, w2G2: w2G2, f2: negZMinusEvalG,
            g2Gen: g2Gen
        )
    }

    // MARK: - KZG pairing data extraction

    /// Extract pairing data from a KZG opening for cross-scheme batching.
    ///
    /// Given: commitment C, evaluation y at point z, proof pi.
    /// Pairing equation: e(C - [y]*G, [1]_2) * e(-pi, [s-z]_2) = 1
    ///
    /// - Parameters:
    ///   - commitment: KZG polynomial commitment C
    ///   - point: evaluation point z
    ///   - value: claimed evaluation y = p(z)
    ///   - proof: KZG opening proof pi
    ///   - srs0: SRS generator G (first SRS point)
    ///   - g2Gen: [1]_2
    ///   - sG2: [s]_2
    /// - Returns: Pairing data for batch verification
    public static func extractKZGPairingData(
        commitment: PointProjective,
        point: Fr,
        value: Fr,
        proof: PointProjective,
        srs0: PointAffine,
        g2Gen: G2ProjectivePoint,
        sG2: G2ProjectivePoint
    ) -> KZGPairingData {
        let g1 = pointFromAffine(srs0)

        // C - [y]*G
        let lhs = pointAdd(commitment, cPointScalarMul(g1, frSub(Fr.zero, value)))

        // -pi
        let negProof = pointNeg(proof)

        // [s]_2 - [z]*[1]_2
        let zG2 = g2ScalarMul(g2Gen, frToInt(point))
        let sMinusZG2 = g2Add(sG2, g2Negate(zG2))

        return KZGPairingData(
            lhs: lhs, g2Gen: g2Gen,
            negProof: negProof, sMinusZG2: sMinusZG2
        )
    }

    // MARK: - Helpers

    /// Compute r(zeta) for Plonk linearization (matches PlonkVerifier logic).
    private static func computeLinearizationEvalForCross(
        proof: PlonkProof, alpha: Fr, beta: Fr, gamma: Fr, l1Zeta: Fr
    ) -> Fr {
        let alpha2 = frSqr(alpha)
        let term1 = frAdd(frAdd(proof.aEval, frMul(beta, proof.sigma1Eval)), gamma)
        let term2 = frAdd(frAdd(proof.bEval, frMul(beta, proof.sigma2Eval)), gamma)
        let term3 = frAdd(proof.cEval, gamma)
        let permCorr = frMul(frMul(frMul(term1, term2), term3), proof.zOmegaEval)
        return frAdd(frMul(alpha, permCorr), frMul(alpha2, l1Zeta))
    }
}

// MARK: - Transcript helpers (module-private)

/// Absorb a projective point into transcript (for cross-scheme Plonk extraction).
private func absorbPointForCross(_ transcript: Transcript, _ p: PointProjective) {
    if let aff = pointToAffine(p) {
        transcript.absorb(fpToFr(aff.x))
        transcript.absorb(fpToFr(aff.y))
    }
}
