// PlonkVerifier — O(1) Plonk proof verification
//
// Reconstructs challenges from transcript, checks the polynomial identity
// using KZG opening verification. Without a pairing engine, we verify
// using the known SRS secret (suitable for testing; production would use
// e(W, [s]_2 - zeta*[1]_2) == e(C - y*[1]_1, [1]_2)).

import Foundation
import NeonFieldOps

public class PlonkVerifier {
    public let setup: PlonkSetup
    public let kzg: KZGEngine

    public init(setup: PlonkSetup, kzg: KZGEngine) {
        self.setup = setup
        self.kzg = kzg
    }

    /// Verify a Plonk proof. Returns true if valid.
    public func verify(proof: PlonkProof) -> Bool {
        let n = setup.n
        let omega = setup.omega
        let k1 = setup.k1
        let k2 = setup.k2

        // --- Reconstruct transcript challenges ---
        let transcript = Transcript(label: "plonk", backend: .keccak256)

        for c in setup.selectorCommitments {
            absorbPoint(transcript, c)
        }
        for c in setup.permutationCommitments {
            absorbPoint(transcript, c)
        }

        // Round 1
        absorbPoint(transcript, proof.aCommit)
        absorbPoint(transcript, proof.bCommit)
        absorbPoint(transcript, proof.cCommit)

        // Round 2
        let beta = transcript.squeeze()
        let gamma = transcript.squeeze()

        absorbPoint(transcript, proof.zCommit)

        // Round 3
        let alpha = transcript.squeeze()
        let alpha2 = frSqr(alpha)

        absorbPoint(transcript, proof.tLoCommit)
        absorbPoint(transcript, proof.tMidCommit)
        absorbPoint(transcript, proof.tHiCommit)
        for extraCommit in proof.tExtraCommits {
            absorbPoint(transcript, extraCommit)
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

        // --- Compute evaluation of linearization at zeta ---

        // L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
        let zetaN = frPow(zeta, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)
        let nInv = frInverse(frFromInt(UInt64(n)))
        let zetaMinusOne = frSub(zeta, Fr.one)
        // Guard: if zeta == 1 (astronomically unlikely), bail
        if zetaMinusOne.isZero { return false }
        let l1Zeta = frMul(zhZeta, frMul(nInv, frInverse(zetaMinusOne)))

        // Gate constraint check at zeta:
        // r_gate = a*b*qM(zeta) + a*qL(zeta) + b*qR(zeta) + c*qO(zeta) + qC(zeta)
        // But we don't have qM(zeta) etc. directly from proof — we recompute from commitments.

        // Permutation constraint scalars
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

        // Reconstruct the commitment to r(x) from proof components
        // [r] = ab*[qM] + a*[qL] + b*[qR] + c*[qO] + [qC]
        //      + alpha*(permNum*[z] - permDenPartial*[sigma3])
        //      + alpha^2*L_1(zeta)*[z]
        //      - Z_H(zeta)*(tLo + zeta^n*tMid + zeta^{2n}*tHi)
        let abEval = frMul(proof.aEval, proof.bEval)

        var rCommit = pointIdentity()

        // Selector part
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[3], abEval))  // qM
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[0], proof.aEval))  // qL
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[1], proof.bEval))  // qR
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[2], proof.cEval))  // qO
        rCommit = pointAdd(rCommit, setup.selectorCommitments[4])  // qC (coeff = 1)

        // Custom gate selectors
        // Range: scalar = a*(1-a) = a - a^2
        let aEvalSq = frSqr(proof.aEval)
        let rangeScalar = frSub(proof.aEval, aEvalSq)
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[5], rangeScalar))  // qRange

        // Lookup: scalar = prod(a - t_i) for all table values
        var lookupScalar = Fr.zero
        for table in setup.lookupTables {
            if table.values.isEmpty { continue }
            var prod = Fr.one
            for tVal in table.values {
                prod = frMul(prod, frSub(proof.aEval, tVal))
            }
            lookupScalar = frAdd(lookupScalar, prod)
        }
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[6], lookupScalar))  // qLookup

        // Poseidon: scalar = c - a*b^2
        let bEvalSq = frSqr(proof.bEval)
        let poseidonScalar = frSub(proof.cEval, frMul(proof.aEval, bEvalSq))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.selectorCommitments[7], poseidonScalar))  // qPoseidon

        // Permutation part with z
        let zCoeff = frAdd(frMul(alpha, permNum), frMul(alpha2, l1Zeta))
        rCommit = pointAdd(rCommit, cPointScalarMul(proof.zCommit, zCoeff))

        // Permutation part with sigma3
        let sigma3Coeff = frSub(Fr.zero, frMul(alpha, permDenPartial))
        rCommit = pointAdd(rCommit, cPointScalarMul(setup.permutationCommitments[2], sigma3Coeff))

        // Quotient part: sum_k(zeta^{k*n} * [t_k])
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

        // --- Verify KZG openings ---
        // We need to verify:
        //   1. W_zeta proves that r + v*a + v^2*b + v^3*c + v^4*sigma1 + v^5*sigma2
        //      evaluates correctly at zeta
        //   2. W_{zeta*omega} proves z evaluates correctly at zeta*omega

        // Compute r(zeta) from the polynomial identity
        // r(zeta) should be zero if the proof is correct (the numerator vanishes)
        // Actually r(zeta) is what we can derive:
        // r(zeta) = gate + alpha*perm + alpha^2*boundary - Z_H(zeta)*t(zeta)
        // This should equal 0. We verify via the opening proof instead.

        // Combined commitment for batch opening at zeta:
        // F = [r] + v*[a] + v^2*[b] + v^3*[c] + v^4*[sigma1] + v^5*[sigma2]
        var fCommit = rCommit
        var vPow = v
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.aCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.bCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(proof.cCommit, vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(setup.permutationCommitments[0], vPow)); vPow = frMul(vPow, v)
        fCommit = pointAdd(fCommit, cPointScalarMul(setup.permutationCommitments[1], vPow))

        // Combined evaluation at zeta:
        // Compute r(zeta) = (evaluate the linearization)
        // Since the prover computed r(x) and opened it, r(zeta) is implicitly
        // verified through the opening. We reconstruct the expected evaluation:
        let rZeta = computeLinearizationEval(proof: proof, alpha: alpha, beta: beta, gamma: gamma,
                                              zeta: zeta, k1: k1, k2: k2, l1Zeta: l1Zeta, zhZeta: zhZeta)

        var combinedEval = rZeta
        vPow = v
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.aEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.bEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.cEval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.sigma1Eval)); vPow = frMul(vPow, v)
        combinedEval = frAdd(combinedEval, frMul(vPow, proof.sigma2Eval))

        // Verify opening at zeta: e(W_zeta, [s-zeta]) == e(F - y*G, G2)
        // Using SRS secret: F - y*G == (s - zeta) * W_zeta
        let s = setup.srsSecret
        let g1 = pointFromAffine(setup.srs[0])
        let sMinusZeta = frSub(s, zeta)

        let lhs1 = cPointScalarMul(proof.openingProof, sMinusZeta)
        let rhs1 = pointAdd(fCommit, cPointScalarMul(g1, frSub(Fr.zero, combinedEval)))

        let lhs1Aff = batchToAffine([lhs1])
        let rhs1Aff = batchToAffine([rhs1])

        let check1 = fpToInt(lhs1Aff[0].x) == fpToInt(rhs1Aff[0].x) &&
                      fpToInt(lhs1Aff[0].y) == fpToInt(rhs1Aff[0].y)

        if !check1 { return false }

        // Verify shifted opening at zeta*omega
        let zetaOmega = frMul(zeta, omega)
        let sMinusZetaOmega = frSub(s, zetaOmega)

        let lhs2 = cPointScalarMul(proof.shiftedOpeningProof, sMinusZetaOmega)
        let rhs2 = pointAdd(proof.zCommit, cPointScalarMul(g1, frSub(Fr.zero, proof.zOmegaEval)))

        let lhs2Aff = batchToAffine([lhs2])
        let rhs2Aff = batchToAffine([rhs2])

        let check2 = fpToInt(lhs2Aff[0].x) == fpToInt(rhs2Aff[0].x) &&
                      fpToInt(lhs2Aff[0].y) == fpToInt(rhs2Aff[0].y)

        return check2
    }

    /// Compute the evaluation of the linearization polynomial r at zeta.
    ///
    /// The linearization keeps sigma3(x) and z(x) as polynomials, substituting
    /// known evaluations for the other wires. At x=zeta, this differs from the
    /// full (zero) constraint by two correction terms:
    ///   1. Permutation: alpha * (a+beta*s1+gamma)(b+beta*s2+gamma)(c+gamma)*z(zw)
    ///   2. Boundary:    alpha^2 * L_1(zeta)
    /// giving r(zeta) = alpha*permCorr + alpha^2*L_1(zeta).
    private func computeLinearizationEval(proof: PlonkProof, alpha: Fr, beta: Fr, gamma: Fr,
                                           zeta: Fr, k1: Fr, k2: Fr, l1Zeta: Fr, zhZeta: Fr) -> Fr {
        _ = (zeta, k1, k2, zhZeta)  // unused in this calculation
        let alpha2 = frSqr(alpha)
        let term1 = frAdd(frAdd(proof.aEval, frMul(beta, proof.sigma1Eval)), gamma)
        let term2 = frAdd(frAdd(proof.bEval, frMul(beta, proof.sigma2Eval)), gamma)
        let term3 = frAdd(proof.cEval, gamma)
        let permCorr = frMul(frMul(frMul(term1, term2), term3), proof.zOmegaEval)
        // Note: custom gate contributions (range, lookup, poseidon) are fully determined
        // by wire evaluations and don't add correction terms to the linearization eval,
        // because the custom selector polynomials are opened via their commitments
        // with known scalar multipliers in the commitment reconstruction.
        return frAdd(frMul(alpha, permCorr), frMul(alpha2, l1Zeta))
    }
}
