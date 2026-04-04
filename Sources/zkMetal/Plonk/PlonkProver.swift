// PlonkProver — Plonk proof generation
//
// Implements the 5-round Plonk protocol:
//   Round 1: Commit to witness polynomials a(x), b(x), c(x)
//   Round 2: Compute and commit to permutation accumulator z(x)
//   Round 3: Compute and commit to quotient polynomial t(x)
//   Round 4: Evaluate polynomials at challenge zeta
//   Round 5: Compute linearization and KZG opening proofs
//
// All polynomial arithmetic uses NTT for O(n log n) operations.

import Foundation
import NeonFieldOps

// MARK: - Proof

public struct PlonkProof {
    public let aCommit: PointProjective
    public let bCommit: PointProjective
    public let cCommit: PointProjective
    public let zCommit: PointProjective          // permutation accumulator
    public let tLoCommit: PointProjective        // quotient poly, low degree chunk
    public let tMidCommit: PointProjective       // quotient poly, mid degree chunk
    public let tHiCommit: PointProjective        // quotient poly, high degree chunk
    public let aEval: Fr                         // a(zeta)
    public let bEval: Fr                         // b(zeta)
    public let cEval: Fr                         // c(zeta)
    public let sigma1Eval: Fr                    // sigma1(zeta)
    public let sigma2Eval: Fr                    // sigma2(zeta)
    public let zOmegaEval: Fr                    // z(zeta * omega)
    public let openingProof: PointProjective     // W_zeta
    public let shiftedOpeningProof: PointProjective  // W_{zeta*omega}
}

// MARK: - Prover

public class PlonkProver {
    public let setup: PlonkSetup
    public let kzg: KZGEngine
    public let ntt: NTTEngine

    public init(setup: PlonkSetup, kzg: KZGEngine, ntt: NTTEngine) {
        self.setup = setup
        self.kzg = kzg
        self.ntt = ntt
    }

    /// Generate a Plonk proof from a witness assignment.
    /// witness[variable_index] = field value for that variable.
    public func prove(witness: [Fr], circuit: PlonkCircuit) throws -> PlonkProof {
        let n = setup.n
        let omega = setup.omega

        // Build wire polynomials from witness and circuit wire assignments
        var aEvals = [Fr](repeating: Fr.zero, count: n)
        var bEvals = [Fr](repeating: Fr.zero, count: n)
        var cEvals = [Fr](repeating: Fr.zero, count: n)

        for i in 0..<circuit.numGates {
            let wires = circuit.wireAssignments[i]
            aEvals[i] = witness[wires[0]]
            bEvals[i] = witness[wires[1]]
            cEvals[i] = witness[wires[2]]
        }

        // --- Transcript ---
        let transcript = Transcript(label: "plonk", backend: .keccak256)

        // Absorb setup commitments
        for c in setup.selectorCommitments {
            absorbPoint(transcript, c)
        }
        for c in setup.permutationCommitments {
            absorbPoint(transcript, c)
        }

        // ========== Round 1: Witness commitments ==========

        // Add random blinding: a(x) = a_eval(x) + (b1*x + b2) * Z_H(x)
        // For simplicity (and soundness in the honest-verifier model),
        // we use the raw witness polynomials without blinding.
        let aCoeffs = try ntt.intt(aEvals)
        let bCoeffs = try ntt.intt(bEvals)
        let cCoeffs = try ntt.intt(cEvals)

        let aCommit = try kzg.commit(aCoeffs)
        let bCommit = try kzg.commit(bCoeffs)
        let cCommit = try kzg.commit(cCoeffs)

        absorbPoint(transcript, aCommit)
        absorbPoint(transcript, bCommit)
        absorbPoint(transcript, cCommit)

        // ========== Round 2: Permutation accumulator z(x) ==========

        let beta = transcript.squeeze()
        let gamma = transcript.squeeze()

        // z(omega^0) = 1
        // z(omega^{i+1}) = z(omega^i) * prod_j (w_j(omega^i) + beta*id_j(omega^i) + gamma) /
        //                                       (w_j(omega^i) + beta*sigma_j(omega^i) + gamma)
        // where id_j(omega^i) = omega^i * {1, k1, k2} (identity permutation)

        var zEvals = [Fr](repeating: Fr.zero, count: n)
        zEvals[0] = Fr.one

        let domain = setup.domain
        let k1 = setup.k1
        let k2 = setup.k2
        let sigma1Evals = setup.permutationEvals[0]
        let sigma2Evals = setup.permutationEvals[1]
        let sigma3Evals = setup.permutationEvals[2]

        for i in 0..<(n - 1) {
            // Numerator: (a + beta*omega^i + gamma)(b + beta*k1*omega^i + gamma)(c + beta*k2*omega^i + gamma)
            let num1 = frAdd(frAdd(aEvals[i], frMul(beta, domain[i])), gamma)
            let num2 = frAdd(frAdd(bEvals[i], frMul(beta, frMul(k1, domain[i]))), gamma)
            let num3 = frAdd(frAdd(cEvals[i], frMul(beta, frMul(k2, domain[i]))), gamma)
            let num = frMul(frMul(num1, num2), num3)

            // Denominator: (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma)
            let den1 = frAdd(frAdd(aEvals[i], frMul(beta, sigma1Evals[i])), gamma)
            let den2 = frAdd(frAdd(bEvals[i], frMul(beta, sigma2Evals[i])), gamma)
            let den3 = frAdd(frAdd(cEvals[i], frMul(beta, sigma3Evals[i])), gamma)
            let den = frMul(frMul(den1, den2), den3)

            zEvals[i + 1] = frMul(zEvals[i], frMul(num, frInverse(den)))
        }

        let zCoeffs = try ntt.intt(zEvals)
        let zCommit = try kzg.commit(zCoeffs)
        absorbPoint(transcript, zCommit)

        // ========== Round 3: Quotient polynomial t(x) ==========

        let alpha = transcript.squeeze()

        // The quotient polynomial satisfies:
        //   t(x) * Z_H(x) = gate_constraint(x) + alpha * perm_constraint(x) + alpha^2 * boundary_constraint(x)
        //
        // We compute t in evaluation form on a coset of size 4n, then split into 3 degree-n chunks.
        //
        // For efficiency, we compute on the evaluation domain and use the relation:
        //   t(omega^i) = [gate(omega^i) + alpha * perm(omega^i) + alpha^2 * boundary(omega^i)] / Z_H(omega^i)
        //
        // But Z_H(omega^i) = 0 on the domain! So we must work on a larger coset.
        // Instead, we compute the numerator polynomial and divide by Z_H in coefficient form.

        // Gate constraint: qL*a + qR*b + qO*c + qM*a*b + qC (in eval form)
        var gateEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let g = circuit.gates[min(i, circuit.numGates - 1)]
            var val = frMul(g.qL, aEvals[i])
            val = frAdd(val, frMul(g.qR, bEvals[i]))
            val = frAdd(val, frMul(g.qO, cEvals[i]))
            val = frAdd(val, frMul(g.qM, frMul(aEvals[i], bEvals[i])))
            val = frAdd(val, g.qC)
            gateEvals[i] = val
        }

        // Permutation constraint (in eval form):
        // (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * z(x)
        // - (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * z(omega*x)
        var permEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let nextI = (i + 1) % n

            let num1 = frAdd(frAdd(aEvals[i], frMul(beta, domain[i])), gamma)
            let num2 = frAdd(frAdd(bEvals[i], frMul(beta, frMul(k1, domain[i]))), gamma)
            let num3 = frAdd(frAdd(cEvals[i], frMul(beta, frMul(k2, domain[i]))), gamma)
            let numProd = frMul(frMul(frMul(num1, num2), num3), zEvals[i])

            let den1 = frAdd(frAdd(aEvals[i], frMul(beta, sigma1Evals[i])), gamma)
            let den2 = frAdd(frAdd(bEvals[i], frMul(beta, sigma2Evals[i])), gamma)
            let den3 = frAdd(frAdd(cEvals[i], frMul(beta, sigma3Evals[i])), gamma)
            let denProd = frMul(frMul(frMul(den1, den2), den3), zEvals[nextI])

            permEvals[i] = frSub(numProd, denProd)
        }

        // Boundary constraint: (z(x) - 1) * L_1(x) where L_1(omega^i) = n if i==0, else 0
        // Actually L_1(omega^0) = 1 in the normalized Lagrange basis
        // L_1(x) = (x^n - 1) / (n * (x - 1)). At omega^i: L_1(omega^0) = 1, L_1(omega^i) = 0 for i>0
        var boundaryEvals = [Fr](repeating: Fr.zero, count: n)
        // z(omega^0) - 1 should be 0, enforce via L_1
        // In eval form: boundary[0] = (z[0] - 1) * 1 = z[0] - 1, boundary[i] = 0
        boundaryEvals[0] = frSub(zEvals[0], Fr.one)

        // Combine: numerator = gate + alpha * perm + alpha^2 * boundary
        let alpha2 = frSqr(alpha)
        var numCoeffs: [Fr]
        do {
            var numEvals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                var val = gateEvals[i]
                val = frAdd(val, frMul(alpha, permEvals[i]))
                val = frAdd(val, frMul(alpha2, boundaryEvals[i]))
                numEvals[i] = val
            }
            numCoeffs = try ntt.intt(numEvals)
        }

        // Divide by Z_H(x) = x^n - 1 in coefficient form
        // Since the numerator vanishes on the domain, it is divisible by Z_H
        let tCoeffs = polyDivideByVanishing(numCoeffs, n: n)

        // Split t into 3 chunks of degree n: t = t_lo + x^n * t_mid + x^{2n} * t_hi
        let tLoCoeffs = Array(tCoeffs.prefix(n)) + [Fr](repeating: Fr.zero, count: max(0, n - tCoeffs.prefix(n).count))
        let tMidCoeffs: [Fr]
        let tHiCoeffs: [Fr]
        if tCoeffs.count > n {
            let mid = Array(tCoeffs.dropFirst(n).prefix(n))
            tMidCoeffs = mid + [Fr](repeating: Fr.zero, count: max(0, n - mid.count))
        } else {
            tMidCoeffs = [Fr](repeating: Fr.zero, count: n)
        }
        if tCoeffs.count > 2 * n {
            let hi = Array(tCoeffs.dropFirst(2 * n).prefix(n))
            tHiCoeffs = hi + [Fr](repeating: Fr.zero, count: max(0, n - hi.count))
        } else {
            tHiCoeffs = [Fr](repeating: Fr.zero, count: n)
        }

        let tLoCommit = try kzg.commit(tLoCoeffs)
        let tMidCommit = try kzg.commit(tMidCoeffs)
        let tHiCommit = try kzg.commit(tHiCoeffs)

        absorbPoint(transcript, tLoCommit)
        absorbPoint(transcript, tMidCommit)
        absorbPoint(transcript, tHiCommit)

        // ========== Round 4: Evaluate at challenge zeta ==========

        let zeta = transcript.squeeze()
        let zetaOmega = frMul(zeta, omega)

        let aZeta = polyEval(aCoeffs, at: zeta)
        let bZeta = polyEval(bCoeffs, at: zeta)
        let cZeta = polyEval(cCoeffs, at: zeta)
        let sigma1Zeta = polyEval(setup.permutationPolys[0], at: zeta)
        let sigma2Zeta = polyEval(setup.permutationPolys[1], at: zeta)
        let zOmegaZeta = polyEval(zCoeffs, at: zetaOmega)

        transcript.absorb(aZeta)
        transcript.absorb(bZeta)
        transcript.absorb(cZeta)
        transcript.absorb(sigma1Zeta)
        transcript.absorb(sigma2Zeta)
        transcript.absorb(zOmegaZeta)

        // ========== Round 5: Opening proofs ==========

        let v = transcript.squeeze()  // batching challenge

        // Linearization polynomial r(x):
        // r(x) = a_zeta*b_zeta*qM(x) + a_zeta*qL(x) + b_zeta*qR(x) + c_zeta*qO(x) + qC(x)
        //       + alpha * [ (a_zeta + beta*zeta + gamma)(b_zeta + beta*k1*zeta + gamma)(c_zeta + beta*k2*zeta + gamma) * z(x)
        //                 - (a_zeta + beta*sigma1_zeta + gamma)(b_zeta + beta*sigma2_zeta + gamma) * beta * z_omega_zeta * sigma3(x) ]
        //       + alpha^2 * L_1(zeta) * z(x)
        //       - Z_H(zeta) * (t_lo(x) + zeta^n * t_mid(x) + zeta^{2n} * t_hi(x))

        let qLCoeffs = setup.selectorPolys[0]
        let qRCoeffs = setup.selectorPolys[1]
        let qOCoeffs = setup.selectorPolys[2]
        let qMCoeffs = setup.selectorPolys[3]
        let qCCoeffs = setup.selectorPolys[4]
        let sigma3Coeffs = setup.permutationPolys[2]

        // Compute L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
        let zetaN = frPow(zeta, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)  // Z_H(zeta) = zeta^n - 1
        let nInv = frInverse(frFromInt(UInt64(n)))
        let l1Zeta = frMul(zhZeta, frMul(nInv, frInverse(frSub(zeta, Fr.one))))

        // Build r(x) coefficient by coefficient
        var rCoeffs = [Fr](repeating: Fr.zero, count: n)

        // Gate part: a_z*b_z*qM + a_z*qL + b_z*qR + c_z*qO + qC
        let abZeta = frMul(aZeta, bZeta)
        for i in 0..<n {
            var val = frMul(abZeta, qMCoeffs[i])
            val = frAdd(val, frMul(aZeta, qLCoeffs[i]))
            val = frAdd(val, frMul(bZeta, qRCoeffs[i]))
            val = frAdd(val, frMul(cZeta, qOCoeffs[i]))
            val = frAdd(val, qCCoeffs[i])
            rCoeffs[i] = val
        }

        // Permutation part with z(x)
        let permNum = frMul(
            frMul(frAdd(frAdd(aZeta, frMul(beta, zeta)), gamma),
                  frAdd(frAdd(bZeta, frMul(beta, frMul(k1, zeta))), gamma)),
            frAdd(frAdd(cZeta, frMul(beta, frMul(k2, zeta))), gamma)
        )

        let permDenPartial = frMul(
            frMul(frAdd(frAdd(aZeta, frMul(beta, sigma1Zeta)), gamma),
                  frAdd(frAdd(bZeta, frMul(beta, sigma2Zeta)), gamma)),
            frMul(beta, zOmegaZeta)
        )

        for i in 0..<n {
            // + alpha * permNum * z[i]
            rCoeffs[i] = frAdd(rCoeffs[i], frMul(alpha, frMul(permNum, zCoeffs[i])))
            // - alpha * permDenPartial * sigma3[i]
            rCoeffs[i] = frSub(rCoeffs[i], frMul(alpha, frMul(permDenPartial, sigma3Coeffs[i])))
            // + alpha^2 * L_1(zeta) * z[i]
            rCoeffs[i] = frAdd(rCoeffs[i], frMul(alpha2, frMul(l1Zeta, zCoeffs[i])))
        }

        // Subtract Z_H(zeta) * (t_lo + zeta^n * t_mid + zeta^{2n} * t_hi)
        let zetaN2 = frSqr(zetaN)
        for i in 0..<n {
            var tVal = tLoCoeffs[i]
            tVal = frAdd(tVal, frMul(zetaN, tMidCoeffs[i]))
            tVal = frAdd(tVal, frMul(zetaN2, tHiCoeffs[i]))
            rCoeffs[i] = frSub(rCoeffs[i], frMul(zhZeta, tVal))
        }

        // Opening proof at zeta: batch open r, a, b, c, sigma1, sigma2
        // W_zeta = (r(x) + v*a(x) + v^2*b(x) + v^3*c(x) + v^4*sigma1(x) + v^5*sigma2(x) - [r(z) + v*a(z) + ...]) / (x - zeta)
        let rZeta = polyEval(rCoeffs, at: zeta)

        var combinedCoeffs = [Fr](repeating: Fr.zero, count: n)
        let polysToOpen = [rCoeffs, aCoeffs, bCoeffs, cCoeffs, setup.permutationPolys[0], setup.permutationPolys[1]]
        let evalsAtZeta = [rZeta, aZeta, bZeta, cZeta, sigma1Zeta, sigma2Zeta]

        var vPow = Fr.one
        var combinedEval = Fr.zero
        for idx in 0..<polysToOpen.count {
            let poly = polysToOpen[idx]
            for i in 0..<min(poly.count, n) {
                combinedCoeffs[i] = frAdd(combinedCoeffs[i], frMul(vPow, poly[i]))
            }
            combinedEval = frAdd(combinedEval, frMul(vPow, evalsAtZeta[idx]))
            vPow = frMul(vPow, v)
        }

        // Subtract combined eval from constant term
        combinedCoeffs[0] = frSub(combinedCoeffs[0], combinedEval)

        // Divide by (x - zeta) via synthetic division
        let wZetaCoeffs = syntheticDivide(combinedCoeffs, root: zeta)
        let openingProof = try kzg.commit(wZetaCoeffs)

        // Shifted opening proof at zeta*omega: just z(x)
        var zShifted = zCoeffs
        zShifted[0] = frSub(zShifted[0], zOmegaZeta)
        let wZetaOmegaCoeffs = syntheticDivide(zShifted, root: zetaOmega)
        let shiftedOpeningProof = try kzg.commit(wZetaOmegaCoeffs)

        return PlonkProof(
            aCommit: aCommit, bCommit: bCommit, cCommit: cCommit,
            zCommit: zCommit,
            tLoCommit: tLoCommit, tMidCommit: tMidCommit, tHiCommit: tHiCommit,
            aEval: aZeta, bEval: bZeta, cEval: cZeta,
            sigma1Eval: sigma1Zeta, sigma2Eval: sigma2Zeta,
            zOmegaEval: zOmegaZeta,
            openingProof: openingProof,
            shiftedOpeningProof: shiftedOpeningProof
        )
    }
}

// MARK: - Polynomial helpers

/// Evaluate polynomial (coefficient form) at a point using Horner's method
public func polyEval(_ coeffs: [Fr], at x: Fr) -> Fr {
    var result = Fr.zero
    for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}

/// Divide polynomial by Z_H(x) = x^n - 1.
/// If f(x) = q(x) * (x^n - 1), returns q(x).
/// Performs long division: process from high degree down.
func polyDivideByVanishing(_ coeffs: [Fr], n: Int) -> [Fr] {
    // f = sum c_i x^i, divide by x^n - 1
    // Quotient degree = deg(f) - n
    let fDeg = coeffs.count
    if fDeg <= n { return [] }

    var rem = coeffs
    let qDeg = fDeg - n
    var q = [Fr](repeating: Fr.zero, count: qDeg)

    // Long division by x^n - 1: for each degree from top down,
    // q[i] = rem[i+n], then rem[i] += q[i] (since divisor has -1 at x^0)
    for i in stride(from: qDeg - 1, through: 0, by: -1) {
        q[i] = rem[i + n]
        rem[i] = frAdd(rem[i], q[i])
    }

    return q
}

/// Synthetic division: divide polynomial by (x - root), return quotient.
/// Uses Horner-like recurrence: q[n-2] = c[n-1], q[i-1] = c[i] + root * q[i]
func syntheticDivide(_ coeffs: [Fr], root: Fr) -> [Fr] {
    let n = coeffs.count
    if n <= 1 { return [] }
    var q = [Fr](repeating: Fr.zero, count: n - 1)
    q[n - 2] = coeffs[n - 1]
    for i in stride(from: n - 2, through: 1, by: -1) {
        q[i - 1] = frAdd(coeffs[i], frMul(root, q[i]))
    }
    return q
}

/// Absorb a projective point into transcript (convert to affine, absorb x and y coordinates)
func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
    let aff = batchToAffine([p])
    // Absorb x coordinate limbs as field elements
    let xInt = fpToInt(aff[0].x)
    let yInt = fpToInt(aff[0].y)
    let xFr = Fr.from64(xInt)
    let yFr = Fr.from64(yInt)
    transcript.absorb(xFr)
    transcript.absorb(yFr)
}
