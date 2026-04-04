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
        // The numerator has degree > n (from polynomial products), so we must compute it
        // in coefficient form using polynomial multiplication, NOT via evaluation on the
        // size-n domain (which aliases to zero since all constraints are satisfied on-domain).

        let qLCoeffsR3 = setup.selectorPolys[0]
        let qRCoeffsR3 = setup.selectorPolys[1]
        let qOCoeffsR3 = setup.selectorPolys[2]
        let qMCoeffsR3 = setup.selectorPolys[3]
        let qCCoeffsR3 = setup.selectorPolys[4]
        let sigma1CoeffsR3 = setup.permutationPolys[0]
        let sigma2CoeffsR3 = setup.permutationPolys[1]
        let sigma3CoeffsR3 = setup.permutationPolys[2]

        // Gate constraint: qL*a + qR*b + qO*c + qM*a*b + qC
        var gateCoeffs = polyMulCoeffs(qLCoeffsR3, aCoeffs)
        gateCoeffs = polyAddCoeffs(gateCoeffs, polyMulCoeffs(qRCoeffsR3, bCoeffs))
        gateCoeffs = polyAddCoeffs(gateCoeffs, polyMulCoeffs(qOCoeffsR3, cCoeffs))
        gateCoeffs = polyAddCoeffs(gateCoeffs, polyMulCoeffs(qMCoeffsR3, polyMulCoeffs(aCoeffs, bCoeffs)))
        gateCoeffs = polyAddCoeffs(gateCoeffs, qCCoeffsR3)

        // Build id_k(x) polynomials: id1(x) = x (identity), id2(x) = k1*x, id3(x) = k2*x
        // On domain omega^i: id1(omega^i) = omega^i = domain[i]
        // id1(x) has coefficients [0, 1, 0, ...], id2(x) = [0, k1, 0, ...], id3(x) = [0, k2, 0, ...]
        var id1Coeffs = [Fr](repeating: Fr.zero, count: n)
        id1Coeffs[1] = Fr.one
        var id2Coeffs = [Fr](repeating: Fr.zero, count: n)
        id2Coeffs[1] = k1
        var id3Coeffs = [Fr](repeating: Fr.zero, count: n)
        id3Coeffs[1] = k2

        // betaConst(x) = beta (constant poly), gammaConst(x) = gamma (constant poly)
        let betaConst = [beta]
        let gammaConst = [gamma]

        // Permutation numerator: (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * z
        let permN1 = polyAddCoeffs(polyAddCoeffs(aCoeffs, polyScaleCoeffs(id1Coeffs, beta)), gammaConst)
        let permN2 = polyAddCoeffs(polyAddCoeffs(bCoeffs, polyScaleCoeffs(id2Coeffs, beta)), gammaConst)
        let permN3 = polyAddCoeffs(polyAddCoeffs(cCoeffs, polyScaleCoeffs(id3Coeffs, beta)), gammaConst)
        let permNumPoly = polyMulCoeffs(polyMulCoeffs(polyMulCoeffs(permN1, permN2), permN3), zCoeffs)

        // Permutation denominator: (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * z(omega*x)
        let permD1 = polyAddCoeffs(polyAddCoeffs(aCoeffs, polyScaleCoeffs(sigma1CoeffsR3, beta)), gammaConst)
        let permD2 = polyAddCoeffs(polyAddCoeffs(bCoeffs, polyScaleCoeffs(sigma2CoeffsR3, beta)), gammaConst)
        let permD3 = polyAddCoeffs(polyAddCoeffs(cCoeffs, polyScaleCoeffs(sigma3CoeffsR3, beta)), gammaConst)
        // z(omega*x) has coefficients z[i] * omega^i
        let zOmegaCoeffs = polyShift(zCoeffs, omega: omega)
        let permDenPoly = polyMulCoeffs(polyMulCoeffs(polyMulCoeffs(permD1, permD2), permD3), zOmegaCoeffs)

        let permCoeffs = polySubCoeffs(permNumPoly, permDenPoly)

        // Boundary constraint: (z(x) - 1) * L_1(x)
        // L_1(x) = (x^n - 1) / (n * (x - 1))
        // In coefficient form: L_1 has coefficients from iNTT of [1, 0, 0, ..., 0]
        var l1Evals = [Fr](repeating: Fr.zero, count: n)
        l1Evals[0] = Fr.one
        let l1Coeffs = try ntt.intt(l1Evals)
        var zMinus1Coeffs = zCoeffs
        zMinus1Coeffs[0] = frSub(zMinus1Coeffs[0], Fr.one)
        let boundaryCoeffs = polyMulCoeffs(zMinus1Coeffs, l1Coeffs)

        // Combine: numerator = gate + alpha * perm + alpha^2 * boundary
        let alpha2 = frSqr(alpha)
        var numCoeffs = gateCoeffs
        numCoeffs = polyAddCoeffs(numCoeffs, polyScaleCoeffs(permCoeffs, alpha))
        numCoeffs = polyAddCoeffs(numCoeffs, polyScaleCoeffs(boundaryCoeffs, alpha2))

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

/// Multiply two polynomials (coefficient form) naively in O(n*m).
func polyMulCoeffs(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    if a.isEmpty || b.isEmpty { return [] }
    let n = a.count + b.count - 1
    var result = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<a.count {
        for j in 0..<b.count {
            result[i + j] = frAdd(result[i + j], frMul(a[i], b[j]))
        }
    }
    return result
}

/// Add two polynomials (coefficient form).
func polyAddCoeffs(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    let n = max(a.count, b.count)
    var result = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<a.count { result[i] = a[i] }
    for i in 0..<b.count { result[i] = frAdd(result[i], b[i]) }
    return result
}

/// Subtract two polynomials (coefficient form): a - b.
func polySubCoeffs(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    let n = max(a.count, b.count)
    var result = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<a.count { result[i] = a[i] }
    for i in 0..<b.count { result[i] = frSub(result[i], b[i]) }
    return result
}

/// Scale polynomial by a scalar.
func polyScaleCoeffs(_ a: [Fr], _ s: Fr) -> [Fr] {
    return a.map { frMul($0, s) }
}

/// Compute f(omega * x) from f(x): coefficient i becomes c_i * omega^i.
func polyShift(_ coeffs: [Fr], omega: Fr) -> [Fr] {
    var result = [Fr](repeating: Fr.zero, count: coeffs.count)
    var omegaPow = Fr.one
    for i in 0..<coeffs.count {
        result[i] = frMul(coeffs[i], omegaPow)
        omegaPow = frMul(omegaPow, omega)
    }
    return result
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
