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
// Optimized: CPU C NTT for small sizes, batch field ops, C synthetic division.

import Foundation
import NeonFieldOps

// MARK: - CPU NTT threshold

/// Below this size, use CPU C NTT instead of GPU dispatch (avoids Metal overhead).
private let kCPU_NTT_THRESHOLD = 8192

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
        let logN = Int(log2(Double(n)))
        let aCoeffs: [Fr], bCoeffs: [Fr], cCoeffs: [Fr]
        if n <= kCPU_NTT_THRESHOLD {
            aCoeffs = cINTT_Fr(aEvals, logN: logN)
            bCoeffs = cINTT_Fr(bEvals, logN: logN)
            cCoeffs = cINTT_Fr(cEvals, logN: logN)
        } else {
            aCoeffs = try ntt.intt(aEvals)
            bCoeffs = try ntt.intt(bEvals)
            cCoeffs = try ntt.intt(cEvals)
        }

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

        let domain = setup.domain
        let k1 = setup.k1
        let k2 = setup.k2
        let sigma1Evals = setup.permutationEvals[0]
        let sigma2Evals = setup.permutationEvals[1]
        let sigma3Evals = setup.permutationEvals[2]

        // Fused C path: numerator/denominator loop + batch inverse + running product
        aEvals.withUnsafeBytes { aBuf in
            bEvals.withUnsafeBytes { bBuf in
                cEvals.withUnsafeBytes { cBuf in
                    sigma1Evals.withUnsafeBytes { s1Buf in
                        sigma2Evals.withUnsafeBytes { s2Buf in
                            sigma3Evals.withUnsafeBytes { s3Buf in
                                domain.withUnsafeBytes { dBuf in
                                    withUnsafeBytes(of: beta) { betaBuf in
                                        withUnsafeBytes(of: gamma) { gammaBuf in
                                            withUnsafeBytes(of: k1) { k1Buf in
                                                withUnsafeBytes(of: k2) { k2Buf in
                                                    zEvals.withUnsafeMutableBytes { zBuf in
                                                        plonk_compute_z_accumulator(
                                                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            s3Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            betaBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            gammaBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            k1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            k2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                                            Int32(n),
                                                            zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let zCoeffs = n <= kCPU_NTT_THRESHOLD ? cINTT_Fr(zEvals, logN: logN) : try ntt.intt(zEvals)
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
        let qRangeCoeffsR3 = setup.selectorPolys[5]
        let qLookupCoeffsR3 = setup.selectorPolys[6]
        let qPoseidonCoeffsR3 = setup.selectorPolys[7]

        // Gate constraint: qL*a + qR*b + qO*c + qM*a*b + qC
        var gateCoeffs = try polyMulNTT(qLCoeffsR3, aCoeffs, ntt: ntt)
        gateCoeffs = polyAddCoeffs(gateCoeffs, try polyMulNTT(qRCoeffsR3, bCoeffs, ntt: ntt))
        gateCoeffs = polyAddCoeffs(gateCoeffs, try polyMulNTT(qOCoeffsR3, cCoeffs, ntt: ntt))
        // Reuse a*b product for qM and Poseidon
        let abProd = try polyMulNTT(aCoeffs, bCoeffs, ntt: ntt)
        gateCoeffs = polyAddCoeffs(gateCoeffs, try polyMulNTT(qMCoeffsR3, abProd, ntt: ntt))
        gateCoeffs = polyAddCoeffs(gateCoeffs, qCCoeffsR3)

        // Custom gate: Range constraint: qRange * (a - a^2)
        let aSqProd = try polyMulNTT(aCoeffs, aCoeffs, ntt: ntt)
        let aMinusASq = polySubCoeffs(aCoeffs, aSqProd)
        gateCoeffs = polyAddCoeffs(gateCoeffs, try polyMulNTT(qRangeCoeffsR3, aMinusASq, ntt: ntt))

        // Custom gate: Lookup constraint
        if !setup.lookupTables.isEmpty {
            for table in setup.lookupTables {
                if table.values.isEmpty { continue }
                var vanishPoly = polySubCoeffs(aCoeffs, [table.values[0]])
                for k in 1..<table.values.count {
                    let factor = polySubCoeffs(aCoeffs, [table.values[k]])
                    vanishPoly = try polyMulNTT(vanishPoly, factor, ntt: ntt)
                }
                let lookupConstraint = try polyMulNTT(qLookupCoeffsR3, vanishPoly, ntt: ntt)
                gateCoeffs = polyAddCoeffs(gateCoeffs, lookupConstraint)
            }
        }

        // Custom gate: Poseidon S-box constraint: qPoseidon * (c - a*b*b)
        // Reuse abProd from gate constraint
        let abbCoeffs = try polyMulNTT(abProd, bCoeffs, ntt: ntt)
        let sboxDiff = polySubCoeffs(cCoeffs, abbCoeffs)
        let poseidonConstraint = try polyMulNTT(qPoseidonCoeffsR3, sboxDiff, ntt: ntt)
        gateCoeffs = polyAddCoeffs(gateCoeffs, poseidonConstraint)

        // Build id_k(x) polynomials: id1(x)=x, id2(x)=k1*x, id3(x)=k2*x
        var id1Coeffs = [Fr](repeating: Fr.zero, count: n); id1Coeffs[1] = Fr.one
        var id2Coeffs = [Fr](repeating: Fr.zero, count: n); id2Coeffs[1] = k1
        var id3Coeffs = [Fr](repeating: Fr.zero, count: n); id3Coeffs[1] = k2

        let gammaConst = [gamma]

        // Permutation numerator: (a + beta*id1 + gamma)(b + beta*id2 + gamma)(c + beta*id3 + gamma) * z
        let permN1 = polyAddCoeffs(polyAddCoeffs(aCoeffs, polyScaleCoeffs(id1Coeffs, beta)), gammaConst)
        let permN2 = polyAddCoeffs(polyAddCoeffs(bCoeffs, polyScaleCoeffs(id2Coeffs, beta)), gammaConst)
        let permN3 = polyAddCoeffs(polyAddCoeffs(cCoeffs, polyScaleCoeffs(id3Coeffs, beta)), gammaConst)
        // For triple+ products, use the optimized polyMulNTT which now uses CPU C NTT
        let permNumPoly = try polyMulNTT(try polyMulNTT(try polyMulNTT(permN1, permN2, ntt: ntt), permN3, ntt: ntt), zCoeffs, ntt: ntt)

        // Permutation denominator: (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma) * z(omega*x)
        let sigma1CoeffsR3 = setup.permutationPolys[0]
        let sigma2CoeffsR3 = setup.permutationPolys[1]
        let sigma3CoeffsR3 = setup.permutationPolys[2]
        let permD1 = polyAddCoeffs(polyAddCoeffs(aCoeffs, polyScaleCoeffs(sigma1CoeffsR3, beta)), gammaConst)
        let permD2 = polyAddCoeffs(polyAddCoeffs(bCoeffs, polyScaleCoeffs(sigma2CoeffsR3, beta)), gammaConst)
        let permD3 = polyAddCoeffs(polyAddCoeffs(cCoeffs, polyScaleCoeffs(sigma3CoeffsR3, beta)), gammaConst)
        let zOmegaCoeffs = polyShift(zCoeffs, omega: omega)
        let permDenPoly = try polyMulNTT(try polyMulNTT(try polyMulNTT(permD1, permD2, ntt: ntt), permD3, ntt: ntt), zOmegaCoeffs, ntt: ntt)

        let permCoeffs = polySubCoeffs(permNumPoly, permDenPoly)

        // Boundary constraint: (z(x) - 1) * L_1(x)
        var l1EvalsRaw = [Fr](repeating: Fr.zero, count: n)
        l1EvalsRaw[0] = Fr.one
        let l1Coeffs = n <= kCPU_NTT_THRESHOLD ? cINTT_Fr(l1EvalsRaw, logN: logN) : try ntt.intt(l1EvalsRaw)
        var zMinus1Coeffs = zCoeffs
        zMinus1Coeffs[0] = frSub(zMinus1Coeffs[0], Fr.one)
        let boundaryCoeffs = try polyMulNTT(zMinus1Coeffs, l1Coeffs, ntt: ntt)

        // Combine: numerator = gate + alpha * perm + alpha^2 * boundary
        let alpha2 = frSqr(alpha)
        var numCoeffs = gateCoeffs
        numCoeffs = polyAddCoeffs(numCoeffs, polyScaleCoeffs(permCoeffs, alpha))
        numCoeffs = polyAddCoeffs(numCoeffs, polyScaleCoeffs(boundaryCoeffs, alpha2))

        // Divide by Z_H(x) = x^n - 1 in coefficient form
        // Since the numerator vanishes on the domain, it is divisible by Z_H
        let tCoeffs = polyDivideByVanishing(numCoeffs, n: n)

        // Split t into chunks of degree n: t = t_0 + x^n * t_1 + x^{2n} * t_2 + ...
        let numChunks = max(3, (tCoeffs.count + n - 1) / n)
        var tChunkCoeffs = [[Fr]]()
        for c in 0..<numChunks {
            let start = c * n
            if start < tCoeffs.count {
                let chunk = Array(tCoeffs.dropFirst(start).prefix(n))
                tChunkCoeffs.append(chunk + [Fr](repeating: Fr.zero, count: max(0, n - chunk.count)))
            } else {
                tChunkCoeffs.append([Fr](repeating: Fr.zero, count: n))
            }
        }

        let tLoCoeffs = tChunkCoeffs[0]
        let tMidCoeffs = tChunkCoeffs[1]
        let tHiCoeffs = tChunkCoeffs[2]

        var tChunkCommits = [PointProjective]()
        for chunk in tChunkCoeffs {
            tChunkCommits.append(try kzg.commit(chunk))
        }

        let tLoCommit = tChunkCommits[0]
        let tMidCommit = tChunkCommits[1]
        let tHiCommit = tChunkCommits[2]

        for commit in tChunkCommits {
            absorbPoint(transcript, commit)
        }

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
        let qRangeCoeffs = setup.selectorPolys[5]
        let qLookupCoeffs = setup.selectorPolys[6]
        let qPoseidonCoeffs = setup.selectorPolys[7]
        let sigma3Coeffs = setup.permutationPolys[2]

        // Compute L_1(zeta) = (zeta^n - 1) / (n * (zeta - 1))
        let zetaN = frPow(zeta, UInt64(n))
        let zhZeta = frSub(zetaN, Fr.one)  // Z_H(zeta) = zeta^n - 1
        let nInv = frInverse(frFromInt(UInt64(n)))
        let l1Zeta = frMul(zhZeta, frMul(nInv, frInverse(frSub(zeta, Fr.one))))

        // Build r(x) coefficient by coefficient
        var rCoeffs = [Fr](repeating: Fr.zero, count: n)

        // Gate part: a_z*b_z*qM + a_z*qL + b_z*qR + c_z*qO + qC
        //   + qRange*(a_z - a_z^2)
        //   + qLookup*vanish(a_z)
        //   + qPoseidon*(c_z - a_z*b_z*b_z)
        let abZeta = frMul(aZeta, bZeta)

        // Range: a*(1-a) = a - a^2
        let aZetaSq = frSqr(aZeta)
        let rangeScalar = frSub(aZeta, aZetaSq)

        // Lookup: prod(a_z - t_i) for all table values
        var lookupScalar = Fr.zero
        for table in setup.lookupTables {
            if table.values.isEmpty { continue }
            var prod = Fr.one
            for tVal in table.values {
                prod = frMul(prod, frSub(aZeta, tVal))
            }
            lookupScalar = frAdd(lookupScalar, prod)
        }

        // Poseidon: c - a*b*b = c - a*b^2
        let bZetaSq = frSqr(bZeta)
        let poseidonScalar = frSub(cZeta, frMul(aZeta, bZetaSq))

        // Gate part via batch ops: r = abZeta*qM + aZeta*qL + bZeta*qR + cZeta*qO + qC
        //                         + rangeScalar*qRange + lookupScalar*qLookup + poseidonScalar*qPoseidon
        rCoeffs = polyScaleCoeffs(qMCoeffs, abZeta)
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qLCoeffs, aZeta))
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qRCoeffs, bZeta))
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qOCoeffs, cZeta))
        rCoeffs = polyAddCoeffs(rCoeffs, qCCoeffs)
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qRangeCoeffs, rangeScalar))
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qLookupCoeffs, lookupScalar))
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(qPoseidonCoeffs, poseidonScalar))

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

        // r += alpha*permNum*z - alpha*permDenPartial*sigma3 + alpha^2*L1(zeta)*z
        let alphaPermNum = frMul(alpha, permNum)
        let alphaPermDen = frMul(alpha, permDenPartial)
        let alpha2L1 = frMul(alpha2, l1Zeta)
        // Combined z coefficient: (alpha*permNum + alpha^2*L1(zeta))
        let zScale = frAdd(alphaPermNum, alpha2L1)
        rCoeffs = polyAddCoeffs(rCoeffs, polyScaleCoeffs(zCoeffs, zScale))
        rCoeffs = polySubCoeffs(rCoeffs, polyScaleCoeffs(sigma3Coeffs, alphaPermDen))

        // Subtract Z_H(zeta) * sum_k(zeta^{k*n} * t_k)
        // Build combined t polynomial: sum_k(zeta^{k*n} * t_k)
        var combinedT = tChunkCoeffs[0]
        var zetaNPow = zetaN
        for c in 1..<numChunks {
            combinedT = polyAddCoeffs(combinedT, polyScaleCoeffs(tChunkCoeffs[c], zetaNPow))
            zetaNPow = frMul(zetaNPow, zetaN)
        }
        rCoeffs = polySubCoeffs(rCoeffs, polyScaleCoeffs(combinedT, zhZeta))

        // Opening proof at zeta: batch open r, a, b, c, sigma1, sigma2
        // W_zeta = (r(x) + v*a(x) + v^2*b(x) + v^3*c(x) + v^4*sigma1(x) + v^5*sigma2(x) - [r(z) + v*a(z) + ...]) / (x - zeta)
        let rZeta = polyEval(rCoeffs, at: zeta)

        // Evaluate custom selector polys at zeta for the opening proof
        let qRangeZeta = polyEval(qRangeCoeffs, at: zeta)
        let qLookupZeta = polyEval(qLookupCoeffs, at: zeta)
        let qPoseidonZeta = polyEval(qPoseidonCoeffs, at: zeta)

        // Batch combined polynomial: sum_k v^k * poly_k using C batch ops
        let polysToOpen = [rCoeffs, aCoeffs, bCoeffs, cCoeffs, setup.permutationPolys[0], setup.permutationPolys[1]]
        let evalsAtZeta = [rZeta, aZeta, bZeta, cZeta, sigma1Zeta, sigma2Zeta]

        var combinedCoeffs = rCoeffs  // v^0 * r (v^0 = 1)
        var vPow = v
        var combinedEval = rZeta
        for idx in 1..<polysToOpen.count {
            combinedCoeffs = polyAddCoeffs(combinedCoeffs, polyScaleCoeffs(polysToOpen[idx], vPow))
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

        let tExtraCommits = numChunks > 3 ? Array(tChunkCommits.dropFirst(3)) : []

        // Collect public input values from witness
        var pubInputs = [Fr]()
        for idx in circuit.publicInputIndices {
            pubInputs.append(witness[idx])
        }

        return PlonkProof(
            aCommit: aCommit, bCommit: bCommit, cCommit: cCommit,
            zCommit: zCommit,
            tLoCommit: tLoCommit, tMidCommit: tMidCommit, tHiCommit: tHiCommit,
            tExtraCommits: tExtraCommits,
            aEval: aZeta, bEval: bZeta, cEval: cZeta,
            sigma1Eval: sigma1Zeta, sigma2Eval: sigma2Zeta,
            zOmegaEval: zOmegaZeta,
            openingProof: openingProof,
            shiftedOpeningProof: shiftedOpeningProof,
            publicInputs: pubInputs
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
/// Uses C CIOS for fast Montgomery field ops.
func syntheticDivide(_ coeffs: [Fr], root: Fr) -> [Fr] {
    let n = coeffs.count
    if n <= 1 { return [] }
    var quotient = [Fr](repeating: Fr.zero, count: n - 1)
    coeffs.withUnsafeBytes { cBuf in
        withUnsafeBytes(of: root) { rBuf in
            quotient.withUnsafeMutableBytes { qBuf in
                bn254_fr_synthetic_div(
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return quotient
}

/// Multiply two polynomials (coefficient form) naively in O(n*m).
/// Used only for small polynomials; large ones should use polyMulNTT.
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

/// Check if a polynomial is all zeros.
func polyIsZero(_ a: [Fr]) -> Bool {
    for c in a {
        if !c.isZero { return false }
    }
    return true
}

/// NTT-based polynomial multiplication in O(n log n).
/// Uses CPU C NTT for small sizes to avoid GPU dispatch overhead.
func polyMulNTT(_ a: [Fr], _ b: [Fr], ntt: NTTEngine) throws -> [Fr] {
    if a.isEmpty || b.isEmpty { return [] }
    if polyIsZero(a) || polyIsZero(b) { return [] }
    let resultLen = a.count + b.count - 1
    var logM = 0
    while (1 << logM) < resultLen { logM += 1 }
    let m = 1 << logM

    var aPad = [Fr](repeating: Fr.zero, count: m)
    for i in 0..<a.count { aPad[i] = a[i] }
    var bPad = [Fr](repeating: Fr.zero, count: m)
    for i in 0..<b.count { bPad[i] = b[i] }

    if m <= kCPU_NTT_THRESHOLD {
        // CPU path: C CIOS NTT + batch pairwise multiply
        var aData = aPad
        var bData = bPad
        aData.withUnsafeMutableBytes { buf in
            bn254_fr_ntt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logM))
        }
        bData.withUnsafeMutableBytes { buf in
            bn254_fr_ntt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logM))
        }
        // Pointwise multiply using C batch pairwise mul
        var cData = [Fr](repeating: Fr.zero, count: m)
        cData.withUnsafeMutableBytes { cBuf in
            aData.withUnsafeBytes { aBuf in
                bData.withUnsafeBytes { bBuf in
                    mont_mul_pair_batch_asm(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(m)
                    )
                }
            }
        }
        cData.withUnsafeMutableBytes { buf in
            bn254_fr_intt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logM))
        }
        return Array(cData.prefix(resultLen))
    }

    // GPU path for large sizes
    let aEvals = try ntt.ntt(aPad)
    let bEvals = try ntt.ntt(bPad)

    var cEvals = [Fr](repeating: Fr.zero, count: m)
    for i in 0..<m { cEvals[i] = frMul(aEvals[i], bEvals[i]) }

    let result = try ntt.intt(cEvals)
    return Array(result.prefix(resultLen))
}

/// Add two polynomials (coefficient form). Uses C batch add for equal-length hot path.
func polyAddCoeffs(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    let n = max(a.count, b.count)
    let minLen = min(a.count, b.count)
    var result = [Fr](repeating: Fr.zero, count: n)

    if minLen > 0 {
        result.withUnsafeMutableBytes { rBuf in
            a.withUnsafeBytes { aBuf in
                b.withUnsafeBytes { bBuf in
                    bn254_fr_batch_add_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(minLen)
                    )
                }
            }
        }
    }
    // Copy remaining elements from whichever is longer
    if a.count > minLen {
        for i in minLen..<a.count { result[i] = a[i] }
    } else if b.count > minLen {
        for i in minLen..<b.count { result[i] = b[i] }
    }
    return result
}

/// Subtract two polynomials (coefficient form): a - b. Uses C batch sub.
func polySubCoeffs(_ a: [Fr], _ b: [Fr]) -> [Fr] {
    let n = max(a.count, b.count)
    let minLen = min(a.count, b.count)
    var result = [Fr](repeating: Fr.zero, count: n)

    if minLen > 0 {
        result.withUnsafeMutableBytes { rBuf in
            a.withUnsafeBytes { aBuf in
                b.withUnsafeBytes { bBuf in
                    bn254_fr_batch_sub_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(minLen)
                    )
                }
            }
        }
    }
    if a.count > minLen {
        for i in minLen..<a.count { result[i] = a[i] }
    } else if b.count > minLen {
        // result[i] = 0 - b[i] = neg(b[i])
        for i in minLen..<b.count { result[i] = frSub(Fr.zero, b[i]) }
    }
    return result
}

/// Scale polynomial by a scalar. Uses C batch scalar mul.
func polyScaleCoeffs(_ a: [Fr], _ s: Fr) -> [Fr] {
    if a.isEmpty { return [] }
    var result = [Fr](repeating: Fr.zero, count: a.count)
    result.withUnsafeMutableBytes { rBuf in
        a.withUnsafeBytes { aBuf in
            withUnsafeBytes(of: s) { sBuf in
                bn254_fr_batch_mul_scalar_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(a.count)
                )
            }
        }
    }
    return result
}

/// Compute f(omega * x) from f(x): coefficient i becomes c_i * omega^i.
/// Uses C pairwise batch multiply for the omega power scaling.
func polyShift(_ coeffs: [Fr], omega: Fr) -> [Fr] {
    let n = coeffs.count
    if n == 0 { return [] }
    // Build omega powers: [1, omega, omega^2, ..., omega^{n-1}]
    var powers = [Fr](repeating: Fr.one, count: n)
    for i in 1..<n { powers[i] = frMul(powers[i-1], omega) }
    // Pairwise multiply
    var result = [Fr](repeating: Fr.zero, count: n)
    result.withUnsafeMutableBytes { rBuf in
        coeffs.withUnsafeBytes { cBuf in
            powers.withUnsafeBytes { pBuf in
                mont_mul_pair_batch_asm(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }
        }
    }
    return result
}

/// Absorb a projective point into transcript (convert to affine, absorb x and y coordinates).
/// Uses C CIOS single-point conversion to avoid batchToAffine overhead.
func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
    var affine = [UInt64](repeating: 0, count: 8)  // x[4], y[4]
    withUnsafeBytes(of: p) { pBuf in
        affine.withUnsafeMutableBufferPointer { aBuf in
            bn254_projective_to_affine(
                pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                aBuf.baseAddress!
            )
        }
    }
    let xFr = Fr.from64(Array(affine[0..<4]))
    let yFr = Fr.from64(Array(affine[4..<8]))
    transcript.absorb(xFr)
    transcript.absorb(yFr)
}
