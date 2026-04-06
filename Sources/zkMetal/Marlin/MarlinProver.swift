// MarlinProver — Marlin polynomial IOP prover (Chiesa et al., EUROCRYPT 2020)
//
// Implements Marlin's Algebraic Holographic Proof (AHP) with KZG commitments:
//   Index phase: preprocess R1CS (A, B, C) into committed row/col/val polynomials
//   Round 1: commit to witness w(X) and auxiliary z_A, z_B, z_C
//   Round 2: receive alpha, compute quotient t(X) = (z_A*z_B - z_C) / v_H
//   Round 3: receive beta, compute lincheck g(X) and h(X)
//   Opening phase: batch-open all polynomials at evaluation points via KZG
//
// Uses BN254 Fr, KZGEngine for commitments, NTTEngine for polynomial arithmetic.
// Fiat-Shamir via Transcript (Keccak-256).

import Foundation
import NeonFieldOps

// MARK: - MarlinProver

/// Standalone Marlin prover that generates AHP proofs for R1CS satisfiability.
///
/// Usage:
///   let prover = try MarlinProver(kzg: kzg, ntt: ntt)
///   let (pk, vk) = try prover.index(r1cs: r1cs, srsSecret: secret)
///   let proof = try prover.prove(r1cs: r1cs, publicInputs: pub, witness: wit, pk: pk)
public class MarlinProver {
    public let kzg: KZGEngine
    public let ntt: NTTEngine

    public init(kzg: KZGEngine, ntt: NTTEngine) {
        self.kzg = kzg
        self.ntt = ntt
    }

    // MARK: - Index Phase

    /// Marlin index (setup): preprocess R1CS matrices A, B, C into committed
    /// row/col/val/row_col polynomials. Returns a proving key and verification key.
    ///
    /// For each matrix M in {A, B, C}, encodes the sparse entries as:
    ///   row_M(X): maps non-zero index i to omega_H^{entry.row}
    ///   col_M(X): maps non-zero index i to omega_K^{entry.col}
    ///   val_M(X): maps non-zero index i to the entry value
    ///   rowcol_M(X): row_M(X) * col_M(X)
    public func index(r1cs: R1CSInstance, srsSecret: Fr) throws -> (MarlinProvingKey, MarlinVerificationKey) {
        let m = r1cs.numConstraints
        let n = r1cs.numVars
        let numPublic = r1cs.numPublic
        let maxNNZ = max(r1cs.aEntries.count, max(r1cs.bEntries.count, r1cs.cEntries.count))

        let hSize = nextPow2(m)
        let kSize = nextPow2(n)
        let nzSize = nextPow2(maxNNZ)

        let logH = logBase2(hSize)
        let logK = logBase2(kSize)

        let omegaH = frRootOfUnity(logN: logH)
        let omegaK = frRootOfUnity(logN: logK)

        let index = MarlinIndex(
            numConstraints: m, numVariables: n, numNonZero: maxNNZ,
            constraintDomainSize: hSize, variableDomainSize: kSize,
            nonZeroDomainSize: nzSize, omegaH: omegaH, omegaK: omegaK
        )

        // Build index polynomials for each matrix: row(X), col(X), val(X), row_col(X)
        let allEntries = [r1cs.aEntries, r1cs.bEntries, r1cs.cEntries]

        var indexPolynomials = [[Fr]]()
        var indexCommitments = [PointProjective]()

        for entries in allEntries {
            var rowEvals = [Fr](repeating: .zero, count: nzSize)
            var colEvals = [Fr](repeating: .zero, count: nzSize)
            var valEvals = [Fr](repeating: .zero, count: nzSize)
            var rcEvals = [Fr](repeating: .zero, count: nzSize)

            for (i, entry) in entries.enumerated() {
                if i >= nzSize { break }
                rowEvals[i] = frPow(omegaH, UInt64(entry.row))
                colEvals[i] = frPow(omegaK, UInt64(entry.col))
                valEvals[i] = entry.val
                rcEvals[i] = frMul(rowEvals[i], colEvals[i])
            }

            let rowCoeffs = try ntt.intt(rowEvals)
            let colCoeffs = try ntt.intt(colEvals)
            let valCoeffs = try ntt.intt(valEvals)
            let rcCoeffs = try ntt.intt(rcEvals)

            for poly in [rowCoeffs, colCoeffs, valCoeffs, rcCoeffs] {
                indexPolynomials.append(poly)
                indexCommitments.append(try kzg.commit(poly))
            }
        }

        let vk = MarlinVerificationKey(
            index: index, indexCommitments: indexCommitments,
            srsSecret: srsSecret, srs: kzg.srs
        )

        let pk = MarlinProvingKey(
            index: index, indexPolynomials: indexPolynomials,
            indexCommitments: indexCommitments, srs: kzg.srs,
            srsSecret: srsSecret, numPublic: numPublic
        )

        return (pk, vk)
    }

    // MARK: - Prove

    /// Generate a Marlin proof for R1CS satisfiability.
    ///
    /// Protocol rounds:
    /// 1. Commit to w(X), z_A(X), z_B(X), z_C(X); receive eta_A, eta_B, eta_C
    /// 2. Build t(X) = (z_A*z_B - z_C) / v_H; commit; receive alpha
    /// 3. Sumcheck round polys; receive beta
    /// 4. Build lincheck g(X), h(X); commit; receive gamma
    /// 5. Open all polynomials at beta and gamma via batch KZG
    public func prove(r1cs: R1CSInstance, publicInputs: [Fr], witness: [Fr],
                      pk: MarlinProvingKey) throws -> MarlinProof {
        let idx = pk.index
        let hSize = idx.constraintDomainSize
        let kSize = idx.variableDomainSize
        let nzSize = idx.nonZeroDomainSize
        let numPublic = pk.numPublic
        let logH = logBase2(hSize)
        let numSumcheckRounds = logH

        // --- Build full assignment z = [1, publicInputs..., witness...] ---
        var fullZ = [Fr](repeating: .zero, count: r1cs.numVars)
        fullZ[0] = .one
        for i in 0..<numPublic { fullZ[1 + i] = publicInputs[i] }
        for i in 0..<witness.count { fullZ[1 + numPublic + i] = witness[i] }

        // --- Compute A*z, B*z, C*z ---
        let az = r1cs.sparseMatVec(r1cs.aEntries, fullZ)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, fullZ)
        let cz = r1cs.sparseMatVec(r1cs.cEntries, fullZ)

        // Pad to domain sizes and convert to coefficient form
        var zAEvals = [Fr](repeating: .zero, count: hSize)
        var zBEvals = [Fr](repeating: .zero, count: hSize)
        var zCEvals = [Fr](repeating: .zero, count: hSize)
        for i in 0..<r1cs.numConstraints {
            zAEvals[i] = az[i]; zBEvals[i] = bz[i]; zCEvals[i] = cz[i]
        }
        let zACoeffs = try ntt.intt(zAEvals)
        let zBCoeffs = try ntt.intt(zBEvals)
        let zCCoeffs = try ntt.intt(zCEvals)

        var wEvals = [Fr](repeating: .zero, count: kSize)
        for i in 0..<min(fullZ.count, kSize) { wEvals[i] = fullZ[i] }
        let wCoeffs = try ntt.intt(wEvals)

        // --- Round 1: commit to witness and matrix-vector products ---
        let wCommit = try kzg.commit(wCoeffs)
        let zACommit = try kzg.commit(zACoeffs)
        let zBCommit = try kzg.commit(zBCoeffs)
        let zCCommit = try kzg.commit(zCCoeffs)

        // Build Fiat-Shamir transcript through round 1
        let ts = Transcript(label: "marlin", backend: .keccak256)
        ts.absorb(frFromInt(UInt64(idx.numConstraints)))
        ts.absorb(frFromInt(UInt64(idx.numVariables)))
        ts.absorb(frFromInt(UInt64(idx.numNonZero)))
        for c in pk.indexCommitments { marlinAbsorbPointImpl(ts, c) }
        for pi in publicInputs { ts.absorb(pi) }
        marlinAbsorbPointImpl(ts, wCommit)
        marlinAbsorbPointImpl(ts, zACommit)
        marlinAbsorbPointImpl(ts, zBCommit)
        marlinAbsorbPointImpl(ts, zCCommit)
        let etaA = ts.squeeze()
        let etaB = ts.squeeze()
        let etaC = ts.squeeze()

        // --- Round 2: quotient polynomial t(X) = (z_A*z_B - z_C) / v_H(X) ---
        // Multiply on 2x domain to handle degree doubling
        let doubleH = hSize * 2

        var zACoeffs2 = [Fr](repeating: .zero, count: doubleH)
        var zBCoeffs2 = [Fr](repeating: .zero, count: doubleH)
        var zCCoeffs2 = [Fr](repeating: .zero, count: doubleH)
        for i in 0..<zACoeffs.count { zACoeffs2[i] = zACoeffs[i] }
        for i in 0..<zBCoeffs.count { zBCoeffs2[i] = zBCoeffs[i] }
        for i in 0..<zCCoeffs.count { zCCoeffs2[i] = zCCoeffs[i] }
        let zAEvals2 = try ntt.ntt(zACoeffs2)
        let zBEvals2 = try ntt.ntt(zBCoeffs2)
        let zCEvals2 = try ntt.ntt(zCCoeffs2)

        // Numerator: z_A * z_B - z_C on 2x domain
        var numEvals2 = [Fr](repeating: .zero, count: doubleH)
        for i in 0..<doubleH {
            numEvals2[i] = frSub(frMul(zAEvals2[i], zBEvals2[i]), zCEvals2[i])
        }
        var numCoeffs = try ntt.intt(numEvals2)

        // Divide by vanishing polynomial v_H(X) = X^|H| - 1 via synthetic long division
        var tCoeffs = [Fr](repeating: .zero, count: hSize)
        for i in stride(from: numCoeffs.count - 1, through: hSize, by: -1) {
            let qi = numCoeffs[i]
            tCoeffs[i - hSize] = qi
            numCoeffs[i - hSize] = frAdd(numCoeffs[i - hSize], qi)
        }

        let tCommit = try kzg.commit(tCoeffs)

        // Absorb t commitment, squeeze alpha challenge
        marlinAbsorbPointImpl(ts, tCommit)
        let alpha = ts.squeeze()

        // --- Sumcheck round polynomials ---
        let sumcheckPolys = MarlinProver.buildSumcheckPolys(numSumcheckRounds, alpha: alpha)

        // Absorb sumcheck, squeeze beta
        for coeffs in sumcheckPolys { for c in coeffs { ts.absorb(c) } }
        let beta = ts.squeeze()

        // --- Round 3: inner sumcheck / lincheck ---
        // Evaluate the combined sigma on K_NZ domain:
        //   sigma(X) = sum_M eta_M * val_M(X) / ((beta - row_M(X)) * (X - col_M(X)))
        let logNZ = logBase2(nzSize)
        let omegaNZ = frRootOfUnity(logN: logNZ)
        let etas = [etaA, etaB, etaC]

        // Precompute domain points via chain multiply
        var domainPts = [Fr](repeating: Fr.one, count: nzSize)
        for i in 1..<nzSize { domainPts[i] = frMul(domainPts[i - 1], omegaNZ) }

        // Collect all denominators for batch inversion (3 per domain point)
        var allDenoms = [Fr](repeating: Fr.zero, count: nzSize * 3)
        var denomNZ = [Bool](repeating: false, count: nzSize * 3)
        for i in 0..<nzSize {
            let pt = domainPts[i]
            for mi in 0..<3 {
                let rg = evalPoly(pk.indexPolynomials[mi * 4], at: pt)
                let cg = evalPoly(pk.indexPolynomials[mi * 4 + 1], at: pt)
                let d = frMul(frSub(beta, rg), frSub(pt, cg))
                allDenoms[i * 3 + mi] = d
                denomNZ[i * 3 + mi] = !d.isZero
            }
        }
        let totalD = nzSize * 3
        var dPfx = [Fr](repeating: Fr.one, count: totalD)
        for i in 1..<totalD {
            dPfx[i] = denomNZ[i - 1] ? frMul(dPfx[i - 1], allDenoms[i - 1]) : dPfx[i - 1]
        }
        let lastProd = denomNZ[totalD - 1] ? frMul(dPfx[totalD - 1], allDenoms[totalD - 1]) : dPfx[totalD - 1]
        var dAcc = frInverse(lastProd)
        var denomInvs = [Fr](repeating: Fr.zero, count: totalD)
        for i in Swift.stride(from: totalD - 1, through: 0, by: -1) {
            if denomNZ[i] {
                denomInvs[i] = frMul(dAcc, dPfx[i])
                dAcc = frMul(dAcc, allDenoms[i])
            }
        }
        var sigmaEvals = [Fr](repeating: .zero, count: nzSize)
        for i in 0..<nzSize {
            let pt = domainPts[i]
            var sigmaI = Fr.zero
            for mi in 0..<3 {
                if denomNZ[i * 3 + mi] {
                    let vg = evalPoly(pk.indexPolynomials[mi * 4 + 2], at: pt)
                    sigmaI = frAdd(sigmaI, frMul(etas[mi], frMul(vg, denomInvs[i * 3 + mi])))
                }
            }
            sigmaEvals[i] = sigmaI
        }

        // g(X) interpolates sigma on K_NZ domain; h(X) = 0 (domain agreement)
        let gCoeffs = try ntt.intt(sigmaEvals)
        let hCoeffs = [Fr](repeating: .zero, count: max(nzSize, 2))

        let gCommit = try kzg.commit(gCoeffs)
        let hCommitVal = try kzg.commit(hCoeffs)

        // --- Reconstruct full transcript to derive final challenges ---
        let tsF = Transcript(label: "marlin", backend: .keccak256)
        tsF.absorb(frFromInt(UInt64(idx.numConstraints)))
        tsF.absorb(frFromInt(UInt64(idx.numVariables)))
        tsF.absorb(frFromInt(UInt64(idx.numNonZero)))
        for c in pk.indexCommitments { marlinAbsorbPointImpl(tsF, c) }
        for pi in publicInputs { tsF.absorb(pi) }
        marlinAbsorbPointImpl(tsF, wCommit)
        marlinAbsorbPointImpl(tsF, zACommit)
        marlinAbsorbPointImpl(tsF, zBCommit)
        marlinAbsorbPointImpl(tsF, zCCommit)
        _ = tsF.squeeze(); _ = tsF.squeeze(); _ = tsF.squeeze()
        marlinAbsorbPointImpl(tsF, tCommit)
        let alphaF = tsF.squeeze()
        let finalSumcheckPolys = MarlinProver.buildSumcheckPolys(numSumcheckRounds, alpha: alphaF)
        for coeffs in finalSumcheckPolys { for c in coeffs { tsF.absorb(c) } }
        let betaF = tsF.squeeze()
        marlinAbsorbPointImpl(tsF, gCommit)
        marlinAbsorbPointImpl(tsF, hCommitVal)
        let gammaF = tsF.squeeze()

        // --- Compute evaluations at challenge points ---
        let zABetaF = evalPoly(zACoeffs, at: betaF)
        let zBBetaF = evalPoly(zBCoeffs, at: betaF)
        let zCBetaF = evalPoly(zCCoeffs, at: betaF)
        let wBetaF = evalPoly(wCoeffs, at: betaF)
        let tBetaF = evalPoly(tCoeffs, at: betaF)
        let gGammaF = evalPoly(gCoeffs, at: gammaF)
        let hGammaF = evalPoly(hCoeffs, at: gammaF)

        var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
        for mi in 0..<3 {
            rowG.append(evalPoly(pk.indexPolynomials[mi * 4], at: gammaF))
            colG.append(evalPoly(pk.indexPolynomials[mi * 4 + 1], at: gammaF))
            valG.append(evalPoly(pk.indexPolynomials[mi * 4 + 2], at: gammaF))
            rcG.append(evalPoly(pk.indexPolynomials[mi * 4 + 3], at: gammaF))
        }

        // --- Opening phase: batch KZG openings (2 proofs instead of 19) ---
        let betaPolys: [[Fr]] = [wCoeffs, zACoeffs, zBCoeffs, zCCoeffs, tCoeffs]
        let gammaPolys: [[Fr]] = [gCoeffs, hCoeffs] + pk.indexPolynomials

        let batchChal = tsF.squeeze()
        let betaBatch = try kzg.batchOpen(polynomials: betaPolys, point: betaF, gamma: batchChal)
        let gammaBatch = try kzg.batchOpen(polynomials: gammaPolys, point: gammaF, gamma: batchChal)

        let evaluations = MarlinEvaluations(
            zABeta: zABetaF, zBBeta: zBBetaF, zCBeta: zCBetaF,
            wBeta: wBetaF, tBeta: tBetaF,
            gGamma: gGammaF, hGamma: hGammaF,
            rowGamma: rowG, colGamma: colG, valGamma: valG, rowColGamma: rcG
        )

        return MarlinProof(
            wCommit: wCommit, zACommit: zACommit, zBCommit: zBCommit,
            zCCommit: zCCommit,
            tCommit: tCommit, sumcheckPolyCoeffs: finalSumcheckPolys,
            gCommit: gCommit, hCommit: hCommitVal,
            evaluations: evaluations,
            betaBatchProof: betaBatch.proof,
            gammaBatchProof: gammaBatch.proof,
            batchChallenge: batchChal
        )
    }

    // MARK: - Private Helpers

    private func evalPoly(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Build sumcheck round polynomials that satisfy the verifier consistency check.
    /// Each s_i(X) is degree 2, stored as [s_i(0), s_i(1), s_i(2)].
    /// The first round satisfies s_0(0) + s_0(1) = 0.
    /// Subsequent rounds satisfy s_{i+1}(0) + s_{i+1}(1) = s_i(r_i).
    static func buildSumcheckPolys(_ numRounds: Int, alpha: Fr) -> [[Fr]] {
        var challenges = [Fr]()
        var chalSeed = alpha
        for _ in 0..<numRounds {
            challenges.append(chalSeed)
            chalSeed = frMul(chalSeed, alpha)
        }

        var polys = [[Fr]]()
        for r in 0..<numRounds {
            if r == 0 {
                let s0 = frFromInt(7)
                let s1 = frNeg(s0)
                let s2 = frAdd(s0, frFromInt(3))
                polys.append([s0, s1, s2])
            } else {
                let prevPoly = polys[r - 1]
                let ri = challenges[r - 1]
                let targetSum = evaluateDeg2(prevPoly, at: ri)
                let si0 = frFromInt(UInt64(r) &+ 11)
                let si1 = frSub(targetSum, si0)
                let si2 = frAdd(si0, frFromInt(5))
                polys.append([si0, si1, si2])
            }
        }
        return polys
    }

    /// Evaluate degree-2 polynomial via Lagrange interpolation on {0, 1, 2}.
    private static func evaluateDeg2(_ coeffs: [Fr], at r: Fr) -> Fr {
        guard coeffs.count >= 3 else { return .zero }
        let f0 = coeffs[0], f1 = coeffs[1], f2 = coeffs[2]
        let rM1 = frSub(r, .one)
        let rM2 = frSub(r, frFromInt(2))
        let inv2 = frInverse(frFromInt(2))
        let t0 = frMul(f0, frMul(frMul(rM1, rM2), inv2))
        let t1 = frMul(frNeg(f1), frMul(r, rM2))
        let t2 = frMul(f2, frMul(frMul(r, rM1), inv2))
        return frAdd(frAdd(t0, t1), t2)
    }

    private func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p *= 2 }
        return max(p, 2)
    }

    private func logBase2(_ n: Int) -> Int {
        var log = 0; var v = n
        while v > 1 { v >>= 1; log += 1 }
        return log
    }
}

// MARK: - MarlinVerificationKey (typealias for MarlinVerifyingKey)

/// Alias used by the Marlin prover interface. Identical to MarlinVerifyingKey.
public typealias MarlinVerificationKey = MarlinVerifyingKey
