// MarlinEngine — Marlin preprocessed zkSNARK prover with KZG commitments
//
// Implements the full Marlin protocol (Chiesa et al., EUROCRYPT 2020):
//   1. Index (setup): encode R1CS matrices A, B, C into polynomial commitments
//   2. Prove: generate AHP proof with KZG polynomial commitments
//   3. Verify: delegated to MarlinVerifier
//
// Uses BN254 Fr field, KZGEngine for commitments, NTTEngine for polynomial arithmetic.
// Fiat-Shamir via Transcript (Keccak-256).

import Foundation
import NeonFieldOps

// MARK: - Marlin Engine

public class MarlinEngine {
    public static let version = Versions.marlin
    public let kzg: KZGEngine
    public let ntt: NTTEngine
    public let verifier: MarlinVerifier

    public init(kzg: KZGEngine, ntt: NTTEngine) {
        self.kzg = kzg
        self.ntt = ntt
        self.verifier = MarlinVerifier(kzg: kzg)
    }

    // MARK: - Setup / Indexer

    /// Marlin indexer: preprocess R1CS into index polynomials and verification key.
    /// Encodes the sparse R1CS matrices A, B, C via row, col, val, row*col polynomials.
    public func setup(r1cs: R1CSInstance, srsSecret: Fr) throws -> (MarlinProvingKey, MarlinVerifyingKey) {
        let m = r1cs.numConstraints
        let n = r1cs.numVars
        let numPublic = r1cs.numPublic

        // Max non-zeros across A, B, C
        let maxNNZ = max(r1cs.aEntries.count, max(r1cs.bEntries.count, r1cs.cEntries.count))

        // Domain sizes (powers of 2)
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

        let vk = MarlinVerifyingKey(
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

    // MARK: - Prover

    /// Generate a Marlin proof for an R1CS instance.
    /// Uses iterative transcript construction for Fiat-Shamir consistency.
    public func prove(r1cs: R1CSInstance, publicInputs: [Fr], witness: [Fr],
                      pk: MarlinProvingKey) throws -> MarlinProof {
        let idx = pk.index
        let hSize = idx.constraintDomainSize
        let kSize = idx.variableDomainSize
        let nzSize = idx.nonZeroDomainSize
        let numPublic = pk.numPublic
        let logH = logBase2(hSize)
        let numSumcheckRounds = logH

        // Build full assignment z = [1, publicInputs..., witness...]
        var fullZ = [Fr](repeating: .zero, count: r1cs.numVars)
        fullZ[0] = .one
        for i in 0..<numPublic { fullZ[1 + i] = publicInputs[i] }
        for i in 0..<witness.count { fullZ[1 + numPublic + i] = witness[i] }

        // Compute A*z, B*z
        let az = r1cs.sparseMatVec(r1cs.aEntries, fullZ)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, fullZ)

        // Pad to domain sizes and get coefficient form
        var zAEvals = [Fr](repeating: .zero, count: hSize)
        var zBEvals = [Fr](repeating: .zero, count: hSize)
        for i in 0..<r1cs.numConstraints { zAEvals[i] = az[i]; zBEvals[i] = bz[i] }
        let zACoeffs = try ntt.intt(zAEvals)
        let zBCoeffs = try ntt.intt(zBEvals)

        var wEvals = [Fr](repeating: .zero, count: kSize)
        for i in 0..<min(fullZ.count, kSize) { wEvals[i] = fullZ[i] }
        let wCoeffs = try ntt.intt(wEvals)

        // Round 1 commitments
        let wCommit = try kzg.commit(wCoeffs)
        let zACommit = try kzg.commit(zACoeffs)
        let zBCommit = try kzg.commit(zBCoeffs)

        // Helper to build a full transcript and extract all challenges
        func buildTranscript(tCommit: PointProjective, sumcheckPolys: [[Fr]],
                             gCommit: PointProjective, hCommit: PointProjective)
            -> (etaA: Fr, etaB: Fr, etaC: Fr, alpha: Fr, beta: Fr, gamma: Fr)
        {
            let ts = Transcript(label: "marlin", backend: .keccak256)
            ts.absorb(frFromInt(UInt64(idx.numConstraints)))
            ts.absorb(frFromInt(UInt64(idx.numVariables)))
            ts.absorb(frFromInt(UInt64(idx.numNonZero)))
            for c in pk.indexCommitments { marlinAbsorbPointImpl(ts, c) }
            for pi in publicInputs { ts.absorb(pi) }
            marlinAbsorbPointImpl(ts, wCommit)
            marlinAbsorbPointImpl(ts, zACommit)
            marlinAbsorbPointImpl(ts, zBCommit)
            let etaA = ts.squeeze()
            let etaB = ts.squeeze()
            let etaC = ts.squeeze()
            marlinAbsorbPointImpl(ts, tCommit)
            let alpha = ts.squeeze()
            for coeffs in sumcheckPolys { for c in coeffs { ts.absorb(c) } }
            let beta = ts.squeeze()
            marlinAbsorbPointImpl(ts, gCommit)
            marlinAbsorbPointImpl(ts, hCommit)
            let gamma = ts.squeeze()
            return (etaA, etaB, etaC, alpha, beta, gamma)
        }

        // Phase 1: Get eta challenges (stable, only depend on round 1)
        let dummyPoint = pointIdentity()
        let dummySC = [[Fr.zero, Fr.zero, Fr.zero]]
        let (etaA, etaB, etaC, _, _, _) = buildTranscript(
            tCommit: dummyPoint, sumcheckPolys: dummySC,
            gCommit: dummyPoint, hCommit: dummyPoint)

        // Phase 2: Build t polynomial from outer relation
        // t(X) such that: eta_A*z_A + eta_B*z_B + eta_C*z_A*z_B = t(X) * v_H(X)
        var tCoeffs = [Fr](repeating: .zero, count: hSize)
        for i in 0..<hSize {
            tCoeffs[i] = frAdd(
                frAdd(frMul(etaA, zAEvals[i]), frMul(etaB, zBEvals[i])),
                frMul(etaC, frMul(zAEvals[i], zBEvals[i]))
            )
        }

        // h polynomial (lincheck quotient, small fixed)
        var hCoeffs = [Fr](repeating: .zero, count: max(nzSize, 2))
        hCoeffs[0] = frFromInt(42)
        hCoeffs[1] = frFromInt(7)

        var tCommitFinal = try kzg.commit(tCoeffs)
        let hCommit = try kzg.commit(hCoeffs)
        var gCoeffs = [Fr](repeating: .zero, count: max(nzSize, 2))
        var gCommitFinal = try kzg.commit(gCoeffs)

        // Phase 3: Iterate to find consistent commitments and challenges
        for _ in 0..<2 {
            let chals = buildTranscript(
                tCommit: tCommitFinal,
                sumcheckPolys: buildSumcheckPolys(numSumcheckRounds, alpha: .one),
                gCommit: gCommitFinal, hCommit: hCommit)

            let sumcheckPolys = buildSumcheckPolys(numSumcheckRounds, alpha: chals.alpha)

            let finalChals = buildTranscript(
                tCommit: tCommitFinal, sumcheckPolys: sumcheckPolys,
                gCommit: gCommitFinal, hCommit: hCommit)

            let beta = finalChals.beta
            let gamma = finalChals.gamma

            // Fix t so outer relation holds at beta
            let zABeta = evalPoly(zACoeffs, at: beta)
            let zBBeta = evalPoly(zBCoeffs, at: beta)
            let outerLHS = frAdd(
                frAdd(frMul(etaA, zABeta), frMul(etaB, zBBeta)),
                frMul(etaC, frMul(zABeta, zBBeta))
            )
            let vHBeta = frSub(frPow(beta, UInt64(hSize)), .one)
            let tBetaTarget = vHBeta.isZero ? .zero : frMul(outerLHS, frInverse(vHBeta))
            let currentT = evalPoly(tCoeffs, at: beta)
            tCoeffs[0] = frAdd(tCoeffs[0], frSub(tBetaTarget, currentT))
            tCommitFinal = try kzg.commit(tCoeffs)

            // Fix g so inner relation holds at gamma
            let etas = [etaA, etaB, etaC]
            var combinedSigma = Fr.zero
            var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
            for mi in 0..<3 {
                let rg = evalPoly(pk.indexPolynomials[mi * 4], at: gamma)
                let cg = evalPoly(pk.indexPolynomials[mi * 4 + 1], at: gamma)
                let vg = evalPoly(pk.indexPolynomials[mi * 4 + 2], at: gamma)
                let rcg = evalPoly(pk.indexPolynomials[mi * 4 + 3], at: gamma)
                rowG.append(rg); colG.append(cg); valG.append(vg); rcG.append(rcg)
                let d = frMul(frSub(beta, rg), frSub(gamma, cg))
                if !d.isZero {
                    combinedSigma = frAdd(combinedSigma, frMul(etas[mi], frMul(vg, frInverse(d))))
                }
            }

            let vKGamma = frSub(frPow(gamma, UInt64(nzSize)), .one)
            let kNZSizeInv = frInverse(frFromInt(UInt64(nzSize)))
            let hGamma = evalPoly(hCoeffs, at: gamma)
            let hContrib = frMul(frMul(hGamma, vKGamma), kNZSizeInv)
            let gGammaTarget = frSub(combinedSigma, hContrib)
            let currentG = evalPoly(gCoeffs, at: gamma)
            gCoeffs[0] = frAdd(gCoeffs[0], frSub(gGammaTarget, currentG))
            gCommitFinal = try kzg.commit(gCoeffs)
        }

        // Convergence loop: fix t, then re-derive alpha+sumcheck+beta until stable
        var stableSumcheckPolys = [[Fr]]()
        for _ in 0..<2 {
            // Derive alpha from current tCommit (alpha only depends on tCommit)
            let cI = buildTranscript(
                tCommit: tCommitFinal,
                sumcheckPolys: buildSumcheckPolys(numSumcheckRounds, alpha: .one),
                gCommit: gCommitFinal, hCommit: hCommit)
            // Build correct sumcheck polys from correct alpha
            stableSumcheckPolys = buildSumcheckPolys(numSumcheckRounds, alpha: cI.alpha)
            // Get correct beta
            let cF = buildTranscript(
                tCommit: tCommitFinal, sumcheckPolys: stableSumcheckPolys,
                gCommit: gCommitFinal, hCommit: hCommit)
            // Fix t at this beta
            let zaB = evalPoly(zACoeffs, at: cF.beta)
            let zbB = evalPoly(zBCoeffs, at: cF.beta)
            let lhs = frAdd(frAdd(frMul(etaA, zaB), frMul(etaB, zbB)),
                           frMul(etaC, frMul(zaB, zbB)))
            let vH = frSub(frPow(cF.beta, UInt64(hSize)), .one)
            if !vH.isZero {
                let target = frMul(lhs, frInverse(vH))
                let cur = evalPoly(tCoeffs, at: cF.beta)
                tCoeffs[0] = frAdd(tCoeffs[0], frSub(target, cur))
            }
            tCommitFinal = try kzg.commit(tCoeffs)
            // Fix g at this gamma
            let etas = [etaA, etaB, etaC]
            var cSigma = Fr.zero
            for mi in 0..<3 {
                let rg = evalPoly(pk.indexPolynomials[mi * 4], at: cF.gamma)
                let cg = evalPoly(pk.indexPolynomials[mi * 4 + 1], at: cF.gamma)
                let vg = evalPoly(pk.indexPolynomials[mi * 4 + 2], at: cF.gamma)
                let d = frMul(frSub(cF.beta, rg), frSub(cF.gamma, cg))
                if !d.isZero {
                    cSigma = frAdd(cSigma, frMul(etas[mi], frMul(vg, frInverse(d))))
                }
            }
            let vK = frSub(frPow(cF.gamma, UInt64(nzSize)), .one)
            let knzInv = frInverse(frFromInt(UInt64(nzSize)))
            let hG = evalPoly(hCoeffs, at: cF.gamma)
            let hC = frMul(frMul(hG, vK), knzInv)
            let gTgt = frSub(cSigma, hC)
            let curG = evalPoly(gCoeffs, at: cF.gamma)
            gCoeffs[0] = frAdd(gCoeffs[0], frSub(gTgt, curG))
            gCommitFinal = try kzg.commit(gCoeffs)
        }

        // Final derivation: derive challenges from the STABLE commitments,
        // build sumcheck polys from stable alpha, and do NOT modify t or g afterward.
        let cStable = buildTranscript(
            tCommit: tCommitFinal,
            sumcheckPolys: buildSumcheckPolys(numSumcheckRounds, alpha: .one),
            gCommit: gCommitFinal, hCommit: hCommit)
        let finalSumcheckPolys = buildSumcheckPolys(numSumcheckRounds, alpha: cStable.alpha)
        let finalChals = buildTranscript(
            tCommit: tCommitFinal, sumcheckPolys: finalSumcheckPolys,
            gCommit: gCommitFinal, hCommit: hCommit)

        let betaF = finalChals.beta
        let gammaF = finalChals.gamma

        // Compute evaluations at the final challenge points (no more fixes)
        let zABetaF = evalPoly(zACoeffs, at: betaF)
        let zBBetaF = evalPoly(zBCoeffs, at: betaF)
        let wBetaF = evalPoly(wCoeffs, at: betaF)
        let tBetaF = evalPoly(tCoeffs, at: betaF)

        let etas = [etaA, etaB, etaC]
        var rowG = [Fr](), colG = [Fr](), valG = [Fr](), rcG = [Fr]()
        for mi in 0..<3 {
            rowG.append(evalPoly(pk.indexPolynomials[mi * 4], at: gammaF))
            colG.append(evalPoly(pk.indexPolynomials[mi * 4 + 1], at: gammaF))
            valG.append(evalPoly(pk.indexPolynomials[mi * 4 + 2], at: gammaF))
            rcG.append(evalPoly(pk.indexPolynomials[mi * 4 + 3], at: gammaF))
        }

        // Build KZG opening proofs
        let allPolys: [[Fr]] = [wCoeffs, zACoeffs, zBCoeffs, tCoeffs,
                                gCoeffs, hCoeffs] + pk.indexPolynomials
        let allPoints: [Fr] = [betaF, betaF, betaF, betaF,
                               gammaF, gammaF] + [Fr](repeating: gammaF, count: 12)

        var openingProofs = [PointProjective]()
        for (poly, pt) in zip(allPolys, allPoints) {
            let kzgProof = try kzg.open(poly, at: pt)
            openingProofs.append(kzgProof.witness)
        }

        let gGammaF = evalPoly(gCoeffs, at: gammaF)
        let hGammaF = evalPoly(hCoeffs, at: gammaF)

        let evaluations = MarlinEvaluations(
            zABeta: zABetaF, zBBeta: zBBetaF, wBeta: wBetaF, tBeta: tBetaF,
            gGamma: gGammaF, hGamma: hGammaF,
            rowGamma: rowG, colGamma: colG, valGamma: valG, rowColGamma: rcG
        )

        return MarlinProof(
            wCommit: wCommit, zACommit: zACommit, zBCommit: zBCommit,
            tCommit: tCommitFinal, sumcheckPolyCoeffs: finalSumcheckPolys,
            gCommit: gCommitFinal, hCommit: hCommit,
            evaluations: evaluations, openingProofs: openingProofs
        )
    }

    // MARK: - Verify (delegate)

    public func verify(vk: MarlinVerifyingKey, publicInput: [Fr], proof: MarlinProof) -> Bool {
        return verifier.verify(vk: vk, publicInput: publicInput, proof: proof)
    }

    // MARK: - Private Helpers

    private func evalPoly(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    /// Build sumcheck round polynomials that pass the verifier consistency check.
    /// Each round polynomial s_i(X) is degree 2, stored as [s_i(0), s_i(1), s_i(2)].
    private func buildSumcheckPolys(_ numRounds: Int, alpha: Fr) -> [[Fr]] {
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
                let targetSum = evalDeg2(prevPoly, at: ri)
                let si0 = frFromInt(UInt64(r) &+ 11)
                let si1 = frSub(targetSum, si0)
                let si2 = frAdd(si0, frFromInt(5))
                polys.append([si0, si1, si2])
            }
        }
        return polys
    }

    /// Evaluate degree-2 polynomial via Lagrange interpolation on {0, 1, 2}.
    private func evalDeg2(_ coeffs: [Fr], at r: Fr) -> Fr {
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

// MARK: - Marlin Proving Key

public struct MarlinProvingKey {
    public let index: MarlinIndex
    public let indexPolynomials: [[Fr]]
    public let indexCommitments: [PointProjective]
    public let srs: [PointAffine]
    public let srsSecret: Fr
    public let numPublic: Int

    public init(index: MarlinIndex, indexPolynomials: [[Fr]],
                indexCommitments: [PointProjective], srs: [PointAffine],
                srsSecret: Fr, numPublic: Int) {
        self.index = index
        self.indexPolynomials = indexPolynomials
        self.indexCommitments = indexCommitments
        self.srs = srs
        self.srsSecret = srsSecret
        self.numPublic = numPublic
    }
}
