// Groth16 Prover
import Foundation
import Metal

public class Groth16Prover {
    public let msm: MetalMSM; public let ntt: NTTEngine
    public var profileGroth16 = false
    public init() throws { self.msm = try MetalMSM(); self.ntt = try NTTEngine() }

    public func prove(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                      publicInputs: [Fr], witness: [Fr]) throws -> Groth16Proof {
        let nP = r1cs.numPublic
        var z = [Fr](repeating: .zero, count: r1cs.numVars)
        z[0] = .one; for i in 0..<nP { z[1+i] = publicInputs[i] }
        for i in 0..<witness.count { z[1+nP+i] = witness[i] }
        precondition(r1cs.isSatisfied(z: z), "R1CS not satisfied")
        let r = groth16RandomFr(); let s = groth16RandomFr()

        // Phase 1: computeH, proofA, proofBG2, proofBG1 are all independent
        var _t = CFAbsoluteTimeGetCurrent()
        var h: [Fr]?; var pA: PointProjective?; var pB: G2ProjectivePoint?; var pBg1: PointProjective?
        var hError: Error?; var aError: Error?; var bg1Error: Error?
        let group = DispatchGroup()

        // proofBG2 is CPU-only and the slowest single phase - run on separate thread
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            pB = proofBG2(pk: pk, z: z, s: s)
            group.leave()
        }

        // computeH uses CPU NTT for small sizes, proofA/BG1 use GPU MSM
        // Run computeH on a CPU thread while proofA uses GPU on main thread
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do { h = try computeH(r1cs: r1cs, z: z) } catch { hError = error }
            group.leave()
        }
        // proofA and proofBG1 use GPU MSM, run on main thread
        do { pA = try proofA(pk: pk, z: z, r: r) } catch { aError = error }
        do { pBg1 = try proofBG1(pk: pk, z: z, s: s) } catch { bg1Error = error }

        group.wait()
        if profileGroth16 { let _e = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [groth16] phase1 (H+A+BG2+BG1): %.2f ms\n", (_e - _t) * 1000), stderr); _t = _e }

        if let e = hError { throw e }
        if let e = aError { throw e }
        if let e = bg1Error { throw e }

        let pC = try proofC(pk: pk, w: witness, h: h!, a: pA!, bg1: pBg1!, r: r, s: s)
        if profileGroth16 { let _e = CFAbsoluteTimeGetCurrent(); fputs(String(format: "  [groth16] proofC: %.2f ms\n", (_e - _t) * 1000), stderr); _t = _e }
        return Groth16Proof(a: pA!, b: pB!, c: pC)
    }

    private func computeH(r1cs: R1CSInstance, z: [Fr]) throws -> [Fr] {
        let m = r1cs.numConstraints
        var domainN = 1; var logN = 0
        while domainN < m { domainN <<= 1; logN += 1 }

        let az = r1cs.sparseMatVec(r1cs.aEntries, z)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, z)
        let cz = r1cs.sparseMatVec(r1cs.cEntries, z)

        // Pad to domain size
        var aEvals = [Fr](repeating: .zero, count: domainN)
        var bEvals = [Fr](repeating: .zero, count: domainN)
        var cEvals = [Fr](repeating: .zero, count: domainN)
        for i in 0..<m { aEvals[i] = az[i]; bEvals[i] = bz[i]; cEvals[i] = cz[i] }

        let bigN = domainN * 2
        let logBigN = logN + 1

        // Use CPU NTT for small sizes to avoid GPU dispatch overhead
        let useCPU = bigN <= 4096

        let aCoeffs: [Fr]; let bCoeffs: [Fr]; let cCoeffs: [Fr]
        if useCPU {
            aCoeffs = cINTT_Fr(aEvals, logN: logN)
            bCoeffs = cINTT_Fr(bEvals, logN: logN)
            cCoeffs = cINTT_Fr(cEvals, logN: logN)
        } else {
            aCoeffs = try ntt.intt(aEvals); bCoeffs = try ntt.intt(bEvals); cCoeffs = try ntt.intt(cEvals)
        }

        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domainN { aPad[i] = aCoeffs[i]; bPad[i] = bCoeffs[i]; cPad[i] = cCoeffs[i] }

        let aE: [Fr]; let bE: [Fr]; let cE: [Fr]
        if useCPU {
            aE = cNTT_Fr(aPad, logN: logBigN)
            bE = cNTT_Fr(bPad, logN: logBigN)
            cE = cNTT_Fr(cPad, logN: logBigN)
        } else {
            aE = try ntt.ntt(aPad); bE = try ntt.ntt(bPad); cE = try ntt.ntt(cPad)
        }

        var pE = [Fr](repeating: .zero, count: bigN)
        for i in 0..<bigN { pE[i] = frSub(frMul(aE[i], bE[i]), cE[i]) }

        let pCoeffs: [Fr]
        if useCPU {
            pCoeffs = cINTT_Fr(pE, logN: logBigN)
        } else {
            pCoeffs = try ntt.intt(pE)
        }

        var hCoeffs = [Fr](repeating: .zero, count: domainN)
        var rem = Array(pCoeffs.prefix(bigN))
        for i in stride(from: bigN - 1, through: domainN, by: -1) {
            let q = rem[i]
            hCoeffs[i - domainN] = q
            rem[i] = .zero
            rem[i - domainN] = frAdd(rem[i - domainN], q)
        }
        return hCoeffs
    }

    private func proofA(pk: Groth16ProvingKey, z: [Fr], r: Fr) throws -> PointProjective {
        let n = min(z.count, pk.a_query_affine.count)
        let m = try doMSMAffine(aff: Array(pk.a_query_affine.prefix(n)), sc: Array(z.prefix(n)))
        return pointAdd(pointAdd(pk.alpha_g1, m), pointScalarMul(pk.delta_g1, r))
    }

    private func proofBG2(pk: Groth16ProvingKey, z: [Fr], s: Fr) -> G2ProjectivePoint {
        let n = min(z.count, pk.b_g2_query.count)
        var pts = [G2ProjectivePoint](); var scs = [[UInt64]]()
        pts.reserveCapacity(n + 1); scs.reserveCapacity(n + 1)
        for i in 0..<n {
            if !z[i].isZero { pts.append(pk.b_g2_query[i]); scs.append(frToInt(z[i])) }
        }
        pts.append(pk.delta_g2); scs.append(frToInt(s))
        let m = g2PippengerMSM(points: pts, scalars: scs)
        return g2Add(pk.beta_g2, m)
    }

    private func proofBG1(pk: Groth16ProvingKey, z: [Fr], s: Fr) throws -> PointProjective {
        let n = min(z.count, pk.b_g1_query_affine.count)
        let m = try doMSMAffine(aff: Array(pk.b_g1_query_affine.prefix(n)), sc: Array(z.prefix(n)))
        return pointAdd(pointAdd(pk.beta_g1, m), pointScalarMul(pk.delta_g1, s))
    }

    private func proofC(pk: Groth16ProvingKey, w: [Fr], h: [Fr],
                         a: PointProjective, bg1: PointProjective, r: Fr, s: Fr) throws -> PointProjective {
        let lN = min(w.count, pk.l_query_affine.count)
        let lM = try doMSMAffine(aff: Array(pk.l_query_affine.prefix(lN)), sc: Array(w.prefix(lN)))
        let hN = min(h.count, pk.h_query_affine.count)
        let hM = hN > 0 ? try doMSMAffine(aff: Array(pk.h_query_affine.prefix(hN)), sc: Array(h.prefix(hN))) : pointIdentity()
        var res = pointAdd(lM, hM)
        res = pointAdd(res, pointScalarMul(a, s))
        res = pointAdd(res, pointScalarMul(bg1, r))
        return pointAdd(res, pointNeg(pointScalarMul(pk.delta_g1, frMul(r, s))))
    }

    /// MSM with pre-computed affine points (avoids batchToAffine per call)
    private func doMSMAffine(aff: [PointAffine], sc: [Fr]) throws -> PointProjective {
        let n = aff.count; if n == 0 { return pointIdentity() }
        var sl = [[UInt32]](); sl.reserveCapacity(n)
        for s in sc { sl.append(frToLimbs(s)) }
        return n >= 256 ? try msm.msm(points: aff, scalars: sl) : cPippengerMSM(points: aff, scalars: sl)
    }
}

// MARK: - G2 Straus MSM

/// Straus (Shamir's trick) multi-scalar multiplication for G2 points.
/// Processes all scalars bit-by-bit from MSB to LSB with a single shared doubling.
/// Much faster than N sequential scalar muls: O(256 doubles + 256*density adds)
/// vs O(N * 256 doubles + N * 128 adds).
public func g2PippengerMSM(points: [G2ProjectivePoint], scalars: [[UInt64]]) -> G2ProjectivePoint {
    let n = points.count
    if n == 0 { return g2Identity() }
    if n == 1 { return g2ScalarMul(points[0], scalars[0]) }

    // Find highest set bit to skip leading zeros
    var maxBit = 0
    for sc in scalars {
        for w in 0..<sc.count where sc[w] != 0 {
            let bit = w * 64 + (63 - sc[w].leadingZeroBitCount)
            if bit > maxBit { maxBit = bit }
        }
    }

    // Straus: scan from highest set bit to LSB
    var result = g2Identity()
    for bit in stride(from: maxBit, through: 0, by: -1) {
        result = g2Double(result)
        let wordIdx = bit / 64
        let bitIdx = bit % 64
        for i in 0..<n {
            let sc = scalars[i]
            if wordIdx < sc.count && (sc[wordIdx] >> bitIdx) & 1 == 1 {
                result = g2Add(result, points[i])
            }
        }
    }
    return result
}
