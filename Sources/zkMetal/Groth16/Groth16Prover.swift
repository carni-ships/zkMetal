// Groth16 Prover
import Foundation
import Metal

public class Groth16Prover {
    public let msm: MetalMSM; public let ntt: NTTEngine
    public init() throws { self.msm = try MetalMSM(); self.ntt = try NTTEngine() }

    public func prove(pk: Groth16ProvingKey, r1cs: R1CSInstance,
                      publicInputs: [Fr], witness: [Fr]) throws -> Groth16Proof {
        let nP = r1cs.numPublic
        var z = [Fr](repeating: .zero, count: r1cs.numVars)
        z[0] = .one; for i in 0..<nP { z[1+i] = publicInputs[i] }
        for i in 0..<witness.count { z[1+nP+i] = witness[i] }
        precondition(r1cs.isSatisfied(z: z), "R1CS not satisfied")
        let h = try computeH(r1cs: r1cs, z: z)
        let r = groth16RandomFr(); let s = groth16RandomFr()
        let pA = try proofA(pk: pk, z: z, r: r)
        let pB = proofBG2(pk: pk, z: z, s: s)
        let pBg1 = try proofBG1(pk: pk, z: z, s: s)
        let pC = try proofC(pk: pk, w: witness, h: h, a: pA, bg1: pBg1, r: r, s: s)
        return Groth16Proof(a: pA, b: pB, c: pC)
    }

    private func computeH(r1cs: R1CSInstance, z: [Fr]) throws -> [Fr] {
        let m = r1cs.numConstraints
        // Domain size: smallest power of 2 >= m
        var domainN = 1
        while domainN < m { domainN <<= 1 }

        // Az, Bz, Cz are the evaluations of A(x), B(x), C(x) on the NTT domain
        let az = r1cs.sparseMatVec(r1cs.aEntries, z)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, z)
        let cz = r1cs.sparseMatVec(r1cs.cEntries, z)

        // Pad to domain size
        var aEvals = [Fr](repeating: .zero, count: domainN)
        var bEvals = [Fr](repeating: .zero, count: domainN)
        var cEvals = [Fr](repeating: .zero, count: domainN)
        for i in 0..<m { aEvals[i] = az[i]; bEvals[i] = bz[i]; cEvals[i] = cz[i] }

        // INTT to get coefficient form of A, B, C
        let aCoeffs = try ntt.intt(aEvals)
        let bCoeffs = try ntt.intt(bEvals)
        let cCoeffs = try ntt.intt(cEvals)

        // Pad to 2*domainN for polynomial multiplication
        let bigN = domainN * 2
        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domainN { aPad[i] = aCoeffs[i]; bPad[i] = bCoeffs[i]; cPad[i] = cCoeffs[i] }

        // Forward NTT on 2*domainN to get evaluations
        let aE = try ntt.ntt(aPad)
        let bE = try ntt.ntt(bPad)
        let cE = try ntt.ntt(cPad)

        // Pointwise: P(x) = A(x)*B(x) - C(x)
        var pE = [Fr](repeating: .zero, count: bigN)
        for i in 0..<bigN { pE[i] = frSub(frMul(aE[i], bE[i]), cE[i]) }

        // INTT to get P in coefficient form
        let pCoeffs = try ntt.intt(pE)

        // Polynomial long division: H = P / (x^domainN - 1)
        // Since P is divisible by Z = x^domainN - 1, remainder is 0
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
        let n = min(z.count, pk.a_query.count)
        let m = try doMSM(pts: Array(pk.a_query.prefix(n)), sc: Array(z.prefix(n)))
        return pointAdd(pointAdd(pk.alpha_g1, m), pointScalarMul(pk.delta_g1, r))
    }

    private func proofBG2(pk: Groth16ProvingKey, z: [Fr], s: Fr) -> G2ProjectivePoint {
        let n = min(z.count, pk.b_g2_query.count); var r = pk.beta_g2
        for i in 0..<n { if !z[i].isZero { r = g2Add(r, g2ScalarMul(pk.b_g2_query[i], frToInt(z[i]))) } }
        return g2Add(r, g2ScalarMul(pk.delta_g2, frToInt(s)))
    }

    private func proofBG1(pk: Groth16ProvingKey, z: [Fr], s: Fr) throws -> PointProjective {
        let n = min(z.count, pk.b_g1_query.count)
        let m = try doMSM(pts: Array(pk.b_g1_query.prefix(n)), sc: Array(z.prefix(n)))
        return pointAdd(pointAdd(pk.beta_g1, m), pointScalarMul(pk.delta_g1, s))
    }

    private func proofC(pk: Groth16ProvingKey, w: [Fr], h: [Fr],
                         a: PointProjective, bg1: PointProjective, r: Fr, s: Fr) throws -> PointProjective {
        let lM = try doMSM(pts: Array(pk.l_query.prefix(min(w.count, pk.l_query.count))),
                            sc: Array(w.prefix(min(w.count, pk.l_query.count))))
        let hN = min(h.count, pk.h_query.count)
        let hM = hN > 0 ? try doMSM(pts: Array(pk.h_query.prefix(hN)), sc: Array(h.prefix(hN))) : pointIdentity()
        var res = pointAdd(lM, hM)
        res = pointAdd(res, pointScalarMul(a, s))
        res = pointAdd(res, pointScalarMul(bg1, r))
        return pointAdd(res, pointNeg(pointScalarMul(pk.delta_g1, frMul(r, s))))
    }

    private func doMSM(pts: [PointProjective], sc: [Fr]) throws -> PointProjective {
        let n = pts.count; if n == 0 { return pointIdentity() }
        // Use naive scalar mul for correctness (MSM has a bug for small inputs)
        var result = pointIdentity()
        for i in 0..<n {
            if !sc[i].isZero { result = pointAdd(result, pointScalarMul(pts[i], sc[i])) }
        }
        return result
    }
}
