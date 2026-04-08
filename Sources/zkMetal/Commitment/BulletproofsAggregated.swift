// Bulletproofs Aggregated Range Proofs
//
// Proves M values are all in range [0, 2^n) in a single proof.
// Proof size is O(log(M*n)) — logarithmic in the number of values,
// which is the key optimization making Bulletproofs practical for
// confidential transactions (e.g., Monero, Mimblewimble).
//
// Protocol:
//   - Each value v_j gets a bit decomposition a_L^(j)
//   - Vectors are concatenated into length M*n
//   - Generator vectors G, H of length M*n are used
//   - A single inner product argument proves all M range constraints
//
// References:
//   - Bulletproofs: Short Proofs for Confidential Transactions (Bunz et al. 2018) Section 4.2

import Foundation
import NeonFieldOps

// MARK: - Aggregated Range Proof

/// An aggregated Bulletproofs range proof for M values.
public struct BulletproofsAggregatedProof {
    /// Commitment to concatenated bit decompositions
    public let A: PointProjective
    /// Blinding commitment
    public let S: PointProjective
    /// T1 commitment (t1 coefficient)
    public let T1: PointProjective
    /// T2 commitment (t2 coefficient)
    public let T2: PointProjective
    /// Blinding scalar taux
    public let taux: Fr
    /// Blinding scalar mu
    public let mu: Fr
    /// Evaluated inner product tHat
    public let tHat: Fr
    /// Inner product proof
    public let innerProductProof: InnerProductProof
    /// Number of values proved
    public let m: Int

    public init(A: PointProjective, S: PointProjective,
                T1: PointProjective, T2: PointProjective,
                taux: Fr, mu: Fr, tHat: Fr,
                innerProductProof: InnerProductProof, m: Int) {
        self.A = A
        self.S = S
        self.T1 = T1
        self.T2 = T2
        self.taux = taux
        self.mu = mu
        self.tHat = tHat
        self.innerProductProof = innerProductProof
        self.m = m
    }
}

// MARK: - Aggregated Prover

/// Prover for aggregated Bulletproofs range proofs.
/// Proves M values are all in [0, 2^n) in a single proof.
public class BulletproofsAggregatedProver {

    /// Create an aggregated range proof for M values.
    ///
    /// - Parameters:
    ///   - values: array of M values to prove in range
    ///   - gammas: blinding factors for each value commitment
    ///   - params: Bulletproofs parameters (must have n >= bit size)
    /// - Returns: (Vs, proof) where Vs are value commitments and proof is the aggregated proof
    public static func prove(
        values: [UInt64], gammas: [Fr],
        params: BulletproofsParams
    ) -> (Vs: [PointProjective], proof: BulletproofsAggregatedProof) {
        let m = values.count
        let n = params.n
        precondition(m > 0, "Need at least one value")
        precondition(gammas.count == m, "Need one gamma per value")
        precondition(n >= 8 && (n & (n - 1)) == 0, "n must be power of 2")
        for v in values {
            precondition(n == 64 || v < (1 << n), "Value out of range")
        }

        // Need M*n generators. Generate extended params if needed.
        let mn = m * n
        let extParams = BulletproofsParams.generate(n: nextPowerOf2(mn))

        let gProj = pointFromAffine(extParams.g)
        let hProj = pointFromAffine(extParams.h)

        // Step 1: Commit to each value
        var Vs = [PointProjective]()
        Vs.reserveCapacity(m)
        for j in 0..<m {
            let vFr = frFromInt(values[j])
            let Vj = pointAdd(cPointScalarMul(gProj, vFr), cPointScalarMul(hProj, gammas[j]))
            Vs.append(Vj)
        }

        // Step 2: Concatenated bit decompositions
        // a_L = [bits(v_0), bits(v_1), ..., bits(v_{m-1})]  (length M*n)
        // a_R = a_L - 1^{M*n}
        var aL = [Fr](repeating: Fr.zero, count: mn)
        var aR = [Fr](repeating: Fr.zero, count: mn)
        for j in 0..<m {
            for i in 0..<n {
                let idx = j * n + i
                if (values[j] >> i) & 1 == 1 {
                    aL[idx] = Fr.one
                    // aR[idx] = 0
                } else {
                    // aL[idx] = 0
                    aR[idx] = frNeg(Fr.one)
                }
            }
        }

        // Blinding factors
        let alpha = deterministicBlinding(seed: 10, gammas: gammas)
        let rho = deterministicBlinding(seed: 20, gammas: gammas)
        let tau1 = deterministicBlinding(seed: 30, gammas: gammas)
        let tau2 = deterministicBlinding(seed: 40, gammas: gammas)

        // Random s_L, s_R
        var sL = [Fr](repeating: Fr.zero, count: mn)
        var sR = [Fr](repeating: Fr.zero, count: mn)
        for i in 0..<mn {
            sL[i] = deterministicBlinding(seed: UInt64(1000 + i), gammas: gammas)
            sR[i] = deterministicBlinding(seed: UInt64(2000 + i), gammas: gammas)
        }

        // Pad vectors to next power of 2 if needed
        let padN = nextPowerOf2(mn)
        if padN > mn {
            aL.append(contentsOf: [Fr](repeating: Fr.zero, count: padN - mn))
            aR.append(contentsOf: [Fr](repeating: frNeg(Fr.one), count: padN - mn))
            sL.append(contentsOf: [Fr](repeating: Fr.zero, count: padN - mn))
            sR.append(contentsOf: [Fr](repeating: Fr.zero, count: padN - mn))
        }

        // A = <a_L, G> + <a_R, H> + alpha*h
        let A = computeVectorCommitment(aL, aR, alpha, extParams)

        // S = <s_L, G> + <s_R, H> + rho*h
        let S = computeVectorCommitment(sL, sR, rho, extParams)

        // Fiat-Shamir: y, z
        var transcript = BPTranscript()
        transcript.appendLabel("bulletproofs-aggregated")
        for j in 0..<m { transcript.appendPoint(Vs[j]) }
        transcript.appendPoint(A)
        transcript.appendPoint(S)
        transcript.appendLabel("y")
        let y = transcript.challenge()

        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        // Precompute y powers (length padN)
        var yPow = [Fr](repeating: Fr.one, count: padN)
        for i in 1..<padN {
            yPow[i] = frMul(yPow[i - 1], y)
        }

        // Precompute 2^n powers for each sub-proof
        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            twoPow[i] = frMul(twoPow[i - 1], two)
        }

        // z powers: z, z^2, z^3, ..., z^(m+1)
        var zPow = [Fr](repeating: Fr.one, count: m + 2)
        for i in 1..<(m + 2) {
            zPow[i] = frMul(zPow[i - 1], z)
        }

        // Build r0, r1 for the aggregated case
        // l(x) = (a_L - z*1^{M*n}) + s_L * x
        // r(x) = y^{M*n} . (a_R + z*1^{M*n} + s_R*x) + sum_j z^{j+2} * (0...0, 2^n, 0...0)
        //         where the 2^n block is in position j

        var l0 = [Fr](repeating: Fr.zero, count: padN)
        var r0 = [Fr](repeating: Fr.zero, count: padN)
        var r1 = [Fr](repeating: Fr.zero, count: padN)

        for i in 0..<padN {
            l0[i] = frSub(aL[i], z)
            let aRpZ = frAdd(aR[i], z)
            r0[i] = frMul(yPow[i], aRpZ)
            r1[i] = frMul(yPow[i], sR[i])
        }

        // Add z^{j+2} * 2^{i mod n} terms to r0
        for j in 0..<m {
            for i in 0..<n {
                let idx = j * n + i
                if idx < padN {
                    r0[idx] = frAdd(r0[idx], frMul(zPow[j + 2], twoPow[i]))
                }
            }
        }

        // t0, t1, t2
        let t0 = frInnerProductPad(l0, r0)
        let t1 = frAdd(frInnerProductPad(l0, r1), frInnerProductPad(sL, r0))
        let t2 = frInnerProductPad(sL, r1)

        // Commit to T1, T2
        let T1 = pointAdd(cPointScalarMul(gProj, t1), cPointScalarMul(hProj, tau1))
        let T2 = pointAdd(cPointScalarMul(gProj, t2), cPointScalarMul(hProj, tau2))

        // Challenge x
        transcript.appendPoint(T1)
        transcript.appendPoint(T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()
        let x2 = frMul(x, x)

        // Evaluate l, r at x: lVec[i] = l0[i] + x*sL[i], rVec[i] = r0[i] + x*r1[i]
        var lVec = [Fr](repeating: Fr.zero, count: padN)
        var rVec = [Fr](repeating: Fr.zero, count: padN)
        l0.withUnsafeBytes { l0Buf in
            withUnsafeBytes(of: x) { xBuf in
                sL.withUnsafeBytes { slBuf in
                    lVec.withUnsafeMutableBytes { lBuf in
                        bn254_fr_batch_linear_combine(
                            l0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            slBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(padN))
                    }
                }
            }
        }
        r0.withUnsafeBytes { r0Buf in
            withUnsafeBytes(of: x) { xBuf in
                r1.withUnsafeBytes { r1Buf in
                    rVec.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_linear_combine(
                            r0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            r1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(padN))
                    }
                }
            }
        }

        let tHat = frInnerProductPad(lVec, rVec)

        // taux = tau2*x^2 + tau1*x + sum_j z^{j+2} * gamma_j
        var taux = frAdd(frMul(tau2, x2), frMul(tau1, x))
        for j in 0..<m {
            taux = frAdd(taux, frMul(zPow[j + 2], gammas[j]))
        }

        // mu = alpha + rho*x
        let mu = frAdd(alpha, frMul(rho, x))

        // Inner product argument on modified generators
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: padN)
        for i in 1..<padN {
            yInvPow[i] = frMul(yInvPow[i - 1], yInv)
        }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: padN)
        for i in 0..<padN {
            HPrime[i] = cPointScalarMul(pointFromAffine(extParams.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        transcript.appendScalar(tHat)
        transcript.appendLabel("u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        let ipProof = BulletproofsProver.innerProductProve(
            G: Array(extParams.G.prefix(padN)),
            H: HPrimeAffine,
            U: uPoint,
            a: lVec, b: rVec,
            transcript: transcript
        )

        let proof = BulletproofsAggregatedProof(
            A: A, S: S, T1: T1, T2: T2,
            taux: taux, mu: mu, tHat: tHat,
            innerProductProof: ipProof,
            m: m
        )

        return (Vs, proof)
    }

    // MARK: - Private Helpers

    private static func computeVectorCommitment(
        _ aVec: [Fr], _ bVec: [Fr], _ blind: Fr,
        _ params: BulletproofsParams
    ) -> PointProjective {
        let n = aVec.count
        let hProj = pointFromAffine(params.h)
        var result = cPointScalarMul(hProj, blind)

        for i in 0..<n {
            if !aVec[i].isZero {
                result = pointAdd(result, cPointScalarMul(pointFromAffine(params.G[i]), aVec[i]))
            }
            if !bVec[i].isZero {
                result = pointAdd(result, cPointScalarMul(pointFromAffine(params.H[i]), bVec[i]))
            }
        }

        return result
    }

    private static func frInnerProductPad(_ a: [Fr], _ b: [Fr]) -> Fr {
        let n = min(a.count, b.count)
        var result = Fr.zero
        for i in 0..<n {
            result = frAdd(result, frMul(a[i], b[i]))
        }
        return result
    }

    private static func deterministicBlinding(seed: UInt64, gammas: [Fr]) -> Fr {
        var data = [UInt8](repeating: 0, count: 8 + gammas.count * 32)
        withUnsafeBytes(of: seed) { buf in
            data.replaceSubrange(0..<8, with: buf)
        }
        for (j, g) in gammas.enumerated() {
            withUnsafeBytes(of: g) { buf in
                let offset = 8 + j * 32
                data.replaceSubrange(offset..<(offset + 32), with: buf)
            }
        }
        var hash = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]
            limbs[1] = ptr[1]
            limbs[2] = ptr[2]
            limbs[3] = ptr[3]
        }
        limbs[3] &= 0x0FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Aggregated Verifier

/// Verifier for aggregated Bulletproofs range proofs.
public class BulletproofsAggregatedVerifier {

    /// Verify an aggregated range proof for M values.
    ///
    /// - Parameters:
    ///   - Vs: the M value commitments
    ///   - proof: the aggregated range proof
    ///   - params: Bulletproofs parameters (base bit size)
    /// - Returns: true if all M values are in [0, 2^n)
    public static func verify(
        Vs: [PointProjective], proof: BulletproofsAggregatedProof,
        params: BulletproofsParams
    ) -> Bool {
        let m = proof.m
        let n = params.n
        guard Vs.count == m, m > 0, n > 0 else { return false }

        let mn = m * n
        let padN = nextPowerOf2(mn)
        guard padN > 0 else { return false }
        let extParams = BulletproofsParams.generate(n: padN)
        guard extParams.G.count >= padN, extParams.H.count >= padN else { return false }

        let gProj = pointFromAffine(extParams.g)
        let hProj = pointFromAffine(extParams.h)

        // Reconstruct challenges
        var transcript = BPTranscript()
        transcript.appendLabel("bulletproofs-aggregated")
        for j in 0..<m { transcript.appendPoint(Vs[j]) }
        transcript.appendPoint(proof.A)
        transcript.appendPoint(proof.S)
        transcript.appendLabel("y")
        let y = transcript.challenge()

        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        transcript.appendPoint(proof.T1)
        transcript.appendPoint(proof.T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        let x2 = frMul(x, x)

        // y powers
        var yPow = [Fr](repeating: Fr.one, count: padN)
        for i in 1..<padN {
            yPow[i] = frMul(yPow[i - 1], y)
        }

        // 2 powers
        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            twoPow[i] = frMul(twoPow[i - 1], two)
        }

        // z powers — need up to z^{m+2} for zPow[j+3] with j < m
        var zPow = [Fr](repeating: Fr.one, count: m + 3)
        for i in 1..<(m + 3) {
            zPow[i] = frMul(zPow[i - 1], z)
        }
        let z2 = zPow[2]

        // delta(y,z) for aggregated case
        // delta = (z - z^2) * <1^{M*n}, y^{M*n}> - sum_j z^{j+2} * <1^n, 2^n> * <1, y^{j*n+0..n-1}>
        // Simplified: delta = (z - z^2) * sum(y^i, i=0..padN-1) - sum_j z^{j+2} * sum(2^i, i=0..n-1) * ...
        // Actually the standard formula:
        // delta = (z - z^2) * sum_{i=0}^{mn-1} y^i - sum_{j=0}^{m-1} z^{j+3} * sum_{i=0}^{n-1} 2^i
        // But we need to be careful about the padded vs unpadded parts.

        var sumYPow = Fr.zero
        for i in 0..<mn { sumYPow = frAdd(sumYPow, yPow[i]) }

        var sumTwoPow = Fr.zero
        for i in 0..<n { sumTwoPow = frAdd(sumTwoPow, twoPow[i]) }

        var delta = frMul(frSub(z, zPow[2]), sumYPow)
        for j in 0..<m {
            delta = frSub(delta, frMul(zPow[j + 3], sumTwoPow))
        }

        // Check 1: tHat*g + taux*h == sum_j(z^{j+2}*V_j) + delta*g + x*T1 + x^2*T2
        let lhs1 = pointAdd(cPointScalarMul(gProj, proof.tHat), cPointScalarMul(hProj, proof.taux))
        var rhs1 = pointAdd(cPointScalarMul(gProj, delta),
                           pointAdd(cPointScalarMul(proof.T1, x), cPointScalarMul(proof.T2, x2)))
        for j in 0..<m {
            rhs1 = pointAdd(rhs1, cPointScalarMul(Vs[j], zPow[j + 2]))
        }

        guard pointEqual(lhs1, rhs1) else { return false }

        // Check 2: Inner product proof
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: padN)
        for i in 1..<padN {
            yInvPow[i] = frMul(yInvPow[i - 1], yInv)
        }

        var HPrime = [PointProjective](repeating: pointIdentity(), count: padN)
        for i in 0..<padN {
            HPrime[i] = cPointScalarMul(pointFromAffine(extParams.H[i]), yInvPow[i])
        }

        // P = A + x*S - z*sum(G) + sum(modified H terms) - mu*h
        var P = pointAdd(proof.A, cPointScalarMul(proof.S, x))

        for i in 0..<padN {
            P = pointAdd(P, cPointScalarMul(pointFromAffine(extParams.G[i]), frNeg(z)))
        }

        for i in 0..<padN {
            var coeff = frMul(z, yPow[i])
            // Add z^{j+2} * 2^{i mod n} for the appropriate j
            let j = i / n
            let iModN = i % n
            if j < m && iModN < n {
                coeff = frAdd(coeff, frMul(zPow[j + 2], twoPow[iModN]))
            }
            P = pointAdd(P, cPointScalarMul(HPrime[i], coeff))
        }

        P = pointAdd(P, cPointScalarMul(hProj, frNeg(proof.mu)))

        transcript.appendScalar(proof.tHat)
        transcript.appendLabel("u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        let PPrime = pointAdd(P, cPointScalarMul(uPoint, proof.tHat))

        let HPrimeAffine = batchToAffine(HPrime)
        return BulletproofsVerifier.innerProductVerify(
            G: Array(extParams.G.prefix(padN)),
            H: HPrimeAffine,
            U: uPoint,
            P: PPrime,
            proof: proof.innerProductProof,
            transcript: transcript
        )
    }
}

// MARK: - Utility

/// Next power of 2 >= n.
private func nextPowerOf2(_ n: Int) -> Int {
    var v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1
}

// BPTranscript is defined in BulletproofsEngine.swift and shared across the module.
