// Bulletproofs Range Proof Engine — BN254-based range proofs
//
// Implements the Bulletproofs protocol (Bunz et al. 2018) for proving
// that a committed value lies in [0, 2^n) without revealing the value.
//
// Protocol overview:
//   1. Prover commits to value v: V = v*G + gamma*H
//   2. Prover sends A, S commitments (bit decomposition + blinding)
//   3. Verifier sends challenges y, z
//   4. Prover computes polynomial t(x) = <l(x), r(x)> and commits to t1, t2
//   5. Verifier sends challenge x
//   6. Prover evaluates l(x), r(x), sends taux, mu, tHat, and inner product proof
//   7. Verifier checks the proof
//
// Uses BN254 G1 points and Fr scalars via the existing PedersenEngine infrastructure.
//
// References:
//   - Bulletproofs: Short Proofs for Confidential Transactions and More (Bunz et al. 2018)
//   - https://eprint.iacr.org/2017/1066

import Foundation
import NeonFieldOps

// MARK: - Bulletproofs Parameters

/// Parameters for the Bulletproofs range proof system.
/// Contains generator vectors G[], H[] of size n, plus pedersen generators g, h.
public struct BulletproofsParams {
    /// Generator vector G (length n, for left vector)
    public let G: [PointAffine]
    /// Generator vector H (length n, for right vector)
    public let H: [PointAffine]
    /// Pedersen base point g (for value commitment)
    public let g: PointAffine
    /// Pedersen blinding point h (for blinding)
    public let h: PointAffine
    /// Range bit size n (must be power of 2)
    public let n: Int

    /// Generate deterministic parameters for range proofs of n bits.
    /// Uses iterated hash-double from a seed point to produce independent generators.
    public static func generate(n: Int) -> BulletproofsParams {
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")

        // We need 2*n + 2 distinct generators: n for G, n for H, 1 for g, 1 for h
        let totalNeeded = 2 * n + 2
        let seed = pointFromAffine(PointAffine(x: fpFromInt(1), y: fpFromInt(2)))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(totalNeeded)
        var acc = seed
        for _ in 0..<totalNeeded {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, seed))
        }
        let affine = batchToAffine(projPoints)

        return BulletproofsParams(
            G: Array(affine[0..<n]),
            H: Array(affine[n..<(2 * n)]),
            g: affine[2 * n],
            h: affine[2 * n + 1],
            n: n
        )
    }
}

// MARK: - Inner Product Proof

/// An inner product argument proof (sub-protocol of Bulletproofs).
/// Proves <a, b> = c given commitment P = <a, G> + <b, H> + c * U.
public struct InnerProductProof {
    /// Left commitments per halving round
    public let L: [PointProjective]
    /// Right commitments per halving round
    public let R: [PointProjective]
    /// Final scalar a
    public let a: Fr
    /// Final scalar b
    public let b: Fr

    public init(L: [PointProjective], R: [PointProjective], a: Fr, b: Fr) {
        self.L = L
        self.R = R
        self.a = a
        self.b = b
    }
}

// MARK: - Bulletproofs Range Proof

/// A Bulletproofs range proof demonstrating v in [0, 2^n).
public struct BulletproofsRangeProof {
    /// Commitment to bit decomposition: A = <a_L, G> + <a_R, H> + alpha*h
    public let A: PointProjective
    /// Blinding commitment: S = <s_L, G> + <s_R, H> + rho*h
    public let S: PointProjective
    /// Commitment to t1 coefficient: T1 = t1*g + tau1*h
    public let T1: PointProjective
    /// Commitment to t2 coefficient: T2 = t2*g + tau2*h
    public let T2: PointProjective
    /// Blinding factor for tHat: taux = tau2*x^2 + tau1*x + z^2*gamma
    public let taux: Fr
    /// Blinding factor: mu = alpha + rho*x
    public let mu: Fr
    /// Evaluated inner product: tHat = t(x) = t0 + t1*x + t2*x^2
    public let tHat: Fr
    /// Inner product proof on the final l, r vectors
    public let innerProductProof: InnerProductProof

    public init(A: PointProjective, S: PointProjective,
                T1: PointProjective, T2: PointProjective,
                taux: Fr, mu: Fr, tHat: Fr,
                innerProductProof: InnerProductProof) {
        self.A = A
        self.S = S
        self.T1 = T1
        self.T2 = T2
        self.taux = taux
        self.mu = mu
        self.tHat = tHat
        self.innerProductProof = innerProductProof
    }
}

// MARK: - Fiat-Shamir Transcript (Bulletproofs-local)

/// Minimal Fiat-Shamir transcript for Bulletproofs using Blake3.
struct BPTranscript {
    private var state: [UInt8] = []

    mutating func appendPoint(_ p: PointProjective) {
        withUnsafeBytes(of: p) { buf in
            state.append(contentsOf: buf)
        }
    }

    mutating func appendScalar(_ s: Fr) {
        withUnsafeBytes(of: s) { buf in
            state.append(contentsOf: buf)
        }
    }

    mutating func appendLabel(_ label: String) {
        state.append(contentsOf: Array(label.utf8))
    }

    func challenge() -> Fr {
        var hash = [UInt8](repeating: 0, count: 32)
        state.withUnsafeBufferPointer { inp in
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
        limbs[3] &= 0x0FFFFFFFFFFFFFFF  // Ensure < r for BN254 Fr
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
    }
}

// MARK: - Bulletproofs Prover

/// Prover for Bulletproofs range proofs.
/// Proves that a committed value v lies in [0, 2^n) without revealing v.
public class BulletproofsProver {

    /// Create a range proof that v is in [0, 2^n).
    ///
    /// - Parameters:
    ///   - value: the value v to prove in range
    ///   - gamma: blinding factor for the Pedersen commitment V = v*g + gamma*h
    ///   - params: Bulletproofs parameters (generators, bit size)
    /// - Returns: (V, proof) where V is the value commitment and proof is the range proof
    public static func prove(value: UInt64, gamma: Fr,
                             params: BulletproofsParams) -> (V: PointProjective, proof: BulletproofsRangeProof) {
        let n = params.n
        precondition(n >= 8 && n <= 64 && (n & (n - 1)) == 0, "n must be power of 2 in [8, 64]")
        precondition(n == 64 || value < (1 << n), "value out of range [0, 2^\(n))")

        let gProj = pointFromAffine(params.g)
        let hProj = pointFromAffine(params.h)

        // Step 1: Commit to value: V = v*g + gamma*h
        let vFr = frFromInt(value)
        let V = pointAdd(cPointScalarMul(gProj, vFr), cPointScalarMul(hProj, gamma))

        // Step 2: Bit decomposition
        // a_L[i] = (v >> i) & 1, a_R[i] = a_L[i] - 1
        var aL = [Fr](repeating: Fr.zero, count: n)
        var aR = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            if (value >> i) & 1 == 1 {
                aL[i] = Fr.one
                // aR[i] = 0 (already set)
            } else {
                // aL[i] = 0 (already set)
                aR[i] = frNeg(Fr.one)  // -1
            }
        }

        // Generate blinding scalars deterministically from value + gamma (for testability)
        // In production, these would be random.
        let alpha = deterministicBlinding(seed: 1, gamma: gamma)
        let rho = deterministicBlinding(seed: 2, gamma: gamma)
        let tau1 = deterministicBlinding(seed: 3, gamma: gamma)
        let tau2 = deterministicBlinding(seed: 4, gamma: gamma)

        // Random blinding vectors s_L, s_R
        var sL = [Fr](repeating: Fr.zero, count: n)
        var sR = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            sL[i] = deterministicBlinding(seed: UInt64(100 + i), gamma: gamma)
            sR[i] = deterministicBlinding(seed: UInt64(200 + i), gamma: gamma)
        }

        // Step 3: Compute A = <a_L, G> + <a_R, H> + alpha*h
        let A = computeVectorCommitment(aL, aR, alpha, params)

        // Step 4: Compute S = <s_L, G> + <s_R, H> + rho*h
        let S = computeVectorCommitment(sL, sR, rho, params)

        // Fiat-Shamir: get challenges y, z
        var transcript = BPTranscript()
        transcript.appendLabel("bulletproofs-range")
        transcript.appendPoint(V)
        transcript.appendPoint(A)
        transcript.appendPoint(S)
        transcript.appendLabel("y")
        let y = transcript.challenge()

        transcript.appendScalar(y)
        transcript.appendLabel("z")
        let z = transcript.challenge()

        // Step 5: Compute t(x) = <l(x), r(x)>
        // l(x) = (a_L - z*1) + s_L*x
        // r(x) = y^n . (a_R + z*1 + s_R*x) + z^2 * 2^n
        // t(x) = t0 + t1*x + t2*x^2

        // Precompute powers of y: y^0, y^1, ..., y^(n-1)
        var yPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            yPow[i] = frMul(yPow[i - 1], y)
        }

        // Precompute powers of 2: 2^0, 2^1, ..., 2^(n-1)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        let two = frFromInt(2)
        for i in 1..<n {
            twoPow[i] = frMul(twoPow[i - 1], two)
        }

        let z2 = frMul(z, z)

        // l0 = a_L - z*1, l1 = s_L
        var l0 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            l0[i] = frSub(aL[i], z)
        }

        // r0 = y^n . (a_R + z*1) + z^2 * 2^n
        // r1 = y^n . s_R
        var r0 = [Fr](repeating: Fr.zero, count: n)
        var r1 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let aRpZ = frAdd(aR[i], z)
            r0[i] = frAdd(frMul(yPow[i], aRpZ), frMul(z2, twoPow[i]))
            r1[i] = frMul(yPow[i], sR[i])
        }

        // t0 = <l0, r0>
        // t1 = <l0, r1> + <sL, r0>
        // t2 = <sL, r1>
        let t0 = frInnerProduct(l0, r0)
        let t1 = frAdd(frInnerProduct(l0, r1), frInnerProduct(sL, r0))
        let t2 = frInnerProduct(sL, r1)

        // Step 6: Commit to t1, t2
        let T1 = pointAdd(cPointScalarMul(gProj, t1), cPointScalarMul(hProj, tau1))
        let T2 = pointAdd(cPointScalarMul(gProj, t2), cPointScalarMul(hProj, tau2))

        // Fiat-Shamir: get challenge x
        transcript.appendPoint(T1)
        transcript.appendPoint(T2)
        transcript.appendLabel("x")
        let x = transcript.challenge()

        // Step 7: Evaluate at x
        // l = l0 + l1*x = (a_L - z*1) + s_L*x
        // r = r0 + r1*x = y^n . (a_R + z*1 + s_R*x) + z^2 * 2^n
        var lVec = [Fr](repeating: Fr.zero, count: n)
        var rVec = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            lVec[i] = frAdd(l0[i], frMul(sL[i], x))
            rVec[i] = frAdd(r0[i], frMul(r1[i], x))
        }

        let tHat = frInnerProduct(lVec, rVec)

        // taux = tau2*x^2 + tau1*x + z^2*gamma
        let x2 = frMul(x, x)
        let taux = frAdd(frAdd(frMul(tau2, x2), frMul(tau1, x)), frMul(z2, gamma))

        // mu = alpha + rho*x
        let mu = frAdd(alpha, frMul(rho, x))

        // Step 8: Inner product argument
        // We need to prove <l, r> = tHat with respect to modified generators
        // H' = H[i] * y^(-i) for the inner product argument

        // Compute y inverse powers: y^0, y^(-1), ..., y^(-(n-1))
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            yInvPow[i] = frMul(yInvPow[i - 1], yInv)
        }

        // Modified H generators: H'[i] = y^(-i) * H[i]
        var HPrime = [PointProjective](repeating: pointIdentity(), count: n)
        for i in 0..<n {
            HPrime[i] = cPointScalarMul(pointFromAffine(params.H[i]), yInvPow[i])
        }
        let HPrimeAffine = batchToAffine(HPrime)

        // Get inner product proof U point from transcript
        transcript.appendScalar(tHat)
        transcript.appendLabel("u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        // Inner product proof on (G, H', U, l, r)
        let ipProof = innerProductProve(
            G: params.G, H: HPrimeAffine, U: uPoint,
            a: lVec, b: rVec, transcript: transcript
        )

        let proof = BulletproofsRangeProof(
            A: A, S: S, T1: T1, T2: T2,
            taux: taux, mu: mu, tHat: tHat,
            innerProductProof: ipProof
        )

        return (V, proof)
    }

    // MARK: - Inner Product Argument Prover

    /// Prove an inner product argument: <a, b> = c
    /// with P = <a, G> + <b, H> + c * U
    static func innerProductProve(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        a inputA: [Fr], b inputB: [Fr],
        transcript: BPTranscript
    ) -> InnerProductProof {
        var n = inputA.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Vector length must be power of 2")

        var a = inputA
        var b = inputB
        var gPts = G.map { pointFromAffine($0) }
        var hPts = H.map { pointFromAffine($0) }
        var Ls = [PointProjective]()
        var Rs = [PointProjective]()
        var trans = transcript

        while n > 1 {
            let halfN = n / 2

            // Compute cross inner products
            let cL = frInnerProduct(Array(a[0..<halfN]), Array(b[halfN..<n]))
            let cR = frInnerProduct(Array(a[halfN..<n]), Array(b[0..<halfN]))

            // L = <a_lo, G_hi> + <b_hi, H_lo> + cL * U
            var L = cPointScalarMul(U, cL)
            for i in 0..<halfN {
                L = pointAdd(L, cPointScalarMul(gPts[halfN + i], a[i]))
                L = pointAdd(L, cPointScalarMul(hPts[i], b[halfN + i]))
            }

            // R = <a_hi, G_lo> + <b_lo, H_hi> + cR * U
            var R = cPointScalarMul(U, cR)
            for i in 0..<halfN {
                R = pointAdd(R, cPointScalarMul(gPts[i], a[halfN + i]))
                R = pointAdd(R, cPointScalarMul(hPts[halfN + i], b[i]))
            }

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            trans.appendPoint(L)
            trans.appendPoint(R)
            trans.appendLabel("ip_challenge")
            let u = trans.challenge()
            let uInv = frInverse(u)

            // Fold vectors
            var newA = [Fr](repeating: Fr.zero, count: halfN)
            var newB = [Fr](repeating: Fr.zero, count: halfN)
            var newG = [PointProjective](repeating: pointIdentity(), count: halfN)
            var newH = [PointProjective](repeating: pointIdentity(), count: halfN)

            for i in 0..<halfN {
                newA[i] = frAdd(frMul(u, a[i]), frMul(uInv, a[halfN + i]))
                newB[i] = frAdd(frMul(uInv, b[i]), frMul(u, b[halfN + i]))
                newG[i] = pointAdd(cPointScalarMul(gPts[i], uInv), cPointScalarMul(gPts[halfN + i], u))
                newH[i] = pointAdd(cPointScalarMul(hPts[i], u), cPointScalarMul(hPts[halfN + i], uInv))
            }

            a = newA
            b = newB
            gPts = newG
            hPts = newH
            n = halfN
        }

        return InnerProductProof(L: Ls, R: Rs, a: a[0], b: b[0])
    }

    // MARK: - Private Helpers

    /// Compute <a, G> + <b, H> + blind*h
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

    /// Fr inner product: <a, b> = sum(a[i] * b[i])
    private static func frInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        precondition(a.count == b.count)
        var result = Fr.zero
        for i in 0..<a.count {
            result = frAdd(result, frMul(a[i], b[i]))
        }
        return result
    }

    /// Deterministic blinding from seed + gamma (for reproducible tests).
    /// In production, use cryptographic randomness.
    private static func deterministicBlinding(seed: UInt64, gamma: Fr) -> Fr {
        var data = [UInt8](repeating: 0, count: 40)
        withUnsafeBytes(of: seed) { buf in
            data.replaceSubrange(0..<8, with: buf)
        }
        withUnsafeBytes(of: gamma) { buf in
            data.replaceSubrange(8..<40, with: buf)
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

// MARK: - Bulletproofs Verifier

/// Verifier for Bulletproofs range proofs.
public class BulletproofsVerifier {

    /// Verify a Bulletproofs range proof.
    ///
    /// - Parameters:
    ///   - V: the Pedersen commitment to the value (V = v*g + gamma*h)
    ///   - proof: the range proof
    ///   - params: Bulletproofs parameters
    /// - Returns: true if the proof is valid (v is in [0, 2^n))
    public static func verify(V: PointProjective, proof: BulletproofsRangeProof,
                              params: BulletproofsParams) -> Bool {
        let n = params.n
        let gProj = pointFromAffine(params.g)
        let hProj = pointFromAffine(params.h)

        // Reconstruct challenges via Fiat-Shamir
        var transcript = BPTranscript()
        transcript.appendLabel("bulletproofs-range")
        transcript.appendPoint(V)
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
        let z2 = frMul(z, z)

        // Precompute y powers and 2 powers
        var yPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            yPow[i] = frMul(yPow[i - 1], y)
        }
        let two = frFromInt(2)
        var twoPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            twoPow[i] = frMul(twoPow[i - 1], two)
        }

        // delta(y,z) = (z - z^2) * <1^n, y^n> - z^3 * <1^n, 2^n>
        var sumYPow = Fr.zero
        for i in 0..<n { sumYPow = frAdd(sumYPow, yPow[i]) }
        var sumTwoPow = Fr.zero
        for i in 0..<n { sumTwoPow = frAdd(sumTwoPow, twoPow[i]) }

        let z3 = frMul(z2, z)
        let delta = frSub(frMul(frSub(z, z2), sumYPow), frMul(z3, sumTwoPow))

        // Check 1: tHat*g + taux*h == z^2*V + delta*g + x*T1 + x^2*T2
        let lhs1 = pointAdd(cPointScalarMul(gProj, proof.tHat), cPointScalarMul(hProj, proof.taux))
        let rhs1 = pointAdd(
            pointAdd(cPointScalarMul(V, z2), cPointScalarMul(gProj, delta)),
            pointAdd(cPointScalarMul(proof.T1, x), cPointScalarMul(proof.T2, x2))
        )

        guard pointEqual(lhs1, rhs1) else { return false }

        // Check 2: Inner product argument
        // Reconstruct P from A, S, and the challenge-modified generators
        // P = A + x*S - z*<1, G> + (z*y^n + z^2*2^n) . H' - mu*h

        // Compute y inverse powers for H'
        let yInv = frInverse(y)
        var yInvPow = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            yInvPow[i] = frMul(yInvPow[i - 1], yInv)
        }

        // H'[i] = y^(-i) * H[i]
        var HPrime = [PointProjective](repeating: pointIdentity(), count: n)
        for i in 0..<n {
            HPrime[i] = cPointScalarMul(pointFromAffine(params.H[i]), yInvPow[i])
        }

        // P = A + x*S
        var P = pointAdd(proof.A, cPointScalarMul(proof.S, x))

        // - z * sum(G[i])
        for i in 0..<n {
            P = pointAdd(P, cPointScalarMul(pointFromAffine(params.G[i]), frNeg(z)))
        }

        // + sum((z*y^i + z^2*2^i) * H'[i])
        for i in 0..<n {
            let coeff = frAdd(frMul(z, yPow[i]), frMul(z2, twoPow[i]))
            P = pointAdd(P, cPointScalarMul(HPrime[i], coeff))
        }

        // - mu * h
        P = pointAdd(P, cPointScalarMul(hProj, frNeg(proof.mu)))

        // U point for inner product (same derivation as prover)
        transcript.appendScalar(proof.tHat)
        transcript.appendLabel("u")
        let uChallenge = transcript.challenge()
        let uPoint = cPointScalarMul(gProj, uChallenge)

        // P_prime = P + tHat * U  (binding the inner product value)
        let PPrime = pointAdd(P, cPointScalarMul(uPoint, proof.tHat))

        // Verify inner product proof
        let HPrimeAffine = batchToAffine(HPrime)
        return innerProductVerify(
            G: params.G, H: HPrimeAffine, U: uPoint,
            P: PPrime, proof: proof.innerProductProof,
            transcript: transcript
        )
    }

    // MARK: - Inner Product Argument Verifier

    /// Verify an inner product proof.
    /// Checks that the proof correctly demonstrates <a, b> with respect to
    /// the commitment P = <a, G> + <b, H> + <a,b> * U.
    static func innerProductVerify(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        P: PointProjective, proof: InnerProductProof,
        transcript: BPTranscript
    ) -> Bool {
        var n = G.count
        let logN = proof.L.count
        guard proof.R.count == logN else { return false }
        guard (1 << logN) == n else { return false }

        // Reconstruct challenges
        var trans = transcript
        var challenges = [Fr]()
        challenges.reserveCapacity(logN)

        for round in 0..<logN {
            trans.appendPoint(proof.L[round])
            trans.appendPoint(proof.R[round])
            trans.appendLabel("ip_challenge")
            let u = trans.challenge()
            challenges.append(u)
        }

        // Precompute challenge inverses
        let challengeInvs = frBatchInverse(challenges)

        // Compute s-vector: s[i] = product of challenges based on bit decomposition
        var s = [Fr](repeating: Fr.one, count: n)
        for round in 0..<logN {
            let u = challenges[round]
            let uInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[i] = frMul(s[i], uInv)
                } else {
                    s[i] = frMul(s[i], u)
                }
            }
        }

        // G_final = sum(s[i] * G[i])
        var gFinal = pointIdentity()
        for i in 0..<n {
            gFinal = pointAdd(gFinal, cPointScalarMul(pointFromAffine(G[i]), s[i]))
        }

        // H_final = sum(s[i]^(-1) * H[i])
        let sInvs = frBatchInverse(s)
        var hFinal = pointIdentity()
        for i in 0..<n {
            hFinal = pointAdd(hFinal, cPointScalarMul(pointFromAffine(H[i]), sInvs[i]))
        }

        // Expected: P_expected = proof.a * G_final + proof.b * H_final + (proof.a * proof.b) * U
        let ab = frMul(proof.a, proof.b)
        let expected = pointAdd(
            pointAdd(cPointScalarMul(gFinal, proof.a), cPointScalarMul(hFinal, proof.b)),
            cPointScalarMul(U, ab)
        )

        // Fold P: P_folded = P + sum(u_i^2 * L_i + u_i^(-2) * R_i)
        var PFolded = P
        for round in 0..<logN {
            let u2 = frMul(challenges[round], challenges[round])
            let uInv2 = frMul(challengeInvs[round], challengeInvs[round])
            PFolded = pointAdd(PFolded, pointAdd(
                cPointScalarMul(proof.L[round], u2),
                cPointScalarMul(proof.R[round], uInv2)
            ))
        }

        return pointEqual(PFolded, expected)
    }
}

// MARK: - Standalone Inner Product Argument

/// Standalone inner product argument for testing the sub-protocol.
public class InnerProductArgument {

    /// Create an inner product proof for <a, b> = c
    /// with commitment P = <a, G> + <b, H> + c * U.
    public static func prove(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        a: [Fr], b: [Fr]
    ) -> InnerProductProof {
        return BulletproofsProver.innerProductProve(
            G: G, H: H, U: U, a: a, b: b,
            transcript: BPTranscript()
        )
    }

    /// Verify an inner product proof.
    public static func verify(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        P: PointProjective, proof: InnerProductProof
    ) -> Bool {
        return BulletproofsVerifier.innerProductVerify(
            G: G, H: H, U: U, P: P, proof: proof,
            transcript: BPTranscript()
        )
    }

    /// Compute the commitment P = <a, G> + <b, H> + <a,b> * U
    public static func computeCommitment(
        G: [PointAffine], H: [PointAffine], U: PointProjective,
        a: [Fr], b: [Fr]
    ) -> PointProjective {
        let n = a.count
        var P = pointIdentity()
        for i in 0..<n {
            P = pointAdd(P, cPointScalarMul(pointFromAffine(G[i]), a[i]))
            P = pointAdd(P, cPointScalarMul(pointFromAffine(H[i]), b[i]))
        }
        var ip = Fr.zero
        for i in 0..<n { ip = frAdd(ip, frMul(a[i], b[i])) }
        P = pointAdd(P, cPointScalarMul(U, ip))
        return P
    }
}
