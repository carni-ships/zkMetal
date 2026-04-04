// Univariate Sumcheck Engine — Aurora/Fractal/Marlin style
//
// Claim: sum_{x in H} f(x) = v, where H is a multiplicative subgroup of size n = 2^k.
//
// Protocol (for arbitrary degree f, following Aurora/Marlin):
//   1. Decompose: f(X) - v/n = q(X) * Z_H(X) + X * p(X)
//      where Z_H(X) = X^n - 1, q is the quotient, and X*p(X) is the remainder
//      with constant term removed (since it's zero when the claim is correct).
//   2. Prover commits to q(X) and p(X) via KZG
//   3. Verifier picks random challenge r, checks:
//      f(r) - v/n = q(r) * Z_H(r) + r * p(r)
//   4. Verifier checks KZG opening proofs for f(r), q(r), p(r)
//
// Key advantage: single round instead of log(n) rounds, using NTT for poly division.
// Also supports batch mode: k polynomials reduced to 1 via random linear combination.

import Foundation

// MARK: - Proof structures

public struct UnivariateSumcheckProof {
    public let qCommitment: PointProjective    // KZG commitment to quotient q(X)
    public let pCommitment: PointProjective    // KZG commitment to remainder p(X) where rem = X*p(X)
    public let fEval: Fr                       // f(r) at random challenge
    public let qEval: Fr                       // q(r) at random challenge
    public let pEval: Fr                       // p(r) at random challenge
    public let fOpeningProof: PointProjective  // KZG witness for f(r)
    public let qOpeningProof: PointProjective  // KZG witness for q(r)
    public let pOpeningProof: PointProjective  // KZG witness for p(r)

    public init(qCommitment: PointProjective, pCommitment: PointProjective,
                fEval: Fr, qEval: Fr, pEval: Fr,
                fOpeningProof: PointProjective, qOpeningProof: PointProjective,
                pOpeningProof: PointProjective) {
        self.qCommitment = qCommitment
        self.pCommitment = pCommitment
        self.fEval = fEval
        self.qEval = qEval
        self.pEval = pEval
        self.fOpeningProof = fOpeningProof
        self.qOpeningProof = qOpeningProof
        self.pOpeningProof = pOpeningProof
    }
}

public struct BatchUnivariateSumcheckProof {
    public let gCommitment: PointProjective      // commitment to combined g = sum alpha^i * f_i
    public let innerProof: UnivariateSumcheckProof  // univariate sumcheck proof for g
    public let alphas: [Fr]                        // random combination coefficients (for verification)
}

// MARK: - Engine

public class UnivariateSumcheckEngine {
    public static let version = Versions.univariateSumcheck
    public let kzg: KZGEngine
    public let poly: PolyEngine

    /// Convenience accessors
    var ntt: NTTEngine { poly.nttEngine }

    public init(kzg: KZGEngine) throws {
        self.kzg = kzg
        self.poly = kzg.polyEngine
    }

    // MARK: - Single polynomial prove/verify

    /// Prove that sum_{x in H} f(x) = claimedSum, where H is the multiplicative subgroup of size 2^logN.
    /// f is given in coefficient form. Works for any degree (not limited to deg < n).
    ///
    /// Aurora/Marlin decomposition:
    ///   g(X) = f(X) - v/n
    ///   g(X) = q(X) * Z_H(X) + rem(X)     where deg(rem) < n
    ///   rem(X) has rem[0] = 0 when claim is correct, so rem(X) = X * p(X)
    ///   Verifier checks: f(r) - v/n = q(r) * Z_H(r) + r * p(r)
    public func prove(fCoeffs: [Fr], logN: Int, claimedSum: Fr,
                      transcript: Transcript) throws -> UnivariateSumcheckProof {
        let n = 1 << logN

        // 1. Compute v/n
        let nInv = frInverse(frFromInt(UInt64(n)))
        let vOverN = frMul(claimedSum, nInv)

        // 2. g(X) = f(X) - v/n
        var gCoeffs = fCoeffs
        // Pad to at least n+1 so division always works
        while gCoeffs.count < n + 1 {
            gCoeffs.append(Fr.zero)
        }
        gCoeffs[0] = frSub(gCoeffs[0], vOverN)

        // 3. Polynomial division with remainder: g = q * Z_H + rem
        let (qCoeffs, remCoeffs) = divideByVanishingWithRemainder(gCoeffs, vanishingDegree: n)

        // 4. Extract p from rem: rem(X) = X*p(X), so p[i] = rem[i+1]
        //    rem[0] should be 0 if the claim is valid (prover assumes this)
        var pCoeffs: [Fr]
        if remCoeffs.count > 1 {
            pCoeffs = Array(remCoeffs[1...])
        } else {
            pCoeffs = [Fr.zero]
        }
        // Ensure p is non-empty
        if pCoeffs.isEmpty { pCoeffs = [Fr.zero] }

        // 5. Commit f, q, p via KZG
        let fCommitment = try kzg.commit(fCoeffs)
        let qCommitment = try kzg.commit(qCoeffs.isEmpty ? [Fr.zero] : qCoeffs)
        let pCommitment = try kzg.commit(pCoeffs)

        // 6. Fiat-Shamir: absorb commitments, squeeze challenge r
        absorbPoint(transcript, fCommitment)
        absorbPoint(transcript, qCommitment)
        absorbPoint(transcript, pCommitment)
        transcript.absorbLabel("univariate-sumcheck-challenge")
        let r = transcript.squeeze()

        // 7. Open f(r), q(r), p(r)
        let fProof = try kzg.open(fCoeffs, at: r)
        let qProof = try kzg.open(qCoeffs.isEmpty ? [Fr.zero] : qCoeffs, at: r)
        let pProof = try kzg.open(pCoeffs, at: r)

        return UnivariateSumcheckProof(
            qCommitment: qCommitment,
            pCommitment: pCommitment,
            fEval: fProof.evaluation,
            qEval: qProof.evaluation,
            pEval: pProof.evaluation,
            fOpeningProof: fProof.witness,
            qOpeningProof: qProof.witness,
            pOpeningProof: pProof.witness
        )
    }

    /// Verify a univariate sumcheck proof.
    ///
    /// Checks:
    ///   1. f(r) - v/n = q(r) * Z_H(r) + r * p(r)
    ///   2. KZG opening proofs for f(r), q(r), p(r) are valid
    public func verify(proof: UnivariateSumcheckProof, fCommitment: PointProjective,
                       claimedSum: Fr, logN: Int, transcript: Transcript,
                       srsSecret: Fr) -> Bool {
        let n = 1 << logN

        // Reconstruct Fiat-Shamir challenge
        absorbPoint(transcript, fCommitment)
        absorbPoint(transcript, proof.qCommitment)
        absorbPoint(transcript, proof.pCommitment)
        transcript.absorbLabel("univariate-sumcheck-challenge")
        let r = transcript.squeeze()

        // 1. Check sumcheck equation: f(r) - v/n = q(r) * Z_H(r) + r * p(r)
        let nInv = frInverse(frFromInt(UInt64(n)))
        let vOverN = frMul(claimedSum, nInv)

        // Z_H(r) = r^n - 1
        let rN = frPow(r, UInt64(n))
        let zHr = frSub(rN, Fr.one)

        // LHS = f(r) - v/n
        let lhs = frSub(proof.fEval, vOverN)
        // RHS = q(r) * Z_H(r) + r * p(r)
        let rhs = frAdd(frMul(proof.qEval, zHr), frMul(r, proof.pEval))

        let lhsLimbs = frToInt(lhs)
        let rhsLimbs = frToInt(rhs)
        guard lhsLimbs == rhsLimbs else {
            return false
        }

        // 2. Verify KZG openings using SRS secret
        let g1 = pointFromAffine(kzg.srs[0])
        let sMr = frSub(srsSecret, r)

        // Verify f opening: C_f == [f(r)]*G + [s-r]*pi_f
        guard verifyKZGOpening(commitment: fCommitment, eval: proof.fEval,
                               witness: proof.fOpeningProof, g1: g1, sMr: sMr) else {
            return false
        }

        // Verify q opening
        guard verifyKZGOpening(commitment: proof.qCommitment, eval: proof.qEval,
                               witness: proof.qOpeningProof, g1: g1, sMr: sMr) else {
            return false
        }

        // Verify p opening
        guard verifyKZGOpening(commitment: proof.pCommitment, eval: proof.pEval,
                               witness: proof.pOpeningProof, g1: g1, sMr: sMr) else {
            return false
        }

        return true
    }

    // MARK: - Batch univariate sumcheck

    /// Batch prove: given k polynomials f_1,...,f_k with claims v_1,...,v_k,
    /// reduce to a single univariate sumcheck via random linear combination.
    ///
    /// g = sum_i alpha^i * f_i,  w = sum_i alpha^i * v_i
    /// Then prove sum_{x in H} g(x) = w.
    public func batchProve(polynomials: [[Fr]], claims: [Fr], logN: Int,
                           transcript: Transcript) throws -> BatchUnivariateSumcheckProof {
        precondition(polynomials.count == claims.count)
        precondition(!polynomials.isEmpty)
        let k = polynomials.count

        // Get batching challenge alpha from transcript
        transcript.absorbLabel("batch-univariate-sumcheck")
        for i in 0..<k {
            transcript.absorb(claims[i])
        }
        let alpha = transcript.squeeze()

        // Compute g = sum_i alpha^i * f_i (combined polynomial)
        let maxDeg = polynomials.map { $0.count }.max()!
        var gCoeffs = [Fr](repeating: Fr.zero, count: maxDeg)
        var alphaPow = Fr.one
        var alphas = [Fr]()
        alphas.reserveCapacity(k)

        for i in 0..<k {
            alphas.append(alphaPow)
            let fi = polynomials[i]
            for j in 0..<fi.count {
                gCoeffs[j] = frAdd(gCoeffs[j], frMul(alphaPow, fi[j]))
            }
            if i < k - 1 {
                alphaPow = frMul(alphaPow, alpha)
            }
        }

        // Compute w = sum_i alpha^i * v_i (combined claim)
        var w = Fr.zero
        alphaPow = Fr.one
        for i in 0..<k {
            w = frAdd(w, frMul(alphaPow, claims[i]))
            if i < k - 1 {
                alphaPow = frMul(alphaPow, alpha)
            }
        }

        // Commit to g
        let gCommitment = try kzg.commit(gCoeffs)

        // Single univariate sumcheck on (g, w)
        let innerProof = try prove(fCoeffs: gCoeffs, logN: logN, claimedSum: w,
                                   transcript: transcript)

        return BatchUnivariateSumcheckProof(
            gCommitment: gCommitment,
            innerProof: innerProof,
            alphas: alphas
        )
    }

    /// Batch verify: reconstruct the combined claim, then verify the inner proof.
    public func batchVerify(proof: BatchUnivariateSumcheckProof,
                            claims: [Fr], logN: Int,
                            transcript: Transcript, srsSecret: Fr) -> Bool {
        let k = claims.count
        guard k == proof.alphas.count else { return false }

        // Reconstruct batching challenge
        transcript.absorbLabel("batch-univariate-sumcheck")
        for i in 0..<k {
            transcript.absorb(claims[i])
        }
        let _ = transcript.squeeze()  // consume alpha

        // Reconstruct combined claim w = sum_i alpha^i * v_i
        var w = Fr.zero
        for i in 0..<k {
            w = frAdd(w, frMul(proof.alphas[i], claims[i]))
        }

        // Verify the inner univariate sumcheck proof
        return verify(proof: proof.innerProof, fCommitment: proof.gCommitment,
                      claimedSum: w, logN: logN, transcript: transcript,
                      srsSecret: srsSecret)
    }

    // MARK: - Helpers

    /// Verify a single KZG opening: C == [eval]*G + [s-r]*witness
    private func verifyKZGOpening(commitment: PointProjective, eval: Fr,
                                  witness: PointProjective,
                                  g1: PointProjective, sMr: Fr) -> Bool {
        let yG = cPointScalarMul(g1, eval)
        let szP = cPointScalarMul(witness, sMr)
        let expected = pointAdd(yG, szP)
        let cA = batchToAffine([commitment])
        let eA = batchToAffine([expected])
        return fpToInt(cA[0].x) == fpToInt(eA[0].x) &&
               fpToInt(cA[0].y) == fpToInt(eA[0].y)
    }

    /// Divide polynomial g by Z_H(X) = X^n - 1 with remainder.
    /// Returns (quotient, remainder) such that g = quotient * Z_H + remainder, deg(remainder) < n.
    ///
    /// Synthetic division by X^n - 1:
    /// Process from highest degree down. For each coefficient g[i] where i >= n,
    /// the coefficient contributes to quotient[i-n] and adds back to g[i-n] (since Z_H = X^n - 1).
    func divideByVanishingWithRemainder(_ g: [Fr], vanishingDegree n: Int) -> (quotient: [Fr], remainder: [Fr]) {
        let d = g.count
        guard d > n else {
            // g has degree < n, quotient is 0, remainder is g
            return ([], g)
        }

        // Work on a mutable copy
        var work = g

        let qLen = d - n
        var q = [Fr](repeating: Fr.zero, count: qLen)

        // Process from highest degree coefficient down to degree n
        for i in stride(from: d - 1, through: n, by: -1) {
            let coeff = work[i]
            // This coefficient belongs to quotient[i - n]
            q[i - n] = coeff
            // Subtract coeff * Z_H contribution: Z_H = X^n - 1, so
            // coeff * X^{i-n} * (X^n - 1) contributes coeff at position i and -coeff at position i-n
            // We already "consumed" position i; add coeff back to position i-n (since -(−1) = +1)
            work[i - n] = frAdd(work[i - n], coeff)
            work[i] = Fr.zero
        }

        // Remainder is work[0..n-1]
        let rem = Array(work.prefix(n))
        return (q, rem)
    }

    /// Absorb a projective point into the transcript by converting to affine and absorbing x, y.
    func absorbPoint(_ transcript: Transcript, _ point: PointProjective) {
        let affine = batchToAffine([point])
        let xLimbs = fpToInt(affine[0].x)
        let yLimbs = fpToInt(affine[0].y)
        var xBytes = [UInt8]()
        var yBytes = [UInt8]()
        for limb in xLimbs {
            var l = limb
            xBytes.append(contentsOf: withUnsafeBytes(of: &l) { Array($0) })
        }
        for limb in yLimbs {
            var l = limb
            yBytes.append(contentsOf: withUnsafeBytes(of: &l) { Array($0) })
        }
        transcript.absorbBytes(xBytes)
        transcript.absorbBytes(yBytes)
    }

    // MARK: - NTT-based division (for large polynomials)

    /// Divide polynomial f by Z_H(X) = X^n - 1 using NTT.
    /// Returns only the quotient (for use when exact division is known).
    func divideByVanishingNTT(_ fCoeffs: [Fr], vanishingDegree n: Int) throws -> [Fr] {
        let d = fCoeffs.count
        guard d > n else { return [] }

        var m = 1
        while m < d + n { m <<= 1 }

        var fPad = fCoeffs
        if fPad.count < m {
            fPad.append(contentsOf: [Fr](repeating: Fr.zero, count: m - fPad.count))
        }

        var zH = [Fr](repeating: Fr.zero, count: m)
        zH[0] = frSub(Fr.zero, Fr.one)  // -1
        zH[n] = Fr.one

        let fEvals = try ntt.ntt(fPad)
        let zHEvals = try ntt.ntt(zH)

        let zHInv = try poly.batchInverse(zHEvals)
        let hEvals = try poly.hadamard(fEvals, zHInv)

        let hCoeffs = try ntt.intt(hEvals)

        let hLen = d - n
        return Array(hCoeffs.prefix(hLen))
    }
}
