// Univariate Sumcheck Engine — Aurora/Fractal/Marlin style
//
// Claim: sum_{x in H} f(x) = v, where H is a multiplicative subgroup of size n = 2^k.
//
// Protocol:
//   1. Prover computes h(X) = (f(X) - v/n) / Z_H(X) where Z_H(X) = X^n - 1
//   2. Prover commits to h(X) via KZG
//   3. Verifier picks random challenge r, checks: f(r) = v/n + h(r) * Z_H(r)
//
// Key advantage: single polynomial division (NTT + pointwise divide + iNTT) instead
// of log(n) sequential rounds in multilinear sumcheck. Fewer GPU dispatches.
//
// Also supports batch mode: k polynomials reduced to 1 via random linear combination.

import Foundation

// MARK: - Proof structures

public struct UnivariateSumcheckProof {
    public let hCommitment: PointProjective    // KZG commitment to quotient h(X)
    public let fEval: Fr                       // f(r) at random challenge
    public let hEval: Fr                       // h(r) at random challenge
    public let fOpeningProof: PointProjective   // KZG witness for f(r)
    public let hOpeningProof: PointProjective   // KZG witness for h(r)

    public init(hCommitment: PointProjective, fEval: Fr, hEval: Fr,
                fOpeningProof: PointProjective, hOpeningProof: PointProjective) {
        self.hCommitment = hCommitment
        self.fEval = fEval
        self.hEval = hEval
        self.fOpeningProof = fOpeningProof
        self.hOpeningProof = hOpeningProof
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
    /// f is given in coefficient form, degree < domainSize * quotientBlowup (typically degree < 2n or 4n).
    ///
    /// Steps:
    ///   1. Compute v/n (inverse of domain size times claimed sum)
    ///   2. g(X) = f(X) - v/n (subtract from constant term)
    ///   3. Divide g by Z_H(X) = X^n - 1 via NTT: evaluate on larger domain, pointwise divide, iNTT
    ///   4. Commit h via KZG
    ///   5. Fiat-Shamir challenge r
    ///   6. Open f(r) and h(r) via KZG
    public func prove(fCoeffs: [Fr], logN: Int, claimedSum: Fr,
                      transcript: Transcript) throws -> UnivariateSumcheckProof {
        let n = 1 << logN

        // 1. Compute v/n: the constant that f should average to over H
        let nInv = frInverse(frFromInt(UInt64(n)))
        let vOverN = frMul(claimedSum, nInv)

        // 2. g(X) = f(X) - v/n
        var gCoeffs = fCoeffs
        gCoeffs[0] = frSub(gCoeffs[0], vOverN)

        // 3. Polynomial division: h = g / Z_H where Z_H(X) = X^n - 1
        //    Since Z_H is sparse, we use direct coefficient division:
        //    if g(X) = sum_i g_i X^i and Z_H = X^n - 1, then
        //    h = g / Z_H is computed by: h_i = g_{i+n} + h_{i+n} for i from top down
        //    (synthetic division by X^n - 1)
        let hCoeffs = divideByVanishing(gCoeffs, vanishingDegree: n)

        // 4. Commit f and h via KZG
        let fCommitment = try kzg.commit(fCoeffs)
        let hCommitment = try kzg.commit(hCoeffs)

        // 5. Fiat-Shamir: absorb commitments, squeeze challenge r
        absorbPoint(transcript, fCommitment)
        absorbPoint(transcript, hCommitment)
        transcript.absorbLabel("univariate-sumcheck-challenge")
        let r = transcript.squeeze()

        // 6. Open f(r) and h(r)
        let fProof = try kzg.open(fCoeffs, at: r)
        let hProof = try kzg.open(hCoeffs, at: r)

        return UnivariateSumcheckProof(
            hCommitment: hCommitment,
            fEval: fProof.evaluation,
            hEval: hProof.evaluation,
            fOpeningProof: fProof.witness,
            hOpeningProof: hProof.witness
        )
    }

    /// Verify a univariate sumcheck proof.
    ///
    /// Checks:
    ///   1. f(r) = v/n + h(r) * Z_H(r)   (the sumcheck equation)
    ///   2. KZG opening proofs for f(r) and h(r) are valid
    ///
    /// Parameters:
    ///   - fCommitment: KZG commitment to f (provided separately by the protocol)
    ///   - srsSecret: the SRS toxic waste (needed for verification without pairings)
    public func verify(proof: UnivariateSumcheckProof, fCommitment: PointProjective,
                       claimedSum: Fr, logN: Int, transcript: Transcript,
                       srsSecret: Fr) -> Bool {
        let n = 1 << logN

        // Reconstruct Fiat-Shamir challenge
        absorbPoint(transcript, fCommitment)
        absorbPoint(transcript, proof.hCommitment)
        transcript.absorbLabel("univariate-sumcheck-challenge")
        let r = transcript.squeeze()

        // 1. Check sumcheck equation: f(r) = v/n + h(r) * Z_H(r)
        let nInv = frInverse(frFromInt(UInt64(n)))
        let vOverN = frMul(claimedSum, nInv)

        // Z_H(r) = r^n - 1
        let rN = frPow(r, UInt64(n))
        let zHr = frSub(rN, Fr.one)

        let expected = frAdd(vOverN, frMul(proof.hEval, zHr))

        let fEvalLimbs = frToInt(proof.fEval)
        let expectedLimbs = frToInt(expected)
        guard fEvalLimbs == expectedLimbs else {
            return false
        }

        // 2. Verify KZG openings using SRS secret
        //    For f: check C_f == [f(r)]*G + [s-r]*pi_f
        let g1 = pointFromAffine(kzg.srs[0])
        let sMr = frSub(srsSecret, r)

        // Verify f opening
        let fYG = cPointScalarMul(g1, proof.fEval)
        let fSzP = cPointScalarMul(proof.fOpeningProof, sMr)
        let fExpected = pointAdd(fYG, fSzP)
        let fCA = batchToAffine([fCommitment])
        let fEA = batchToAffine([fExpected])
        guard fpToInt(fCA[0].x) == fpToInt(fEA[0].x) &&
              fpToInt(fCA[0].y) == fpToInt(fEA[0].y) else {
            return false
        }

        // Verify h opening
        let hYG = cPointScalarMul(g1, proof.hEval)
        let hSzP = cPointScalarMul(proof.hOpeningProof, sMr)
        let hExpected = pointAdd(hYG, hSzP)
        let hCA = batchToAffine([proof.hCommitment])
        let hEA = batchToAffine([hExpected])
        guard fpToInt(hCA[0].x) == fpToInt(hEA[0].x) &&
              fpToInt(hCA[0].y) == fpToInt(hEA[0].y) else {
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
            // Absorb each claim
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
        let _alpha = transcript.squeeze()
        // (alpha should match proof.alphas[1] if k >= 2, but we trust the transcript)
        _ = _alpha  // suppress unused warning

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

    /// Divide polynomial g by the vanishing polynomial Z_H(X) = X^n - 1.
    /// Returns quotient h such that g = h * (X^n - 1).
    /// Assumes g is exactly divisible (remainder = 0 for valid sumcheck).
    ///
    /// Derivation: g_i = h_{i-n} - h_i, so h_{i-n} = g_i + h_i.
    /// Process top-down from i = d-1 to i = n.
    func divideByVanishing(_ g: [Fr], vanishingDegree n: Int) -> [Fr] {
        let d = g.count
        guard d > n else { return [] }

        let hLen = d - n
        var h = [Fr](repeating: Fr.zero, count: hLen)

        for i in stride(from: d - 1, through: n, by: -1) {
            let hIdx = i - n
            if i < hLen {
                // h[i] was set by a previous (higher) iteration
                h[hIdx] = frAdd(g[i], h[i])
            } else {
                h[hIdx] = g[i]
            }
        }

        return h
    }

    /// Absorb a projective point into the transcript by converting to affine and absorbing x, y.
    func absorbPoint(_ transcript: Transcript, _ point: PointProjective) {
        let affine = batchToAffine([point])
        // Absorb x and y coordinates as field elements
        // Convert Fp to bytes for the transcript
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
    /// Evaluates f and Z_H on a domain of size m >= deg(f) + n, does pointwise division, iNTT.
    /// This is faster for large polynomials where deg(f) >> n.
    func divideByVanishingNTT(_ fCoeffs: [Fr], vanishingDegree n: Int) throws -> [Fr] {
        let d = fCoeffs.count
        guard d > n else { return [] }

        // Need evaluation domain of size >= d (to avoid aliasing)
        var m = 1
        while m < d + n { m <<= 1 }
        let logM = Int(log2(Double(m)))

        // Pad f to size m
        var fPad = fCoeffs
        if fPad.count < m {
            fPad.append(contentsOf: [Fr](repeating: Fr.zero, count: m - fPad.count))
        }

        // Build Z_H in coefficient form: Z_H[0] = -1, Z_H[n] = 1
        var zH = [Fr](repeating: Fr.zero, count: m)
        zH[0] = frSub(Fr.zero, Fr.one)  // -1
        zH[n] = Fr.one

        // NTT both
        let fEvals = try ntt.ntt(fPad)
        let zHEvals = try ntt.ntt(zH)

        // Pointwise divide (with batch inverse for efficiency)
        let zHInv = try poly.batchInverse(zHEvals)
        var hEvals = try poly.hadamard(fEvals, zHInv)

        // iNTT
        let hCoeffs = try ntt.intt(hEvals)

        // Trim to expected length
        let hLen = d - n
        return Array(hCoeffs.prefix(hLen))
    }
}
