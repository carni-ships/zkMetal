// BLS12-381 optimal ate pairing implementation
// e: G1 x G2 -> GT (subgroup of Fp12*)
//
// The ate pairing for BLS12-381 uses parameter x = -0xd201000000010000
// Miller loop iterates over the bits of |x| = 0xd201000000010000
// Since x is negative, we conjugate the result at the end of the Miller loop.
//
// Tower: Fp2 = Fp[u]/(u^2+1), Fp6 = Fp2[v]/(v^3 - (1+u)), Fp12 = Fp6[w]/(w^2 - v)
// BLS12-381 uses a D-type sextic twist: E': y^2 = x^3 + 4(1+u)

import Foundation

// MARK: - Miller Loop (Affine G2 formulation)

/// Affine doubling on the G2 twist curve: T = 2T, returns slope lambda.
/// E': y^2 = x^3 + b' where b' = 4(1+u).
private func g2_381AffineDouble(_ t: inout G2Affine381) -> Fp2_381 {
    // lambda = 3*x^2 / (2*y)  (a=0 for BLS12-381 twist)
    let xsq = fp2_381Sqr(t.x)
    let num = fp2_381Add(fp2_381Double(xsq), xsq)  // 3*x^2
    let den = fp2_381Double(t.y)                     // 2*y
    let lam = fp2_381Mul(num, fp2_381Inverse(den))
    let x3 = fp2_381Sub(fp2_381Sqr(lam), fp2_381Double(t.x))
    let y3 = fp2_381Sub(fp2_381Mul(lam, fp2_381Sub(t.x, x3)), t.y)
    t = G2Affine381(x: x3, y: y3)
    return lam
}

/// Affine addition on the G2 twist curve: T = T + Q, returns slope lambda.
private func g2_381AffineAdd(_ t: inout G2Affine381, _ q: G2Affine381) -> Fp2_381 {
    let dx = fp2_381Sub(q.x, t.x)
    let dy = fp2_381Sub(q.y, t.y)
    let lam = fp2_381Mul(dy, fp2_381Inverse(dx))
    let x3 = fp2_381Sub(fp2_381Sub(fp2_381Sqr(lam), t.x), q.x)
    let y3 = fp2_381Sub(fp2_381Mul(lam, fp2_381Sub(t.x, x3)), t.y)
    t = G2Affine381(x: x3, y: y3)
    return lam
}

/// Line evaluation at G1 point P for a line on the twist with slope lambda through (xT, yT).
///
/// BLS12-381 uses a D-type sextic twist. The twist map psi: E' -> E is:
///   (x', y') -> (x'/w^2, y'/w^3) where w^6 = (1+u) (the non-residue)
/// In our tower w^2 = v, w^3 = v*w, so psi maps:
///   (x', y') -> (x'/v, y'/(v*w))
///
/// The preimage of P = (xP, yP) in G1 under psi^{-1} is:
///   x'_P = xP * v,  y'_P = yP * v*w
///
/// The line on E' through T with slope lambda:
///   l(x',y') = y' - yT - lambda*(x' - xT)
///
/// Evaluating at psi^{-1}(P):
///   l = yP*v*w - yT - lambda*(xP*v - xT)
///     = (lambda*xT - yT) + (-lambda*xP)*v + yP*(v*w)
///
/// Tower positions: 1 = c0.c0, v = c0.c1, v*w = c1.c1
private func lineEval381(lambda: Fp2_381, xT: Fp2_381, yT: Fp2_381,
                          px: Fp381, py: Fp381) -> Fp12_381 {
    let ell_0  = fp2_381Sub(fp2_381Mul(lambda, xT), yT)         // lambda*xT - yT at position 1
    let ell_v  = fp2_381Neg(fp2_381MulByFp(lambda, px))         // -lambda*xP at position v
    let ell_vw = Fp2_381(c0: py, c1: .zero)                     // yP at position v*w
    return Fp12_381(
        c0: Fp6_381(c0: ell_0, c1: ell_v, c2: .zero),
        c1: Fp6_381(c0: .zero, c1: ell_vw, c2: .zero))
}

/// Compute the Miller loop for the optimal ate pairing.
///
/// For BLS12-381, the optimal ate pairing uses the parameter |x| = 0xd201000000010000.
/// The Miller loop computes f_{|x|,Q}(P) using the standard double-and-add algorithm,
/// then conjugates since x < 0.
public func millerLoop381(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
    // |x| = 0xd201000000010000
    // Binary (MSB first, 64 bits):
    // 1101 0010 0000 0001 0000 0000 0000 0000 0000 0000 0000 0001 0000 0000 0000 0000
    // After leading 1 (bit 63), iterate bits 62 down to 0:
    let xBits: [Int] = [
        1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 62-47
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // bits 46-31
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 30-15
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0      // bits 14-0
    ]

    // Initialize: T = Q (affine)
    var tPt = q
    var f = Fp12_381.one

    for i in 0..<xBits.count {
        // Square f
        f = fp12_381Sqr(f)

        // Doubling step: compute tangent line at T evaluated at P, then T = 2T
        let oldT = tPt
        let lam = g2_381AffineDouble(&tPt)
        let lineD = lineEval381(lambda: lam, xT: oldT.x, yT: oldT.y, px: p.x, py: p.y)
        f = fp12_381Mul(f, lineD)

        // Addition step if bit is 1
        if xBits[i] == 1 {
            let oldT2 = tPt
            let lam2 = g2_381AffineAdd(&tPt, q)
            let lineA = lineEval381(lambda: lam2, xT: oldT2.x, yT: oldT2.y, px: p.x, py: p.y)
            f = fp12_381Mul(f, lineA)
        }
    }

    // x is negative, so conjugate the result
    f = fp12_381Conjugate(f)

    return f
}

// MARK: - Final Exponentiation

/// Final exponentiation: f^((p^12 - 1) / r)
/// Split into easy part and hard part:
///   Easy: f^(p^6 - 1) * f^(p^2 + 1)
///   Hard: f^((p^4 - p^2 + 1) / r)
public func finalExponentiation381(_ f: Fp12_381) -> Fp12_381 {
    // Easy part step 1: f^(p^6 - 1) = conj(f) * f^(-1)
    let fConj = fp12_381Conjugate(f)
    let fInv = fp12_381Inverse(f)
    var result = fp12_381Mul(fConj, fInv)

    // Easy part step 2: result^(p^2 + 1) = frobenius2(result) * result
    let resultP2 = fp12_381Frobenius2(result)
    result = fp12_381Mul(resultP2, result)

    // Hard part: result^((p^4 - p^2 + 1) / r)
    result = hardPartExponentiation(result)

    return result
}

/// Hard part of the final exponentiation.
/// Computes f^((p^4 - p^2 + 1) / r).
///
/// Follows the gnark-crypto BLS12-381 implementation exactly
/// (Hayashida-Hayasaka-Teruya, eprint 2020/875).
private func hardPartExponentiation(_ f: Fp12_381) -> Fp12_381 {
    var t0, t1, t2: Fp12_381
    var result = f

    // t[0] = result^2
    t0 = fp12_381Sqr(result)

    // t[1] = t[0]^(x/2)  (ExptHalf: (result^2)^(|x|/2) then conjugate = result^x)
    t1 = fp12_381PowByXHalf(t0)

    // t[2] = result^(-1)
    t2 = fp12_381Conjugate(result)

    // t[1] = t[1] * t[2] = result^(x-1)
    t1 = fp12_381Mul(t1, t2)

    // t[2] = t[1]^x = result^(x(x-1)) = result^(x^2-x)
    t2 = fp12_381PowByX(t1)

    // t[1] = t[1]^(-1) = result^(1-x)
    t1 = fp12_381Conjugate(t1)

    // t[1] = t[1] * t[2] = result^(x^2-2x+1) = result^((x-1)^2)
    t1 = fp12_381Mul(t1, t2)

    // t[2] = t[1]^x = result^(x(x-1)^2) = result^(x^3-2x^2+x)
    t2 = fp12_381PowByX(t1)

    // t[1] = frob(t[1]) = result^((x-1)^2 * p)
    t1 = fp12_381Frobenius(t1)

    // t[1] = t[1] * t[2] = result^((x-1)^2*p + x^3-2x^2+x)
    t1 = fp12_381Mul(t1, t2)

    // result = result * t[0] = f * f^2 = f^3
    result = fp12_381Mul(result, t0)

    // t[0] = t[1]^x
    t0 = fp12_381PowByX(t1)

    // t[2] = t[0]^x
    t2 = fp12_381PowByX(t0)

    // t[0] = frob2(t[1])
    t0 = fp12_381Frobenius2(t1)

    // t[1] = t[1]^(-1)
    t1 = fp12_381Conjugate(t1)

    // t[1] = t[1] * t[2]
    t1 = fp12_381Mul(t1, t2)

    // t[1] = t[1] * t[0]
    t1 = fp12_381Mul(t1, t0)

    // result = result * t[1]
    result = fp12_381Mul(result, t1)

    return result
}

// MARK: - Public API

/// Compute the optimal ate pairing e(P, Q) for BLS12-381.
/// P is a G1 point, Q is a G2 point. Returns an element of GT (subgroup of Fp12*).
public func bls12381Pairing(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
    let f = millerLoop381(p, q)
    return finalExponentiation381(f)
}

/// Pairing check: verify that the product of pairings equals 1.
/// Returns true if prod_i e(P_i, Q_i) = 1 in GT.
public func bls12381PairingCheck(_ pairs: [(G1Affine381, G2Affine381)]) -> Bool {
    var f = Fp12_381.one
    for (p, q) in pairs {
        let miller = millerLoop381(p, q)
        f = fp12_381Mul(f, miller)
    }
    let result = finalExponentiation381(f)

    // Check if result is the identity element of GT (= 1 in Fp12)
    return fp12_381Equal(result, .one)
}

/// Check equality of two Fp12 elements
public func fp12_381Equal(_ a: Fp12_381, _ b: Fp12_381) -> Bool {
    let diff = fp12_381Sub(a, b)
    return diff.c0.c0.c0.isZero && diff.c0.c0.c1.isZero &&
           diff.c0.c1.c0.isZero && diff.c0.c1.c1.isZero &&
           diff.c0.c2.c0.isZero && diff.c0.c2.c1.isZero &&
           diff.c1.c0.c0.isZero && diff.c1.c0.c1.isZero &&
           diff.c1.c1.c0.isZero && diff.c1.c1.c1.isZero &&
           diff.c1.c2.c0.isZero && diff.c1.c2.c1.isZero
}

// MARK: - BLS Signature Verification

/// BLS signature verification: e(pk, H(m)) = e(G1, sig)
/// pk: public key (G1 point)
/// message: hash-to-curve output (G2 point)
/// signature: BLS signature (G2 point)
///
/// Note: This checks e(pk, message) == e(G1_gen, signature)
/// which is equivalent to checking e(pk, message) * e(-G1_gen, signature) == 1
public func bls12381BLSVerify(
    pubkey: G1Affine381,
    message: G2Affine381,
    signature: G2Affine381
) -> Bool {
    let gen = bls12381G1Generator()
    let negGen = g1_381NegateAffine(gen)
    return bls12381PairingCheck([(pubkey, message), (negGen, signature)])
}
