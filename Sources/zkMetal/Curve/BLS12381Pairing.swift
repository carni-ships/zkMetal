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

// MARK: - Miller Loop

/// Compute the Miller loop for the optimal ate pairing.
///
/// For BLS12-381, the optimal ate pairing uses the parameter |x| = 0xd201000000010000.
/// The Miller loop computes f_{|x|,Q}(P) using the standard double-and-add algorithm,
/// then conjugates since x < 0.
public func millerLoop381(_ p: G1Affine381, _ q: G2Affine381) -> Fp12_381 {
    // |x| = 0xd201000000010000
    // Binary (MSB first, 64 bits):
    // 1101 0010 0000 0001 0000 0000 0000 0000 0000 0000 0000 0001 0000 0000 0000 0000
    // Bits 62 down to 0 after leading 1 at bit 63:
    // 63 bits: bits 62 down to 0 of |x| = 0xd201000000010000
    let xBits: [Int] = [
        1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 62-47
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // bits 46-31
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  // bits 30-15
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0      // bits 14-0
    ]

    // Initialize: T = Q (in Jacobian projective)
    var tX = q.x
    var tY = q.y
    var tZ = Fp2_381.one

    var f = Fp12_381.one

    for i in 0..<xBits.count {
        // Square f
        f = fp12_381Sqr(f)

        // Doubling step: compute tangent line at T evaluated at P, then T = 2T
        let lineD = millerDoublingStep(&tX, &tY, &tZ, p.x, p.y)
        f = fp12_381Mul(f, lineD)

        // Addition step if bit is 1
        if xBits[i] == 1 {
            let lineA = millerAdditionStep(&tX, &tY, &tZ, q, p.x, p.y)
            f = fp12_381Mul(f, lineA)
        }
    }

    // x is negative, so conjugate the result
    f = fp12_381Conjugate(f)

    return f
}

/// Doubling step: compute tangent line at T=(X:Y:Z) evaluated at P=(px,py),
/// then update T to 2T.
///
/// For a curve y^2 = x^3 + b in Jacobian coordinates, the tangent at T has slope
/// lambda = 3*X^2 / (2*Y*Z) (projective).
///
/// The line evaluation l_T,T(P) in the pairing context for a D-twist is:
///   l = ell_0 + ell_vw * (v*w) + ell_vv * v
/// where these are placed in the Fp12 tower.
private func millerDoublingStep(
    _ tX: inout Fp2_381, _ tY: inout Fp2_381, _ tZ: inout Fp2_381,
    _ px: Fp381, _ py: Fp381
) -> Fp12_381 {
    // Precompute
    let xx = fp2_381Sqr(tX)           // X^2
    let yy = fp2_381Sqr(tY)           // Y^2
    let yyyy = fp2_381Sqr(yy)         // Y^4
    let zz = fp2_381Sqr(tZ)           // Z^2

    // S = 2*((X+Y^2)^2 - X^2 - Y^4) = 4*X*Y^2
    let s = fp2_381Double(fp2_381Sub(fp2_381Sub(
        fp2_381Sqr(fp2_381Add(tX, yy)), xx), yyyy))

    // M = 3*X^2  (since a=0 for BLS12-381)
    let m = fp2_381Add(fp2_381Double(xx), xx)

    // Line evaluation coefficients (BEFORE updating T):
    // For D-type twist on BLS12-381:
    //   ell_0   = 3*b'*Z^2 - 2*Y^2  (constant term, lives at Fp6.c0.c0 = position 1)
    //   ell_vw  = -2*Y*Z * py        (lives at Fp12.c1.c1 = position v*w)
    //   ell_vv  = 3*X^2 * px         (lives at Fp12.c0.c1 = position v)
    // where b' = 4(1+u) is the twist curve coefficient.

    let bPrime = Fp2_381(c0: fp381FromInt(4), c1: fp381FromInt(4))
    let threeBZZ = fp2_381Mul(fp2_381Add(fp2_381Double(bPrime), bPrime), zz)
    let ell_0 = fp2_381Sub(threeBZZ, fp2_381Double(yy))

    // 2*Y*Z
    let twoYZ = fp2_381Double(fp2_381Mul(tY, tZ))
    let ell_vw = fp2_381Neg(fp2_381MulByFp(twoYZ, py))

    let ell_vv = fp2_381MulByFp(m, px)

    // Update T = 2T (Jacobian doubling for a=0)
    let x3 = fp2_381Sub(fp2_381Sqr(m), fp2_381Double(s))
    let y3 = fp2_381Sub(fp2_381Mul(m, fp2_381Sub(s, x3)),
                        fp2_381Double(fp2_381Double(fp2_381Double(yyyy))))
    let z3 = fp2_381Sub(fp2_381Sqr(fp2_381Add(tY, tZ)),
                        fp2_381Add(yy, zz))

    tX = x3
    tY = y3
    tZ = z3

    // Construct sparse Fp12 element.
    // Tower: Fp12 = Fp6.c0 + Fp6.c1 * w, Fp6 = c0 + c1*v + c2*v^2
    // Positions: 1 = c0.c0, v = c0.c1, v^2 = c0.c2, w = c1.c0, v*w = c1.c1, v^2*w = c1.c2
    return Fp12_381(
        c0: Fp6_381(c0: ell_0, c1: ell_vv, c2: .zero),
        c1: Fp6_381(c0: .zero, c1: ell_vw, c2: .zero))
}

/// Addition step: compute chord line through T and Q evaluated at P,
/// then update T to T + Q.
///
/// T is in Jacobian projective (X:Y:Z), Q is affine.
private func millerAdditionStep(
    _ tX: inout Fp2_381, _ tY: inout Fp2_381, _ tZ: inout Fp2_381,
    _ q: G2Affine381,
    _ px: Fp381, _ py: Fp381
) -> Fp12_381 {
    let zz = fp2_381Sqr(tZ)
    let zzz = fp2_381Mul(zz, tZ)

    // H = Q.x * Z^2 - X
    let h = fp2_381Sub(fp2_381Mul(q.x, zz), tX)
    // R = Q.y * Z^3 - Y
    let r = fp2_381Sub(fp2_381Mul(q.y, zzz), tY)

    // Line evaluation coefficients:
    // The chord through T and Q has slope lambda = R / H (projectively).
    // For D-type twist:
    //   ell_0   = T.x * Q.y * Z^3 - T.y * Q.x * Z^2  (cross term)
    //           Equivalently: X*R' - Y*H' where R'=Q.y*Z^3, H'=Q.x*Z^2
    //           Actually: ell_0 = R * T.x/Z^2 - H * T.y/Z^3 (unscaled)
    //           In projective form: ell_0 = R*X - H*Y (/ Z^5, absorbed into pairing)
    //   ell_vw  = -H * py   (= -(Q.x*Z^2 - X) * py, lives at v*w position)
    //   ell_vv  = R * px    (= (Q.y*Z^3 - Y) * px, lives at v position)
    //
    // But we need to be consistent with the projective scaling.
    // The standard approach: line l(P) = R * px - H * py + (stuff)
    // where the "stuff" term is the constant.

    let ell_vw = fp2_381Neg(fp2_381MulByFp(h, py))
    let ell_vv = fp2_381MulByFp(r, px)

    // Constant term: we need (T.x * (Q.y * Z^3) - T.y * (Q.x * Z^2)) but
    // properly scaled. In practice:
    // ell_0 = R * tX - H * tY  (absorbing Z factors consistently)
    // However, since R = Q.y*Z^3 - tY and H = Q.x*Z^2 - tX, we have:
    // lambda = R/H, and the line is: lambda*(xP - xQ) - (yP - yQ)
    // In projective form over the twist, the constant term is:
    let ell_0 = fp2_381Sub(fp2_381Mul(r, tX), fp2_381Mul(h, tY))

    // Update T = T + Q using standard Jacobian mixed addition
    let hh = fp2_381Sqr(h)
    let hhh = fp2_381Mul(h, hh)
    let rr = fp2_381Sqr(r)

    // X3 = R^2 - H^3 - 2*X*H^2
    let x3 = fp2_381Sub(fp2_381Sub(rr, hhh), fp2_381Double(fp2_381Mul(tX, hh)))
    // Y3 = R*(X*H^2 - X3) - Y*H^3
    let y3 = fp2_381Sub(fp2_381Mul(r, fp2_381Sub(fp2_381Mul(tX, hh), x3)),
                        fp2_381Mul(tY, hhh))
    // Z3 = H * Z
    let z3 = fp2_381Mul(h, tZ)

    tX = x3
    tY = y3
    tZ = z3

    // Sparse Fp12: same positions as doubling step
    return Fp12_381(
        c0: Fp6_381(c0: ell_0, c1: ell_vv, c2: .zero),
        c1: Fp6_381(c0: .zero, c1: ell_vw, c2: .zero))
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
/// Uses Algorithm 1 from Hayashida-Hayasaka-Teruya (eprint 2020/875),
/// as implemented in gnark-crypto bls12-381.
///
/// The hard exponent decomposes as:
///   (p^4 - p^2 + 1)/r = (2u^2-3u+3)*p + (2u^3-3u^2+u)*p^2 + (2u^4-3u^3+u^2)*p^3
/// where u = x = -0xd201000000010000 is the BLS12-381 parameter.
/// fp12_381PowByX computes f^|x| then conjugates (since x < 0), giving f^x.
private func hardPartExponentiation(_ f: Fp12_381) -> Fp12_381 {
    // t0 = f^2
    let t0 = fp12_381Sqr(f)

    // t1 = t0^x = f^(2x)
    let t1a = fp12_381PowByX(t0)

    // t2 = f^(-1)
    let t2a = fp12_381Conjugate(f)

    // t1 = f^(2x) * f^(-1) = f^(2x - 1)
    let t1b = fp12_381Mul(t1a, t2a)

    // t2 = (f^(2x-1))^x = f^(2x^2 - x)
    let t2b = fp12_381PowByX(t1b)

    // t3 = (f^(2x-1))^(-1) = f^(1 - 2x)
    let t3a = fp12_381Conjugate(t1b)

    // t1 = f^(2x^2-x) * f^(1-2x) = f^(2x^2 - 3x + 1)
    let t1c = fp12_381Mul(t2b, t3a)

    // t2 = (f^(2x^2-3x+1))^x = f^(2x^3 - 3x^2 + x)
    let t2c = fp12_381PowByX(t1c)

    // t3 = (f^(2x^3-3x^2+x))^x = f^(2x^4 - 3x^3 + x^2)
    let t3b = fp12_381PowByX(t2c)

    // t1 = f^(2x^2-3x+1) * f^2 = f^(2x^2-3x+3)
    let t1d = fp12_381Mul(t1c, t0)

    // Apply Frobenius maps:
    // t1 = f^((2x^2-3x+3)*p)
    let t1e = fp12_381Frobenius(t1d)

    // t2 = f^((2x^3-3x^2+x)*p^2)
    let t2d = fp12_381Frobenius2(t2c)

    // t3 = f^((2x^4-3x^3+x^2)*p^3)
    let t3c = fp12_381Frobenius3(t3b)

    // result = t1 * t2 * t3
    return fp12_381Mul(fp12_381Mul(t1e, t2d), t3c)
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
