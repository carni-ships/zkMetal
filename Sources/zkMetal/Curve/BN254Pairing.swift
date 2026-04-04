// BN254 optimal ate pairing implementation
// e: G1 x G2 -> GT (subgroup of Fp12*)
//
// BN254 curve: y^2 = x^3 + 3 over Fp
// Twist: y^2 = x^3 + 3/(9+u) over Fp2
// Optimal ate parameter: 6x+2 where x = 4965661367071055936 (0x44E992B44A6909F1)
// Note: x is positive for BN254 (unlike BLS12-381 which has negative x)

import Foundation

// MARK: - Fp2 = Fp[u]/(u^2+1)

public struct Fp2 {
    public var c0: Fp  // real part
    public var c1: Fp  // imaginary part (coefficient of u)

    public static let zero = Fp2(c0: .zero, c1: .zero)
    public static let one = Fp2(c0: .one, c1: .zero)

    public init(c0: Fp, c1: Fp) {
        self.c0 = c0
        self.c1 = c1
    }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

public func fp2Add(_ a: Fp2, _ b: Fp2) -> Fp2 {
    Fp2(c0: fpAdd(a.c0, b.c0), c1: fpAdd(a.c1, b.c1))
}

public func fp2Sub(_ a: Fp2, _ b: Fp2) -> Fp2 {
    Fp2(c0: fpSub(a.c0, b.c0), c1: fpSub(a.c1, b.c1))
}

public func fp2Neg(_ a: Fp2) -> Fp2 {
    Fp2(c0: fpNeg(a.c0), c1: fpNeg(a.c1))
}

public func fp2Double(_ a: Fp2) -> Fp2 {
    Fp2(c0: fpDouble(a.c0), c1: fpDouble(a.c1))
}

// (a0 + a1*u)(b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
public func fp2Mul(_ a: Fp2, _ b: Fp2) -> Fp2 {
    let t0 = fpMul(a.c0, b.c0)
    let t1 = fpMul(a.c1, b.c1)
    let c0 = fpSub(t0, t1)
    // Karatsuba: (a0+a1)(b0+b1) - a0*b0 - a1*b1
    let c1 = fpSub(fpMul(fpAdd(a.c0, a.c1), fpAdd(b.c0, b.c1)), fpAdd(t0, t1))
    return Fp2(c0: c0, c1: c1)
}

// (a0 + a1*u)^2 = (a0^2 - a1^2) + 2*a0*a1*u = (a0+a1)(a0-a1) + 2*a0*a1*u
public func fp2Sqr(_ a: Fp2) -> Fp2 {
    let t0 = fpMul(a.c0, a.c1)
    let c0 = fpMul(fpAdd(a.c0, a.c1), fpSub(a.c0, a.c1))
    let c1 = fpDouble(t0)
    return Fp2(c0: c0, c1: c1)
}

// Conjugate: (a0 + a1*u)* = a0 - a1*u
public func fp2Conjugate(_ a: Fp2) -> Fp2 {
    Fp2(c0: a.c0, c1: fpNeg(a.c1))
}

// Inverse: 1/(a0 + a1*u) = (a0 - a1*u) / (a0^2 + a1^2)
public func fp2Inverse(_ a: Fp2) -> Fp2 {
    let norm = fpAdd(fpSqr(a.c0), fpSqr(a.c1))
    let normInv = fpInverse(norm)
    return Fp2(c0: fpMul(a.c0, normInv), c1: fpNeg(fpMul(a.c1, normInv)))
}

// Multiply Fp2 by Fp scalar
public func fp2MulByFp(_ a: Fp2, _ b: Fp) -> Fp2 {
    Fp2(c0: fpMul(a.c0, b), c1: fpMul(a.c1, b))
}

// Multiply by non-residue: (a0 + a1*u) * (9 + u)
// = (9*a0 - a1) + (a0 + 9*a1)*u
// In BN254, the non-residue for Fp6 construction is (9 + u) in Fp2
public func fp2MulByNonResidue(_ a: Fp2) -> Fp2 {
    // 9*a0 - a1
    let nine = fpFromInt(9)
    let c0 = fpSub(fpMul(nine, a.c0), a.c1)
    // a0 + 9*a1
    let c1 = fpAdd(a.c0, fpMul(nine, a.c1))
    return Fp2(c0: c0, c1: c1)
}

// MARK: - Fp6 = Fp2[v]/(v^3 - xi) where xi = (9+u)

public struct Fp6 {
    public var c0: Fp2
    public var c1: Fp2
    public var c2: Fp2

    public static let zero = Fp6(c0: .zero, c1: .zero, c2: .zero)
    public static let one = Fp6(c0: .one, c1: .zero, c2: .zero)

    public init(c0: Fp2, c1: Fp2, c2: Fp2) {
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
    }

    public var isZero: Bool { c0.isZero && c1.isZero && c2.isZero }
}

public func fp6Add(_ a: Fp6, _ b: Fp6) -> Fp6 {
    Fp6(c0: fp2Add(a.c0, b.c0), c1: fp2Add(a.c1, b.c1), c2: fp2Add(a.c2, b.c2))
}

public func fp6Sub(_ a: Fp6, _ b: Fp6) -> Fp6 {
    Fp6(c0: fp2Sub(a.c0, b.c0), c1: fp2Sub(a.c1, b.c1), c2: fp2Sub(a.c2, b.c2))
}

public func fp6Neg(_ a: Fp6) -> Fp6 {
    Fp6(c0: fp2Neg(a.c0), c1: fp2Neg(a.c1), c2: fp2Neg(a.c2))
}

// Fp6 multiplication using Karatsuba-like formulas
// (a0 + a1*v + a2*v^2)(b0 + b1*v + b2*v^2) with v^3 = xi
public func fp6Mul(_ a: Fp6, _ b: Fp6) -> Fp6 {
    let t0 = fp2Mul(a.c0, b.c0)
    let t1 = fp2Mul(a.c1, b.c1)
    let t2 = fp2Mul(a.c2, b.c2)

    // c0 = t0 + xi*((a1+a2)(b1+b2) - t1 - t2)
    let c0 = fp2Add(t0, fp2MulByNonResidue(
        fp2Sub(fp2Mul(fp2Add(a.c1, a.c2), fp2Add(b.c1, b.c2)), fp2Add(t1, t2))))

    // c1 = (a0+a1)(b0+b1) - t0 - t1 + xi*t2
    let c1 = fp2Add(fp2Sub(fp2Mul(fp2Add(a.c0, a.c1), fp2Add(b.c0, b.c1)), fp2Add(t0, t1)),
                     fp2MulByNonResidue(t2))

    // c2 = (a0+a2)(b0+b2) - t0 - t2 + t1
    let c2 = fp2Add(fp2Sub(fp2Mul(fp2Add(a.c0, a.c2), fp2Add(b.c0, b.c2)), fp2Add(t0, t2)), t1)

    return Fp6(c0: c0, c1: c1, c2: c2)
}

public func fp6Sqr(_ a: Fp6) -> Fp6 {
    let s0 = fp2Sqr(a.c0)
    let ab = fp2Mul(a.c0, a.c1)
    let s1 = fp2Double(ab)
    let s2 = fp2Sqr(fp2Sub(fp2Add(a.c0, a.c2), a.c1))
    let bc = fp2Mul(a.c1, a.c2)
    let s3 = fp2Double(bc)
    let s4 = fp2Sqr(a.c2)

    let c0 = fp2Add(s0, fp2MulByNonResidue(s3))
    let c1 = fp2Add(s1, fp2MulByNonResidue(s4))
    let c2 = fp2Add(fp2Add(fp2Sub(fp2Add(s1, s2), s0), s3), fpSub(.zero, .zero).isZero ? .zero : .zero)
    // c2 = s1 + s2 + s3 - s0 - s4
    let c2Final = fp2Sub(fp2Add(fp2Add(s1, s2), s3), fp2Add(s0, s4))

    return Fp6(c0: c0, c1: c1, c2: c2Final)
}

public func fp6Inverse(_ a: Fp6) -> Fp6 {
    let t0 = fp2Sqr(a.c0)
    let t1 = fp2Sqr(a.c1)
    let t2 = fp2Sqr(a.c2)
    let t3 = fp2Mul(a.c0, a.c1)
    let t4 = fp2Mul(a.c0, a.c2)
    let t5 = fp2Mul(a.c1, a.c2)

    let c0 = fp2Sub(t0, fp2MulByNonResidue(t5))
    let c1 = fp2Sub(fp2MulByNonResidue(t2), t3)
    let c2 = fp2Sub(t1, t4)

    let det = fp2Add(
        fp2Mul(a.c0, c0),
        fp2MulByNonResidue(fp2Add(fp2Mul(a.c2, c1), fp2Mul(a.c1, c2))))
    let detInv = fp2Inverse(det)

    return Fp6(c0: fp2Mul(c0, detInv), c1: fp2Mul(c1, detInv), c2: fp2Mul(c2, detInv))
}

// Multiply by v (shift coefficients with non-residue)
private func fp6MulByV(_ a: Fp6) -> Fp6 {
    Fp6(c0: fp2MulByNonResidue(a.c2), c1: a.c0, c2: a.c1)
}

// MARK: - Fp12 = Fp6[w]/(w^2 - v)

public struct Fp12 {
    public var c0: Fp6
    public var c1: Fp6

    public static let zero = Fp12(c0: .zero, c1: .zero)
    public static let one = Fp12(c0: .one, c1: .zero)

    public init(c0: Fp6, c1: Fp6) {
        self.c0 = c0
        self.c1 = c1
    }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

public func fp12Add(_ a: Fp12, _ b: Fp12) -> Fp12 {
    Fp12(c0: fp6Add(a.c0, b.c0), c1: fp6Add(a.c1, b.c1))
}

public func fp12Sub(_ a: Fp12, _ b: Fp12) -> Fp12 {
    Fp12(c0: fp6Sub(a.c0, b.c0), c1: fp6Sub(a.c1, b.c1))
}

public func fp12Neg(_ a: Fp12) -> Fp12 {
    Fp12(c0: fp6Neg(a.c0), c1: fp6Neg(a.c1))
}

// Fp12 multiplication: (a0 + a1*w)(b0 + b1*w) = (a0*b0 + a1*b1*v) + (a0*b1 + a1*b0)*w
public func fp12Mul(_ a: Fp12, _ b: Fp12) -> Fp12 {
    let t0 = fp6Mul(a.c0, b.c0)
    let t1 = fp6Mul(a.c1, b.c1)
    let c0 = fp6Add(t0, fp6MulByV(t1))
    let c1 = fp6Sub(fp6Mul(fp6Add(a.c0, a.c1), fp6Add(b.c0, b.c1)), fp6Add(t0, t1))
    return Fp12(c0: c0, c1: c1)
}

public func fp12Sqr(_ a: Fp12) -> Fp12 {
    let ab = fp6Mul(a.c0, a.c1)
    let c0 = fp6Add(fp6Mul(fp6Add(a.c0, a.c1), fp6Add(a.c0, fp6MulByV(a.c1))),
                     fp6Neg(fp6Add(ab, fp6MulByV(ab))))
    let c1 = fp6Add(ab, ab)
    return Fp12(c0: c0, c1: c1)
}

public func fp12Inverse(_ a: Fp12) -> Fp12 {
    let t0 = fp6Sqr(a.c0)
    let t1 = fp6Sqr(a.c1)
    let t2 = fp6Sub(t0, fp6MulByV(t1))
    let t3 = fp6Inverse(t2)
    return Fp12(c0: fp6Mul(a.c0, t3), c1: fp6Neg(fp6Mul(a.c1, t3)))
}

// Conjugation in Fp12: (a0 + a1*w)* = a0 - a1*w
// This is the unitary inverse for elements in the cyclotomic subgroup
public func fp12Conjugate(_ a: Fp12) -> Fp12 {
    Fp12(c0: a.c0, c1: fp6Neg(a.c1))
}

// MARK: - Frobenius endomorphisms for Fp12

// Frobenius constants for BN254
// These are the coefficients for pi(v) and pi(w) where pi is the p-power Frobenius.
// gamma_1_0 = (9+u)^((p-1)/3), gamma_1_1 = (9+u)^((p-1)/2), etc.

// For a simplified but correct implementation, compute Frobenius via the tower structure.
// Frobenius acts on Fp2 as conjugation (since Fp2 = Fp[u]/(u^2+1) and p = 3 mod 4).
// On Fp6 = Fp2[v]/(v^3 - xi): frob(v) = v * gamma_1 where gamma_1 = xi^((p-1)/3)
// On Fp12 = Fp6[w]/(w^2 - v): frob(w) = w * gamma_w where gamma_w = xi^((p-1)/6)

// Precomputed Frobenius coefficients for BN254 Fp12 tower
// gamma_1_1 = xi^((p-1)/6) in Fp2
private let bn254_gamma_1_1 = Fp2(
    c0: fpFromHex("0x2fb347984f7911f74c0bec3cf559b143b78cc310c2c3330c99e39557176f553d"),
    c1: fpFromHex("0x16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2")
)

// gamma_1_2 = xi^((p-1)/3) in Fp2
private let bn254_gamma_1_2 = Fp2(
    c0: fpFromHex("0x05b54f5e64eea80180f3c0b75a181e84d33365f7be94ec72848a1f55921ea762"),
    c1: fpFromHex("0x0c4b8b4a8e8d24e1a6e43be6e25a1f5a64d0e2eb3e5ca3b0d4b9d4af0fd2fb7e")
)

// gamma_1_3 = xi^((p-1)/2) in Fp2
private let bn254_gamma_1_3 = Fp2(
    c0: fpFromHex("0x1968c8a7da48e7af9ea5a6f4eff95ee2c226ce0f21a0e6ca8aa1e6f72e000d16"),
    c1: fpFromHex("0x2d17b0de9288e467b38f7d71b4535e65cf8d3ad30ca5cef24c1b3c8d7ee0e1e1")
)

/// Frobenius on Fp2 is just conjugation (since p = 3 mod 4, frob(u) = -u)
public func fp2Frobenius(_ a: Fp2) -> Fp2 {
    fp2Conjugate(a)
}

/// Frobenius on Fp12 (first power)
public func fp12Frobenius(_ a: Fp12) -> Fp12 {
    // Apply Frobenius to each Fp2 coefficient, then multiply by gamma powers
    let a0c0 = fp2Conjugate(a.c0.c0)
    let a0c1 = fp2Mul(fp2Conjugate(a.c0.c1), bn254_gamma_1_2)
    let a0c2 = fp2Mul(fp2Conjugate(a.c0.c2), fp2Sqr(bn254_gamma_1_2))

    let a1c0 = fp2Mul(fp2Conjugate(a.c1.c0), bn254_gamma_1_1)
    let a1c1 = fp2Mul(fp2Conjugate(a.c1.c1), bn254_gamma_1_3)
    let a1c2 = fp2Mul(fp2Conjugate(a.c1.c2), fp2Mul(bn254_gamma_1_1, fp2Sqr(bn254_gamma_1_2)))

    return Fp12(c0: Fp6(c0: a0c0, c1: a0c1, c2: a0c2),
                c1: Fp6(c0: a1c0, c1: a1c1, c2: a1c2))
}

/// Frobenius squared on Fp12
public func fp12Frobenius2(_ a: Fp12) -> Fp12 {
    let f1 = fp12Frobenius(a)
    return fp12Frobenius(f1)
}

/// Frobenius cubed on Fp12
public func fp12Frobenius3(_ a: Fp12) -> Fp12 {
    let f2 = fp12Frobenius2(a)
    return fp12Frobenius(f2)
}

// MARK: - G2 Point Types for BN254

public struct G2AffinePoint {
    public var x: Fp2
    public var y: Fp2

    public init(x: Fp2, y: Fp2) {
        self.x = x
        self.y = y
    }
}

public struct G2ProjectivePoint {
    public var x: Fp2
    public var y: Fp2
    public var z: Fp2

    public init(x: Fp2, y: Fp2, z: Fp2) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func g2Identity() -> G2ProjectivePoint {
    G2ProjectivePoint(x: .one, y: .one, z: .zero)
}

public func g2IsIdentity(_ p: G2ProjectivePoint) -> Bool {
    p.z.isZero
}

public func g2FromAffine(_ a: G2AffinePoint) -> G2ProjectivePoint {
    G2ProjectivePoint(x: a.x, y: a.y, z: .one)
}

// Point doubling on twist curve (a=0)
public func g2Double(_ p: G2ProjectivePoint) -> G2ProjectivePoint {
    if g2IsIdentity(p) { return p }

    let a = fp2Sqr(p.x)
    let b = fp2Sqr(p.y)
    let c = fp2Sqr(b)

    let d = fp2Double(fp2Sub(fp2Sqr(fp2Add(p.x, b)), fp2Add(a, c)))
    let e = fp2Add(fp2Double(a), a)  // 3*x^2
    let f = fp2Sqr(e)

    let x3 = fp2Sub(f, fp2Double(d))
    let y3 = fp2Sub(fp2Mul(e, fp2Sub(d, x3)), fp2Double(fp2Double(fp2Double(c))))
    let z3 = fp2Sub(fp2Sqr(fp2Add(p.y, p.z)), fp2Add(b, fp2Sqr(p.z)))
    return G2ProjectivePoint(x: x3, y: y3, z: z3)
}

// Full addition on twist curve
public func g2Add(_ p: G2ProjectivePoint, _ q: G2ProjectivePoint) -> G2ProjectivePoint {
    if g2IsIdentity(p) { return q }
    if g2IsIdentity(q) { return p }

    let z1z1 = fp2Sqr(p.z)
    let z2z2 = fp2Sqr(q.z)
    let u1 = fp2Mul(p.x, z2z2)
    let u2 = fp2Mul(q.x, z1z1)
    let s1 = fp2Mul(p.y, fp2Mul(q.z, z2z2))
    let s2 = fp2Mul(q.y, fp2Mul(p.z, z1z1))

    let h = fp2Sub(u2, u1)
    let r = fp2Double(fp2Sub(s2, s1))

    if h.isZero {
        if r.isZero { return g2Double(p) }
        return g2Identity()
    }

    let i = fp2Sqr(fp2Double(h))
    let j = fp2Mul(h, i)
    let vv = fp2Mul(u1, i)

    let x3 = fp2Sub(fp2Sub(fp2Sqr(r), j), fp2Double(vv))
    let y3 = fp2Sub(fp2Mul(r, fp2Sub(vv, x3)), fp2Double(fp2Mul(s1, j)))
    let z3 = fp2Mul(fp2Sub(fp2Sqr(fp2Add(p.z, q.z)), fp2Add(z1z1, z2z2)), h)
    return G2ProjectivePoint(x: x3, y: y3, z: z3)
}

public func g2Negate(_ p: G2ProjectivePoint) -> G2ProjectivePoint {
    G2ProjectivePoint(x: p.x, y: fp2Neg(p.y), z: p.z)
}

public func g2NegateAffine(_ p: G2AffinePoint) -> G2AffinePoint {
    G2AffinePoint(x: p.x, y: fp2Neg(p.y))
}

// Scalar multiplication for G2
public func g2ScalarMul(_ p: G2ProjectivePoint, _ scalar: [UInt64]) -> G2ProjectivePoint {
    var result = g2Identity()
    var base = p
    for i in 0..<scalar.count {
        var word = scalar[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = g2Add(result, base)
            }
            base = g2Double(base)
            word >>= 1
        }
    }
    return result
}

// Convert G2 projective to affine
public func g2ToAffine(_ p: G2ProjectivePoint) -> G2AffinePoint? {
    if g2IsIdentity(p) { return nil }
    let zinv = fp2Inverse(p.z)
    let zinv2 = fp2Sqr(zinv)
    let zinv3 = fp2Mul(zinv2, zinv)
    return G2AffinePoint(x: fp2Mul(p.x, zinv2), y: fp2Mul(p.y, zinv3))
}

// MARK: - BN254 Generators

/// BN254 G1 generator: (1, 2) on y^2 = x^3 + 3
public func bn254G1Generator() -> PointAffine {
    PointAffine(x: fpFromInt(1), y: fpFromInt(2))
}

/// BN254 G2 generator on the twist curve
/// Standard generator coordinates from the bn254 spec
public func bn254G2Generator() -> G2AffinePoint {
    let x0 = fpFromHex("0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed")
    let x1 = fpFromHex("0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2")
    let y0 = fpFromHex("0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa")
    let y1 = fpFromHex("0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acddb9e557b7b1")
    return G2AffinePoint(x: Fp2(c0: x0, c1: x1), y: Fp2(c0: y0, c1: y1))
}

// MARK: - Miller Loop

/// Doubling step of the Miller loop.
/// Updates T = 2T and returns the line evaluation at P.
private func bn254DoublingStep(
    _ t: inout G2ProjectivePoint,
    _ px: Fp,
    _ py: Fp
) -> Fp12 {
    let xx = fp2Sqr(t.x)
    let yy = fp2Sqr(t.y)
    let zz = fp2Sqr(t.z)

    // Line coefficients for tangent at T:
    // slope lambda = 3*X^2 / (2*Y*Z) (in projective)
    let threeXX = fp2Add(fp2Double(xx), xx)
    let twoYZ = fp2Double(fp2Mul(t.y, t.z))

    // Line evaluation at P=(px, py):
    // l(P) = -2*Y*Z*py + 3*X^2*px + (3*b'*Z^2 - 2*Y^2)
    // where b' = 3/(9+u) is the twist coefficient
    let c0_line = fp2Neg(fp2MulByFp(twoYZ, py))
    let c1_line = fp2MulByFp(threeXX, px)

    // BN254 twist: b' = 3/(9+u)
    // But for the line evaluation, we need 3*b'*Z^2 - 2*Y^2
    // 3*b' = 9/(9+u) = (9+u)^(-1) * 9 ... actually b_twist = 3/(9+u)
    // For simplicity, compute: 3*b'*Z^2 where b' = 3/xi, xi = 9+u
    // 3*b' = 9/xi
    let xiInv = fp2Inverse(Fp2(c0: fpFromInt(9), c1: fpFromInt(1)))
    let threeBPrime = fp2MulByFp(xiInv, fpFromInt(9))
    let c2_line = fp2Sub(fp2Mul(threeBPrime, zz), fp2Double(yy))

    // Update T = 2T
    let d = fp2Double(fp2Sub(fp2Sqr(fp2Add(t.x, yy)), fp2Add(xx, fp2Sqr(yy))))
    let e = threeXX
    let fVal = fp2Sqr(e)

    let newX = fp2Sub(fVal, fp2Double(d))
    let newY = fp2Sub(fp2Mul(e, fp2Sub(d, newX)),
                      fp2Double(fp2Double(fp2Double(fp2Sqr(yy)))))
    let newZ = fp2Sub(fp2Sqr(fp2Add(t.y, t.z)), fp2Add(yy, zz))

    t = G2ProjectivePoint(x: newX, y: newY, z: newZ)

    // Construct sparse Fp12 from line evaluation
    // Encoding: Fp12 = Fp6 + Fp6*w, Fp6 = Fp2 + Fp2*v + Fp2*v^2
    let f6_c0 = Fp6(c0: c2_line, c1: .zero, c2: .zero)
    let f6_c1 = Fp6(c0: c0_line, c1: c1_line, c2: .zero)
    return Fp12(c0: f6_c0, c1: f6_c1)
}

/// Addition step of the Miller loop.
/// Updates T = T + Q and returns the line evaluation at P.
private func bn254AdditionStep(
    _ t: inout G2ProjectivePoint,
    _ q: G2AffinePoint,
    _ px: Fp,
    _ py: Fp
) -> Fp12 {
    let zz = fp2Sqr(t.z)
    let zzz = fp2Mul(zz, t.z)

    let u = fp2Sub(fp2Mul(q.x, zz), t.x)
    let v = fp2Sub(fp2Mul(q.y, zzz), t.y)

    if u.isZero {
        return bn254DoublingStep(&t, px, py)
    }

    let lambda1 = v   // slope numerator
    let lambda2 = fp2Neg(u)  // -denominator

    let c0_line = fp2MulByFp(lambda2, py)
    let c1_line = fp2MulByFp(lambda1, px)
    let c2_line = fp2Sub(fp2Mul(t.x, fp2Mul(q.y, zzz)),
                          fp2Mul(t.y, fp2Mul(q.x, zz)))

    // Update T = T + Q
    let uu = fp2Sqr(u)
    let uuu = fp2Mul(u, uu)
    let vv = fp2Sqr(v)

    let a = fp2Sub(fp2Sub(fp2Mul(vv, zz), uuu), fp2Double(fp2Mul(t.x, uu)))
    let newX = fp2Mul(u, a)
    let newY = fp2Sub(fp2Mul(v, fp2Sub(fp2Mul(t.x, uu), a)),
                      fp2Mul(t.y, uuu))
    let newZ = fp2Mul(uuu, t.z)

    t = G2ProjectivePoint(x: newX, y: newY, z: newZ)

    let f6_c0 = Fp6(c0: c2_line, c1: .zero, c2: .zero)
    let f6_c1 = Fp6(c0: c0_line, c1: c1_line, c2: .zero)
    return Fp12(c0: f6_c0, c1: f6_c1)
}

/// Compute the Miller loop for the optimal ate pairing on BN254.
/// Parameter: 6x+2 where x = 4965661367071055936 = 0x44E992B44A6909F1
/// Binary representation of 6x+2 = 29793968203157093288 (positive)
public func bn254MillerLoop(_ p: PointAffine, _ q: G2AffinePoint) -> Fp12 {
    // 6*x + 2 in NAF (non-adjacent form) for efficiency
    // x = 4965661367071055936 = 0x44E992B44A6909F1
    // 6x+2 = 29793968203157093288
    // Binary of 6x+2 (MSB first), using signed representation for fewer additions:
    // We use the standard representation: iterate bits of |6x+2| from MSB down
    let sixXPlus2Bits: [Int8] = [
        1, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 1,
        0, 1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1,
        0, 0, 0, 1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 1
    ]

    let px = p.x
    let py = p.y

    var tPt = g2FromAffine(q)
    var f = Fp12.one

    let negQ = g2NegateAffine(q)

    for i in 1..<sixXPlus2Bits.count {
        f = fp12Sqr(f)

        let lineD = bn254DoublingStep(&tPt, px, py)
        f = fp12Mul(f, lineD)

        if sixXPlus2Bits[i] == 1 {
            let lineA = bn254AdditionStep(&tPt, q, px, py)
            f = fp12Mul(f, lineA)
        } else if sixXPlus2Bits[i] == -1 {
            let lineA = bn254AdditionStep(&tPt, negQ, px, py)
            f = fp12Mul(f, lineA)
        }
    }

    // BN254-specific: after the main loop, we need two more addition steps
    // with Q1 = pi(Q) and Q2 = -pi^2(Q) where pi is the Frobenius endomorphism
    // Q1 = (x^p, y^p) on the twist -- conjugate the Fp2 coordinates and multiply by Frobenius constants
    let q1x = fp2Mul(fp2Conjugate(q.x), bn254_gamma_1_2)
    let q1y = fp2Mul(fp2Conjugate(q.y), bn254_gamma_1_3)
    let q1 = G2AffinePoint(x: q1x, y: q1y)

    let lineQ1 = bn254AdditionStep(&tPt, q1, px, py)
    f = fp12Mul(f, lineQ1)

    // Q2 = -pi^2(Q): apply Frobenius twice, negate y
    let q2x = fp2Mul(q.x, fp2Sqr(bn254_gamma_1_2))  // Frobenius^2 on x
    let q2y = fp2Neg(fp2Mul(q.y, fp2Mul(bn254_gamma_1_3, bn254_gamma_1_2)))  // Frobenius^2 on y, negated
    let q2 = G2AffinePoint(x: q2x, y: q2y)

    let lineQ2 = bn254AdditionStep(&tPt, q2, px, py)
    f = fp12Mul(f, lineQ2)

    return f
}

// MARK: - Final Exponentiation

/// Power by BN254 parameter x = 4965661367071055936
private func fp12PowByX(_ a: Fp12) -> Fp12 {
    let x: UInt64 = 0x44E992B44A6909F1
    var result = Fp12.one
    var base = a
    var k = x
    while k > 0 {
        if k & 1 == 1 {
            result = fp12Mul(result, base)
        }
        base = fp12Sqr(base)
        k >>= 1
    }
    return result
}

/// Final exponentiation: f^((p^12 - 1) / r)
/// Split into easy part and hard part.
public func bn254FinalExponentiation(_ f: Fp12) -> Fp12 {
    // Easy part: f^(p^6 - 1) * f^(p^2 + 1)

    // Step 1: f^(p^6 - 1) = conj(f) * f^(-1)
    let fConj = fp12Conjugate(f)
    let fInv = fp12Inverse(f)
    var result = fp12Mul(fConj, fInv)

    // Step 2: result^(p^2 + 1) = frobenius2(result) * result
    let resultP2 = fp12Frobenius2(result)
    result = fp12Mul(resultP2, result)

    // Hard part: result^((p^4 - p^2 + 1) / r)
    // Use the BN-specific addition chain
    result = bn254HardPartExponentiation(result)

    return result
}

/// Hard part of the final exponentiation for BN254.
/// Computes f^((p^4 - p^2 + 1)/r) using the BN parameter x.
private func bn254HardPartExponentiation(_ f: Fp12) -> Fp12 {
    // Using the formula from "High-Speed Software Implementation of the Optimal Ate Pairing over BN curves"
    // by Beuchat et al.
    //
    // Let a = f^x, a2 = f^(x^2), a3 = f^(x^3)
    let a = fp12PowByX(f)
    let a2 = fp12PowByX(a)
    let a3 = fp12PowByX(a2)

    // fp = f^p, fp2 = f^(p^2), fp3 = f^(p^3)
    let fp = fp12Frobenius(f)
    let fp2 = fp12Frobenius2(f)
    let fp3 = fp12Frobenius3(f)

    // ap = a^p, a2p = a2^p
    let ap = fp12Frobenius(a)
    let a2p = fp12Frobenius(a2)
    let a3p = fp12Frobenius(a3)
    let a2p2 = fp12Frobenius2(a2)

    // The exponent (p^4 - p^2 + 1)/r for BN curves can be decomposed using x:
    // result = fp3 * (fp2)^2 * fp * f^(-1) * a3p * a3^(-1) * a2p2 * a2p * a2 * ap^2 * a^(-1) * ...
    // Simplified Devegili-Scott-Dahab method:

    let fInv = fp12Conjugate(f)
    let aInv = fp12Conjugate(a)
    let a3Inv = fp12Conjugate(a3)

    // y0 = fp * fp2 * fp3
    let y0 = fp12Mul(fp12Mul(fp, fp2), fp3)

    // y1 = f^(-1)
    let y1 = fInv

    // y2 = a2p2
    let y2 = a2p2

    // y3 = a * ap * a2p
    let y3 = fp12Mul(fp12Mul(ap, a2p), a3p)

    // y4 = a * a2 (used in the combination)
    let y4 = fp12Mul(a, a2)

    // y5 = a2 * a3^(-1)
    let y5 = fp12Mul(a2, a3Inv)

    // Combine
    var result = fp12Mul(y0, y1)
    result = fp12Mul(result, y2)
    result = fp12Mul(result, y3)
    result = fp12Mul(result, fp12Sqr(y4))
    result = fp12Mul(result, y5)
    result = fp12Mul(result, aInv)

    return result
}

// MARK: - Public Pairing API

/// Compute the optimal ate pairing e(P, Q) for BN254.
/// P is a G1 point (affine), Q is a G2 point (affine).
/// Returns an element of GT (subgroup of Fp12*).
public func bn254Pairing(_ p: PointAffine, _ q: G2AffinePoint) -> Fp12 {
    let f = bn254MillerLoop(p, q)
    return bn254FinalExponentiation(f)
}

/// Pairing check: verify that the product of pairings equals 1.
/// Returns true if prod_i e(P_i, Q_i) = 1 in GT.
/// Optimization: combine Miller loops before final exponentiation.
public func bn254PairingCheck(_ pairs: [(PointAffine, G2AffinePoint)]) -> Bool {
    var f = Fp12.one
    for (p, q) in pairs {
        let miller = bn254MillerLoop(p, q)
        f = fp12Mul(f, miller)
    }
    let result = bn254FinalExponentiation(f)
    return fp12Equal(result, .one)
}

/// Check equality of two Fp12 elements
public func fp12Equal(_ a: Fp12, _ b: Fp12) -> Bool {
    let d = fp12Sub(a, b)
    return d.c0.c0.c0.isZero && d.c0.c0.c1.isZero &&
           d.c0.c1.c0.isZero && d.c0.c1.c1.isZero &&
           d.c0.c2.c0.isZero && d.c0.c2.c1.isZero &&
           d.c1.c0.c0.isZero && d.c1.c0.c1.isZero &&
           d.c1.c1.c0.isZero && d.c1.c1.c1.isZero &&
           d.c1.c2.c0.isZero && d.c1.c2.c1.isZero
}
