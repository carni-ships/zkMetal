// BN254 optimal ate pairing implementation
// e: G1 x G2 -> GT (subgroup of Fp12*)
//
// BN254 curve: y^2 = x^3 + 3 over Fp
// Twist: y^2 = x^3 + 3/(9+u) over Fp2
// Optimal ate parameter: 6x+2 where x = 4965661367071055936 (0x44E992B44A6909F1)

import Foundation
import NeonFieldOps

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
    let c1 = fpSub(fpMul(fpAdd(a.c0, a.c1), fpAdd(b.c0, b.c1)), fpAdd(t0, t1))
    return Fp2(c0: c0, c1: c1)
}

// (a0 + a1*u)^2 = (a0+a1)(a0-a1) + 2*a0*a1*u
public func fp2Sqr(_ a: Fp2) -> Fp2 {
    let t0 = fpMul(a.c0, a.c1)
    let c0 = fpMul(fpAdd(a.c0, a.c1), fpSub(a.c0, a.c1))
    let c1 = fpDouble(t0)
    return Fp2(c0: c0, c1: c1)
}

public func fp2Conjugate(_ a: Fp2) -> Fp2 {
    Fp2(c0: a.c0, c1: fpNeg(a.c1))
}

public func fp2Inverse(_ a: Fp2) -> Fp2 {
    let norm = fpAdd(fpSqr(a.c0), fpSqr(a.c1))
    let normInv = fpInverse(norm)
    return Fp2(c0: fpMul(a.c0, normInv), c1: fpNeg(fpMul(a.c1, normInv)))
}

public func fp2MulByFp(_ a: Fp2, _ b: Fp) -> Fp2 {
    Fp2(c0: fpMul(a.c0, b), c1: fpMul(a.c1, b))
}

// Multiply by non-residue xi = 9 + u: (a0 + a1*u)(9 + u) = (9*a0 - a1) + (a0 + 9*a1)*u
public func fp2MulByNonResidue(_ a: Fp2) -> Fp2 {
    let nine = fpFromInt(9)
    let c0 = fpSub(fpMul(nine, a.c0), a.c1)
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

// Fp6 multiplication using Karatsuba
public func fp6Mul(_ a: Fp6, _ b: Fp6) -> Fp6 {
    let t0 = fp2Mul(a.c0, b.c0)
    let t1 = fp2Mul(a.c1, b.c1)
    let t2 = fp2Mul(a.c2, b.c2)

    let c0 = fp2Add(t0, fp2MulByNonResidue(
        fp2Sub(fp2Mul(fp2Add(a.c1, a.c2), fp2Add(b.c1, b.c2)), fp2Add(t1, t2))))
    let c1 = fp2Add(fp2Sub(fp2Mul(fp2Add(a.c0, a.c1), fp2Add(b.c0, b.c1)), fp2Add(t0, t1)),
                     fp2MulByNonResidue(t2))
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
    let c2 = fp2Sub(fp2Add(fp2Add(s1, s2), s3), fp2Add(s0, s4))

    return Fp6(c0: c0, c1: c1, c2: c2)
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

// (a0 + a1*w)(b0 + b1*w) = (a0*b0 + a1*b1*v) + (a0*b1 + a1*b0)*w
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

public func fp12Conjugate(_ a: Fp12) -> Fp12 {
    Fp12(c0: a.c0, c1: fp6Neg(a.c1))
}

// MARK: - Frobenius endomorphisms for BN254 Fp12

// Precomputed Frobenius coefficients (exact values)
// gamma_1_1 = xi^((p-1)/6)
private let bn254_gamma_1_1 = Fp2(
    c0: fpFromHex("0x1284b71c2865a7dfe8b99fdd76e68b605c521e08292f2176d60b35dadcc9e470"),
    c1: fpFromHex("0x246996f3b4fae7e6a6327cfe12150b8e747992778eeec7e5ca5cf05f80f362ac")
)

// gamma_1_2 = xi^((p-1)/3)
private let bn254_gamma_1_2 = Fp2(
    c0: fpFromHex("0x2fb347984f7911f74c0bec3cf559b143b78cc310c2c3330c99e39557176f553d"),
    c1: fpFromHex("0x16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2")
)

// gamma_1_3 = xi^((p-1)/2)
private let bn254_gamma_1_3 = Fp2(
    c0: fpFromHex("0x063cf305489af5dcdc5ec698b6e2f9b9dbaae0eda9c95998dc54014671a0135a"),
    c1: fpFromHex("0x07c03cbcac41049a0704b5a7ec796f2b21807dc98fa25bd282d37f632623b0e3")
)

// Frobenius^2 constants (all in Fp, c1=0)
private let bn254_gamma_2_1 = fpFromHex("0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd49")
private let bn254_gamma_2_2 = fpFromHex("0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48")
private let bn254_gamma_2_3 = fpFromHex("0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46")

// Frobenius^3 constants
private let bn254_gamma_3_1 = Fp2(
    c0: fpFromHex("0x19dc81cfcc82e4bbefe9608cd0acaa90894cb38dbe55d24ae86f7d391ed4a67f"),
    c1: fpFromHex("0x00abf8b60be77d7306cbeee33576139d7f03a5e397d439ec7694aa2bf4c0c101")
)

public func fp2Frobenius(_ a: Fp2) -> Fp2 { fp2Conjugate(a) }

// Public accessors for Frobenius constants (for testing)
public func bn254_gamma_1_2_pub() -> Fp2 { bn254_gamma_1_2 }
public func bn254_gamma_1_3_pub() -> Fp2 { bn254_gamma_1_3 }

/// Frobenius on Fp12
public func fp12Frobenius(_ a: Fp12) -> Fp12 {
    // Frobenius acts on Fp2 as conjugation, then multiply by gamma powers
    let a0c0 = fp2Conjugate(a.c0.c0)
    let a0c1 = fp2Mul(fp2Conjugate(a.c0.c1), bn254_gamma_1_2)
    let a0c2 = fp2Mul(fp2Conjugate(a.c0.c2), fp2Sqr(bn254_gamma_1_2))

    let a1c0 = fp2Mul(fp2Conjugate(a.c1.c0), bn254_gamma_1_1)
    let a1c1 = fp2Mul(fp2Conjugate(a.c1.c1), bn254_gamma_1_3)
    let a1c2 = fp2Mul(fp2Conjugate(a.c1.c2), fp2Mul(bn254_gamma_1_1, fp2Sqr(bn254_gamma_1_2)))

    return Fp12(c0: Fp6(c0: a0c0, c1: a0c1, c2: a0c2),
                c1: Fp6(c0: a1c0, c1: a1c1, c2: a1c2))
}

/// Frobenius squared
public func fp12Frobenius2(_ a: Fp12) -> Fp12 {
    // Frobenius^2 constants are in Fp, so just multiply by scalars
    let a0c0 = a.c0.c0
    let a0c1 = fp2MulByFp(a.c0.c1, bn254_gamma_2_2)
    let a0c2 = fp2MulByFp(a.c0.c2, fpMul(bn254_gamma_2_2, bn254_gamma_2_2))

    let a1c0 = fp2MulByFp(a.c1.c0, bn254_gamma_2_1)
    let a1c1 = fp2MulByFp(a.c1.c1, bn254_gamma_2_3)
    let a1c2 = fp2MulByFp(a.c1.c2, fpMul(bn254_gamma_2_1, fpMul(bn254_gamma_2_2, bn254_gamma_2_2)))

    return Fp12(c0: Fp6(c0: a0c0, c1: a0c1, c2: a0c2),
                c1: Fp6(c0: a1c0, c1: a1c1, c2: a1c2))
}

public func fp12Frobenius3(_ a: Fp12) -> Fp12 {
    fp12Frobenius(fp12Frobenius2(a))
}

// MARK: - G2 Point Types for BN254

public struct G2AffinePoint {
    public var x: Fp2
    public var y: Fp2
    public init(x: Fp2, y: Fp2) { self.x = x; self.y = y }
}

public struct G2ProjectivePoint {
    public var x: Fp2
    public var y: Fp2
    public var z: Fp2
    public init(x: Fp2, y: Fp2, z: Fp2) { self.x = x; self.y = y; self.z = z }
}

public func g2Identity() -> G2ProjectivePoint {
    G2ProjectivePoint(x: .one, y: .one, z: .zero)
}

public func g2IsIdentity(_ p: G2ProjectivePoint) -> Bool { p.z.isZero }

public func g2FromAffine(_ a: G2AffinePoint) -> G2ProjectivePoint {
    G2ProjectivePoint(x: a.x, y: a.y, z: .one)
}

public func g2Double(_ p: G2ProjectivePoint) -> G2ProjectivePoint {
    if g2IsIdentity(p) { return p }
    let a = fp2Sqr(p.x)
    let b = fp2Sqr(p.y)
    let c = fp2Sqr(b)
    let d = fp2Double(fp2Sub(fp2Sqr(fp2Add(p.x, b)), fp2Add(a, c)))
    let e = fp2Add(fp2Double(a), a)
    let f = fp2Sqr(e)
    let x3 = fp2Sub(f, fp2Double(d))
    let y3 = fp2Sub(fp2Mul(e, fp2Sub(d, x3)), fp2Double(fp2Double(fp2Double(c))))
    let z3 = fp2Sub(fp2Sqr(fp2Add(p.y, p.z)), fp2Add(b, fp2Sqr(p.z)))
    return G2ProjectivePoint(x: x3, y: y3, z: z3)
}

public func g2Add(_ p: G2ProjectivePoint, _ q: G2ProjectivePoint) -> G2ProjectivePoint {
    if g2IsIdentity(p) { return q }
    if g2IsIdentity(q) { return p }
    let z1z1 = fp2Sqr(p.z); let z2z2 = fp2Sqr(q.z)
    let u1 = fp2Mul(p.x, z2z2); let u2 = fp2Mul(q.x, z1z1)
    let s1 = fp2Mul(p.y, fp2Mul(q.z, z2z2))
    let s2 = fp2Mul(q.y, fp2Mul(p.z, z1z1))
    let h = fp2Sub(u2, u1); let r = fp2Double(fp2Sub(s2, s1))
    if h.isZero { if r.isZero { return g2Double(p) }; return g2Identity() }
    let i = fp2Sqr(fp2Double(h)); let j = fp2Mul(h, i); let vv = fp2Mul(u1, i)
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

public func g2ScalarMul(_ p: G2ProjectivePoint, _ scalar: [UInt64]) -> G2ProjectivePoint {
    var result = g2Identity(); var base = p
    for i in 0..<scalar.count {
        var word = scalar[i]
        for _ in 0..<64 {
            if word & 1 == 1 { result = g2Add(result, base) }
            base = g2Double(base); word >>= 1
        }
    }
    return result
}

public func g2ToAffine(_ p: G2ProjectivePoint) -> G2AffinePoint? {
    if g2IsIdentity(p) { return nil }
    let zinv = fp2Inverse(p.z); let zinv2 = fp2Sqr(zinv); let zinv3 = fp2Mul(zinv2, zinv)
    return G2AffinePoint(x: fp2Mul(p.x, zinv2), y: fp2Mul(p.y, zinv3))
}

// MARK: - BN254 Generators

public func bn254G1Generator() -> PointAffine {
    PointAffine(x: fpFromInt(1), y: fpFromInt(2))
}

/// BN254 G2 generator on the twist curve
/// Coordinates from EIP-197 / go-ethereum bn256
/// Fp2 convention: c0 + c1*u, matching gfP2{c0, c1} = c0 + c1*i
public func bn254G2Generator() -> G2AffinePoint {
    let x0 = fpFromHex("0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed")
    let x1 = fpFromHex("0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2")
    let y0 = fpFromHex("0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa")
    let y1 = fpFromHex("0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b")
    return G2AffinePoint(x: Fp2(c0: x0, c1: x1), y: Fp2(c0: y0, c1: y1))
}

// MARK: - Miller Loop (Affine G2 formulation)

// NAF representation of 6x+2 (MSB first), x = 4965661367071055936
private let sixXPlus2NAF: [Int8] = [
     1,  0, -1,  0,  1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1,  0,
     1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0, -1,  0,
     1,  0,  0, -1,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0, -1,  0,
    -1,  0,  0,  1,  0,  0,  0, -1,  0,  0, -1,  0,  1,  0,  1,  0,
     0,  0
]

// Use affine coordinates for T in the Miller loop for simplicity and correctness.
// This is slower than projective but easier to verify.

/// Affine doubling on the twist: T = 2T, returns slope lambda
private func g2AffineDouble(_ t: inout G2AffinePoint) -> Fp2 {
    // lambda = 3*x^2 / (2*y)  (a=0 for BN254 twist)
    let xsq = fp2Sqr(t.x)
    let num = fp2Add(fp2Double(xsq), xsq)  // 3*x^2
    let den = fp2Double(t.y)                // 2*y
    let lam = fp2Mul(num, fp2Inverse(den))
    let x3 = fp2Sub(fp2Sqr(lam), fp2Double(t.x))
    let y3 = fp2Sub(fp2Mul(lam, fp2Sub(t.x, x3)), t.y)
    t = G2AffinePoint(x: x3, y: y3)
    return lam
}

/// Affine addition on the twist: T = T + Q, returns slope lambda
private func g2AffineAdd(_ t: inout G2AffinePoint, _ q: G2AffinePoint) -> Fp2 {
    let dx = fp2Sub(q.x, t.x)
    let dy = fp2Sub(q.y, t.y)
    let lam = fp2Mul(dy, fp2Inverse(dx))
    let x3 = fp2Sub(fp2Sub(fp2Sqr(lam), t.x), q.x)
    let y3 = fp2Sub(fp2Mul(lam, fp2Sub(t.x, x3)), t.y)
    t = G2AffinePoint(x: x3, y: y3)
    return lam
}

/// Line evaluation at G1 point P for a line on the twist with slope lambda through (xT, yT).
///
/// BN254 uses an M-type sextic twist. The twist map is psi: E' -> E via
///   (x', y') -> (x' * w^2, y' * w^3)   where w^6 = xi = 9+u
/// The line on E' through T with slope lambda is:
///   y' - yT = lambda * (x' - xT)
/// Pulling back via psi^{-1}(P) where P = (xP, yP) in G1:
///   xP lives in Fp, yP lives in Fp, but we embed into Fp12 via the twist:
///   yP/w^3 - yT = lambda * (xP/w^2 - xT)
/// Multiply through by w^3:
///   yP - yT*w^3 = lambda*xP*w - lambda*xT*w^3
///   yP - lambda*xP*w + (lambda*xT - yT)*w^3
///
/// Tower: Fp12 = Fp6[w]/(w^2 - v), Fp6 = Fp2[v]/(v^3 - xi)
///   w^2 = v, w^3 = vw
/// Positions in the 6-basis {1, v, v^2, w, vw, v^2w}:
///   w^0 = 1     -> c0.c0  (position 0)
///   w^1 = w     -> c1.c0  (position 3)
///   w^3 = vw    -> c1.c1  (position 4)
///
/// Result: yP at c0.c0, -lam*xP at c1.c0, (lam*xT - yT) at c1.c1
/// This matches gnark's MulBy034 sparse structure.
private func lineEval(lambda: Fp2, xT: Fp2, yT: Fp2, p: PointAffine) -> Fp12 {
    let yPfp2 = Fp2(c0: p.y, c1: .zero)                    // yP at position 0 -> c0.c0
    let negLamXP = fp2Neg(fp2MulByFp(lambda, p.x))         // -lam*xP at position 3 -> c1.c0
    let lamXT_yT = fp2Sub(fp2Mul(lambda, xT), yT)          // (lam*xT - yT) at position 4 -> c1.c1
    return Fp12(c0: Fp6(c0: yPfp2, c1: .zero, c2: .zero),
                c1: Fp6(c0: negLamXP, c1: lamXT_yT, c2: .zero))
}

/// Compute the optimal ate Miller loop for BN254.
public func bn254MillerLoop(_ p: PointAffine, _ q: G2AffinePoint) -> Fp12 {
    var tPt = q
    var f = Fp12.one
    let negQ = g2NegateAffine(q)

    for i in 1..<sixXPlus2NAF.count {
        f = fp12Sqr(f)

        let oldT = tPt
        let lam = g2AffineDouble(&tPt)
        let lineD = lineEval(lambda: lam, xT: oldT.x, yT: oldT.y, p: p)
        f = fp12Mul(f, lineD)

        if sixXPlus2NAF[i] == 1 {
            let oldT2 = tPt
            let lam2 = g2AffineAdd(&tPt, q)
            let lineA = lineEval(lambda: lam2, xT: oldT2.x, yT: oldT2.y, p: p)
            f = fp12Mul(f, lineA)
        } else if sixXPlus2NAF[i] == -1 {
            let oldT2 = tPt
            let lam2 = g2AffineAdd(&tPt, negQ)
            let lineA = lineEval(lambda: lam2, xT: oldT2.x, yT: oldT2.y, p: p)
            f = fp12Mul(f, lineA)
        }
    }

    // Frobenius correction: Q1 = pi(Q), Q2 = -pi^2(Q)
    let q1x = fp2Mul(fp2Conjugate(q.x), bn254_gamma_1_2)
    let q1y = fp2Mul(fp2Conjugate(q.y), bn254_gamma_1_3)
    let q1 = G2AffinePoint(x: q1x, y: q1y)

    let oldT3 = tPt
    let lam3 = g2AffineAdd(&tPt, q1)
    let lineQ1 = lineEval(lambda: lam3, xT: oldT3.x, yT: oldT3.y, p: p)
    f = fp12Mul(f, lineQ1)

    let q2x = fp2MulByFp(q.x, bn254_gamma_2_2)
    let q2y = fp2Neg(fp2MulByFp(q.y, bn254_gamma_2_3))
    let q2 = G2AffinePoint(x: q2x, y: q2y)

    let oldT4 = tPt
    let lam4 = g2AffineAdd(&tPt, q2)
    let lineQ2 = lineEval(lambda: lam4, xT: oldT4.x, yT: oldT4.y, p: p)
    f = fp12Mul(f, lineQ2)

    return f
}

// MARK: - Final Exponentiation (C-accelerated with cyclotomic squaring)

/// Marshal Fp12 from Swift (8x32 Montgomery per Fp) to C (4x64 Montgomery per Fp, 48 UInt64 total).
/// Layout: c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
///         c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1
private func fp12ToFlat64(_ f: Fp12) -> [UInt64] {
    var flat = [UInt64](repeating: 0, count: 48)
    let fps: [Fp] = [
        f.c0.c0.c0, f.c0.c0.c1, f.c0.c1.c0, f.c0.c1.c1, f.c0.c2.c0, f.c0.c2.c1,
        f.c1.c0.c0, f.c1.c0.c1, f.c1.c1.c0, f.c1.c1.c1, f.c1.c2.c0, f.c1.c2.c1
    ]
    for i in 0..<12 {
        let limbs64 = fps[i].to64()
        flat[i * 4]     = limbs64[0]
        flat[i * 4 + 1] = limbs64[1]
        flat[i * 4 + 2] = limbs64[2]
        flat[i * 4 + 3] = limbs64[3]
    }
    return flat
}

/// Unmarshal Fp12 from C (48 UInt64) back to Swift.
private func fp12FromFlat64(_ flat: [UInt64]) -> Fp12 {
    func readFp(_ offset: Int) -> Fp {
        Fp.from64([flat[offset], flat[offset + 1], flat[offset + 2], flat[offset + 3]])
    }
    func readFp2(_ offset: Int) -> Fp2 {
        Fp2(c0: readFp(offset), c1: readFp(offset + 4))
    }
    func readFp6(_ offset: Int) -> Fp6 {
        Fp6(c0: readFp2(offset), c1: readFp2(offset + 8), c2: readFp2(offset + 16))
    }
    return Fp12(c0: readFp6(0), c1: readFp6(24))
}

/// Final exponentiation: f^((p^12 - 1) / r) — C accelerated with cyclotomic squaring.
/// Uses NEON-optimized C implementation with Granger-Scott cyclotomic squaring in the hard part,
/// which is ~3x faster than generic Fp12 squaring for elements in the cyclotomic subgroup.
public func bn254FinalExponentiation(_ f: Fp12) -> Fp12 {
    var fFlat = fp12ToFlat64(f)
    var result = [UInt64](repeating: 0, count: 48)
    bn254_final_exp(&fFlat, &result)
    return fp12FromFlat64(result)
}

// MARK: - Public Pairing API

public func bn254Pairing(_ p: PointAffine, _ q: G2AffinePoint) -> Fp12 {
    bn254FinalExponentiation(bn254MillerLoop(p, q))
}

public func bn254PairingCheck(_ pairs: [(PointAffine, G2AffinePoint)]) -> Bool {
    var f = Fp12.one
    for (p, q) in pairs {
        f = fp12Mul(f, bn254MillerLoop(p, q))
    }
    return fp12Equal(bn254FinalExponentiation(f), .one)
}

public func fp12Equal(_ a: Fp12, _ b: Fp12) -> Bool {
    let d = fp12Sub(a, b)
    return d.c0.c0.c0.isZero && d.c0.c0.c1.isZero &&
           d.c0.c1.c0.isZero && d.c0.c1.c1.isZero &&
           d.c0.c2.c0.isZero && d.c0.c2.c1.isZero &&
           d.c1.c0.c0.isZero && d.c1.c0.c1.isZero &&
           d.c1.c1.c0.isZero && d.c1.c1.c1.isZero &&
           d.c1.c2.c0.isZero && d.c1.c2.c1.isZero
}

// MARK: - C-Accelerated BN254 Pairing

/// Pack a G1 affine point (8x UInt32 limbs) into 4x UInt64 layout for C code
private func packG1Affine(_ p: PointAffine) -> [UInt64] {
    let xl = p.x.to64()
    let yl = p.y.to64()
    return xl + yl  // 8 x UInt64
}

/// Pack a G2 affine point (Fp2 coords, each 8x UInt32) into 8x UInt64 layout
private func packG2Affine(_ q: G2AffinePoint) -> [UInt64] {
    let xc0 = q.x.c0.to64()
    let xc1 = q.x.c1.to64()
    let yc0 = q.y.c0.to64()
    let yc1 = q.y.c1.to64()
    return xc0 + xc1 + yc0 + yc1  // 16 x UInt64
}

/// C-accelerated BN254 pairing check: verify prod_i e(P_i, Q_i) = 1
public func cBN254PairingCheck(_ pairs: [(PointAffine, G2AffinePoint)]) -> Bool {
    let n = pairs.count
    if n == 0 { return true }

    var buffer = [UInt64]()
    buffer.reserveCapacity(n * 24)
    for (p, q) in pairs {
        buffer.append(contentsOf: packG1Affine(p))
        buffer.append(contentsOf: packG2Affine(q))
    }

    return buffer.withUnsafeBufferPointer { ptr in
        bn254_pairing_check(ptr.baseAddress!, Int32(n)) == 1
    }
}

/// C-accelerated BN254 pairing check with precomputed G2 lines
public func cBN254PairingCheckPrecomp(_ pairs: [(PointAffine, G2AffinePoint)]) -> Bool {
    let n = pairs.count
    if n == 0 { return true }

    var buffer = [UInt64]()
    buffer.reserveCapacity(n * 24)
    for (p, q) in pairs {
        buffer.append(contentsOf: packG1Affine(p))
        buffer.append(contentsOf: packG2Affine(q))
    }

    return buffer.withUnsafeBufferPointer { ptr in
        bn254_pairing_check_precomp(ptr.baseAddress!, Int32(n)) == 1
    }
}

/// C-accelerated single BN254 pairing: e(P, Q)
public func cBN254Pairing(_ p: PointAffine, _ q: G2AffinePoint) -> [UInt64] {
    var pBuf = packG1Affine(p)
    var qBuf = packG2Affine(q)
    var result = [UInt64](repeating: 0, count: 48)
    pBuf.withUnsafeBufferPointer { pp in
        qBuf.withUnsafeBufferPointer { qp in
            result.withUnsafeMutableBufferPointer { rp in
                bn254_pairing(pp.baseAddress!, qp.baseAddress!, rp.baseAddress!)
            }
        }
    }
    return result
}
