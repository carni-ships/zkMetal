// BLS12-381 base field Fp arithmetic (CPU-side)
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// 381-bit prime, field elements as 12x32-bit limbs in Montgomery form (little-endian).

import Foundation

public struct Fp381 {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                   UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    // p in 6x64-bit limbs (little-endian)
    public static let P: [UInt64] = [
        0xb9feffffffffaaab, 0x1eabfffeb153ffff,
        0x6730d2a0f6b0f624, 0x64774b84f38512bf,
        0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a
    ]

    // R mod p (Montgomery form of 1): 2^384 mod p
    public static let R_MOD_P: [UInt64] = [
        0x760900000002fffd, 0xebf4000bc40c0002,
        0x5f48985753c758ba, 0x77ce585370525745,
        0x5c071a97a256ec6d, 0x15f65ec3fa80e493
    ]

    // R^2 mod p: 2^768 mod p
    public static let R2_MOD_P: [UInt64] = [
        0xf4df1f341c341746, 0x0a76e6a609d104f1,
        0x8de5476c4c95b6d5, 0x67eb88a9939d83c0,
        0x9a793e85b519952d, 0x11988fe592cae3aa
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0x89f3fffcfffcfffd

    public static var zero: Fp381 {
        Fp381(v: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    }

    public static var one: Fp381 {
        // R mod p in 32-bit limbs (little-endian)
        Fp381(v: (0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
                  0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
                  0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3))
    }

    public init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                     UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    // Convert to 6x64-bit limbs for arithmetic
    public func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8, v.9, v.10, v.11]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
            UInt64(l[8]) | (UInt64(l[9]) << 32),
            UInt64(l[10]) | (UInt64(l[11]) << 32),
        ]
    }

    public static func from64(_ limbs: [UInt64]) -> Fp381 {
        Fp381(v: (
            UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
            UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
            UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
            UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32),
            UInt32(limbs[4] & 0xFFFFFFFF), UInt32(limbs[4] >> 32),
            UInt32(limbs[5] & 0xFFFFFFFF), UInt32(limbs[5] >> 32)
        ))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0 &&
        v.4 == 0 && v.5 == 0 && v.6 == 0 && v.7 == 0 &&
        v.8 == 0 && v.9 == 0 && v.10 == 0 && v.11 == 0
    }
}

// MARK: - Field Operations

// Montgomery multiplication: (a * b * R^-1) mod p
public func fp381Mul(_ a: Fp381, _ b: Fp381) -> Fp381 {
    let al = a.to64(), bl = b.to64()
    var t = [UInt64](repeating: 0, count: 7)

    for i in 0..<6 {
        var carry: UInt64 = 0
        for j in 0..<6 {
            let (hi, lo) = al[i].multipliedFullWidth(by: bl[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[6] = t[6] &+ carry

        let m = t[0] &* Fp381.INV
        carry = 0
        for j in 0..<6 {
            let (hi, lo) = m.multipliedFullWidth(by: Fp381.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[6] = t[6] &+ carry

        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = t[5]; t[5] = t[6]; t[6] = 0
    }

    var r = Array(t[0..<6])
    if gte384(r, Fp381.P) {
        (r, _) = sub384(r, Fp381.P)
    }
    return Fp381.from64(r)
}

public func fp381Add(_ a: Fp381, _ b: Fp381) -> Fp381 {
    var (r, carry) = add384(a.to64(), b.to64())
    if carry != 0 || gte384(r, Fp381.P) {
        (r, _) = sub384(r, Fp381.P)
    }
    return Fp381.from64(r)
}

public func fp381Sub(_ a: Fp381, _ b: Fp381) -> Fp381 {
    var (r, borrow) = sub384(a.to64(), b.to64())
    if borrow {
        (r, _) = add384(r, Fp381.P)
    }
    return Fp381.from64(r)
}

public func fp381Sqr(_ a: Fp381) -> Fp381 { fp381Mul(a, a) }
public func fp381Double(_ a: Fp381) -> Fp381 { fp381Add(a, a) }

// Convert integer to Montgomery form
public func fp381FromInt(_ val: UInt64) -> Fp381 {
    let limbs: [UInt64] = [val, 0, 0, 0, 0, 0]
    let raw = Fp381.from64(limbs)
    return fp381Mul(raw, Fp381.from64(Fp381.R2_MOD_P))
}

// Convert from Montgomery form to integer
public func fp381ToInt(_ a: Fp381) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0, 0, 0]
    return fp381Mul(a, Fp381.from64(one)).to64()
}

// Field negation
public func fp381Neg(_ a: Fp381) -> Fp381 {
    if a.isZero { return a }
    let (r, _) = sub384(Fp381.P, a.to64())
    return Fp381.from64(r)
}

// Field inverse via Fermat's little theorem: a^(p-2) mod p
public func fp381Inverse(_ a: Fp381) -> Fp381 {
    var result = Fp381.one
    var base = a
    var exp = Fp381.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<6 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = fp381Mul(result, base)
            }
            base = fp381Sqr(base)
            word >>= 1
        }
    }
    return result
}

/// Parse a hex string into Fp381 in Montgomery form.
public func fp381FromHex(_ hex: String) -> Fp381 {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 96 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0, 0, 0]
    for i in 0..<6 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[5 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = Fp381.from64(limbs)
    return fp381Mul(raw, Fp381.from64(Fp381.R2_MOD_P))
}

/// Convert Fp381 (Montgomery form) to hex string.
public func fp381ToHex(_ a: Fp381) -> String {
    let limbs = fp381ToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}

// MARK: - Fp2 = Fp[u]/(u^2 + 1)

/// Fp2 element: c0 + c1 * u where u^2 = -1
public struct Fp2_381 {
    public var c0: Fp381
    public var c1: Fp381

    public init(c0: Fp381, c1: Fp381) {
        self.c0 = c0
        self.c1 = c1
    }

    public static var zero: Fp2_381 { Fp2_381(c0: .zero, c1: .zero) }
    public static var one: Fp2_381 { Fp2_381(c0: .one, c1: .zero) }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

/// Fp2 addition
public func fp2_381Add(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: fp381Add(a.c0, b.c0), c1: fp381Add(a.c1, b.c1))
}

/// Fp2 subtraction
public func fp2_381Sub(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: fp381Sub(a.c0, b.c0), c1: fp381Sub(a.c1, b.c1))
}

/// Fp2 negation
public func fp2_381Neg(_ a: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: fp381Neg(a.c0), c1: fp381Neg(a.c1))
}

/// Fp2 doubling
public func fp2_381Double(_ a: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: fp381Double(a.c0), c1: fp381Double(a.c1))
}

/// Fp2 multiplication: (a0 + a1*u)(b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
public func fp2_381Mul(_ a: Fp2_381, _ b: Fp2_381) -> Fp2_381 {
    let a0b0 = fp381Mul(a.c0, b.c0)
    let a1b1 = fp381Mul(a.c1, b.c1)
    // Karatsuba: (a0+a1)(b0+b1) - a0b0 - a1b1 = a0b1 + a1b0
    let t = fp381Mul(fp381Add(a.c0, a.c1), fp381Add(b.c0, b.c1))
    return Fp2_381(
        c0: fp381Sub(a0b0, a1b1),
        c1: fp381Sub(fp381Sub(t, a0b0), a1b1)
    )
}

/// Fp2 squaring: (a0 + a1*u)^2 = (a0^2 - a1^2) + 2*a0*a1*u
/// Using complex squaring: (a0+a1)(a0-a1) and 2*a0*a1
public func fp2_381Sqr(_ a: Fp2_381) -> Fp2_381 {
    let v0 = fp381Mul(a.c0, a.c1)
    let c0 = fp381Mul(fp381Add(a.c0, a.c1), fp381Sub(a.c0, a.c1))
    let c1 = fp381Double(v0)
    return Fp2_381(c0: c0, c1: c1)
}

/// Fp2 conjugation: (a0 + a1*u)* = a0 - a1*u
public func fp2_381Conjugate(_ a: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: a.c0, c1: fp381Neg(a.c1))
}

/// Fp2 inverse: 1/(a0 + a1*u) = (a0 - a1*u) / (a0^2 + a1^2)
public func fp2_381Inverse(_ a: Fp2_381) -> Fp2_381 {
    let norm = fp381Add(fp381Sqr(a.c0), fp381Sqr(a.c1))
    let normInv = fp381Inverse(norm)
    return Fp2_381(c0: fp381Mul(a.c0, normInv), c1: fp381Neg(fp381Mul(a.c1, normInv)))
}

/// Multiply Fp2 element by Fp element
public func fp2_381MulByFp(_ a: Fp2_381, _ b: Fp381) -> Fp2_381 {
    Fp2_381(c0: fp381Mul(a.c0, b), c1: fp381Mul(a.c1, b))
}

/// Multiply by the non-residue (1 + u) for the Fp6 tower
/// (a0 + a1*u)(1 + u) = (a0 - a1) + (a0 + a1)*u
public func fp2_381MulByNonResidue(_ a: Fp2_381) -> Fp2_381 {
    Fp2_381(c0: fp381Sub(a.c0, a.c1), c1: fp381Add(a.c0, a.c1))
}

// MARK: - Fp6 = Fp2[v]/(v^3 - (1+u))

/// Fp6 element: c0 + c1*v + c2*v^2 where v^3 = 1+u
public struct Fp6_381 {
    public var c0: Fp2_381
    public var c1: Fp2_381
    public var c2: Fp2_381

    public init(c0: Fp2_381, c1: Fp2_381, c2: Fp2_381) {
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
    }

    public static var zero: Fp6_381 { Fp6_381(c0: .zero, c1: .zero, c2: .zero) }
    public static var one: Fp6_381 { Fp6_381(c0: .one, c1: .zero, c2: .zero) }

    public var isZero: Bool { c0.isZero && c1.isZero && c2.isZero }
}

/// Fp6 addition
public func fp6_381Add(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 {
    Fp6_381(c0: fp2_381Add(a.c0, b.c0),
            c1: fp2_381Add(a.c1, b.c1),
            c2: fp2_381Add(a.c2, b.c2))
}

/// Fp6 subtraction
public func fp6_381Sub(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 {
    Fp6_381(c0: fp2_381Sub(a.c0, b.c0),
            c1: fp2_381Sub(a.c1, b.c1),
            c2: fp2_381Sub(a.c2, b.c2))
}

/// Fp6 negation
public func fp6_381Neg(_ a: Fp6_381) -> Fp6_381 {
    Fp6_381(c0: fp2_381Neg(a.c0), c1: fp2_381Neg(a.c1), c2: fp2_381Neg(a.c2))
}

/// Fp6 multiplication using Karatsuba
public func fp6_381Mul(_ a: Fp6_381, _ b: Fp6_381) -> Fp6_381 {
    let v0 = fp2_381Mul(a.c0, b.c0)
    let v1 = fp2_381Mul(a.c1, b.c1)
    let v2 = fp2_381Mul(a.c2, b.c2)

    // c0 = v0 + xi * ((a1+a2)(b1+b2) - v1 - v2)
    let t0 = fp2_381MulByNonResidue(
        fp2_381Sub(fp2_381Sub(
            fp2_381Mul(fp2_381Add(a.c1, a.c2), fp2_381Add(b.c1, b.c2)),
            v1), v2))
    let c0 = fp2_381Add(v0, t0)

    // c1 = (a0+a1)(b0+b1) - v0 - v1 + xi*v2
    let t1 = fp2_381Sub(fp2_381Sub(
        fp2_381Mul(fp2_381Add(a.c0, a.c1), fp2_381Add(b.c0, b.c1)),
        v0), v1)
    let c1 = fp2_381Add(t1, fp2_381MulByNonResidue(v2))

    // c2 = (a0+a2)(b0+b2) - v0 - v2 + v1
    let t2 = fp2_381Sub(fp2_381Sub(
        fp2_381Mul(fp2_381Add(a.c0, a.c2), fp2_381Add(b.c0, b.c2)),
        v0), v2)
    let c2 = fp2_381Add(t2, v1)

    return Fp6_381(c0: c0, c1: c1, c2: c2)
}

/// Fp6 squaring
public func fp6_381Sqr(_ a: Fp6_381) -> Fp6_381 {
    let s0 = fp2_381Sqr(a.c0)
    let ab = fp2_381Mul(a.c0, a.c1)
    let s1 = fp2_381Double(ab)
    let s2 = fp2_381Sqr(fp2_381Sub(fp2_381Add(a.c0, a.c2), a.c1))
    let bc = fp2_381Mul(a.c1, a.c2)
    let s3 = fp2_381Double(bc)
    let s4 = fp2_381Sqr(a.c2)

    let c0 = fp2_381Add(s0, fp2_381MulByNonResidue(s3))
    let c1 = fp2_381Add(s1, fp2_381MulByNonResidue(s4))
    let c2 = fp2_381Add(fp2_381Add(fp2_381Add(s1, s2), s3), fp2_381Sub(fp381_fp2Zero(), s0))
    // c2 = s1 + s2 + s3 - s0 - s4
    let c2final = fp2_381Sub(fp2_381Add(fp2_381Add(fp2_381Add(s1, s2), s3), fp2_381Neg(s0)), s4)

    return Fp6_381(c0: c0, c1: c1, c2: c2final)
}

private func fp381_fp2Zero() -> Fp2_381 { .zero }

/// Fp6 inverse
public func fp6_381Inverse(_ a: Fp6_381) -> Fp6_381 {
    let c0s = fp2_381Sqr(a.c0)
    let c1s = fp2_381Sqr(a.c1)
    let c2s = fp2_381Sqr(a.c2)
    let c01 = fp2_381Mul(a.c0, a.c1)
    let c02 = fp2_381Mul(a.c0, a.c2)
    let c12 = fp2_381Mul(a.c1, a.c2)

    // A = c0^2 - xi*c1*c2
    let capA = fp2_381Sub(c0s, fp2_381MulByNonResidue(c12))
    // B = xi*c2^2 - c0*c1
    let capB = fp2_381Sub(fp2_381MulByNonResidue(c2s), c01)
    // C = c1^2 - c0*c2
    let capC = fp2_381Sub(c1s, c02)

    // F = c0*A + xi*(c2*B + c1*C)
    let capF = fp2_381Add(
        fp2_381Mul(a.c0, capA),
        fp2_381MulByNonResidue(fp2_381Add(
            fp2_381Mul(a.c2, capB),
            fp2_381Mul(a.c1, capC))))
    let fInv = fp2_381Inverse(capF)

    return Fp6_381(
        c0: fp2_381Mul(capA, fInv),
        c1: fp2_381Mul(capB, fInv),
        c2: fp2_381Mul(capC, fInv))
}

/// Multiply Fp6 by scalar in Fp2 (sparse: only c0 component)
public func fp6_381MulByFp2(_ a: Fp6_381, _ b: Fp2_381) -> Fp6_381 {
    Fp6_381(c0: fp2_381Mul(a.c0, b), c1: fp2_381Mul(a.c1, b), c2: fp2_381Mul(a.c2, b))
}

// MARK: - Fp12 = Fp6[w]/(w^2 - v)

/// Fp12 element: c0 + c1*w where w^2 = v
public struct Fp12_381 {
    public var c0: Fp6_381
    public var c1: Fp6_381

    public init(c0: Fp6_381, c1: Fp6_381) {
        self.c0 = c0
        self.c1 = c1
    }

    public static var zero: Fp12_381 { Fp12_381(c0: .zero, c1: .zero) }
    public static var one: Fp12_381 { Fp12_381(c0: .one, c1: .zero) }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

/// Helper: multiply Fp6 by v (shift the tower)
/// v * (a0 + a1*v + a2*v^2) = a2*xi + a0*v + a1*v^2
private func fp6_381MulByV(_ a: Fp6_381) -> Fp6_381 {
    Fp6_381(c0: fp2_381MulByNonResidue(a.c2), c1: a.c0, c2: a.c1)
}

/// Fp12 addition
public func fp12_381Add(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 {
    Fp12_381(c0: fp6_381Add(a.c0, b.c0), c1: fp6_381Add(a.c1, b.c1))
}

/// Fp12 subtraction
public func fp12_381Sub(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 {
    Fp12_381(c0: fp6_381Sub(a.c0, b.c0), c1: fp6_381Sub(a.c1, b.c1))
}

/// Fp12 negation
public func fp12_381Neg(_ a: Fp12_381) -> Fp12_381 {
    Fp12_381(c0: fp6_381Neg(a.c0), c1: fp6_381Neg(a.c1))
}

/// Fp12 multiplication: (a0+a1*w)(b0+b1*w) = (a0*b0 + a1*b1*v) + (a0*b1 + a1*b0)*w
public func fp12_381Mul(_ a: Fp12_381, _ b: Fp12_381) -> Fp12_381 {
    let t0 = fp6_381Mul(a.c0, b.c0)
    let t1 = fp6_381Mul(a.c1, b.c1)

    let c0 = fp6_381Add(t0, fp6_381MulByV(t1))
    let c1 = fp6_381Sub(fp6_381Sub(
        fp6_381Mul(fp6_381Add(a.c0, a.c1), fp6_381Add(b.c0, b.c1)),
        t0), t1)

    return Fp12_381(c0: c0, c1: c1)
}

/// Fp12 squaring
public func fp12_381Sqr(_ a: Fp12_381) -> Fp12_381 {
    let ab = fp6_381Mul(a.c0, a.c1)
    let c0 = fp6_381Add(
        fp6_381Mul(fp6_381Add(a.c0, a.c1), fp6_381Add(a.c0, fp6_381MulByV(a.c1))),
        fp6_381Neg(fp6_381Add(ab, fp6_381MulByV(ab))))
    let c1 = fp6_381Add(ab, ab)
    return Fp12_381(c0: c0, c1: c1)
}

/// Fp12 inverse
public func fp12_381Inverse(_ a: Fp12_381) -> Fp12_381 {
    // 1/(a0 + a1*w) = (a0 - a1*w) / (a0^2 - a1^2*v)
    let t0 = fp6_381Sqr(a.c0)
    let t1 = fp6_381Sqr(a.c1)
    let denom = fp6_381Sub(t0, fp6_381MulByV(t1))
    let denomInv = fp6_381Inverse(denom)

    return Fp12_381(
        c0: fp6_381Mul(a.c0, denomInv),
        c1: fp6_381Neg(fp6_381Mul(a.c1, denomInv)))
}

/// Fp12 conjugation (unitary inverse for elements on cyclotomic subgroup)
public func fp12_381Conjugate(_ a: Fp12_381) -> Fp12_381 {
    Fp12_381(c0: a.c0, c1: fp6_381Neg(a.c1))
}

// MARK: - Fp2 Exponentiation (needed for Frobenius computation)

/// Exponentiate Fp2 element by a big integer given as [UInt64] limbs (little-endian).
private func fp2_381Pow(_ base: Fp2_381, _ exp: [UInt64]) -> Fp2_381 {
    var result = Fp2_381.one
    var b = base
    for limb in exp {
        var w = limb
        for _ in 0..<64 {
            if w & 1 == 1 {
                result = fp2_381Mul(result, b)
            }
            b = fp2_381Sqr(b)
            w >>= 1
        }
    }
    return result
}

// MARK: - Frobenius Coefficients for BLS12-381
// Computed at first use from (1+u)^((p^i-1)/k) using Fp2 exponentiation.
// This avoids hardcoded constants that are error-prone.

/// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
/// (p-1)/3 in little-endian 64-bit limbs
private let pMinus1Over3: [UInt64] = {
    // (p-1) = p - 1
    // p-1 = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa
    // (p-1)/3 = 0x08b0a9e4b08d2538d6090b44ed80e6e2c01e1d4c07ba5aa8cee1d4cc9c3274fe0dab943d0b12f8a2e2eb92f03f5f69c87600000015555
    // Computed: (p-1)/3
    var pM1: [UInt64] = Fp381.P
    pM1[0] -= 1  // p-1
    // Divide by 3
    return div384by3(pM1)
}()

/// Divide a 384-bit number (6 x UInt64, little-endian) by 3.
/// Assumes the number is divisible by 3.
private func div384by3(_ a: [UInt64]) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: 6)
    var remainder: UInt64 = 0
    for i in stride(from: 5, through: 0, by: -1) {
        // We need to divide (remainder * 2^64 + a[i]) by 3.
        // remainder is at most 2 (since we're dividing by 3).
        // Use bit-by-bit division to avoid needing 128-bit arithmetic.
        var word: UInt64 = 0
        var r = remainder
        for bit in stride(from: 63, through: 0, by: -1) {
            r = r << 1 | ((a[i] >> bit) & 1)
            if r >= 3 {
                word |= 1 << bit
                r -= 3
            }
        }
        result[i] = word
        remainder = r
    }
    return result
}

/// Compute (p-1)/6 as [UInt64] limbs (little-endian)
private let pMinus1Over6: [UInt64] = {
    div384by3(div384by2(pMinus1()))
}()

private func pMinus1() -> [UInt64] {
    var result = Fp381.P
    result[0] -= 1
    return result
}

private func div384by2(_ a: [UInt64]) -> [UInt64] {
    var result = [UInt64](repeating: 0, count: 6)
    for i in 0..<6 {
        result[i] = a[i] >> 1
        if i < 5 {
            result[i] |= (a[i + 1] & 1) << 63
        }
    }
    return result
}

/// The non-residue (1 + u) in Fp2
private let fp2_381NonResidue = Fp2_381(c0: .one, c1: .one)

/// Frobenius coefficients, computed at first use.
/// gamma_{1,1} = (1+u)^((p-1)/3)
private let FROBENIUS_COEFF_FP6_C1_1: Fp2_381 = fp2_381Pow(fp2_381NonResidue, pMinus1Over3)
/// gamma_{2,1} = (1+u)^(2(p-1)/3) = gamma_{1,1}^2
private let FROBENIUS_COEFF_FP6_C2_1: Fp2_381 = fp2_381Sqr(FROBENIUS_COEFF_FP6_C1_1)

/// For Frobenius^2: (1+u)^((p^2-1)/3)
/// = (1+u)^((p-1)(p+1)/3) = [(1+u)^((p-1)/3)]^(p+1) = gamma_{1,1}^p * gamma_{1,1}
/// Since Frobenius on Fp2 is conjugation: gamma_{1,1}^p = conj(gamma_{1,1})
private let FROBENIUS_COEFF_FP6_C1_2: Fp2_381 = fp2_381Mul(fp2_381Conjugate(FROBENIUS_COEFF_FP6_C1_1), FROBENIUS_COEFF_FP6_C1_1)
/// (1+u)^(2(p^2-1)/3) = C1_2^2
private let FROBENIUS_COEFF_FP6_C2_2: Fp2_381 = fp2_381Sqr(FROBENIUS_COEFF_FP6_C1_2)

/// For Frobenius^3: (1+u)^((p^3-1)/3)
/// = (1+u)^((p-1)(p^2+p+1)/3) = [(1+u)^((p-1)/3)]^(p^2+p+1)
/// = gamma_{1,1}^(p^2) * gamma_{1,1}^p * gamma_{1,1}
/// = conj(conj(gamma_{1,1})) * conj(gamma_{1,1}) * gamma_{1,1}  [since frob^2 on Fp2 = identity]
/// = gamma_{1,1} * conj(gamma_{1,1}) * gamma_{1,1}
/// = |gamma_{1,1}|^2 * gamma_{1,1}  ... hmm, this isn't right
/// Actually: gamma_{1,1}^(p^2) = gamma_{1,1} (since p^2 acts as identity on Fp2)
/// So: C1_3 = gamma_{1,1}^(p^2+p+1) = gamma_{1,1} * conj(gamma_{1,1}) * gamma_{1,1}
///         = |gamma_{1,1}|^2 * gamma_{1,1}
/// Wait, p^2 on Fp2 IS identity (since [Fp2:Fp]=2 and p^2 ≡ 1 on Fp2).
/// So: C1_3 = gamma_{1,1}^(1+p+1) = gamma_{1,1}^(p+2)  ... no, (p^3-1)/3 = (p-1)(p^2+p+1)/3
/// Since (p^2+p+1) = (p^2+p+1), and we know gamma_{1,1}^(p-1) is in Fp (norm of gamma_{1,1}),
/// this gets complicated. Let me just compute it directly.
private let FROBENIUS_COEFF_FP6_C1_3: Fp2_381 = {
    // (1+u)^((p^3-1)/3) = [(1+u)^((p-1)/3)]^(p^2+p+1)
    let g = FROBENIUS_COEFF_FP6_C1_1
    let gp = fp2_381Conjugate(g)  // g^p
    let gp2 = g  // g^(p^2) = g since p^2 acts as identity on Fp2
    return fp2_381Mul(fp2_381Mul(gp2, gp), g)
}()
private let FROBENIUS_COEFF_FP6_C2_3: Fp2_381 = fp2_381Sqr(FROBENIUS_COEFF_FP6_C1_3)

/// For Fp12 Frobenius: (1+u)^((p-1)/6)
private let FROBENIUS_COEFF_FP12_C1_1: Fp2_381 = fp2_381Pow(fp2_381NonResidue, pMinus1Over6)
/// (1+u)^((p^2-1)/6) = [(1+u)^((p-1)/6)]^(p+1) = conj(C12_1) * C12_1
private let FROBENIUS_COEFF_FP12_C1_2: Fp2_381 = fp2_381Mul(fp2_381Conjugate(FROBENIUS_COEFF_FP12_C1_1), FROBENIUS_COEFF_FP12_C1_1)
/// (1+u)^((p^3-1)/6) = [(1+u)^((p-1)/6)]^(p^2+p+1) = C12_1^(p^2) * C12_1^p * C12_1
/// = C12_1 * conj(C12_1) * C12_1 = |C12_1|^2 * C12_1
private let FROBENIUS_COEFF_FP12_C1_3: Fp2_381 = {
    let g = FROBENIUS_COEFF_FP12_C1_1
    let gp = fp2_381Conjugate(g)
    return fp2_381Mul(fp2_381Mul(g, gp), g)
}()

/// Frobenius on Fp6: frob_p(c0 + c1*v + c2*v^2) = conj(c0) + conj(c1)*gamma1 * v + conj(c2)*gamma2 * v^2
private func fp6_381Frobenius(_ a: Fp6_381) -> Fp6_381 {
    let c0 = fp2_381Conjugate(a.c0)
    let c1 = fp2_381Mul(fp2_381Conjugate(a.c1), FROBENIUS_COEFF_FP6_C1_1)
    let c2 = fp2_381Mul(fp2_381Conjugate(a.c2), FROBENIUS_COEFF_FP6_C2_1)
    return Fp6_381(c0: c0, c1: c1, c2: c2)
}

/// Frobenius^2 on Fp6
private func fp6_381Frobenius2(_ a: Fp6_381) -> Fp6_381 {
    let c0 = a.c0  // conj(conj(x)) = x for Fp2 Frobenius applied twice
    let c1 = fp2_381Mul(a.c1, FROBENIUS_COEFF_FP6_C1_2)
    let c2 = fp2_381Mul(a.c2, FROBENIUS_COEFF_FP6_C2_2)
    return Fp6_381(c0: c0, c1: c1, c2: c2)
}

/// Frobenius^3 on Fp6
private func fp6_381Frobenius3(_ a: Fp6_381) -> Fp6_381 {
    let c0 = fp2_381Conjugate(a.c0)
    let c1 = fp2_381Mul(fp2_381Conjugate(a.c1), FROBENIUS_COEFF_FP6_C1_3)
    let c2 = fp2_381Mul(fp2_381Conjugate(a.c2), FROBENIUS_COEFF_FP6_C2_3)
    return Fp6_381(c0: c0, c1: c1, c2: c2)
}

/// Frobenius endomorphism on Fp12 (p-th power)
/// frob(a0 + a1*w) = frob_fp6(a0) + frob_fp6(a1) * gamma_w * w
/// where gamma_w = (1+u)^((p-1)/6) is FROBENIUS_COEFF_FP12_C1_1
public func fp12_381Frobenius(_ a: Fp12_381) -> Fp12_381 {
    let c0 = fp6_381Frobenius(a.c0)
    let c1 = fp6_381MulByFp2(fp6_381Frobenius(a.c1), FROBENIUS_COEFF_FP12_C1_1)
    return Fp12_381(c0: c0, c1: c1)
}

/// Frobenius^2 on Fp12
public func fp12_381Frobenius2(_ a: Fp12_381) -> Fp12_381 {
    let c0 = fp6_381Frobenius2(a.c0)
    let c1 = fp6_381MulByFp2(fp6_381Frobenius2(a.c1), FROBENIUS_COEFF_FP12_C1_2)
    return Fp12_381(c0: c0, c1: c1)
}

/// Frobenius^3 on Fp12
public func fp12_381Frobenius3(_ a: Fp12_381) -> Fp12_381 {
    let c0 = fp6_381Frobenius3(a.c0)
    let c1 = fp6_381MulByFp2(fp6_381Frobenius3(a.c1), FROBENIUS_COEFF_FP12_C1_3)
    return Fp12_381(c0: c0, c1: c1)
}

/// Exponentiation by the BLS parameter x = 0xd201000000010000
/// This is used in the final exponentiation hard part
public func fp12_381PowByX(_ a: Fp12_381) -> Fp12_381 {
    // x = 0xd201000000010000 (the BLS12-381 parameter, used as positive)
    // Binary: 1101001000000001000000000000000000000000000000010000000000000000
    let xBits: [UInt8] = [
        1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]

    var result = Fp12_381.one
    for bit in xBits {
        result = fp12_381Sqr(result)
        if bit == 1 {
            result = fp12_381Mul(result, a)
        }
    }
    // x is negative in BLS12-381, so negate (conjugate for unitary elements)
    return fp12_381Conjugate(result)
}
