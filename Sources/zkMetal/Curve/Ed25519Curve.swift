// Ed25519 / Curve25519 elliptic curve operations (CPU-side)
//
// Twisted Edwards form: -x^2 + y^2 = 1 + d*x^2*y^2
// d = -121665/121666 mod p
//
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = X*Y/Z
// This representation allows unified addition (no special cases for doubling).

import Foundation

// MARK: - Point Types

public struct Ed25519PointAffine {
    public var x: Ed25519Fp
    public var y: Ed25519Fp

    public init(x: Ed25519Fp, y: Ed25519Fp) {
        self.x = x
        self.y = y
    }
}

/// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
public struct Ed25519PointExtended {
    public var x: Ed25519Fp
    public var y: Ed25519Fp
    public var z: Ed25519Fp
    public var t: Ed25519Fp

    public init(x: Ed25519Fp, y: Ed25519Fp, z: Ed25519Fp, t: Ed25519Fp) {
        self.x = x
        self.y = y
        self.z = z
        self.t = t
    }
}

// MARK: - Curve Constants

/// d = -121665/121666 mod p
/// Computed as: -121665 * inverse(121666) mod p
public func ed25519D() -> Ed25519Fp {
    let n121665 = ed25519FpFromInt(121665)
    let n121666 = ed25519FpFromInt(121666)
    let inv121666 = ed25519FpInverse(n121666)
    return ed25519FpNeg(ed25519FpMul(n121665, inv121666))
}

/// 2*d (used frequently in addition formulas)
public func ed25519D2() -> Ed25519Fp {
    return ed25519FpDouble(ed25519D())
}

/// Ed25519 base point (generator)
/// y = 4/5 mod p, x is the positive square root
public func ed25519Generator() -> Ed25519PointAffine {
    // y = 4/5 mod p
    let four = ed25519FpFromInt(4)
    let five = ed25519FpFromInt(5)
    let y = ed25519FpMul(four, ed25519FpInverse(five))

    // x^2 = (y^2 - 1) / (d*y^2 + 1)
    let y2 = ed25519FpSqr(y)
    let d = ed25519D()
    let num = ed25519FpSub(y2, Ed25519Fp.one)
    let den = ed25519FpAdd(ed25519FpMul(d, y2), Ed25519Fp.one)
    let x2 = ed25519FpMul(num, ed25519FpInverse(den))

    guard let x = ed25519FpSqrt(x2) else {
        fatalError("Ed25519 generator x coordinate has no square root")
    }

    // Choose the positive root (even, i.e. x mod 2 == 0)
    let xBytes = ed25519FpToBytes(x)
    if xBytes[0] & 1 != 0 {
        return Ed25519PointAffine(x: ed25519FpNeg(x), y: y)
    }
    return Ed25519PointAffine(x: x, y: y)
}

// MARK: - Identity and Predicates

/// Identity point: (0, 1) in affine, (0, 1, 1, 0) in extended
public func ed25519PointIdentity() -> Ed25519PointExtended {
    Ed25519PointExtended(
        x: Ed25519Fp.zero,
        y: Ed25519Fp.one,
        z: Ed25519Fp.one,
        t: Ed25519Fp.zero
    )
}

public func ed25519PointIsIdentity(_ p: Ed25519PointExtended) -> Bool {
    // Identity: X == 0 and Y == Z (and T == 0)
    let xInt = ed25519FpToInt(p.x)
    let yInt = ed25519FpToInt(p.y)
    let zInt = ed25519FpToInt(p.z)
    let xIsZero = xInt[0] == 0 && xInt[1] == 0 && xInt[2] == 0 && xInt[3] == 0
    return xIsZero && yInt == zInt
}

// MARK: - Conversions

public func ed25519PointFromAffine(_ a: Ed25519PointAffine) -> Ed25519PointExtended {
    Ed25519PointExtended(
        x: a.x,
        y: a.y,
        z: Ed25519Fp.one,
        t: ed25519FpMul(a.x, a.y)
    )
}

public func ed25519PointToAffine(_ p: Ed25519PointExtended) -> Ed25519PointAffine {
    if p.z.isZero {
        return Ed25519PointAffine(x: Ed25519Fp.zero, y: Ed25519Fp.one)
    }
    let zinv = ed25519FpInverse(p.z)
    return Ed25519PointAffine(
        x: ed25519FpMul(p.x, zinv),
        y: ed25519FpMul(p.y, zinv)
    )
}

// MARK: - Point Operations

/// Point addition using extended coordinates (unified formula)
/// Algorithm from "Twisted Edwards Curves Revisited" (Hisil et al.)
/// Cost: 8M + 1D (where D = mul by 2*d constant)
public func ed25519PointAdd(_ p: Ed25519PointExtended, _ q: Ed25519PointExtended) -> Ed25519PointExtended {
    let d2 = ed25519D2()
    let a = ed25519FpMul(p.x, q.x)
    let b = ed25519FpMul(p.y, q.y)
    let c = ed25519FpMul(ed25519FpMul(p.t, q.t), d2)
    let d = ed25519FpMul(p.z, q.z)
    let dd = ed25519FpDouble(d)
    let e = ed25519FpSub(ed25519FpMul(ed25519FpAdd(p.x, p.y), ed25519FpAdd(q.x, q.y)), ed25519FpAdd(a, b))
    let f = ed25519FpSub(dd, c)
    let g = ed25519FpAdd(dd, c)
    // For -x^2 + y^2: h = b + a (since a = x1*x2 and curve is -x^2 + y^2)
    let h = ed25519FpAdd(b, a)

    return Ed25519PointExtended(
        x: ed25519FpMul(e, f),
        y: ed25519FpMul(g, h),
        z: ed25519FpMul(f, g),
        t: ed25519FpMul(e, h)
    )
}

/// Point doubling (more efficient than generic add)
/// Cost: 4M + 4S + 1D
public func ed25519PointDouble(_ p: Ed25519PointExtended) -> Ed25519PointExtended {
    let a = ed25519FpSqr(p.x)
    let b = ed25519FpSqr(p.y)
    let c = ed25519FpDouble(ed25519FpSqr(p.z))
    // For -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1):
    // D = -A (since a = -1)
    let d_val = ed25519FpNeg(a)
    let e = ed25519FpSub(ed25519FpSqr(ed25519FpAdd(p.x, p.y)), ed25519FpAdd(a, b))
    let g = ed25519FpAdd(d_val, b)
    let f = ed25519FpSub(g, c)
    let h = ed25519FpSub(d_val, b)

    return Ed25519PointExtended(
        x: ed25519FpMul(e, f),
        y: ed25519FpMul(g, h),
        z: ed25519FpMul(f, g),
        t: ed25519FpMul(e, h)
    )
}

/// Negate a point: -(x, y) = (-x, y)
public func ed25519PointNeg(_ p: Ed25519PointExtended) -> Ed25519PointExtended {
    Ed25519PointExtended(
        x: ed25519FpNeg(p.x),
        y: p.y,
        z: p.z,
        t: ed25519FpNeg(p.t)
    )
}

public func ed25519PointNegAffine(_ p: Ed25519PointAffine) -> Ed25519PointAffine {
    Ed25519PointAffine(x: ed25519FpNeg(p.x), y: p.y)
}

/// Scalar multiplication: double-and-add
public func ed25519PointMulScalar(_ p: Ed25519PointExtended, _ scalar: [UInt64]) -> Ed25519PointExtended {
    var result = ed25519PointIdentity()
    var base = p
    for i in 0..<4 {
        var word = scalar[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = ed25519PointAdd(result, base)
            }
            base = ed25519PointDouble(base)
            word >>= 1
        }
    }
    return result
}

/// Scalar multiplication with small integer
public func ed25519PointMulInt(_ p: Ed25519PointExtended, _ n: Int) -> Ed25519PointExtended {
    if n == 0 { return ed25519PointIdentity() }
    if n < 0 { return ed25519PointNeg(ed25519PointMulInt(p, -n)) }
    var result = ed25519PointIdentity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = ed25519PointAdd(result, base)
        }
        base = ed25519PointDouble(base)
        k >>= 1
    }
    return result
}

/// Batch extended -> affine using Montgomery's trick (single inversion)
public func ed25519BatchToAffine(_ points: [Ed25519PointExtended]) -> [Ed25519PointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Ed25519Fp](repeating: Ed25519Fp.one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = ed25519FpMul(prods[i - 1], points[i].z)
    }

    var inv = ed25519FpInverse(prods[n - 1])

    var result = [Ed25519PointAffine](repeating: Ed25519PointAffine(x: Ed25519Fp.zero, y: Ed25519Fp.one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        let zinv = (i > 0) ? ed25519FpMul(inv, prods[i - 1]) : inv
        if i > 0 { inv = ed25519FpMul(inv, points[i].z) }
        result[i] = Ed25519PointAffine(
            x: ed25519FpMul(points[i].x, zinv),
            y: ed25519FpMul(points[i].y, zinv)
        )
    }
    return result
}

/// Check if an affine point is on the Ed25519 curve
public func ed25519PointOnCurve(_ p: Ed25519PointAffine) -> Bool {
    // -x^2 + y^2 = 1 + d*x^2*y^2
    let x2 = ed25519FpSqr(p.x)
    let y2 = ed25519FpSqr(p.y)
    let d = ed25519D()
    let lhs = ed25519FpAdd(ed25519FpNeg(x2), y2)  // -x^2 + y^2
    let rhs = ed25519FpAdd(Ed25519Fp.one, ed25519FpMul(d, ed25519FpMul(x2, y2)))  // 1 + d*x^2*y^2
    return ed25519FpToInt(lhs) == ed25519FpToInt(rhs)
}

// MARK: - Ed25519 Encoding/Decoding

/// Encode an affine point to 32 bytes (RFC 8032)
/// Format: y coordinate with the sign bit of x in the top bit
public func ed25519PointEncode(_ p: Ed25519PointAffine) -> [UInt8] {
    var bytes = ed25519FpToBytes(p.y)
    let xBytes = ed25519FpToBytes(p.x)
    bytes[31] |= (xBytes[0] & 1) << 7  // sign bit of x in top bit of last byte
    return bytes
}

/// Decode 32 bytes to an Ed25519 point (RFC 8032)
public func ed25519PointDecode(_ bytes: [UInt8]) -> Ed25519PointAffine? {
    guard bytes.count == 32 else { return nil }
    let xSign = (bytes[31] >> 7) & 1
    var yBytes = bytes
    yBytes[31] &= 0x7F  // clear sign bit

    let y = ed25519FpFromBytes(yBytes)
    let y2 = ed25519FpSqr(y)
    let d = ed25519D()
    // x^2 = (y^2 - 1) / (d*y^2 + 1)
    let num = ed25519FpSub(y2, Ed25519Fp.one)
    let den = ed25519FpAdd(ed25519FpMul(d, y2), Ed25519Fp.one)
    let denInv = ed25519FpInverse(den)
    let x2 = ed25519FpMul(num, denInv)

    if x2.isZero {
        if xSign != 0 { return nil }
        return Ed25519PointAffine(x: Ed25519Fp.zero, y: y)
    }

    guard var x = ed25519FpSqrt(x2) else { return nil }

    let xB = ed25519FpToBytes(x)
    if (xB[0] & 1) != xSign {
        x = ed25519FpNeg(x)
    }

    return Ed25519PointAffine(x: x, y: y)
}

// MARK: - Curve25519 / Montgomery Form Conversion

/// Convert Ed25519 point to Curve25519 (Montgomery form) u-coordinate
/// u = (1 + y) / (1 - y) mod p
public func ed25519ToMontgomery(_ p: Ed25519PointAffine) -> Ed25519Fp {
    let one = Ed25519Fp.one
    let num = ed25519FpAdd(one, p.y)
    let den = ed25519FpSub(one, p.y)
    return ed25519FpMul(num, ed25519FpInverse(den))
}

/// Convert Curve25519 u-coordinate to Ed25519 affine point
/// y = (u - 1) / (u + 1) mod p
/// x recovered from the curve equation
public func ed25519FromMontgomery(_ u: Ed25519Fp) -> Ed25519PointAffine? {
    let one = Ed25519Fp.one
    let num = ed25519FpSub(u, one)
    let den = ed25519FpAdd(u, one)
    let y = ed25519FpMul(num, ed25519FpInverse(den))

    let y2 = ed25519FpSqr(y)
    let d = ed25519D()
    let x2num = ed25519FpSub(y2, one)
    let x2den = ed25519FpAdd(ed25519FpMul(d, y2), one)
    let x2 = ed25519FpMul(x2num, ed25519FpInverse(x2den))

    guard let x = ed25519FpSqrt(x2) else { return nil }
    return Ed25519PointAffine(x: x, y: y)
}
