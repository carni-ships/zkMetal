// BabyJubjub twisted Edwards curve operations (CPU-side)
//
// Twisted Edwards form: ax^2 + y^2 = 1 + dx^2y^2
// a = 168700, d = 168696 (standard BabyJubjub parameters)
//
// Base field: BN254 Fr (already exists with full GPU support)
// Subgroup order: q = 2736030358979909402780800718157159386076813972158567259200215660948447373041
//
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
// Used by Circom, Semaphore, Tornado Cash, Polygon for in-circuit signatures.

import Foundation

// MARK: - Point Types

public struct BJJPointAffine {
    public var x: Fr
    public var y: Fr

    public init(x: Fr, y: Fr) {
        self.x = x
        self.y = y
    }
}

/// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
public struct BJJPointExtended {
    public var x: Fr
    public var y: Fr
    public var z: Fr
    public var t: Fr

    public init(x: Fr, y: Fr, z: Fr, t: Fr) {
        self.x = x
        self.y = y
        self.z = z
        self.t = t
    }
}

// MARK: - Curve Constants

/// BabyJubjub curve parameter a = 168700
public func bjjA() -> Fr {
    frFromInt(168700)
}

/// BabyJubjub curve parameter d = 168696
public func bjjD() -> Fr {
    frFromInt(168696)
}

/// Subgroup order q (little-endian 64-bit limbs)
/// q = 2736030358979909402780800718157159386076813972158567259200215660948447373041
public let BJJ_SUBGROUP_ORDER: [UInt64] = [
    0x060c89ce5c263405, 0x24ed9171a53c0f03,
    0x0000000000000000, 0x0c19139cb84c680a
]

/// Cofactor = 8 (BabyJubjub has order 8 * q)
public let BJJ_COFACTOR: UInt64 = 8

/// Standard BabyJubjub generator (base point)
/// Gx = 5299619240641551281634865583518297030282874472190772894086521144482721001553
/// Gy = 16950150798460657717958625567821834550301663161624707787222815936182638968203
public func bjjGenerator() -> BJJPointAffine {
    let gx = Fr.from64([
        0x2893f3f6bb957051, 0x2ab8d8010534e0b6,
        0x4eacb2e09d6277c1, 0x0bb77a6ad63e739b
    ])
    let gy = Fr.from64([
        0x4b3c257a872d7d8b, 0xfce0051fb9e13377,
        0x25572e1cd16bf9ed, 0x25797203f7a0b249
    ])
    // Convert to Montgomery form
    let gxMont = frMul(gx, Fr.from64(Fr.R2_MOD_R))
    let gyMont = frMul(gy, Fr.from64(Fr.R2_MOD_R))
    return BJJPointAffine(x: gxMont, y: gyMont)
}

// MARK: - Identity and Predicates

/// Identity point: (0, 1) in affine, (0, 1, 1, 0) in extended
public func bjjPointIdentity() -> BJJPointExtended {
    BJJPointExtended(
        x: Fr.zero,
        y: Fr.one,
        z: Fr.one,
        t: Fr.zero
    )
}

public func bjjPointIsIdentity(_ p: BJJPointExtended) -> Bool {
    let xInt = frToInt(p.x)
    let yInt = frToInt(p.y)
    let zInt = frToInt(p.z)
    let xIsZero = xInt[0] == 0 && xInt[1] == 0 && xInt[2] == 0 && xInt[3] == 0
    return xIsZero && yInt == zInt
}

// MARK: - Conversions

public func bjjPointFromAffine(_ a: BJJPointAffine) -> BJJPointExtended {
    BJJPointExtended(
        x: a.x,
        y: a.y,
        z: Fr.one,
        t: frMul(a.x, a.y)
    )
}

public func bjjPointToAffine(_ p: BJJPointExtended) -> BJJPointAffine {
    if p.z.isZero {
        return BJJPointAffine(x: Fr.zero, y: Fr.one)
    }
    let zinv = frInverse(p.z)
    return BJJPointAffine(
        x: frMul(p.x, zinv),
        y: frMul(p.y, zinv)
    )
}

// MARK: - Point Operations

/// Point addition using extended coordinates (unified formula)
/// For ax^2 + y^2 = 1 + dx^2y^2 with general a:
/// A = X1*X2, B = Y1*Y2, C = d*T1*T2, D = Z1*Z2
/// E = (X1+Y1)*(X2+Y2) - A - B
/// F = D - C, G = D + C
/// H = B - a*A
/// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
public func bjjPointAdd(_ p: BJJPointExtended, _ q: BJJPointExtended) -> BJJPointExtended {
    let aConst = bjjA()
    let dConst = bjjD()
    let aa = frMul(p.x, q.x)
    let bb = frMul(p.y, q.y)
    let cc = frMul(frMul(p.t, q.t), dConst)
    let dd = frMul(p.z, q.z)
    let e = frSub(frMul(frAdd(p.x, p.y), frAdd(q.x, q.y)), frAdd(aa, bb))
    let f = frSub(dd, cc)
    let g = frAdd(dd, cc)
    let h = frSub(bb, frMul(aConst, aa))

    return BJJPointExtended(
        x: frMul(e, f),
        y: frMul(g, h),
        z: frMul(f, g),
        t: frMul(e, h)
    )
}

/// Point doubling (more efficient than generic add)
/// For ax^2 + y^2 = 1 + dx^2y^2 with general a:
/// A = X^2, B = Y^2, C = 2*Z^2
/// D = a*A
/// E = (X+Y)^2 - A - B
/// G = D + B, F = G - C, H = D - B
/// X3 = E*F, Y3 = G*H, T3 = E*H, Z3 = F*G
public func bjjPointDouble(_ p: BJJPointExtended) -> BJJPointExtended {
    let aConst = bjjA()
    let aa = frSqr(p.x)
    let bb = frSqr(p.y)
    let cc = frDouble(frSqr(p.z))
    let dd = frMul(aConst, aa)
    let e = frSub(frSqr(frAdd(p.x, p.y)), frAdd(aa, bb))
    let g = frAdd(dd, bb)
    let f = frSub(g, cc)
    let h = frSub(dd, bb)

    return BJJPointExtended(
        x: frMul(e, f),
        y: frMul(g, h),
        z: frMul(f, g),
        t: frMul(e, h)
    )
}

/// Negate a point: -(x, y) = (-x, y)
public func bjjPointNeg(_ p: BJJPointExtended) -> BJJPointExtended {
    BJJPointExtended(
        x: frNeg(p.x),
        y: p.y,
        z: p.z,
        t: frNeg(p.t)
    )
}

public func bjjPointNegAffine(_ p: BJJPointAffine) -> BJJPointAffine {
    BJJPointAffine(x: frNeg(p.x), y: p.y)
}

/// Scalar multiplication: double-and-add
/// Scalar is given as raw (non-Montgomery) 64-bit limbs
public func bjjPointMulScalar(_ p: BJJPointExtended, _ scalar: [UInt64]) -> BJJPointExtended {
    var result = bjjPointIdentity()
    var base = p
    for i in 0..<scalar.count {
        var word = scalar[i]
        let bits = (i < scalar.count - 1) ? 64 : 64
        for _ in 0..<bits {
            if word & 1 == 1 {
                result = bjjPointAdd(result, base)
            }
            base = bjjPointDouble(base)
            word >>= 1
        }
    }
    return result
}

/// Scalar multiplication with small integer
public func bjjPointMulInt(_ p: BJJPointExtended, _ n: Int) -> BJJPointExtended {
    if n == 0 { return bjjPointIdentity() }
    if n < 0 { return bjjPointNeg(bjjPointMulInt(p, -n)) }
    var result = bjjPointIdentity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = bjjPointAdd(result, base)
        }
        base = bjjPointDouble(base)
        k >>= 1
    }
    return result
}

/// Batch extended -> affine using Montgomery's trick (single inversion)
public func bjjBatchToAffine(_ points: [BJJPointExtended]) -> [BJJPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fr](repeating: Fr.one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = frMul(prods[i - 1], points[i].z)
    }

    var inv = frInverse(prods[n - 1])

    var result = [BJJPointAffine](repeating: BJJPointAffine(x: Fr.zero, y: Fr.one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        let zinv = (i > 0) ? frMul(inv, prods[i - 1]) : inv
        if i > 0 { inv = frMul(inv, points[i].z) }
        result[i] = BJJPointAffine(
            x: frMul(points[i].x, zinv),
            y: frMul(points[i].y, zinv)
        )
    }
    return result
}

/// Check if an affine point is on the BabyJubjub curve
/// ax^2 + y^2 = 1 + dx^2y^2
public func bjjPointOnCurve(_ p: BJJPointAffine) -> Bool {
    let x2 = frSqr(p.x)
    let y2 = frSqr(p.y)
    let aConst = bjjA()
    let dConst = bjjD()
    let lhs = frAdd(frMul(aConst, x2), y2)  // a*x^2 + y^2
    let rhs = frAdd(Fr.one, frMul(dConst, frMul(x2, y2)))  // 1 + d*x^2*y^2
    return frToInt(lhs) == frToInt(rhs)
}

/// Square root in Fr using Tonelli-Shanks
/// Fr has TWO_ADICITY = 28, so r - 1 = 2^28 * Q (Q odd)
/// Q = 81540058820840996586704275553141814055101440848469862132140264610111
public func frSqrt(_ a: Fr) -> Fr? {
    if a.isZero { return Fr.zero }

    let oneInt = frToInt(Fr.one)

    // Euler criterion: a^((r-1)/2) must equal 1 for QR
    let pMinus1Over2: [UInt64] = [
        0xa1f0fac9f8000000, 0x9419f4243cdcb848,
        0xdc2822db40c0ac2e, 0x183227397098d014
    ]
    let euler = frPowBig(a, pMinus1Over2)
    if frToInt(euler) != oneInt { return nil }

    // Q = (r-1) >> 28 (precomputed)
    let qLimbs: [UInt64] = [
        0x9b9709143e1f593f, 0x181585d2833e8487,
        0x131a029b85045b68, 0x000000030644e72e
    ]
    // (Q+1)/2 (precomputed)
    let qp1h: [UInt64] = [
        0xcdcb848a1f0faca0, 0x0c0ac2e9419f4243,
        0x098d014dc2822db4, 0x0000000183227397
    ]

    var mm = Fr.TWO_ADICITY  // 28
    var c = Fr.from64(Fr.ROOT_OF_UNITY)  // primitive 2^28-th root of unity (a non-residue^Q)
    var tt = frPowBig(a, qLimbs)
    var rr = frPowBig(a, qp1h)

    while true {
        let tInt = frToInt(tt)
        if tInt == oneInt { break }
        var i = 0
        var tmp = tt
        repeat {
            i += 1
            tmp = frSqr(tmp)
        } while frToInt(tmp) != oneInt && i < mm
        if i >= mm { return nil }
        var b = c
        for _ in 0..<(mm - i - 1) {
            b = frSqr(b)
        }
        mm = i
        c = frSqr(b)
        tt = frMul(tt, c)
        rr = frMul(rr, b)
    }
    return rr
}

/// Power with arbitrary-size exponent (4x64-bit limbs, little-endian)
public func frPowBig(_ base: Fr, _ exp: [UInt64]) -> Fr {
    var result = Fr.one
    var b = base
    for i in 0..<exp.count {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = frMul(result, b)
            }
            b = frSqr(b)
            word >>= 1
        }
    }
    return result
}

// MARK: - Encoding / Decoding

/// Encode a BabyJubjub point as two 32-byte field elements (x, y)
public func bjjPointEncode(_ p: BJJPointAffine) -> [UInt8] {
    let xLimbs = frToInt(p.x)
    let yLimbs = frToInt(p.y)
    var bytes = [UInt8](repeating: 0, count: 64)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((xLimbs[i] >> (j * 8)) & 0xFF)
        }
        for j in 0..<8 {
            bytes[32 + i * 8 + j] = UInt8((yLimbs[i] >> (j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Decode 64 bytes to a BabyJubjub affine point
public func bjjPointDecode(_ bytes: [UInt8]) -> BJJPointAffine? {
    guard bytes.count == 64 else { return nil }
    var xLimbs: [UInt64] = [0, 0, 0, 0]
    var yLimbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        for j in 0..<8 {
            xLimbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            yLimbs[i] |= UInt64(bytes[32 + i * 8 + j]) << (j * 8)
        }
    }
    let x = frMul(Fr.from64(xLimbs), Fr.from64(Fr.R2_MOD_R))
    let y = frMul(Fr.from64(yLimbs), Fr.from64(Fr.R2_MOD_R))
    let p = BJJPointAffine(x: x, y: y)
    guard bjjPointOnCurve(p) else { return nil }
    return p
}

/// Compressed encoding: y coordinate + sign bit of x in top bit
public func bjjPointCompress(_ p: BJJPointAffine) -> [UInt8] {
    let xLimbs = frToInt(p.x)
    let yLimbs = frToInt(p.y)
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((yLimbs[i] >> (j * 8)) & 0xFF)
        }
    }
    // Sign of x: use parity (least significant bit)
    bytes[31] |= UInt8((xLimbs[0] & 1) << 7)
    return bytes
}
