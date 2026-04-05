// Jubjub twisted Edwards curve operations (CPU-side)
//
// Twisted Edwards form: ax^2 + y^2 = 1 + dx^2y^2
// a = -1, d = -(10240/10241) mod p
//
// Base field: BLS12-381 Fr (p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001)
// Subgroup order: r_s = 6554484396890773809930967563523245729705921265872317281365359162392183254199
// Cofactor: 8
//
// Used by Zcash Sapling for in-circuit signatures and commitments.
// Reference: https://zips.z.cash/protocol/protocol.pdf (Section 5.4.9.3)

import Foundation
import NeonFieldOps

// MARK: - Point Types

public struct JubjubPointAffine {
    public var x: Fr381
    public var y: Fr381

    public init(x: Fr381, y: Fr381) {
        self.x = x
        self.y = y
    }
}

/// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z
public struct JubjubPointExtended {
    public var x: Fr381
    public var y: Fr381
    public var z: Fr381
    public var t: Fr381

    public init(x: Fr381, y: Fr381, z: Fr381, t: Fr381) {
        self.x = x
        self.y = y
        self.z = z
        self.t = t
    }
}

// MARK: - Curve Constants

/// Jubjub curve parameter a = -1 (i.e. p - 1 mod p)
public func jubjubA() -> Fr381 {
    fr381Neg(Fr381.one)
}

/// Jubjub curve parameter d = -(10240/10241) mod p
/// = 19257038036680949359750312669786877991949435402254120286184196891950884077233
public func jubjubD() -> Fr381 {
    let raw = Fr381.from64([
        0x01065fd6d6343eb1, 0x292d7f6d37579d26,
        0xf5fd9207e6bd7fd4, 0x2a9318e74bfa2b48
    ])
    return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
}

/// Subgroup order r_s (little-endian 64-bit limbs)
/// r_s = 6554484396890773809930967563523245729705921265872317281365359162392183254199
public let JUBJUB_SUBGROUP_ORDER: [UInt64] = [
    0xd0970e5ed6f72cb7, 0xa6682093ccc81082,
    0x06673b0101343b00, 0x0e7db4ea6533afa9
]

/// Cofactor = 8 (Jubjub has order 8 * r_s)
public let JUBJUB_COFACTOR: UInt64 = 8

/// Standard Jubjub generator (derived from 8 * P where P is first valid point at x=3)
/// Gx = 0x5183972af8eff38ca624b4df00384882000c546bf2f39ede7f4ecf1a74f976c4
/// Gy = 0x3b43f8472ca2fc2c9e8fcc5abd9dc308096c8707ffa6833b146bad709349702e
public func jubjubGenerator() -> JubjubPointAffine {
    let gx = Fr381.from64([
        0x7f4ecf1a74f976c4, 0x000c546bf2f39ede,
        0xa624b4df00384882, 0x5183972af8eff38c
    ])
    let gy = Fr381.from64([
        0x146bad709349702e, 0x096c8707ffa6833b,
        0x9e8fcc5abd9dc308, 0x3b43f8472ca2fc2c
    ])
    // Convert to Montgomery form
    let gxMont = fr381Mul(gx, Fr381.from64(Fr381.R2_MOD_R))
    let gyMont = fr381Mul(gy, Fr381.from64(Fr381.R2_MOD_R))
    return JubjubPointAffine(x: gxMont, y: gyMont)
}

// MARK: - Identity and Predicates

/// Identity point: (0, 1) in affine, (0, 1, 1, 0) in extended
public func jubjubPointIdentity() -> JubjubPointExtended {
    JubjubPointExtended(
        x: Fr381.zero,
        y: Fr381.one,
        z: Fr381.one,
        t: Fr381.zero
    )
}

public func jubjubPointIsIdentity(_ p: JubjubPointExtended) -> Bool {
    let xInt = fr381ToInt(p.x)
    let yInt = fr381ToInt(p.y)
    let zInt = fr381ToInt(p.z)
    let xIsZero = xInt[0] == 0 && xInt[1] == 0 && xInt[2] == 0 && xInt[3] == 0
    return xIsZero && yInt == zInt
}

// MARK: - Conversions

public func jubjubPointFromAffine(_ a: JubjubPointAffine) -> JubjubPointExtended {
    JubjubPointExtended(
        x: a.x,
        y: a.y,
        z: Fr381.one,
        t: fr381Mul(a.x, a.y)
    )
}

public func jubjubPointToAffine(_ p: JubjubPointExtended) -> JubjubPointAffine {
    if p.z.isZero {
        return JubjubPointAffine(x: Fr381.zero, y: Fr381.one)
    }
    let zinv = fr381Inverse(p.z)
    return JubjubPointAffine(
        x: fr381Mul(p.x, zinv),
        y: fr381Mul(p.y, zinv)
    )
}

// MARK: - Point Operations

/// Point addition using extended coordinates (unified formula)
/// Uses C CIOS Montgomery field ops for ~10-30x speedup.
public func jubjubPointAdd(_ p: JubjubPointExtended, _ q: JubjubPointExtended) -> JubjubPointExtended {
    var result = jubjubPointIdentity()
    withUnsafeBytes(of: p) { pBuf in
        withUnsafeBytes(of: q) { qBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                jubjub_point_add(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// Point doubling (more efficient than generic add)
/// Uses C CIOS Montgomery field ops for ~10-30x speedup.
public func jubjubPointDouble(_ p: JubjubPointExtended) -> JubjubPointExtended {
    var result = jubjubPointIdentity()
    withUnsafeBytes(of: p) { pBuf in
        withUnsafeMutableBytes(of: &result) { resBuf in
            jubjub_point_double(
                pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

/// Negate a point: -(x, y) = (-x, y)
public func jubjubPointNeg(_ p: JubjubPointExtended) -> JubjubPointExtended {
    JubjubPointExtended(
        x: fr381Neg(p.x),
        y: p.y,
        z: p.z,
        t: fr381Neg(p.t)
    )
}

public func jubjubPointNegAffine(_ p: JubjubPointAffine) -> JubjubPointAffine {
    JubjubPointAffine(x: fr381Neg(p.x), y: p.y)
}

/// Scalar multiplication using windowed method (w=4)
/// Uses C CIOS Montgomery field ops via jubjub_scalar_mul.
/// Scalar is given as raw (non-Montgomery) 64-bit limbs
public func jubjubPointMulScalar(_ p: JubjubPointExtended, _ scalar: [UInt64]) -> JubjubPointExtended {
    // Ensure scalar is exactly 4 limbs
    var scalarLimbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<min(scalar.count, 4) {
        scalarLimbs[i] = scalar[i]
    }
    var result = jubjubPointIdentity()
    withUnsafeBytes(of: p) { pBuf in
        scalarLimbs.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                jubjub_scalar_mul(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    scBuf.baseAddress!,
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// Scalar multiplication with small integer
public func jubjubPointMulInt(_ p: JubjubPointExtended, _ n: Int) -> JubjubPointExtended {
    if n == 0 { return jubjubPointIdentity() }
    if n < 0 { return jubjubPointNeg(jubjubPointMulInt(p, -n)) }
    var result = jubjubPointIdentity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = jubjubPointAdd(result, base)
        }
        base = jubjubPointDouble(base)
        k >>= 1
    }
    return result
}

/// Batch extended -> affine using Montgomery's trick (single inversion)
public func jubjubBatchToAffine(_ points: [JubjubPointExtended]) -> [JubjubPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fr381](repeating: Fr381.one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = fr381Mul(prods[i - 1], points[i].z)
    }

    var inv = fr381Inverse(prods[n - 1])

    var result = [JubjubPointAffine](repeating: JubjubPointAffine(x: Fr381.zero, y: Fr381.one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        let zinv = (i > 0) ? fr381Mul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fr381Mul(inv, points[i].z) }
        result[i] = JubjubPointAffine(
            x: fr381Mul(points[i].x, zinv),
            y: fr381Mul(points[i].y, zinv)
        )
    }
    return result
}

/// Check if an affine point is on the Jubjub curve
/// ax^2 + y^2 = 1 + dx^2y^2 (a = -1)
public func jubjubPointOnCurve(_ p: JubjubPointAffine) -> Bool {
    let x2 = fr381Sqr(p.x)
    let y2 = fr381Sqr(p.y)
    let aConst = jubjubA()
    let dConst = jubjubD()
    let lhs = fr381Add(fr381Mul(aConst, x2), y2)  // a*x^2 + y^2
    let rhs = fr381Add(Fr381.one, fr381Mul(dConst, fr381Mul(x2, y2)))  // 1 + d*x^2*y^2
    return fr381ToInt(lhs) == fr381ToInt(rhs)
}

/// Power with arbitrary-size exponent (4x64-bit limbs, little-endian)
public func fr381PowBig(_ base: Fr381, _ exp: [UInt64]) -> Fr381 {
    var result = Fr381.one
    var b = base
    for i in 0..<exp.count {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = fr381Mul(result, b)
            }
            b = fr381Sqr(b)
            word >>= 1
        }
    }
    return result
}

/// Square root in Fr381 using Tonelli-Shanks
/// Fr381 has TWO_ADICITY = 32, so r - 1 = 2^32 * Q (Q odd)
public func fr381Sqrt(_ a: Fr381) -> Fr381? {
    if a.isZero { return Fr381.zero }

    let oneInt = fr381ToInt(Fr381.one)

    // Euler criterion: a^((r-1)/2) must equal 1 for QR
    let pMinus1Over2: [UInt64] = [
        0x7fffffff80000000, 0xa9ded2017fff2dff,
        0x199cec0404d0ec02, 0x39f6d3a994cebea4
    ]
    let euler = fr381PowBig(a, pMinus1Over2)
    if fr381ToInt(euler) != oneInt { return nil }

    // Q = (r-1) >> 32
    let qLimbs: [UInt64] = [
        0xfffe5bfeffffffff, 0x09a1d80553bda402,
        0x299d7d483339d808, 0x0000000073eda753
    ]
    // (Q+1)/2
    let qp1h: [UInt64] = [
        0x7fff2dff80000000, 0x04d0ec02a9ded201,
        0x94cebea4199cec04, 0x0000000039f6d3a9
    ]

    var mm = Fr381.TWO_ADICITY  // 32
    var c = fr381Mul(Fr381.from64(Fr381.ROOT_OF_UNITY), Fr381.from64(Fr381.R2_MOD_R))
    var tt = fr381PowBig(a, qLimbs)
    var rr = fr381PowBig(a, qp1h)

    while true {
        let tInt = fr381ToInt(tt)
        if tInt == oneInt { break }
        var i = 0
        var tmp = tt
        repeat {
            i += 1
            tmp = fr381Sqr(tmp)
        } while fr381ToInt(tmp) != oneInt && i < mm
        if i >= mm { return nil }
        var b = c
        for _ in 0..<(mm - i - 1) {
            b = fr381Sqr(b)
        }
        mm = i
        c = fr381Sqr(b)
        tt = fr381Mul(tt, c)
        rr = fr381Mul(rr, b)
    }
    return rr
}

// MARK: - Encoding / Decoding

/// Encode a Jubjub point as two 32-byte field elements (x, y)
public func jubjubPointEncode(_ p: JubjubPointAffine) -> [UInt8] {
    let xLimbs = fr381ToInt(p.x)
    let yLimbs = fr381ToInt(p.y)
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

/// Decode 64 bytes to a Jubjub affine point
public func jubjubPointDecode(_ bytes: [UInt8]) -> JubjubPointAffine? {
    guard bytes.count == 64 else { return nil }
    var xLimbs: [UInt64] = [0, 0, 0, 0]
    var yLimbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        for j in 0..<8 {
            xLimbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            yLimbs[i] |= UInt64(bytes[32 + i * 8 + j]) << (j * 8)
        }
    }
    let x = fr381Mul(Fr381.from64(xLimbs), Fr381.from64(Fr381.R2_MOD_R))
    let y = fr381Mul(Fr381.from64(yLimbs), Fr381.from64(Fr381.R2_MOD_R))
    let p = JubjubPointAffine(x: x, y: y)
    guard jubjubPointOnCurve(p) else { return nil }
    return p
}

/// Compressed encoding: y coordinate + sign bit of x in top bit
public func jubjubPointCompress(_ p: JubjubPointAffine) -> [UInt8] {
    let xLimbs = fr381ToInt(p.x)
    let yLimbs = fr381ToInt(p.y)
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
