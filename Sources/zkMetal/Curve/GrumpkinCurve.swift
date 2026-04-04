// Grumpkin elliptic curve operations (CPU-side)
// y^2 = x^3 - 17 (a=0, b=-17)
// Grumpkin is BN254's inner curve (used by Aztec for recursive proving).
// Base field = BN254 Fr, Scalar field = BN254 Fq (Fp).
// Jacobian projective coordinates: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)

import Foundation

public struct GrumpkinPointAffine {
    public var x: Fr
    public var y: Fr

    public init(x: Fr, y: Fr) {
        self.x = x
        self.y = y
    }
}

public struct GrumpkinPointProjective {
    public var x: Fr
    public var y: Fr
    public var z: Fr

    public init(x: Fr, y: Fr, z: Fr) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func grumpkinPointIdentity() -> GrumpkinPointProjective {
    GrumpkinPointProjective(x: Fr.one, y: Fr.one, z: Fr.zero)
}

public func grumpkinPointIsIdentity(_ p: GrumpkinPointProjective) -> Bool {
    p.z.isZero
}

public func grumpkinPointFromAffine(_ a: GrumpkinPointAffine) -> GrumpkinPointProjective {
    GrumpkinPointProjective(x: a.x, y: a.y, z: Fr.one)
}

public func grumpkinPointToAffine(_ p: GrumpkinPointProjective) -> GrumpkinPointAffine {
    if grumpkinPointIsIdentity(p) {
        return GrumpkinPointAffine(x: Fr.zero, y: Fr.zero)
    }
    let zinv = frInverse(p.z)
    let zinv2 = frSqr(zinv)
    let zinv3 = frMul(zinv2, zinv)
    return GrumpkinPointAffine(x: frMul(p.x, zinv2), y: frMul(p.y, zinv3))
}

// Point doubling (a=0 for Grumpkin: y^2 = x^3 - 17)
public func grumpkinPointDouble(_ p: GrumpkinPointProjective) -> GrumpkinPointProjective {
    if grumpkinPointIsIdentity(p) { return p }

    let a = frSqr(p.x)
    let b = frSqr(p.y)
    let c = frSqr(b)

    let xpb = frAdd(p.x, b)
    let d = frDouble(frSub(frSqr(xpb), frAdd(a, c)))

    let e = frAdd(frDouble(a), a) // 3*X^2
    let f = frSqr(e)

    let rx = frSub(f, frDouble(d))
    let ry = frSub(frMul(e, frSub(d, rx)),
                   frDouble(frDouble(frDouble(c))))
    let yz = frAdd(p.y, p.z)
    let rz = frSub(frSqr(yz), frAdd(b, frSqr(p.z)))

    return GrumpkinPointProjective(x: rx, y: ry, z: rz)
}

// Full addition: projective + projective
public func grumpkinPointAdd(_ p: GrumpkinPointProjective, _ q: GrumpkinPointProjective) -> GrumpkinPointProjective {
    if grumpkinPointIsIdentity(p) { return q }
    if grumpkinPointIsIdentity(q) { return p }

    let z1z1 = frSqr(p.z)
    let z2z2 = frSqr(q.z)
    let u1 = frMul(p.x, z2z2)
    let u2 = frMul(q.x, z1z1)
    let s1 = frMul(p.y, frMul(q.z, z2z2))
    let s2 = frMul(q.y, frMul(p.z, z1z1))

    let h = frSub(u2, u1)
    let rr = frDouble(frSub(s2, s1))

    if h.isZero {
        if rr.isZero { return grumpkinPointDouble(p) }
        return grumpkinPointIdentity()
    }

    let rz = frMul(frDouble(frMul(p.z, q.z)), h)
    let dh = frDouble(h)
    let i = frSqr(dh)
    let v = frMul(u1, i)
    let j = frMul(h, i)

    let rx = frSub(frSub(frSqr(rr), j), frDouble(v))
    let ry = frSub(frMul(rr, frSub(v, rx)), frDouble(frMul(s1, j)))

    return GrumpkinPointProjective(x: rx, y: ry, z: rz)
}

// Negate affine point
public func grumpkinPointNegateAffine(_ p: GrumpkinPointAffine) -> GrumpkinPointAffine {
    GrumpkinPointAffine(x: p.x, y: frNeg(p.y))
}

// Negate projective point
public func grumpkinPointNegate(_ p: GrumpkinPointProjective) -> GrumpkinPointProjective {
    GrumpkinPointProjective(x: p.x, y: frNeg(p.y), z: p.z)
}

// Scalar multiplication (double-and-add)
public func grumpkinPointMulInt(_ p: GrumpkinPointProjective, _ n: Int) -> GrumpkinPointProjective {
    if n == 0 { return grumpkinPointIdentity() }
    var result = grumpkinPointIdentity()
    var base = p
    var scalar = n
    while scalar > 0 {
        if scalar & 1 == 1 {
            result = grumpkinPointAdd(result, base)
        }
        base = grumpkinPointDouble(base)
        scalar >>= 1
    }
    return result
}

// Scalar multiplication with 256-bit scalar (as Fp / BN254 Fq)
public func grumpkinPointScalarMul(_ p: GrumpkinPointProjective, _ scalar: Fp) -> GrumpkinPointProjective {
    // Convert from Montgomery form to get raw limbs
    let limbs = fpToInt(scalar)
    var result = grumpkinPointIdentity()
    var base = p
    for i in 0..<4 {
        var word = limbs[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = grumpkinPointAdd(result, base)
            }
            base = grumpkinPointDouble(base)
            word >>= 1
        }
    }
    return result
}

// Batch projective -> affine using Montgomery's trick
public func batchGrumpkinToAffine(_ points: [GrumpkinPointProjective]) -> [GrumpkinPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var products = [Fr](repeating: Fr.zero, count: n)
    products[0] = points[0].z
    for i in 1..<n {
        if grumpkinPointIsIdentity(points[i]) {
            products[i] = products[i - 1]
        } else {
            products[i] = frMul(products[i - 1], points[i].z)
        }
    }

    var inv = frInverse(products[n - 1])
    var result = [GrumpkinPointAffine](repeating: GrumpkinPointAffine(x: Fr.zero, y: Fr.zero), count: n)

    for i in stride(from: n - 1, through: 0, by: -1) {
        if grumpkinPointIsIdentity(points[i]) {
            result[i] = GrumpkinPointAffine(x: Fr.zero, y: Fr.zero)
            continue
        }
        let zinv = (i == 0) ? inv : frMul(inv, products[i - 1])
        if i > 0 {
            inv = frMul(inv, points[i].z)
        }
        let zinv2 = frSqr(zinv)
        let zinv3 = frMul(zinv2, zinv)
        result[i] = GrumpkinPointAffine(x: frMul(points[i].x, zinv2),
                                        y: frMul(points[i].y, zinv3))
    }
    return result
}

// Check if affine point is on curve: y^2 = x^3 - 17
public func grumpkinPointIsOnCurve(_ p: GrumpkinPointAffine) -> Bool {
    let x2 = frSqr(p.x)
    let x3 = frMul(x2, p.x)
    let b = frFromInt(17)
    let rhs = frSub(x3, b) // x^3 - 17
    let lhs = frSqr(p.y)
    return frToInt(lhs) == frToInt(rhs)
}

// Grumpkin generator point
// G = (1, sqrt(1 - 17)) = (1, sqrt(-16))
// From the Aztec/Noir specification:
// x = 1
// y = 17631683881184975370165255887551781615748388533673675138860 (mod r)
// We compute: Gx = frFromInt(1), Gy = sqrt(1^3 - 17) = sqrt(-16 mod r)
// sqrt(-16 mod r) where r = BN254 scalar field order
public func grumpkinGenerator() -> GrumpkinPointAffine {
    // x = 1 in Montgomery form
    let gx = Fr.one

    // y^2 = 1 - 17 = -16 mod r
    // We need sqrt(-16 mod r). Compute via Tonelli-Shanks or directly.
    // -16 mod r = r - 16
    // r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    // r - 16 = 21888242871839275222246405745257275088548364400416034343698204186575808495601
    // sqrt(r - 16) mod r = computed value

    // We compute sqrt(-16) using exponentiation: (-16)^((r+1)/4) since r = 3 mod 4
    // Actually r mod 4: r = ...01, so r mod 4 = 1. We need Tonelli-Shanks.
    // TWO_ADICITY = 28, so (r-1) = 2^28 * t where t is odd.

    // Instead, compute directly: y = sqrt(x^3 - 17) for x = 1
    let negSixteen = frSub(frFromInt(1), frFromInt(17)) // 1^3 - 17 = -16 mod r
    let gy = grumpkinFrSqrt(negSixteen)

    return GrumpkinPointAffine(x: gx, y: gy)
}

// Tonelli-Shanks square root for Fr (BN254 scalar field)
// Returns sqrt(a) if it exists, undefined behavior if not a QR
private func grumpkinFrSqrt(_ a: Fr) -> Fr {
    // Fr has TWO_ADICITY = 28, so r - 1 = 2^28 * t, t odd
    // t = (r - 1) / 2^28
    // We use Tonelli-Shanks algorithm

    if a.isZero { return Fr.zero }

    // r - 1 as 64-bit limbs
    let rMinus1: [UInt64] = [
        Fr.P[0] - 1, Fr.P[1], Fr.P[2], Fr.P[3]
    ]

    // t = (r-1) >> 28
    var t = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        t[i] = rMinus1[i] >> 28
        if i + 1 < 4 {
            t[i] |= rMinus1[i + 1] << 36
        }
    }

    // Non-residue: g = GENERATOR^t (generator = 5 for Fr)
    // Root of unity omega = 5^t mod r (already have ROOT_OF_UNITY = 5^((r-1)/2^28))
    var z = Fr.from64(Fr.ROOT_OF_UNITY)

    // a^((t+1)/2) — initial guess
    // (t+1)/2: t is odd, so t+1 is even
    var tPlus1Over2 = t
    // Add 1 to t
    var carry: UInt64 = 1
    for i in 0..<4 {
        let (s, c) = tPlus1Over2[i].addingReportingOverflow(carry)
        tPlus1Over2[i] = s
        carry = c ? 1 : 0
    }
    // Divide by 2
    for i in 0..<4 {
        tPlus1Over2[i] >>= 1
        if i + 1 < 4 {
            tPlus1Over2[i] |= tPlus1Over2[i + 1] << 63
        }
    }

    // x = a^((t+1)/2)
    var x = grumpkinFrPowBig(a, tPlus1Over2)
    // b = a^t
    var b = grumpkinFrPowBig(a, t)

    var m: Int = Fr.TWO_ADICITY // 28

    while true {
        if frIsOne(b) { return x }

        // Find least i such that b^(2^i) = 1
        var i = 0
        var temp = b
        while !frIsOne(temp) {
            temp = frSqr(temp)
            i += 1
        }

        // z' = z^(2^(m-i-1))
        var zPow = z
        for _ in 0..<(m - i - 1) {
            zPow = frSqr(zPow)
        }

        x = frMul(x, zPow)
        z = frSqr(zPow)
        b = frMul(b, z)
        m = i
    }
}

// Check if Fr element equals 1 (Montgomery form)
private func frIsOne(_ a: Fr) -> Bool {
    let one = Fr.one
    return a.v.0 == one.v.0 && a.v.1 == one.v.1 && a.v.2 == one.v.2 && a.v.3 == one.v.3 &&
           a.v.4 == one.v.4 && a.v.5 == one.v.5 && a.v.6 == one.v.6 && a.v.7 == one.v.7
}

// Exponentiation with big exponent (4x64-bit limbs)
private func grumpkinFrPowBig(_ base: Fr, _ exp: [UInt64]) -> Fr {
    var result = Fr.one
    var b = base
    for i in 0..<4 {
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
