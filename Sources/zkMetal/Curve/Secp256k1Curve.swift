// secp256k1 elliptic curve operations (CPU-side)
// y^2 = x^3 + 7
// Jacobian projective coordinates: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)

import Foundation

public struct SecpPointAffine {
    public var x: SecpFp
    public var y: SecpFp

    public init(x: SecpFp, y: SecpFp) {
        self.x = x
        self.y = y
    }
}

public struct SecpPointProjective {
    public var x: SecpFp
    public var y: SecpFp
    public var z: SecpFp

    public init(x: SecpFp, y: SecpFp, z: SecpFp) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func secpPointIdentity() -> SecpPointProjective {
    SecpPointProjective(x: SecpFp.one, y: SecpFp.one, z: SecpFp.zero)
}

public func secpPointIsIdentity(_ p: SecpPointProjective) -> Bool {
    p.z.isZero
}

public func secpPointFromAffine(_ a: SecpPointAffine) -> SecpPointProjective {
    SecpPointProjective(x: a.x, y: a.y, z: SecpFp.one)
}

public func secpPointToAffine(_ p: SecpPointProjective) -> SecpPointAffine {
    if secpPointIsIdentity(p) {
        return SecpPointAffine(x: SecpFp.zero, y: SecpFp.zero)
    }
    let zinv = secpInverse(p.z)
    let zinv2 = secpSqr(zinv)
    let zinv3 = secpMul(zinv2, zinv)
    return SecpPointAffine(x: secpMul(p.x, zinv2), y: secpMul(p.y, zinv3))
}

// Point doubling (a=0 for secp256k1)
public func secpPointDouble(_ p: SecpPointProjective) -> SecpPointProjective {
    if secpPointIsIdentity(p) { return p }

    let a = secpSqr(p.x)
    let b = secpSqr(p.y)
    let c = secpSqr(b)

    let xpb = secpAdd(p.x, b)
    let d = secpDouble(secpSub(secpSqr(xpb), secpAdd(a, c)))

    let e = secpAdd(secpDouble(a), a) // 3*X^2
    let f = secpSqr(e)

    let rx = secpSub(f, secpDouble(d))
    let ry = secpSub(secpMul(e, secpSub(d, rx)),
                     secpDouble(secpDouble(secpDouble(c))))
    let yz = secpAdd(p.y, p.z)
    let rz = secpSub(secpSqr(yz), secpAdd(b, secpSqr(p.z)))

    return SecpPointProjective(x: rx, y: ry, z: rz)
}

// Full addition: projective + projective
public func secpPointAdd(_ p: SecpPointProjective, _ q: SecpPointProjective) -> SecpPointProjective {
    if secpPointIsIdentity(p) { return q }
    if secpPointIsIdentity(q) { return p }

    let z1z1 = secpSqr(p.z)
    let z2z2 = secpSqr(q.z)
    let u1 = secpMul(p.x, z2z2)
    let u2 = secpMul(q.x, z1z1)
    let s1 = secpMul(p.y, secpMul(q.z, z2z2))
    let s2 = secpMul(q.y, secpMul(p.z, z1z1))

    let h = secpSub(u2, u1)
    let rr = secpDouble(secpSub(s2, s1))

    if h.isZero {
        if rr.isZero { return secpPointDouble(p) }
        return secpPointIdentity()
    }

    let rz = secpMul(secpDouble(secpMul(p.z, q.z)), h)
    let dh = secpDouble(h)
    let i = secpSqr(dh)
    let v = secpMul(u1, i)
    let j = secpMul(h, i)

    let rx = secpSub(secpSub(secpSqr(rr), j), secpDouble(v))
    let ry = secpSub(secpMul(rr, secpSub(v, rx)), secpDouble(secpMul(s1, j)))

    return SecpPointProjective(x: rx, y: ry, z: rz)
}

// Negate affine point
public func secpPointNegateAffine(_ p: SecpPointAffine) -> SecpPointAffine {
    SecpPointAffine(x: p.x, y: secpNeg(p.y))
}

// Scalar multiplication (double-and-add)
public func secpPointMulInt(_ p: SecpPointProjective, _ n: Int) -> SecpPointProjective {
    if n == 0 { return secpPointIdentity() }
    var result = secpPointIdentity()
    var base = p
    var scalar = n
    while scalar > 0 {
        if scalar & 1 == 1 {
            result = secpPointAdd(result, base)
        }
        base = secpPointDouble(base)
        scalar >>= 1
    }
    return result
}

// Batch projective → affine using Montgomery's trick
public func batchSecpToAffine(_ points: [SecpPointProjective]) -> [SecpPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    // Accumulate products of z-coordinates
    var products = [SecpFp](repeating: SecpFp.zero, count: n)
    products[0] = points[0].z
    for i in 1..<n {
        if secpPointIsIdentity(points[i]) {
            products[i] = products[i - 1]
        } else {
            products[i] = secpMul(products[i - 1], points[i].z)
        }
    }

    // Single inversion
    var inv = secpInverse(products[n - 1])
    var result = [SecpPointAffine](repeating: SecpPointAffine(x: SecpFp.zero, y: SecpFp.zero), count: n)

    // Back-propagate inverses
    for i in stride(from: n - 1, through: 0, by: -1) {
        if secpPointIsIdentity(points[i]) {
            result[i] = SecpPointAffine(x: SecpFp.zero, y: SecpFp.zero)
            continue
        }
        let zinv = (i == 0) ? inv : secpMul(inv, products[i - 1])
        if i > 0 {
            inv = secpMul(inv, points[i].z)
        }
        let zinv2 = secpSqr(zinv)
        let zinv3 = secpMul(zinv2, zinv)
        result[i] = SecpPointAffine(x: secpMul(points[i].x, zinv2),
                                    y: secpMul(points[i].y, zinv3))
    }
    return result
}

// secp256k1 generator point G in Montgomery form
public func secp256k1Generator() -> SecpPointAffine {
    let gx = SecpFp(v: (0x487e2097, 0xd7362e5a, 0x29bc66db, 0x231e2953,
                        0x33fd129c, 0x979f48c0, 0xe9089f48, 0x9981e643))
    let gy = SecpFp(v: (0xd3dbabe2, 0xb15ea6d2, 0x1f1dc64d, 0x8dfc5d5d,
                        0xac19c136, 0x70b6b59a, 0xd4a582d6, 0xcf3f851f))
    return SecpPointAffine(x: gx, y: gy)
}
