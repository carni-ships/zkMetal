// Vesta elliptic curve operations (CPU-side)
// y^2 = x^3 + 5
// Jacobian projective coordinates: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)

import Foundation

public struct VestaPointAffine {
    public var x: VestaFp
    public var y: VestaFp

    public init(x: VestaFp, y: VestaFp) {
        self.x = x
        self.y = y
    }
}

public struct VestaPointProjective {
    public var x: VestaFp
    public var y: VestaFp
    public var z: VestaFp

    public init(x: VestaFp, y: VestaFp, z: VestaFp) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func vestaPointIdentity() -> VestaPointProjective {
    VestaPointProjective(x: VestaFp.one, y: VestaFp.one, z: VestaFp.zero)
}

public func vestaPointIsIdentity(_ p: VestaPointProjective) -> Bool {
    p.z.isZero
}

public func vestaPointFromAffine(_ a: VestaPointAffine) -> VestaPointProjective {
    VestaPointProjective(x: a.x, y: a.y, z: VestaFp.one)
}

public func vestaPointToAffine(_ p: VestaPointProjective) -> VestaPointAffine {
    if vestaPointIsIdentity(p) {
        return VestaPointAffine(x: VestaFp.zero, y: VestaFp.zero)
    }
    let zinv = vestaInverse(p.z)
    let zinv2 = vestaSqr(zinv)
    let zinv3 = vestaMul(zinv2, zinv)
    return VestaPointAffine(x: vestaMul(p.x, zinv2), y: vestaMul(p.y, zinv3))
}

// Point doubling (a=0 for Vesta: y^2 = x^3 + 5)
public func vestaPointDouble(_ p: VestaPointProjective) -> VestaPointProjective {
    if vestaPointIsIdentity(p) { return p }

    let a = vestaSqr(p.x)
    let b = vestaSqr(p.y)
    let c = vestaSqr(b)

    let xpb = vestaAdd(p.x, b)
    let d = vestaDouble(vestaSub(vestaSqr(xpb), vestaAdd(a, c)))

    let e = vestaAdd(vestaDouble(a), a) // 3*X^2
    let f = vestaSqr(e)

    let rx = vestaSub(f, vestaDouble(d))
    let ry = vestaSub(vestaMul(e, vestaSub(d, rx)),
                     vestaDouble(vestaDouble(vestaDouble(c))))
    let yz = vestaAdd(p.y, p.z)
    let rz = vestaSub(vestaSqr(yz), vestaAdd(b, vestaSqr(p.z)))

    return VestaPointProjective(x: rx, y: ry, z: rz)
}

// Full addition: projective + projective
public func vestaPointAdd(_ p: VestaPointProjective, _ q: VestaPointProjective) -> VestaPointProjective {
    if vestaPointIsIdentity(p) { return q }
    if vestaPointIsIdentity(q) { return p }

    let z1z1 = vestaSqr(p.z)
    let z2z2 = vestaSqr(q.z)
    let u1 = vestaMul(p.x, z2z2)
    let u2 = vestaMul(q.x, z1z1)
    let s1 = vestaMul(p.y, vestaMul(q.z, z2z2))
    let s2 = vestaMul(q.y, vestaMul(p.z, z1z1))

    let h = vestaSub(u2, u1)
    let rr = vestaDouble(vestaSub(s2, s1))

    if h.isZero {
        if rr.isZero { return vestaPointDouble(p) }
        return vestaPointIdentity()
    }

    let rz = vestaMul(vestaDouble(vestaMul(p.z, q.z)), h)
    let dh = vestaDouble(h)
    let i = vestaSqr(dh)
    let v = vestaMul(u1, i)
    let j = vestaMul(h, i)

    let rx = vestaSub(vestaSub(vestaSqr(rr), j), vestaDouble(v))
    let ry = vestaSub(vestaMul(rr, vestaSub(v, rx)), vestaDouble(vestaMul(s1, j)))

    return VestaPointProjective(x: rx, y: ry, z: rz)
}

// Negate affine point
public func vestaPointNegateAffine(_ p: VestaPointAffine) -> VestaPointAffine {
    VestaPointAffine(x: p.x, y: vestaNeg(p.y))
}

// Scalar multiplication (double-and-add)
public func vestaPointMulInt(_ p: VestaPointProjective, _ n: Int) -> VestaPointProjective {
    if n == 0 { return vestaPointIdentity() }
    var result = vestaPointIdentity()
    var base = p
    var scalar = n
    while scalar > 0 {
        if scalar & 1 == 1 {
            result = vestaPointAdd(result, base)
        }
        base = vestaPointDouble(base)
        scalar >>= 1
    }
    return result
}

// Batch projective -> affine using Montgomery's trick
public func batchVestaToAffine(_ points: [VestaPointProjective]) -> [VestaPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var products = [VestaFp](repeating: VestaFp.zero, count: n)
    products[0] = points[0].z
    for i in 1..<n {
        if vestaPointIsIdentity(points[i]) {
            products[i] = products[i - 1]
        } else {
            products[i] = vestaMul(products[i - 1], points[i].z)
        }
    }

    var inv = vestaInverse(products[n - 1])
    var result = [VestaPointAffine](repeating: VestaPointAffine(x: VestaFp.zero, y: VestaFp.zero), count: n)

    for i in stride(from: n - 1, through: 0, by: -1) {
        if vestaPointIsIdentity(points[i]) {
            result[i] = VestaPointAffine(x: VestaFp.zero, y: VestaFp.zero)
            continue
        }
        let zinv = (i == 0) ? inv : vestaMul(inv, products[i - 1])
        if i > 0 {
            inv = vestaMul(inv, points[i].z)
        }
        let zinv2 = vestaSqr(zinv)
        let zinv3 = vestaMul(zinv2, zinv)
        result[i] = VestaPointAffine(x: vestaMul(points[i].x, zinv2),
                                     y: vestaMul(points[i].y, zinv3))
    }
    return result
}

// Vesta generator: G = (-1, 2) — verified: (-1)^3 + 5 = 4 = 2^2
public func vestaGenerator() -> VestaPointAffine {
    // (-1 mod p, 2) in Montgomery form
    let gx = VestaFp(v: (0x00000004, 0x311bac84, 0x2652a376, 0x891a63f0,
                          0x00000000, 0x00000000, 0x00000000, 0x00000000))
    let gy = VestaFp(v: (0xfffffff9, 0x2a0f9218, 0xbcef61f1, 0x1011d11b,
                          0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff))
    return VestaPointAffine(x: gx, y: gy)
}
