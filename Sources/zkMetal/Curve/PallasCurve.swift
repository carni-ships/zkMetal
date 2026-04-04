// Pallas elliptic curve operations (CPU-side)
// y^2 = x^3 + 5
// Jacobian projective coordinates: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)

import Foundation

public struct PallasPointAffine {
    public var x: PallasFp
    public var y: PallasFp

    public init(x: PallasFp, y: PallasFp) {
        self.x = x
        self.y = y
    }
}

public struct PallasPointProjective {
    public var x: PallasFp
    public var y: PallasFp
    public var z: PallasFp

    public init(x: PallasFp, y: PallasFp, z: PallasFp) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func pallasPointIdentity() -> PallasPointProjective {
    PallasPointProjective(x: PallasFp.one, y: PallasFp.one, z: PallasFp.zero)
}

public func pallasPointIsIdentity(_ p: PallasPointProjective) -> Bool {
    p.z.isZero
}

public func pallasPointFromAffine(_ a: PallasPointAffine) -> PallasPointProjective {
    PallasPointProjective(x: a.x, y: a.y, z: PallasFp.one)
}

public func pallasPointToAffine(_ p: PallasPointProjective) -> PallasPointAffine {
    if pallasPointIsIdentity(p) {
        return PallasPointAffine(x: PallasFp.zero, y: PallasFp.zero)
    }
    let zinv = pallasInverse(p.z)
    let zinv2 = pallasSqr(zinv)
    let zinv3 = pallasMul(zinv2, zinv)
    return PallasPointAffine(x: pallasMul(p.x, zinv2), y: pallasMul(p.y, zinv3))
}

// Point doubling (a=0 for Pallas: y^2 = x^3 + 5)
public func pallasPointDouble(_ p: PallasPointProjective) -> PallasPointProjective {
    if pallasPointIsIdentity(p) { return p }

    let a = pallasSqr(p.x)
    let b = pallasSqr(p.y)
    let c = pallasSqr(b)

    let xpb = pallasAdd(p.x, b)
    let d = pallasDouble(pallasSub(pallasSqr(xpb), pallasAdd(a, c)))

    let e = pallasAdd(pallasDouble(a), a) // 3*X^2
    let f = pallasSqr(e)

    let rx = pallasSub(f, pallasDouble(d))
    let ry = pallasSub(pallasMul(e, pallasSub(d, rx)),
                     pallasDouble(pallasDouble(pallasDouble(c))))
    let yz = pallasAdd(p.y, p.z)
    let rz = pallasSub(pallasSqr(yz), pallasAdd(b, pallasSqr(p.z)))

    return PallasPointProjective(x: rx, y: ry, z: rz)
}

// Full addition: projective + projective
public func pallasPointAdd(_ p: PallasPointProjective, _ q: PallasPointProjective) -> PallasPointProjective {
    if pallasPointIsIdentity(p) { return q }
    if pallasPointIsIdentity(q) { return p }

    let z1z1 = pallasSqr(p.z)
    let z2z2 = pallasSqr(q.z)
    let u1 = pallasMul(p.x, z2z2)
    let u2 = pallasMul(q.x, z1z1)
    let s1 = pallasMul(p.y, pallasMul(q.z, z2z2))
    let s2 = pallasMul(q.y, pallasMul(p.z, z1z1))

    let h = pallasSub(u2, u1)
    let rr = pallasDouble(pallasSub(s2, s1))

    if h.isZero {
        if rr.isZero { return pallasPointDouble(p) }
        return pallasPointIdentity()
    }

    let rz = pallasMul(pallasDouble(pallasMul(p.z, q.z)), h)
    let dh = pallasDouble(h)
    let i = pallasSqr(dh)
    let v = pallasMul(u1, i)
    let j = pallasMul(h, i)

    let rx = pallasSub(pallasSub(pallasSqr(rr), j), pallasDouble(v))
    let ry = pallasSub(pallasMul(rr, pallasSub(v, rx)), pallasDouble(pallasMul(s1, j)))

    return PallasPointProjective(x: rx, y: ry, z: rz)
}

// Negate affine point
public func pallasPointNegateAffine(_ p: PallasPointAffine) -> PallasPointAffine {
    PallasPointAffine(x: p.x, y: pallasNeg(p.y))
}

// Scalar multiplication (double-and-add)
public func pallasPointMulInt(_ p: PallasPointProjective, _ n: Int) -> PallasPointProjective {
    if n == 0 { return pallasPointIdentity() }
    var result = pallasPointIdentity()
    var base = p
    var scalar = n
    while scalar > 0 {
        if scalar & 1 == 1 {
            result = pallasPointAdd(result, base)
        }
        base = pallasPointDouble(base)
        scalar >>= 1
    }
    return result
}

// Batch projective -> affine using Montgomery's trick
public func batchPallasToAffine(_ points: [PallasPointProjective]) -> [PallasPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var products = [PallasFp](repeating: PallasFp.zero, count: n)
    products[0] = points[0].z
    for i in 1..<n {
        if pallasPointIsIdentity(points[i]) {
            products[i] = products[i - 1]
        } else {
            products[i] = pallasMul(products[i - 1], points[i].z)
        }
    }

    var inv = pallasInverse(products[n - 1])
    var result = [PallasPointAffine](repeating: PallasPointAffine(x: PallasFp.zero, y: PallasFp.zero), count: n)

    for i in stride(from: n - 1, through: 0, by: -1) {
        if pallasPointIsIdentity(points[i]) {
            result[i] = PallasPointAffine(x: PallasFp.zero, y: PallasFp.zero)
            continue
        }
        let zinv = (i == 0) ? inv : pallasMul(inv, products[i - 1])
        if i > 0 {
            inv = pallasMul(inv, points[i].z)
        }
        let zinv2 = pallasSqr(zinv)
        let zinv3 = pallasMul(zinv2, zinv)
        result[i] = PallasPointAffine(x: pallasMul(points[i].x, zinv2),
                                      y: pallasMul(points[i].y, zinv3))
    }
    return result
}

// Pallas generator: G = (-1, 2) — verified: (-1)^3 + 5 = 4 = 2^2
public func pallasGenerator() -> PallasPointAffine {
    // (-1 mod p, 2) in Montgomery form
    let gx = PallasFp(v: (0x00000004, 0x64b4c3b4, 0x2533e46e, 0x891a63f0,
                           0x00000000, 0x00000000, 0x00000000, 0x00000000))
    let gy = PallasFp(v: (0xfffffff9, 0xcfc3a984, 0xbee5303e, 0x1011d11b,
                           0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff))
    return PallasPointAffine(x: gx, y: gy)
}
