// BN254 elliptic curve point operations (CPU-side)
// y^2 = x^3 + 3, Jacobian projective coordinates

import Foundation

public struct PointAffine {
    public var x: Fp
    public var y: Fp

    public init(x: Fp, y: Fp) {
        self.x = x
        self.y = y
    }
}

public struct PointProjective {
    public var x: Fp
    public var y: Fp
    public var z: Fp

    public init(x: Fp, y: Fp, z: Fp) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public struct MsmParams {
    public var n_points: UInt32
    public var window_bits: UInt32
    public var n_buckets: UInt32

    public init(n_points: UInt32, window_bits: UInt32, n_buckets: UInt32) {
        self.n_points = n_points
        self.window_bits = window_bits
        self.n_buckets = n_buckets
    }
}

public func pointIdentity() -> PointProjective {
    PointProjective(x: .one, y: .one, z: .zero)
}

public func pointIsIdentity(_ p: PointProjective) -> Bool {
    p.z.isZero
}

public func pointFromAffine(_ a: PointAffine) -> PointProjective {
    PointProjective(x: a.x, y: a.y, z: .one)
}

// Point doubling: a=0 for BN254
public func pointDouble(_ p: PointProjective) -> PointProjective {
    if pointIsIdentity(p) { return p }

    let a = fpSqr(p.x)
    let b = fpSqr(p.y)
    let c = fpSqr(b)

    let d = fpDouble(fpSub(fpSqr(fpAdd(p.x, b)), fpAdd(a, c)))
    let e = fpAdd(fpDouble(a), a) // 3*x^2
    let f = fpSqr(e)

    let x3 = fpSub(f, fpDouble(d))
    let y3 = fpSub(fpMul(e, fpSub(d, x3)), fpDouble(fpDouble(fpDouble(c))))
    let z3 = fpSub(fpSqr(fpAdd(p.y, p.z)), fpAdd(b, fpSqr(p.z)))
    return PointProjective(x: x3, y: y3, z: z3)
}

// Full addition: projective + projective
public func pointAdd(_ p: PointProjective, _ q: PointProjective) -> PointProjective {
    if pointIsIdentity(p) { return q }
    if pointIsIdentity(q) { return p }

    let z1z1 = fpSqr(p.z)
    let z2z2 = fpSqr(q.z)
    let u1 = fpMul(p.x, z2z2)
    let u2 = fpMul(q.x, z1z1)
    let s1 = fpMul(p.y, fpMul(q.z, z2z2))
    let s2 = fpMul(q.y, fpMul(p.z, z1z1))

    let h = fpSub(u2, u1)
    let r = fpDouble(fpSub(s2, s1))

    if h.isZero {
        if r.isZero { return pointDouble(p) }
        return pointIdentity()
    }

    let i = fpSqr(fpDouble(h))
    let j = fpMul(h, i)
    let vv = fpMul(u1, i)

    let x3 = fpSub(fpSub(fpSqr(r), j), fpDouble(vv))
    let y3 = fpSub(fpMul(r, fpSub(vv, x3)), fpDouble(fpMul(s1, j)))
    let z3 = fpMul(fpSub(fpSqr(fpAdd(p.z, q.z)), fpAdd(z1z1, z2z2)), h)
    return PointProjective(x: x3, y: y3, z: z3)
}

/// Scalar multiplication using double-and-add. O(log n).
public func pointMulInt(_ p: PointProjective, _ n: Int) -> PointProjective {
    if n == 0 { return pointIdentity() }
    if n == 1 { return p }
    var result = pointIdentity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = pointIsIdentity(result) ? base : pointAdd(result, base)
        }
        base = pointDouble(base)
        k >>= 1
    }
    return result
}

// Convert projective to affine: (X/Z^2, Y/Z^3)
public func pointToAffine(_ p: PointProjective) -> PointAffine? {
    if pointIsIdentity(p) { return nil }
    let zinv = fpInverse(p.z)
    let zinv2 = fpSqr(zinv)
    let zinv3 = fpMul(zinv2, zinv)
    return PointAffine(x: fpMul(p.x, zinv2), y: fpMul(p.y, zinv3))
}

/// Batch convert projective points to affine using Montgomery's trick (single inversion).
public func batchToAffine(_ points: [PointProjective]) -> [PointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fp](repeating: .one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = pointIsIdentity(points[i]) ? prods[i-1] : fpMul(prods[i-1], points[i].z)
    }

    var inv = fpInverse(prods[n - 1])

    var result = [PointAffine](repeating: PointAffine(x: .one, y: .one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if pointIsIdentity(points[i]) {
            continue
        }
        let zinv = (i > 0) ? fpMul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fpMul(inv, points[i].z) }
        let zinv2 = fpSqr(zinv)
        let zinv3 = fpMul(zinv2, zinv)
        result[i] = PointAffine(x: fpMul(points[i].x, zinv2), y: fpMul(points[i].y, zinv3))
    }
    return result
}

// Negate a point in projective: -P = (x, -y, z)
public func pointNeg(_ p: PointProjective) -> PointProjective {
    return PointProjective(x: p.x, y: fpNeg(p.y), z: p.z)
}

// Negate a point in affine: -P = (x, -y)
public func pointNegateAffine(_ p: PointAffine) -> PointAffine {
    return PointAffine(x: p.x, y: fpNeg(p.y))
}
