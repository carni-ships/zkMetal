// BLS12-377 elliptic curve G1 point operations (CPU-side)
// y^2 = x^3 + 1, Jacobian projective coordinates

import Foundation
import NeonFieldOps

public struct Point377Affine {
    public var x: Fq377
    public var y: Fq377

    public init(x: Fq377, y: Fq377) {
        self.x = x
        self.y = y
    }
}

public struct Point377Projective {
    public var x: Fq377
    public var y: Fq377
    public var z: Fq377

    public init(x: Fq377, y: Fq377, z: Fq377) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func point377Identity() -> Point377Projective {
    Point377Projective(x: .one, y: .one, z: .zero)
}

public func point377IsIdentity(_ p: Point377Projective) -> Bool {
    p.z.isZero
}

public func point377FromAffine(_ a: Point377Affine) -> Point377Projective {
    Point377Projective(x: a.x, y: a.y, z: .one)
}

// Point doubling: a=0 for BLS12-377
public func point377Double(_ p: Point377Projective) -> Point377Projective {
    if point377IsIdentity(p) { return p }

    let a = fq377Sqr(p.x)
    let b = fq377Sqr(p.y)
    let c = fq377Sqr(b)

    let d = fq377Double(fq377Sub(fq377Sqr(fq377Add(p.x, b)), fq377Add(a, c)))
    let e = fq377Add(fq377Double(a), a) // 3*x^2
    let f = fq377Sqr(e)

    let x3 = fq377Sub(f, fq377Double(d))
    let y3 = fq377Sub(fq377Mul(e, fq377Sub(d, x3)), fq377Double(fq377Double(fq377Double(c))))
    let z3 = fq377Sub(fq377Sqr(fq377Add(p.y, p.z)), fq377Add(b, fq377Sqr(p.z)))
    return Point377Projective(x: x3, y: y3, z: z3)
}

// Full addition: projective + projective
public func point377Add(_ p: Point377Projective, _ q: Point377Projective) -> Point377Projective {
    if point377IsIdentity(p) { return q }
    if point377IsIdentity(q) { return p }

    let z1z1 = fq377Sqr(p.z)
    let z2z2 = fq377Sqr(q.z)
    let u1 = fq377Mul(p.x, z2z2)
    let u2 = fq377Mul(q.x, z1z1)
    let s1 = fq377Mul(p.y, fq377Mul(q.z, z2z2))
    let s2 = fq377Mul(q.y, fq377Mul(p.z, z1z1))

    let h = fq377Sub(u2, u1)
    let r = fq377Double(fq377Sub(s2, s1))

    if h.isZero {
        if r.isZero { return point377Double(p) }
        return point377Identity()
    }

    let i = fq377Sqr(fq377Double(h))
    let j = fq377Mul(h, i)
    let vv = fq377Mul(u1, i)

    let x3 = fq377Sub(fq377Sub(fq377Sqr(r), j), fq377Double(vv))
    let y3 = fq377Sub(fq377Mul(r, fq377Sub(vv, x3)), fq377Double(fq377Mul(s1, j)))
    let z3 = fq377Mul(fq377Sub(fq377Sqr(fq377Add(p.z, q.z)), fq377Add(z1z1, z2z2)), h)
    return Point377Projective(x: x3, y: y3, z: z3)
}

// Convert projective to affine: (X/Z^2, Y/Z^3)
public func point377ToAffine(_ p: Point377Projective) -> Point377Affine? {
    if point377IsIdentity(p) { return nil }
    let zinv = fq377Inverse(p.z)
    let zinv2 = fq377Sqr(zinv)
    let zinv3 = fq377Mul(zinv2, zinv)
    return Point377Affine(x: fq377Mul(p.x, zinv2), y: fq377Mul(p.y, zinv3))
}

// Batch convert projective points to affine using Montgomery's trick
public func batch377ToAffine(_ points: [Point377Projective]) -> [Point377Affine] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fq377](repeating: .one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = point377IsIdentity(points[i]) ? prods[i-1] : fq377Mul(prods[i-1], points[i].z)
    }

    var inv = fq377Inverse(prods[n - 1])

    var result = [Point377Affine](repeating: Point377Affine(x: .one, y: .one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if point377IsIdentity(points[i]) { continue }
        let zinv = (i > 0) ? fq377Mul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fq377Mul(inv, points[i].z) }
        let zinv2 = fq377Sqr(zinv)
        let zinv3 = fq377Mul(zinv2, zinv)
        result[i] = Point377Affine(x: fq377Mul(points[i].x, zinv2), y: fq377Mul(points[i].y, zinv3))
    }
    return result
}

// Negate a point in affine: -P = (x, -y)
public func point377NegateAffine(_ p: Point377Affine) -> Point377Affine {
    Point377Affine(x: p.x, y: fq377Neg(p.y))
}

// Scalar multiplication using double-and-add
public func point377MulInt(_ p: Point377Projective, _ n: Int) -> Point377Projective {
    if n == 0 { return point377Identity() }
    if n == 1 { return p }
    var result = point377Identity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = point377IsIdentity(result) ? base : point377Add(result, base)
        }
        base = point377Double(base)
        k >>= 1
    }
    return result
}

// BLS12-377 G1 generator point (in Montgomery form)
public func bls12377Generator() -> Point377Affine {
    // gx = 81937999373150964239938255573465948239988671502647976594219695644855304257327692006745978603320413799295628339695
    // gy = 241266749859715473739788878240585681733927191168601896383759122102112907357779751001206799952863815012735208165030
    let gx = fq377Mul(Fq377.from64([
        0xeab9b16eb21be9ef, 0xd5481512ffcd394e,
        0x188282c8bd37cb5c, 0x85951e2caa9d41bb,
        0xc8fc6225bf87ff54, 0x008848defe740a67
    ]), Fq377.from64(Fq377.R2_MOD_Q))
    let gy = fq377Mul(Fq377.from64([
        0xfd82de55559c8ea6, 0xc2fe3d3634a9591a,
        0x6d182ad44fb82305, 0xbd7fb348ca3e52d9,
        0x1f674f5d30afeec4, 0x01914a69c5102eff
    ]), Fq377.from64(Fq377.R2_MOD_Q))
    return Point377Affine(x: gx, y: gy)
}

// CPU Pippenger MSM for BLS12-377 G1 — uses GLV endomorphism for n > 16
public func bls12377CpuMSM(points: [Point377Affine], scalars: [[UInt32]]) -> Point377Projective {
    let n = points.count
    guard n > 0, n == scalars.count else { return point377Identity() }

    var result = point377Identity()
    points.withUnsafeBufferPointer { ptsBuf in
        let ptsPtr = UnsafeRawPointer(ptsBuf.baseAddress!).assumingMemoryBound(to: UInt64.self)
        var flatScalars = [UInt32](repeating: 0, count: n * 8)
        for i in 0..<n {
            let s = scalars[i]
            for j in 0..<min(s.count, 8) {
                flatScalars[i * 8 + j] = s[j]
            }
        }
        flatScalars.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bls12_377_g1_glv_pippenger_msm(
                    ptsPtr,
                    scBuf.baseAddress!,
                    Int32(n),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}
