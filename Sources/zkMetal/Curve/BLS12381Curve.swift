// BLS12-381 elliptic curve G1 and G2 point operations (CPU-side)
// G1: y^2 = x^3 + 4 over Fp
// G2: y^2 = x^3 + 4(1+u) over Fp2 = Fp[u]/(u^2+1)
// Jacobian projective coordinates: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)

import Foundation
import NeonFieldOps

// MARK: - G1 Point Types

public struct G1Affine381 {
    public var x: Fp381
    public var y: Fp381

    public init(x: Fp381, y: Fp381) {
        self.x = x
        self.y = y
    }
}

public struct G1Projective381 {
    public var x: Fp381
    public var y: Fp381
    public var z: Fp381

    public init(x: Fp381, y: Fp381, z: Fp381) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func g1_381Identity() -> G1Projective381 {
    G1Projective381(x: .one, y: .one, z: .zero)
}

public func g1_381IsIdentity(_ p: G1Projective381) -> Bool {
    p.z.isZero
}

public func g1_381FromAffine(_ a: G1Affine381) -> G1Projective381 {
    G1Projective381(x: a.x, y: a.y, z: .one)
}

// Point doubling for y^2 = x^3 + b (a=0) — C accelerated
public func g1_381Double(_ p: G1Projective381) -> G1Projective381 {
    var result = G1Projective381(x: .one, y: .one, z: .zero)
    withUnsafeBytes(of: p) { pBuf in
        withUnsafeMutableBytes(of: &result) { rBuf in
            bls12_381_g1_point_double(
                pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            )
        }
    }
    return result
}

// Full addition: projective + projective — C accelerated
public func g1_381Add(_ p: G1Projective381, _ q: G1Projective381) -> G1Projective381 {
    var result = G1Projective381(x: .one, y: .one, z: .zero)
    withUnsafeBytes(of: p) { pBuf in
        withUnsafeBytes(of: q) { qBuf in
            withUnsafeMutableBytes(of: &result) { rBuf in
                bls12_381_g1_point_add(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

// Mixed addition: projective + affine (Z=1 optimization) — C accelerated
public func g1_381AddMixed(_ p: G1Projective381, _ q: G1Affine381) -> G1Projective381 {
    var result = G1Projective381(x: .one, y: .one, z: .zero)
    withUnsafeBytes(of: p) { pBuf in
        withUnsafeBytes(of: q) { qBuf in
            withUnsafeMutableBytes(of: &result) { rBuf in
                bls12_381_g1_point_add_mixed(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

// Convert to affine
public func g1_381ToAffine(_ p: G1Projective381) -> G1Affine381? {
    if g1_381IsIdentity(p) { return nil }
    let zinv = fp381Inverse(p.z)
    let zinv2 = fp381Sqr(zinv)
    let zinv3 = fp381Mul(zinv2, zinv)
    return G1Affine381(x: fp381Mul(p.x, zinv2), y: fp381Mul(p.y, zinv3))
}

// Negate a point
public func g1_381NegateAffine(_ p: G1Affine381) -> G1Affine381 {
    G1Affine381(x: p.x, y: fp381Neg(p.y))
}

public func g1_381Negate(_ p: G1Projective381) -> G1Projective381 {
    G1Projective381(x: p.x, y: fp381Neg(p.y), z: p.z)
}

// Scalar multiplication — C accelerated windowed (w=4)
public func g1_381ScalarMul(_ p: G1Projective381, _ scalar: [UInt64]) -> G1Projective381 {
    var result = G1Projective381(x: .one, y: .one, z: .zero)
    withUnsafeBytes(of: p) { pBuf in
        scalar.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { rBuf in
                bls12_381_g1_scalar_mul(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    scBuf.baseAddress!,
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

// Integer scalar mul (for small scalars)
public func g1_381MulInt(_ p: G1Projective381, _ n: Int) -> G1Projective381 {
    if n == 0 { return g1_381Identity() }
    if n == 1 { return p }
    var result = g1_381Identity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = g1_381IsIdentity(result) ? base : g1_381Add(result, base)
        }
        base = g1_381Double(base)
        k >>= 1
    }
    return result
}

// Batch affine conversion using Montgomery's trick
public func batchG1_381ToAffine(_ points: [G1Projective381]) -> [G1Affine381] {
    let n = points.count
    if n == 0 { return [] }

    var prods = [Fp381](repeating: .one, count: n)
    prods[0] = points[0].z
    for i in 1..<n {
        prods[i] = g1_381IsIdentity(points[i]) ? prods[i-1] : fp381Mul(prods[i-1], points[i].z)
    }

    var inv = fp381Inverse(prods[n - 1])
    var result = [G1Affine381](repeating: G1Affine381(x: .one, y: .one), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if g1_381IsIdentity(points[i]) { continue }
        let zinv = (i > 0) ? fp381Mul(inv, prods[i - 1]) : inv
        if i > 0 { inv = fp381Mul(inv, points[i].z) }
        let zinv2 = fp381Sqr(zinv)
        let zinv3 = fp381Mul(zinv2, zinv)
        result[i] = G1Affine381(x: fp381Mul(points[i].x, zinv2), y: fp381Mul(points[i].y, zinv3))
    }
    return result
}

// BLS12-381 G1 generator point
// gx = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
// gy = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
public func bls12381G1Generator() -> G1Affine381 {
    let gx = fp381Mul(Fp381.from64([
        0xfb3af00adb22c6bb, 0x6c55e83ff97a1aef,
        0xa14e3a3f171bac58, 0xc3688c4f9774b905,
        0x2695638c4fa9ac0f, 0x17f1d3a73197d794
    ]), Fp381.from64(Fp381.R2_MOD_P))
    let gy = fp381Mul(Fp381.from64([
        0x0caa232946c5e7e1, 0xd03cc744a2888ae4,
        0x00db18cb2c04b3ed, 0xfcf5e095d5d00af6,
        0xa09e30ed741d8ae4, 0x08b3f481e3aaa0f1
    ]), Fp381.from64(Fp381.R2_MOD_P))
    return G1Affine381(x: gx, y: gy)
}

// MARK: - G2 Point Types

public struct G2Affine381 {
    public var x: Fp2_381
    public var y: Fp2_381

    public init(x: Fp2_381, y: Fp2_381) {
        self.x = x
        self.y = y
    }
}

public struct G2Projective381 {
    public var x: Fp2_381
    public var y: Fp2_381
    public var z: Fp2_381

    public init(x: Fp2_381, y: Fp2_381, z: Fp2_381) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public func g2_381Identity() -> G2Projective381 {
    G2Projective381(x: .one, y: .one, z: .zero)
}

public func g2_381IsIdentity(_ p: G2Projective381) -> Bool {
    p.z.isZero
}

public func g2_381FromAffine(_ a: G2Affine381) -> G2Projective381 {
    G2Projective381(x: a.x, y: a.y, z: .one)
}

// Point doubling for y^2 = x^3 + b' (a=0)
public func g2_381Double(_ p: G2Projective381) -> G2Projective381 {
    if g2_381IsIdentity(p) { return p }

    let a = fp2_381Sqr(p.x)
    let b = fp2_381Sqr(p.y)
    let c = fp2_381Sqr(b)

    let d = fp2_381Double(fp2_381Sub(fp2_381Sqr(fp2_381Add(p.x, b)), fp2_381Add(a, c)))
    let e = fp2_381Add(fp2_381Double(a), a) // 3*x^2
    let f = fp2_381Sqr(e)

    let x3 = fp2_381Sub(f, fp2_381Double(d))
    let y3 = fp2_381Sub(fp2_381Mul(e, fp2_381Sub(d, x3)),
                        fp2_381Double(fp2_381Double(fp2_381Double(c))))
    let z3 = fp2_381Sub(fp2_381Sqr(fp2_381Add(p.y, p.z)), fp2_381Add(b, fp2_381Sqr(p.z)))
    return G2Projective381(x: x3, y: y3, z: z3)
}

// Full addition over Fp2
public func g2_381Add(_ p: G2Projective381, _ q: G2Projective381) -> G2Projective381 {
    if g2_381IsIdentity(p) { return q }
    if g2_381IsIdentity(q) { return p }

    let z1z1 = fp2_381Sqr(p.z)
    let z2z2 = fp2_381Sqr(q.z)
    let u1 = fp2_381Mul(p.x, z2z2)
    let u2 = fp2_381Mul(q.x, z1z1)
    let s1 = fp2_381Mul(p.y, fp2_381Mul(q.z, z2z2))
    let s2 = fp2_381Mul(q.y, fp2_381Mul(p.z, z1z1))

    let h = fp2_381Sub(u2, u1)
    let r = fp2_381Double(fp2_381Sub(s2, s1))

    if h.isZero {
        if r.isZero { return g2_381Double(p) }
        return g2_381Identity()
    }

    let i = fp2_381Sqr(fp2_381Double(h))
    let j = fp2_381Mul(h, i)
    let vv = fp2_381Mul(u1, i)

    let x3 = fp2_381Sub(fp2_381Sub(fp2_381Sqr(r), j), fp2_381Double(vv))
    let y3 = fp2_381Sub(fp2_381Mul(r, fp2_381Sub(vv, x3)), fp2_381Double(fp2_381Mul(s1, j)))
    let z3 = fp2_381Mul(fp2_381Sub(fp2_381Sqr(fp2_381Add(p.z, q.z)),
                                    fp2_381Add(z1z1, z2z2)), h)
    return G2Projective381(x: x3, y: y3, z: z3)
}

// Convert to affine
public func g2_381ToAffine(_ p: G2Projective381) -> G2Affine381? {
    if g2_381IsIdentity(p) { return nil }
    let zinv = fp2_381Inverse(p.z)
    let zinv2 = fp2_381Sqr(zinv)
    let zinv3 = fp2_381Mul(zinv2, zinv)
    return G2Affine381(x: fp2_381Mul(p.x, zinv2), y: fp2_381Mul(p.y, zinv3))
}

// Negate G2 point
public func g2_381Negate(_ p: G2Projective381) -> G2Projective381 {
    G2Projective381(x: p.x, y: fp2_381Neg(p.y), z: p.z)
}

public func g2_381NegateAffine(_ p: G2Affine381) -> G2Affine381 {
    G2Affine381(x: p.x, y: fp2_381Neg(p.y))
}

// Scalar multiplication for G2
public func g2_381ScalarMul(_ p: G2Projective381, _ scalar: [UInt64]) -> G2Projective381 {
    var result = g2_381Identity()
    var base = p
    for i in 0..<scalar.count {
        var word = scalar[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = g2_381Add(result, base)
            }
            base = g2_381Double(base)
            word >>= 1
        }
    }
    return result
}

// Integer scalar mul for G2
public func g2_381MulInt(_ p: G2Projective381, _ n: Int) -> G2Projective381 {
    if n == 0 { return g2_381Identity() }
    if n == 1 { return p }
    var result = g2_381Identity()
    var base = p
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = g2_381IsIdentity(result) ? base : g2_381Add(result, base)
        }
        base = g2_381Double(base)
        k >>= 1
    }
    return result
}

// Standard BLS12-381 G2 generator on the twist curve E': y^2 = x^3 + 4(1+u)
// This is the well-known generator in the r-torsion subgroup.
// x = (0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8,
//      0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e)
// y = (0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801,
//      0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be)
public func bls12381G2Generator() -> G2Affine381 {
    let xc0 = fp381Mul(Fp381.from64([
        0xd48056c8c121bdb8, 0x0bac0326a805bbef,
        0xb4510b647ae3d177, 0xc6e47ad4fa403b02,
        0x260805272dc51051, 0x024aa2b2f08f0a91
    ]), Fp381.from64(Fp381.R2_MOD_P))
    let xc1 = fp381Mul(Fp381.from64([
        0xe5ac7d055d042b7e, 0x334cf11213945d57,
        0xb5da61bbdc7f5049, 0x596bd0d09920b61a,
        0x7dacd3a088274f65, 0x13e02b6052719f60
    ]), Fp381.from64(Fp381.R2_MOD_P))
    let yc0 = fp381Mul(Fp381.from64([
        0xe193548608b82801, 0x923ac9cc3baca289,
        0x6d429a695160d12c, 0xadfd9baa8cbdd3a7,
        0x8cc9cdc6da2e351a, 0x0ce5d527727d6e11
    ]), Fp381.from64(Fp381.R2_MOD_P))
    let yc1 = fp381Mul(Fp381.from64([
        0xaaa9075ff05f79be, 0x3f370d275cec1da1,
        0x267492ab572e99ab, 0xcb3e287e85a763af,
        0x32acd2b02bc28b99, 0x0606c4a02ea734cc
    ]), Fp381.from64(Fp381.R2_MOD_P))
    return G2Affine381(x: Fp2_381(c0: xc0, c1: xc1), y: Fp2_381(c0: yc0, c1: yc1))
}

/// Alias for G2 generator, used in tests.
public func bls12381G2SimplePoint() -> G2Affine381 {
    bls12381G2Generator()
}
