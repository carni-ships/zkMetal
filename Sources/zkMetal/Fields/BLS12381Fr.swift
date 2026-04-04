// BLS12-381 scalar field Fr arithmetic (CPU-side)
// r = 52435875175126190479447740508185965837690552500527637822603658699938581184513
// 255-bit prime, field elements as 8x32-bit limbs in Montgomery form (little-endian).
// Used for NTT, scalar operations, and pairing arithmetic.

import Foundation

public struct Fr381 {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    // r in 4x64-bit limbs (little-endian)
    public static let P: [UInt64] = [
        0xffffffff00000001, 0x53bda402fffe5bfe,
        0x3339d80809a1d805, 0x73eda753299d7d48
    ]

    // R mod r (Montgomery form of 1): 2^256 mod r
    public static let R_MOD_R: [UInt64] = [
        0x00000001fffffffe, 0x5884b7fa00034802,
        0x998c4fefecbc4ff5, 0x1824b159acc5056f
    ]

    // R^2 mod r: 2^512 mod r
    public static let R2_MOD_R: [UInt64] = [
        0xc999e990f3f29c6d, 0x2b6cedcb87925c23,
        0x05d314967254398f, 0x0748d9d99f59ff11
    ]

    // -r^(-1) mod 2^64
    public static let INV: UInt64 = 0xfffffffeffffffff

    // TWO_ADICITY: r - 1 = 2^32 * t (t odd)
    public static let TWO_ADICITY: Int = 32

    // Primitive 2^32-th root of unity in standard form
    // = 7^((r-1)/2^32) mod r
    public static let ROOT_OF_UNITY: [UInt64] = [
        0x3829971f439f0d2b, 0xb63683508c2280b9,
        0xd09b681922c813b4, 0x16a2a19edfe81f20
    ]

    // Multiplicative generator of Fr381*
    public static let GENERATOR: UInt64 = 7

    public static var zero: Fr381 { Fr381(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: Fr381 {
        // R mod r in 32-bit limbs (little-endian)
        Fr381(v: (0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa,
                  0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159))
    }

    public init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    public func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
        ]
    }

    public static func from64(_ limbs: [UInt64]) -> Fr381 {
        Fr381(v: (
            UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
            UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
            UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
            UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32)
        ))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0 &&
        v.4 == 0 && v.5 == 0 && v.6 == 0 && v.7 == 0
    }
}

// MARK: - Fr381 Field Operations

public func fr381Mul(_ a: Fr381, _ b: Fr381) -> Fr381 {
    let al = a.to64(), bl = b.to64()
    var t = [UInt64](repeating: 0, count: 5)

    for i in 0..<4 {
        var carry: UInt64 = 0
        for j in 0..<4 {
            let (hi, lo) = al[i].multipliedFullWidth(by: bl[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        let m = t[0] &* Fr381.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: Fr381.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = 0
    }

    var r = Array(t[0..<4])
    if gte256(r, Fr381.P) {
        (r, _) = sub256(r, Fr381.P)
    }
    return Fr381.from64(r)
}

public func fr381Add(_ a: Fr381, _ b: Fr381) -> Fr381 {
    var (r, carry) = add256(a.to64(), b.to64())
    if carry != 0 || gte256(r, Fr381.P) {
        (r, _) = sub256(r, Fr381.P)
    }
    return Fr381.from64(r)
}

public func fr381Sub(_ a: Fr381, _ b: Fr381) -> Fr381 {
    var (r, borrow) = sub256(a.to64(), b.to64())
    if borrow {
        (r, _) = add256(r, Fr381.P)
    }
    return Fr381.from64(r)
}

public func fr381Sqr(_ a: Fr381) -> Fr381 { fr381Mul(a, a) }
public func fr381Double(_ a: Fr381) -> Fr381 { fr381Add(a, a) }

public func fr381Neg(_ a: Fr381) -> Fr381 {
    if a.isZero { return a }
    let (r, _) = sub256(Fr381.P, a.to64())
    return Fr381.from64(r)
}

public func fr381FromInt(_ val: UInt64) -> Fr381 {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = Fr381.from64(limbs)
    return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
}

public func fr381ToInt(_ a: Fr381) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return fr381Mul(a, Fr381.from64(one)).to64()
}

public func fr381Inverse(_ a: Fr381) -> Fr381 {
    var result = Fr381.one
    var base = a
    var exp = Fr381.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = fr381Mul(result, base)
            }
            base = fr381Sqr(base)
            word >>= 1
        }
    }
    return result
}

/// Compute a^n mod r using square-and-multiply.
public func fr381Pow(_ a: Fr381, _ n: UInt64) -> Fr381 {
    if n == 0 { return Fr381.one }
    if n == 1 { return a }
    var result = Fr381.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = fr381Mul(result, base)
        }
        base = fr381Sqr(base)
        k >>= 1
    }
    return result
}

/// Get the primitive 2^k-th root of unity (k <= TWO_ADICITY=32).
public func fr381RootOfUnity(logN: Int) -> Fr381 {
    precondition(logN <= Fr381.TWO_ADICITY, "logN exceeds TWO_ADICITY")
    // ROOT_OF_UNITY is in standard form; convert to Montgomery
    var omega = fr381Mul(Fr381.from64(Fr381.ROOT_OF_UNITY), Fr381.from64(Fr381.R2_MOD_R))
    for _ in 0..<(Fr381.TWO_ADICITY - logN) {
        omega = fr381Sqr(omega)
    }
    return omega
}

/// Precompute twiddle factors for NTT.
public func precomputeTwiddles381(logN: Int) -> [Fr381] {
    let n = 1 << logN
    let omega = fr381RootOfUnity(logN: logN)
    var twiddles = [Fr381](repeating: Fr381.one, count: n)
    for i in 1..<n {
        twiddles[i] = fr381Mul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for iNTT.
public func precomputeInverseTwiddles381(logN: Int) -> [Fr381] {
    let n = 1 << logN
    let omega = fr381RootOfUnity(logN: logN)
    let omegaInv = fr381Inverse(omega)
    var twiddles = [Fr381](repeating: Fr381.one, count: n)
    for i in 1..<n {
        twiddles[i] = fr381Mul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
