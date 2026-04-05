// StarkNet/Cairo native field (Stark252) arithmetic (CPU-side)
// p = 2^251 + 17 * 2^192 + 1
//   = 3618502788666131213697322783095070105623107215331596699973092056135872020481
// Used for StarkNet compatibility, NTT, and scalar operations.

import Foundation
import NeonFieldOps

public struct Stark252 {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    // p in 64-bit limbs (little-endian)
    public static let P: [UInt64] = [
        0x0000000000000001, 0x0000000000000000,
        0x0000000000000000, 0x0800000000000011
    ]

    // R mod p (Montgomery form of 1) = 2^256 mod p
    public static let R_MOD_P: [UInt64] = [
        0xffffffffffffffe1, 0xffffffffffffffff,
        0xffffffffffffffff, 0x07fffffffffffdf0
    ]

    // R^2 mod p
    public static let R2_MOD_P: [UInt64] = [
        0xfffffd737e000401, 0x00000001330fffff,
        0xffffffffff6f8000, 0x07ffd4ab5e008810
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0xffffffffffffffff

    // TWO_ADICITY: p - 1 = 2^192 * t (t odd, t = 576460752303423505)
    public static let TWO_ADICITY: Int = 192

    // Primitive 2^192-th root of unity in Montgomery form
    // = 3^((p-1)/2^192) mod p, then converted to Montgomery form
    public static let ROOT_OF_UNITY: [UInt64] = [
        0x4106bccd64a2bdd8, 0xaaada25731fe3be9,
        0x0a35c5be60505574, 0x07222e32c47afc26
    ]

    // Multiplicative generator of Fp*
    public static let GENERATOR: UInt64 = 3

    public static var zero: Stark252 { Stark252(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    // Montgomery form of 1 = R mod p
    public static var one: Stark252 {
        Stark252(v: (0xffffffe1, 0xffffffff, 0xffffffff, 0xffffffff,
                     0xffffffff, 0xffffffff, 0xfffffdf0, 0x07ffffff))
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

    public static func from64(_ limbs: [UInt64]) -> Stark252 {
        Stark252(v: (
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

// MARK: - Stark252 Field Operations

public func stark252Mul(_ a: Stark252, _ b: Stark252) -> Stark252 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    stark252_fp_mul(&al, &bl, &r)
    return Stark252.from64(r)
}

public func stark252Add(_ a: Stark252, _ b: Stark252) -> Stark252 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    stark252_fp_add(&al, &bl, &r)
    return Stark252.from64(r)
}

public func stark252Sub(_ a: Stark252, _ b: Stark252) -> Stark252 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    stark252_fp_sub(&al, &bl, &r)
    return Stark252.from64(r)
}

public func stark252Sqr(_ a: Stark252) -> Stark252 {
    var al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    stark252_fp_sqr(&al, &r)
    return Stark252.from64(r)
}

public func stark252Neg(_ a: Stark252) -> Stark252 {
    var al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    stark252_fp_neg(&al, &r)
    return Stark252.from64(r)
}

public func stark252FromInt(_ val: UInt64) -> Stark252 {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = Stark252.from64(limbs)
    return stark252Mul(raw, Stark252.from64(Stark252.R2_MOD_P))
}

public func stark252ToInt(_ a: Stark252) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return stark252Mul(a, Stark252.from64(one)).to64()
}

/// Fast extraction of low 64-bit integer value from Stark252.
@inline(__always)
public func stark252ToUInt64(_ a: Stark252) -> UInt64 {
    let rawOne = Stark252(v: (1, 0, 0, 0, 0, 0, 0, 0))
    let reduced = stark252Mul(a, rawOne)
    return UInt64(reduced.v.0) | (UInt64(reduced.v.1) << 32)
}

public func stark252Inverse(_ a: Stark252) -> Stark252 {
    // a^(p-2) mod p via Fermat's little theorem
    var result = Stark252.one
    var base = a
    // p-2 = 0x0800000000000010_ffffffffffffffff_ffffffffffffffff_ffffffffffffffff
    let exp: [UInt64] = [
        0xffffffffffffffff, 0xffffffffffffffff,
        0xffffffffffffffff, 0x0800000000000010
    ]

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = stark252Mul(result, base)
            }
            base = stark252Sqr(base)
            word >>= 1
        }
    }
    return result
}

/// Montgomery batch inversion using a single Fermat inverse.
public func stark252BatchInverse(_ a: [Stark252]) -> [Stark252] {
    let n = a.count
    if n == 0 { return [] }
    if n == 1 { return [stark252Inverse(a[0])] }

    var prefix = [Stark252](repeating: Stark252.zero, count: n)
    prefix[0] = a[0]
    for i in 1..<n {
        prefix[i] = stark252Mul(prefix[i - 1], a[i])
    }

    var inv = stark252Inverse(prefix[n - 1])

    var result = [Stark252](repeating: Stark252.zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = stark252Mul(inv, prefix[i - 1])
        inv = stark252Mul(inv, a[i])
    }
    result[0] = inv

    return result
}

/// Compute a^n mod p using square-and-multiply.
public func stark252Pow(_ a: Stark252, _ n: UInt64) -> Stark252 {
    if n == 0 { return Stark252.one }
    if n == 1 { return a }
    var result = Stark252.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = stark252Mul(result, base)
        }
        base = stark252Sqr(base)
        k >>= 1
    }
    return result
}

/// Tonelli-Shanks sqrt. Returns nil if a is not a QR.
public func stark252Sqrt(_ a: Stark252) -> Stark252? {
    // p mod 4: p = 2^251 + 17*2^192 + 1. p mod 4 = 1, so we need Tonelli-Shanks.
    // TWO_ADICITY = 192, which is very high. For practical use, Cipolla or
    // a direct exponentiation-based approach may be preferred.
    // Here we use the simple Tonelli-Shanks.

    if a.isZero { return Stark252.zero }

    // Check if a is a quadratic residue: a^((p-1)/2) == 1
    let pm1over2 = Stark252.from64([
        0x0000000000000000, 0x8000000000000000,
        0x0000000000000000, 0x0400000000000008
    ])
    let legendre = stark252PowBig(a, pm1over2.to64())
    let legendreInt = stark252ToInt(legendre)
    if !(legendreInt[0] == 1 && legendreInt[1] == 0 && legendreInt[2] == 0 && legendreInt[3] == 0) {
        return nil // not a QR
    }

    // Tonelli-Shanks with s=192, q = (p-1)/2^192 = 576460752303423505
    let s = 192
    let q: UInt64 = 576460752303423505 // (p-1) >> 192

    // Find a non-residue z. 3 is the generator, so 3^((p-1)/2) = -1.
    // Actually, we need to verify. Let's use the generator.
    let z = stark252FromInt(3)

    var m = s
    var c = stark252Pow(z, q) // z^q (the 2^s-th root of unity path)
    var t = stark252Pow(a, q)
    var r = stark252Pow(a, (q + 1) / 2)

    while true {
        let tInt = stark252ToInt(t)
        if tInt[0] == 1 && tInt[1] == 0 && tInt[2] == 0 && tInt[3] == 0 {
            return r
        }

        // Find least i such that t^(2^i) = 1
        var i = 0
        var tmp = t
        while true {
            i += 1
            tmp = stark252Sqr(tmp)
            let tmpInt = stark252ToInt(tmp)
            if tmpInt[0] == 1 && tmpInt[1] == 0 && tmpInt[2] == 0 && tmpInt[3] == 0 {
                break
            }
        }

        if i == m { return nil } // should not happen for a QR

        // b = c^(2^(m-i-1))
        var b = c
        for _ in 0..<(m - i - 1) {
            b = stark252Sqr(b)
        }

        m = i
        c = stark252Sqr(b)
        t = stark252Mul(t, c)
        r = stark252Mul(r, b)
    }
}

/// Exponentiation with a 256-bit exponent (4x64-bit limbs, little-endian).
public func stark252PowBig(_ a: Stark252, _ exp: [UInt64]) -> Stark252 {
    var result = Stark252.one
    var base = a
    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = stark252Mul(result, base)
            }
            base = stark252Sqr(base)
            word >>= 1
        }
    }
    return result
}

/// Get the primitive 2^k-th root of unity (k <= TWO_ADICITY=192).
public func stark252RootOfUnity(logN: Int) -> Stark252 {
    precondition(logN <= Stark252.TWO_ADICITY, "logN exceeds TWO_ADICITY")
    // ROOT_OF_UNITY is already in Montgomery form
    var omega = Stark252.from64(Stark252.ROOT_OF_UNITY)
    for _ in 0..<(Stark252.TWO_ADICITY - logN) {
        omega = stark252Sqr(omega)
    }
    return omega
}

/// Precompute twiddle factors: [omega^0, omega^1, ..., omega^(n-1)] in Montgomery form.
public func precomputeTwiddlesStark252(logN: Int) -> [Stark252] {
    let n = 1 << logN
    let omega = stark252RootOfUnity(logN: logN)
    var twiddles = [Stark252](repeating: Stark252.one, count: n)
    for i in 1..<n {
        twiddles[i] = stark252Mul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for iNTT.
public func precomputeInverseTwiddlesStark252(logN: Int) -> [Stark252] {
    let n = 1 << logN
    let omega = stark252RootOfUnity(logN: logN)
    let omegaInv = stark252Inverse(omega)
    var twiddles = [Stark252](repeating: Stark252.one, count: n)
    for i in 1..<n {
        twiddles[i] = stark252Mul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
