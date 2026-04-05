// BLS12-377 scalar field Fr377 arithmetic (CPU-side)
// r = 8444461749428370424248824938781546531375899335154063827935233455917409239041
// Used for NTT and scalar operations.

import Foundation
import NeonFieldOps

public struct Fr377 {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0x0a11800000000001, 0x59aa76fed0000001,
        0x60b44d1e5c37b001, 0x12ab655e9a2ca556
    ]

    // R mod r (Montgomery form of 1)
    public static let R_MOD_R: [UInt64] = [
        0x7d1c7ffffffffff3, 0x7257f50f6ffffff2,
        0x16d81575512c0fee, 0x0d4bda322bbb9a9d
    ]

    // R^2 mod r
    public static let R2_MOD_R: [UInt64] = [
        0x25d577bab861857b, 0xcc2c27b58860591f,
        0xa7cc008fe5dc8593, 0x011fdae7eff1c939
    ]

    // -r^(-1) mod 2^64
    public static let INV: UInt64 = 0x0a117fffffffffff

    // TWO_ADICITY: r - 1 = 2^47 * t (t odd)
    public static let TWO_ADICITY: Int = 47

    // Primitive 2^47-th root of unity in Montgomery form
    // = GENERATOR^((r-1)/2^47) mod r (computed and verified)
    // Stored in standard form; converted to Montgomery at runtime.
    public static let ROOT_OF_UNITY: [UInt64] = [
        0x476ef4a4ec2a895e, 0x9b506ee363e3f04a,
        0x60c69477d1a8a12f, 0x11d4b7f60cb92cc1
    ]

    // Multiplicative generator of Fr377*
    public static let GENERATOR: UInt64 = 22

    public static var zero: Fr377 { Fr377(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: Fr377 {
        Fr377(v: (0xfffffff3, 0x7d1c7fff, 0x6ffffff2, 0x7257f50f,
                  0x512c0fee, 0x16d81575, 0x2bbb9a9d, 0x0d4bda32))
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

    public static func from64(_ limbs: [UInt64]) -> Fr377 {
        Fr377(v: (
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

// MARK: - Fr377 Field Operations

public func fr377Mul(_ a: Fr377, _ b: Fr377) -> Fr377 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    bls12_377_fr_mul(&al, &bl, &r)
    return Fr377.from64(r)
}

public func fr377Add(_ a: Fr377, _ b: Fr377) -> Fr377 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    bls12_377_fr_add(&al, &bl, &r)
    return Fr377.from64(r)
}

public func fr377Sub(_ a: Fr377, _ b: Fr377) -> Fr377 {
    var al = a.to64(), bl = b.to64()
    var r = [UInt64](repeating: 0, count: 4)
    bls12_377_fr_sub(&al, &bl, &r)
    return Fr377.from64(r)
}

public func fr377Sqr(_ a: Fr377) -> Fr377 {
    var al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    bls12_377_fr_sqr(&al, &r)
    return Fr377.from64(r)
}

public func fr377Neg(_ a: Fr377) -> Fr377 {
    var al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    bls12_377_fr_neg(&al, &r)
    return Fr377.from64(r)
}

public func fr377FromInt(_ val: UInt64) -> Fr377 {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = Fr377.from64(limbs)
    return fr377Mul(raw, Fr377.from64(Fr377.R2_MOD_R))
}

public func fr377ToInt(_ a: Fr377) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return fr377Mul(a, Fr377.from64(one)).to64()
}

public func fr377Inverse(_ a: Fr377) -> Fr377 {
    var result = Fr377.one
    var base = a
    var exp = Fr377.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = fr377Mul(result, base)
            }
            base = fr377Sqr(base)
            word >>= 1
        }
    }
    return result
}

/// Compute a^n mod r using square-and-multiply.
public func fr377Pow(_ a: Fr377, _ n: UInt64) -> Fr377 {
    if n == 0 { return Fr377.one }
    if n == 1 { return a }
    var result = Fr377.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = fr377Mul(result, base)
        }
        base = fr377Sqr(base)
        k >>= 1
    }
    return result
}

/// Get the primitive 2^k-th root of unity (k <= TWO_ADICITY=47).
public func fr377RootOfUnity(logN: Int) -> Fr377 {
    precondition(logN <= Fr377.TWO_ADICITY, "logN exceeds TWO_ADICITY")
    // ROOT_OF_UNITY is in standard form; convert to Montgomery
    var omega = fr377Mul(Fr377.from64(Fr377.ROOT_OF_UNITY), Fr377.from64(Fr377.R2_MOD_R))
    for _ in 0..<(Fr377.TWO_ADICITY - logN) {
        omega = fr377Sqr(omega)
    }
    return omega
}

/// Precompute twiddle factors: [omega^0, omega^1, ..., omega^(n - 1)] in Montgomery form.
/// Full N entries needed for four-step FFT twiddle multiply.
public func precomputeTwiddles377(logN: Int) -> [Fr377] {
    let n = 1 << logN
    let omega = fr377RootOfUnity(logN: logN)
    var twiddles = [Fr377](repeating: Fr377.one, count: n)
    for i in 1..<n {
        twiddles[i] = fr377Mul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for iNTT.
/// Full N entries needed for four-step iFFT inverse twiddle multiply.
public func precomputeInverseTwiddles377(logN: Int) -> [Fr377] {
    let n = 1 << logN
    let omega = fr377RootOfUnity(logN: logN)
    let omegaInv = fr377Inverse(omega)
    var twiddles = [Fr377](repeating: Fr377.one, count: n)
    for i in 1..<n {
        twiddles[i] = fr377Mul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
