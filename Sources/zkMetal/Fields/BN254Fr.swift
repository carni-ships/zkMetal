// BN254 scalar field Fr arithmetic (CPU-side)
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Used for NTT and scalar operations.

import Foundation
import NeonFieldOps

public struct Fr {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0x43e1f593f0000001, 0x2833e84879b97091,
        0xb85045b68181585d, 0x30644e72e131a029
    ]

    // R mod r (Montgomery form of 1)
    public static let R_MOD_R: [UInt64] = [
        0xac96341c4ffffffb, 0x36fc76959f60cd29,
        0x666ea36f7879462e, 0x0e0a77c19a07df2f
    ]

    // R^2 mod r
    public static let R2_MOD_R: [UInt64] = [
        0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3,
        0x8c49833d53bb8085, 0x0216d0b17f4e44a5
    ]

    // -r^(-1) mod 2^64
    public static let INV: UInt64 = 0xc2e1f593efffffff

    // TWO_ADICITY: r - 1 = 2^28 * t (t odd)
    public static let TWO_ADICITY: Int = 28

    // Primitive 2^28-th root of unity in Montgomery form
    // = 5^((r-1)/2^28) mod r (computed and verified)
    public static let ROOT_OF_UNITY: [UInt64] = [
        0x636e735580d13d9c, 0xa22bf3742445ffd6,
        0x56452ac01eb203d8, 0x1860ef942963f9e7
    ]

    // Multiplicative generator of Fr*
    public static let GENERATOR: UInt64 = 5

    public static var zero: Fr { Fr(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: Fr {
        Fr(v: (0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
               0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1))
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

    public static func from64(_ limbs: [UInt64]) -> Fr {
        Fr(v: (
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

// MARK: - Fr Equatable & Hashable

extension Fr: Equatable {
    public static func == (lhs: Fr, rhs: Fr) -> Bool {
        lhs.v.0 == rhs.v.0 && lhs.v.1 == rhs.v.1 &&
        lhs.v.2 == rhs.v.2 && lhs.v.3 == rhs.v.3 &&
        lhs.v.4 == rhs.v.4 && lhs.v.5 == rhs.v.5 &&
        lhs.v.6 == rhs.v.6 && lhs.v.7 == rhs.v.7
    }
}

extension Fr: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(v.0)
        hasher.combine(v.1)
        hasher.combine(v.2)
        hasher.combine(v.3)
        hasher.combine(v.4)
        hasher.combine(v.5)
        hasher.combine(v.6)
        hasher.combine(v.7)
    }
}

// MARK: - Fr Field Operations (zero-copy C CIOS)
// Fr.v is 8×UInt32 = 32 bytes = same layout as uint64_t[4] on little-endian

@inline(__always)
private func withFrPtr<T>(_ a: Fr, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func frFromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> Fr {
    var rv: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return Fr(v: rv)
}

@inline(__always)
public func frMul(_ a: Fr, _ b: Fr) -> Fr {
    withFrPtr(a) { ap in withFrPtr(b) { bp in
        frFromRaw { rp in bn254_fr_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func frAdd(_ a: Fr, _ b: Fr) -> Fr {
    withFrPtr(a) { ap in withFrPtr(b) { bp in
        frFromRaw { rp in bn254_fr_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func frSub(_ a: Fr, _ b: Fr) -> Fr {
    withFrPtr(a) { ap in withFrPtr(b) { bp in
        frFromRaw { rp in bn254_fr_sub(ap, bp, rp) }
    }}
}

@inline(__always)
public func frSqr(_ a: Fr) -> Fr {
    withFrPtr(a) { ap in
        frFromRaw { rp in bn254_fr_sqr(ap, rp) }
    }
}

public func frFromInt(_ val: UInt64) -> Fr {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))
}

public func frToInt(_ a: Fr) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return frMul(a, Fr.from64(one)).to64()
}

/// Fast extraction of low 64-bit integer value from Fr (no heap allocation).
/// Only correct when the integer value fits in 64 bits.
/// Performs Montgomery reduction: multiply by raw 1 and take low limb.
@inline(__always)
public func frToUInt64(_ a: Fr) -> UInt64 {
    let rawOne = Fr(v: (1, 0, 0, 0, 0, 0, 0, 0))
    let reduced = frMul(a, rawOne)
    return UInt64(reduced.v.0) | (UInt64(reduced.v.1) << 32)
}

@inline(__always)
public func frInverse(_ a: Fr) -> Fr {
    withFrPtr(a) { ap in
        frFromRaw { rp in bn254_fr_inverse(ap, rp) }
    }
}

/// Montgomery batch inversion: compute 1/a[i] for all i using a single frInverse.
/// Uses O(3n) multiplications instead of O(n * 256) squarings+muls.
public func frBatchInverse(_ a: [Fr]) -> [Fr] {
    let n = a.count
    if n == 0 { return [] }
    if n == 1 { return [frInverse(a[0])] }

    var result = [Fr](repeating: Fr.zero, count: n)
    a.withUnsafeBytes { aBuf in
        result.withUnsafeMutableBytes { rBuf in
            bn254_fr_batch_inverse(
                aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                Int32(n),
                rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
        }
    }
    return result
}

/// Compute a^n mod r using square-and-multiply.
public func frPow(_ a: Fr, _ n: UInt64) -> Fr {
    if n == 0 { return Fr.one }
    if n == 1 { return a }
    var result = Fr.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 {
            result = frMul(result, base)
        }
        base = frSqr(base)
        k >>= 1
    }
    return result
}

/// Cached roots of unity — only 29 possible values (logN 0...28).
private let _rootOfUnityCache: [Fr] = {
    var cache = [Fr](repeating: Fr.zero, count: Fr.TWO_ADICITY + 1)
    cache[Fr.TWO_ADICITY] = Fr.from64(Fr.ROOT_OF_UNITY)
    for k in stride(from: Fr.TWO_ADICITY - 1, through: 0, by: -1) {
        cache[k] = frSqr(cache[k + 1])
    }
    return cache
}()

/// Get the primitive 2^k-th root of unity (k <= TWO_ADICITY=28).
public func frRootOfUnity(logN: Int) -> Fr {
    precondition(logN >= 0 && logN <= Fr.TWO_ADICITY, "logN out of range")
    return _rootOfUnityCache[logN]
}

/// Precompute twiddle factors: [omega^0, omega^1, ..., omega^(n - 1)] in Montgomery form.
/// Full N entries needed for four-step FFT twiddle multiply.
public func precomputeTwiddles(logN: Int) -> [Fr] {
    let n = 1 << logN
    let omega = frRootOfUnity(logN: logN)
    var twiddles = [Fr](repeating: Fr.one, count: n)
    for i in 1..<n {
        twiddles[i] = frMul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for iNTT.
/// Full N entries needed for four-step iFFT inverse twiddle multiply.
public func precomputeInverseTwiddles(logN: Int) -> [Fr] {
    let n = 1 << logN
    let omega = frRootOfUnity(logN: logN)
    let omegaInv = frInverse(omega)
    var twiddles = [Fr](repeating: Fr.one, count: n)
    for i in 1..<n {
        twiddles[i] = frMul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
