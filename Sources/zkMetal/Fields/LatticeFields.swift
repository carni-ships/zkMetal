// Lattice-based cryptography field arithmetic (CPU-side)
// Kyber: Z_q where q = 3329 (12-bit modulus, fits in UInt16)
// Dilithium: Z_q where q = 8380417 (23-bit modulus, fits in UInt32)
//
// These small moduli are the key advantage on Metal's 32-bit ALU:
// - Kyber: single 16-bit multiply + Barrett reduction
// - Dilithium: single 32-bit multiply + Montgomery reduction
// No multi-limb arithmetic needed (unlike BN254's 256-bit field).

import Foundation

// MARK: - Kyber Field (q = 3329)

public struct KyberField: Equatable {
    public var value: UInt16  // 0..<3329

    public static let Q: UInt16 = 3329
    // Barrett constant: floor(2^24 / q) = floor(16777216 / 3329) = 5039
    // We use k=24 so m = 5039
    public static let BARRETT_M: UInt32 = 5039
    public static let BARRETT_SHIFT: UInt32 = 24

    // Montgomery constant for NTT: R = 2^16 mod q = 62209  (unused on CPU, used in GPU Montgomery form)
    // But we use Barrett on CPU for simplicity since q is so small.

    // Primitive 256th root of unity: zeta = 17 (17^128 = -1 mod 3329)
    public static let ZETA: UInt16 = 17

    public static var zero: KyberField { KyberField(value: 0) }
    public static var one: KyberField { KyberField(value: 1) }

    public init(value: UInt16) {
        self.value = value
    }

    public init(reducing v: UInt32) {
        // Barrett reduction: v mod 3329
        let t = (UInt64(v) * UInt64(KyberField.BARRETT_M)) >> KyberField.BARRETT_SHIFT
        var r = Int32(v) - Int32(t) * Int32(KyberField.Q)
        if r >= Int32(KyberField.Q) { r -= Int32(KyberField.Q) }
        if r < 0 { r += Int32(KyberField.Q) }
        self.value = UInt16(r)
    }

    public var isZero: Bool { value == 0 }
}

@inline(__always)
public func kyberAdd(_ a: KyberField, _ b: KyberField) -> KyberField {
    let s = UInt32(a.value) + UInt32(b.value)
    return KyberField(value: s >= UInt32(KyberField.Q) ? UInt16(s - UInt32(KyberField.Q)) : UInt16(s))
}

@inline(__always)
public func kyberSub(_ a: KyberField, _ b: KyberField) -> KyberField {
    if a.value >= b.value {
        return KyberField(value: a.value - b.value)
    }
    return KyberField(value: a.value &+ KyberField.Q &- b.value)
}

@inline(__always)
public func kyberNeg(_ a: KyberField) -> KyberField {
    if a.value == 0 { return a }
    return KyberField(value: KyberField.Q - a.value)
}

@inline(__always)
public func kyberMul(_ a: KyberField, _ b: KyberField) -> KyberField {
    let prod = UInt32(a.value) * UInt32(b.value)
    return KyberField(reducing: prod)
}

@inline(__always)
public func kyberSqr(_ a: KyberField) -> KyberField { kyberMul(a, a) }

public func kyberPow(_ base: KyberField, _ exp: UInt16) -> KyberField {
    if exp == 0 { return KyberField.one }
    var result = KyberField.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = kyberMul(result, b) }
        b = kyberSqr(b)
        e >>= 1
    }
    return result
}

public func kyberInverse(_ a: KyberField) -> KyberField {
    // Fermat: a^(q-2) mod q
    return kyberPow(a, KyberField.Q - 2)
}

// MARK: - Dilithium Field (q = 8380417)

public struct DilithiumField: Equatable {
    public var value: UInt32  // 0..<8380417

    public static let Q: UInt32 = 8380417
    // Montgomery constant R = 2^32 mod q = 4236238847 (but we prefer Barrett for CPU)
    // Barrett: m = floor(2^48 / q) = floor(281474976710656 / 8380417) = 33579385
    // Actually let's use a simpler Barrett with shift=32:
    // m = floor(2^32 / q) = 512 (too coarse). Use shift=48.
    // m = floor(2^48 / q) = 33579385 (25 bits, fits in UInt32)
    public static let BARRETT_M: UInt64 = 33579385
    public static let BARRETT_SHIFT: UInt64 = 48

    // Primitive 512th root of unity: 1753 (1753^256 = -1 mod q, order 512)
    // For the 256-element negacyclic NTT, we need zeta^2 = 1753^2 mod q = 3073009
    // which satisfies (3073009)^128 = -1 mod q, (3073009)^256 = 1 mod q
    public static let ZETA: UInt32 = 3073009

    // Montgomery form constants for GPU
    public static let MONT_R: UInt32 = 4193792  // 2^23 mod q (since q is 23 bits, use R=2^23)
    // Actually for GPU we'll use direct Barrett. Montgomery is overkill for single-word.

    public static var zero: DilithiumField { DilithiumField(value: 0) }
    public static var one: DilithiumField { DilithiumField(value: 1) }

    public init(value: UInt32) {
        self.value = value
    }

    public init(reducing v: UInt64) {
        // Direct modulo — safe and fast for 47-bit products on 64-bit CPU
        // (q fits in 23 bits, product of two elements fits in 46 bits)
        self.value = UInt32(v % UInt64(DilithiumField.Q))
    }

    public var isZero: Bool { value == 0 }
}

@inline(__always)
public func dilithiumAdd(_ a: DilithiumField, _ b: DilithiumField) -> DilithiumField {
    let s = UInt64(a.value) + UInt64(b.value)
    return DilithiumField(value: s >= UInt64(DilithiumField.Q) ? UInt32(s - UInt64(DilithiumField.Q)) : UInt32(s))
}

@inline(__always)
public func dilithiumSub(_ a: DilithiumField, _ b: DilithiumField) -> DilithiumField {
    if a.value >= b.value {
        return DilithiumField(value: a.value - b.value)
    }
    return DilithiumField(value: a.value &+ DilithiumField.Q &- b.value)
}

@inline(__always)
public func dilithiumNeg(_ a: DilithiumField) -> DilithiumField {
    if a.value == 0 { return a }
    return DilithiumField(value: DilithiumField.Q - a.value)
}

@inline(__always)
public func dilithiumMul(_ a: DilithiumField, _ b: DilithiumField) -> DilithiumField {
    let prod = UInt64(a.value) * UInt64(b.value)
    return DilithiumField(reducing: prod)
}

@inline(__always)
public func dilithiumSqr(_ a: DilithiumField) -> DilithiumField { dilithiumMul(a, a) }

public func dilithiumPow(_ base: DilithiumField, _ exp: UInt32) -> DilithiumField {
    if exp == 0 { return DilithiumField.one }
    var result = DilithiumField.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = dilithiumMul(result, b) }
        b = dilithiumSqr(b)
        e >>= 1
    }
    return result
}

public func dilithiumInverse(_ a: DilithiumField) -> DilithiumField {
    // Fermat: a^(q-2) mod q
    return dilithiumPow(a, DilithiumField.Q - 2)
}

// MARK: - Kyber NTT twiddle generation

/// Compute Kyber NTT twiddle factors (precomputed powers of zeta)
/// Kyber uses a negacyclic NTT with the "bit-reversed" ordering.
/// zeta_brv[i] = zeta^(bitrev(i)) for i in 0..<128
public func kyberTwiddles() -> [KyberField] {
    let n = 256
    let zeta = KyberField(value: KyberField.ZETA)
    // Compute powers of zeta: zeta^0, zeta^1, ..., zeta^255
    var powers = [KyberField](repeating: KyberField.one, count: n)
    for i in 1..<n {
        powers[i] = kyberMul(powers[i-1], zeta)
    }
    // Bit-reversed twiddle table for NTT (128 entries)
    var twiddles = [KyberField](repeating: KyberField.one, count: 128)
    for i in 0..<128 {
        let brv = bitReverse7(UInt8(i))
        twiddles[i] = powers[Int(brv)]
    }
    return twiddles
}

/// Compute inverse twiddle factors for Kyber INTT
public func kyberInvTwiddles() -> [KyberField] {
    let fwd = kyberTwiddles()
    return fwd.map { kyberInverse($0) }
}

/// Bit-reverse a 7-bit number
@inline(__always)
func bitReverse7(_ x: UInt8) -> UInt8 {
    var v = x
    v = ((v & 0x55) << 1) | ((v >> 1) & 0x55)
    v = ((v & 0x33) << 2) | ((v >> 2) & 0x33)
    v = ((v & 0x0F) << 4) | ((v >> 4) & 0x0F)
    return v >> 1  // 7-bit reversal: reverse 8 bits then shift right 1
}

// MARK: - Dilithium NTT twiddle generation

/// Compute Dilithium NTT twiddle factors
public func dilithiumTwiddles() -> [DilithiumField] {
    let n = 256
    let zeta = DilithiumField(value: DilithiumField.ZETA)
    var powers = [DilithiumField](repeating: DilithiumField.one, count: n)
    for i in 1..<n {
        powers[i] = dilithiumMul(powers[i-1], zeta)
    }
    var twiddles = [DilithiumField](repeating: DilithiumField.one, count: 128)
    for i in 0..<128 {
        let brv = bitReverse7(UInt8(i))
        twiddles[i] = powers[Int(brv)]
    }
    return twiddles
}

/// Compute inverse twiddle factors for Dilithium INTT
public func dilithiumInvTwiddles() -> [DilithiumField] {
    let fwd = dilithiumTwiddles()
    return fwd.map { dilithiumInverse($0) }
}

// MARK: - CPU NTT implementations

/// CPU Kyber NTT (Cooley-Tukey, in-place, negacyclic)
/// Standard Kyber NTT as specified in FIPS 203
public func kyberNTTCPU(_ poly: inout [KyberField]) {
    precondition(poly.count == 256)
    let twiddles = kyberTwiddles()
    var k = 1
    var len = 128
    while len >= 2 {
        var start = 0
        while start < 256 {
            let tw = twiddles[k]
            k += 1
            for j in start..<(start + len) {
                let t = kyberMul(tw, poly[j + len])
                poly[j + len] = kyberSub(poly[j], t)
                poly[j] = kyberAdd(poly[j], t)
            }
            start += 2 * len
        }
        len >>= 1
    }
}

/// CPU Kyber inverse NTT (Gentleman-Sande)
/// Uses forward twiddles (not inverse), matching the standard Kyber INTT formulation.
public func kyberInvNTTCPU(_ poly: inout [KyberField]) {
    precondition(poly.count == 256)
    let twiddles = kyberTwiddles()  // use FORWARD twiddles
    var k = 127
    var len = 2
    while len <= 128 {
        var start = 0
        while start < 256 {
            let tw = twiddles[k]
            k -= 1
            for j in start..<(start + len) {
                let t = poly[j]
                poly[j] = kyberAdd(t, poly[j + len])
                poly[j + len] = kyberMul(tw, kyberSub(t, poly[j + len]))
            }
            start += 2 * len
        }
        len <<= 1
    }
    let invN = kyberInverse(KyberField(value: 128))
    for i in 0..<256 {
        poly[i] = kyberMul(poly[i], invN)
    }
}

/// CPU Dilithium NTT
public func dilithiumNTTCPU(_ poly: inout [DilithiumField]) {
    precondition(poly.count == 256)
    let twiddles = dilithiumTwiddles()
    var k = 1
    var len = 128
    while len >= 2 {
        var start = 0
        while start < 256 {
            let tw = twiddles[k]
            k += 1
            for j in start..<(start + len) {
                let t = dilithiumMul(tw, poly[j + len])
                poly[j + len] = dilithiumSub(poly[j], t)
                poly[j] = dilithiumAdd(poly[j], t)
            }
            start += 2 * len
        }
        len >>= 1
    }
}

/// CPU Dilithium inverse NTT (Gentleman-Sande)
/// Uses forward twiddles, matching the standard formulation.
public func dilithiumInvNTTCPU(_ poly: inout [DilithiumField]) {
    precondition(poly.count == 256)
    let twiddles = dilithiumTwiddles()  // use FORWARD twiddles
    var k = 127
    var len = 2
    while len <= 128 {
        var start = 0
        while start < 256 {
            let tw = twiddles[k]
            k -= 1
            for j in start..<(start + len) {
                let t = poly[j]
                poly[j] = dilithiumAdd(t, poly[j + len])
                poly[j + len] = dilithiumMul(tw, dilithiumSub(t, poly[j + len]))
            }
            start += 2 * len
        }
        len <<= 1
    }
    let invN = dilithiumInverse(DilithiumField(value: 128))
    for i in 0..<256 {
        poly[i] = dilithiumMul(poly[i], invN)
    }
}
