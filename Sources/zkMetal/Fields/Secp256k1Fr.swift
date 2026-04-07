// secp256k1 scalar field Fr arithmetic (CPU-side)
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Field elements as 4x64-bit limbs in Montgomery form (little-endian).

import Foundation
import NeonFieldOps

public struct SecpFr: Equatable {
    public var v: (UInt64, UInt64, UInt64, UInt64)

    public static let N: [UInt64] = [
        0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
        0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF
    ]

    // R mod n (Montgomery form of 1): 2^256 mod n
    public static let R_MOD_N: [UInt64] = [
        0x402DA1732FC9BEBF, 0x4551231950B75FC4,
        0x0000000000000001, 0x0000000000000000
    ]

    // R^2 mod n: 2^512 mod n
    public static let R2_MOD_N: [UInt64] = [
        0x896CF21467D7D140, 0x741496C20E7CF878,
        0xE697F5E45BCD07C6, 0x9D671CD581C69BC5
    ]

    // -n^(-1) mod 2^64
    public static let INV: UInt64 = 0x4B0DFF665588B13F

    public static var zero: SecpFr { SecpFr(v: (0, 0, 0, 0)) }

    public static var one: SecpFr {
        SecpFr(v: (R_MOD_N[0], R_MOD_N[1], R_MOD_N[2], R_MOD_N[3]))
    }

    public init(v: (UInt64, UInt64, UInt64, UInt64)) {
        self.v = v
    }

    public func toLimbs() -> [UInt64] { [v.0, v.1, v.2, v.3] }

    public static func fromLimbs(_ l: [UInt64]) -> SecpFr {
        SecpFr(v: (l[0], l[1], l[2], l[3]))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0
    }

    public static func == (lhs: SecpFr, rhs: SecpFr) -> Bool {
        lhs.v.0 == rhs.v.0 && lhs.v.1 == rhs.v.1 &&
        lhs.v.2 == rhs.v.2 && lhs.v.3 == rhs.v.3
    }

    // Convert to 8x32-bit limbs (for MSM scalar interface)
    public func to32() -> [UInt32] {
        [UInt32(v.0 & 0xFFFFFFFF), UInt32(v.0 >> 32),
         UInt32(v.1 & 0xFFFFFFFF), UInt32(v.1 >> 32),
         UInt32(v.2 & 0xFFFFFFFF), UInt32(v.2 >> 32),
         UInt32(v.3 & 0xFFFFFFFF), UInt32(v.3 >> 32)]
    }
}

// MARK: - Field Operations (zero-copy C CIOS)

@inline(__always)
private func withSecpFrPtr<T>(_ a: SecpFr, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func secpFrFromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> SecpFr {
    var rv: (UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return SecpFr(v: rv)
}

@inline(__always)
public func secpFrMul(_ a: SecpFr, _ b: SecpFr) -> SecpFr {
    withSecpFrPtr(a) { ap in withSecpFrPtr(b) { bp in
        secpFrFromRaw { rp in secp256k1_fr_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpFrAdd(_ a: SecpFr, _ b: SecpFr) -> SecpFr {
    withSecpFrPtr(a) { ap in withSecpFrPtr(b) { bp in
        secpFrFromRaw { rp in secp256k1_fr_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpFrSub(_ a: SecpFr, _ b: SecpFr) -> SecpFr {
    withSecpFrPtr(a) { ap in withSecpFrPtr(b) { bp in
        secpFrFromRaw { rp in secp256k1_fr_sub(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpFrSqr(_ a: SecpFr) -> SecpFr { secpFrMul(a, a) }

@inline(__always)
public func secpFrNeg(_ a: SecpFr) -> SecpFr {
    withSecpFrPtr(a) { ap in
        secpFrFromRaw { rp in secp256k1_fr_neg(ap, rp) }
    }
}

/// Convert raw integer to Montgomery form: val * R mod n
public func secpFrFromInt(_ val: UInt64) -> SecpFr {
    let raw = SecpFr(v: (val, 0, 0, 0))
    return secpFrMul(raw, SecpFr.fromLimbs(SecpFr.R2_MOD_N))
}

/// Convert from Montgomery form to integer: a * R^-1 mod n
public func secpFrToInt(_ a: SecpFr) -> [UInt64] {
    let one = SecpFr(v: (1, 0, 0, 0))
    return secpFrMul(a, one).toLimbs()
}

/// Convert raw 256-bit value (4×64 LE limbs) to Montgomery form
public func secpFrFromRaw(_ limbs: [UInt64]) -> SecpFr {
    let raw = SecpFr.fromLimbs(limbs)
    return secpFrMul(raw, SecpFr.fromLimbs(SecpFr.R2_MOD_N))
}

/// Field inverse via C Fermat's little theorem: a^(n-2) mod n
@inline(__always)
public func secpFrInverse(_ a: SecpFr) -> SecpFr {
    withSecpFrPtr(a) { ap in
        secpFrFromRaw { rp in secp256k1_fr_inverse(ap, rp) }
    }
}

/// Batch modular inverse using Montgomery's trick.
/// Computes inverse of each element in the array.
public func secpFrBatchInverse(_ elems: [SecpFr]) -> [SecpFr] {
    let n = elems.count
    if n == 0 { return [] }

    // Build prefix products: prods[i] = elems[0] * ... * elems[i]
    var prods = [SecpFr](repeating: SecpFr.one, count: n)
    prods[0] = elems[0]
    for i in 1..<n {
        prods[i] = secpFrMul(prods[i - 1], elems[i])
    }

    // Single inversion of the full product
    var inv = secpFrInverse(prods[n - 1])

    // Walk backwards to compute individual inverses
    var result = [SecpFr](repeating: SecpFr.zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = secpFrMul(inv, prods[i - 1])
        inv = secpFrMul(inv, elems[i])
    }
    result[0] = inv

    return result
}
