// Ed25519 base field Fp arithmetic (CPU-side)
// p = 2^255 - 19
// Field elements as 4x64-bit limbs in direct integer representation (little-endian).
// Uses Solinas special reduction via C implementation for mul/sqr (2-4x faster than CIOS Montgomery).

import Foundation
import NeonFieldOps

public struct Ed25519Fp: Equatable {
    public var v: (UInt64, UInt64, UInt64, UInt64)

    // p = 2^255 - 19 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
    public static let P: [UInt64] = [
        0xffffffffffffffed, 0xffffffffffffffff,
        0xffffffffffffffff, 0x7fffffffffffffff
    ]

    // In direct integer representation, "one" is simply 1
    // (no Montgomery R factor needed)
    public static let R_MOD_P: [UInt64] = [
        0x0000000000000026, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000
    ]

    // R^2 mod p (kept for backward compat if anything references it)
    public static let R2_MOD_P: [UInt64] = [
        0x00000000000005a4, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000
    ]

    // -p^(-1) mod 2^64 (kept for conversion helpers)
    public static let INV: UInt64 = 0x86bca1af286bca1b

    public static var zero: Ed25519Fp { Ed25519Fp(v: (0, 0, 0, 0)) }

    // In direct integer form, one = 1
    public static var one: Ed25519Fp {
        Ed25519Fp(v: (1, 0, 0, 0))
    }

    public init(v: (UInt64, UInt64, UInt64, UInt64)) {
        self.v = v
    }

    public func toLimbs() -> [UInt64] { [v.0, v.1, v.2, v.3] }

    public static func fromLimbs(_ l: [UInt64]) -> Ed25519Fp {
        Ed25519Fp(v: (l[0], l[1], l[2], l[3]))
    }

    // Convert to 8x32-bit limbs (for GPU interface)
    public func to32() -> [UInt32] {
        [UInt32(v.0 & 0xFFFFFFFF), UInt32(v.0 >> 32),
         UInt32(v.1 & 0xFFFFFFFF), UInt32(v.1 >> 32),
         UInt32(v.2 & 0xFFFFFFFF), UInt32(v.2 >> 32),
         UInt32(v.3 & 0xFFFFFFFF), UInt32(v.3 >> 32)]
    }

    public static func from32(_ l: [UInt32]) -> Ed25519Fp {
        Ed25519Fp(v: (
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32)
        ))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0
    }

    /// Convert from direct integer representation to Montgomery form for GPU interop.
    /// Montgomery form = val * R mod p, where R = 2^256 ≡ 38 mod p.
    public func toMontgomery() -> Ed25519Fp {
        _fpUnaryOp(self, ed25519_direct_to_mont)
    }

    /// Convert from Montgomery form (GPU) to direct integer representation (CPU).
    public static func fromMontgomery(_ mont: Ed25519Fp) -> Ed25519Fp {
        _fpUnaryOp(mont, ed25519_mont_to_direct)
    }

    public static func == (lhs: Ed25519Fp, rhs: Ed25519Fp) -> Bool {
        lhs.v.0 == rhs.v.0 && lhs.v.1 == rhs.v.1 &&
        lhs.v.2 == rhs.v.2 && lhs.v.3 == rhs.v.3
    }
}

// MARK: - Field Operations (Solinas reduction via C)

// Helper: call a binary C field op on two Ed25519Fp values.
// Swift tuples of 4 x UInt64 are laid out contiguously, matching C's uint64_t[4].
@inline(__always)
private func _fpBinOp(_ a: Ed25519Fp, _ b: Ed25519Fp,
                      _ op: (UnsafePointer<UInt64>, UnsafePointer<UInt64>, UnsafeMutablePointer<UInt64>) -> Void) -> Ed25519Fp {
    var r = Ed25519Fp.zero
    withUnsafePointer(to: a.v) { ap in
        withUnsafePointer(to: b.v) { bp in
            withUnsafeMutablePointer(to: &r.v) { rp in
                op(UnsafeRawPointer(ap).assumingMemoryBound(to: UInt64.self),
                   UnsafeRawPointer(bp).assumingMemoryBound(to: UInt64.self),
                   UnsafeMutableRawPointer(rp).assumingMemoryBound(to: UInt64.self))
            }
        }
    }
    return r
}

@inline(__always)
private func _fpUnaryOp(_ a: Ed25519Fp,
                        _ op: (UnsafePointer<UInt64>, UnsafeMutablePointer<UInt64>) -> Void) -> Ed25519Fp {
    var r = Ed25519Fp.zero
    withUnsafePointer(to: a.v) { ap in
        withUnsafeMutablePointer(to: &r.v) { rp in
            op(UnsafeRawPointer(ap).assumingMemoryBound(to: UInt64.self),
               UnsafeMutableRawPointer(rp).assumingMemoryBound(to: UInt64.self))
        }
    }
    return r
}

/// Modular multiplication using Solinas reduction: (a * b) mod p
/// Direct integer form -- no Montgomery R factor.
public func ed25519FpMul(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    _fpBinOp(a, b, ed25519_fp_mul)
}

public func ed25519FpAdd(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    _fpBinOp(a, b, ed25519_fp_add)
}

public func ed25519FpSub(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    _fpBinOp(a, b, ed25519_fp_sub)
}

public func ed25519FpSqr(_ a: Ed25519Fp) -> Ed25519Fp {
    _fpUnaryOp(a, ed25519_fp_sqr)
}

public func ed25519FpDouble(_ a: Ed25519Fp) -> Ed25519Fp { ed25519FpAdd(a, a) }

public func ed25519FpNeg(_ a: Ed25519Fp) -> Ed25519Fp {
    if a.isZero { return a }
    return _fpUnaryOp(a, ed25519_fp_neg)
}

/// In direct integer form, fromInt just returns the value directly (no Montgomery conversion).
public func ed25519FpFromInt(_ val: UInt64) -> Ed25519Fp {
    Ed25519Fp(v: (val, 0, 0, 0))
}

/// In direct integer form, toInt returns the limbs directly.
public func ed25519FpToInt(_ a: Ed25519Fp) -> [UInt64] {
    a.toLimbs()
}

/// In direct form, fromRaw returns the limbs directly (just ensure < p).
public func ed25519FpFromRaw(_ limbs: [UInt64]) -> Ed25519Fp {
    // Reduce mod p if needed
    var r = limbs
    if ed25519FpGte(r, Ed25519Fp.P) {
        r = ed25519FpSub256(r, Ed25519Fp.P).0
    }
    return Ed25519Fp.fromLimbs(r)
}

/// Field inverse via Fermat's little theorem: a^(p-2) mod p
/// Uses C Solinas implementation.
public func ed25519FpInverse(_ a: Ed25519Fp) -> Ed25519Fp {
    _fpUnaryOp(a, ed25519_fp_inverse)
}

/// Square root in Fp using p ≡ 5 mod 8 formula
/// candidate = a^((p+3)/8) mod p
public func ed25519FpSqrt(_ a: Ed25519Fp) -> Ed25519Fp? {
    if a.isZero { return Ed25519Fp.zero }

    // (p+3)/8 = (2^255 - 19 + 3) / 8 = (2^255 - 16) / 8 = 2^252 - 2
    let exp: [UInt64] = [
        0xfffffffffffffffe, 0xffffffffffffffff,
        0xffffffffffffffff, 0x0fffffffffffffff
    ]

    var result = Ed25519Fp.one
    var base = a
    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = ed25519FpMul(result, base)
            }
            base = ed25519FpSqr(base)
            word >>= 1
        }
    }

    // Check if result^2 == a
    let check = ed25519FpSqr(result)
    if check == a { return result }

    // Try result * sqrt(-1)
    let sqrtMinusOne = ed25519FpComputeSqrtMinusOne()
    let result2 = ed25519FpMul(result, sqrtMinusOne)
    let check2 = ed25519FpSqr(result2)
    if check2 == a { return result2 }

    return nil
}

/// Compute sqrt(-1) mod p = 2^((p-1)/4) mod p
private func ed25519FpComputeSqrtMinusOne() -> Ed25519Fp {
    // (p-1)/4 = (2^255 - 20) / 4 = 2^253 - 5
    let exp: [UInt64] = [
        0xfffffffffffffffb, 0xffffffffffffffff,
        0xffffffffffffffff, 0x1fffffffffffffff
    ]
    var result = Ed25519Fp.one
    var base = ed25519FpFromInt(2)
    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = ed25519FpMul(result, base)
            }
            base = ed25519FpSqr(base)
            word >>= 1
        }
    }
    return result
}

/// Parse hex string to Ed25519Fp (direct integer form)
public func ed25519FpFromHex(_ hex: String) -> Ed25519Fp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    return ed25519FpFromRaw(limbs)
}

/// Convert to hex string
public func ed25519FpToHex(_ a: Ed25519Fp) -> String {
    let limbs = ed25519FpToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}

// MARK: - 256-bit helpers

func ed25519FpGte(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] > b[i] { return true }
        if a[i] < b[i] { return false }
    }
    return true
}

func ed25519FpSub256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], Bool) {
    var r = [UInt64](repeating: 0, count: 4)
    var borrow: UInt64 = 0
    for i in 0..<4 {
        let (d1, b1) = a[i].subtractingReportingOverflow(b[i])
        let (d2, b2) = d1.subtractingReportingOverflow(borrow)
        r[i] = d2
        borrow = (b1 ? 1 : 0) + (b2 ? 1 : 0)
    }
    return (r, borrow != 0)
}

/// Batch modular inverse using Montgomery's trick.
public func ed25519FpBatchInverse(_ elems: [Ed25519Fp]) -> [Ed25519Fp] {
    let n = elems.count
    if n == 0 { return [] }

    var prods = [Ed25519Fp](repeating: Ed25519Fp.one, count: n)
    prods[0] = elems[0]
    for i in 1..<n {
        prods[i] = ed25519FpMul(prods[i - 1], elems[i])
    }

    var inv = ed25519FpInverse(prods[n - 1])
    var result = [Ed25519Fp](repeating: Ed25519Fp.zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = ed25519FpMul(inv, prods[i - 1])
        inv = ed25519FpMul(inv, elems[i])
    }
    result[0] = inv
    return result
}

/// Encode Ed25519Fp to 32 bytes (little-endian, standard Ed25519 wire format)
public func ed25519FpToBytes(_ a: Ed25519Fp) -> [UInt8] {
    let limbs = ed25519FpToInt(a)
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((limbs[i] >> (j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Decode 32 bytes (little-endian) to Ed25519Fp
public func ed25519FpFromBytes(_ bytes: [UInt8]) -> Ed25519Fp {
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        for j in 0..<8 {
            if i * 8 + j < bytes.count {
                limbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
        }
    }
    // Clear top bit (Ed25519 field elements are at most 255 bits)
    limbs[3] &= 0x7fffffffffffffff
    return ed25519FpFromRaw(limbs)
}
