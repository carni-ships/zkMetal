// Ed25519 base field Fp arithmetic (CPU-side)
// p = 2^255 - 19
// Field elements as 4x64-bit limbs in Montgomery form (little-endian).

import Foundation

public struct Ed25519Fp: Equatable {
    public var v: (UInt64, UInt64, UInt64, UInt64)

    // p = 2^255 - 19 = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed
    public static let P: [UInt64] = [
        0xffffffffffffffed, 0xffffffffffffffff,
        0xffffffffffffffff, 0x7fffffffffffffff
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    // R = 2^256 mod p = 2^256 - p = 2^256 - (2^255 - 19) = 2^255 + 19 = 38
    // But more precisely: 2^256 mod (2^255-19) = 2*p + 38 mod p = 38
    // Actually 2^256 = 2 * (2^255) = 2 * (p + 19) = 2p + 38, so R mod p = 38
    public static let R_MOD_P: [UInt64] = [
        0x0000000000000026, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000
    ]

    // R^2 mod p: 2^512 mod p
    // 2^512 mod p = (2^256)^2 mod p = 38^2 mod p = 1444 mod p
    // Wait, need proper computation. 2^256 mod p = 38.
    // 2^512 mod p = 38^2 mod p = 1444. But this seems too small.
    // Let me verify: 38 * 38 = 1444, and 1444 < p, so R^2 mod p = 1444 = 0x5A4
    public static let R2_MOD_P: [UInt64] = [
        0x00000000000005a4, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000
    ]

    // -p^(-1) mod 2^64
    // p = 0xffffffffffffffed
    // We need inv such that p * inv ≡ -1 (mod 2^64)
    // p[0] = 0xffffffffffffffed
    // p[0] * inv ≡ -1 (mod 2^64)
    // inv = 0x86bca1af286bca1b (computed via extended GCD)
    public static let INV: UInt64 = 0x86bca1af286bca1b

    public static var zero: Ed25519Fp { Ed25519Fp(v: (0, 0, 0, 0)) }

    public static var one: Ed25519Fp {
        Ed25519Fp(v: (R_MOD_P[0], R_MOD_P[1], R_MOD_P[2], R_MOD_P[3]))
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

    public static func == (lhs: Ed25519Fp, rhs: Ed25519Fp) -> Bool {
        lhs.v.0 == rhs.v.0 && lhs.v.1 == rhs.v.1 &&
        lhs.v.2 == rhs.v.2 && lhs.v.3 == rhs.v.3
    }
}

// MARK: - Field Operations

/// Montgomery multiplication: (a * b * R^-1) mod p
public func ed25519FpMul(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    let al = a.toLimbs(), bl = b.toLimbs()
    var t = [UInt64](repeating: 0, count: 6)

    for i in 0..<4 {
        var carry: UInt64 = 0
        for j in 0..<4 {
            let (hi, lo) = al[i].multipliedFullWidth(by: bl[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
        }
        let (s4, c4) = t[4].addingReportingOverflow(carry)
        t[4] = s4
        t[5] = t[5] &+ (c4 ? 1 : 0)

        let m = t[0] &* Ed25519Fp.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: Ed25519Fp.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
        }
        let (s4r, c4r) = t[4].addingReportingOverflow(carry)
        t[4] = s4r
        t[5] = t[5] &+ (c4r ? 1 : 0)

        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = t[5]; t[5] = 0
    }

    var r = Array(t[0..<4])
    if t[4] != 0 || ed25519FpGte(r, Ed25519Fp.P) {
        r = ed25519FpSub256(r, Ed25519Fp.P).0
    }
    return Ed25519Fp.fromLimbs(r)
}

public func ed25519FpAdd(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let (s1, c1) = a.toLimbs()[i].addingReportingOverflow(b.toLimbs()[i])
        let (s2, c2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
    }
    if carry != 0 || ed25519FpGte(r, Ed25519Fp.P) {
        r = ed25519FpSub256(r, Ed25519Fp.P).0
    }
    return Ed25519Fp.fromLimbs(r)
}

public func ed25519FpSub(_ a: Ed25519Fp, _ b: Ed25519Fp) -> Ed25519Fp {
    let (r, borrow) = ed25519FpSub256(a.toLimbs(), b.toLimbs())
    if borrow {
        var result = [UInt64](repeating: 0, count: 4)
        var carry: UInt64 = 0
        for i in 0..<4 {
            let (s1, c1) = r[i].addingReportingOverflow(Ed25519Fp.P[i])
            let (s2, c2) = s1.addingReportingOverflow(carry)
            result[i] = s2
            carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        return Ed25519Fp.fromLimbs(result)
    }
    return Ed25519Fp.fromLimbs(r)
}

public func ed25519FpSqr(_ a: Ed25519Fp) -> Ed25519Fp { ed25519FpMul(a, a) }
public func ed25519FpDouble(_ a: Ed25519Fp) -> Ed25519Fp { ed25519FpAdd(a, a) }

public func ed25519FpNeg(_ a: Ed25519Fp) -> Ed25519Fp {
    if a.isZero { return a }
    return Ed25519Fp.fromLimbs(ed25519FpSub256(Ed25519Fp.P, a.toLimbs()).0)
}

/// Convert raw integer to Montgomery form
public func ed25519FpFromInt(_ val: UInt64) -> Ed25519Fp {
    let raw = Ed25519Fp(v: (val, 0, 0, 0))
    return ed25519FpMul(raw, Ed25519Fp.fromLimbs(Ed25519Fp.R2_MOD_P))
}

/// Convert from Montgomery form to integer
public func ed25519FpToInt(_ a: Ed25519Fp) -> [UInt64] {
    let one = Ed25519Fp(v: (1, 0, 0, 0))
    return ed25519FpMul(a, one).toLimbs()
}

/// Convert raw 256-bit limbs to Montgomery form
public func ed25519FpFromRaw(_ limbs: [UInt64]) -> Ed25519Fp {
    let raw = Ed25519Fp.fromLimbs(limbs)
    return ed25519FpMul(raw, Ed25519Fp.fromLimbs(Ed25519Fp.R2_MOD_P))
}

/// Field inverse via Fermat's little theorem: a^(p-2) mod p
public func ed25519FpInverse(_ a: Ed25519Fp) -> Ed25519Fp {
    var result = Ed25519Fp.one
    var base = a
    var exp = Ed25519Fp.P.map { $0 }
    // p - 2
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

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

/// Square root in Fp using p ≡ 5 mod 8 formula
/// For p = 2^255 - 19, we use the Tonelli-Shanks variant:
/// candidate = a^((p+3)/8) mod p
/// If candidate^2 = a, return candidate
/// If candidate^2 = -a, return candidate * sqrt(-1)
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
    // sqrt(-1) mod p = 2^((p-1)/4) mod p
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

/// Parse hex string to Ed25519Fp in Montgomery form
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

/// Decode 32 bytes (little-endian) to Ed25519Fp in Montgomery form
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
