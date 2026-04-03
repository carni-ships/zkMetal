// secp256k1 base field Fp arithmetic (CPU-side)
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//   = 2^256 - 2^32 - 977
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

import Foundation

public struct SecpFp {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0xfffffffefffffc2f, 0xffffffffffffffff,
        0xffffffffffffffff, 0xffffffffffffffff
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    public static let R_MOD_P: [UInt64] = [
        0x00000001000003d1, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000
    ]

    // R^2 mod p: 2^512 mod p
    public static let R2_MOD_P: [UInt64] = [
        0x000007a2000e90a1, 0x0000000000000001,
        0x0000000000000000, 0x0000000000000000
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0xd838091dd2253531

    public static var zero: SecpFp { SecpFp(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: SecpFp {
        // R mod p in 32-bit limbs (little-endian)
        SecpFp(v: (0x000003d1, 0x00000001, 0x00000000, 0x00000000,
                   0x00000000, 0x00000000, 0x00000000, 0x00000000))
    }

    public init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    // Convert to 4x64-bit limbs for arithmetic
    public func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
        ]
    }

    public static func from64(_ limbs: [UInt64]) -> SecpFp {
        SecpFp(v: (
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

// MARK: - Field Operations

// Montgomery multiplication: (a * b * R^-1) mod p
// Uses 6-limb accumulator to handle carries for near-256-bit modulus
public func secpMul(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    let al = a.to64(), bl = b.to64()
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

        let m = t[0] &* SecpFp.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: SecpFp.P[j])
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
    if t[4] != 0 || gte256(r, SecpFp.P) {
        (r, _) = sub256(r, SecpFp.P)
    }
    return SecpFp.from64(r)
}

public func secpAdd(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    var (r, carry) = add256(a.to64(), b.to64())
    if carry != 0 || gte256(r, SecpFp.P) {
        (r, _) = sub256(r, SecpFp.P)
    }
    return SecpFp.from64(r)
}

public func secpSub(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    var (r, borrow) = sub256(a.to64(), b.to64())
    if borrow {
        (r, _) = add256(r, SecpFp.P)
    }
    return SecpFp.from64(r)
}

public func secpSqr(_ a: SecpFp) -> SecpFp { secpMul(a, a) }
public func secpDouble(_ a: SecpFp) -> SecpFp { secpAdd(a, a) }

// Convert integer to Montgomery form: a * R mod p
public func secpFromInt(_ val: UInt64) -> SecpFp {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = SecpFp.from64(limbs)
    return secpMul(raw, SecpFp.from64(SecpFp.R2_MOD_P))
}

// Convert from Montgomery form to integer: a * R^-1 mod p
public func secpToInt(_ a: SecpFp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return secpMul(a, SecpFp.from64(one)).to64()
}

// Field negation: -a mod p
public func secpNeg(_ a: SecpFp) -> SecpFp {
    if a.isZero { return a }
    let (r, _) = sub256(SecpFp.P, a.to64())
    return SecpFp.from64(r)
}

// Field inverse via Fermat's little theorem: a^(p-2) mod p
public func secpInverse(_ a: SecpFp) -> SecpFp {
    var result = SecpFp.one
    var base = a
    var exp = SecpFp.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = secpMul(result, base)
            }
            base = secpSqr(base)
            word >>= 1
        }
    }
    return result
}

/// Parse a hex string into a SecpFp in Montgomery form.
public func secpFromHex(_ hex: String) -> SecpFp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = SecpFp.from64(limbs)
    return secpMul(raw, SecpFp.from64(SecpFp.R2_MOD_P))
}

/// Convert SecpFp (Montgomery form) to a "0x"-prefixed big-endian hex string.
public func secpToHex(_ a: SecpFp) -> String {
    let limbs = secpToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}
