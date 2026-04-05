// BN254 base field Fp arithmetic (CPU-side)
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

import Foundation
import NeonFieldOps

public struct Fp {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0x3c208c16d87cfd47, 0x97816a916871ca8d,
        0xb85045b68181585d, 0x30644e72e131a029
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    public static let R_MOD_P: [UInt64] = [
        0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d,
        0x666ea36f7879462c, 0x0e0a77c19a07df2f
    ]

    // R^2 mod p: 2^512 mod p
    public static let R2_MOD_P: [UInt64] = [
        0xf32cfc5b538afa89, 0xb5e71911d44501fb,
        0x47ab1eff0a417ff6, 0x06d89f71cab8351f
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0x87d20782e4866389

    public static var zero: Fp { Fp(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: Fp {
        // R mod p in 32-bit limbs (little-endian)
        Fp(v: (0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
               0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1))
    }

    public init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    public init(from bytes: [UInt8]) {
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for i in 0..<min(32, bytes.count) {
            limbs[i / 4] |= UInt32(bytes[i]) << ((i % 4) * 8)
        }
        self.v = (limbs[0], limbs[1], limbs[2], limbs[3],
                  limbs[4], limbs[5], limbs[6], limbs[7])
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

    public static func from64(_ limbs: [UInt64]) -> Fp {
        Fp(v: (
            UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
            UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
            UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
            UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32)
        ))
    }

    public func toBytes() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 32)
        let limbs = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        for i in 0..<8 {
            bytes[i * 4 + 0] = UInt8(limbs[i] & 0xFF)
            bytes[i * 4 + 1] = UInt8((limbs[i] >> 8) & 0xFF)
            bytes[i * 4 + 2] = UInt8((limbs[i] >> 16) & 0xFF)
            bytes[i * 4 + 3] = UInt8((limbs[i] >> 24) & 0xFF)
        }
        return bytes
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0 &&
        v.4 == 0 && v.5 == 0 && v.6 == 0 && v.7 == 0
    }
}

// MARK: - 256-bit Arithmetic Helpers (Array)

public func add256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], UInt64) {
    var r = [UInt64](repeating: 0, count: 4)
    var carry: UInt64 = 0
    for i in 0..<4 {
        let (s1, c1) = a[i].addingReportingOverflow(b[i])
        let (s2, c2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
    }
    return (r, carry)
}

public func sub256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], Bool) {
    var r = [UInt64](repeating: 0, count: 4)
    var borrow: Bool = false
    for i in 0..<4 {
        let (s1, b1) = a[i].subtractingReportingOverflow(b[i])
        let (s2, b2) = s1.subtractingReportingOverflow(borrow ? 1 : 0)
        r[i] = s2
        borrow = b1 || b2
    }
    return (r, borrow)
}

public func gte256(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] > b[i] { return true }
        if a[i] < b[i] { return false }
    }
    return true
}

// MARK: - 256-bit Arithmetic Helpers (Tuple, zero allocation)

public typealias U256 = (UInt64, UInt64, UInt64, UInt64)

@inline(__always)
public func add256t(_ a: U256, _ b: U256) -> (U256, Bool) {
    let (s0, c0) = a.0.addingReportingOverflow(b.0)
    let (t1, c1a) = a.1.addingReportingOverflow(b.1)
    let (s1, c1b) = t1.addingReportingOverflow(c0 ? 1 : 0)
    let c1 = c1a || c1b
    let (t2, c2a) = a.2.addingReportingOverflow(b.2)
    let (s2, c2b) = t2.addingReportingOverflow(c1 ? 1 : 0)
    let c2 = c2a || c2b
    let (t3, c3a) = a.3.addingReportingOverflow(b.3)
    let (s3, c3b) = t3.addingReportingOverflow(c2 ? 1 : 0)
    return ((s0, s1, s2, s3), c3a || c3b)
}

@inline(__always)
public func sub256t(_ a: U256, _ b: U256) -> (U256, Bool) {
    let (s0, b0) = a.0.subtractingReportingOverflow(b.0)
    let (t1, b1a) = a.1.subtractingReportingOverflow(b.1)
    let (s1, b1b) = t1.subtractingReportingOverflow(b0 ? 1 : 0)
    let bw1 = b1a || b1b
    let (t2, b2a) = a.2.subtractingReportingOverflow(b.2)
    let (s2, b2b) = t2.subtractingReportingOverflow(bw1 ? 1 : 0)
    let bw2 = b2a || b2b
    let (t3, b3a) = a.3.subtractingReportingOverflow(b.3)
    let (s3, b3b) = t3.subtractingReportingOverflow(bw2 ? 1 : 0)
    return ((s0, s1, s2, s3), b3a || b3b)
}

@inline(__always)
public func gte256t(_ a: U256, _ b: U256) -> Bool {
    if a.3 != b.3 { return a.3 > b.3 }
    if a.2 != b.2 { return a.2 > b.2 }
    if a.1 != b.1 { return a.1 > b.1 }
    return a.0 >= b.0
}

@inline(__always)
public func neg256t(_ a: U256) -> U256 {
    let n0 = ~a.0 &+ 1
    let c0: UInt64 = (a.0 == 0) ? 1 : 0
    let n1 = ~a.1 &+ c0
    let c1: UInt64 = (a.1 == 0 && c0 == 1) ? 1 : 0
    let n2 = ~a.2 &+ c1
    let c2: UInt64 = (a.2 == 0 && c1 == 1) ? 1 : 0
    let n3 = ~a.3 &+ c2
    return (n0, n1, n2, n3)
}

// MARK: - Field Operations

// Montgomery multiplication: (a * b * R^-1) mod p
public func fpMul(_ a: Fp, _ b: Fp) -> Fp {
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

        let m = t[0] &* Fp.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: Fp.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = 0
    }

    var r = Array(t[0..<4])
    if gte256(r, Fp.P) {
        (r, _) = sub256(r, Fp.P)
    }
    return Fp.from64(r)
}

public func fpAdd(_ a: Fp, _ b: Fp) -> Fp {
    var (r, carry) = add256(a.to64(), b.to64())
    if carry != 0 || gte256(r, Fp.P) {
        (r, _) = sub256(r, Fp.P)
    }
    return Fp.from64(r)
}

public func fpSub(_ a: Fp, _ b: Fp) -> Fp {
    var (r, borrow) = sub256(a.to64(), b.to64())
    if borrow {
        (r, _) = add256(r, Fp.P)
    }
    return Fp.from64(r)
}

public func fpSqr(_ a: Fp) -> Fp {
    let al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    al.withUnsafeBufferPointer { aPtr in
        r.withUnsafeMutableBufferPointer { rPtr in
            bn254_fp_sqr(aPtr.baseAddress!, rPtr.baseAddress!)
        }
    }
    return Fp.from64(r)
}
public func fpDouble(_ a: Fp) -> Fp { fpAdd(a, a) }

// Convert integer to Montgomery form: a * R mod p
public func fpFromInt(_ val: UInt64) -> Fp {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

// Convert from Montgomery form to integer: a * R^-1 mod p
public func fpToInt(_ a: Fp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return fpMul(a, Fp.from64(one)).to64()
}

// Field negation: -a mod p
public func fpNeg(_ a: Fp) -> Fp {
    if a.isZero { return a }
    let (r, _) = sub256(Fp.P, a.to64())
    return Fp.from64(r)
}

// Field inverse via Fermat's little theorem: a^(p-2) mod p — C accelerated
public func fpInverse(_ a: Fp) -> Fp {
    let al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    al.withUnsafeBufferPointer { aPtr in
        r.withUnsafeMutableBufferPointer { rPtr in
            bn254_fp_inv(aPtr.baseAddress!, rPtr.baseAddress!)
        }
    }
    return Fp.from64(r)
}

/// Fp square root via a^((p+1)/4) — C accelerated.
/// BN254 Fp has p ≡ 3 mod 4. Returns nil if a is not a quadratic residue.
public func fpSqrt(_ a: Fp) -> Fp? {
    let al = a.to64()
    var r = [UInt64](repeating: 0, count: 4)
    let ok = al.withUnsafeBufferPointer { aPtr -> Int32 in
        r.withUnsafeMutableBufferPointer { rPtr in
            bn254_fp_sqrt(aPtr.baseAddress!, rPtr.baseAddress!)
        }
    }
    return ok != 0 ? Fp.from64(r) : nil
}

/// Parse a hex string (with or without "0x" prefix) into an Fp in Montgomery form.
public func fpFromHex(_ hex: String) -> Fp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Convert Fp (Montgomery form) to a "0x"-prefixed big-endian hex string.
public func fpToHex(_ a: Fp) -> String {
    let limbs = fpToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}
