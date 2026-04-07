// secp256k1 base field Fp arithmetic (CPU-side)
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//   = 2^256 - 2^32 - 977
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

import Foundation
import NeonFieldOps

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

// MARK: - Field Operations (zero-copy C CIOS)

@inline(__always)
private func withSecpFpPtr<T>(_ a: SecpFp, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func secpFpFromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> SecpFp {
    var rv: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return SecpFp(v: rv)
}

@inline(__always)
public func secpMul(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in withSecpFpPtr(b) { bp in
        secpFpFromRaw { rp in secp256k1_fp_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpAdd(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in withSecpFpPtr(b) { bp in
        secpFpFromRaw { rp in secp256k1_fp_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpSub(_ a: SecpFp, _ b: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in withSecpFpPtr(b) { bp in
        secpFpFromRaw { rp in secp256k1_fp_sub(ap, bp, rp) }
    }}
}

@inline(__always)
public func secpSqr(_ a: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in
        secpFpFromRaw { rp in secp256k1_fp_sqr(ap, rp) }
    }
}

public func secpDouble(_ a: SecpFp) -> SecpFp { secpAdd(a, a) }

public func secpFromInt(_ val: UInt64) -> SecpFp {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = SecpFp.from64(limbs)
    return secpMul(raw, SecpFp.from64(SecpFp.R2_MOD_P))
}

public func secpToInt(_ a: SecpFp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return secpMul(a, SecpFp.from64(one)).to64()
}

@inline(__always)
public func secpNeg(_ a: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in
        secpFpFromRaw { rp in secp256k1_fp_neg(ap, rp) }
    }
}

@inline(__always)
public func secpInverse(_ a: SecpFp) -> SecpFp {
    withSecpFpPtr(a) { ap in
        secpFpFromRaw { rp in secp256k1_fp_inv(ap, rp) }
    }
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
