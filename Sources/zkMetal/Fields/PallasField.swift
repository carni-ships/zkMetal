// Pallas base field Fp arithmetic (CPU-side)
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).
// Note: Pallas Fp = Vesta Fq (scalar field). This is the cycle property.

import Foundation
import NeonFieldOps

public struct PallasFp {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0x992d30ed00000001, 0x224698fc094cf91b,
        0x0000000000000000, 0x4000000000000000
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    public static let R_MOD_P: [UInt64] = [
        0x34786d38fffffffd, 0x992c350be41914ad,
        0xffffffffffffffff, 0x3fffffffffffffff
    ]

    // R^2 mod p: 2^512 mod p
    public static let R2_MOD_P: [UInt64] = [
        0x8c78ecb30000000f, 0xd7d30dbd8b0de0e7,
        0x7797a99bc3c95d18, 0x096d41af7b9cb714
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0x992d30ecffffffff

    public static var zero: PallasFp { PallasFp(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: PallasFp {
        PallasFp(v: (0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
                     0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff))
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

    public func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
        ]
    }

    public static func from64(_ limbs: [UInt64]) -> PallasFp {
        PallasFp(v: (
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

// MARK: - Zero-copy pointer helpers
// PallasFp.v is 8×UInt32 = 32 bytes = same layout as uint64_t[4] on little-endian

@inline(__always)
private func withPallasPtr<T>(_ a: PallasFp, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func pallasFromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> PallasFp {
    var rv: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return PallasFp(v: rv)
}

// MARK: - Field Operations

@inline(__always)
public func pallasMul(_ a: PallasFp, _ b: PallasFp) -> PallasFp {
    withPallasPtr(a) { ap in withPallasPtr(b) { bp in
        pallasFromRaw { rp in pallas_fp_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func pallasAdd(_ a: PallasFp, _ b: PallasFp) -> PallasFp {
    withPallasPtr(a) { ap in withPallasPtr(b) { bp in
        pallasFromRaw { rp in pallas_fp_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func pallasSub(_ a: PallasFp, _ b: PallasFp) -> PallasFp {
    withPallasPtr(a) { ap in withPallasPtr(b) { bp in
        pallasFromRaw { rp in pallas_fp_sub(ap, bp, rp) }
    }}
}

public func pallasSqr(_ a: PallasFp) -> PallasFp { pallasMul(a, a) }
public func pallasDouble(_ a: PallasFp) -> PallasFp { pallasAdd(a, a) }

public func pallasFromInt(_ val: UInt64) -> PallasFp {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = PallasFp.from64(limbs)
    return pallasMul(raw, PallasFp.from64(PallasFp.R2_MOD_P))
}

public func pallasToInt(_ a: PallasFp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return pallasMul(a, PallasFp.from64(one)).to64()
}

@inline(__always)
public func pallasNeg(_ a: PallasFp) -> PallasFp {
    withPallasPtr(a) { ap in
        pallasFromRaw { rp in pallas_fp_neg(ap, rp) }
    }
}

public func pallasInverse(_ a: PallasFp) -> PallasFp {
    var result = PallasFp.one
    var base = a
    var exp = PallasFp.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = pallasMul(result, base)
            }
            base = pallasSqr(base)
            word >>= 1
        }
    }
    return result
}

public func pallasFromHex(_ hex: String) -> PallasFp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = PallasFp.from64(limbs)
    return pallasMul(raw, PallasFp.from64(PallasFp.R2_MOD_P))
}

public func pallasToHex(_ a: PallasFp) -> String {
    let limbs = pallasToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}
