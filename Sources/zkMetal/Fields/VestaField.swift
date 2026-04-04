// Vesta base field Fp arithmetic (CPU-side)
// p = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// (= Pallas scalar field Fq)
// Field elements as 8x32-bit limbs in Montgomery form (little-endian).

import Foundation

public struct VestaFp {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    public static let P: [UInt64] = [
        0x8c46eb2100000001, 0x224698fc0994a8dd,
        0x0000000000000000, 0x4000000000000000
    ]

    // R mod p (Montgomery form of 1): 2^256 mod p
    public static let R_MOD_P: [UInt64] = [
        0x5b2b3e9cfffffffd, 0x992c350be3420567,
        0xffffffffffffffff, 0x3fffffffffffffff
    ]

    // R^2 mod p: 2^512 mod p
    public static let R2_MOD_P: [UInt64] = [
        0xfc9678ff0000000f, 0x67bb433d891a16e3,
        0x7fae231004ccf590, 0x096d41af7ccfdaa9
    ]

    // -p^(-1) mod 2^64
    public static let INV: UInt64 = 0x8c46eb20ffffffff

    public static var zero: VestaFp { VestaFp(v: (0, 0, 0, 0, 0, 0, 0, 0)) }

    public static var one: VestaFp {
        VestaFp(v: (0xfffffffd, 0x5b2b3e9c, 0xe3420567, 0x992c350b,
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

    public static func from64(_ limbs: [UInt64]) -> VestaFp {
        VestaFp(v: (
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

// MARK: - Field Operations

public func vestaMul(_ a: VestaFp, _ b: VestaFp) -> VestaFp {
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

        let m = t[0] &* VestaFp.INV
        carry = 0
        for j in 0..<4 {
            let (hi, lo) = m.multipliedFullWidth(by: VestaFp.P[j])
            let (s1, c1) = t[j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            t[j] = s2
            carry = hi + (c1 ? 1 : 0) + (c2 ? 1 : 0)
        }
        t[4] = t[4] &+ carry

        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = 0
    }

    var r = Array(t[0..<4])
    if gte256(r, VestaFp.P) {
        (r, _) = sub256(r, VestaFp.P)
    }
    return VestaFp.from64(r)
}

public func vestaAdd(_ a: VestaFp, _ b: VestaFp) -> VestaFp {
    var (r, carry) = add256(a.to64(), b.to64())
    if carry != 0 || gte256(r, VestaFp.P) {
        (r, _) = sub256(r, VestaFp.P)
    }
    return VestaFp.from64(r)
}

public func vestaSub(_ a: VestaFp, _ b: VestaFp) -> VestaFp {
    var (r, borrow) = sub256(a.to64(), b.to64())
    if borrow {
        (r, _) = add256(r, VestaFp.P)
    }
    return VestaFp.from64(r)
}

public func vestaSqr(_ a: VestaFp) -> VestaFp { vestaMul(a, a) }
public func vestaDouble(_ a: VestaFp) -> VestaFp { vestaAdd(a, a) }

public func vestaFromInt(_ val: UInt64) -> VestaFp {
    let limbs: [UInt64] = [val, 0, 0, 0]
    let raw = VestaFp.from64(limbs)
    return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
}

public func vestaToInt(_ a: VestaFp) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0]
    return vestaMul(a, VestaFp.from64(one)).to64()
}

public func vestaNeg(_ a: VestaFp) -> VestaFp {
    if a.isZero { return a }
    let (r, _) = sub256(VestaFp.P, a.to64())
    return VestaFp.from64(r)
}

public func vestaInverse(_ a: VestaFp) -> VestaFp {
    var result = VestaFp.one
    var base = a
    var exp = VestaFp.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = vestaMul(result, base)
            }
            base = vestaSqr(base)
            word >>= 1
        }
    }
    return result
}

public func vestaFromHex(_ hex: String) -> VestaFp {
    let clean = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - clean.count)) + clean
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: i * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[3 - i] = UInt64(padded[start..<end], radix: 16) ?? 0
    }
    let raw = VestaFp.from64(limbs)
    return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
}

public func vestaToHex(_ a: VestaFp) -> String {
    let limbs = vestaToInt(a)
    return "0x" + limbs.reversed().map { String(format: "%016llx", $0) }.joined()
}
