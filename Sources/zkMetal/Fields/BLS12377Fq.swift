// BLS12-377 base field Fq arithmetic (CPU-side)
// q = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
// 377-bit prime, field elements as 12x32-bit limbs in Montgomery form (little-endian).

import Foundation
import NeonFieldOps

public struct Fq377 {
    public var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                   UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    // q in 6x64-bit limbs (little-endian)
    public static let P: [UInt64] = [
        0x8508c00000000001, 0x170b5d4430000000,
        0x1ef3622fba094800, 0x1a22d9f300f5138f,
        0xc63b05c06ca1493b, 0x01ae3a4617c510ea
    ]

    // R mod q (Montgomery form of 1): 2^384 mod q
    public static let R_MOD_Q: [UInt64] = [
        0x02cdffffffffff68, 0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2, 0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8, 0x008d6661e2fdf49a
    ]

    // R^2 mod q: 2^768 mod q
    public static let R2_MOD_Q: [UInt64] = [
        0xb786686c9400cd22, 0x0329fcaab00431b1,
        0x22a5f11162d6b46d, 0xbfdf7d03827dc3ac,
        0x837e92f041790bf9, 0x006dfccb1e914b88
    ]

    // -q^(-1) mod 2^64
    public static let INV: UInt64 = 0x8508bfffffffffff

    public static var zero: Fq377 {
        Fq377(v: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    }

    public static var one: Fq377 {
        // R mod q in 32-bit limbs (little-endian)
        Fq377(v: (0xffffff68, 0x02cdffff, 0x7fffffb1, 0x51409f83,
                  0x8a7d3ff2, 0x9f7db3a9, 0x6e7c6305, 0x7b4e97b7,
                  0x803c84e8, 0x4cf495bf, 0xe2fdf49a, 0x008d6661))
    }

    public init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                     UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    // Convert to 6x64-bit limbs for arithmetic
    public func to64() -> [UInt64] {
        let l = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8, v.9, v.10, v.11]
        return [
            UInt64(l[0]) | (UInt64(l[1]) << 32),
            UInt64(l[2]) | (UInt64(l[3]) << 32),
            UInt64(l[4]) | (UInt64(l[5]) << 32),
            UInt64(l[6]) | (UInt64(l[7]) << 32),
            UInt64(l[8]) | (UInt64(l[9]) << 32),
            UInt64(l[10]) | (UInt64(l[11]) << 32),
        ]
    }

    public static func from64(_ limbs: [UInt64]) -> Fq377 {
        Fq377(v: (
            UInt32(limbs[0] & 0xFFFFFFFF), UInt32(limbs[0] >> 32),
            UInt32(limbs[1] & 0xFFFFFFFF), UInt32(limbs[1] >> 32),
            UInt32(limbs[2] & 0xFFFFFFFF), UInt32(limbs[2] >> 32),
            UInt32(limbs[3] & 0xFFFFFFFF), UInt32(limbs[3] >> 32),
            UInt32(limbs[4] & 0xFFFFFFFF), UInt32(limbs[4] >> 32),
            UInt32(limbs[5] & 0xFFFFFFFF), UInt32(limbs[5] >> 32)
        ))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0 &&
        v.4 == 0 && v.5 == 0 && v.6 == 0 && v.7 == 0 &&
        v.8 == 0 && v.9 == 0 && v.10 == 0 && v.11 == 0
    }
}

// MARK: - 384-bit Arithmetic Helpers

public func add384(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], UInt64) {
    var r = [UInt64](repeating: 0, count: 6)
    var carry: UInt64 = 0
    for i in 0..<6 {
        let (s1, c1) = a[i].addingReportingOverflow(b[i])
        let (s2, c2) = s1.addingReportingOverflow(carry)
        r[i] = s2
        carry = (c1 ? 1 : 0) + (c2 ? 1 : 0)
    }
    return (r, carry)
}

public func sub384(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], Bool) {
    var r = [UInt64](repeating: 0, count: 6)
    var borrow: Bool = false
    for i in 0..<6 {
        let (s1, b1) = a[i].subtractingReportingOverflow(b[i])
        let (s2, b2) = s1.subtractingReportingOverflow(borrow ? 1 : 0)
        r[i] = s2
        borrow = b1 || b2
    }
    return (r, borrow)
}

public func gte384(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    for i in stride(from: 5, through: 0, by: -1) {
        if a[i] > b[i] { return true }
        if a[i] < b[i] { return false }
    }
    return true
}

// MARK: - Zero-copy pointer helpers
// Fq377.v is 12×UInt32 = 48 bytes = same layout as uint64_t[6] on little-endian

@inline(__always)
private func withFq377Ptr<T>(_ a: Fq377, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func fq377FromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> Fq377 {
    var rv: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
             UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) = (0,0,0,0,0,0,0,0,0,0,0,0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return Fq377(v: rv)
}

// MARK: - Field Operations

// Montgomery multiplication: (a * b * R^-1) mod q — C CIOS
@inline(__always)
public func fq377Mul(_ a: Fq377, _ b: Fq377) -> Fq377 {
    withFq377Ptr(a) { ap in withFq377Ptr(b) { bp in
        fq377FromRaw { rp in bls12_377_fq_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func fq377Add(_ a: Fq377, _ b: Fq377) -> Fq377 {
    withFq377Ptr(a) { ap in withFq377Ptr(b) { bp in
        fq377FromRaw { rp in bls12_377_fq_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func fq377Sub(_ a: Fq377, _ b: Fq377) -> Fq377 {
    withFq377Ptr(a) { ap in withFq377Ptr(b) { bp in
        fq377FromRaw { rp in bls12_377_fq_sub(ap, bp, rp) }
    }}
}

@inline(__always)
public func fq377Sqr(_ a: Fq377) -> Fq377 {
    withFq377Ptr(a) { ap in
        fq377FromRaw { rp in bls12_377_fq_sqr(ap, rp) }
    }
}
public func fq377Double(_ a: Fq377) -> Fq377 { fq377Add(a, a) }

// Convert integer to Montgomery form
public func fq377FromInt(_ val: UInt64) -> Fq377 {
    let limbs: [UInt64] = [val, 0, 0, 0, 0, 0]
    let raw = Fq377.from64(limbs)
    return fq377Mul(raw, Fq377.from64(Fq377.R2_MOD_Q))
}

// Convert from Montgomery form to integer
public func fq377ToInt(_ a: Fq377) -> [UInt64] {
    let one: [UInt64] = [1, 0, 0, 0, 0, 0]
    return fq377Mul(a, Fq377.from64(one)).to64()
}

// Field negation
@inline(__always)
public func fq377Neg(_ a: Fq377) -> Fq377 {
    withFq377Ptr(a) { ap in
        fq377FromRaw { rp in bls12_377_fq_neg(ap, rp) }
    }
}

// Field inverse via Fermat's little theorem: a^(q-2) mod q
public func fq377Inverse(_ a: Fq377) -> Fq377 {
    var result = Fq377.one
    var base = a
    var exp = Fq377.P.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<6 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = fq377Mul(result, base)
            }
            base = fq377Sqr(base)
            word >>= 1
        }
    }
    return result
}
