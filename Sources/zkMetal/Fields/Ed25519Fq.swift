// Ed25519 scalar field Fq arithmetic (CPU-side)
// q = 2^252 + 27742317777372353535851937790883648493
//   = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed
// Field elements as 4x64-bit limbs in Montgomery form (little-endian).

import Foundation
import NeonFieldOps

public struct Ed25519Fq: Equatable {
    public var v: (UInt64, UInt64, UInt64, UInt64)

    public static let Q: [UInt64] = [
        0x5812631a5cf5d3ed, 0x14def9dea2f79cd6,
        0x0000000000000000, 0x1000000000000000
    ]

    // R mod q: 2^256 mod q
    // 2^256 / q ≈ 16, remainder = 2^256 - 16*q
    // 16 * q = 0x10000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed0
    // Actually, q < 2^253, so 2^256 > 8*q. Let's compute:
    // R = 2^256 mod q. Since q ≈ 2^252, R ≈ 2^4 * (something).
    // 2^256 = q * 16 + r => r = 2^256 - 16*q
    // 16*q[0] = 0x8126314a5cf5d3ed0 -- this gets complicated. Let me use proper formula.
    // R = 2^256 mod q
    // We compute: 2^256 = 2^4 * 2^252 = 16 * (q - 27742317777372353535851937790883648493)
    // So 2^256 = 16*q - 16*27742317777372353535851937790883648493
    // R = -16 * 27742317777372353535851937790883648493 mod q
    //   = q - (16 * 27742317777372353535851937790883648493 mod q)
    // 16 * 27742317777372353535851937790883648493 = 443877084437957656573631004654138375888
    // R = q - 443877084437957656573631004654138375888
    // q = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    // R = 7237005577332262213973186563042994240413239274941949949428319933631315875101
    // In hex: 0x0ffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951d
    public static let R_MOD_Q: [UInt64] = [
        0xd6ec31748d98951d, 0xc6ef5bf4737dcf70,
        0xfffffffffffffffe, 0x0fffffffffffffff
    ]

    // R^2 mod q
    // Need to compute (2^256)^2 mod q = R * R mod q
    // R_MOD_Q^2 mod q -- we'll compute via schoolbook
    // For bootstrap, we can compute this numerically.
    // R^2 mod q = 0x0399411b7c309a3dceec73d217f5be65d00e1ba768859347a40611e3449c0f01
    public static let R2_MOD_Q: [UInt64] = [
        0xa40611e3449c0f01, 0xd00e1ba768859347,
        0xceec73d217f5be65, 0x0399411b7c309a3d
    ]

    // -q^(-1) mod 2^64
    // q[0] = 0x5812631a5cf5d3ed
    // inv such that q[0] * inv ≡ -1 (mod 2^64)
    public static let INV: UInt64 = 0xd2b51da312547e1b

    public static var zero: Ed25519Fq { Ed25519Fq(v: (0, 0, 0, 0)) }

    public static var one: Ed25519Fq {
        Ed25519Fq(v: (R_MOD_Q[0], R_MOD_Q[1], R_MOD_Q[2], R_MOD_Q[3]))
    }

    public init(v: (UInt64, UInt64, UInt64, UInt64)) {
        self.v = v
    }

    public func toLimbs() -> [UInt64] { [v.0, v.1, v.2, v.3] }

    public static func fromLimbs(_ l: [UInt64]) -> Ed25519Fq {
        Ed25519Fq(v: (l[0], l[1], l[2], l[3]))
    }

    public var isZero: Bool {
        v.0 == 0 && v.1 == 0 && v.2 == 0 && v.3 == 0
    }

    public static func == (lhs: Ed25519Fq, rhs: Ed25519Fq) -> Bool {
        lhs.v.0 == rhs.v.0 && lhs.v.1 == rhs.v.1 &&
        lhs.v.2 == rhs.v.2 && lhs.v.3 == rhs.v.3
    }

    public func to32() -> [UInt32] {
        [UInt32(v.0 & 0xFFFFFFFF), UInt32(v.0 >> 32),
         UInt32(v.1 & 0xFFFFFFFF), UInt32(v.1 >> 32),
         UInt32(v.2 & 0xFFFFFFFF), UInt32(v.2 >> 32),
         UInt32(v.3 & 0xFFFFFFFF), UInt32(v.3 >> 32)]
    }
}

// MARK: - Field Operations (zero-copy C CIOS)

@inline(__always)
private func withFqPtr<T>(_ a: Ed25519Fq, _ body: (UnsafePointer<UInt64>) -> T) -> T {
    var v = a.v
    return withUnsafePointer(to: &v) { p in
        body(UnsafeRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
}

@inline(__always)
private func fqFromRaw(_ body: (UnsafeMutablePointer<UInt64>) -> Void) -> Ed25519Fq {
    var rv: (UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0)
    withUnsafeMutablePointer(to: &rv) { p in
        body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: UInt64.self))
    }
    return Ed25519Fq(v: rv)
}

@inline(__always)
public func ed25519FqMul(_ a: Ed25519Fq, _ b: Ed25519Fq) -> Ed25519Fq {
    withFqPtr(a) { ap in withFqPtr(b) { bp in
        fqFromRaw { rp in ed25519_fq_mul(ap, bp, rp) }
    }}
}

@inline(__always)
public func ed25519FqAdd(_ a: Ed25519Fq, _ b: Ed25519Fq) -> Ed25519Fq {
    withFqPtr(a) { ap in withFqPtr(b) { bp in
        fqFromRaw { rp in ed25519_fq_add(ap, bp, rp) }
    }}
}

@inline(__always)
public func ed25519FqSub(_ a: Ed25519Fq, _ b: Ed25519Fq) -> Ed25519Fq {
    withFqPtr(a) { ap in withFqPtr(b) { bp in
        fqFromRaw { rp in ed25519_fq_sub(ap, bp, rp) }
    }}
}

@inline(__always)
public func ed25519FqSqr(_ a: Ed25519Fq) -> Ed25519Fq { ed25519FqMul(a, a) }
public func ed25519FqNeg(_ a: Ed25519Fq) -> Ed25519Fq {
    if a.isZero { return a }
    return ed25519FqSub(Ed25519Fq(v: (Ed25519Fq.Q[0], Ed25519Fq.Q[1], Ed25519Fq.Q[2], Ed25519Fq.Q[3])), a)
}

/// Convert raw integer to Montgomery form
public func ed25519FqFromInt(_ val: UInt64) -> Ed25519Fq {
    let raw = Ed25519Fq(v: (val, 0, 0, 0))
    return ed25519FqMul(raw, Ed25519Fq.fromLimbs(Ed25519Fq.R2_MOD_Q))
}

/// Convert from Montgomery form to integer
public func ed25519FqToInt(_ a: Ed25519Fq) -> [UInt64] {
    let one = Ed25519Fq(v: (1, 0, 0, 0))
    return ed25519FqMul(a, one).toLimbs()
}

/// Convert raw 256-bit limbs to Montgomery form
public func ed25519FqFromRaw(_ limbs: [UInt64]) -> Ed25519Fq {
    let raw = Ed25519Fq.fromLimbs(limbs)
    return ed25519FqMul(raw, Ed25519Fq.fromLimbs(Ed25519Fq.R2_MOD_Q))
}

/// Field inverse via Fermat's little theorem: a^(q-2) mod q
public func ed25519FqInverse(_ a: Ed25519Fq) -> Ed25519Fq {
    var result = Ed25519Fq.one
    var base = a
    var exp = Ed25519Fq.Q.map { $0 }
    if exp[0] >= 2 { exp[0] -= 2 }
    else { exp[0] = exp[0] &- 2; exp[1] -= 1 }

    for i in 0..<4 {
        var word = exp[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = ed25519FqMul(result, base)
            }
            base = ed25519FqSqr(base)
            word >>= 1
        }
    }
    return result
}

/// Batch modular inverse using Montgomery's trick
public func ed25519FqBatchInverse(_ elems: [Ed25519Fq]) -> [Ed25519Fq] {
    let n = elems.count
    if n == 0 { return [] }

    var prods = [Ed25519Fq](repeating: Ed25519Fq.one, count: n)
    prods[0] = elems[0]
    for i in 1..<n {
        prods[i] = ed25519FqMul(prods[i - 1], elems[i])
    }

    var inv = ed25519FqInverse(prods[n - 1])
    var result = [Ed25519Fq](repeating: Ed25519Fq.zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = ed25519FqMul(inv, prods[i - 1])
        inv = ed25519FqMul(inv, elems[i])
    }
    result[0] = inv
    return result
}

/// Encode Fq to bytes (little-endian, 32 bytes)
public func ed25519FqToBytes(_ a: Ed25519Fq) -> [UInt8] {
    let limbs = ed25519FqToInt(a)
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((limbs[i] >> (j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Decode bytes to Fq (little-endian, with reduction)
public func ed25519FqFromBytes(_ bytes: [UInt8]) -> Ed25519Fq {
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        for j in 0..<8 {
            if i * 8 + j < bytes.count {
                limbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
        }
    }
    // Reduce mod q
    while ed25519FqGte(limbs, Ed25519Fq.Q) {
        limbs = ed25519FqSub256(limbs, Ed25519Fq.Q).0
    }
    return ed25519FqFromRaw(limbs)
}

/// Reduce a 64-byte (512-bit) value mod q, as needed for Ed25519 hash-to-scalar
public func ed25519FqFromBytes64(_ bytes: [UInt8]) -> Ed25519Fq {
    // Split into two 256-bit halves, reduce high half by R mod q, add
    var lo: [UInt64] = [0, 0, 0, 0]
    var hi: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        for j in 0..<8 {
            if i * 8 + j < 32 { lo[i] |= UInt64(bytes[i * 8 + j]) << (j * 8) }
            if 32 + i * 8 + j < bytes.count { hi[i] |= UInt64(bytes[32 + i * 8 + j]) << (j * 8) }
        }
    }
    // Convert both to Montgomery form, compute hi * R + lo = hi * 2^256 + lo (mod q)
    let loMont = ed25519FqFromRaw(lo)
    let hiMont = ed25519FqFromRaw(hi)
    // hi * 2^256 mod q = hi * R_MOD_Q (in integer domain, but we're in Montgomery)
    // In Montgomery: to compute (hi_raw * 2^256 + lo_raw) mod q
    // = hi_mont * R_mont + lo_mont where R_mont = R mod q in Montgomery = R * R mod q / R = R mod q
    // Actually simpler: hiMont represents hi*R in Mont. We want hi*2^256 = hi * R in the integer domain.
    // In Montgomery form, the value representing hi*R in integer = hi*R * R = hiMont * R.
    // That's hiMont * (something). Let me use a different approach.
    // Direct: compute hi_raw * (2^256 mod q) + lo_raw, all mod q.
    // 2^256 mod q is R_MOD_Q in the integer domain.
    // So: result = hi_raw * R_MOD_Q_int + lo_raw mod q
    let rModQMont = ed25519FqFromRaw(Ed25519Fq.R_MOD_Q)
    let hiTimesR = ed25519FqMul(hiMont, rModQMont)
    return ed25519FqAdd(hiTimesR, loMont)
}

// MARK: - 256-bit helpers

func ed25519FqGte(_ a: [UInt64], _ b: [UInt64]) -> Bool {
    for i in stride(from: 3, through: 0, by: -1) {
        if a[i] > b[i] { return true }
        if a[i] < b[i] { return false }
    }
    return true
}

func ed25519FqSub256(_ a: [UInt64], _ b: [UInt64]) -> ([UInt64], Bool) {
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
