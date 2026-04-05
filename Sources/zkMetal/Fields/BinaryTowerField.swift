// Binary tower field arithmetic engine for Binius-style proofs over GF(2) extensions.
//
// Tower construction:
//   GF(2) -> GF(2^2) -> GF(2^4) -> GF(2^8) -> GF(2^16) -> GF(2^32) -> GF(2^64) -> GF(2^128)
//
// Each level doubles the extension degree:
//   GF(2^{2k}) = GF(2^k)[X] / (X^2 + X + alpha_k)
//
// Addition = XOR (free), multiplication = carry-less multiply + reduction.
// Uses ARM64 PMULL via NeonFieldOps C library for hardware acceleration.

import Foundation
import NeonFieldOps

// MARK: - BinaryTowerProtocol

/// Unified API across all binary tower levels.
/// Every tower field element supports add (XOR), mul (Karatsuba + reduction),
/// inverse, squaring, and exponentiation.
public protocol BinaryTowerProtocol: Equatable, CustomStringConvertible {
    static var zero: Self { get }
    static var one: Self { get }
    static var extensionDegree: Int { get }

    var isZero: Bool { get }

    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self

    func inverse() -> Self
    func squared() -> Self
    func pow(_ n: Int) -> Self

    /// Embed a GF(2^8) element into this tower level (lo coefficient, zeros above)
    init(fromGF8 value: UInt8)

    /// Project back to GF(2^8) (returns lo byte, only valid if hi parts are zero)
    var toGF8: UInt8 { get }
}

// Default pow implementation via square-and-multiply
extension BinaryTowerProtocol {
    public func pow(_ n: Int) -> Self {
        if n == 0 { return Self.one }
        var result = Self.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base.squared()
            exp >>= 1
        }
        return result
    }
}

// MARK: - BinaryTower8 — GF(2^8) with AES polynomial

/// GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B).
/// Uses NEON-accelerated log/exp table multiplication via NeonFieldOps.
public struct BinaryTower8: BinaryTowerProtocol {
    public var value: UInt8

    public static let zero = BinaryTower8(value: 0)
    public static let one = BinaryTower8(value: 1)
    public static let extensionDegree = 8

    /// AES S-box generator element
    public static let generator = BinaryTower8(value: 3)

    public init(value: UInt8) {
        self.value = value
    }

    public init(fromGF8 value: UInt8) {
        self.value = value
    }

    public var toGF8: UInt8 { value }
    public var isZero: Bool { value == 0 }
    public var description: String { "BT8(0x\(String(value, radix: 16, uppercase: true)))" }

    public static func == (lhs: BinaryTower8, rhs: BinaryTower8) -> Bool {
        lhs.value == rhs.value
    }

    // Addition = XOR
    @inline(__always)
    public static func + (a: BinaryTower8, b: BinaryTower8) -> BinaryTower8 {
        BinaryTower8(value: a.value ^ b.value)
    }

    // Subtraction = XOR (char 2)
    @inline(__always)
    public static func - (a: BinaryTower8, b: BinaryTower8) -> BinaryTower8 {
        BinaryTower8(value: a.value ^ b.value)
    }

    // Multiplication via NEON-accelerated log/exp tables
    @inline(__always)
    public static func * (a: BinaryTower8, b: BinaryTower8) -> BinaryTower8 {
        BinaryTower8(value: bt_gf8_mul(a.value, b.value))
    }

    // Inverse via Fermat's little theorem: a^(-1) = a^(2^8-2) = a^254
    public func inverse() -> BinaryTower8 {
        precondition(!isZero, "Cannot invert zero in GF(2^8)")
        return BinaryTower8(value: bt_gf8_inv(value))
    }

    // Squaring is a linear operation in characteristic 2 (Frobenius endomorphism)
    @inline(__always)
    public func squared() -> BinaryTower8 {
        BinaryTower8(value: bt_gf8_sqr(value))
    }
}

// MARK: - BinaryTower16 — GF(2^16) via Karatsuba over GF(2^8)

/// GF(2^16) = GF(2^8)[X] / (X^2 + X + alpha).
/// Uses PMULL-accelerated carry-less multiply via NeonFieldOps.
public struct BinaryTower16: BinaryTowerProtocol {
    public var lo: UInt8  // coefficient of 1
    public var hi: UInt8  // coefficient of X

    public static let zero = BinaryTower16(lo: 0, hi: 0)
    public static let one = BinaryTower16(lo: 1, hi: 0)
    public static let extensionDegree = 16

    public init(lo: UInt8, hi: UInt8) {
        self.lo = lo
        self.hi = hi
    }

    public init(value: UInt16) {
        self.lo = UInt8(value & 0xFF)
        self.hi = UInt8(value >> 8)
    }

    public init(fromGF8 value: UInt8) {
        self.lo = value
        self.hi = 0
    }

    public var toGF8: UInt8 { lo }

    public var toUInt16: UInt16 {
        UInt16(lo) | (UInt16(hi) << 8)
    }

    public var isZero: Bool { lo == 0 && hi == 0 }
    public var description: String { "BT16(0x\(String(toUInt16, radix: 16, uppercase: true)))" }

    public static func == (lhs: BinaryTower16, rhs: BinaryTower16) -> Bool {
        lhs.lo == rhs.lo && lhs.hi == rhs.hi
    }

    @inline(__always)
    public static func + (a: BinaryTower16, b: BinaryTower16) -> BinaryTower16 {
        BinaryTower16(lo: a.lo ^ b.lo, hi: a.hi ^ b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryTower16, b: BinaryTower16) -> BinaryTower16 {
        a + b  // char 2
    }

    // PMULL-accelerated multiply
    @inline(__always)
    public static func * (a: BinaryTower16, b: BinaryTower16) -> BinaryTower16 {
        let result = bt_gf16_mul(a.toUInt16, b.toUInt16)
        return BinaryTower16(value: result)
    }

    // Inverse via norm to GF(2^8) then invert in subfield
    public func inverse() -> BinaryTower16 {
        precondition(!isZero, "Cannot invert zero in GF(2^16)")
        let result = bt_gf16_inv(toUInt16)
        return BinaryTower16(value: result)
    }

    @inline(__always)
    public func squared() -> BinaryTower16 {
        let result = bt_gf16_sqr(toUInt16)
        return BinaryTower16(value: result)
    }
}

// MARK: - BinaryTower32 — GF(2^32) via Karatsuba over GF(2^16)

/// GF(2^32) = GF(2^16)[X] / (X^2 + X + beta).
/// Uses PMULL-accelerated carry-less multiply via NeonFieldOps.
public struct BinaryTower32: BinaryTowerProtocol {
    public var lo: UInt16
    public var hi: UInt16

    public static let zero = BinaryTower32(lo: 0, hi: 0)
    public static let one = BinaryTower32(lo: 1, hi: 0)
    public static let extensionDegree = 32

    public init(lo: UInt16, hi: UInt16) {
        self.lo = lo
        self.hi = hi
    }

    public init(value: UInt32) {
        self.lo = UInt16(value & 0xFFFF)
        self.hi = UInt16(value >> 16)
    }

    public init(fromGF8 value: UInt8) {
        self.lo = UInt16(value)
        self.hi = 0
    }

    public var toGF8: UInt8 { UInt8(lo & 0xFF) }

    public var toUInt32: UInt32 {
        UInt32(lo) | (UInt32(hi) << 16)
    }

    public var isZero: Bool { lo == 0 && hi == 0 }
    public var description: String { "BT32(0x\(String(toUInt32, radix: 16, uppercase: true)))" }

    public static func == (lhs: BinaryTower32, rhs: BinaryTower32) -> Bool {
        lhs.lo == rhs.lo && lhs.hi == rhs.hi
    }

    @inline(__always)
    public static func + (a: BinaryTower32, b: BinaryTower32) -> BinaryTower32 {
        BinaryTower32(lo: a.lo ^ b.lo, hi: a.hi ^ b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryTower32, b: BinaryTower32) -> BinaryTower32 {
        a + b
    }

    // PMULL-accelerated multiply
    @inline(__always)
    public static func * (a: BinaryTower32, b: BinaryTower32) -> BinaryTower32 {
        let result = bt_gf32_mul(a.toUInt32, b.toUInt32)
        return BinaryTower32(value: result)
    }

    public func inverse() -> BinaryTower32 {
        precondition(!isZero, "Cannot invert zero in GF(2^32)")
        let result = bt_gf32_inv(toUInt32)
        return BinaryTower32(value: result)
    }

    @inline(__always)
    public func squared() -> BinaryTower32 {
        let result = bt_gf32_sqr(toUInt32)
        return BinaryTower32(value: result)
    }
}

// MARK: - BinaryTower64 — GF(2^64) via PMULL

/// GF(2^64) with irreducible x^64 + x^4 + x^3 + x + 1.
/// Direct PMULL carry-less multiply — single hardware instruction for the core multiply.
public struct BinaryTower64: BinaryTowerProtocol {
    public var value: UInt64

    public static let zero = BinaryTower64(value: 0)
    public static let one = BinaryTower64(value: 1)
    public static let extensionDegree = 64

    public init(value: UInt64) {
        self.value = value
    }

    public init(fromGF8 value: UInt8) {
        self.value = UInt64(value)
    }

    public var toGF8: UInt8 { UInt8(value & 0xFF) }
    public var isZero: Bool { value == 0 }
    public var description: String { "BT64(0x\(String(value, radix: 16, uppercase: true)))" }

    public static func == (lhs: BinaryTower64, rhs: BinaryTower64) -> Bool {
        lhs.value == rhs.value
    }

    @inline(__always)
    public static func + (a: BinaryTower64, b: BinaryTower64) -> BinaryTower64 {
        BinaryTower64(value: a.value ^ b.value)
    }

    @inline(__always)
    public static func - (a: BinaryTower64, b: BinaryTower64) -> BinaryTower64 {
        BinaryTower64(value: a.value ^ b.value)
    }

    // PMULL-accelerated multiply
    @inline(__always)
    public static func * (a: BinaryTower64, b: BinaryTower64) -> BinaryTower64 {
        BinaryTower64(value: bt_gf64_mul(a.value, b.value))
    }

    // Itoh-Tsujii inversion via C
    public func inverse() -> BinaryTower64 {
        precondition(!isZero, "Cannot invert zero in GF(2^64)")
        return BinaryTower64(value: bt_gf64_inv(value))
    }

    @inline(__always)
    public func squared() -> BinaryTower64 {
        BinaryTower64(value: bt_gf64_sqr(value))
    }
}

// MARK: - BinaryTower128 — GF(2^128) via Karatsuba over GF(2^64) with PMULL2

/// GF(2^128) with irreducible x^128 + x^7 + x^2 + x + 1 (AES-GCM polynomial).
/// Karatsuba multiplication using 3 PMULL instructions, same as GHASH.
public struct BinaryTower128: BinaryTowerProtocol {
    public var lo: UInt64
    public var hi: UInt64

    public static let zero = BinaryTower128(lo: 0, hi: 0)
    public static let one = BinaryTower128(lo: 1, hi: 0)
    public static let extensionDegree = 128

    public init(lo: UInt64, hi: UInt64) {
        self.lo = lo
        self.hi = hi
    }

    public init(fromGF8 value: UInt8) {
        self.lo = UInt64(value)
        self.hi = 0
    }

    public var toGF8: UInt8 { UInt8(lo & 0xFF) }
    public var isZero: Bool { lo == 0 && hi == 0 }

    public var description: String {
        "BT128(0x\(String(hi, radix: 16, uppercase: true))_\(String(lo, radix: 16, uppercase: true)))"
    }

    public static func == (lhs: BinaryTower128, rhs: BinaryTower128) -> Bool {
        lhs.lo == rhs.lo && lhs.hi == rhs.hi
    }

    @inline(__always)
    public static func + (a: BinaryTower128, b: BinaryTower128) -> BinaryTower128 {
        BinaryTower128(lo: a.lo ^ b.lo, hi: a.hi ^ b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryTower128, b: BinaryTower128) -> BinaryTower128 {
        a + b
    }

    // PMULL2-accelerated Karatsuba multiply (3 carry-less multiplies + Barrett reduction)
    @inline(__always)
    public static func * (a: BinaryTower128, b: BinaryTower128) -> BinaryTower128 {
        var aArr: [UInt64] = [a.lo, a.hi]
        var bArr: [UInt64] = [b.lo, b.hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_mul(&aArr, &bArr, &rArr)
        return BinaryTower128(lo: rArr[0], hi: rArr[1])
    }

    // Itoh-Tsujii inversion via C
    public func inverse() -> BinaryTower128 {
        precondition(!isZero, "Cannot invert zero in GF(2^128)")
        var aArr: [UInt64] = [lo, hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_inv(&aArr, &rArr)
        return BinaryTower128(lo: rArr[0], hi: rArr[1])
    }

    @inline(__always)
    public func squared() -> BinaryTower128 {
        var aArr: [UInt64] = [lo, hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_sqr(&aArr, &rArr)
        return BinaryTower128(lo: rArr[0], hi: rArr[1])
    }
}

// MARK: - Tower Embedding Utilities

/// Embed a GF(2^8) element up through the tower levels.
/// This is the canonical embedding: place the byte in the lo-most position.
public enum BinaryTowerEmbed {

    /// Embed GF(2^8) into GF(2^128) for cross-level consistency checks.
    @inline(__always)
    public static func gf8Into128(_ x: UInt8) -> BinaryTower128 {
        BinaryTower128(lo: UInt64(x), hi: 0)
    }

    /// Embed GF(2^8) into GF(2^64)
    @inline(__always)
    public static func gf8Into64(_ x: UInt8) -> BinaryTower64 {
        BinaryTower64(value: UInt64(x))
    }

    /// Embed GF(2^8) into GF(2^32)
    @inline(__always)
    public static func gf8Into32(_ x: UInt8) -> BinaryTower32 {
        BinaryTower32(value: UInt32(x))
    }

    /// Embed GF(2^8) into GF(2^16)
    @inline(__always)
    public static func gf8Into16(_ x: UInt8) -> BinaryTower16 {
        BinaryTower16(value: UInt16(x))
    }

    /// Check if a BinaryTower128 element is actually in the GF(2^8) subfield
    @inline(__always)
    public static func isInGF8(_ x: BinaryTower128) -> Bool {
        x.hi == 0 && x.lo <= 0xFF
    }

    /// Check if a BinaryTower64 element is in the GF(2^8) subfield
    @inline(__always)
    public static func isInGF8(_ x: BinaryTower64) -> Bool {
        x.value <= 0xFF
    }
}

// MARK: - Batch Operations (NEON-vectorized)

/// NEON-accelerated batch operations for binary tower fields.
public enum BinaryTowerBatch {

    /// Batch GF(2^64) multiply: out[i] = a[i] * b[i]
    public static func mul64(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf64_batch_mul(a, b, &out, Int32(a.count))
        return out
    }

    /// Batch GF(2^64) add (XOR): out[i] = a[i] ^ b[i]
    public static func add64(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf64_batch_add(a, b, &out, Int32(a.count))
        return out
    }

    /// Batch GF(2^128) multiply: out[i] = a[i] * b[i]
    /// Input/output arrays interleave lo/hi: [a0_lo, a0_hi, a1_lo, a1_hi, ...]
    public static func mul128(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        precondition(a.count % 2 == 0)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf128_batch_mul(a, b, &out, Int32(a.count / 2))
        return out
    }

    /// Batch GF(2^128) add (XOR)
    public static func add128(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        precondition(a.count % 2 == 0)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf128_batch_add(a, b, &out, Int32(a.count / 2))
        return out
    }
}
