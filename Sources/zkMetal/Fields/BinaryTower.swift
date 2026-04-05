// Binary tower field arithmetic (CPU-side)
// GF(2) → GF(2^2) → GF(2^4) → GF(2^8) → GF(2^16) → GF(2^32) → GF(2^64) → GF(2^128)
// Each extension GF(2^{2k}) = GF(2^k)[X] / (X^2 + X + α)
// Addition = XOR (free), multiplication = carry-less multiply + reduce
// Foundation for Binius-style binary STARKs

import Foundation

// MARK: - GF(2^8) — AES field, the practical base

/// GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B)
/// This is the AES polynomial, extremely well-studied.
public struct BinaryField8: Equatable, CustomStringConvertible {
    public var value: UInt8

    public static let zero = BinaryField8(value: 0)
    public static let one = BinaryField8(value: 1)

    public init(value: UInt8) {
        self.value = value
    }

    public var isZero: Bool { value == 0 }

    public var description: String { "GF256(\(value))" }

    // Addition = XOR
    @inline(__always)
    public static func + (a: BinaryField8, b: BinaryField8) -> BinaryField8 {
        BinaryField8(value: a.value ^ b.value)
    }

    // Subtraction = XOR (same as addition in char 2)
    @inline(__always)
    public static func - (a: BinaryField8, b: BinaryField8) -> BinaryField8 {
        BinaryField8(value: a.value ^ b.value)
    }

    // Multiplication via carry-less multiply + reduction by x^8+x^4+x^3+x+1
    @inline(__always)
    public static func * (a: BinaryField8, b: BinaryField8) -> BinaryField8 {
        BinaryField8(value: gf8Mul(a.value, b.value))
    }

    // Inverse via lookup table (precomputed from Fermat: a^254)
    public func inverse() -> BinaryField8 {
        precondition(!isZero, "Cannot invert zero")
        return BinaryField8(value: GF8Tables.inv[Int(value)])
    }

    // Square — in GF(2^k), squaring is a linear operation (Frobenius)
    @inline(__always)
    public func squared() -> BinaryField8 {
        self * self
    }

    // Exponentiation via square-and-multiply
    public func pow(_ n: Int) -> BinaryField8 {
        if n == 0 { return .one }
        var result = BinaryField8.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base * base
            exp >>= 1
        }
        return result
    }
}

// Carry-less multiply for GF(2^8) with reduction by 0x11B
@inline(__always)
func gf8Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
    var result: UInt8 = 0
    var aa = a
    var bb = b
    for _ in 0..<8 {
        if bb & 1 != 0 { result ^= aa }
        let carry = aa >> 7
        aa = (aa << 1) ^ (carry &* 0x1B)  // reduce by x^8+x^4+x^3+x+1 (0x11B, low byte = 0x1B)
        bb >>= 1
    }
    return result
}

// MARK: - GF(2^8) Lookup Tables

/// Precomputed tables for GF(2^8) — log/exp for fast multiply, inverse table
public enum GF8Tables {
    // Inverse table: inv[a] = a^(-1) in GF(2^8), inv[0] = 0
    public static let inv: [UInt8] = {
        var table = [UInt8](repeating: 0, count: 256)
        for i in 1..<256 {
            // a^254 = a^(-1) in GF(2^8) since a^255 = 1
            let a = UInt8(i)
            var result: UInt8 = 1
            var base = a
            var exp = 254
            while exp > 0 {
                if exp & 1 == 1 { result = gf8Mul(result, base) }
                base = gf8Mul(base, base)
                exp >>= 1
            }
            table[i] = result
        }
        return table
    }()

    // Log table (base generator=3): log[a] = discrete log of a
    public static let log: [UInt8] = {
        var table = [UInt8](repeating: 0, count: 256)
        var x: UInt8 = 1
        for i in 0..<255 {
            table[Int(x)] = UInt8(i)
            x = gf8Mul(x, 3)  // generator = 3
        }
        return table
    }()

    // Exp table: exp[i] = 3^i in GF(2^8)
    public static let exp: [UInt8] = {
        var table = [UInt8](repeating: 0, count: 512)  // double for wraparound
        var x: UInt8 = 1
        for i in 0..<255 {
            table[i] = x
            table[i + 255] = x  // wraparound
            x = gf8Mul(x, 3)
        }
        return table
    }()

    /// Fast multiply using log/exp tables (avoids bit-serial loop)
    @inline(__always)
    public static func mulViaLog(_ a: UInt8, _ b: UInt8) -> UInt8 {
        if a == 0 || b == 0 { return 0 }
        let logSum = Int(log[Int(a)]) + Int(log[Int(b)])
        return exp[logSum]
    }
}

// MARK: - GF(2^16) — Tower extension over GF(2^8)

/// GF(2^16) = GF(2^8)[X] / (X^2 + X + α)
/// Represented as lo + hi*X where lo, hi in GF(2^8)
/// α = 0x2B chosen so that X^2 + X + 0x2B is irreducible over GF(2^8)
public struct BinaryField16: Equatable, CustomStringConvertible {
    public var lo: BinaryField8  // coefficient of 1
    public var hi: BinaryField8  // coefficient of X

    /// Tower extension parameter: X^2 + X + ALPHA = 0
    public static let ALPHA = BinaryField8(value: 0x2B)

    public static let zero = BinaryField16(lo: .zero, hi: .zero)
    public static let one = BinaryField16(lo: .one, hi: .zero)

    public init(lo: BinaryField8, hi: BinaryField8) {
        self.lo = lo
        self.hi = hi
    }

    public init(value: UInt16) {
        self.lo = BinaryField8(value: UInt8(value & 0xFF))
        self.hi = BinaryField8(value: UInt8(value >> 8))
    }

    public var isZero: Bool { lo.isZero && hi.isZero }

    public var toUInt16: UInt16 {
        UInt16(lo.value) | (UInt16(hi.value) << 8)
    }

    public var description: String { "GF2^16(\(toUInt16))" }

    // Addition = component-wise XOR
    @inline(__always)
    public static func + (a: BinaryField16, b: BinaryField16) -> BinaryField16 {
        BinaryField16(lo: a.lo + b.lo, hi: a.hi + b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryField16, b: BinaryField16) -> BinaryField16 {
        a + b  // char 2
    }

    // Karatsuba multiplication in the tower
    // (a + bX)(c + dX) mod (X^2 + X + α)
    // = ac + (ac + (a+b)(c+d) + bd·α)·X + bd·X^2
    // Since X^2 = X + α:  bd·X^2 = bd·X + bd·α
    // Result: (ac + bd·α) + ((a+b)(c+d) + ac)·X
    //
    // Wait — let's be precise. X^2 + X + α = 0 means X^2 = X + α.
    // (a + bX)(c + dX) = ac + (ad + bc)X + bd·X^2
    //                   = ac + (ad + bc)X + bd(X + α)
    //                   = (ac + bd·α) + (ad + bc + bd)X
    //
    // Karatsuba form: let e = (a+b)(c+d) = ac + ad + bc + bd
    // ad + bc + bd = e + ac  (since ad + bc = e + ac + bd, then +bd = e + ac)
    // Wait: e = ac + ad + bc + bd, so ad + bc = e + ac + bd
    // Then ad + bc + bd = e + ac + bd + bd = e + ac (bd cancels in char 2)
    //
    // So: result = (ac + bd·α) + (e + ac)·X
    // where e = (a+b)(c+d), and we compute ac, bd — 3 multiplications total
    @inline(__always)
    public static func * (a: BinaryField16, b: BinaryField16) -> BinaryField16 {
        let ac = a.lo * b.lo
        let bd = a.hi * b.hi
        let e = (a.lo + a.hi) * (b.lo + b.hi)
        let bdAlpha = bd * BinaryField16.ALPHA
        return BinaryField16(lo: ac + bdAlpha, hi: e + ac)
    }

    public func inverse() -> BinaryField16 {
        precondition(!isZero, "Cannot invert zero")
        // For (a + bX), inverse uses norm:
        // N = a^2 + a·b + b^2·α (the norm from GF(2^16) to GF(2^8))
        // (a + bX)^(-1) = ((a + b) + bX) / N
        // = N^(-1) * ((a + b) + bX)
        let a2 = lo * lo
        let ab = lo * hi
        let b2Alpha = hi * hi * BinaryField16.ALPHA
        let norm = a2 + ab + b2Alpha
        let normInv = norm.inverse()
        return BinaryField16(lo: (lo + hi) * normInv, hi: hi * normInv)
    }

    public func squared() -> BinaryField16 { self * self }

    public func pow(_ n: Int) -> BinaryField16 {
        if n == 0 { return .one }
        var result = BinaryField16.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base * base
            exp >>= 1
        }
        return result
    }
}

// MARK: - GF(2^32) — Tower extension over GF(2^16)

/// GF(2^32) = GF(2^16)[X] / (X^2 + X + β)
/// β chosen so polynomial is irreducible over GF(2^16)
public struct BinaryField32: Equatable, CustomStringConvertible {
    public var lo: BinaryField16
    public var hi: BinaryField16

    /// Tower extension parameter at this level
    public static let BETA = BinaryField16(lo: BinaryField8(value: 0x02), hi: BinaryField8(value: 0x01))

    public static let zero = BinaryField32(lo: .zero, hi: .zero)
    public static let one = BinaryField32(lo: .one, hi: .zero)

    public init(lo: BinaryField16, hi: BinaryField16) {
        self.lo = lo
        self.hi = hi
    }

    public init(value: UInt32) {
        self.lo = BinaryField16(value: UInt16(value & 0xFFFF))
        self.hi = BinaryField16(value: UInt16(value >> 16))
    }

    public var isZero: Bool { lo.isZero && hi.isZero }

    public var toUInt32: UInt32 {
        UInt32(lo.toUInt16) | (UInt32(hi.toUInt16) << 16)
    }

    public var description: String { "GF2^32(\(toUInt32))" }

    @inline(__always)
    public static func + (a: BinaryField32, b: BinaryField32) -> BinaryField32 {
        BinaryField32(lo: a.lo + b.lo, hi: a.hi + b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryField32, b: BinaryField32) -> BinaryField32 {
        a + b
    }

    // Same Karatsuba as GF(2^16) but one level up
    @inline(__always)
    public static func * (a: BinaryField32, b: BinaryField32) -> BinaryField32 {
        let ac = a.lo * b.lo
        let bd = a.hi * b.hi
        let e = (a.lo + a.hi) * (b.lo + b.hi)
        let bdBeta = bd * BinaryField32.BETA
        return BinaryField32(lo: ac + bdBeta, hi: e + ac)
    }

    // Multiply BinaryField16 by the tower constant BETA
    // This is just a regular GF(2^16) multiplication
    @inline(__always)
    static func mulBeta(_ x: BinaryField16) -> BinaryField16 {
        x * BinaryField32.BETA
    }

    public func inverse() -> BinaryField32 {
        precondition(!isZero, "Cannot invert zero")
        let a2 = lo * lo
        let ab = lo * hi
        let b2Beta = hi * hi * BinaryField32.BETA
        let norm = a2 + ab + b2Beta
        let normInv = norm.inverse()
        return BinaryField32(lo: (lo + hi) * normInv, hi: hi * normInv)
    }

    public func squared() -> BinaryField32 { self * self }

    public func pow(_ n: Int) -> BinaryField32 {
        if n == 0 { return .one }
        var result = BinaryField32.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base * base
            exp >>= 1
        }
        return result
    }
}

// MARK: - GF(2^64) — Tower extension over GF(2^32)

/// GF(2^64) = GF(2^32)[X] / (X^2 + X + γ)
public struct BinaryField64: Equatable, CustomStringConvertible {
    public var lo: BinaryField32
    public var hi: BinaryField32

    /// Tower extension parameter
    public static let GAMMA = BinaryField32(value: 0x00000002)

    public static let zero = BinaryField64(lo: .zero, hi: .zero)
    public static let one = BinaryField64(lo: .one, hi: .zero)

    public init(lo: BinaryField32, hi: BinaryField32) {
        self.lo = lo
        self.hi = hi
    }

    public init(value: UInt64) {
        self.lo = BinaryField32(value: UInt32(value & 0xFFFFFFFF))
        self.hi = BinaryField32(value: UInt32(value >> 32))
    }

    public var isZero: Bool { lo.isZero && hi.isZero }

    public var toUInt64: UInt64 {
        UInt64(lo.toUInt32) | (UInt64(hi.toUInt32) << 32)
    }

    public var description: String { "GF2^64(\(toUInt64))" }

    @inline(__always)
    public static func + (a: BinaryField64, b: BinaryField64) -> BinaryField64 {
        BinaryField64(lo: a.lo + b.lo, hi: a.hi + b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryField64, b: BinaryField64) -> BinaryField64 {
        a + b
    }

    @inline(__always)
    public static func * (a: BinaryField64, b: BinaryField64) -> BinaryField64 {
        let ac = a.lo * b.lo
        let bd = a.hi * b.hi
        let e = (a.lo + a.hi) * (b.lo + b.hi)
        let bdGamma = bd * BinaryField64.GAMMA
        return BinaryField64(lo: ac + bdGamma, hi: e + ac)
    }

    public func inverse() -> BinaryField64 {
        precondition(!isZero, "Cannot invert zero")
        let a2 = lo * lo
        let ab = lo * hi
        let b2Gamma = hi * hi * BinaryField64.GAMMA
        let norm = a2 + ab + b2Gamma
        let normInv = norm.inverse()
        return BinaryField64(lo: (lo + hi) * normInv, hi: hi * normInv)
    }

    public func squared() -> BinaryField64 { self * self }

    public func pow(_ n: Int) -> BinaryField64 {
        if n == 0 { return .one }
        var result = BinaryField64.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base * base
            exp >>= 1
        }
        return result
    }
}

// MARK: - GF(2^128) — Tower extension over GF(2^64)

/// GF(2^128) = GF(2^64)[X] / (X^2 + X + δ)
public struct BinaryField128: Equatable, CustomStringConvertible {
    public var lo: BinaryField64
    public var hi: BinaryField64

    /// Tower extension parameter
    public static let DELTA = BinaryField64(value: 0x0000000000000002)

    public static let zero = BinaryField128(lo: .zero, hi: .zero)
    public static let one = BinaryField128(lo: .one, hi: .zero)

    public init(lo: BinaryField64, hi: BinaryField64) {
        self.lo = lo
        self.hi = hi
    }

    public var isZero: Bool { lo.isZero && hi.isZero }

    public var description: String { "GF2^128(\(lo.toUInt64), \(hi.toUInt64))" }

    @inline(__always)
    public static func + (a: BinaryField128, b: BinaryField128) -> BinaryField128 {
        BinaryField128(lo: a.lo + b.lo, hi: a.hi + b.hi)
    }

    @inline(__always)
    public static func - (a: BinaryField128, b: BinaryField128) -> BinaryField128 {
        a + b
    }

    @inline(__always)
    public static func * (a: BinaryField128, b: BinaryField128) -> BinaryField128 {
        let ac = a.lo * b.lo
        let bd = a.hi * b.hi
        let e = (a.lo + a.hi) * (b.lo + b.hi)
        let bdDelta = bd * BinaryField128.DELTA
        return BinaryField128(lo: ac + bdDelta, hi: e + ac)
    }

    public func inverse() -> BinaryField128 {
        precondition(!isZero, "Cannot invert zero")
        let a2 = lo * lo
        let ab = lo * hi
        let b2Delta = hi * hi * BinaryField128.DELTA
        let norm = a2 + ab + b2Delta
        let normInv = norm.inverse()
        return BinaryField128(lo: (lo + hi) * normInv, hi: hi * normInv)
    }

    public func squared() -> BinaryField128 { self * self }

    public func pow(_ n: Int) -> BinaryField128 {
        if n == 0 { return .one }
        var result = BinaryField128.one
        var base = self
        var exp = n
        while exp > 0 {
            if exp & 1 == 1 { result = result * base }
            base = base * base
            exp >>= 1
        }
        return result
    }
}

// MARK: - Free function wrappers (matching project convention)

@inline(__always) public func bf8Add(_ a: BinaryField8, _ b: BinaryField8) -> BinaryField8 { a + b }
@inline(__always) public func bf8Sub(_ a: BinaryField8, _ b: BinaryField8) -> BinaryField8 { a - b }
@inline(__always) public func bf8Mul(_ a: BinaryField8, _ b: BinaryField8) -> BinaryField8 { a * b }
@inline(__always) public func bf8Inv(_ a: BinaryField8) -> BinaryField8 { a.inverse() }

@inline(__always) public func bf16Add(_ a: BinaryField16, _ b: BinaryField16) -> BinaryField16 { a + b }
@inline(__always) public func bf16Sub(_ a: BinaryField16, _ b: BinaryField16) -> BinaryField16 { a - b }
@inline(__always) public func bf16Mul(_ a: BinaryField16, _ b: BinaryField16) -> BinaryField16 { a * b }
@inline(__always) public func bf16Inv(_ a: BinaryField16) -> BinaryField16 { a.inverse() }

@inline(__always) public func bf32Add(_ a: BinaryField32, _ b: BinaryField32) -> BinaryField32 { a + b }
@inline(__always) public func bf32Sub(_ a: BinaryField32, _ b: BinaryField32) -> BinaryField32 { a - b }
@inline(__always) public func bf32Mul(_ a: BinaryField32, _ b: BinaryField32) -> BinaryField32 { a * b }
@inline(__always) public func bf32Inv(_ a: BinaryField32) -> BinaryField32 { a.inverse() }

@inline(__always) public func bf64Add(_ a: BinaryField64, _ b: BinaryField64) -> BinaryField64 { a + b }
@inline(__always) public func bf64Sub(_ a: BinaryField64, _ b: BinaryField64) -> BinaryField64 { a - b }
@inline(__always) public func bf64Mul(_ a: BinaryField64, _ b: BinaryField64) -> BinaryField64 { a * b }
@inline(__always) public func bf64Inv(_ a: BinaryField64) -> BinaryField64 { a.inverse() }

@inline(__always) public func bf128Add(_ a: BinaryField128, _ b: BinaryField128) -> BinaryField128 { a + b }
@inline(__always) public func bf128Sub(_ a: BinaryField128, _ b: BinaryField128) -> BinaryField128 { a - b }
@inline(__always) public func bf128Mul(_ a: BinaryField128, _ b: BinaryField128) -> BinaryField128 { a * b }
@inline(__always) public func bf128Inv(_ a: BinaryField128) -> BinaryField128 { a.inverse() }

// MARK: - NEON/PMULL-Accelerated Binary Tower (C bridge)

import NeonFieldOps

/// NEON-accelerated binary tower fields using ARM64 PMULL carry-less multiply.
/// These call through to C implementations in binary_tower.c for maximum performance.
public enum BinaryTowerNeon {

    /// Initialize GF(2^8) lookup tables. Call once at startup.
    public static func initialize() {
        bt_gf8_init()
    }

    // MARK: - GF(2^64) accelerated ops

    /// GF(2^64) multiply using PMULL (single instruction for carry-less multiply).
    @inline(__always)
    public static func gf64Mul(_ a: UInt64, _ b: UInt64) -> UInt64 {
        bt_gf64_mul(a, b)
    }

    /// GF(2^64) square.
    @inline(__always)
    public static func gf64Sqr(_ a: UInt64) -> UInt64 {
        bt_gf64_sqr(a)
    }

    /// GF(2^64) inverse via Itoh-Tsujii.
    @inline(__always)
    public static func gf64Inv(_ a: UInt64) -> UInt64 {
        bt_gf64_inv(a)
    }

    // MARK: - GF(2^128) flat (AES-GCM polynomial) accelerated ops

    /// GF(2^128) multiply (flat representation, AES-GCM polynomial).
    @inline(__always)
    public static func gf128Mul(_ a: (UInt64, UInt64), _ b: (UInt64, UInt64)) -> (UInt64, UInt64) {
        var aArr: [UInt64] = [a.0, a.1]
        var bArr: [UInt64] = [b.0, b.1]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_mul(&aArr, &bArr, &rArr)
        return (rArr[0], rArr[1])
    }

    /// GF(2^128) inverse (flat representation).
    @inline(__always)
    public static func gf128Inv(_ a: (UInt64, UInt64)) -> (UInt64, UInt64) {
        var aArr: [UInt64] = [a.0, a.1]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_inv(&aArr, &rArr)
        return (rArr[0], rArr[1])
    }

    // MARK: - GF(2^128) tower form accelerated ops

    /// Tower GF(2^128) = GF(2^64)[X]/(X^2+X+2) multiply, using PMULL at the GF(2^64) level.
    @inline(__always)
    public static func tower128Mul(_ a: (UInt64, UInt64), _ b: (UInt64, UInt64)) -> (UInt64, UInt64) {
        var aArr: [UInt64] = [a.0, a.1]
        var bArr: [UInt64] = [b.0, b.1]
        var rArr: [UInt64] = [0, 0]
        bt_tower128_mul(&aArr, &bArr, &rArr)
        return (rArr[0], rArr[1])
    }

    /// Tower GF(2^128) inverse.
    @inline(__always)
    public static func tower128Inv(_ a: (UInt64, UInt64)) -> (UInt64, UInt64) {
        var aArr: [UInt64] = [a.0, a.1]
        var rArr: [UInt64] = [0, 0]
        bt_tower128_inv(&aArr, &rArr)
        return (rArr[0], rArr[1])
    }

    // MARK: - GF(2^32) and GF(2^16) accelerated ops

    /// GF(2^32) multiply using PMULL.
    @inline(__always)
    public static func gf32Mul(_ a: UInt32, _ b: UInt32) -> UInt32 {
        bt_gf32_mul(a, b)
    }

    /// GF(2^16) multiply using PMULL.
    @inline(__always)
    public static func gf16Mul(_ a: UInt16, _ b: UInt16) -> UInt16 {
        bt_gf16_mul(a, b)
    }

    /// GF(2^8) multiply using log/exp table.
    @inline(__always)
    public static func gf8Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        bt_gf8_mul(a, b)
    }

    // MARK: - Batch operations

    /// Batch GF(2^64) multiply: out[i] = a[i] * b[i].
    public static func gf64BatchMul(_ a: UnsafePointer<UInt64>, _ b: UnsafePointer<UInt64>,
                                     _ out: UnsafeMutablePointer<UInt64>, count: Int) {
        bt_gf64_batch_mul(a, b, out, Int32(count))
    }

    /// Batch GF(2^64) add (XOR), NEON-vectorized.
    public static func gf64BatchAdd(_ a: UnsafePointer<UInt64>, _ b: UnsafePointer<UInt64>,
                                     _ out: UnsafeMutablePointer<UInt64>, count: Int) {
        bt_gf64_batch_add(a, b, out, Int32(count))
    }

    /// Batch GF(2^128) multiply: out[i] = a[i] * b[i]. Each element is 2 x UInt64.
    public static func gf128BatchMul(_ a: UnsafePointer<UInt64>, _ b: UnsafePointer<UInt64>,
                                      _ out: UnsafeMutablePointer<UInt64>, count: Int) {
        bt_gf128_batch_mul(a, b, out, Int32(count))
    }

    /// Batch GF(2^128) add (XOR), NEON-vectorized. Each element is 2 x UInt64.
    public static func gf128BatchAdd(_ a: UnsafePointer<UInt64>, _ b: UnsafePointer<UInt64>,
                                      _ out: UnsafeMutablePointer<UInt64>, count: Int) {
        bt_gf128_batch_add(a, b, out, Int32(count))
    }

    /// Batch tower GF(2^128) multiply.
    public static func tower128BatchMul(_ a: UnsafePointer<UInt64>, _ b: UnsafePointer<UInt64>,
                                         _ out: UnsafeMutablePointer<UInt64>, count: Int) {
        bt_tower128_batch_mul(a, b, out, Int32(count))
    }
}
