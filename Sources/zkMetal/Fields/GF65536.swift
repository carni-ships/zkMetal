// GF(2^16) = GF(2)[x] / (x^16 + x^12 + x^3 + x + 1)
// Primitive polynomial: 0x1100B (g=2 generates full multiplicative group of order 65535)
// Used for Reed-Solomon erasure coding over binary fields.
// Multiplication via log/antilog tables (each 64K entries, fast lookup).

import Foundation

public struct GF16: Equatable {
    public var value: UInt16

    public static let zero = GF16(value: 0)
    public static let one = GF16(value: 1)

    /// Primitive polynomial: x^16 + x^12 + x^3 + x + 1
    /// g=2 has order 65535 (generates full multiplicative group)
    static let POLY: UInt32 = 0x1100B

    public init(value: UInt16) {
        self.value = value
    }

    public var isZero: Bool { value == 0 }
}

// MARK: - Log/Antilog Tables

/// GF(2^16) has 65535 nonzero elements. The multiplicative group is cyclic of order 65535.
/// We use generator g = 2 (x) since x is a generator for our irreducible polynomial.
/// log[a] = i such that g^i = a (for a != 0)
/// antilog[i] = g^i mod poly

/// Shared tables (computed once, lazily)
public let gf16Tables: GF16Tables = GF16Tables()

public class GF16Tables {
    public let logTable: [UInt16]      // 65536 entries, logTable[0] unused
    public let antilogTable: [UInt16]  // 131070 entries (doubled for mod-free sum)

    init() {
        var log = [UInt16](repeating: 0, count: 65536)
        var alog = [UInt16](repeating: 0, count: 131070) // 2 * 65535

        // Build antilog table: alog[i] = g^i where g = 2 (primitive element)
        var val: UInt32 = 1
        for i in 0..<65535 {
            alog[i] = UInt16(val)
            log[Int(val)] = UInt16(i)

            // Multiply by generator (x): shift left by 1, reduce if needed
            val <<= 1
            if val & 0x10000 != 0 {
                val ^= GF16.POLY
            }
        }

        // Double the antilog table so alog[i + j] works without mod for i,j < 65535
        for i in 0..<65535 {
            alog[i + 65535] = alog[i]
        }

        self.logTable = log
        self.antilogTable = alog
    }
}

// MARK: - Arithmetic

public func gf16Add(_ a: GF16, _ b: GF16) -> GF16 {
    GF16(value: a.value ^ b.value)  // XOR in GF(2^k)
}

public func gf16Sub(_ a: GF16, _ b: GF16) -> GF16 {
    GF16(value: a.value ^ b.value)  // Same as add in characteristic 2
}

public func gf16Mul(_ a: GF16, _ b: GF16) -> GF16 {
    if a.value == 0 || b.value == 0 { return .zero }
    let tables = gf16Tables
    let sum = Int(tables.logTable[Int(a.value)]) + Int(tables.logTable[Int(b.value)])
    return GF16(value: tables.antilogTable[sum])
}

public func gf16Div(_ a: GF16, _ b: GF16) -> GF16 {
    precondition(b.value != 0, "Division by zero in GF(2^16)")
    if a.value == 0 { return .zero }
    let tables = gf16Tables
    let logA = Int(tables.logTable[Int(a.value)])
    let logB = Int(tables.logTable[Int(b.value)])
    var diff = logA - logB
    if diff < 0 { diff += 65535 }
    return GF16(value: tables.antilogTable[diff])
}

public func gf16Inverse(_ a: GF16) -> GF16 {
    precondition(a.value != 0, "Inverse of zero in GF(2^16)")
    let tables = gf16Tables
    let logA = Int(tables.logTable[Int(a.value)])
    return GF16(value: tables.antilogTable[65535 - logA])
}

public func gf16Pow(_ base: GF16, _ exp: UInt32) -> GF16 {
    if exp == 0 { return .one }
    if base.value == 0 { return .zero }
    let tables = gf16Tables
    let logBase = Int(tables.logTable[Int(base.value)])
    let logResult = Int(UInt64(logBase) * UInt64(exp) % 65535)
    return GF16(value: tables.antilogTable[logResult])
}

// MARK: - Evaluation points for RS coding

/// Generate n distinct evaluation points in GF(2^16).
/// Uses successive powers of the generator: g^0, g^1, ..., g^(n-1)
public func gf16EvalPoints(_ n: Int) -> [GF16] {
    precondition(n <= 65535, "GF(2^16) has at most 65535 nonzero elements")
    let tables = gf16Tables
    return (0..<n).map { GF16(value: tables.antilogTable[$0]) }
}
