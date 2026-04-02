// Goldilocks field arithmetic (CPU-side)
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// Used by Plonky2, Plonky3, and various STARK systems.
// TWO_ADICITY = 32 (p - 1 = 2^32 * (2^32 - 1))

import Foundation

public struct Gl: Equatable {
    public var v: UInt64

    public static let P: UInt64 = 0xFFFFFFFF00000001
    public static let EPS: UInt64 = 0xFFFFFFFF  // 2^32 - 1
    public static let TWO_ADICITY: Int = 32

    // Primitive 2^32-th root of unity: 7^((p-1)/2^32) mod p
    // = 1753635133440165772 (verified)
    public static let ROOT_OF_UNITY: UInt64 = 1753635133440165772

    // Multiplicative generator
    public static let GENERATOR: UInt64 = 7

    public static var zero: Gl { Gl(v: 0) }
    public static var one: Gl { Gl(v: 1) }

    public init(v: UInt64) {
        self.v = v
    }

    public var isZero: Bool { v == 0 }
}

public func glAdd(_ a: Gl, _ b: Gl) -> Gl {
    let (sum, carry) = a.v.addingReportingOverflow(b.v)
    var r = sum
    if carry {
        r = r &+ Gl.EPS
        if r < Gl.EPS { r = r &+ Gl.EPS }
    }
    return Gl(v: r >= Gl.P ? r &- Gl.P : r)
}

public func glSub(_ a: Gl, _ b: Gl) -> Gl {
    if a.v >= b.v { return Gl(v: a.v - b.v) }
    return Gl(v: a.v &+ Gl.P &- b.v)
}

public func glNeg(_ a: Gl) -> Gl {
    if a.v == 0 { return a }
    return Gl(v: Gl.P - a.v)
}

public func glMul(_ a: Gl, _ b: Gl) -> Gl {
    let (hi, lo) = a.v.multipliedFullWidth(by: b.v)
    return glReduce128(hi: hi, lo: lo)
}

/// Reduce a 128-bit value (hi, lo) modulo p = 2^64 - 2^32 + 1
/// Uses: val = hi*2^64 + lo ≡ lo + hi_lo*eps - hi_hi (mod p)
/// where eps = 2^32 - 1, hi = hi_hi*2^32 + hi_lo
private func glReduce128(hi: UInt64, lo: UInt64) -> Gl {
    let hiLo = hi & 0xFFFFFFFF
    let hiHi = hi >> 32

    // Step 1: t = lo + hiLo * eps (may overflow u64 by at most one eps)
    let hiLoEps = hiLo &* Gl.EPS  // fits in u64
    let (t1, c1) = lo.addingReportingOverflow(hiLoEps)

    // Step 2: t = t1 - hiHi (may underflow)
    let (t2, b2) = t1.subtractingReportingOverflow(hiHi)

    // Adjust: if c1 (overflow), add eps; if b2 (underflow), add p
    var r = t2
    if c1 {
        r = r &+ Gl.EPS  // overflow: 2^64 ≡ eps mod p
    }
    if b2 {
        r = r &+ Gl.P  // underflow: add p
    }
    // Final: might still be >= p
    if r >= Gl.P { r = r &- Gl.P }
    return Gl(v: r)
}

public func glSqr(_ a: Gl) -> Gl { glMul(a, a) }

public func glPow(_ base: Gl, _ exp: UInt64) -> Gl {
    if exp == 0 { return Gl.one }
    var result = Gl.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = glMul(result, b) }
        b = glSqr(b)
        e >>= 1
    }
    return result
}

public func glInverse(_ a: Gl) -> Gl {
    // a^(p-2) mod p
    return glPow(a, Gl.P - 2)
}
