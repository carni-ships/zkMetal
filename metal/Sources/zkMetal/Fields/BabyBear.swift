// BabyBear field arithmetic (CPU-side)
// p = 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
// Used by SP1 (Succinct), RISC Zero, Plonky3
// TWO_ADICITY = 27 (p - 1 = 2^27 * 15)

import Foundation

public struct Bb: Equatable {
    public var v: UInt32

    public static let P: UInt32 = 0x78000001  // 2013265921
    public static let TWO_ADICITY: Int = 27

    // Primitive 2^27-th root of unity: 3^((p-1)/2^27) mod p
    public static let ROOT_OF_UNITY: UInt32 = 440564289

    // Multiplicative generator
    public static let GENERATOR: UInt32 = 31

    public static var zero: Bb { Bb(v: 0) }
    public static var one: Bb { Bb(v: 1) }

    public init(v: UInt32) {
        self.v = v
    }

    public var isZero: Bool { v == 0 }
}

public func bbAdd(_ a: Bb, _ b: Bb) -> Bb {
    let sum = UInt64(a.v) + UInt64(b.v)
    return Bb(v: UInt32(sum >= UInt64(Bb.P) ? sum - UInt64(Bb.P) : sum))
}

public func bbSub(_ a: Bb, _ b: Bb) -> Bb {
    if a.v >= b.v { return Bb(v: a.v - b.v) }
    return Bb(v: a.v &+ Bb.P &- b.v)
}

public func bbNeg(_ a: Bb) -> Bb {
    if a.v == 0 { return a }
    return Bb(v: Bb.P - a.v)
}

public func bbMul(_ a: Bb, _ b: Bb) -> Bb {
    let prod = UInt64(a.v) * UInt64(b.v)
    return Bb(v: UInt32(prod % UInt64(Bb.P)))
}

public func bbSqr(_ a: Bb) -> Bb { bbMul(a, a) }

public func bbPow(_ base: Bb, _ exp: UInt32) -> Bb {
    if exp == 0 { return Bb.one }
    var result = Bb.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = bbMul(result, b) }
        b = bbSqr(b)
        e >>= 1
    }
    return result
}

public func bbInverse(_ a: Bb) -> Bb {
    return bbPow(a, Bb.P - 2)
}

/// Get primitive 2^k-th root of unity for BabyBear.
public func bbRootOfUnity(logN: Int) -> Bb {
    precondition(logN <= Bb.TWO_ADICITY)
    var omega = Bb(v: Bb.ROOT_OF_UNITY)
    for _ in 0..<(Bb.TWO_ADICITY - logN) {
        omega = bbSqr(omega)
    }
    return omega
}
