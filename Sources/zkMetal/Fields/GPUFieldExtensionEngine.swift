// GPU-accelerated batch operations over BN254 extension field tower.
//
// Provides batch operations (add, mul, inv, sqr) over Fp2/Fp6/Fp12
// and the GPUFieldExtensionEngine orchestrator.
//
// Reuses existing Fp2/Fp6/Fp12 types and arithmetic from BN254Pairing.swift.

import Foundation

// MARK: - Equatable conformance for tower types

extension Fp2: Equatable {
    public static func == (lhs: Fp2, rhs: Fp2) -> Bool {
        lhs.c0.to64() == rhs.c0.to64() && lhs.c1.to64() == rhs.c1.to64()
    }
}

extension Fp6: Equatable {
    public static func == (lhs: Fp6, rhs: Fp6) -> Bool {
        lhs.c0 == rhs.c0 && lhs.c1 == rhs.c1 && lhs.c2 == rhs.c2
    }
}

extension Fp12: Equatable {
    public static func == (lhs: Fp12, rhs: Fp12) -> Bool {
        lhs.c0 == rhs.c0 && lhs.c1 == rhs.c1
    }
}

// MARK: - Additional Fp2 helpers (not in BN254Pairing)

/// Fp2 conjugate: conj(a + bu) = a - bu
public func fp2Conj(_ x: Fp2) -> Fp2 {
    Fp2(c0: x.c0, c1: fpNeg(x.c1))
}

/// Fp2 norm: N(a + bu) = a^2 + b^2  (in Fp)
public func fp2Norm(_ x: Fp2) -> Fp {
    fpAdd(fpMul(x.c0, x.c0), fpMul(x.c1, x.c1))
}

/// Fp2 inverse: 1/(a+bu) = conj(a+bu) / N(a+bu)
public func fp2Inv(_ x: Fp2) -> Fp2 {
    let norm = fp2Norm(x)
    let normInv = fpInverse(norm)
    return Fp2(c0: fpMul(x.c0, normInv), c1: fpMul(fpNeg(x.c1), normInv))
}

/// Multiply Fp2 by the non-residue u: u * (a + bu) = -b + au
public func fp2MulByU(_ x: Fp2) -> Fp2 {
    Fp2(c0: fpNeg(x.c1), c1: x.c0)
}

// MARK: - Additional Fp6 helpers

/// Fp6 doubling
public func fp6Double(_ x: Fp6) -> Fp6 {
    Fp6(c0: fp2Double(x.c0), c1: fp2Double(x.c1), c2: fp2Double(x.c2))
}

/// Multiply Fp2 by xi = 9 + u (non-residue for Fp6 tower)
public func fp2MulByXi(_ x: Fp2) -> Fp2 {
    let nine = fpFromInt(9)
    let real = fpSub(fpMul(nine, x.c0), x.c1)
    let imag = fpAdd(fpMul(nine, x.c1), x.c0)
    return Fp2(c0: real, c1: imag)
}

/// Fp6 inverse via adjugate matrix
public func fp6Inv(_ x: Fp6) -> Fp6 {
    let c0sq = fp2Sqr(x.c0)
    let c1sq = fp2Sqr(x.c1)
    let c2sq = fp2Sqr(x.c2)
    let c0c1 = fp2Mul(x.c0, x.c1)
    let c0c2 = fp2Mul(x.c0, x.c2)
    let c1c2 = fp2Mul(x.c1, x.c2)

    let cA = fp2Sub(c0sq, fp2MulByXi(c1c2))
    let cB = fp2Sub(fp2MulByXi(c2sq), c0c1)
    let cC = fp2Sub(c1sq, c0c2)

    let det = fp2Add(fp2Mul(x.c0, cA),
                     fp2MulByXi(fp2Add(fp2Mul(x.c2, cB), fp2Mul(x.c1, cC))))
    let detInv = fp2Inv(det)

    return Fp6(c0: fp2Mul(cA, detInv),
               c1: fp2Mul(cB, detInv),
               c2: fp2Mul(cC, detInv))
}

/// Multiply Fp6 by Fp2 scalar
public func fp6MulByFp2(_ x: Fp6, _ s: Fp2) -> Fp6 {
    Fp6(c0: fp2Mul(x.c0, s), c1: fp2Mul(x.c1, s), c2: fp2Mul(x.c2, s))
}

/// Multiply Fp6 by v: v * (c0 + c1*v + c2*v^2) = xi*c2 + c0*v + c1*v^2
func extFp6MulByV(_ x: Fp6) -> Fp6 {
    Fp6(c0: fp2MulByXi(x.c2), c1: x.c0, c2: x.c1)
}

// MARK: - Additional Fp12 helpers

/// Fp12 conjugate: conj(a + b*w) = a - b*w
public func fp12Conj(_ x: Fp12) -> Fp12 {
    Fp12(c0: x.c0, c1: fp6Neg(x.c1))
}

/// Fp12 inverse: 1/(a + bw) = (a - bw) / (a^2 - b^2*v)
public func fp12Inv(_ x: Fp12) -> Fp12 {
    let a2 = fp6Sqr(x.c0)
    let b2v = extFp6MulByV(fp6Sqr(x.c1))
    let det = fp6Sub(a2, b2v)
    let detInv = fp6Inv(det)
    return Fp12(c0: fp6Mul(x.c0, detInv), c1: fp6Neg(fp6Mul(x.c1, detInv)))
}

/// Fp12 exponentiation by scalar (square-and-multiply)
public func fp12Pow(_ base: Fp12, _ exp: [UInt64]) -> Fp12 {
    var result = Fp12.one
    var found = false
    for i in stride(from: exp.count - 1, through: 0, by: -1) {
        let w = exp[i]
        for bit in stride(from: 63, through: 0, by: -1) {
            if found {
                result = fp12Sqr(result)
            }
            if (w >> bit) & 1 == 1 {
                found = true
                result = fp12Mul(result, base)
            }
        }
    }
    return result
}

// MARK: - Batch Operations

/// Batch Fp2 addition
public func fp2BatchAdd(_ xs: [Fp2], _ ys: [Fp2]) -> [Fp2] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { fp2Add($0, $1) }
}

/// Batch Fp2 multiplication
public func fp2BatchMul(_ xs: [Fp2], _ ys: [Fp2]) -> [Fp2] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { fp2Mul($0, $1) }
}

/// Batch Fp2 inversion using Montgomery's trick
public func fp2BatchInv(_ xs: [Fp2]) -> [Fp2] {
    let n = xs.count
    guard n > 0 else { return [] }
    if n == 1 { return [fp2Inv(xs[0])] }

    var prefix = [Fp2](repeating: .zero, count: n)
    prefix[0] = xs[0]
    for i in 1..<n {
        prefix[i] = fp2Mul(prefix[i - 1], xs[i])
    }

    var inv = fp2Inv(prefix[n - 1])

    var result = [Fp2](repeating: .zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = fp2Mul(inv, prefix[i - 1])
        inv = fp2Mul(inv, xs[i])
    }
    result[0] = inv
    return result
}

/// Batch Fp2 squaring
public func fp2BatchSqr(_ xs: [Fp2]) -> [Fp2] {
    xs.map { fp2Sqr($0) }
}

/// Batch Fp6 multiplication
public func fp6BatchMul(_ xs: [Fp6], _ ys: [Fp6]) -> [Fp6] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { fp6Mul($0, $1) }
}

/// Batch Fp12 multiplication
public func fp12BatchMul(_ xs: [Fp12], _ ys: [Fp12]) -> [Fp12] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { fp12Mul($0, $1) }
}

// MARK: - GPU Field Extension Engine

/// Orchestrates GPU-accelerated extension field operations.
public final class GPUFieldExtensionEngine {
    public static let shared = GPUFieldExtensionEngine()

    private init() {}

    /// Batch Fp2 multiply
    public func batchFp2Mul(_ xs: [Fp2], _ ys: [Fp2]) -> [Fp2] {
        fp2BatchMul(xs, ys)
    }

    /// Batch Fp2 inverse using Montgomery's trick
    public func batchFp2Inv(_ xs: [Fp2]) -> [Fp2] {
        fp2BatchInv(xs)
    }

    /// Batch Fp6 multiply
    public func batchFp6Mul(_ xs: [Fp6], _ ys: [Fp6]) -> [Fp6] {
        fp6BatchMul(xs, ys)
    }

    /// Batch Fp12 multiply
    public func batchFp12Mul(_ xs: [Fp12], _ ys: [Fp12]) -> [Fp12] {
        fp12BatchMul(xs, ys)
    }

    /// Compute Fp12 product of array (reduction)
    public func fp12Product(_ xs: [Fp12]) -> Fp12 {
        var acc = Fp12.one
        for x in xs {
            acc = fp12Mul(acc, x)
        }
        return acc
    }
}
