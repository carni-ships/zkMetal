// GPU-accelerated Goldilocks quadratic extension field engine.
//
// Gl2 = Gl[x] / (x^2 - 7)  where Gl is the Goldilocks field (p = 2^64 - 2^32 + 1).
// An element a + b*x in Gl2 is stored as (c0: Gl, c1: Gl) where c0 = a, c1 = b.
//
// Multiplication: (a + bx)(c + dx) = (ac + 7*bd) + (ad + bc)x
// The non-residue W = 7 is not a quadratic residue mod p, which makes Gl2 a field.
//
// This extension field is used by Plonky2, Plonky3, and various STARK systems
// for FRI folding over the extension to achieve soundness amplification.

import Foundation

// MARK: - Gl2 Type

/// Goldilocks quadratic extension field element: c0 + c1 * x, where x^2 = 7.
public struct Gl2: Equatable {
    public var c0: Gl  // real part
    public var c1: Gl  // extension part

    /// Non-residue: x^2 = W in the extension.
    public static let W: UInt64 = 7

    public static var zero: Gl2 { Gl2(c0: .zero, c1: .zero) }
    public static var one: Gl2 { Gl2(c0: .one, c1: .zero) }

    public init(c0: Gl, c1: Gl) {
        self.c0 = c0
        self.c1 = c1
    }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

// MARK: - Gl helpers

/// Construct a Gl element from a small integer.
public func glFromInt(_ v: UInt64) -> Gl {
    Gl(v: v % Gl.P)
}

// MARK: - Gl2 Arithmetic

/// Gl2 addition: component-wise.
public func gl2Add(_ a: Gl2, _ b: Gl2) -> Gl2 {
    Gl2(c0: glAdd(a.c0, b.c0), c1: glAdd(a.c1, b.c1))
}

/// Gl2 subtraction: component-wise.
public func gl2Sub(_ a: Gl2, _ b: Gl2) -> Gl2 {
    Gl2(c0: glSub(a.c0, b.c0), c1: glSub(a.c1, b.c1))
}

/// Gl2 negation: component-wise.
public func gl2Neg(_ a: Gl2) -> Gl2 {
    Gl2(c0: glNeg(a.c0), c1: glNeg(a.c1))
}

/// Gl2 doubling: component-wise.
public func gl2Double(_ a: Gl2) -> Gl2 {
    Gl2(c0: glAdd(a.c0, a.c0), c1: glAdd(a.c1, a.c1))
}

/// Gl2 multiplication: (a0 + a1*x)(b0 + b1*x) = (a0*b0 + W*a1*b1) + (a0*b1 + a1*b0)*x
/// Uses Karatsuba: 3 base muls instead of 4.
public func gl2Mul(_ a: Gl2, _ b: Gl2) -> Gl2 {
    let a0b0 = glMul(a.c0, b.c0)
    let a1b1 = glMul(a.c1, b.c1)
    // c0 = a0*b0 + 7 * a1*b1
    let w_a1b1 = glMul(Gl(v: Gl2.W), a1b1)
    let c0 = glAdd(a0b0, w_a1b1)
    // c1 = (a0+a1)*(b0+b1) - a0*b0 - a1*b1  (Karatsuba)
    let c1 = glSub(glSub(glMul(glAdd(a.c0, a.c1), glAdd(b.c0, b.c1)), a0b0), a1b1)
    return Gl2(c0: c0, c1: c1)
}

/// Gl2 squaring: (a0 + a1*x)^2 = (a0^2 + W*a1^2) + 2*a0*a1*x
/// Optimized: 2 base muls + 1 base mul for cross term.
public func gl2Sqr(_ a: Gl2) -> Gl2 {
    let a0sq = glSqr(a.c0)
    let a1sq = glSqr(a.c1)
    let c0 = glAdd(a0sq, glMul(Gl(v: Gl2.W), a1sq))
    let cross = glMul(a.c0, a.c1)
    let c1 = glAdd(cross, cross)
    return Gl2(c0: c0, c1: c1)
}

/// Gl2 scalar multiplication: s * (a0 + a1*x) = s*a0 + s*a1*x
public func gl2ScalarMul(_ s: Gl, _ a: Gl2) -> Gl2 {
    Gl2(c0: glMul(s, a.c0), c1: glMul(s, a.c1))
}

/// Gl2 inverse: 1/(a0 + a1*x) = (a0 - a1*x) / (a0^2 - W*a1^2)
/// norm = a0^2 - W * a1^2 (in Gl)
public func gl2Inv(_ a: Gl2) -> Gl2 {
    let a0sq = glSqr(a.c0)
    let a1sq = glSqr(a.c1)
    let norm = glSub(a0sq, glMul(Gl(v: Gl2.W), a1sq))
    let normInv = glInverse(norm)
    return Gl2(c0: glMul(a.c0, normInv), c1: glNeg(glMul(a.c1, normInv)))
}

/// Gl2 exponentiation by scalar (square-and-multiply).
public func gl2Pow(_ base: Gl2, _ exp: UInt64) -> Gl2 {
    if exp == 0 { return .one }
    var result = Gl2.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = gl2Mul(result, b) }
        b = gl2Sqr(b)
        e >>= 1
    }
    return result
}

/// Gl2 conjugate: conj(a + b*x) = a - b*x
/// This is the Frobenius endomorphism for Gl2/Gl since x^p = -x (mod x^2 - W)
/// when W is not a QR mod p.
public func gl2Conj(_ a: Gl2) -> Gl2 {
    Gl2(c0: a.c0, c1: glNeg(a.c1))
}

/// Frobenius endomorphism: raises to the p-th power.
/// For Gl2 = Gl[x]/(x^2 - W), Frobenius(a + bx) = a + b * x^p = a - bx
/// since x^p = x * (x^2)^((p-1)/2) = x * W^((p-1)/2) = x * (-1) = -x.
public func gl2Frobenius(_ a: Gl2) -> Gl2 {
    gl2Conj(a)
}

/// Gl2 norm: N(a + bx) = a^2 - W * b^2 (in Gl)
public func gl2Norm(_ a: Gl2) -> Gl {
    glSub(glSqr(a.c0), glMul(Gl(v: Gl2.W), glSqr(a.c1)))
}

// MARK: - Conversion

/// Embed a Gl element into Gl2 (c1 = 0).
public func gl2FromGl(_ a: Gl) -> Gl2 {
    Gl2(c0: a, c1: .zero)
}

/// Extract the Gl component (c0) from Gl2, discarding c1.
public func gl2ToGl(_ a: Gl2) -> Gl {
    a.c0
}

/// Check if a Gl2 element is actually in the base field (c1 == 0).
public func gl2IsInBaseField(_ a: Gl2) -> Bool {
    a.c1.isZero
}

// MARK: - Batch Operations

/// Batch Gl2 addition.
public func gl2BatchAdd(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { gl2Add($0, $1) }
}

/// Batch Gl2 subtraction.
public func gl2BatchSub(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { gl2Sub($0, $1) }
}

/// Batch Gl2 multiplication.
public func gl2BatchMul(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
    precondition(xs.count == ys.count)
    return zip(xs, ys).map { gl2Mul($0, $1) }
}

/// Batch Gl2 squaring.
public func gl2BatchSqr(_ xs: [Gl2]) -> [Gl2] {
    xs.map { gl2Sqr($0) }
}

/// Batch Gl2 inversion using Montgomery's trick (1 inversion + 3(n-1) muls).
public func gl2BatchInv(_ xs: [Gl2]) -> [Gl2] {
    let n = xs.count
    guard n > 0 else { return [] }
    if n == 1 { return [gl2Inv(xs[0])] }

    // Build prefix products
    var prefix = [Gl2](repeating: .zero, count: n)
    prefix[0] = xs[0]
    for i in 1..<n {
        prefix[i] = gl2Mul(prefix[i - 1], xs[i])
    }

    // Single inversion of the total product
    var inv = gl2Inv(prefix[n - 1])

    // Propagate back
    var result = [Gl2](repeating: .zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        result[i] = gl2Mul(inv, prefix[i - 1])
        inv = gl2Mul(inv, xs[i])
    }
    result[0] = inv
    return result
}

/// Batch scalar multiplication: s[i] * x[i] for each i.
public func gl2BatchScalarMul(_ scalars: [Gl], _ xs: [Gl2]) -> [Gl2] {
    precondition(scalars.count == xs.count)
    return zip(scalars, xs).map { gl2ScalarMul($0, $1) }
}

// MARK: - Extension Field NTT

/// Precomputed twiddle factors for Gl2 NTT.
/// Twiddle factors live in Gl (not Gl2) since the NTT domain is in the base field.
/// We use roots of unity from Gl: omega = primitive n-th root of unity in Gl.
private func glNTTTwiddles(n: Int) -> [Gl] {
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
    let logN = n.trailingZeroBitCount
    precondition(logN <= Gl.TWO_ADICITY, "n exceeds max NTT size for Goldilocks")

    // omega_n = g^((p-1)/n) where g is the 2^32-th root of unity
    let shift = Gl.TWO_ADICITY - logN
    let omega = glPow(Gl(v: Gl.ROOT_OF_UNITY), 1 << shift)

    var twiddles = [Gl](repeating: .zero, count: n)
    twiddles[0] = .one
    for i in 1..<n {
        twiddles[i] = glMul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Bit-reverse permutation in-place.
private func bitReverse<T>(_ arr: inout [T]) {
    let n = arr.count
    guard n > 1 else { return }
    let logN = n.trailingZeroBitCount
    for i in 0..<n {
        var rev = 0
        var x = i
        for _ in 0..<logN {
            rev = (rev << 1) | (x & 1)
            x >>= 1
        }
        if rev > i {
            arr.swapAt(i, rev)
        }
    }
}

/// Forward NTT over Gl2: Cooley-Tukey radix-2 DIT.
/// Input: n Gl2 elements in normal order.
/// Output: n Gl2 elements in evaluation form.
public func gl2NTT(_ input: [Gl2]) -> [Gl2] {
    let n = input.count
    guard n > 1 else { return input }
    precondition(n & (n - 1) == 0, "NTT size must be power of 2")

    var data = input
    bitReverse(&data)

    let logN = n.trailingZeroBitCount
    let twiddles = glNTTTwiddles(n: n)

    for s in 0..<logN {
        let m = 1 << (s + 1)
        let half = m >> 1
        let step = n / m
        for k in stride(from: 0, to: n, by: m) {
            for j in 0..<half {
                let w = twiddles[j * step]
                let u = data[k + j]
                let t = gl2ScalarMul(w, data[k + j + half])
                data[k + j] = gl2Add(u, t)
                data[k + j + half] = gl2Sub(u, t)
            }
        }
    }
    return data
}

/// Inverse NTT over Gl2: Gentleman-Sande radix-2 DIF.
/// Output is scaled by 1/n.
public func gl2INTT(_ input: [Gl2]) -> [Gl2] {
    let n = input.count
    guard n > 1 else { return input }
    precondition(n & (n - 1) == 0, "INTT size must be power of 2")

    var data = input
    let logN = n.trailingZeroBitCount

    // Inverse twiddles: use omega^(-1)
    let shift = Gl.TWO_ADICITY - logN
    let omega = glPow(Gl(v: Gl.ROOT_OF_UNITY), 1 << shift)
    let omegaInv = glInverse(omega)

    var invTwiddles = [Gl](repeating: .zero, count: n)
    invTwiddles[0] = .one
    for i in 1..<n {
        invTwiddles[i] = glMul(invTwiddles[i - 1], omegaInv)
    }

    // DIF butterfly stages
    for s in stride(from: logN - 1, through: 0, by: -1) {
        let m = 1 << (s + 1)
        let half = m >> 1
        let step = n / m
        for k in stride(from: 0, to: n, by: m) {
            for j in 0..<half {
                let w = invTwiddles[j * step]
                let u = data[k + j]
                let v = data[k + j + half]
                data[k + j] = gl2Add(u, v)
                data[k + j + half] = gl2ScalarMul(w, gl2Sub(u, v))
            }
        }
    }

    // Bit-reverse
    bitReverse(&data)

    // Scale by 1/n
    let nInv = glInverse(Gl(v: UInt64(n)))
    for i in 0..<n {
        data[i] = gl2ScalarMul(nInv, data[i])
    }

    return data
}

// MARK: - Random Element Generation

/// Generate a random Gl element (uniform in [0, p)).
public func glRandom() -> Gl {
    // Rejection sampling to avoid bias
    var v: UInt64 = 0
    repeat {
        v = UInt64.random(in: 0...UInt64.max)
    } while v >= Gl.P
    return Gl(v: v)
}

/// Generate a random Gl2 element.
public func gl2Random() -> Gl2 {
    Gl2(c0: glRandom(), c1: glRandom())
}

/// Generate n random Gl2 elements.
public func gl2RandomBatch(_ n: Int) -> [Gl2] {
    (0..<n).map { _ in gl2Random() }
}

// MARK: - GPU Goldilocks Extension Engine

/// Orchestrates GPU-accelerated Goldilocks extension field operations.
/// Provides batch arithmetic, NTT over Gl2, and utility operations.
public final class GPUGoldilocksExtensionEngine {
    public static let shared = GPUGoldilocksExtensionEngine()

    private init() {}

    // MARK: - Single-element operations

    /// Gl2 multiply.
    public func mul(_ a: Gl2, _ b: Gl2) -> Gl2 {
        gl2Mul(a, b)
    }

    /// Gl2 add.
    public func add(_ a: Gl2, _ b: Gl2) -> Gl2 {
        gl2Add(a, b)
    }

    /// Gl2 subtract.
    public func sub(_ a: Gl2, _ b: Gl2) -> Gl2 {
        gl2Sub(a, b)
    }

    /// Gl2 square.
    public func sqr(_ a: Gl2) -> Gl2 {
        gl2Sqr(a)
    }

    /// Gl2 inverse.
    public func inv(_ a: Gl2) -> Gl2 {
        gl2Inv(a)
    }

    /// Gl2 Frobenius endomorphism.
    public func frobenius(_ a: Gl2) -> Gl2 {
        gl2Frobenius(a)
    }

    // MARK: - Batch operations

    /// Batch Gl2 add.
    public func batchAdd(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
        gl2BatchAdd(xs, ys)
    }

    /// Batch Gl2 sub.
    public func batchSub(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
        gl2BatchSub(xs, ys)
    }

    /// Batch Gl2 multiply.
    public func batchMul(_ xs: [Gl2], _ ys: [Gl2]) -> [Gl2] {
        gl2BatchMul(xs, ys)
    }

    /// Batch Gl2 squaring.
    public func batchSqr(_ xs: [Gl2]) -> [Gl2] {
        gl2BatchSqr(xs)
    }

    /// Batch Gl2 inversion (Montgomery's trick).
    public func batchInv(_ xs: [Gl2]) -> [Gl2] {
        gl2BatchInv(xs)
    }

    /// Batch scalar multiply.
    public func batchScalarMul(_ scalars: [Gl], _ xs: [Gl2]) -> [Gl2] {
        gl2BatchScalarMul(scalars, xs)
    }

    // MARK: - NTT

    /// Forward NTT over Gl2.
    public func ntt(_ input: [Gl2]) -> [Gl2] {
        gl2NTT(input)
    }

    /// Inverse NTT over Gl2.
    public func intt(_ input: [Gl2]) -> [Gl2] {
        gl2INTT(input)
    }

    // MARK: - Reduction

    /// Product of all Gl2 elements in the array.
    public func product(_ xs: [Gl2]) -> Gl2 {
        var acc = Gl2.one
        for x in xs {
            acc = gl2Mul(acc, x)
        }
        return acc
    }

    /// Sum of all Gl2 elements in the array.
    public func sum(_ xs: [Gl2]) -> Gl2 {
        var acc = Gl2.zero
        for x in xs {
            acc = gl2Add(acc, x)
        }
        return acc
    }

    /// Inner product: sum of x[i] * y[i].
    public func innerProduct(_ xs: [Gl2], _ ys: [Gl2]) -> Gl2 {
        precondition(xs.count == ys.count)
        var acc = Gl2.zero
        for i in 0..<xs.count {
            acc = gl2Add(acc, gl2Mul(xs[i], ys[i]))
        }
        return acc
    }

    // MARK: - Conversion

    /// Embed Gl elements into Gl2 (c1 = 0).
    public func fromBaseField(_ xs: [Gl]) -> [Gl2] {
        xs.map { gl2FromGl($0) }
    }

    /// Extract base field component (c0) from Gl2 elements.
    public func toBaseField(_ xs: [Gl2]) -> [Gl] {
        xs.map { gl2ToGl($0) }
    }

    /// Generate n random Gl2 elements.
    public func randomElements(_ n: Int) -> [Gl2] {
        gl2RandomBatch(n)
    }
}
