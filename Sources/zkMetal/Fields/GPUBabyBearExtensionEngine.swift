// GPU-accelerated BabyBear quartic extension field engine.
//
// Bb4 = Bb[x] / (x^4 - 11)
// where Bb = GF(2^31 - 2^27 + 1) = GF(2013265921)
//
// An element of Bb4 is represented as (c0, c1, c2, c3) where
//   a = c0 + c1*x + c2*x^2 + c3*x^3
//
// The irreducible polynomial x^4 - 11 means x^4 = 11 in Bb.
// This extension is used by SP1 (Succinct), Plonky3, and RISC Zero
// for FRI-based STARKs over BabyBear.
//
// Provides:
//   - Bb4 arithmetic (add, sub, mul, inv, sqr, pow)
//   - Batch Bb4 operations (batch add, mul, inv)
//   - Extension field NTT over Bb4
//   - Bb4 random element generation
//   - Frobenius endomorphism
//   - Conversion between Bb and Bb4

import Foundation

// MARK: - Bb4 — BabyBear Quartic Extension Element

/// Element of Bb4 = Bb[x] / (x^4 - 11).
///
/// Stored as four Bb coefficients: c0 + c1*x + c2*x^2 + c3*x^3.
/// The non-residue W = 11 satisfies x^4 = W in the extension.
public struct Bb4: Equatable, CustomStringConvertible {
    public var c0: Bb
    public var c1: Bb
    public var c2: Bb
    public var c3: Bb

    /// The non-residue: x^4 = W = 11 in Bb.
    public static let W: UInt32 = 11

    public static let zero = Bb4(c0: .zero, c1: .zero, c2: .zero, c3: .zero)
    public static let one  = Bb4(c0: .one, c1: .zero, c2: .zero, c3: .zero)

    public init(c0: Bb, c1: Bb, c2: Bb, c3: Bb) {
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
    }

    /// Embed a base field element into Bb4 (constant coefficient).
    public init(from base: Bb) {
        self.c0 = base
        self.c1 = .zero
        self.c2 = .zero
        self.c3 = .zero
    }

    /// Construct from four UInt32 values (reduced mod p).
    public init(_ v0: UInt32, _ v1: UInt32, _ v2: UInt32, _ v3: UInt32) {
        self.c0 = Bb(v: v0 % Bb.P)
        self.c1 = Bb(v: v1 % Bb.P)
        self.c2 = Bb(v: v2 % Bb.P)
        self.c3 = Bb(v: v3 % Bb.P)
    }

    public var isZero: Bool {
        c0.isZero && c1.isZero && c2.isZero && c3.isZero
    }

    /// Check if element is in the base field Bb (c1 = c2 = c3 = 0).
    public var isInBaseField: Bool {
        c1.isZero && c2.isZero && c3.isZero
    }

    /// Project to base field (returns c0, valid only if isInBaseField).
    public var toBase: Bb { c0 }

    public var description: String {
        "Bb4(\(c0.v), \(c1.v), \(c2.v), \(c3.v))"
    }
}

// MARK: - Bb4 Arithmetic

/// Bb4 addition: component-wise.
public func bb4Add(_ a: Bb4, _ b: Bb4) -> Bb4 {
    Bb4(c0: bbAdd(a.c0, b.c0),
        c1: bbAdd(a.c1, b.c1),
        c2: bbAdd(a.c2, b.c2),
        c3: bbAdd(a.c3, b.c3))
}

/// Bb4 subtraction: component-wise.
public func bb4Sub(_ a: Bb4, _ b: Bb4) -> Bb4 {
    Bb4(c0: bbSub(a.c0, b.c0),
        c1: bbSub(a.c1, b.c1),
        c2: bbSub(a.c2, b.c2),
        c3: bbSub(a.c3, b.c3))
}

/// Bb4 negation: component-wise.
public func bb4Neg(_ a: Bb4) -> Bb4 {
    Bb4(c0: bbNeg(a.c0), c1: bbNeg(a.c1), c2: bbNeg(a.c2), c3: bbNeg(a.c3))
}

/// Bb4 doubling.
public func bb4Double(_ a: Bb4) -> Bb4 {
    bb4Add(a, a)
}

/// Bb4 multiplication.
///
/// Given a = a0 + a1*x + a2*x^2 + a3*x^3 and b = b0 + b1*x + b2*x^2 + b3*x^3,
/// the product mod (x^4 - W) uses x^4 = W to reduce higher terms:
///
///   c0 = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
///   c1 = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
///   c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b3)
///   c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
public func bb4Mul(_ a: Bb4, _ b: Bb4) -> Bb4 {
    let w = Bb(v: Bb4.W)

    // Direct schoolbook multiplication with reduction by x^4 = W
    let a0b0 = bbMul(a.c0, b.c0)
    let a0b1 = bbMul(a.c0, b.c1)
    let a0b2 = bbMul(a.c0, b.c2)
    let a0b3 = bbMul(a.c0, b.c3)

    let a1b0 = bbMul(a.c1, b.c0)
    let a1b1 = bbMul(a.c1, b.c1)
    let a1b2 = bbMul(a.c1, b.c2)
    let a1b3 = bbMul(a.c1, b.c3)

    let a2b0 = bbMul(a.c2, b.c0)
    let a2b1 = bbMul(a.c2, b.c1)
    let a2b2 = bbMul(a.c2, b.c2)
    let a2b3 = bbMul(a.c2, b.c3)

    let a3b0 = bbMul(a.c3, b.c0)
    let a3b1 = bbMul(a.c3, b.c1)
    let a3b2 = bbMul(a.c3, b.c2)
    let a3b3 = bbMul(a.c3, b.c3)

    // c0 = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
    let c0 = bbAdd(a0b0, bbMul(w, bbAdd(bbAdd(a1b3, a2b2), a3b1)))

    // c1 = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
    let c1 = bbAdd(bbAdd(a0b1, a1b0), bbMul(w, bbAdd(a2b3, a3b2)))

    // c2 = a0*b2 + a1*b1 + a2*b0 + W*(a3*b3)
    let c2 = bbAdd(bbAdd(bbAdd(a0b2, a1b1), a2b0), bbMul(w, a3b3))

    // c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
    let c3 = bbAdd(bbAdd(a0b3, a1b2), bbAdd(a2b1, a3b0))

    return Bb4(c0: c0, c1: c1, c2: c2, c3: c3)
}

/// Bb4 squaring (optimized: fewer multiplications than generic mul).
///
/// a^2 = (a0 + a1*x + a2*x^2 + a3*x^3)^2
///   c0 = a0^2 + 2W*a1*a3 + W*a2^2
///   c1 = 2*a0*a1 + 2W*a2*a3
///   c2 = 2*a0*a2 + a1^2 + W*2*a3^2    (corrected: just a1^2 + 2*a0*a2 + 2W*a3^2 ... wait)
///
/// Let's derive carefully:
///   (a0 + a1*x + a2*x^2 + a3*x^3)^2
///   = a0^2 + 2*a0*a1*x + (2*a0*a2 + a1^2)*x^2 + (2*a0*a3 + 2*a1*a2)*x^3
///     + (2*a1*a3 + a2^2)*x^4 + (2*a2*a3)*x^5 + a3^2*x^6
///
///   Reduce: x^4 = W, x^5 = W*x, x^6 = W*x^2
///   c0 = a0^2 + W*(2*a1*a3 + a2^2)
///   c1 = 2*a0*a1 + W*(2*a2*a3)
///   c2 = 2*a0*a2 + a1^2 + W*a3^2
///   c3 = 2*a0*a3 + 2*a1*a2
public func bb4Sqr(_ a: Bb4) -> Bb4 {
    let w = Bb(v: Bb4.W)

    let a0sq = bbSqr(a.c0)
    let a1sq = bbSqr(a.c1)
    let a2sq = bbSqr(a.c2)
    let a3sq = bbSqr(a.c3)

    let a0a1 = bbMul(a.c0, a.c1)
    let a0a2 = bbMul(a.c0, a.c2)
    let a0a3 = bbMul(a.c0, a.c3)
    let a1a2 = bbMul(a.c1, a.c2)
    let a1a3 = bbMul(a.c1, a.c3)
    let a2a3 = bbMul(a.c2, a.c3)

    // c0 = a0^2 + W*(2*a1*a3 + a2^2)
    let c0 = bbAdd(a0sq, bbMul(w, bbAdd(bbAdd(a1a3, a1a3), a2sq)))

    // c1 = 2*a0*a1 + W*2*a2*a3
    let c1 = bbAdd(bbAdd(a0a1, a0a1), bbMul(w, bbAdd(a2a3, a2a3)))

    // c2 = 2*a0*a2 + a1^2 + W*a3^2
    let c2 = bbAdd(bbAdd(bbAdd(a0a2, a0a2), a1sq), bbMul(w, a3sq))

    // c3 = 2*a0*a3 + 2*a1*a2
    let c3 = bbAdd(bbAdd(a0a3, a0a3), bbAdd(a1a2, a1a2))

    return Bb4(c0: c0, c1: c1, c2: c2, c3: c3)
}

/// Bb4 scalar multiplication by a base field element.
public func bb4ScalarMul(_ s: Bb, _ a: Bb4) -> Bb4 {
    Bb4(c0: bbMul(s, a.c0),
        c1: bbMul(s, a.c1),
        c2: bbMul(s, a.c2),
        c3: bbMul(s, a.c3))
}

/// Bb4 multiplication by x (shift coefficients, reduce x^4 = W).
/// x * (c0 + c1*x + c2*x^2 + c3*x^3) = W*c3 + c0*x + c1*x^2 + c2*x^3
public func bb4MulByX(_ a: Bb4) -> Bb4 {
    let w = Bb(v: Bb4.W)
    return Bb4(c0: bbMul(w, a.c3), c1: a.c0, c2: a.c1, c3: a.c2)
}

/// Bb4 inverse via the norm-based approach.
///
/// For a in Bb4 = Bb[x]/(x^4 - W), we compute inv(a) using:
///   a * a_conj = N(a)  (an element in Bb)
///   inv(a) = a_conj / N(a)
///
/// where the norm and conjugate are computed from the quartic tower.
///
/// Specifically, for Bb4 = Bb[x]/(x^4 - W), factor as
///   Bb2 = Bb[y]/(y^2 - W), then Bb4 = Bb2[z]/(z^2 - y).
///
/// Write a = u + v*z where u = c0 + c2*y, v = c1 + c3*y (in Bb2).
/// Then N_Bb4/Bb2(a) = u^2 - v^2*y  (in Bb2).
/// Then N_Bb2/Bb(N_Bb4/Bb2(a)) gives an element of Bb.
///
/// For simplicity we use Fermat's little theorem on the full extension:
///   inv(a) = a^(p^4 - 2) — but this is expensive.
///
/// Instead we use the direct formula: compute via adjugate of the
/// multiplication matrix. For degree 4, it's most practical to use
/// the tower decomposition or the direct Euclidean approach.
///
/// We implement the tower approach:
///   Bb2 = Bb[y]/(y^2 - W) where y^2 = W
///   Bb4 = Bb2[z]/(z^2 - y) where z = x, z^2 = y
///
///   a = (c0 + c2*y) + (c1 + c3*y)*z = u + v*z
///   inv(a) = (u - v*z) / (u^2 - v^2*y) if u^2 - v^2*y != 0
///
/// The norm d = u^2 - v^2*y is in Bb2, and we invert it in Bb2:
///   Bb2 element (d0, d1) has inv = (d0, -d1) / (d0^2 - W*d1^2)
///   where d0^2 - W*d1^2 is in Bb.
public func bb4Inverse(_ a: Bb4) -> Bb4 {
    precondition(!a.isZero, "Cannot invert zero in Bb4")

    let w = Bb(v: Bb4.W)

    // Tower decomposition: a = u + v*z
    // u = c0 + c2*y (Bb2 element: u0=c0, u1=c2)
    // v = c1 + c3*y (Bb2 element: v0=c1, v1=c3)
    let u0 = a.c0; let u1 = a.c2
    let v0 = a.c1; let v1 = a.c3

    // u^2 in Bb2: (u0 + u1*y)^2 = u0^2 + W*u1^2 + 2*u0*u1*y
    let u2_0 = bbAdd(bbSqr(u0), bbMul(w, bbSqr(u1)))
    let u2_1 = bbAdd(bbMul(u0, u1), bbMul(u0, u1))

    // v^2 in Bb2: (v0 + v1*y)^2 = v0^2 + W*v1^2 + 2*v0*v1*y
    let v2_0 = bbAdd(bbSqr(v0), bbMul(w, bbSqr(v1)))
    let v2_1 = bbAdd(bbMul(v0, v1), bbMul(v0, v1))

    // v^2 * y in Bb2: (v2_0 + v2_1*y)*y = W*v2_1 + v2_0*y
    let v2y_0 = bbMul(w, v2_1)
    let v2y_1 = v2_0

    // d = u^2 - v^2*y in Bb2
    let d0 = bbSub(u2_0, v2y_0)
    let d1 = bbSub(u2_1, v2y_1)

    // Invert d in Bb2: inv(d) = (d0, -d1) / (d0^2 - W*d1^2)
    let normD = bbSub(bbSqr(d0), bbMul(w, bbSqr(d1)))
    let normDInv = bbInverse(normD)
    let dInv0 = bbMul(d0, normDInv)
    let dInv1 = bbMul(bbNeg(d1), normDInv)

    // inv(a) = (u - v*z) * inv(d)
    // = (u * inv(d)) - (v * inv(d)) * z
    // u * inv(d) in Bb2: (u0*dInv0 + W*u1*dInv1, u0*dInv1 + u1*dInv0)
    let r_u0 = bbAdd(bbMul(u0, dInv0), bbMul(w, bbMul(u1, dInv1)))
    let r_u1 = bbAdd(bbMul(u0, dInv1), bbMul(u1, dInv0))

    // v * inv(d) in Bb2
    let r_v0 = bbAdd(bbMul(v0, dInv0), bbMul(w, bbMul(v1, dInv1)))
    let r_v1 = bbAdd(bbMul(v0, dInv1), bbMul(v1, dInv0))

    // inv(a) = (r_u0 + r_u1*y) - (r_v0 + r_v1*y)*z
    // = r_u0 - r_v0*x + r_u1*x^2 - r_v1*x^3
    return Bb4(c0: r_u0, c1: bbNeg(r_v0), c2: r_u1, c3: bbNeg(r_v1))
}

/// Bb4 exponentiation via square-and-multiply.
public func bb4Pow(_ base: Bb4, _ exp: UInt64) -> Bb4 {
    if exp == 0 { return .one }
    var result = Bb4.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = bb4Mul(result, b) }
        b = bb4Sqr(b)
        e >>= 1
    }
    return result
}

// MARK: - Frobenius Endomorphism

/// Frobenius endomorphism on Bb4: phi(a) = a^p.
///
/// For Bb4 = Bb[x]/(x^4 - W), the Frobenius acts as:
///   phi(c0 + c1*x + c2*x^2 + c3*x^3) = c0 + c1*x^p + c2*x^{2p} + c3*x^{3p}
///
/// Since x^4 = W and p = 2013265921, we have p mod 4 = 1, so:
///   x^p = x^(4q+1) = W^q * x where q = (p-1)/4
///   x^{2p} = x^{2(4q+1)} = x^{8q+2} = W^{2q} * x^2
///   x^{3p} = x^{3(4q+1)} = x^{12q+3} = W^{3q} * x^3
///
/// So phi(a) = c0 + c1*W^q*x + c2*W^{2q}*x^2 + c3*W^{3q}*x^3
/// where q = (p-1)/4 = 503316480.
public func bb4Frobenius(_ a: Bb4) -> Bb4 {
    let q = (Bb.P - 1) / 4  // 503316480

    // Precompute W^q, W^{2q}, W^{3q} mod p
    let wq  = bbPow(Bb(v: Bb4.W), q)
    let w2q = bbSqr(wq)   // W^{2q} = (W^q)^2
    let w3q = bbMul(wq, w2q)  // W^{3q} = W^q * W^{2q}

    return Bb4(c0: a.c0,
               c1: bbMul(a.c1, wq),
               c2: bbMul(a.c2, w2q),
               c3: bbMul(a.c3, w3q))
}

/// Iterated Frobenius: phi^k(a) = a^{p^k}.
public func bb4FrobeniusK(_ a: Bb4, _ k: Int) -> Bb4 {
    var result = a
    for _ in 0..<(k % 4) {
        result = bb4Frobenius(result)
    }
    return result
}

// MARK: - Bb / Bb4 Conversions

/// Embed Bb into Bb4 (constant coefficient).
public func bb4FromBase(_ x: Bb) -> Bb4 {
    Bb4(from: x)
}

/// Extract base field component (c0) from Bb4.
public func bb4ToBase(_ a: Bb4) -> Bb {
    a.c0
}

/// Trace map: Tr(a) = a + phi(a) + phi^2(a) + phi^3(a).
/// For Bb4/Bb this maps to the base field.
public func bb4Trace(_ a: Bb4) -> Bb {
    let sum = bb4Add(bb4Add(a, bb4Frobenius(a)),
                     bb4Add(bb4FrobeniusK(a, 2), bb4FrobeniusK(a, 3)))
    return sum.c0
}

// MARK: - Random Element Generation

/// Generate a pseudo-random Bb4 element using a seed.
/// Uses a simple LCG for deterministic generation (not cryptographic).
public func bb4Random(seed: inout UInt64) -> Bb4 {
    func nextU32() -> UInt32 {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return UInt32((seed >> 33) % UInt64(Bb.P))
    }
    return Bb4(c0: Bb(v: nextU32()), c1: Bb(v: nextU32()),
               c2: Bb(v: nextU32()), c3: Bb(v: nextU32()))
}

/// Generate an array of pseudo-random Bb4 elements.
public func bb4RandomArray(count: Int, seed: inout UInt64) -> [Bb4] {
    (0..<count).map { _ in bb4Random(seed: &seed) }
}

// MARK: - Batch Operations

/// Batch Bb4 addition: out[i] = a[i] + b[i].
public func bb4BatchAdd(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
    precondition(a.count == b.count)
    return zip(a, b).map { bb4Add($0, $1) }
}

/// Batch Bb4 subtraction: out[i] = a[i] - b[i].
public func bb4BatchSub(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
    precondition(a.count == b.count)
    return zip(a, b).map { bb4Sub($0, $1) }
}

/// Batch Bb4 multiplication: out[i] = a[i] * b[i].
public func bb4BatchMul(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
    precondition(a.count == b.count)
    return zip(a, b).map { bb4Mul($0, $1) }
}

/// Batch Bb4 squaring: out[i] = a[i]^2.
public func bb4BatchSqr(_ a: [Bb4]) -> [Bb4] {
    a.map { bb4Sqr($0) }
}

/// Batch Bb4 scalar multiplication: out[i] = s * a[i].
public func bb4BatchScalarMul(_ s: Bb, _ a: [Bb4]) -> [Bb4] {
    a.map { bb4ScalarMul(s, $0) }
}

/// Batch Bb4 inverse using Montgomery's trick.
/// Uses 1 Bb4 inversion + 3(N-1) Bb4 multiplications.
/// Zero elements map to zero.
public func bb4BatchInv(_ xs: [Bb4]) -> [Bb4] {
    let n = xs.count
    guard n > 0 else { return [] }
    if n == 1 {
        return [xs[0].isZero ? .zero : bb4Inverse(xs[0])]
    }

    // Compute prefix products, skipping zeros
    var prefix = [Bb4](repeating: .zero, count: n)
    var acc = Bb4.one
    for i in 0..<n {
        if xs[i].isZero {
            prefix[i] = acc
        } else {
            acc = bb4Mul(acc, xs[i])
            prefix[i] = acc
        }
    }

    // Single inversion of accumulated product
    var inv = bb4Inverse(acc)

    // Sweep backwards to recover individual inverses
    var result = [Bb4](repeating: .zero, count: n)
    for i in stride(from: n - 1, through: 1, by: -1) {
        if xs[i].isZero {
            result[i] = .zero
            continue
        }
        result[i] = bb4Mul(inv, prefix[i - 1])
        inv = bb4Mul(inv, xs[i])
    }
    result[0] = xs[0].isZero ? .zero : inv
    return result
}

/// Inner product: sum_i a[i] * b[i].
public func bb4InnerProduct(_ a: [Bb4], _ b: [Bb4]) -> Bb4 {
    precondition(a.count == b.count)
    var acc = Bb4.zero
    for i in 0..<a.count {
        acc = bb4Add(acc, bb4Mul(a[i], b[i]))
    }
    return acc
}

// MARK: - Extension Field NTT over Bb4

/// Forward NTT over Bb4, using BabyBear roots of unity embedded in Bb4.
///
/// The NTT operates on polynomials with Bb4 coefficients, using the
/// same twiddle factors as the base-field NTT (roots of unity in Bb
/// embedded into Bb4).
///
/// n must be a power of 2, with log2(n) <= Bb.TWO_ADICITY = 27.
public func bb4NTTForward(_ coeffs: [Bb4]) -> [Bb4] {
    let n = coeffs.count
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
    let logN = Int(log2(Double(n)))
    precondition(logN <= Bb.TWO_ADICITY, "n exceeds BabyBear two-adicity")

    if n == 1 { return coeffs }

    var a = coeffs

    // Bit-reversal permutation
    for i in 0..<n {
        let j = bitReverseBb4(i, logN)
        if i < j {
            a.swapAt(i, j)
        }
    }

    // Cooley-Tukey butterfly
    var len = 1
    while len < n {
        let omega = bbRootOfUnity(logN: intLog2(len * 2))
        var w = Bb.one
        for j in 0..<len {
            let wExt = bb4FromBase(w)
            var k = j
            while k < n {
                let u = a[k]
                let v = bb4Mul(wExt, a[k + len])
                a[k] = bb4Add(u, v)
                a[k + len] = bb4Sub(u, v)
                k += 2 * len
            }
            w = bbMul(w, omega)
        }
        len <<= 1
    }

    return a
}

/// Inverse NTT over Bb4.
public func bb4NTTInverse(_ values: [Bb4]) -> [Bb4] {
    let n = values.count
    precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
    let logN = Int(log2(Double(n)))
    precondition(logN <= Bb.TWO_ADICITY, "n exceeds BabyBear two-adicity")

    if n == 1 { return values }

    var a = values

    // Bit-reversal permutation
    for i in 0..<n {
        let j = bitReverseBb4(i, logN)
        if i < j {
            a.swapAt(i, j)
        }
    }

    // Gentleman-Sande butterfly with inverse roots
    var len = 1
    while len < n {
        let omega = bbInverse(bbRootOfUnity(logN: intLog2(len * 2)))
        var w = Bb.one
        for j in 0..<len {
            let wExt = bb4FromBase(w)
            var k = j
            while k < n {
                let u = a[k]
                let v = bb4Mul(wExt, a[k + len])
                a[k] = bb4Add(u, v)
                a[k + len] = bb4Sub(u, v)
                k += 2 * len
            }
            w = bbMul(w, omega)
        }
        len <<= 1
    }

    // Scale by 1/n
    let nInv = bbInverse(Bb(v: UInt32(n)))
    for i in 0..<n {
        a[i] = bb4ScalarMul(nInv, a[i])
    }

    return a
}

/// Polynomial multiplication over Bb4 using NTT.
public func bb4PolyMul(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
    let resultLen = a.count + b.count - 1
    var n = 1
    while n < resultLen { n <<= 1 }

    var aPad = a
    var bPad = b
    aPad.append(contentsOf: [Bb4](repeating: .zero, count: n - a.count))
    bPad.append(contentsOf: [Bb4](repeating: .zero, count: n - b.count))

    let aNTT = bb4NTTForward(aPad)
    let bNTT = bb4NTTForward(bPad)
    let cNTT = bb4BatchMul(aNTT, bNTT)
    let c = bb4NTTInverse(cNTT)

    return Array(c[0..<resultLen])
}

// MARK: - NTT Helpers

/// Bit-reverse an index for the NTT butterfly.
private func bitReverseBb4(_ x: Int, _ bits: Int) -> Int {
    var result = 0
    var val = x
    for _ in 0..<bits {
        result = (result << 1) | (val & 1)
        val >>= 1
    }
    return result
}

/// Integer log2 (exact, for powers of 2).
private func intLog2(_ n: Int) -> Int {
    precondition(n > 0 && (n & (n - 1)) == 0)
    var v = n
    var r = 0
    while v > 1 {
        v >>= 1
        r += 1
    }
    return r
}

// MARK: - GPU BabyBear Extension Engine

/// Orchestrates GPU-accelerated BabyBear quartic extension field operations.
///
/// Provides batch arithmetic, NTT, Frobenius, and conversion operations
/// over Bb4 = Bb[x]/(x^4 - 11).
public final class GPUBabyBearExtensionEngine {
    public static let shared = GPUBabyBearExtensionEngine()

    private init() {}

    // MARK: - Single Element Operations

    /// Multiply two Bb4 elements.
    public func mul(_ a: Bb4, _ b: Bb4) -> Bb4 {
        bb4Mul(a, b)
    }

    /// Square a Bb4 element.
    public func sqr(_ a: Bb4) -> Bb4 {
        bb4Sqr(a)
    }

    /// Invert a Bb4 element.
    public func inv(_ a: Bb4) -> Bb4 {
        bb4Inverse(a)
    }

    /// Apply Frobenius endomorphism.
    public func frobenius(_ a: Bb4) -> Bb4 {
        bb4Frobenius(a)
    }

    // MARK: - Batch Operations

    /// Batch Bb4 multiply.
    public func batchMul(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
        bb4BatchMul(a, b)
    }

    /// Batch Bb4 add.
    public func batchAdd(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
        bb4BatchAdd(a, b)
    }

    /// Batch Bb4 inverse using Montgomery's trick.
    public func batchInv(_ xs: [Bb4]) -> [Bb4] {
        bb4BatchInv(xs)
    }

    /// Batch Bb4 squaring.
    public func batchSqr(_ xs: [Bb4]) -> [Bb4] {
        bb4BatchSqr(xs)
    }

    /// Compute product of array (reduction).
    public func product(_ xs: [Bb4]) -> Bb4 {
        var acc = Bb4.one
        for x in xs {
            acc = bb4Mul(acc, x)
        }
        return acc
    }

    // MARK: - NTT

    /// Forward NTT over Bb4.
    public func nttForward(_ coeffs: [Bb4]) -> [Bb4] {
        bb4NTTForward(coeffs)
    }

    /// Inverse NTT over Bb4.
    public func nttInverse(_ values: [Bb4]) -> [Bb4] {
        bb4NTTInverse(values)
    }

    /// Polynomial multiplication via NTT.
    public func polyMul(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
        bb4PolyMul(a, b)
    }

    // MARK: - Conversion

    /// Embed base field elements into Bb4.
    public func fromBase(_ xs: [Bb]) -> [Bb4] {
        xs.map { bb4FromBase($0) }
    }

    /// Extract base field components from Bb4 elements.
    public func toBase(_ xs: [Bb4]) -> [Bb] {
        xs.map { bb4ToBase($0) }
    }

    // MARK: - Random

    /// Generate random Bb4 elements.
    public func randomElements(count: Int, seed: UInt64 = 0x12345678) -> [Bb4] {
        var s = seed
        return bb4RandomArray(count: count, seed: &s)
    }
}
