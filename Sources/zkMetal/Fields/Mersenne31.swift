// Mersenne31 field arithmetic (CPU-side)
// p = 2^31 - 1 = 0x7FFFFFFF = 2147483647
// Used by Stwo (StarkWare), Circle STARKs
// The circle group over M31 has order p+1 = 2^31 (full 2-adicity!)

import Foundation

public struct M31: Equatable {
    public var v: UInt32

    public static let P: UInt32 = 0x7FFFFFFF  // 2147483647

    // Circle group order: p + 1 = 2^31
    public static let CIRCLE_ORDER_LOG: Int = 31

    // Circle group generator: (2, 1268011823)
    // This point has order 2^31 on the circle x^2 + y^2 = 1 mod p
    public static let CIRCLE_GEN_X: UInt32 = 2
    public static let CIRCLE_GEN_Y: UInt32 = 1268011823

    public static var zero: M31 { M31(v: 0) }
    public static var one: M31 { M31(v: 1) }

    public init(v: UInt32) {
        self.v = v
    }

    public var isZero: Bool { v == 0 }
}

@inline(__always)
public func m31Add(_ a: M31, _ b: M31) -> M31 {
    let s = UInt32(a.v &+ b.v)
    // Since a, b < p < 2^31, sum < 2^32, and we need mod (2^31 - 1)
    // (s & p) + (s >> 31) handles the wraparound
    let r = (s & M31.P) &+ (s >> 31)
    // One more reduction in case r == p
    return M31(v: r == M31.P ? 0 : r)
}

@inline(__always)
public func m31Sub(_ a: M31, _ b: M31) -> M31 {
    if a.v >= b.v {
        return M31(v: a.v - b.v)
    }
    return M31(v: a.v &+ M31.P &- b.v)
}

@inline(__always)
public func m31Neg(_ a: M31) -> M31 {
    if a.v == 0 { return a }
    return M31(v: M31.P - a.v)
}

@inline(__always)
public func m31Mul(_ a: M31, _ b: M31) -> M31 {
    let prod = UInt64(a.v) * UInt64(b.v)
    let lo = UInt32(truncatingIfNeeded: prod & UInt64(M31.P))
    let hi = UInt32(truncatingIfNeeded: prod >> 31)
    let s = lo &+ hi
    let r = (s & M31.P) &+ (s >> 31)
    return M31(v: r == M31.P ? 0 : r)
}

@inline(__always)
public func m31Sqr(_ a: M31) -> M31 { m31Mul(a, a) }

public func m31Pow(_ base: M31, _ exp: UInt32) -> M31 {
    if exp == 0 { return M31.one }
    var result = M31.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = m31Mul(result, b) }
        b = m31Sqr(b)
        e >>= 1
    }
    return result
}

public func m31Inverse(_ a: M31) -> M31 {
    // Fermat's little theorem: a^(p-2) mod p
    return m31Pow(a, M31.P - 2)
}

// MARK: - CM31: Complex extension M31[i] / (i^2 + 1)

public struct CM31: Equatable {
    public var a: M31  // real part
    public var b: M31  // imaginary part

    public static var zero: CM31 { CM31(a: M31.zero, b: M31.zero) }
    public static var one: CM31 { CM31(a: M31.one, b: M31.zero) }
    public static var i: CM31 { CM31(a: M31.zero, b: M31.one) }

    public init(a: M31, b: M31) {
        self.a = a
        self.b = b
    }

    public var isZero: Bool { a.isZero && b.isZero }
}

@inline(__always)
public func cm31Add(_ a: CM31, _ b: CM31) -> CM31 {
    CM31(a: m31Add(a.a, b.a), b: m31Add(a.b, b.b))
}

@inline(__always)
public func cm31Sub(_ a: CM31, _ b: CM31) -> CM31 {
    CM31(a: m31Sub(a.a, b.a), b: m31Sub(a.b, b.b))
}

@inline(__always)
public func cm31Neg(_ a: CM31) -> CM31 {
    CM31(a: m31Neg(a.a), b: m31Neg(a.b))
}

@inline(__always)
public func cm31Mul(_ a: CM31, _ b: CM31) -> CM31 {
    // (a.a + a.b*i)(b.a + b.b*i) = (a.a*b.a - a.b*b.b) + (a.a*b.b + a.b*b.a)*i
    let real = m31Sub(m31Mul(a.a, b.a), m31Mul(a.b, b.b))
    let imag = m31Add(m31Mul(a.a, b.b), m31Mul(a.b, b.a))
    return CM31(a: real, b: imag)
}

public func cm31Inverse(_ a: CM31) -> CM31 {
    // 1/(a+bi) = (a-bi)/(a^2+b^2)
    let normSq = m31Add(m31Sqr(a.a), m31Sqr(a.b))
    let invNorm = m31Inverse(normSq)
    return CM31(a: m31Mul(a.a, invNorm), b: m31Mul(m31Neg(a.b), invNorm))
}

/// Multiply CM31 by a real M31 scalar
@inline(__always)
public func cm31ScalarMul(_ c: CM31, _ s: M31) -> CM31 {
    CM31(a: m31Mul(c.a, s), b: m31Mul(c.b, s))
}

// MARK: - Circle Group

/// A point on the circle x^2 + y^2 = 1 over M31
public struct CirclePoint: Equatable {
    public var x: M31
    public var y: M31

    public static var identity: CirclePoint { CirclePoint(x: M31.one, y: M31.zero) }
    public static var generator: CirclePoint {
        CirclePoint(x: M31(v: M31.CIRCLE_GEN_X), y: M31(v: M31.CIRCLE_GEN_Y))
    }

    public init(x: M31, y: M31) {
        self.x = x
        self.y = y
    }

    /// Verify this point is on the circle
    public var isOnCircle: Bool {
        let lhs = m31Add(m31Sqr(x), m31Sqr(y))
        return lhs.v == 1
    }
}

/// Circle group operation: complex multiplication
/// (x1,y1) * (x2,y2) = (x1*x2 - y1*y2, x1*y2 + y1*x2)
@inline(__always)
public func circleGroupMul(_ a: CirclePoint, _ b: CirclePoint) -> CirclePoint {
    let x = m31Sub(m31Mul(a.x, b.x), m31Mul(a.y, b.y))
    let y = m31Add(m31Mul(a.x, b.y), m31Mul(a.y, b.x))
    return CirclePoint(x: x, y: y)
}

/// Circle group conjugate (inverse): (x, y) -> (x, -y)
@inline(__always)
public func circleGroupConj(_ a: CirclePoint) -> CirclePoint {
    CirclePoint(x: a.x, y: m31Neg(a.y))
}

/// Circle group exponentiation by scalar
public func circleGroupPow(_ base: CirclePoint, _ exp: Int) -> CirclePoint {
    if exp == 0 { return CirclePoint.identity }
    var result = CirclePoint.identity
    var b = exp < 0 ? circleGroupConj(base) : base
    var e = exp < 0 ? -exp : exp
    while e > 0 {
        if e & 1 == 1 { result = circleGroupMul(result, b) }
        b = circleGroupMul(b, b)
        e >>= 1
    }
    return result
}

/// Circle doubling map (squaring): pi(x, y) = (2x^2 - 1, 2xy)
@inline(__always)
public func circleDouble(_ p: CirclePoint) -> CirclePoint {
    circleGroupMul(p, p)
}

/// Get generator of order 2^k subgroup of the circle group
public func circleSubgroupGenerator(logN: Int) -> CirclePoint {
    precondition(logN <= M31.CIRCLE_ORDER_LOG)
    return circleGroupPow(CirclePoint.generator, 1 << (M31.CIRCLE_ORDER_LOG - logN))
}

/// Build circle domain (coset) of size 2^k for Circle FFT
/// Uses generator as coset shift to avoid zero y-coordinates
public func circleCosetDomain(logN: Int) -> [CirclePoint] {
    let n = 1 << logN
    let genK = circleSubgroupGenerator(logN: logN)
    let shift = CirclePoint.generator
    var domain = [CirclePoint](repeating: CirclePoint.identity, count: n)
    domain[0] = shift
    for i in 1..<n {
        domain[i] = circleGroupMul(domain[i - 1], genK)
    }
    return domain
}
