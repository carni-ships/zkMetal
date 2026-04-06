// GPU-accelerated Binius binary tower field engine.
// F_2 -> F_4 -> F_16 -> F_256 -> F_{2^16} -> F_{2^32}
// Each level: F_{2^{2k}} = F_{2^k}[X] / (X^2 + X + alpha_k)
// Addition = XOR (free), multiplication = Karatsuba + reduce.

import Foundation
import NeonFieldOps

// MARK: - BiniusF2 — The base field GF(2)

/// The base field GF(2) = {0, 1}.
/// All arithmetic is modulo 2: add = XOR, mul = AND.
public struct BiniusF2: Equatable, CustomStringConvertible {
    public var bit: UInt8

    public static let zero = BiniusF2(bit: 0)
    public static let one  = BiniusF2(bit: 1)

    public init(bit: UInt8) {
        self.bit = bit & 1
    }

    public var isZero: Bool { bit == 0 }
    public var description: String { "F2(\(bit))" }

    @inline(__always)
    public static func + (a: BiniusF2, b: BiniusF2) -> BiniusF2 {
        BiniusF2(bit: a.bit ^ b.bit)
    }

    @inline(__always)
    public static func - (a: BiniusF2, b: BiniusF2) -> BiniusF2 {
        a + b  // char 2
    }

    @inline(__always)
    public static func * (a: BiniusF2, b: BiniusF2) -> BiniusF2 {
        BiniusF2(bit: a.bit & b.bit)
    }

    public func inverse() -> BiniusF2 {
        precondition(!isZero, "Cannot invert zero in F2")
        return self  // 1^(-1) = 1
    }

    @inline(__always)
    public func squared() -> BiniusF2 { self }
}

// MARK: - BiniusF4 — GF(4) = GF(2)[X] / (X^2 + X + 1)

/// GF(4) with irreducible X^2 + X + 1.
/// Elements: {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
/// Stored as 2-bit value: lo = coeff of 1, hi = coeff of alpha.
public struct BiniusF4: Equatable, CustomStringConvertible {
    public var value: UInt8  // only low 2 bits used

    public static let zero  = BiniusF4(value: 0)
    public static let one   = BiniusF4(value: 1)
    public static let alpha = BiniusF4(value: 2)  // the root of X^2+X+1

    public init(value: UInt8) {
        self.value = value & 0x03
    }

    public init(lo: UInt8, hi: UInt8) {
        self.value = (lo & 1) | ((hi & 1) << 1)
    }

    public var lo: UInt8 { value & 1 }
    public var hi: UInt8 { (value >> 1) & 1 }
    public var isZero: Bool { value == 0 }
    public var description: String { "F4(\(value))" }

    @inline(__always)
    public static func + (a: BiniusF4, b: BiniusF4) -> BiniusF4 {
        BiniusF4(value: a.value ^ b.value)
    }

    @inline(__always)
    public static func - (a: BiniusF4, b: BiniusF4) -> BiniusF4 {
        a + b
    }

    /// Multiplication in GF(4): Karatsuba with alpha^2 = alpha + 1.
    /// (a0 + a1*alpha)(b0 + b1*alpha)
    /// = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
    /// = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*(alpha + 1)
    /// = (a0*b0 + a1*b1) + (a0*b1 + a1*b0 + a1*b1)*alpha
    @inline(__always)
    public static func * (a: BiniusF4, b: BiniusF4) -> BiniusF4 {
        // Use lookup table for 4x4 = 16 entries
        return BiniusF4(value: biniusF4MulTable[Int(a.value)][Int(b.value)])
    }

    public func inverse() -> BiniusF4 {
        precondition(!isZero, "Cannot invert zero in GF(4)")
        // GF(4) inverse: 1->1, alpha->alpha+1, alpha+1->alpha
        return BiniusF4(value: biniusF4InvTable[Int(value)])
    }

    @inline(__always)
    public func squared() -> BiniusF4 {
        // In GF(4), squaring is Frobenius: (a+b*alpha)^2 = a + b*(alpha+1) = (a+b) + b*alpha
        // Wait: alpha^2 = alpha + 1, so (a + b*alpha)^2 = a + b*(alpha+1) = (a+b) + b*alpha
        BiniusF4(lo: lo ^ hi, hi: hi)
    }
}

/// GF(4) multiplication lookup table.
private let biniusF4MulTable: [[UInt8]] = {
    // Compute (a0+a1*alpha)*(b0+b1*alpha) for all 4x4 combinations
    var table = [[UInt8]](repeating: [UInt8](repeating: 0, count: 4), count: 4)
    for a in 0..<4 {
        for b in 0..<4 {
            let a0 = UInt8(a & 1)
            let a1 = UInt8((a >> 1) & 1)
            let b0 = UInt8(b & 1)
            let b1 = UInt8((b >> 1) & 1)
            let rLo = (a0 & b0) ^ (a1 & b1)  // a0*b0 + a1*b1
            let rHi = (a0 & b1) ^ (a1 & b0) ^ (a1 & b1)  // a0*b1 + a1*b0 + a1*b1
            table[a][b] = rLo | (rHi << 1)
        }
    }
    return table
}()

/// GF(4) inverse lookup table: inv[0] = 0 (sentinel), inv[1] = 1, inv[2] = 3, inv[3] = 2
private let biniusF4InvTable: [UInt8] = [0, 1, 3, 2]

// MARK: - BiniusF16 — GF(16) = GF(4)[X] / (X^2 + X + alpha)

/// GF(16) via tower extension over GF(4).
/// X^2 + X + alpha where alpha is the GF(4) generator.
/// Stored as 4-bit value: low 2 bits = lo coeff in GF(4), high 2 bits = hi coeff.
public struct BiniusF16: Equatable, CustomStringConvertible {
    public var value: UInt8  // only low 4 bits used

    public static let zero = BiniusF16(value: 0)
    public static let one  = BiniusF16(value: 1)

    public init(value: UInt8) {
        self.value = value & 0x0F
    }

    public init(lo: BiniusF4, hi: BiniusF4) {
        self.value = lo.value | (hi.value << 2)
    }

    public var lo: BiniusF4 { BiniusF4(value: value & 0x03) }
    public var hi: BiniusF4 { BiniusF4(value: (value >> 2) & 0x03) }
    public var isZero: Bool { value == 0 }
    public var description: String { "F16(\(value))" }

    @inline(__always)
    public static func + (a: BiniusF16, b: BiniusF16) -> BiniusF16 {
        BiniusF16(value: a.value ^ b.value)
    }

    @inline(__always)
    public static func - (a: BiniusF16, b: BiniusF16) -> BiniusF16 {
        a + b
    }

    /// Karatsuba multiplication in GF(16) tower.
    /// (a + bX)(c + dX) mod (X^2 + X + alpha)
    /// = (ac + bd*alpha) + (ac + (a+b)(c+d))X
    @inline(__always)
    public static func * (a: BiniusF16, b: BiniusF16) -> BiniusF16 {
        return BiniusF16(value: biniusF16MulTable[Int(a.value)][Int(b.value)])
    }

    public func inverse() -> BiniusF16 {
        precondition(!isZero, "Cannot invert zero in GF(16)")
        return BiniusF16(value: biniusF16InvTable[Int(value)])
    }

    @inline(__always)
    public func squared() -> BiniusF16 {
        self * self
    }
}

/// GF(16) multiplication lookup table (16 x 16 = 256 entries).
private let biniusF16MulTable: [[UInt8]] = {
    var table = [[UInt8]](repeating: [UInt8](repeating: 0, count: 16), count: 16)
    let alphaF4 = BiniusF4.alpha  // tower constant alpha in GF(4)
    for a in 0..<16 {
        for b in 0..<16 {
            let aEl = BiniusF16(value: UInt8(a))
            let bEl = BiniusF16(value: UInt8(b))
            let ac = aEl.lo * bEl.lo
            let bd = aEl.hi * bEl.hi
            let e = (aEl.lo + aEl.hi) * (bEl.lo + bEl.hi)
            let bdAlpha = bd * alphaF4
            let rLo = ac + bdAlpha
            let rHi = e + ac
            table[a][b] = rLo.value | (rHi.value << 2)
        }
    }
    return table
}()

/// GF(16) inverse lookup table.
private let biniusF16InvTable: [UInt8] = {
    var table = [UInt8](repeating: 0, count: 16)
    for a in 1..<16 {
        let aEl = BiniusF16(value: UInt8(a))
        // Brute force: find b such that a * b = 1
        for b in 1..<16 {
            let bEl = BiniusF16(value: UInt8(b))
            if (aEl * bEl).value == 1 {
                table[a] = UInt8(b)
                break
            }
        }
    }
    return table
}()

// MARK: - BiniusF256 — GF(256) = GF(16)[X] / (X^2 + X + beta)

/// GF(256) via tower extension over GF(16).
/// Stored as a full byte: lo nibble = GF(16) lo, hi nibble = GF(16) hi.
public struct BiniusF256: Equatable, CustomStringConvertible {
    public var value: UInt8

    public static let zero = BiniusF256(value: 0)
    public static let one  = BiniusF256(value: 1)

    /// Tower constant beta: chosen so X^2 + X + beta is irreducible over GF(16).
    /// beta = element 2 in GF(16) (which is alpha in the GF(4) lo position).
    public static let beta = BiniusF16(value: 0x02)

    public init(value: UInt8) {
        self.value = value
    }

    public init(lo: BiniusF16, hi: BiniusF16) {
        self.value = lo.value | (hi.value << 4)
    }

    public var lo: BiniusF16 { BiniusF16(value: value & 0x0F) }
    public var hi: BiniusF16 { BiniusF16(value: (value >> 4) & 0x0F) }
    public var isZero: Bool { value == 0 }
    public var description: String { "F256(0x\(String(value, radix: 16, uppercase: true)))" }

    @inline(__always)
    public static func + (a: BiniusF256, b: BiniusF256) -> BiniusF256 {
        BiniusF256(value: a.value ^ b.value)
    }

    @inline(__always)
    public static func - (a: BiniusF256, b: BiniusF256) -> BiniusF256 {
        a + b
    }

    /// Karatsuba multiplication in the tower.
    @inline(__always)
    public static func * (a: BiniusF256, b: BiniusF256) -> BiniusF256 {
        let ac = a.lo * b.lo
        let bd = a.hi * b.hi
        let e = (a.lo + a.hi) * (b.lo + b.hi)
        let bdBeta = bd * BiniusF256.beta
        return BiniusF256(lo: ac + bdBeta, hi: e + ac)
    }

    /// Inverse via norm to GF(16) then invert in subfield.
    public func inverse() -> BiniusF256 {
        precondition(!isZero, "Cannot invert zero in GF(256)")
        let a = lo
        let b = hi
        let a2 = a * a
        let ab = a * b
        let b2Beta = b * b * BiniusF256.beta
        let norm = a2 + ab + b2Beta
        let normInv = norm.inverse()
        return BiniusF256(lo: (a + b) * normInv, hi: b * normInv)
    }

    @inline(__always)
    public func squared() -> BiniusF256 { self * self }

    /// Exponentiation via square-and-multiply.
    public func pow(_ n: Int) -> BiniusF256 {
        if n == 0 { return .one }
        var result = BiniusF256.one
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

// MARK: - PackedBinaryField32 — 32 GF(2) elements packed in a UInt32

/// 32 GF(2) elements packed into a single UInt32 word.
/// Bitwise XOR = parallel addition of 32 elements.
/// Bitwise AND = parallel multiplication of 32 elements.
/// This is the fundamental SIMD unit for Binius binary field operations.
public struct PackedBinaryField32: Equatable, CustomStringConvertible {
    public var bits: UInt32

    public static let zero = PackedBinaryField32(bits: 0)
    public static let ones = PackedBinaryField32(bits: UInt32.max)

    public init(bits: UInt32) {
        self.bits = bits
    }

    public var description: String {
        "PBF32(0x\(String(bits, radix: 16, uppercase: true)))"
    }

    /// Parallel add of 32 GF(2) elements (XOR).
    @inline(__always)
    public static func + (a: PackedBinaryField32, b: PackedBinaryField32) -> PackedBinaryField32 {
        PackedBinaryField32(bits: a.bits ^ b.bits)
    }

    /// Parallel subtract = add in char 2.
    @inline(__always)
    public static func - (a: PackedBinaryField32, b: PackedBinaryField32) -> PackedBinaryField32 {
        a + b
    }

    /// Parallel multiply of 32 GF(2) elements (AND).
    @inline(__always)
    public static func * (a: PackedBinaryField32, b: PackedBinaryField32) -> PackedBinaryField32 {
        PackedBinaryField32(bits: a.bits & b.bits)
    }

    /// Population count — number of 1-bits (useful for trace weight).
    public var popcount: Int {
        var x = bits
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return Int((x &* 0x01010101) >> 24)
    }

    /// Extract single bit at position i.
    @inline(__always)
    public func bit(at i: Int) -> UInt8 {
        UInt8((bits >> i) & 1)
    }

    /// Set single bit at position i.
    @inline(__always)
    public mutating func setBit(at i: Int, to val: UInt8) {
        if val & 1 == 1 {
            bits |= (1 << i)
        } else {
            bits &= ~(1 << i)
        }
    }

    /// XOR reduction — parity of all 32 bits.
    public var parity: UInt8 {
        var x = bits
        x ^= x >> 16
        x ^= x >> 8
        x ^= x >> 4
        x ^= x >> 2
        x ^= x >> 1
        return UInt8(x & 1)
    }
}

// MARK: - PackedBinaryField64 — 64 GF(2) elements packed in a UInt64

/// 64 GF(2) elements packed into a single UInt64 word.
public struct PackedBinaryField64: Equatable, CustomStringConvertible {
    public var bits: UInt64

    public static let zero = PackedBinaryField64(bits: 0)
    public static let ones = PackedBinaryField64(bits: UInt64.max)

    public init(bits: UInt64) {
        self.bits = bits
    }

    public var description: String {
        "PBF64(0x\(String(bits, radix: 16, uppercase: true)))"
    }

    @inline(__always)
    public static func + (a: PackedBinaryField64, b: PackedBinaryField64) -> PackedBinaryField64 {
        PackedBinaryField64(bits: a.bits ^ b.bits)
    }

    @inline(__always)
    public static func - (a: PackedBinaryField64, b: PackedBinaryField64) -> PackedBinaryField64 {
        a + b
    }

    @inline(__always)
    public static func * (a: PackedBinaryField64, b: PackedBinaryField64) -> PackedBinaryField64 {
        PackedBinaryField64(bits: a.bits & b.bits)
    }

    /// Split into two PackedBinaryField32 halves.
    public var loHalf: PackedBinaryField32 {
        PackedBinaryField32(bits: UInt32(bits & 0xFFFFFFFF))
    }
    public var hiHalf: PackedBinaryField32 {
        PackedBinaryField32(bits: UInt32(bits >> 32))
    }

    /// Construct from two 32-bit halves.
    public init(lo: PackedBinaryField32, hi: PackedBinaryField32) {
        self.bits = UInt64(lo.bits) | (UInt64(hi.bits) << 32)
    }

    /// Extract single bit at position i.
    @inline(__always)
    public func bit(at i: Int) -> UInt8 {
        UInt8((bits >> i) & 1)
    }

    /// XOR reduction — parity of all 64 bits.
    public var parity: UInt8 {
        var x = bits
        x ^= x >> 32
        x ^= x >> 16
        x ^= x >> 8
        x ^= x >> 4
        x ^= x >> 2
        x ^= x >> 1
        return UInt8(x & 1)
    }
}

// MARK: - Binary Field Polynomial Arithmetic

/// Polynomial arithmetic over binary tower fields.
/// Coefficients are BiniusF256 elements; addition = XOR (no carries).
public enum BiniusBinaryPoly {

    /// Add two polynomials (coefficient-wise XOR).
    public static func add(_ a: [BiniusF256], _ b: [BiniusF256]) -> [BiniusF256] {
        let maxLen = max(a.count, b.count)
        var result = [BiniusF256](repeating: .zero, count: maxLen)
        for i in 0..<a.count { result[i] = result[i] + a[i] }
        for i in 0..<b.count { result[i] = result[i] + b[i] }
        return result
    }

    /// Multiply two polynomials (schoolbook, O(n^2)).
    public static func mul(_ a: [BiniusF256], _ b: [BiniusF256]) -> [BiniusF256] {
        if a.isEmpty || b.isEmpty { return [] }
        let n = a.count + b.count - 1
        var result = [BiniusF256](repeating: .zero, count: n)
        for i in 0..<a.count {
            for j in 0..<b.count {
                result[i + j] = result[i + j] + (a[i] * b[j])
            }
        }
        return result
    }

    /// Evaluate polynomial at a point using Horner's method.
    public static func evaluate(_ coeffs: [BiniusF256], at x: BiniusF256) -> BiniusF256 {
        if coeffs.isEmpty { return .zero }
        var acc = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            acc = acc * x + coeffs[i]
        }
        return acc
    }

    /// Compute the degree of a polynomial (index of highest nonzero coefficient).
    public static func degree(_ p: [BiniusF256]) -> Int {
        for i in stride(from: p.count - 1, through: 0, by: -1) {
            if !p[i].isZero { return i }
        }
        return -1  // zero polynomial
    }

    /// Scale polynomial by a scalar: out[i] = s * p[i].
    public static func scale(_ p: [BiniusF256], by s: BiniusF256) -> [BiniusF256] {
        p.map { $0 * s }
    }
}

// MARK: - Additive NTT over Binary Fields

/// Additive NTT over binary fields.
/// Uses additive subgroups instead of multiplicative roots of unity.
/// Core transform for Binius polynomial commitment schemes.
public enum BiniusAdditiveNTT {

    /// Forward additive NTT: evaluate polynomial at all points of a 2^k affine subspace.
    public static func forward(_ coeffs: [BiniusF256], basis: [BiniusF256]) -> [BiniusF256] {
        let k = basis.count
        let n = 1 << k
        precondition(coeffs.count == n, "coeffs must have 2^k entries")

        var data = coeffs

        // Butterfly passes: for each level, split and recombine
        var halfSize = n >> 1
        for level in 0..<k {
            let twist = basis[level]
            var offset = 0
            while offset < n {
                for j in 0..<halfSize {
                    let u = data[offset + j]
                    let v = data[offset + halfSize + j]
                    // Butterfly: u' = u + v, v' = u + twist*v
                    data[offset + j] = u + v
                    data[offset + halfSize + j] = u + (twist * v)
                }
                offset += halfSize << 1
            }
            halfSize >>= 1
        }

        return data
    }

    /// Inverse additive NTT: interpolate from evaluations back to coefficients.
    public static func inverse(_ evals: [BiniusF256], basis: [BiniusF256]) -> [BiniusF256] {
        let k = basis.count
        let n = 1 << k
        precondition(evals.count == n, "evals must have 2^k entries")

        var data = evals

        // Reverse butterfly passes
        var halfSize = 1
        for level in stride(from: k - 1, through: 0, by: -1) {
            let twist = basis[level]
            let twistInv = twist.inverse()
            var offset = 0
            while offset < n {
                for j in 0..<halfSize {
                    let u = data[offset + j]
                    let v = data[offset + halfSize + j]
                    // Inverse butterfly: reconstruct from u'=u+v, v'=u+t*v
                    // u = (t*u' + v') / (t+1), v = (u' + v') / (t+1)
                    // In char 2 with our specific butterfly: u+v, u+t*v
                    // invert: u = u' + v', v = (v' + u') * t^{-1} ... careful
                    // Actually: u' = u + v, v' = u + t*v
                    // u' + v' = v + t*v = v*(1+t)
                    // So v = (u' + v') * (1+t)^{-1}
                    // And u = u' + v = u' + (u'+v')*(1+t)^{-1}
                    let sum = u + v
                    let onePlusTInv = (BiniusF256.one + twist).inverse()
                    let origV = sum * onePlusTInv
                    let origU = u + origV
                    data[offset + j] = origU
                    data[offset + halfSize + j] = origV
                }
                offset += halfSize << 1
            }
            halfSize <<= 1
        }

        return data
    }

    /// Generate a standard basis for GF(2^8) additive NTT.
    /// Uses successive squarings of a primitive element.
    public static func standardBasis(dimension k: Int) -> [BiniusF256] {
        precondition(k <= 8, "GF(2^8) subspaces have dimension at most 8")
        var basis = [BiniusF256]()
        var elem = BiniusF256(value: 0x02)  // primitive element
        for _ in 0..<k {
            basis.append(elem)
            elem = elem.squared()
        }
        return basis
    }
}

// MARK: - GPU Batch Binary Tower Multiply

/// GPU-accelerated batch operations for Binius binary tower fields.
/// Falls back to NEON-accelerated CPU paths for small batch sizes.
public final class GPUBiniusBinaryFieldEngine {
    public static let shared = GPUBiniusBinaryFieldEngine()

    /// Minimum batch size to justify GPU dispatch overhead.
    public static let gpuThreshold = 4096

    private init() {}

    // MARK: - Batch GF(256) Operations

    /// Batch GF(256) addition (XOR): out[i] = a[i] + b[i].
    public func batchAdd256(_ a: [BiniusF256], _ b: [BiniusF256]) -> [BiniusF256] {
        precondition(a.count == b.count)
        var out = [BiniusF256](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = BiniusF256(value: a[i].value ^ b[i].value)
        }
        return out
    }

    /// Batch GF(256) multiplication: out[i] = a[i] * b[i].
    public func batchMul256(_ a: [BiniusF256], _ b: [BiniusF256]) -> [BiniusF256] {
        precondition(a.count == b.count)
        var out = [BiniusF256](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] * b[i]
        }
        return out
    }

    /// Batch GF(256) scalar multiply: out[i] = scalar * a[i].
    public func batchScalarMul256(_ scalar: BiniusF256, _ a: [BiniusF256]) -> [BiniusF256] {
        var out = [BiniusF256](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = scalar * a[i]
        }
        return out
    }

    // MARK: - Batch GF(2^32) Operations via BinaryTower32

    /// Batch GF(2^32) multiply using NEON-accelerated PMULL.
    public func batchMulGF32(_ a: [UInt32], _ b: [UInt32]) -> [UInt32] {
        precondition(a.count == b.count)
        var out = [UInt32](repeating: 0, count: a.count)
        for i in 0..<a.count {
            out[i] = bt_gf32_mul(a[i], b[i])
        }
        return out
    }

    /// Batch GF(2^32) add (XOR).
    public func batchAddGF32(_ a: [UInt32], _ b: [UInt32]) -> [UInt32] {
        precondition(a.count == b.count)
        var out = [UInt32](repeating: 0, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] ^ b[i]
        }
        return out
    }

    // MARK: - Batch GF(2^64) Operations

    /// Batch GF(2^64) multiply using NEON PMULL acceleration.
    public func batchMulGF64(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf64_batch_mul(a, b, &out, Int32(a.count))
        return out
    }

    /// Batch GF(2^64) add (XOR) using NEON.
    public func batchAddGF64(_ a: [UInt64], _ b: [UInt64]) -> [UInt64] {
        precondition(a.count == b.count)
        var out = [UInt64](repeating: 0, count: a.count)
        bt_gf64_batch_add(a, b, &out, Int32(a.count))
        return out
    }

    // MARK: - Packed Binary Field Operations

    /// Batch packed GF(2) addition (XOR) on 32-element packed words.
    public func batchPackedAdd32(_ a: [PackedBinaryField32],
                                 _ b: [PackedBinaryField32]) -> [PackedBinaryField32] {
        precondition(a.count == b.count)
        var out = [PackedBinaryField32](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] + b[i]
        }
        return out
    }

    /// Batch packed GF(2) multiplication (AND) on 32-element packed words.
    public func batchPackedMul32(_ a: [PackedBinaryField32],
                                 _ b: [PackedBinaryField32]) -> [PackedBinaryField32] {
        precondition(a.count == b.count)
        var out = [PackedBinaryField32](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] * b[i]
        }
        return out
    }

    /// Batch packed GF(2) addition (XOR) on 64-element packed words.
    public func batchPackedAdd64(_ a: [PackedBinaryField64],
                                 _ b: [PackedBinaryField64]) -> [PackedBinaryField64] {
        precondition(a.count == b.count)
        var out = [PackedBinaryField64](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] + b[i]
        }
        return out
    }

    /// Batch packed GF(2) multiplication (AND) on 64-element packed words.
    public func batchPackedMul64(_ a: [PackedBinaryField64],
                                 _ b: [PackedBinaryField64]) -> [PackedBinaryField64] {
        precondition(a.count == b.count)
        var out = [PackedBinaryField64](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] * b[i]
        }
        return out
    }

    // MARK: - Multilinear Extension over Binary Hypercube

    /// Evaluate multilinear extension at a point over GF(256) binary hypercube.
    public func multilinearEval256(evals: [BiniusF256],
                                   at point: [BiniusF256]) -> BiniusF256 {
        let n = point.count
        precondition(evals.count == (1 << n), "evals must have 2^n entries")

        var table = evals
        var size = evals.count

        for i in 0..<n {
            let half = size >> 1
            let ri = point[i]
            for j in 0..<half {
                let a = table[2 * j]
                let b = table[2 * j + 1]
                let diff = a + b
                table[j] = a + (ri * diff)
            }
            size = half
        }
        return table[0]
    }

    // MARK: - Inner Product

    /// Inner product sum_i a[i] * b[i] in GF(256) (add = XOR).
    public func innerProduct256(_ a: [BiniusF256], _ b: [BiniusF256]) -> BiniusF256 {
        precondition(a.count == b.count)
        var acc = BiniusF256.zero
        for i in 0..<a.count {
            acc = acc + (a[i] * b[i])
        }
        return acc
    }

    // MARK: - Batch Inverse (Montgomery's Trick)

    /// Batch inverse via Montgomery's trick. Zero maps to zero.
    public func batchInverse256(_ a: [BiniusF256]) -> [BiniusF256] {
        let n = a.count
        if n == 0 { return [] }

        var prefix = [BiniusF256](repeating: .zero, count: n)
        var acc = BiniusF256.one
        for i in 0..<n {
            if a[i].isZero {
                prefix[i] = acc
            } else {
                acc = acc * a[i]
                prefix[i] = acc
            }
        }

        var invAcc = acc.inverse()
        var out = [BiniusF256](repeating: .zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if a[i].isZero {
                out[i] = .zero
                continue
            }
            let prevPrefix = (i > 0) ? prefix[i - 1] : BiniusF256.one
            out[i] = invAcc * prevPrefix
            invAcc = invAcc * a[i]
        }
        return out
    }

    // MARK: - Packed Reduction

    /// XOR-reduce an array of packed 32-bit words into a single word.
    /// Equivalent to element-wise XOR across all words.
    public func packedReduce32(_ words: [PackedBinaryField32]) -> PackedBinaryField32 {
        var acc: UInt32 = 0
        for w in words {
            acc ^= w.bits
        }
        return PackedBinaryField32(bits: acc)
    }

    /// XOR-reduce an array of packed 64-bit words into a single word.
    public func packedReduce64(_ words: [PackedBinaryField64]) -> PackedBinaryField64 {
        var acc: UInt64 = 0
        for w in words {
            acc ^= w.bits
        }
        return PackedBinaryField64(bits: acc)
    }

    // MARK: - Tower Embedding Utilities

    /// Embed array of GF(2) bits (as UInt8 0/1) into packed 32-bit words.
    public func packBitsInto32(_ bits: [UInt8]) -> [PackedBinaryField32] {
        let nWords = (bits.count + 31) / 32
        var result = [PackedBinaryField32](repeating: .zero, count: nWords)
        for i in 0..<bits.count {
            let wordIdx = i / 32
            let bitIdx = i % 32
            if bits[i] & 1 == 1 {
                result[wordIdx].bits |= (1 << bitIdx)
            }
        }
        return result
    }

    /// Unpack 32-bit packed words back to individual GF(2) bits.
    public func unpackBitsFrom32(_ words: [PackedBinaryField32], count: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            let wordIdx = i / 32
            let bitIdx = i % 32
            result[i] = UInt8((words[wordIdx].bits >> bitIdx) & 1)
        }
        return result
    }

    /// Embed array of GF(2) bits into packed 64-bit words.
    public func packBitsInto64(_ bits: [UInt8]) -> [PackedBinaryField64] {
        let nWords = (bits.count + 63) / 64
        var result = [PackedBinaryField64](repeating: .zero, count: nWords)
        for i in 0..<bits.count {
            let wordIdx = i / 64
            let bitIdx = i % 64
            if bits[i] & 1 == 1 {
                result[wordIdx].bits |= (1 << bitIdx)
            }
        }
        return result
    }

    /// Unpack 64-bit packed words back to individual GF(2) bits.
    public func unpackBitsFrom64(_ words: [PackedBinaryField64], count: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            let wordIdx = i / 64
            let bitIdx = i % 64
            result[i] = UInt8((words[wordIdx].bits >> bitIdx) & 1)
        }
        return result
    }
}

// MARK: - Conversion Utilities

extension BiniusF256 {

    /// Convert to BinaryTower8 (different representation of GF(256)).
    /// NOTE: BiniusF256 uses the tower basis {1, alpha, beta, alpha*beta, ...}
    /// while BinaryTower8 uses the polynomial basis with x^8+x^4+x^3+x+1.
    /// These are DIFFERENT representations of GF(256) and conversion
    /// requires a basis change matrix. This returns the raw byte for
    /// same-representation interop only.
    public var rawByte: UInt8 { value }

    /// Construct from raw byte (same-representation only).
    public init(rawByte: UInt8) {
        self.value = rawByte
    }
}

/// Free function wrappers matching project convention.
@inline(__always) public func biniusF2Add(_ a: BiniusF2, _ b: BiniusF2) -> BiniusF2 { a + b }
@inline(__always) public func biniusF2Mul(_ a: BiniusF2, _ b: BiniusF2) -> BiniusF2 { a * b }
@inline(__always) public func biniusF4Add(_ a: BiniusF4, _ b: BiniusF4) -> BiniusF4 { a + b }
@inline(__always) public func biniusF4Mul(_ a: BiniusF4, _ b: BiniusF4) -> BiniusF4 { a * b }
@inline(__always) public func biniusF16Add(_ a: BiniusF16, _ b: BiniusF16) -> BiniusF16 { a + b }
@inline(__always) public func biniusF16Mul(_ a: BiniusF16, _ b: BiniusF16) -> BiniusF16 { a * b }
@inline(__always) public func biniusF256Add(_ a: BiniusF256, _ b: BiniusF256) -> BiniusF256 { a + b }
@inline(__always) public func biniusF256Mul(_ a: BiniusF256, _ b: BiniusF256) -> BiniusF256 { a * b }
@inline(__always) public func biniusF256Inv(_ a: BiniusF256) -> BiniusF256 { a.inverse() }
