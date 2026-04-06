// GPU-accelerated binary tower field engine for Binius-style proofs.
//
// Binary tower extension chain:
//   GF(2) -> GF(2^2) -> GF(2^4) -> GF(2^8) -> GF(2^16) -> GF(2^32)
//   -> GF(2^64) -> GF(2^128)
//
// Each level doubles the extension degree:
//   GF(2^{2k}) = GF(2^k)[X] / (X^2 + X + alpha_k)
//
// Standalone BiniusTower128 field type with 4 x UInt32 limbs for
// Binius-compatible packed representation. Provides:
//   - Karatsuba multiplication at each tower level
//   - Tower lifting (embed subfield into extension)
//   - Tower projection (extract subfield component)
//   - Multilinear extension evaluation over binary tower
//   - Batch operations for GPU/SIMD parallelism
//
// Addition = XOR (free), subtraction = XOR (characteristic 2).
// Uses NEON/PMULL acceleration via NeonFieldOps for the core multiply.

import Foundation
import NeonFieldOps

// MARK: - BiniusTower128 — 4 x UInt32 limb representation

/// GF(2^128) element in Binius-compatible packed representation.
///
/// Stored as 4 x UInt32 limbs: [w0, w1, w2, w3] where
///   value = w0 + w1*2^32 + w2*2^64 + w3*2^96
///
/// This matches the Binius packed field layout where each 32-bit word
/// corresponds to a GF(2^32) subfield element in the tower decomposition.
///
/// The tower structure is:
///   w0, w1 form the lo GF(2^64) half
///   w2, w3 form the hi GF(2^64) half
///   GF(2^128) = GF(2^64)[X] / (X^2 + X + delta)
public struct BiniusTower128: Equatable, CustomStringConvertible {
    public var w0: UInt32
    public var w1: UInt32
    public var w2: UInt32
    public var w3: UInt32

    public static let zero = BiniusTower128(w0: 0, w1: 0, w2: 0, w3: 0)
    public static let one  = BiniusTower128(w0: 1, w1: 0, w2: 0, w3: 0)

    public init(w0: UInt32, w1: UInt32, w2: UInt32, w3: UInt32) {
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    }

    /// Construct from lo/hi UInt64 pair (for interop with BinaryTower128).
    public init(lo: UInt64, hi: UInt64) {
        self.w0 = UInt32(lo & 0xFFFFFFFF)
        self.w1 = UInt32(lo >> 32)
        self.w2 = UInt32(hi & 0xFFFFFFFF)
        self.w3 = UInt32(hi >> 32)
    }

    /// Construct from a single UInt32 (embeds into lowest limb).
    public init(value: UInt32) {
        self.w0 = value
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
    }

    /// Embed a GF(2^8) byte into the tower.
    public init(fromGF8 value: UInt8) {
        self.w0 = UInt32(value)
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
    }

    public var isZero: Bool { w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0 }

    /// Extract lo UInt64 (w0 + w1 << 32).
    public var lo: UInt64 { UInt64(w0) | (UInt64(w1) << 32) }

    /// Extract hi UInt64 (w2 + w3 << 32).
    public var hi: UInt64 { UInt64(w2) | (UInt64(w3) << 32) }

    /// Project to GF(2^8) subfield (lowest byte of w0).
    public var toGF8: UInt8 { UInt8(w0 & 0xFF) }

    /// Project to GF(2^32) subfield (w0 only, valid if upper limbs are zero).
    public var toGF32: UInt32 { w0 }

    public var description: String {
        let hex3 = String(w3, radix: 16, uppercase: true)
        let hex2 = String(w2, radix: 16, uppercase: true)
        let hex1 = String(w1, radix: 16, uppercase: true)
        let hex0 = String(w0, radix: 16, uppercase: true)
        return "BT128(\(hex3)_\(hex2)_\(hex1)_\(hex0))"
    }

    // MARK: - Arithmetic

    /// Addition = XOR (characteristic 2).
    @inline(__always)
    public static func + (a: BiniusTower128, b: BiniusTower128) -> BiniusTower128 {
        BiniusTower128(w0: a.w0 ^ b.w0, w1: a.w1 ^ b.w1,
                       w2: a.w2 ^ b.w2, w3: a.w3 ^ b.w3)
    }

    /// Subtraction = XOR (same as addition in characteristic 2).
    @inline(__always)
    public static func - (a: BiniusTower128, b: BiniusTower128) -> BiniusTower128 {
        a + b
    }

    /// Multiplication via NEON/PMULL-accelerated GF(2^128) multiply.
    /// Delegates to bt_gf128_mul from NeonFieldOps (Karatsuba + Barrett reduction).
    @inline(__always)
    public static func * (a: BiniusTower128, b: BiniusTower128) -> BiniusTower128 {
        var aArr: [UInt64] = [a.lo, a.hi]
        var bArr: [UInt64] = [b.lo, b.hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_mul(&aArr, &bArr, &rArr)
        return BiniusTower128(lo: rArr[0], hi: rArr[1])
    }

    /// Squaring via NEON/PMULL (Frobenius endomorphism is linear in char 2).
    @inline(__always)
    public func squared() -> BiniusTower128 {
        var aArr: [UInt64] = [lo, hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_sqr(&aArr, &rArr)
        return BiniusTower128(lo: rArr[0], hi: rArr[1])
    }

    /// Multiplicative inverse via Itoh-Tsujii in C.
    public func inverse() -> BiniusTower128 {
        precondition(!isZero, "Cannot invert zero in GF(2^128)")
        var aArr: [UInt64] = [lo, hi]
        var rArr: [UInt64] = [0, 0]
        bt_gf128_inv(&aArr, &rArr)
        return BiniusTower128(lo: rArr[0], hi: rArr[1])
    }

    /// Exponentiation via square-and-multiply.
    public func pow(_ n: Int) -> BiniusTower128 {
        if n == 0 { return .one }
        var result = BiniusTower128.one
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

// MARK: - Tower Lifting and Projection

/// Tower lifting and projection operations for Binius packed field elements.
///
/// Lifting: embed a smaller subfield element into a larger tower level.
/// Projection: extract the lo/hi subfield components from a tower element.
///
/// These are fundamental for Binius multilinear evaluation and sumcheck.
public enum BiniusTowerOps {

    // MARK: - Lifting (embed subfield into extension)

    /// Lift GF(2^32) into GF(2^128) (places value in w0, zeros above).
    @inline(__always)
    public static func liftGF32(_ x: UInt32) -> BiniusTower128 {
        BiniusTower128(w0: x, w1: 0, w2: 0, w3: 0)
    }

    /// Lift GF(2^64) into GF(2^128) (places value in lo half, zeros hi).
    @inline(__always)
    public static func liftGF64(_ x: UInt64) -> BiniusTower128 {
        BiniusTower128(lo: x, hi: 0)
    }

    /// Lift GF(2^8) into GF(2^128).
    @inline(__always)
    public static func liftGF8(_ x: UInt8) -> BiniusTower128 {
        BiniusTower128(fromGF8: x)
    }

    // MARK: - Projection (extract subfield component)

    /// Project to GF(2^64) lo half.
    @inline(__always)
    public static func projectLo64(_ x: BiniusTower128) -> UInt64 {
        x.lo
    }

    /// Project to GF(2^64) hi half.
    @inline(__always)
    public static func projectHi64(_ x: BiniusTower128) -> UInt64 {
        x.hi
    }

    /// Project to GF(2^32) — returns w0 (lowest 32-bit word).
    @inline(__always)
    public static func projectGF32(_ x: BiniusTower128) -> UInt32 {
        x.w0
    }

    /// Check if element is in GF(2^8) subfield.
    @inline(__always)
    public static func isInGF8(_ x: BiniusTower128) -> Bool {
        x.w0 <= 0xFF && x.w1 == 0 && x.w2 == 0 && x.w3 == 0
    }

    /// Check if element is in GF(2^32) subfield.
    @inline(__always)
    public static func isInGF32(_ x: BiniusTower128) -> Bool {
        x.w1 == 0 && x.w2 == 0 && x.w3 == 0
    }

    /// Check if element is in GF(2^64) subfield.
    @inline(__always)
    public static func isInGF64(_ x: BiniusTower128) -> Bool {
        x.w2 == 0 && x.w3 == 0
    }

    // MARK: - Tower decompose / recompose

    /// Decompose GF(2^128) element into two GF(2^64) tower halves.
    /// In the tower GF(2^128) = GF(2^64)[X]/(X^2+X+delta),
    /// an element a = a_lo + a_hi * X.
    @inline(__always)
    public static func decompose128(_ x: BiniusTower128) -> (lo: UInt64, hi: UInt64) {
        (x.lo, x.hi)
    }

    /// Recompose from two GF(2^64) halves.
    @inline(__always)
    public static func recompose128(lo: UInt64, hi: UInt64) -> BiniusTower128 {
        BiniusTower128(lo: lo, hi: hi)
    }

    /// Decompose GF(2^64) into two GF(2^32) halves (w0, w1).
    @inline(__always)
    public static func decompose64(_ x: UInt64) -> (lo: UInt32, hi: UInt32) {
        (UInt32(x & 0xFFFFFFFF), UInt32(x >> 32))
    }

    /// Recompose GF(2^64) from two GF(2^32) halves.
    @inline(__always)
    public static func recompose64(lo: UInt32, hi: UInt32) -> UInt64 {
        UInt64(lo) | (UInt64(hi) << 32)
    }
}

// MARK: - Multilinear Extension over Binary Tower

/// Multilinear extension evaluation over Binius binary tower fields.
///
/// Given f: {0,1}^n -> GF(2^128) as evaluations on the boolean hypercube,
/// compute the unique multilinear extension MLE(f) at an arbitrary point
/// r = (r_0, ..., r_{n-1}) in GF(2^128)^n.
///
/// Uses the standard streaming evaluation:
///   At each variable i, fold the table by half:
///     table[j] = table[2j] + r_i * (table[2j+1] + table[2j])
///   (addition is XOR in char 2)
///
/// This is the core primitive for Binius sumcheck and PCS evaluation.
public enum BiniusMultilinear {

    /// Evaluate multilinear extension at point r.
    ///
    /// - Parameters:
    ///   - evals: Array of 2^n field element evaluations on {0,1}^n
    ///   - point: Array of n challenge coordinates in GF(2^128)
    /// - Returns: MLE(evals)(point)
    public static func evaluate(evals: [BiniusTower128],
                                at point: [BiniusTower128]) -> BiniusTower128 {
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
                // table[j] = a + ri * (b + a)  [add = XOR in char 2]
                let diff = a + b
                table[j] = a + (ri * diff)
            }
            size = half
        }

        return table[0]
    }

    /// Evaluate multilinear extension at point, working with flat UInt64 buffers.
    /// Each element is 2 x UInt64 (lo, hi) packed contiguously.
    ///
    /// This is the batch-friendly version for GPU dispatch.
    public static func evaluateFlat(evals: [UInt64],
                                    at point: [BiniusTower128]) -> BiniusTower128 {
        let n = point.count
        let elemCount = 1 << n
        precondition(evals.count == elemCount * 2, "evals must have 2^(n+1) UInt64s")

        // Convert flat buffer to BiniusTower128
        var table = [BiniusTower128](repeating: .zero, count: elemCount)
        for i in 0..<elemCount {
            table[i] = BiniusTower128(lo: evals[2 * i], hi: evals[2 * i + 1])
        }

        return evaluate(evals: table, at: point)
    }
}

// MARK: - Batch Operations

/// Batch binary tower field operations for SIMD/GPU parallelism.
///
/// Element-wise operations on arrays of BiniusTower128 elements,
/// using NEON-accelerated GF(2^128) arithmetic from NeonFieldOps.
public enum BiniusTowerBatch {

    /// Batch add (XOR): out[i] = a[i] + b[i].
    public static func add(_ a: [BiniusTower128], _ b: [BiniusTower128]) -> [BiniusTower128] {
        precondition(a.count == b.count)
        var out = [BiniusTower128](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i] + b[i]
        }
        return out
    }

    /// Batch multiply: out[i] = a[i] * b[i].
    /// Uses NEON/PMULL-accelerated GF(2^128) multiply via NeonFieldOps.
    public static func mul(_ a: [BiniusTower128], _ b: [BiniusTower128]) -> [BiniusTower128] {
        precondition(a.count == b.count)
        let n = a.count
        // Pack into flat UInt64 buffers for bt_gf128_batch_mul
        var aFlat = [UInt64](repeating: 0, count: n * 2)
        var bFlat = [UInt64](repeating: 0, count: n * 2)
        for i in 0..<n {
            aFlat[2 * i] = a[i].lo
            aFlat[2 * i + 1] = a[i].hi
            bFlat[2 * i] = b[i].lo
            bFlat[2 * i + 1] = b[i].hi
        }
        var rFlat = [UInt64](repeating: 0, count: n * 2)
        bt_gf128_batch_mul(aFlat, bFlat, &rFlat, Int32(n))

        var out = [BiniusTower128](repeating: .zero, count: n)
        for i in 0..<n {
            out[i] = BiniusTower128(lo: rFlat[2 * i], hi: rFlat[2 * i + 1])
        }
        return out
    }

    /// Batch scalar multiply: out[i] = scalar * a[i].
    public static func scalarMul(_ scalar: BiniusTower128,
                                 _ a: [BiniusTower128]) -> [BiniusTower128] {
        let n = a.count
        let scalars = [BiniusTower128](repeating: scalar, count: n)
        return mul(scalars, a)
    }

    /// Batch inverse via Montgomery's trick: N inverses using 1 inversion + 3(N-1) multiplies.
    /// Zero elements map to zero (standard ZK convention).
    public static func batchInverse(_ a: [BiniusTower128]) -> [BiniusTower128] {
        let n = a.count
        if n == 0 { return [] }

        // Compute prefix products, skipping zeros
        var prefix = [BiniusTower128](repeating: .zero, count: n)
        var acc = BiniusTower128.one
        for i in 0..<n {
            if a[i].isZero {
                prefix[i] = acc
            } else {
                acc = acc * a[i]
                prefix[i] = acc
            }
        }

        // Single inversion of accumulated product
        var invAcc = acc.inverse()

        // Sweep backwards to recover individual inverses
        var out = [BiniusTower128](repeating: .zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if a[i].isZero {
                out[i] = .zero
                continue
            }
            let prevPrefix = (i > 0) ? prefix[i - 1] : BiniusTower128.one
            out[i] = invAcc * prevPrefix
            invAcc = invAcc * a[i]
        }
        return out
    }

    /// Batch square: out[i] = a[i]^2.
    public static func batchSquare(_ a: [BiniusTower128]) -> [BiniusTower128] {
        var out = [BiniusTower128](repeating: .zero, count: a.count)
        for i in 0..<a.count {
            out[i] = a[i].squared()
        }
        return out
    }

    /// Inner product: sum_i a[i] * b[i] (addition = XOR).
    public static func innerProduct(_ a: [BiniusTower128],
                                    _ b: [BiniusTower128]) -> BiniusTower128 {
        precondition(a.count == b.count)
        let products = mul(a, b)
        var acc = BiniusTower128.zero
        for p in products {
            acc = acc + p
        }
        return acc
    }
}

// MARK: - Conversion between BiniusTower128 and BinaryTower128

extension BiniusTower128 {

    /// Convert to BinaryTower128 (flat polynomial representation).
    public func toBinaryTower128() -> BinaryTower128 {
        BinaryTower128(lo: lo, hi: hi)
    }

    /// Construct from BinaryTower128.
    public init(from bt: BinaryTower128) {
        self.init(lo: bt.lo, hi: bt.hi)
    }
}
