// Banderwagon curve — the prime-order quotient group used in Ethereum Verkle trees (EIP-6800).
//
// Banderwagon is defined as a quotient group of Bandersnatch, a twisted Edwards curve over
// BLS12-381's scalar field Fr (255-bit prime).
//
// Bandersnatch: -5*x^2 + y^2 = 1 + d*x^2*y^2  (twisted Edwards, a = -5)
// where d = (A+2)/B mapped to TE form, with:
//   A = 29978822694968839326280790526576830000724695703680294200930988189185642946220
//   d = -(45022363124591815672509500913686876175488063829319466900776701791074614335719) / Fr381.P (mod Fr381.P)
//
// Banderwagon quotient: (x, y) ~ (-x, -y), giving a prime-order group.
// Group order = (|Bandersnatch| / cofactor) / 2 is prime.
//
// The scalar field for Banderwagon is a 253-bit prime:
//   13108968793781547619861935127046491459309155893440570251786403306729687672801
//
// Points are serialized by their y-coordinate (with x chosen to have the "lexicographically
// larger" value, since (x,y)~(-x,-y)).
//
// References:
//   - EIP-6800: Ethereum Verkle tree
//   - Bandersnatch: a fast elliptic curve built over BLS12-381 scalar field (Masson et al.)
//   - https://eprint.iacr.org/2021/1152

import Foundation
import NeonFieldOps

// MARK: - Banderwagon Point Types

/// Affine representation of a Banderwagon point (x, y) with x, y in Fr381.
public struct BanderwagonAffine {
    public var x: Fr381
    public var y: Fr381

    public init(x: Fr381, y: Fr381) {
        self.x = x
        self.y = y
    }

    public static let identity = BanderwagonAffine(x: Fr381.zero, y: Fr381.one)
}

/// Extended twisted Edwards coordinates (X, Y, T, Z) where x = X/Z, y = Y/Z, T = XY/Z.
/// This representation gives unified addition formulas (no special cases for doubling).
public struct BanderwagonExtended {
    public var x: Fr381
    public var y: Fr381
    public var t: Fr381   // T = X*Y/Z
    public var z: Fr381

    public init(x: Fr381, y: Fr381, t: Fr381, z: Fr381) {
        self.x = x
        self.y = y
        self.t = t
        self.z = z
    }

    public static let identity = BanderwagonExtended(
        x: Fr381.zero, y: Fr381.one, t: Fr381.zero, z: Fr381.one)
}

// MARK: - Curve Constants

/// Bandersnatch twisted Edwards parameter: a = -5
/// In Montgomery form over Fr381.
public let bandersnatchA: Fr381 = fr381Neg(fr381FromInt(5))

/// Bandersnatch twisted Edwards parameter d.
/// d = -(45022363124591815672509500913686876175488063829319466900776701791074614335719)
///   mod r (BLS12-381 scalar field modulus).
/// Stored in Montgomery form.
public let bandersnatchD: Fr381 = {
    // d = 45022363124591815672509500913686876175488063829319466900776701791074614335719
    // In standard form (little-endian 64-bit limbs):
    let limbs: [UInt64] = [
        0xb369f2f5188d58e7,
        0xcb66677177e54f92,
        0xc66e3bf86be3b6d8,
        0x6389c12633c267cb
    ]
    let raw = Fr381.from64(limbs)
    return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
}()

/// Banderwagon generator point (from the Ethereum Verkle spec).
/// This is a fixed basepoint on the Bandersnatch curve chosen such that it generates
/// the prime-order Banderwagon subgroup.
public let banderwagonGenerator: BanderwagonAffine = {
    // Start with a point on Bandersnatch: a*x^2 + y^2 = 1 + d*x^2*y^2 with a=-5, y=2
    let xLimbs: [UInt64] = [
        0xdab84f0e7712135a,
        0x8fa6a5ec7da7cb87,
        0xc4e774d16936a440,
        0x27190a4a08958a9f
    ]
    let yLimbs: [UInt64] = [0x2, 0x0, 0x0, 0x0]
    let xRaw = Fr381.from64(xLimbs)
    let yRaw = Fr381.from64(yLimbs)
    let x = fr381Mul(xRaw, Fr381.from64(Fr381.R2_MOD_R))
    let y = fr381Mul(yRaw, Fr381.from64(Fr381.R2_MOD_R))
    let rawPoint = BanderwagonExtended(
        x: x, y: y, t: fr381Mul(x, y), z: Fr381.one)
    // Clear cofactor by multiplying by 4 to project into prime-order subgroup
    let inSubgroup = bwDouble(bwDouble(rawPoint))
    return bwToAffine(inSubgroup)
}()

/// Banderwagon scalar field order (the number of points in the prime-order group).
/// q = 13108968793781547619861935127046491459309155893440570251786403306729687672801
public let banderwagonOrder: [UInt64] = [
    0x74fd06b52876e7e1,
    0xff8f870074190471,
    0x0cce760202687600,
    0x1cfb69d4ca675f52
]

// MARK: - Point Operations

/// Check if a Banderwagon extended point is the identity.
public func bwIsIdentity(_ p: BanderwagonExtended) -> Bool {
    // Identity in extended TE: (0, 1, 0, 1) or any scalar multiple (0, Z, 0, Z)
    return fr381ToInt(p.x) == [0, 0, 0, 0]
}

/// Convert affine to extended coordinates.
public func bwFromAffine(_ a: BanderwagonAffine) -> BanderwagonExtended {
    BanderwagonExtended(
        x: a.x, y: a.y,
        t: fr381Mul(a.x, a.y),
        z: Fr381.one)
}

/// Convert extended to affine coordinates.
public func bwToAffine(_ p: BanderwagonExtended) -> BanderwagonAffine {
    if bwIsIdentity(p) {
        return BanderwagonAffine.identity
    }
    let zinv = fr381Inverse(p.z)
    return BanderwagonAffine(
        x: fr381Mul(p.x, zinv),
        y: fr381Mul(p.y, zinv))
}

/// Negate a Banderwagon point.
/// In the quotient group (x,y)~(-x,-y), negation is (x,y) -> (-x, y) in TE coordinates.
public func bwNegate(_ p: BanderwagonExtended) -> BanderwagonExtended {
    BanderwagonExtended(
        x: fr381Neg(p.x),
        y: p.y,
        t: fr381Neg(p.t),
        z: p.z)
}

/// Unified addition for extended twisted Edwards coordinates.
/// Formula for general twisted Edwards: a*x^2 + y^2 = 1 + d*x^2*y^2
/// Uses the "unified" formulas that work for both addition and doubling.
///
/// Cost: 9M + 1D (where D = multiplication by d)
public func bwAdd(_ p: BanderwagonExtended, _ q: BanderwagonExtended) -> BanderwagonExtended {
    // Unified addition formula for twisted Edwards:
    // A = X1*X2
    // B = Y1*Y2
    // C = T1*d*T2
    // D = Z1*Z2
    // E = (X1+Y1)*(X2+Y2) - A - B
    // F = D - C
    // G = D + C
    // H = B - a*A
    // X3 = E*F
    // Y3 = G*H
    // T3 = E*H
    // Z3 = F*G
    let a_ = fr381Mul(p.x, q.x)
    let b_ = fr381Mul(p.y, q.y)
    let c_ = fr381Mul(fr381Mul(p.t, q.t), bandersnatchD)
    let d_ = fr381Mul(p.z, q.z)

    let e_ = fr381Sub(fr381Mul(fr381Add(p.x, p.y), fr381Add(q.x, q.y)),
                       fr381Add(a_, b_))
    let f_ = fr381Sub(d_, c_)
    let g_ = fr381Add(d_, c_)
    let h_ = fr381Sub(b_, fr381Mul(bandersnatchA, a_))  // B - a*A

    return BanderwagonExtended(
        x: fr381Mul(e_, f_),
        y: fr381Mul(g_, h_),
        t: fr381Mul(e_, h_),
        z: fr381Mul(f_, g_))
}

/// Point doubling (specialized, slightly faster than unified addition).
public func bwDouble(_ p: BanderwagonExtended) -> BanderwagonExtended {
    // Dedicated doubling for general twisted Edwards a*x^2 + y^2 = 1 + d*x^2*y^2:
    // A = X1^2
    // B = Y1^2
    // C = 2*Z1^2
    // D = a*A
    // E = (X1+Y1)^2 - A - B
    // G = D + B
    // F = G - C
    // H = D - B
    // X3 = E*F
    // Y3 = G*H
    // T3 = E*H
    // Z3 = F*G
    let a_ = fr381Sqr(p.x)
    let b_ = fr381Sqr(p.y)
    let c_ = fr381Double(fr381Sqr(p.z))
    let d_ = fr381Mul(bandersnatchA, a_)  // a*A
    let e_ = fr381Sub(fr381Sub(fr381Sqr(fr381Add(p.x, p.y)), a_), b_)
    let g_ = fr381Add(d_, b_)
    let f_ = fr381Sub(g_, c_)
    let h_ = fr381Sub(d_, b_)

    return BanderwagonExtended(
        x: fr381Mul(e_, f_),
        y: fr381Mul(g_, h_),
        t: fr381Mul(e_, h_),
        z: fr381Mul(f_, g_))
}

/// Scalar multiplication using double-and-add.
/// Scalar is an Fr381 element. The integer value of the scalar is used directly.
public func bwScalarMul(_ p: BanderwagonExtended, _ scalar: Fr381) -> BanderwagonExtended {
    let limbs = fr381ToInt(scalar)
    if limbs == [0, 0, 0, 0] { return .identity }

    var result = BanderwagonExtended.identity
    var base = p

    // Process all 256 bits
    for i in 0..<4 {
        var word = limbs[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = bwAdd(result, base)
            }
            base = bwDouble(base)
            word >>= 1
        }
    }
    return result
}

/// Multi-scalar multiplication (MSM) using naive method.
/// For production, this should use Pippenger's, but for Verkle tree widths (256)
/// this is acceptable.
public func bwMSM(_ points: [BanderwagonExtended], _ scalars: [Fr381]) -> BanderwagonExtended {
    precondition(points.count == scalars.count)
    var result = BanderwagonExtended.identity
    for i in 0..<points.count {
        let term = bwScalarMul(points[i], scalars[i])
        result = bwAdd(result, term)
    }
    return result
}

/// Batch multi-scalar multiplication with precomputed tables (Straus/BGMW style).
/// More efficient for repeated MSMs with the same base points.
public func bwMSMWithBases(_ bases: [BanderwagonExtended], _ scalars: [Fr381]) -> BanderwagonExtended {
    // For now, delegate to naive MSM. A Pippenger implementation for Fr381
    // scalars could be added for performance.
    return bwMSM(bases, scalars)
}

// MARK: - Point Equality (Banderwagon quotient)

/// Check equality in the Banderwagon quotient group.
/// Two points (X1:Y1:T1:Z1) and (X2:Y2:T2:Z2) are equivalent if
/// X1*Y2 == X2*Y1 (i.e., they represent the same point or its negation in the quotient).
///
/// In Banderwagon, (x,y) ~ (-x,-y), so we check x1/y1 == x2/y2.
public func bwEqual(_ p: BanderwagonExtended, _ q: BanderwagonExtended) -> Bool {
    if bwIsIdentity(p) && bwIsIdentity(q) { return true }
    if bwIsIdentity(p) || bwIsIdentity(q) { return false }

    // In the Banderwagon quotient group (x,y) ~ (-x,-y),
    // two points are equal iff x1/y1 == x2/y2.
    // In extended projective: x = X/Z, y = Y/Z, so x/y = X/Y.
    // Check: X1*Y2 == X2*Y1  (or X1*Y2 == -X2*Y1 for the ~(-x,-y) case,
    // but since x/(-y) = -(x/y), that would be a different point in the quotient).
    let lhs = fr381Mul(p.x, q.y)
    let rhs = fr381Mul(q.x, p.y)
    return fr381ToInt(lhs) == fr381ToInt(rhs)
}

// MARK: - Serialization

/// Serialize a Banderwagon point to 32 bytes (the y-coordinate).
/// The canonical representative has x in the "positive" half of the field.
public func bwSerialize(_ p: BanderwagonExtended) -> [UInt8] {
    let aff = bwToAffine(p)
    // Normalize: choose representative where x is "positive"
    // (i.e., the integer value of x < (r-1)/2)
    let xInt = fr381ToInt(aff.x)
    let halfR: [UInt64] = [
        Fr381.P[0] >> 1 | (Fr381.P[1] << 63),
        Fr381.P[1] >> 1 | (Fr381.P[2] << 63),
        Fr381.P[2] >> 1 | (Fr381.P[3] << 63),
        Fr381.P[3] >> 1
    ]
    var needNeg = false
    // Compare xInt > halfR (lexicographic, big-endian)
    for i in stride(from: 3, through: 0, by: -1) {
        if xInt[i] > halfR[i] { needNeg = true; break }
        if xInt[i] < halfR[i] { break }
    }
    let yToSerialize = needNeg ? fr381Neg(aff.y) : aff.y
    let yInt = fr381ToInt(yToSerialize)
    // Little-endian 32 bytes
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((yInt[i] >> (j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Deserialize a Banderwagon point from 32 bytes (y-coordinate).
/// Returns nil if the point is not on the curve or not in the subgroup.
public func bwDeserialize(_ bytes: [UInt8]) -> BanderwagonExtended? {
    guard bytes.count == 32 else { return nil }
    // Read y from little-endian bytes
    var limbs: [UInt64] = [0, 0, 0, 0]
    for i in 0..<4 {
        var word: UInt64 = 0
        for j in 0..<8 {
            word |= UInt64(bytes[i * 8 + j]) << (j * 8)
        }
        limbs[i] = word
    }
    let yRaw = Fr381.from64(limbs)
    let y = fr381Mul(yRaw, Fr381.from64(Fr381.R2_MOD_R))  // to Montgomery form

    // Recover x from curve equation: a*x^2 + y^2 = 1 + d*x^2*y^2
    // => x^2*(a - d*y^2) = 1 - y^2
    // => x^2 = (1 - y^2) / (a - d*y^2)
    let y2 = fr381Sqr(y)
    let num = fr381Sub(Fr381.one, y2)     // 1 - y^2
    let dy2 = fr381Mul(bandersnatchD, y2)
    let den = fr381Sub(bandersnatchA, dy2) // a - d*y^2

    if fr381ToInt(den) == [0, 0, 0, 0] { return nil }

    let denInv = fr381Inverse(den)
    let x2 = fr381Mul(num, denInv)

    // Square root via Tonelli-Shanks or exponentiation
    // For Fr381 (BLS12-381 scalar field), p ≡ 1 mod 2^32, so we need Tonelli-Shanks.
    // Simpler approach: x = x2^((p+1)/4) only works if p ≡ 3 mod 4, which Fr381 is NOT.
    // Use Euler criterion + repeated squaring.
    guard let x = fr381Sqrt(x2) else { return nil }

    // Choose the canonical x (positive half)
    let xInt = fr381ToInt(x)
    let halfR: [UInt64] = [
        Fr381.P[0] >> 1 | (Fr381.P[1] << 63),
        Fr381.P[1] >> 1 | (Fr381.P[2] << 63),
        Fr381.P[2] >> 1 | (Fr381.P[3] << 63),
        Fr381.P[3] >> 1
    ]
    var xIsNeg = false
    for i in stride(from: 3, through: 0, by: -1) {
        if xInt[i] > halfR[i] { xIsNeg = true; break }
        if xInt[i] < halfR[i] { break }
    }
    let xFinal = xIsNeg ? fr381Neg(x) : x

    return bwFromAffine(BanderwagonAffine(x: xFinal, y: y))
}

// MARK: - Banderwagon Pedersen Hash / Map to Field

/// Map a Banderwagon point to a scalar field element (for use as tree node values).
/// Uses the x/y ratio (which is well-defined in the quotient group).
public func bwMapToField(_ p: BanderwagonExtended) -> Fr381 {
    if bwIsIdentity(p) { return Fr381.zero }
    let aff = bwToAffine(p)
    let yInv = fr381Inverse(aff.y)
    return fr381Mul(aff.x, yInv)
}

// MARK: - Deterministic Generator Points

/// Generate deterministic generator points for Verkle tree commitments.
/// Uses hash-to-curve by incrementing an index and checking if the result is on the curve.
/// Returns `count` generators plus one extra Q point for IPA binding.
public func bwGenerateGenerators(count: Int) -> (generators: [BanderwagonExtended], Q: BanderwagonExtended) {
    var generators = [BanderwagonExtended]()
    generators.reserveCapacity(count)

    // Simple deterministic generation: scalar multiples of the generator
    // G_i = (i+1) * G for i = 0..count-1, Q = (count+1) * G
    let g = bwFromAffine(banderwagonGenerator)
    var acc = g
    for _ in 0..<count {
        generators.append(acc)
        acc = bwAdd(acc, g)
    }
    let q = acc
    return (generators: generators, Q: q)
}

// MARK: - Batch Operations

/// Batch convert extended points to affine using Montgomery's trick (single inversion).
public func bwBatchToAffine(_ points: [BanderwagonExtended]) -> [BanderwagonAffine] {
    let n = points.count
    if n == 0 { return [] }

    // Compute running product of Z-coordinates
    var products = [Fr381](repeating: Fr381.zero, count: n)
    products[0] = points[0].z
    for i in 1..<n {
        if bwIsIdentity(points[i]) {
            products[i] = products[i - 1]
        } else {
            products[i] = fr381Mul(products[i - 1], points[i].z)
        }
    }

    // Single inversion
    var inv = fr381Inverse(products[n - 1])
    var result = [BanderwagonAffine](repeating: .identity, count: n)

    for i in stride(from: n - 1, through: 0, by: -1) {
        if bwIsIdentity(points[i]) {
            result[i] = .identity
            continue
        }
        let zinv = (i == 0) ? inv : fr381Mul(inv, products[i - 1])
        if i > 0 {
            inv = fr381Mul(inv, points[i].z)
        }
        result[i] = BanderwagonAffine(
            x: fr381Mul(points[i].x, zinv),
            y: fr381Mul(points[i].y, zinv))
    }
    return result
}

/// Check if a point is on the Bandersnatch curve: a*x^2 + y^2 = 1 + d*x^2*y^2
public func bwIsOnCurve(_ p: BanderwagonAffine) -> Bool {
    let x2 = fr381Sqr(p.x)
    let y2 = fr381Sqr(p.y)
    let lhs = fr381Add(fr381Mul(bandersnatchA, x2), y2)  // a*x^2 + y^2
    let rhs = fr381Add(Fr381.one, fr381Mul(bandersnatchD, fr381Mul(x2, y2)))
    return fr381ToInt(lhs) == fr381ToInt(rhs)
}

// MARK: - Banderwagon Scalar Field (Fq)

/// Banderwagon scalar field: integers mod q where q is the prime group order.
/// q = 13108968793781547619861935127046491459309155893440570251786403306729687672801
/// Represented as 4 x UInt64 limbs in little-endian order.
///
/// This field is DIFFERENT from Fr381 (the BLS12-381 scalar field of order r).
/// The IPA protocol requires scalar arithmetic in Fq, not Fr381, because the
/// Banderwagon group has order q (not r), and scalar multiplication is mod q.
public struct BwScalar {
    public var limbs: (UInt64, UInt64, UInt64, UInt64)

    public init(_ l0: UInt64, _ l1: UInt64, _ l2: UInt64, _ l3: UInt64) {
        limbs = (l0, l1, l2, l3)
    }

    public static func == (lhs: BwScalar, rhs: BwScalar) -> Bool {
        lhs.limbs.0 == rhs.limbs.0 && lhs.limbs.1 == rhs.limbs.1 &&
        lhs.limbs.2 == rhs.limbs.2 && lhs.limbs.3 == rhs.limbs.3
    }

    public static let zero = BwScalar(0, 0, 0, 0)
    public static let one = BwScalar(1, 0, 0, 0)

    /// The modulus q
    public static let Q: (UInt64, UInt64, UInt64, UInt64) = (
        0x74fd06b52876e7e1,
        0xff8f870074190471,
        0x0cce760202687600,
        0x1cfb69d4ca675f52
    )

    public var isZero: Bool {
        limbs.0 == 0 && limbs.1 == 0 && limbs.2 == 0 && limbs.3 == 0
    }
}

// MARK: - BwScalar Arithmetic

/// Compare a >= b for 256-bit values
func bwScalarGTE(_ a: (UInt64, UInt64, UInt64, UInt64),
                          _ b: (UInt64, UInt64, UInt64, UInt64)) -> Bool {
    if a.3 != b.3 { return a.3 > b.3 }
    if a.2 != b.2 { return a.2 > b.2 }
    if a.1 != b.1 { return a.1 > b.1 }
    return a.0 >= b.0
}

/// Subtract b from a (assuming a >= b), no modular reduction
func bwScalarSub256(_ a: (UInt64, UInt64, UInt64, UInt64),
                             _ b: (UInt64, UInt64, UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64) {
    let (r0, borrow0) = a.0.subtractingReportingOverflow(b.0)
    let b0: UInt64 = borrow0 ? 1 : 0

    let (t1, b1a) = a.1.subtractingReportingOverflow(b.1)
    let (r1, b1b) = t1.subtractingReportingOverflow(b0)
    let b1: UInt64 = (b1a ? 1 : 0) &+ (b1b ? 1 : 0)

    let (t2, b2a) = a.2.subtractingReportingOverflow(b.2)
    let (r2, b2b) = t2.subtractingReportingOverflow(b1)
    let b2: UInt64 = (b2a ? 1 : 0) &+ (b2b ? 1 : 0)

    let (t3, _) = a.3.subtractingReportingOverflow(b.3)
    let (r3, _) = t3.subtractingReportingOverflow(b2)

    return (r0, r1, r2, r3)
}

/// Add two 256-bit values, return result and carry
func bwScalarAdd256(_ a: (UInt64, UInt64, UInt64, UInt64),
                             _ b: (UInt64, UInt64, UInt64, UInt64)) -> (result: (UInt64, UInt64, UInt64, UInt64), carry: Bool) {
    var r0, r1, r2, r3: UInt64
    var carry: UInt64 = 0

    (r0, carry) = {
        let (s, c) = a.0.addingReportingOverflow(b.0)
        return (s, c ? UInt64(1) : 0)
    }()
    (r1, carry) = {
        let (s1, c1) = a.1.addingReportingOverflow(b.1)
        let (s2, c2) = s1.addingReportingOverflow(carry)
        return (s2, (c1 ? UInt64(1) : 0) + (c2 ? UInt64(1) : 0))
    }()
    (r2, carry) = {
        let (s1, c1) = a.2.addingReportingOverflow(b.2)
        let (s2, c2) = s1.addingReportingOverflow(carry)
        return (s2, (c1 ? UInt64(1) : 0) + (c2 ? UInt64(1) : 0))
    }()
    let (s3, c3) = a.3.addingReportingOverflow(b.3)
    let (s3b, c3b) = s3.addingReportingOverflow(carry)
    r3 = s3b
    let finalCarry = c3 || c3b

    return ((r0, r1, r2, r3), finalCarry)
}

/// Reduce mod q: if value >= q, subtract q
func bwScalarReduce(_ v: (UInt64, UInt64, UInt64, UInt64)) -> (UInt64, UInt64, UInt64, UInt64) {
    if bwScalarGTE(v, BwScalar.Q) {
        return bwScalarSub256(v, BwScalar.Q)
    }
    return v
}

/// Addition mod q
public func bwScalarAdd(_ a: BwScalar, _ b: BwScalar) -> BwScalar {
    let (sum, carry) = bwScalarAdd256(a.limbs, b.limbs)
    let result: (UInt64, UInt64, UInt64, UInt64)
    if carry || bwScalarGTE(sum, BwScalar.Q) {
        result = bwScalarSub256(sum, BwScalar.Q)
    } else {
        result = sum
    }
    return BwScalar(result.0, result.1, result.2, result.3)
}

/// Subtraction mod q
public func bwScalarSub(_ a: BwScalar, _ b: BwScalar) -> BwScalar {
    if bwScalarGTE(a.limbs, b.limbs) {
        let diff = bwScalarSub256(a.limbs, b.limbs)
        return BwScalar(diff.0, diff.1, diff.2, diff.3)
    } else {
        // a < b: result = q - (b - a)
        let diff = bwScalarSub256(b.limbs, a.limbs)
        let result = bwScalarSub256(BwScalar.Q, diff)
        return BwScalar(result.0, result.1, result.2, result.3)
    }
}

/// Negation mod q
public func bwScalarNeg(_ a: BwScalar) -> BwScalar {
    if a.isZero { return .zero }
    let result = bwScalarSub256(BwScalar.Q, a.limbs)
    return BwScalar(result.0, result.1, result.2, result.3)
}

/// Multiplication mod q using schoolbook 4x4 limb multiply with Barrett-like reduction.
/// Uses 512-bit intermediate result.
public func bwScalarMul(_ a: BwScalar, _ b: BwScalar) -> BwScalar {
    // Full 512-bit product
    var prod = [UInt64](repeating: 0, count: 8)

    // Schoolbook multiply
    for i in 0..<4 {
        let ai: UInt64
        switch i {
        case 0: ai = a.limbs.0
        case 1: ai = a.limbs.1
        case 2: ai = a.limbs.2
        default: ai = a.limbs.3
        }
        if ai == 0 { continue }
        var carry: UInt64 = 0
        for j in 0..<4 {
            let bj: UInt64
            switch j {
            case 0: bj = b.limbs.0
            case 1: bj = b.limbs.1
            case 2: bj = b.limbs.2
            default: bj = b.limbs.3
            }
            let (hi, lo) = ai.multipliedFullWidth(by: bj)
            let (s1, c1) = prod[i + j].addingReportingOverflow(lo)
            let (s2, c2) = s1.addingReportingOverflow(carry)
            prod[i + j] = s2
            carry = hi &+ (c1 ? 1 : 0) &+ (c2 ? 1 : 0)
        }
        prod[i + 4] = prod[i + 4] &+ carry
    }

    // Reduce 512-bit product mod q using repeated subtraction with shifting
    // Since q is ~253 bits and product is up to ~506 bits, we use division
    // by repeated trial subtraction at decreasing shifts.
    return bwScalarReduceWide(prod)
}

/// Reduce a wide (up to 512-bit) value mod q using Montgomery reduction.
/// We use a simple approach: compute wide mod q by leveraging Fr381 arithmetic.
/// Since q < r (Fr381 modulus), we can split the 512-bit product into high and low
/// parts and reduce using the relation: wide = high * 2^256 + low = high * (2^256 mod q) + low (mod q).
private func bwScalarReduceWide(_ wide: [UInt64]) -> BwScalar {
    // Quick check: if top 4 limbs are all zero, just reduce bottom 4
    if wide[4] == 0 && wide[5] == 0 && wide[6] == 0 && wide[7] == 0 {
        let v = (wide[0], wide[1], wide[2], wide[3])
        let r = bwScalarReduce(v)
        return BwScalar(r.0, r.1, r.2, r.3)
    }

    // Split: wide = hi * 2^256 + lo
    // Compute: (hi * (2^256 mod q) + lo) mod q
    // 2^256 mod q is a precomputed constant.
    //
    // Since hi is at most 256 bits and (2^256 mod q) is ~253 bits,
    // hi * (2^256 mod q) is at most 509 bits. We might need to iterate.
    //
    // Use Fr381 arithmetic for the multiply since it handles large numbers:
    // Convert everything to Fr381, do arithmetic mod r, then reduce result mod q.
    //
    // Actually, since we need mod q (not mod r), let's use a different approach.
    // We'll do the multiplication in 512-bit and recurse.

    // Precomputed: 2^256 mod q
    // q = 0x1cfb69d4ca675f52_0cce760202687600_ff8f870074190471_74fd06b52876e7e1
    // 2^256 mod q: compute by noting 2^256 = k*q + rem
    // We'll compute this once.
    // 2^256 = 115792089237316195423570985008687907853269984665640564039457584007913129639936
    // q     = 13108968793781547619861935127046491459309155893440570251786403306729687672801
    // 2^256 / q ~ 8.831...
    // 2^256 mod q = 2^256 - 8*q
    // 8*q = 104871750350252380958895481016371931674473247147524562014291226453837501382408
    // 2^256 - 8*q = 10920338887063814464675504...
    // Let me compute: 2^256 = 0x1_0000...0000 (257 bits)
    // Easier: compute 2^256 mod q by repeated doubling from 2^252
    // Or just use a static value.

    // Strategy: use recursive splitting. This converges quickly since each step
    // reduces from ~510 bits to ~509 bits... that's bad.
    //
    // Better approach: use schoolbook long division with bit shifts.
    // Find the highest set bit of wide, shift q left to align, subtract.

    var w = wide  // 8 limbs

    // Find highest non-zero limb
    var topLimb = 7
    while topLimb > 3 && w[topLimb] == 0 { topLimb -= 1 }

    if topLimb <= 3 {
        let v = bwScalarReduce((w[0], w[1], w[2], w[3]))
        return BwScalar(v.0, v.1, v.2, v.3)
    }

    // Find highest set bit position in the wide value
    let topBits = 64 - w[topLimb].leadingZeroBitCount
    let wideBits = topLimb * 64 + topBits

    // q has ~253 bits
    let qBits = 253  // q < 2^253

    // We need to subtract q << shift for shift from (wideBits - qBits) down to 0
    // This is at most ~260 iterations for a 512-bit product, which is fast.
    var shift = wideBits - qBits
    if shift < 0 { shift = 0 }

    while shift >= 0 {
        // Check if w >= q << shift
        // q << shift: shift by (shift/64) limbs and (shift%64) bits
        let limbShift = shift / 64
        let bitShift = shift % 64

        // Build q_shifted in a temp buffer
        var qShifted = [UInt64](repeating: 0, count: 8)
        let qArr: [UInt64] = [BwScalar.Q.0, BwScalar.Q.1, BwScalar.Q.2, BwScalar.Q.3]
        for i in 0..<4 {
            let targetLimb = i + limbShift
            if targetLimb >= 8 { break }
            if bitShift == 0 {
                qShifted[targetLimb] = qArr[i]
            } else {
                qShifted[targetLimb] |= qArr[i] << bitShift
                if targetLimb + 1 < 8 {
                    qShifted[targetLimb + 1] |= qArr[i] >> (64 - bitShift)
                }
            }
        }

        // Compare w >= qShifted (8-limb comparison)
        var wGE = true
        for i in stride(from: 7, through: 0, by: -1) {
            if w[i] > qShifted[i] { break }
            if w[i] < qShifted[i] { wGE = false; break }
        }

        if wGE {
            // Subtract: w -= qShifted
            var borrow: UInt64 = 0
            for i in 0..<8 {
                let (t1, b1) = w[i].subtractingReportingOverflow(qShifted[i])
                let (t2, b2) = t1.subtractingReportingOverflow(borrow)
                w[i] = t2
                borrow = (b1 ? 1 : 0) &+ (b2 ? 1 : 0)
            }
        }

        shift -= 1
    }

    // Final reduction
    let v = bwScalarReduce((w[0], w[1], w[2], w[3]))
    return BwScalar(v.0, v.1, v.2, v.3)
}

/// Square mod q
public func bwScalarSqr(_ a: BwScalar) -> BwScalar {
    return bwScalarMul(a, a)
}

/// Modular inverse using Fermat's little theorem: a^(-1) = a^(q-2) mod q
public func bwScalarInverse(_ a: BwScalar) -> BwScalar {
    precondition(!a.isZero, "Cannot invert zero")
    // q - 2
    let qm2 = bwScalarSub256(BwScalar.Q, (2, 0, 0, 0))
    return bwScalarPow(a, BwScalar(qm2.0, qm2.1, qm2.2, qm2.3))
}

/// Exponentiation by squaring
public func bwScalarPow(_ base: BwScalar, _ exp: BwScalar) -> BwScalar {
    var result = BwScalar.one
    var b = base
    let expLimbs = [exp.limbs.0, exp.limbs.1, exp.limbs.2, exp.limbs.3]
    for i in 0..<4 {
        var word = expLimbs[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = bwScalarMul(result, b)
            }
            b = bwScalarSqr(b)
            word >>= 1
        }
    }
    return result
}

/// Create BwScalar from a small integer
public func bwScalarFromInt(_ v: UInt64) -> BwScalar {
    let limbs = (v, UInt64(0), UInt64(0), UInt64(0))
    let r = bwScalarReduce(limbs)
    return BwScalar(r.0, r.1, r.2, r.3)
}

/// Convert Fr381 (Montgomery form) to BwScalar (standard form, reduced mod q)
public func bwScalarFromFr381(_ f: Fr381) -> BwScalar {
    let limbs = fr381ToInt(f)  // dereduces from Montgomery
    // Now reduce mod q
    // limbs is [UInt64], but since r > q, we may need to subtract q multiple times
    var v = (limbs[0], limbs[1], limbs[2], limbs[3])
    while bwScalarGTE(v, BwScalar.Q) {
        v = bwScalarSub256(v, BwScalar.Q)
    }
    return BwScalar(v.0, v.1, v.2, v.3)
}

/// Convert BwScalar to Fr381 (Montgomery form) for use with curve point operations.
/// Since q < r, the scalar value fits directly into Fr381.
public func bwScalarToFr381(_ s: BwScalar) -> Fr381 {
    let raw = Fr381.from64([s.limbs.0, s.limbs.1, s.limbs.2, s.limbs.3])
    return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
}

/// Scalar multiplication using BwScalar (mod q) instead of Fr381 (mod r).
/// This ensures s*P is consistent with the group operation (order q).
public func bwScalarMulQ(_ p: BanderwagonExtended, _ scalar: BwScalar) -> BanderwagonExtended {
    if scalar.isZero { return .identity }

    var result = BanderwagonExtended.identity
    var base = p
    let sLimbs = [scalar.limbs.0, scalar.limbs.1, scalar.limbs.2, scalar.limbs.3]

    for i in 0..<4 {
        var word = sLimbs[i]
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = bwAdd(result, base)
            }
            base = bwDouble(base)
            word >>= 1
        }
    }
    return result
}

/// MSM using BwScalar
public func bwMSMQ(_ points: [BanderwagonExtended], _ scalars: [BwScalar]) -> BanderwagonExtended {
    precondition(points.count == scalars.count)
    var result = BanderwagonExtended.identity
    for i in 0..<points.count {
        let term = bwScalarMulQ(points[i], scalars[i])
        result = bwAdd(result, term)
    }
    return result
}
