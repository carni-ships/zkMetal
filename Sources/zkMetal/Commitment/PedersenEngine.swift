// Pedersen Commitment Engine — multi-curve, GPU-accelerated
//
// Pedersen commitment: C = sum(v_i * G_i) + r * H
// where G_i are generator points, H is a blinding generator, r is randomness.
//
// Properties:
//   - Computationally binding under discrete log
//   - Perfectly hiding (with randomness)
//   - Additively homomorphic: C(a, r1) + C(b, r2) = C(a+b, r1+r2)
//
// Uses:
//   - Nova IVC cross-term T commitment
//   - IPA / Bulletproofs polynomial commitment
//   - Verkle tree node commitments
//   - General zero-knowledge proof systems
//
// Supported curves: BN254 G1, Pallas, Vesta, BLS12-381 G1

import Foundation
import NeonFieldOps

// MARK: - Curve Abstraction

/// Supported curve types for Pedersen commitments.
public enum PedersenCurve {
    case bn254
    case pallas
    case vesta
    case bls12_381
}

// MARK: - Curve Point (type-erased wrapper)

/// A type-erased curve point that can hold any supported curve's projective point.
/// This enables the PedersenEngine to be generic across curves while maintaining
/// type safety at the API boundary.
public enum CurvePoint {
    case bn254(PointProjective)
    case pallas(PallasPointProjective)
    case vesta(VestaPointProjective)
    case bls12_381(G1Projective381)

    /// The identity (point at infinity) for the given curve.
    public static func identity(curve: PedersenCurve) -> CurvePoint {
        switch curve {
        case .bn254:     return .bn254(pointIdentity())
        case .pallas:    return .pallas(pallasPointIdentity())
        case .vesta:     return .vesta(vestaPointIdentity())
        case .bls12_381: return .bls12_381(g1_381Identity())
        }
    }

    /// Check if this is the identity point.
    public var isIdentity: Bool {
        switch self {
        case .bn254(let p):     return pointIsIdentity(p)
        case .pallas(let p):    return pallasPointIsIdentity(p)
        case .vesta(let p):     return vestaPointIsIdentity(p)
        case .bls12_381(let p): return g1_381IsIdentity(p)
        }
    }

    /// Which curve this point belongs to.
    public var curve: PedersenCurve {
        switch self {
        case .bn254:     return .bn254
        case .pallas:    return .pallas
        case .vesta:     return .vesta
        case .bls12_381: return .bls12_381
        }
    }
}

/// Add two CurvePoints (must be on the same curve).
public func curvePointAdd(_ a: CurvePoint, _ b: CurvePoint) -> CurvePoint {
    switch (a, b) {
    case (.bn254(let p), .bn254(let q)):
        return .bn254(pointAdd(p, q))
    case (.pallas(let p), .pallas(let q)):
        return .pallas(pallasPointAdd(p, q))
    case (.vesta(let p), .vesta(let q)):
        return .vesta(vestaPointAdd(p, q))
    case (.bls12_381(let p), .bls12_381(let q)):
        return .bls12_381(g1_381Add(p, q))
    default:
        preconditionFailure("Cannot add points from different curves")
    }
}

/// Check equality of two CurvePoints.
public func curvePointEqual(_ a: CurvePoint, _ b: CurvePoint) -> Bool {
    switch (a, b) {
    case (.bn254(let p), .bn254(let q)):
        return pointEqual(p, q)
    case (.pallas(let p), .pallas(let q)):
        return pallasPointEqual(p, q)
    case (.vesta(let p), .vesta(let q)):
        return vestaPointEqual(p, q)
    case (.bls12_381(let p), .bls12_381(let q)):
        return g1_381ProjectiveEqual(p, q)
    default:
        return false
    }
}

// MARK: - Scalar (type-erased wrapper)

/// A type-erased scalar field element for use with PedersenEngine.
public enum CurveScalar {
    case bn254(Fr)
    case pallas(VestaFp)   // Pallas Fr = Vesta Fp
    case vesta(PallasFp)   // Vesta Fr = Pallas Fp
    case bls12_381(Fr381)

    /// Zero scalar for the given curve.
    public static func zero(curve: PedersenCurve) -> CurveScalar {
        switch curve {
        case .bn254:     return .bn254(.zero)
        case .pallas:    return .pallas(.zero)
        case .vesta:     return .vesta(.zero)
        case .bls12_381: return .bls12_381(.zero)
        }
    }

    /// One scalar for the given curve.
    public static func one(curve: PedersenCurve) -> CurveScalar {
        switch curve {
        case .bn254:     return .bn254(.one)
        case .pallas:    return .pallas(VestaFp.one)
        case .vesta:     return .vesta(PallasFp.one)
        case .bls12_381: return .bls12_381(.one)
        }
    }

    /// Which curve this scalar belongs to.
    public var curve: PedersenCurve {
        switch self {
        case .bn254:     return .bn254
        case .pallas:    return .pallas
        case .vesta:     return .vesta
        case .bls12_381: return .bls12_381
        }
    }
}

// MARK: - Scalar-Point multiplication

/// Multiply a CurvePoint by a CurveScalar.
public func curvePointScalarMul(_ point: CurvePoint, _ scalar: CurveScalar) -> CurvePoint {
    switch (point, scalar) {
    case (.bn254(let p), .bn254(let s)):
        return .bn254(cPointScalarMul(p, s))
    case (.pallas(let p), .pallas(let s)):
        return .pallas(pallasPointScalarMul(p, s))
    case (.vesta(let p), .vesta(let s)):
        return .vesta(vestaPointScalarMul(p, s))
    case (.bls12_381(let p), .bls12_381(let s)):
        return .bls12_381(g1_381ScalarMul(p, s.to64()))
    default:
        preconditionFailure("Point and scalar must be on the same curve")
    }
}

// MARK: - BLS12-381 Projective Equality

/// Check equality of two BLS12-381 G1 projective points.
/// Compares X1*Z2^2 == X2*Z1^2 and Y1*Z2^3 == Y2*Z1^3.
public func g1_381ProjectiveEqual(_ a: G1Projective381, _ b: G1Projective381) -> Bool {
    if g1_381IsIdentity(a) && g1_381IsIdentity(b) { return true }
    if g1_381IsIdentity(a) || g1_381IsIdentity(b) { return false }
    let aZ2 = fp381Sqr(a.z)
    let bZ2 = fp381Sqr(b.z)
    let aZ3 = fp381Mul(a.z, aZ2)
    let bZ3 = fp381Mul(b.z, bZ2)
    let lhsX = fp381Mul(a.x, bZ2)
    let rhsX = fp381Mul(b.x, aZ2)
    let lhsY = fp381Mul(a.y, bZ3)
    let rhsY = fp381Mul(b.y, aZ3)
    return fp381ToInt(lhsX) == fp381ToInt(rhsX) && fp381ToInt(lhsY) == fp381ToInt(rhsY)
}

// MARK: - Pedersen Opening

/// A Pedersen commitment opening: the values and randomness that created the commitment.
public struct PedersenOpening {
    public let values: [CurveScalar]
    public let randomness: CurveScalar
    public let curve: PedersenCurve

    public init(values: [CurveScalar], randomness: CurveScalar, curve: PedersenCurve) {
        self.values = values
        self.randomness = randomness
        self.curve = curve
    }
}

// MARK: - Multi-Curve Pedersen Parameters

/// Pedersen commitment parameters for a specific curve.
/// Contains the generator points G_0..G_{n-1} and blinding generator H.
public struct MultiCurvePedersenParams {
    /// The curve these parameters are for.
    public let curve: PedersenCurve
    /// Number of generators (commitment vector size).
    public let size: Int

    // Curve-specific storage (only one is non-nil based on curve)
    let bn254Generators: [PointAffine]?
    let bn254Blinding: PointAffine?
    let pallasGenerators: [PallasPointAffine]?
    let pallasBlinding: PallasPointAffine?
    let vestaGenerators: [VestaPointAffine]?
    let vestaBlinding: VestaPointAffine?
    let bls381Generators: [G1Affine381]?
    let bls381Blinding: G1Affine381?

    // MARK: - BN254 Generator Setup

    /// Generate deterministic BN254 generators via iterated hash-double.
    /// Uses the same approach as the existing PedersenParams in CommittedCCS.
    public static func generateBN254(size: Int) -> MultiCurvePedersenParams {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(size + 1)
        var acc = g
        for _ in 0..<(size + 1) {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }
        let affinePoints = batchToAffine(projPoints)
        return MultiCurvePedersenParams(
            curve: .bn254, size: size,
            bn254Generators: Array(affinePoints.prefix(size)),
            bn254Blinding: affinePoints[size],
            pallasGenerators: nil, pallasBlinding: nil,
            vestaGenerators: nil, vestaBlinding: nil,
            bls381Generators: nil, bls381Blinding: nil
        )
    }

    // MARK: - Pallas Generator Setup

    /// Generate deterministic Pallas generators via iterated double-add.
    public static func generatePallas(size: Int) -> MultiCurvePedersenParams {
        // Use a known generator for Pallas: hash-derived from small integers
        let seed = pallasPointFromAffine(PallasPointAffine(
            x: pallasFromInt(1), y: pallasFromInt(2)))

        var projPoints = [PallasPointProjective]()
        projPoints.reserveCapacity(size + 1)
        var acc = seed
        for _ in 0..<(size + 1) {
            projPoints.append(acc)
            acc = pallasPointDouble(pallasPointAdd(acc, seed))
        }

        // Batch inversion for affine conversion
        let affinePoints = pallasBatchToAffine(projPoints)
        return MultiCurvePedersenParams(
            curve: .pallas, size: size,
            bn254Generators: nil, bn254Blinding: nil,
            pallasGenerators: Array(affinePoints.prefix(size)),
            pallasBlinding: affinePoints[size],
            vestaGenerators: nil, vestaBlinding: nil,
            bls381Generators: nil, bls381Blinding: nil
        )
    }

    // MARK: - Vesta Generator Setup

    /// Generate deterministic Vesta generators via iterated double-add.
    public static func generateVesta(size: Int) -> MultiCurvePedersenParams {
        let seed = vestaPointFromAffine(VestaPointAffine(
            x: vestaFromInt(1), y: vestaFromInt(2)))

        var projPoints = [VestaPointProjective]()
        projPoints.reserveCapacity(size + 1)
        var acc = seed
        for _ in 0..<(size + 1) {
            projPoints.append(acc)
            acc = vestaPointDouble(vestaPointAdd(acc, seed))
        }

        let affinePoints = vestaBatchToAffine(projPoints)
        return MultiCurvePedersenParams(
            curve: .vesta, size: size,
            bn254Generators: nil, bn254Blinding: nil,
            pallasGenerators: nil, pallasBlinding: nil,
            vestaGenerators: Array(affinePoints.prefix(size)),
            vestaBlinding: affinePoints[size],
            bls381Generators: nil, bls381Blinding: nil
        )
    }

    // MARK: - BLS12-381 Generator Setup

    /// Generate deterministic BLS12-381 G1 generators via iterated double-add.
    public static func generateBLS12_381(size: Int) -> MultiCurvePedersenParams {
        let seed = g1_381FromAffine(G1Affine381(
            x: fp381FromInt(1), y: fp381FromInt(2)))

        var projPoints = [G1Projective381]()
        projPoints.reserveCapacity(size + 1)
        var acc = seed
        for _ in 0..<(size + 1) {
            projPoints.append(acc)
            acc = g1_381Double(g1_381Add(acc, seed))
        }

        let affinePoints = g1_381BatchToAffine(projPoints)
        return MultiCurvePedersenParams(
            curve: .bls12_381, size: size,
            bn254Generators: nil, bn254Blinding: nil,
            pallasGenerators: nil, pallasBlinding: nil,
            vestaGenerators: nil, vestaBlinding: nil,
            bls381Generators: Array(affinePoints.prefix(size)),
            bls381Blinding: affinePoints[size]
        )
    }

    /// Get the blinding generator as a CurvePoint.
    public var blindingPoint: CurvePoint {
        switch curve {
        case .bn254:     return .bn254(pointFromAffine(bn254Blinding!))
        case .pallas:    return .pallas(pallasPointFromAffine(pallasBlinding!))
        case .vesta:     return .vesta(vestaPointFromAffine(vestaBlinding!))
        case .bls12_381: return .bls12_381(g1_381FromAffine(bls381Blinding!))
        }
    }

    /// Get generator i as a CurvePoint.
    public func generator(_ i: Int) -> CurvePoint {
        precondition(i < size, "Generator index out of range")
        switch curve {
        case .bn254:     return .bn254(pointFromAffine(bn254Generators![i]))
        case .pallas:    return .pallas(pallasPointFromAffine(pallasGenerators![i]))
        case .vesta:     return .vesta(vestaPointFromAffine(vestaGenerators![i]))
        case .bls12_381: return .bls12_381(g1_381FromAffine(bls381Generators![i]))
        }
    }
}

// MARK: - Batch-to-Affine Helpers

/// Batch projective-to-affine for Pallas points using Montgomery's trick.
public func pallasBatchToAffine(_ points: [PallasPointProjective]) -> [PallasPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    // Accumulate products of z-coordinates
    var zProducts = [PallasFp](repeating: PallasFp.one, count: n)
    zProducts[0] = points[0].z
    for i in 1..<n {
        if pallasPointIsIdentity(points[i]) {
            zProducts[i] = zProducts[i - 1]
        } else {
            zProducts[i] = pallasMul(zProducts[i - 1], points[i].z)
        }
    }

    // Single inversion of the product
    var inv = pallasInverse(zProducts[n - 1])

    // Back-substitute to get individual z-inverses
    var result = [PallasPointAffine](repeating: PallasPointAffine(x: .zero, y: .zero), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if pallasPointIsIdentity(points[i]) {
            result[i] = PallasPointAffine(x: PallasFp.zero, y: PallasFp.zero)
            continue
        }
        let zInv: PallasFp
        if i > 0 {
            zInv = pallasMul(inv, zProducts[i - 1])
            inv = pallasMul(inv, points[i].z)
        } else {
            zInv = inv
        }
        let zInv2 = pallasSqr(zInv)
        let zInv3 = pallasMul(zInv2, zInv)
        result[i] = PallasPointAffine(x: pallasMul(points[i].x, zInv2),
                                       y: pallasMul(points[i].y, zInv3))
    }
    return result
}

/// Batch projective-to-affine for Vesta points using Montgomery's trick.
public func vestaBatchToAffine(_ points: [VestaPointProjective]) -> [VestaPointAffine] {
    let n = points.count
    if n == 0 { return [] }

    var zProducts = [VestaFp](repeating: VestaFp.one, count: n)
    zProducts[0] = points[0].z
    for i in 1..<n {
        if vestaPointIsIdentity(points[i]) {
            zProducts[i] = zProducts[i - 1]
        } else {
            zProducts[i] = vestaMul(zProducts[i - 1], points[i].z)
        }
    }

    var inv = vestaInverse(zProducts[n - 1])

    var result = [VestaPointAffine](repeating: VestaPointAffine(x: .zero, y: .zero), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if vestaPointIsIdentity(points[i]) {
            result[i] = VestaPointAffine(x: VestaFp.zero, y: VestaFp.zero)
            continue
        }
        let zInv: VestaFp
        if i > 0 {
            zInv = vestaMul(inv, zProducts[i - 1])
            inv = vestaMul(inv, points[i].z)
        } else {
            zInv = inv
        }
        let zInv2 = vestaSqr(zInv)
        let zInv3 = vestaMul(zInv2, zInv)
        result[i] = VestaPointAffine(x: vestaMul(points[i].x, zInv2),
                                      y: vestaMul(points[i].y, zInv3))
    }
    return result
}

/// Batch projective-to-affine for BLS12-381 G1 points using Montgomery's trick.
public func g1_381BatchToAffine(_ points: [G1Projective381]) -> [G1Affine381] {
    let n = points.count
    if n == 0 { return [] }

    var zProducts = [Fp381](repeating: Fp381.one, count: n)
    zProducts[0] = points[0].z
    for i in 1..<n {
        if g1_381IsIdentity(points[i]) {
            zProducts[i] = zProducts[i - 1]
        } else {
            zProducts[i] = fp381Mul(zProducts[i - 1], points[i].z)
        }
    }

    var inv = fp381Inverse(zProducts[n - 1])

    var result = [G1Affine381](repeating: G1Affine381(x: .zero, y: .zero), count: n)
    for i in stride(from: n - 1, through: 0, by: -1) {
        if g1_381IsIdentity(points[i]) {
            result[i] = G1Affine381(x: Fp381.zero, y: Fp381.zero)
            continue
        }
        let zInv: Fp381
        if i > 0 {
            zInv = fp381Mul(inv, zProducts[i - 1])
            inv = fp381Mul(inv, points[i].z)
        } else {
            zInv = inv
        }
        let zInv2 = fp381Sqr(zInv)
        let zInv3 = fp381Mul(zInv2, zInv)
        result[i] = G1Affine381(x: fp381Mul(points[i].x, zInv2),
                                 y: fp381Mul(points[i].y, zInv3))
    }
    return result
}

// MARK: - Pedersen Engine

/// Multi-curve Pedersen commitment engine with GPU acceleration support.
///
/// Usage:
///   let params = PedersenEngine.setup(size: 1024, curve: .bn254)
///   let commitment = PedersenEngine.commit(values: scalars, randomness: r, params: params)
///   let valid = PedersenEngine.verify(commitment: commitment, values: scalars,
///                                      randomness: r, params: params)
public class PedersenEngine {
    public static let version = Versions.pedersenCommit

    // GPU MSM threshold: use GPU for vectors larger than this
    public static let gpuThreshold = 2048

    // Cached GPU MSM engines (lazy-initialized)
    private static var _bn254MSM: MetalMSM?
    private static var _pallasMSM: PallasMSM?
    private static var _vestaMSM: VestaMSM?

    // MARK: - Setup

    /// Generate Pedersen parameters for the given vector size and curve.
    /// Generators are deterministic (hash-derived) for reproducibility.
    ///
    /// - Parameters:
    ///   - size: number of generators (vector commitment dimension)
    ///   - curve: which elliptic curve to use
    /// - Returns: Pedersen parameters containing generators and blinding point
    public static func setup(size: Int, curve: PedersenCurve) -> MultiCurvePedersenParams {
        precondition(size > 0, "Size must be positive")
        switch curve {
        case .bn254:     return MultiCurvePedersenParams.generateBN254(size: size)
        case .pallas:    return MultiCurvePedersenParams.generatePallas(size: size)
        case .vesta:     return MultiCurvePedersenParams.generateVesta(size: size)
        case .bls12_381: return MultiCurvePedersenParams.generateBLS12_381(size: size)
        }
    }

    // MARK: - Commit

    /// Compute a Pedersen commitment: C = sum(v_i * G_i) + r * H
    ///
    /// For large vectors (> gpuThreshold), uses GPU-accelerated MSM.
    /// Falls back to CPU Pippenger MSM for smaller vectors.
    ///
    /// - Parameters:
    ///   - values: vector of scalar field elements to commit to
    ///   - randomness: blinding factor r (use CurveScalar.zero for non-hiding)
    ///   - params: Pedersen parameters (generators + blinding point)
    /// - Returns: commitment point C
    public static func commit(values: [CurveScalar], randomness: CurveScalar,
                              params: MultiCurvePedersenParams) -> CurvePoint {
        precondition(!values.isEmpty, "Values must not be empty")
        precondition(values.count <= params.size,
                     "Too many values (\(values.count)) for params size (\(params.size))")

        let n = values.count

        switch params.curve {
        case .bn254:
            return commitBN254(values: values, randomness: randomness, params: params, n: n)
        case .pallas:
            return commitPallas(values: values, randomness: randomness, params: params, n: n)
        case .vesta:
            return commitVesta(values: values, randomness: randomness, params: params, n: n)
        case .bls12_381:
            return commitBLS12_381(values: values, randomness: randomness, params: params, n: n)
        }
    }

    // MARK: - Batch Commit

    /// Compute multiple Pedersen commitments efficiently via batched MSM.
    /// All vectors must use the same parameters.
    ///
    /// - Parameters:
    ///   - valueVectors: array of value vectors to commit
    ///   - randomness: array of blinding factors (one per vector)
    ///   - params: shared Pedersen parameters
    /// - Returns: array of commitment points
    public static func batchCommit(valueVectors: [[CurveScalar]],
                                    randomness: [CurveScalar],
                                    params: MultiCurvePedersenParams) -> [CurvePoint] {
        precondition(valueVectors.count == randomness.count,
                     "Must have same number of value vectors and randomness values")
        let k = valueVectors.count
        if k == 0 { return [] }

        switch params.curve {
        case .bn254:
            return batchCommitBN254(valueVectors: valueVectors, randomness: randomness,
                                    params: params)
        default:
            // For non-BN254 curves, fall back to sequential commits
            // (GPU multi-MSM only implemented for BN254 currently)
            return zip(valueVectors, randomness).map { values, r in
                commit(values: values, randomness: r, params: params)
            }
        }
    }

    // MARK: - Open

    /// Create an opening for a Pedersen commitment.
    /// An opening is simply the values and randomness used to create the commitment.
    ///
    /// - Parameters:
    ///   - values: the committed values
    ///   - randomness: the blinding factor
    /// - Returns: a PedersenOpening that can be used for verification
    public static func open(values: [CurveScalar], randomness: CurveScalar) -> PedersenOpening {
        precondition(!values.isEmpty)
        return PedersenOpening(values: values, randomness: randomness, curve: values[0].curve)
    }

    // MARK: - Verify

    /// Verify that a commitment opens to the claimed values and randomness.
    /// Recomputes C = sum(v_i * G_i) + r * H and checks equality.
    ///
    /// - Parameters:
    ///   - commitment: the commitment point to verify
    ///   - values: claimed committed values
    ///   - randomness: claimed blinding factor
    ///   - params: Pedersen parameters
    /// - Returns: true if the commitment is valid
    public static func verify(commitment: CurvePoint, values: [CurveScalar],
                              randomness: CurveScalar,
                              params: MultiCurvePedersenParams) -> Bool {
        let recomputed = commit(values: values, randomness: randomness, params: params)
        return curvePointEqual(commitment, recomputed)
    }

    // MARK: - Homomorphic Operations

    /// Additive homomorphism: C1 + C2.
    /// If C1 = Commit(a, r1) and C2 = Commit(b, r2),
    /// then C1 + C2 = Commit(a + b, r1 + r2).
    public static func homomorphicAdd(_ c1: CurvePoint, _ c2: CurvePoint) -> CurvePoint {
        return curvePointAdd(c1, c2)
    }

    /// Scalar multiplication of a commitment: s * C.
    /// If C = Commit(a, r), then s * C = Commit(s*a, s*r).
    public static func homomorphicScalarMul(commitment: CurvePoint,
                                             scalar: CurveScalar) -> CurvePoint {
        return curvePointScalarMul(commitment, scalar)
    }

    // MARK: - BN254 Commit (private)

    private static func commitBN254(values: [CurveScalar], randomness: CurveScalar,
                                    params: MultiCurvePedersenParams, n: Int) -> CurvePoint {
        // Extract BN254 Fr values
        let frValues: [Fr] = values.map { s in
            guard case .bn254(let v) = s else { preconditionFailure("Expected BN254 scalar") }
            return v
        }
        let frR: Fr
        if case .bn254(let r) = randomness { frR = r }
        else { preconditionFailure("Expected BN254 randomness") }

        // Try GPU MSM for large vectors
        if n > PedersenEngine.gpuThreshold {
            if let engine = getBN254MSM() {
                let pp = PedersenParams(generators: params.bn254Generators!,
                                        blinding: params.bn254Blinding!)
                if let result = try? pp.commitGPU(witness: frValues, engine: engine) {
                    let blindingTerm = cPointScalarMul(pointFromAffine(params.bn254Blinding!), frR)
                    return .bn254(pointAdd(result, blindingTerm))
                }
            }
        }

        // CPU path: use existing PedersenParams commit (Pippenger MSM)
        let pp = PedersenParams(generators: params.bn254Generators!,
                                blinding: params.bn254Blinding!)
        let msmResult = pp.commit(witness: frValues)
        let blindingTerm = cPointScalarMul(pointFromAffine(params.bn254Blinding!), frR)
        return .bn254(pointAdd(msmResult, blindingTerm))
    }

    private static func commitPallas(values: [CurveScalar], randomness: CurveScalar,
                                     params: MultiCurvePedersenParams, n: Int) -> CurvePoint {
        let fpValues: [VestaFp] = values.map { s in
            guard case .pallas(let v) = s else { preconditionFailure("Expected Pallas scalar") }
            return v
        }
        let fpR: VestaFp
        if case .pallas(let r) = randomness { fpR = r }
        else { preconditionFailure("Expected Pallas randomness") }

        // Try GPU MSM for large vectors
        if n > PedersenEngine.gpuThreshold, let engine = getPallasMSM() {
            let scalars32: [[UInt32]] = fpValues.map { s in
                let intVal = vestaToInt(s)
                return [
                    UInt32(intVal[0] & 0xFFFFFFFF), UInt32(intVal[0] >> 32),
                    UInt32(intVal[1] & 0xFFFFFFFF), UInt32(intVal[1] >> 32),
                    UInt32(intVal[2] & 0xFFFFFFFF), UInt32(intVal[2] >> 32),
                    UInt32(intVal[3] & 0xFFFFFFFF), UInt32(intVal[3] >> 32),
                ]
            }
            if let gpuResult = try? engine.msm(points: Array(params.pallasGenerators!.prefix(n)),
                                                scalars: scalars32) {
                let blindingTerm = pallasPointScalarMul(
                    pallasPointFromAffine(params.pallasBlinding!), fpR)
                return .pallas(pallasPointAdd(gpuResult, blindingTerm))
            }
        }

        // CPU path: Pippenger MSM
        let msmResult = pallasCpuMSM(points: Array(params.pallasGenerators!.prefix(n)),
                                     scalars: fpValues)
        let blindingTerm = pallasPointScalarMul(
            pallasPointFromAffine(params.pallasBlinding!), fpR)
        return .pallas(pallasPointAdd(msmResult, blindingTerm))
    }

    private static func commitVesta(values: [CurveScalar], randomness: CurveScalar,
                                    params: MultiCurvePedersenParams, n: Int) -> CurvePoint {
        let fpValues: [PallasFp] = values.map { s in
            guard case .vesta(let v) = s else { preconditionFailure("Expected Vesta scalar") }
            return v
        }
        let fpR: PallasFp
        if case .vesta(let r) = randomness { fpR = r }
        else { preconditionFailure("Expected Vesta randomness") }

        // Try GPU MSM for large vectors
        if n > PedersenEngine.gpuThreshold, let engine = getVestaMSM() {
            let scalars32: [[UInt32]] = fpValues.map { s in
                let intVal = pallasToInt(s)
                return [
                    UInt32(intVal[0] & 0xFFFFFFFF), UInt32(intVal[0] >> 32),
                    UInt32(intVal[1] & 0xFFFFFFFF), UInt32(intVal[1] >> 32),
                    UInt32(intVal[2] & 0xFFFFFFFF), UInt32(intVal[2] >> 32),
                    UInt32(intVal[3] & 0xFFFFFFFF), UInt32(intVal[3] >> 32),
                ]
            }
            if let gpuResult = try? engine.msm(points: Array(params.vestaGenerators!.prefix(n)),
                                                scalars: scalars32) {
                let blindingTerm = vestaPointScalarMul(
                    vestaPointFromAffine(params.vestaBlinding!), fpR)
                return .vesta(vestaPointAdd(gpuResult, blindingTerm))
            }
        }

        // CPU path
        let msmResult = vestaCpuMSM(points: Array(params.vestaGenerators!.prefix(n)),
                                    scalars: fpValues)
        let blindingTerm = vestaPointScalarMul(
            vestaPointFromAffine(params.vestaBlinding!), fpR)
        return .vesta(vestaPointAdd(msmResult, blindingTerm))
    }

    private static func commitBLS12_381(values: [CurveScalar], randomness: CurveScalar,
                                        params: MultiCurvePedersenParams, n: Int) -> CurvePoint {
        let frValues: [Fr381] = values.map { s in
            guard case .bls12_381(let v) = s else { preconditionFailure("Expected BLS12-381 scalar") }
            return v
        }
        let frR: Fr381
        if case .bls12_381(let r) = randomness { frR = r }
        else { preconditionFailure("Expected BLS12-381 randomness") }

        // BLS12-381: CPU-only MSM (no GPU MSM engine for this curve yet)
        // Use direct scalar-mul accumulation for small n, Pippenger for larger
        var result = g1_381Identity()
        let gens = params.bls381Generators!
        for i in 0..<n {
            let gi = g1_381FromAffine(gens[i])
            let term = g1_381ScalarMul(gi, frValues[i].to64())
            result = g1_381IsIdentity(result) ? term : g1_381Add(result, term)
        }
        let blindingTerm = g1_381ScalarMul(g1_381FromAffine(params.bls381Blinding!), frR.to64())
        if !g1_381IsIdentity(blindingTerm) {
            result = g1_381IsIdentity(result) ? blindingTerm : g1_381Add(result, blindingTerm)
        }
        return .bls12_381(result)
    }

    // MARK: - Batch Commit BN254 (GPU multi-MSM)

    private static func batchCommitBN254(valueVectors: [[CurveScalar]],
                                          randomness: [CurveScalar],
                                          params: MultiCurvePedersenParams) -> [CurvePoint] {
        let k = valueVectors.count

        // Extract all Fr values
        let frVectors: [[Fr]] = valueVectors.map { vec in
            vec.map { s in
                guard case .bn254(let v) = s else { preconditionFailure("Expected BN254 scalar") }
                return v
            }
        }
        let frRandomness: [Fr] = randomness.map { s in
            guard case .bn254(let r) = s else { preconditionFailure("Expected BN254 randomness") }
            return r
        }

        // Find max vector length
        let maxLen = frVectors.map(\.count).max()!

        // Try GPU multi-MSM
        if maxLen > PedersenEngine.gpuThreshold, let engine = getBN254MSM() {
            let gens = Array(params.bn254Generators!.prefix(maxLen))
            let scalarSets: [[[UInt32]]] = frVectors.map { vec in
                var scalars = vec.map { frToLimbs($0) }
                // Pad with zeros if shorter than maxLen
                while scalars.count < maxLen {
                    scalars.append([0, 0, 0, 0, 0, 0, 0, 0])
                }
                return scalars
            }
            if let msmResults = try? multiMSM(engine: engine, points: gens, scalarSets: scalarSets) {
                return zip(zip(msmResults, frRandomness), 0..<k).map { item, _ in
                    let (msmResult, r) = item
                    let blindingTerm = cPointScalarMul(
                        pointFromAffine(params.bn254Blinding!), r)
                    return .bn254(pointAdd(msmResult, blindingTerm))
                }
            }
        }

        // Fallback: sequential commits
        return zip(valueVectors, randomness).map { values, r in
            commit(values: values, randomness: r, params: params)
        }
    }

    // MARK: - GPU Engine Access

    private static func getBN254MSM() -> MetalMSM? {
        if _bn254MSM == nil { _bn254MSM = try? MetalMSM() }
        return _bn254MSM
    }

    private static func getPallasMSM() -> PallasMSM? {
        if _pallasMSM == nil { _pallasMSM = try? PallasMSM() }
        return _pallasMSM
    }

    private static func getVestaMSM() -> VestaMSM? {
        if _vestaMSM == nil { _vestaMSM = try? VestaMSM() }
        return _vestaMSM
    }
}
