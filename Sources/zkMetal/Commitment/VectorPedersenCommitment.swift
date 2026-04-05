// Vector Pedersen Commitment — IPA-compatible vector commitments with inner product support
//
// A vector Pedersen commitment commits to a vector of field elements:
//   C = MSM(G, a) + r * H
//
// For IPA (Inner Product Arguments), we additionally support:
//   - Inner product commitment: C = MSM(G, a) + <a, b> * Q + r * H
//   - Blinding factor management for zero-knowledge
//   - Vector operations needed for IPA halving rounds
//
// References:
//   - Bulletproofs (Bunz et al. 2018)
//   - Halo (Bowe et al. 2019)

import Foundation
import NeonFieldOps

// MARK: - Vector Pedersen Parameters

/// Parameters for vector Pedersen commitments, extending the base Pedersen params
/// with an inner-product binding point Q.
public struct VectorPedersenParams {
    /// Base Pedersen parameters (generators G[] and blinding H).
    public let baseParams: MultiCurvePedersenParams
    /// Inner product binding point Q (separate from blinding).
    public let Q: CurvePoint

    /// Create vector Pedersen parameters.
    /// - Parameters:
    ///   - size: vector dimension
    ///   - curve: which curve to use
    public static func generate(size: Int, curve: PedersenCurve) -> VectorPedersenParams {
        // Generate base params with size+1 generators; last one becomes Q
        let extended = PedersenEngine.setup(size: size + 1, curve: curve)

        // Take the last generator as Q, rebuild base params with the first `size`
        let baseParams: MultiCurvePedersenParams
        let Q: CurvePoint

        switch curve {
        case .bn254:
            baseParams = MultiCurvePedersenParams(
                curve: .bn254, size: size,
                bn254Generators: Array(extended.bn254Generators!.prefix(size)),
                bn254Blinding: extended.bn254Blinding,
                pallasGenerators: nil, pallasBlinding: nil,
                vestaGenerators: nil, vestaBlinding: nil,
                bls381Generators: nil, bls381Blinding: nil
            )
            Q = .bn254(pointFromAffine(extended.bn254Generators![size]))

        case .pallas:
            baseParams = MultiCurvePedersenParams(
                curve: .pallas, size: size,
                bn254Generators: nil, bn254Blinding: nil,
                pallasGenerators: Array(extended.pallasGenerators!.prefix(size)),
                pallasBlinding: extended.pallasBlinding,
                vestaGenerators: nil, vestaBlinding: nil,
                bls381Generators: nil, bls381Blinding: nil
            )
            Q = .pallas(pallasPointFromAffine(extended.pallasGenerators![size]))

        case .vesta:
            baseParams = MultiCurvePedersenParams(
                curve: .vesta, size: size,
                bn254Generators: nil, bn254Blinding: nil,
                pallasGenerators: nil, pallasBlinding: nil,
                vestaGenerators: Array(extended.vestaGenerators!.prefix(size)),
                vestaBlinding: extended.vestaBlinding,
                bls381Generators: nil, bls381Blinding: nil
            )
            Q = .vesta(vestaPointFromAffine(extended.vestaGenerators![size]))

        case .bls12_381:
            baseParams = MultiCurvePedersenParams(
                curve: .bls12_381, size: size,
                bn254Generators: nil, bn254Blinding: nil,
                pallasGenerators: nil, pallasBlinding: nil,
                vestaGenerators: nil, vestaBlinding: nil,
                bls381Generators: Array(extended.bls381Generators!.prefix(size)),
                bls381Blinding: extended.bls381Blinding
            )
            Q = .bls12_381(g1_381FromAffine(extended.bls381Generators![size]))
        }

        return VectorPedersenParams(baseParams: baseParams, Q: Q)
    }

    /// Vector dimension (number of generators).
    public var size: Int { baseParams.size }

    /// The curve these parameters are for.
    public var curve: PedersenCurve { baseParams.curve }
}

// MARK: - Blinding Factor Manager

/// Manages blinding factors for zero-knowledge Pedersen commitments.
/// Tracks accumulated blindings through homomorphic operations and IPA rounds.
public struct BlindingManager {
    /// The current blinding factor.
    public private(set) var blinding: CurveScalar
    /// The curve this manager operates on.
    public let curve: PedersenCurve

    /// Create a blinding manager with the given initial blinding factor.
    public init(blinding: CurveScalar) {
        self.blinding = blinding
        self.curve = blinding.curve
    }

    /// Create a blinding manager with a random blinding factor.
    /// Uses the given seed for deterministic generation (testing) or 0 for random.
    public static func random(curve: PedersenCurve, seed: UInt64 = 0) -> BlindingManager {
        let scalar: CurveScalar
        if seed == 0 {
            // Non-deterministic: use random bytes
            scalar = randomScalar(curve: curve)
        } else {
            // Deterministic from seed
            scalar = scalarFromInt(seed, curve: curve)
        }
        return BlindingManager(blinding: scalar)
    }

    /// Update the blinding after a homomorphic add: r' = r1 + r2
    public mutating func addBlinding(_ other: CurveScalar) {
        blinding = scalarAdd(blinding, other)
    }

    /// Update the blinding after scalar multiplication: r' = s * r
    public mutating func scaleBlinding(_ s: CurveScalar) {
        blinding = scalarMul(blinding, s)
    }
}

// MARK: - Vector Pedersen Commitment Engine

/// Engine for vector Pedersen commitments with IPA support.
public class VectorPedersenCommitment {
    public static let version = Versions.pedersenCommit

    // MARK: - Basic Vector Commitment

    /// Commit to a vector of scalars: C = MSM(G, a) + r * H
    ///
    /// - Parameters:
    ///   - vector: the values to commit to
    ///   - blinding: randomness for zero-knowledge
    ///   - params: vector Pedersen parameters
    /// - Returns: commitment point
    public static func commit(vector: [CurveScalar], blinding: CurveScalar,
                              params: VectorPedersenParams) -> CurvePoint {
        return PedersenEngine.commit(values: vector, randomness: blinding,
                                     params: params.baseParams)
    }

    // MARK: - Inner Product Commitment

    /// Commit to vectors a, b with inner product binding:
    ///   C = MSM(G, a) + <a, b> * Q + r * H
    ///
    /// This is the commitment used in IPA/Bulletproofs where Q binds the inner product value.
    ///
    /// - Parameters:
    ///   - a: first vector
    ///   - b: second vector (typically evaluation coefficients)
    ///   - blinding: randomness for zero-knowledge
    ///   - params: vector Pedersen parameters with Q point
    /// - Returns: commitment point
    public static func innerProductCommit(a: [CurveScalar], b: [CurveScalar],
                                           blinding: CurveScalar,
                                           params: VectorPedersenParams) -> CurvePoint {
        precondition(a.count == b.count, "Vectors must have equal length")
        precondition(a.count <= params.size, "Vectors too large for params")

        // Compute MSM(G, a) + r * H
        let baseCommit = PedersenEngine.commit(values: a, randomness: blinding,
                                               params: params.baseParams)

        // Compute inner product <a, b>
        let ip = innerProduct(a, b)

        // Add <a, b> * Q
        let ipTerm = curvePointScalarMul(params.Q, ip)
        return curvePointAdd(baseCommit, ipTerm)
    }

    // MARK: - IPA Helper: Compute L, R for a halving round

    /// Compute the left and right commitments for an IPA halving round.
    ///
    /// Given vectors a = [a_lo, a_hi], b = [b_lo, b_hi], generators G = [G_lo, G_hi]:
    ///   L = MSM(G_hi, a_lo) + <a_lo, b_hi> * Q + l_blind * H
    ///   R = MSM(G_lo, a_hi) + <a_hi, b_lo> * Q + r_blind * H
    ///
    /// - Parameters:
    ///   - aLo, aHi: split halves of vector a
    ///   - bLo, bHi: split halves of vector b
    ///   - params: vector Pedersen params (must have size >= aLo.count + aHi.count)
    ///   - lBlinding: blinding for L commitment
    ///   - rBlinding: blinding for R commitment
    /// - Returns: (L, R) commitment points
    public static func ipaHalvingRound(
        aLo: [CurveScalar], aHi: [CurveScalar],
        bLo: [CurveScalar], bHi: [CurveScalar],
        params: VectorPedersenParams,
        lBlinding: CurveScalar, rBlinding: CurveScalar
    ) -> (L: CurvePoint, R: CurvePoint) {
        let halfN = aLo.count
        precondition(aHi.count == halfN && bLo.count == halfN && bHi.count == halfN)

        // Build sub-params for the "hi" generators (indices halfN..<2*halfN)
        // and "lo" generators (indices 0..<halfN)
        let curve = params.curve

        // L = MSM(G_hi, a_lo) + <a_lo, b_hi> * Q + l_blind * H
        let lBase = msmWithGeneratorRange(scalars: aLo, params: params, offset: halfN, count: halfN)
        let lIP = innerProduct(aLo, bHi)
        let lIPTerm = curvePointScalarMul(params.Q, lIP)
        let lBlindTerm = curvePointScalarMul(params.baseParams.blindingPoint, lBlinding)
        let L = curvePointAdd(curvePointAdd(lBase, lIPTerm), lBlindTerm)

        // R = MSM(G_lo, a_hi) + <a_hi, b_lo> * Q + r_blind * H
        let rBase = msmWithGeneratorRange(scalars: aHi, params: params, offset: 0, count: halfN)
        let rIP = innerProduct(aHi, bLo)
        let rIPTerm = curvePointScalarMul(params.Q, rIP)
        let rBlindTerm = curvePointScalarMul(params.baseParams.blindingPoint, rBlinding)
        let R = curvePointAdd(curvePointAdd(rBase, rIPTerm), rBlindTerm)

        return (L, R)
    }

    // MARK: - Verify Vector Opening

    /// Verify that a vector commitment opens to claimed values.
    ///
    /// - Parameters:
    ///   - commitment: the commitment to verify
    ///   - vector: the claimed vector values
    ///   - blinding: the claimed blinding factor
    ///   - params: vector Pedersen parameters
    /// - Returns: true if the commitment is valid
    public static func verify(commitment: CurvePoint, vector: [CurveScalar],
                              blinding: CurveScalar,
                              params: VectorPedersenParams) -> Bool {
        let recomputed = commit(vector: vector, blinding: blinding, params: params)
        return curvePointEqual(commitment, recomputed)
    }

    /// Verify an inner product commitment.
    ///
    /// - Parameters:
    ///   - commitment: the commitment to verify
    ///   - a, b: the claimed vectors
    ///   - blinding: the claimed blinding factor
    ///   - params: vector Pedersen parameters
    /// - Returns: true if the commitment is valid
    public static func verifyInnerProduct(commitment: CurvePoint,
                                           a: [CurveScalar], b: [CurveScalar],
                                           blinding: CurveScalar,
                                           params: VectorPedersenParams) -> Bool {
        let recomputed = innerProductCommit(a: a, b: b, blinding: blinding, params: params)
        return curvePointEqual(commitment, recomputed)
    }

    // MARK: - Private Helpers

    /// Compute MSM using a range of generators from the params.
    private static func msmWithGeneratorRange(scalars: [CurveScalar],
                                               params: VectorPedersenParams,
                                               offset: Int, count: Int) -> CurvePoint {
        let n = scalars.count
        precondition(n == count)
        precondition(offset + count <= params.size)

        switch params.curve {
        case .bn254:
            let frValues: [Fr] = scalars.map {
                guard case .bn254(let v) = $0 else { preconditionFailure("Expected BN254 scalar") }
                return v
            }
            let subGens = Array(params.baseParams.bn254Generators![offset..<(offset + count)])
            // Use small-vector direct path or Pippenger
            if n <= 16 {
                var result = pointIdentity()
                for i in 0..<n {
                    let sp = cPointScalarMul(pointFromAffine(subGens[i]), frValues[i])
                    result = pointAdd(result, sp)
                }
                return .bn254(result)
            }
            let scalarLimbs = frValues.map { frToLimbs($0) }
            return .bn254(cPippengerMSM(points: subGens, scalars: scalarLimbs))

        case .pallas:
            let fpValues: [VestaFp] = scalars.map {
                guard case .pallas(let v) = $0 else { preconditionFailure("Expected Pallas scalar") }
                return v
            }
            let subGens = Array(params.baseParams.pallasGenerators![offset..<(offset + count)])
            return .pallas(pallasCpuMSM(points: subGens, scalars: fpValues))

        case .vesta:
            let fpValues: [PallasFp] = scalars.map {
                guard case .vesta(let v) = $0 else { preconditionFailure("Expected Vesta scalar") }
                return v
            }
            let subGens = Array(params.baseParams.vestaGenerators![offset..<(offset + count)])
            return .vesta(vestaCpuMSM(points: subGens, scalars: fpValues))

        case .bls12_381:
            let frValues: [Fr381] = scalars.map {
                guard case .bls12_381(let v) = $0 else { preconditionFailure("Expected BLS12-381 scalar") }
                return v
            }
            let subGens = Array(params.baseParams.bls381Generators![offset..<(offset + count)])
            var result = g1_381Identity()
            for i in 0..<n {
                let gi = g1_381FromAffine(subGens[i])
                let term = g1_381ScalarMul(gi, frValues[i].to64())
                result = g1_381IsIdentity(result) ? term : g1_381Add(result, term)
            }
            return .bls12_381(result)
        }
    }
}

// MARK: - Scalar Field Operations (type-erased)

/// Compute inner product <a, b> = sum(a_i * b_i) over the scalar field.
public func innerProduct(_ a: [CurveScalar], _ b: [CurveScalar]) -> CurveScalar {
    precondition(a.count == b.count && !a.isEmpty)
    let n = a.count

    switch a[0] {
    case .bn254:
        var sum = Fr.zero
        for i in 0..<n {
            guard case .bn254(let ai) = a[i], case .bn254(let bi) = b[i] else {
                preconditionFailure("Mismatched scalar types")
            }
            sum = frAdd(sum, frMul(ai, bi))
        }
        return .bn254(sum)

    case .pallas:
        var sum = VestaFp.zero
        for i in 0..<n {
            guard case .pallas(let ai) = a[i], case .pallas(let bi) = b[i] else {
                preconditionFailure("Mismatched scalar types")
            }
            sum = vestaAdd(sum, vestaMul(ai, bi))
        }
        return .pallas(sum)

    case .vesta:
        var sum = PallasFp.zero
        for i in 0..<n {
            guard case .vesta(let ai) = a[i], case .vesta(let bi) = b[i] else {
                preconditionFailure("Mismatched scalar types")
            }
            sum = pallasAdd(sum, pallasMul(ai, bi))
        }
        return .vesta(sum)

    case .bls12_381:
        var sum = Fr381.zero
        for i in 0..<n {
            guard case .bls12_381(let ai) = a[i], case .bls12_381(let bi) = b[i] else {
                preconditionFailure("Mismatched scalar types")
            }
            sum = fr381Add(sum, fr381Mul(ai, bi))
        }
        return .bls12_381(sum)
    }
}

/// Add two scalars on the same curve.
public func scalarAdd(_ a: CurveScalar, _ b: CurveScalar) -> CurveScalar {
    switch (a, b) {
    case (.bn254(let x), .bn254(let y)):     return .bn254(frAdd(x, y))
    case (.pallas(let x), .pallas(let y)):   return .pallas(vestaAdd(x, y))
    case (.vesta(let x), .vesta(let y)):     return .vesta(pallasAdd(x, y))
    case (.bls12_381(let x), .bls12_381(let y)): return .bls12_381(fr381Add(x, y))
    default: preconditionFailure("Cannot add scalars from different curves")
    }
}

/// Multiply two scalars on the same curve.
public func scalarMul(_ a: CurveScalar, _ b: CurveScalar) -> CurveScalar {
    switch (a, b) {
    case (.bn254(let x), .bn254(let y)):     return .bn254(frMul(x, y))
    case (.pallas(let x), .pallas(let y)):   return .pallas(vestaMul(x, y))
    case (.vesta(let x), .vesta(let y)):     return .vesta(pallasMul(x, y))
    case (.bls12_381(let x), .bls12_381(let y)): return .bls12_381(fr381Mul(x, y))
    default: preconditionFailure("Cannot multiply scalars from different curves")
    }
}

/// Create a scalar from a small integer on a given curve.
public func scalarFromInt(_ val: UInt64, curve: PedersenCurve) -> CurveScalar {
    switch curve {
    case .bn254:     return .bn254(frFromInt(val))
    case .pallas:    return .pallas(vestaFromInt(val))
    case .vesta:     return .vesta(pallasFromInt(val))
    case .bls12_381: return .bls12_381(fr381FromInt(val))
    }
}

/// Generate a random scalar (non-deterministic) for the given curve.
public func randomScalar(curve: PedersenCurve) -> CurveScalar {
    // Use arc4random_buf for cryptographic randomness, then reduce mod p
    var bytes = [UInt8](repeating: 0, count: 32)
    _ = SecRandomCopyBytes(kSecRandomDefault, 32, &bytes)

    switch curve {
    case .bn254:
        // Interpret bytes as 4xUInt64, reduce to Fr
        let limbs = bytes.withUnsafeBytes { buf -> [UInt64] in
            let ptr = buf.bindMemory(to: UInt64.self)
            return [ptr[0], ptr[1], ptr[2], ptr[3]]
        }
        // Simple reduction: clear top bits to ensure < r
        var reduced = limbs
        reduced[3] &= 0x0FFFFFFFFFFFFFFF  // BN254 Fr is ~254 bits
        let fr = Fr(v: (
            UInt32(reduced[0] & 0xFFFFFFFF), UInt32(reduced[0] >> 32),
            UInt32(reduced[1] & 0xFFFFFFFF), UInt32(reduced[1] >> 32),
            UInt32(reduced[2] & 0xFFFFFFFF), UInt32(reduced[2] >> 32),
            UInt32(reduced[3] & 0xFFFFFFFF), UInt32(reduced[3] >> 32)
        ))
        return .bn254(fr)

    case .pallas:
        return .pallas(vestaFromInt(UInt64(bytes[0]) | (UInt64(bytes[1]) << 8) |
                                    (UInt64(bytes[2]) << 16) | (UInt64(bytes[3]) << 24)))
    case .vesta:
        return .vesta(pallasFromInt(UInt64(bytes[0]) | (UInt64(bytes[1]) << 8) |
                                    (UInt64(bytes[2]) << 16) | (UInt64(bytes[3]) << 24)))
    case .bls12_381:
        return .bls12_381(fr381FromInt(UInt64(bytes[0]) | (UInt64(bytes[1]) << 8) |
                                       (UInt64(bytes[2]) << 16) | (UInt64(bytes[3]) << 24)))
    }
}
