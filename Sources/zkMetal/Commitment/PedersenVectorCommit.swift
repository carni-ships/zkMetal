// Pedersen Vector Commitment Engine — GPU-accelerated via MSM
//
// Provides a BN254 G1-focused Pedersen vector commitment scheme that leverages
// the existing MetalMSM infrastructure for GPU acceleration. This is the
// foundational commitment scheme used by IPA, Bulletproofs, and Hyrax.
//
// Commitment: C = sum(v_i * G_i) + r * H
//   where G_i are public generators, H is a blinding generator, r is randomness.
//
// Properties:
//   - Computationally binding under discrete log assumption
//   - Perfectly hiding (with random blinding factor)
//   - Additively homomorphic: Commit(a, r1) + Commit(b, r2) = Commit(a+b, r1+r2)

import Foundation
import Metal
import NeonFieldOps

// NOTE: PedersenParams is defined in Folding/CommittedCCS.swift and reused here.
// It provides generators, blinding, commit(witness:), and commitGPU(witness:engine:).

// MARK: - Pedersen Vector Commit Engine

/// GPU-accelerated Pedersen vector commitment engine for BN254 G1.
///
/// Usage:
///   let engine = PedersenVectorCommitEngine()
///   let params = engine.setup(n: 256)
///   let commitment = engine.commit(values: frVector, params: params, blinding: r)
///   let valid = engine.open(values: frVector, blinding: r, params: params, commitment: commitment)
public class PedersenVectorCommitEngine {
    /// Cached GPU MSM engine (lazy-initialized).
    private var _msmEngine: MetalMSM?

    /// GPU MSM threshold: use GPU for vectors of this size or larger.
    public static let gpuThreshold = 2048

    public init() {}

    // MARK: - GPU Engine

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - Setup

    /// Generate Pedersen parameters with n random generators G_1,...,G_n plus blinding generator H.
    ///
    /// Generators are derived deterministically from the BN254 G1 generator via iterated
    /// hash-double: G_i = double(G_{i-1} + seed). This is NOT suitable for production
    /// (which requires hash-to-curve); it is deterministic for reproducibility.
    ///
    /// - Parameter n: vector dimension (number of generators)
    /// - Returns: PedersenParams containing generators and blinding point
    public func setup(n: Int) -> PedersenParams {
        precondition(n > 0, "Dimension must be positive")

        // Use the BN254 G1 generator point (1, 2) as seed
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let g = pointFromAffine(PointAffine(x: gx, y: gy))

        // Generate n+1 distinct points via iterated double-add
        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(n + 1)
        var acc = g
        for _ in 0..<(n + 1) {
            projPoints.append(acc)
            acc = pointDouble(pointAdd(acc, g))
        }

        // Batch convert to affine (single field inversion via Montgomery's trick)
        let affinePoints = batchToAffine(projPoints)
        return PedersenParams(
            generators: Array(affinePoints.prefix(n)),
            blinding: affinePoints[n]
        )
    }

    // MARK: - Commit

    /// Compute Pedersen vector commitment: C = sum(v_i * G_i) + r * H
    ///
    /// For vectors >= gpuThreshold, uses GPU-accelerated MSM via Metal.
    /// Falls back to CPU Pippenger MSM for smaller vectors.
    ///
    /// - Parameters:
    ///   - values: vector of BN254 Fr elements to commit to
    ///   - params: Pedersen parameters (generators + blinding point)
    ///   - blinding: optional blinding factor r (defaults to zero = non-hiding)
    /// - Returns: commitment point in projective coordinates
    public func commit(values: [Fr], params: PedersenParams,
                       blinding: Fr? = nil) -> PointProjective {
        precondition(!values.isEmpty, "Values must not be empty")
        precondition(values.count <= params.generators.count,
                     "Too many values (\(values.count)) for params of size \(params.generators.count)")

        let n = values.count

        // MSM: C_base = sum(v_i * G_i)
        let msmResult: PointProjective
        if n >= PedersenVectorCommitEngine.gpuThreshold, let engine = getMSMEngine() {
            let gens = Array(params.generators.prefix(n))
            let scalarLimbs = values.map { frToLimbs($0) }
            msmResult = (try? engine.msm(points: gens, scalars: scalarLimbs))
                ?? params.commit(witness: values)
        } else {
            msmResult = params.commit(witness: values)
        }

        // Add blinding term: r * H
        let r = blinding ?? Fr.zero
        if frToInt(r) != [0, 0, 0, 0] {
            let blindingTerm = cPointScalarMul(pointFromAffine(params.blinding), r)
            return pointAdd(msmResult, blindingTerm)
        }
        return msmResult
    }

    // MARK: - Batch Commit

    /// Batch commit multiple vectors using the same parameters.
    ///
    /// For BN254, leverages multiMSM to share the GPU point upload across all vectors.
    ///
    /// - Parameters:
    ///   - vectors: array of Fr vectors to commit
    ///   - params: shared Pedersen parameters
    /// - Returns: array of commitment points (one per input vector)
    public func batchCommit(vectors: [[Fr]], params: PedersenParams) -> [PointProjective] {
        let k = vectors.count
        if k == 0 { return [] }

        // Find max vector length for padding
        let maxLen = vectors.map(\.count).max()!
        precondition(maxLen <= params.generators.count, "Vector too large for params")

        // For small batch or small vectors, just commit sequentially
        if k <= 2 || maxLen < PedersenVectorCommitEngine.gpuThreshold {
            return vectors.map { commit(values: $0, params: params) }
        }

        // GPU multi-MSM path
        if let engine = getMSMEngine() {
            let gens = Array(params.generators.prefix(maxLen))
            let scalarSets: [[[UInt32]]] = vectors.map { vec in
                var scalars = vec.map { frToLimbs($0) }
                // Pad shorter vectors with zero scalars
                while scalars.count < maxLen {
                    scalars.append([0, 0, 0, 0, 0, 0, 0, 0])
                }
                return scalars
            }
            if let results = try? multiMSM(engine: engine, points: gens, scalarSets: scalarSets) {
                return results
            }
        }

        // Fallback: sequential
        return vectors.map { commit(values: $0, params: params) }
    }

    // MARK: - Open (Verify)

    /// Verify a commitment opening by recomputing the commitment.
    ///
    /// Checks whether C == sum(v_i * G_i) + r * H.
    ///
    /// - Parameters:
    ///   - values: the claimed vector values
    ///   - blinding: the claimed blinding factor r
    ///   - params: Pedersen parameters
    ///   - commitment: the commitment to verify
    /// - Returns: true if the commitment matches
    public func open(values: [Fr], blinding: Fr, params: PedersenParams,
                     commitment: PointProjective) -> Bool {
        let recomputed = commit(values: values, params: params, blinding: blinding)
        return pointEqual(commitment, recomputed)
    }

    // MARK: - Homomorphic Operations

    /// Additive homomorphism: C1 + C2.
    /// If C1 = Commit(a, r1) and C2 = Commit(b, r2),
    /// then add(C1, C2) = Commit(a + b, r1 + r2).
    public static func add(_ c1: PointProjective, _ c2: PointProjective) -> PointProjective {
        return pointAdd(c1, c2)
    }

    /// Scalar multiplication of a commitment: s * C.
    /// If C = Commit(a, r), then scalarMul(s, C) = Commit(s*a, s*r).
    public static func scalarMul(_ s: Fr, _ c: PointProjective) -> PointProjective {
        return cPointScalarMul(c, s)
    }
}
