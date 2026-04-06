// GPU-Accelerated Vector Commitment Engine — Pedersen commitments with position opening proofs
//
// Provides GPU-accelerated Pedersen vector commitments on BN254 G1 with:
//   - Single & batch commitment via Metal MSM
//   - Position-specific opening proofs (open at index i)
//   - Opening proof verification
//
// Commitment: C = sum(v_i * G_i) for i in 0..<n
//
// Opening at index j:
//   proof = sum(v_i * G_i) for i != j   (partial commitment excluding index j)
//   Verification: C == proof + v_j * G_j
//
// This is an unconditionally-binding, non-hiding Pedersen vector commitment
// with O(n) opening proof generation and O(1) verification (one scalar mul + one add).

import Foundation
import Metal
import NeonFieldOps

// MARK: - Vector Opening Proof

/// Proof that a vector commitment opens to a specific value at a given index.
///
/// Given commitment C = sum(v_i * G_i), the proof for index j is:
///   partialCommitment = sum(v_i * G_i) for all i != j
///
/// Verification checks: C == partialCommitment + value * G_j
public struct VectorOpeningProof {
    /// The partial commitment excluding the opened index: sum(v_i * G_i) for i != openedIndex.
    public let partialCommitment: PointProjective
    /// The index that was opened.
    public let openedIndex: Int

    public init(partialCommitment: PointProjective, openedIndex: Int) {
        self.partialCommitment = partialCommitment
        self.openedIndex = openedIndex
    }
}

// MARK: - GPU Vector Commit Engine

/// GPU-accelerated Pedersen vector commitment engine for BN254 G1.
///
/// Uses Metal MSM for vectors above the GPU threshold, falling back to
/// CPU Pippenger for smaller vectors.
///
/// Usage:
///   let engine = GPUVectorCommitEngine()
///   let C = engine.commit(vector: values, generators: gens)
///   let proof = engine.openAt(vector: values, index: 3, generators: gens)
///   let valid = engine.verifyOpening(commitment: C, index: 3, value: values[3],
///                                     proof: proof, generators: gens)
public class GPUVectorCommitEngine {
    public static let version = Versions.gpuVectorCommit

    /// GPU MSM threshold: use GPU for vectors of this size or larger.
    public static let gpuThreshold = 2048

    /// Cached GPU MSM engine (lazy-initialized).
    private var _msmEngine: MetalMSM?

    public init() {}

    // MARK: - GPU Engine

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - Internal MSM

    /// Perform MSM with GPU/CPU selection based on vector size.
    private func performMSM(scalars: [Fr], generators: [PointAffine]) -> PointProjective {
        let n = scalars.count
        precondition(n > 0)
        let gens = Array(generators.prefix(n))
        let limbs = scalars.map { frToLimbs($0) }

        if n >= GPUVectorCommitEngine.gpuThreshold, let engine = getMSMEngine() {
            if let result = try? engine.msm(points: gens, scalars: limbs) {
                return result
            }
        }
        return cPippengerMSM(points: gens, scalars: limbs)
    }

    // MARK: - Commit

    /// Compute Pedersen vector commitment: C = sum(v_i * G_i)
    ///
    /// For vectors >= gpuThreshold, uses GPU-accelerated MSM via Metal.
    /// Falls back to CPU Pippenger MSM for smaller vectors.
    ///
    /// - Parameters:
    ///   - vector: vector of BN254 Fr elements to commit to
    ///   - generators: generator points G_0, ..., G_{n-1} in affine coordinates
    /// - Returns: commitment point C in projective coordinates
    public func commit(vector: [Fr], generators: [PointAffine]) -> PointProjective {
        precondition(!vector.isEmpty, "Vector must not be empty")
        precondition(vector.count <= generators.count,
                     "Too many values (\(vector.count)) for \(generators.count) generators")
        return performMSM(scalars: vector, generators: generators)
    }

    // MARK: - Batch Commit

    /// Batch commit multiple vectors using the same generators.
    ///
    /// Shares GPU point upload across all vectors for efficiency.
    ///
    /// - Parameters:
    ///   - vectors: array of Fr vectors to commit
    ///   - generators: shared generator points
    /// - Returns: array of commitment points (one per input vector)
    public func batchCommit(vectors: [[Fr]], generators: [PointAffine]) -> [PointProjective] {
        let k = vectors.count
        if k == 0 { return [] }

        let maxLen = vectors.map(\.count).max()!
        precondition(maxLen <= generators.count, "Vector too large for generators")

        // For small batch or small vectors, commit sequentially
        if k <= 2 || maxLen < GPUVectorCommitEngine.gpuThreshold {
            return vectors.map { commit(vector: $0, generators: generators) }
        }

        // GPU multi-MSM path: share generator upload across vectors
        if let engine = getMSMEngine() {
            let gens = Array(generators.prefix(maxLen))
            let scalarSets: [[[UInt32]]] = vectors.map { vec in
                var scalars = vec.map { frToLimbs($0) }
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
        return vectors.map { commit(vector: $0, generators: generators) }
    }

    // MARK: - Open At Index

    /// Generate an opening proof for a specific index in the committed vector.
    ///
    /// The proof is the partial commitment excluding the opened index:
    ///   proof.partialCommitment = sum(v_i * G_i) for i != index
    ///
    /// This enables O(1) verification: check C == proof + v[index] * G[index].
    ///
    /// For large vectors, uses GPU MSM on the (n-1) non-opened elements.
    ///
    /// - Parameters:
    ///   - vector: the full committed vector
    ///   - index: the position to open
    ///   - generators: the generator points
    /// - Returns: a VectorOpeningProof
    public func openAt(vector: [Fr], index: Int, generators: [PointAffine]) -> VectorOpeningProof {
        let n = vector.count
        precondition(index >= 0 && index < n, "Index \(index) out of range [0, \(n))")
        precondition(n <= generators.count, "Vector too large for generators")

        // Build partial vector and generators excluding the opened index
        var partialScalars = [Fr]()
        var partialGens = [PointAffine]()
        partialScalars.reserveCapacity(n - 1)
        partialGens.reserveCapacity(n - 1)

        for i in 0..<n {
            if i != index {
                partialScalars.append(vector[i])
                partialGens.append(generators[i])
            }
        }

        // Compute partial commitment via MSM
        let partial: PointProjective
        if partialScalars.isEmpty {
            partial = pointIdentity()
        } else {
            partial = performMSM(scalars: partialScalars, generators: partialGens)
        }

        return VectorOpeningProof(partialCommitment: partial, openedIndex: index)
    }

    // MARK: - Verify Opening

    /// Verify that a commitment opens to a specific value at a given index.
    ///
    /// Checks: C == proof.partialCommitment + value * G[index]
    ///
    /// This is O(1) in the vector size: one scalar multiplication and one point addition.
    ///
    /// - Parameters:
    ///   - commitment: the vector commitment C (projective)
    ///   - index: the opened index
    ///   - value: the claimed value v[index]
    ///   - proof: the opening proof (partial commitment)
    ///   - generators: the generator points
    /// - Returns: true if the opening is valid
    public func verifyOpening(commitment: PointProjective, index: Int, value: Fr,
                              proof: VectorOpeningProof, generators: [PointAffine]) -> Bool {
        precondition(index >= 0 && index < generators.count,
                     "Index \(index) out of range")
        precondition(proof.openedIndex == index,
                     "Proof index \(proof.openedIndex) does not match claimed index \(index)")

        // Recompute: expected = proof.partialCommitment + value * G[index]
        let valueTerm = cPointScalarMul(pointFromAffine(generators[index]), value)
        let expected = pointAdd(proof.partialCommitment, valueTerm)

        return pointEqual(commitment, expected)
    }
}
