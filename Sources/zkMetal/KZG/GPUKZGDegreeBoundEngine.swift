// GPUKZGDegreeBoundEngine — GPU-accelerated KZG degree bound proof engine
//
// Implements degree bound proofs for KZG polynomial commitments. In many ZK proof
// systems (Plonk, Marlin, Sonic), it is necessary to prove that a committed polynomial
// has degree at most d. This engine provides:
//
//   1. Degree bound proofs via shifted commitments:
//      To prove deg(f) <= d, commit to x^(D-d) * f(x) where D is the max SRS degree.
//      The shifted polynomial has degree at most D, which is verifiable against the SRS.
//      If deg(f) > d, the shifted polynomial would have degree > D and the commitment
//      would be inconsistent with the SRS.
//
//   2. Batch degree bound verification:
//      Check multiple polynomials' degree bounds simultaneously using random linear
//      combination. Given N claims (f_i, d_i), derive challenge r and verify:
//        sum_i r^i * ([x^(D-d_i) * f_i(s)] - [s^(D-d_i)] * C_i) == O
//
//   3. GPU-accelerated polynomial shifting:
//      Multiply polynomial coefficients by x^k (prepend k zeros) and compute
//      the shifted commitment via MSM on the shifted SRS points.
//
//   4. Integration with multi-point opening proofs that include degree bound checks:
//      Combined opening + degree bound verification in a single batch.
//
//   5. SRS trimming for degree-specific setups:
//      Extract the minimal SRS subset needed for a given maximum polynomial degree.
//
// The pairing equation for degree bound verification is:
//   e([shifted_f(s)]_1, [1]_2) == e(C_f, [s^(D-d)]_2)
// Which in SRS-secret testing mode becomes:
//   [shifted_f(s)]_1 == [s^(D-d)] * C_f
//
// GPU acceleration is used for:
//   - MSM when computing shifted commitments (large polynomial shifting)
//   - Batch linear combination of degree bound claims
//   - Polynomial coefficient manipulation for shifting

import Foundation
import Metal
import NeonFieldOps

// MARK: - Degree Bound Proof Structures

/// A proof that a committed polynomial has degree at most `degreeBound`.
///
/// Contains the shifted commitment [x^(D - degreeBound) * f(x)] evaluated at the
/// SRS secret point s, plus metadata for verification.
public struct DegreeBoundProof {
    /// The original polynomial commitment C = [f(s)]_1
    public let commitment: PointProjective
    /// The shifted commitment C_shifted = [s^(D-d) * f(s)]_1
    public let shiftedCommitment: PointProjective
    /// The claimed degree bound d (i.e., deg(f) <= d)
    public let degreeBound: Int
    /// The maximum SRS degree D
    public let maxDegree: Int

    public init(commitment: PointProjective, shiftedCommitment: PointProjective,
                degreeBound: Int, maxDegree: Int) {
        self.commitment = commitment
        self.shiftedCommitment = shiftedCommitment
        self.degreeBound = degreeBound
        self.maxDegree = maxDegree
    }
}

/// A combined degree bound + evaluation proof.
///
/// Proves both that deg(f) <= d and that f(z) = y.
public struct DegreeBoundOpeningProof {
    /// The degree bound proof component
    public let degreeBoundProof: DegreeBoundProof
    /// The evaluation point z
    public let point: Fr
    /// The claimed evaluation y = f(z)
    public let evaluation: Fr
    /// The KZG witness for the opening: [q(s)]_1 where q(x) = (f(x) - y)/(x - z)
    public let witness: PointProjective

    public init(degreeBoundProof: DegreeBoundProof, point: Fr,
                evaluation: Fr, witness: PointProjective) {
        self.degreeBoundProof = degreeBoundProof
        self.point = point
        self.evaluation = evaluation
        self.witness = witness
    }
}

/// A batch degree bound claim for simultaneous verification.
public struct BatchDegreeBoundClaim {
    /// Polynomial commitments [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Shifted commitments [C_shifted_0, ..., C_shifted_{N-1}]
    public let shiftedCommitments: [PointProjective]
    /// Degree bounds [d_0, ..., d_{N-1}]
    public let degreeBounds: [Int]
    /// Maximum SRS degree D
    public let maxDegree: Int

    public init(commitments: [PointProjective], shiftedCommitments: [PointProjective],
                degreeBounds: [Int], maxDegree: Int) {
        self.commitments = commitments
        self.shiftedCommitments = shiftedCommitments
        self.degreeBounds = degreeBounds
        self.maxDegree = maxDegree
    }

    /// Number of claims in this batch.
    public var count: Int { commitments.count }

    /// Validate that all arrays have the same length.
    public var isWellFormed: Bool {
        let n = commitments.count
        return n == shiftedCommitments.count && n == degreeBounds.count
    }
}

/// Result of a batch degree bound verification with diagnostics.
public struct DegreeBoundVerifyResult {
    /// Whether all degree bound claims passed verification.
    public let isValid: Bool
    /// Number of claims verified.
    public let claimCount: Int
    /// Whether GPU MSM was used.
    public let usedGPU: Bool
    /// Elapsed time in seconds.
    public let elapsedSeconds: Double
    /// Human-readable summary.
    public let summary: String

    public init(isValid: Bool, claimCount: Int, usedGPU: Bool,
                elapsedSeconds: Double, summary: String) {
        self.isValid = isValid
        self.claimCount = claimCount
        self.usedGPU = usedGPU
        self.elapsedSeconds = elapsedSeconds
        self.summary = summary
    }
}

/// Configuration for a degree-trimmed SRS.
///
/// Contains the subset of SRS points needed for polynomials up to a specific degree,
/// along with the shift parameters for degree bound proofs.
public struct TrimmedSRS {
    /// The trimmed G1 SRS points [G, sG, ..., s^(trimDegree-1) G]
    public let g1Points: [PointAffine]
    /// The maximum degree this trimmed SRS supports
    public let maxDegree: Int
    /// Original (full) SRS degree for reference
    public let fullDegree: Int
    /// Precomputed shift factors [s^1, s^2, ...] for common shift amounts
    public let shiftFactors: [Int: Fr]

    public init(g1Points: [PointAffine], maxDegree: Int, fullDegree: Int,
                shiftFactors: [Int: Fr]) {
        self.g1Points = g1Points
        self.maxDegree = maxDegree
        self.fullDegree = fullDegree
        self.shiftFactors = shiftFactors
    }
}

/// A combined opening + degree bound proof for multi-point scenarios.
public struct MultiPointDegreeBoundProof {
    /// Per-polynomial degree bound proofs
    public let degreeBoundProofs: [DegreeBoundProof]
    /// Per-polynomial opening proofs (witness points)
    public let witnesses: [PointProjective]
    /// Evaluation points
    public let points: [Fr]
    /// Evaluations f_i(z_i)
    public let evaluations: [Fr]
    /// Combined batch proof point (random linear combination)
    public let batchProof: PointProjective
    /// The batching challenge used
    public let gamma: Fr

    public init(degreeBoundProofs: [DegreeBoundProof], witnesses: [PointProjective],
                points: [Fr], evaluations: [Fr], batchProof: PointProjective, gamma: Fr) {
        self.degreeBoundProofs = degreeBoundProofs
        self.witnesses = witnesses
        self.points = points
        self.evaluations = evaluations
        self.batchProof = batchProof
        self.gamma = gamma
    }
}

// MARK: - GPU KZG Degree Bound Engine

/// GPU-accelerated KZG degree bound proof engine.
///
/// Provides:
///   - Degree bound proof generation via shifted commitments
///   - Combined degree bound + evaluation proofs
///   - Batch degree bound verification with GPU MSM
///   - Multi-point opening proofs with degree bound checks
///   - SRS trimming for degree-specific setups
///   - Polynomial shifting (multiply by x^k) with GPU acceleration
///
/// Usage:
///   ```swift
///   let engine = try GPUKZGDegreeBoundEngine(srs: srs)
///   let proof = try engine.proveDegreeBound(polynomial: coeffs, degreeBound: 16)
///   let valid = engine.verifyDegreeBound(proof: proof, srsSecret: secret)
///   ```
public final class GPUKZGDegreeBoundEngine {

    /// GPU MSM threshold: below this count, CPU scalar-mul is faster.
    private static let GPU_THRESHOLD = 64

    /// The Metal MSM engine for GPU-accelerated multi-scalar multiplication.
    public let msmEngine: MetalMSM

    /// SRS points [G, sG, s^2 G, ..., s^(D-1) G] in affine form.
    public let srs: [PointAffine]

    /// Maximum SRS degree D (equals srs.count).
    public var maxDegree: Int { srs.count }

    /// Internal KZG engine for commitments and openings.
    private let kzgEngine: KZGEngine

    // MARK: - Initialization

    /// Create a degree bound engine with an SRS.
    ///
    /// - Parameter srs: Structured reference string [G, sG, s^2 G, ...] in affine form.
    /// - Throws: If Metal device or MSM pipeline initialization fails.
    public init(srs: [PointAffine]) throws {
        self.srs = srs
        self.msmEngine = try MetalMSM()
        self.kzgEngine = try KZGEngine(srs: srs)
    }

    /// Create a degree bound engine with pre-existing engines.
    ///
    /// Avoids GPU pipeline re-creation when the caller already has engines.
    ///
    /// - Parameters:
    ///   - srs: Structured reference string.
    ///   - msmEngine: Pre-existing Metal MSM engine.
    ///   - kzgEngine: Pre-existing KZG engine.
    public init(srs: [PointAffine], msmEngine: MetalMSM, kzgEngine: KZGEngine) {
        self.srs = srs
        self.msmEngine = msmEngine
        self.kzgEngine = kzgEngine
    }

    // MARK: - Polynomial Shifting

    /// Shift a polynomial by multiplying by x^k.
    ///
    /// Given f(x) = a_0 + a_1*x + ... + a_n*x^n, computes
    /// g(x) = x^k * f(x) = a_0*x^k + a_1*x^{k+1} + ... + a_n*x^{n+k}
    ///
    /// In coefficient representation, this prepends k zero coefficients.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array [a_0, a_1, ..., a_n].
    ///   - shift: The power k to shift by.
    /// - Returns: Shifted coefficient array [0, ..., 0, a_0, a_1, ..., a_n] of length n+k+1.
    public func shiftPolynomial(_ polynomial: [Fr], by shift: Int) -> [Fr] {
        guard shift > 0 else { return polynomial }
        guard !polynomial.isEmpty else { return [] }
        var result = [Fr](repeating: Fr.zero, count: shift + polynomial.count)
        for i in 0..<polynomial.count {
            result[shift + i] = polynomial[i]
        }
        return result
    }

    /// Compute the shifted commitment for a degree bound proof.
    ///
    /// For a polynomial f(x) of degree at most d, and max SRS degree D,
    /// computes [x^(D-d-1) * f(x)] evaluated at the SRS secret s.
    ///
    /// This is equivalent to MSM(SRS[D-d-1..D-1], coefficients).
    /// The shifted SRS points are SRS[D-d-1], SRS[D-d], ..., SRS[D-d-1+n-1].
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array of f(x).
    ///   - degreeBound: Claimed degree bound d.
    /// - Returns: The shifted commitment point.
    /// - Throws: If the polynomial is too large for the SRS or the degree bound is invalid.
    public func computeShiftedCommitment(_ polynomial: [Fr], degreeBound: Int) throws -> PointProjective {
        let n = polynomial.count
        let D = maxDegree

        guard degreeBound >= 0 else {
            throw MSMError.invalidInput
        }
        guard n <= degreeBound + 1 else {
            // Polynomial exceeds claimed degree bound
            throw MSMError.invalidInput
        }

        let shift = D - degreeBound - 1
        guard shift >= 0 else {
            throw MSMError.invalidInput
        }
        guard shift + n <= D else {
            throw MSMError.invalidInput
        }

        // Use shifted SRS points: SRS[shift], SRS[shift+1], ..., SRS[shift+n-1]
        let shiftedPoints = Array(srs[shift..<(shift + n)])

        if n <= GPUKZGDegreeBoundEngine.GPU_THRESHOLD {
            let flatLimbs = batchFrToFlatLimbs(polynomial)
            return cPippengerMSMFlat(points: shiftedPoints, flatScalars: flatLimbs)
        } else {
            let scalars = batchFrToLimbs(polynomial)
            return try msmEngine.msm(points: shiftedPoints, scalars: scalars)
        }
    }

    /// Compute a shifted commitment using a pre-trimmed SRS.
    ///
    /// Avoids re-slicing the SRS for repeated operations with the same degree bound.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array.
    ///   - trimmedSRS: Pre-trimmed SRS with appropriate shift points.
    ///   - shiftOffset: The starting index in the trimmed SRS to use.
    /// - Returns: The shifted commitment point.
    public func computeShiftedCommitmentTrimmed(
        _ polynomial: [Fr],
        trimmedSRS: TrimmedSRS,
        shiftOffset: Int
    ) throws -> PointProjective {
        let n = polynomial.count
        guard shiftOffset + n <= trimmedSRS.g1Points.count else {
            throw MSMError.invalidInput
        }

        let pts = Array(trimmedSRS.g1Points[shiftOffset..<(shiftOffset + n)])

        if n <= GPUKZGDegreeBoundEngine.GPU_THRESHOLD {
            let flatLimbs = batchFrToFlatLimbs(polynomial)
            return cPippengerMSMFlat(points: pts, flatScalars: flatLimbs)
        } else {
            let scalars = batchFrToLimbs(polynomial)
            return try msmEngine.msm(points: pts, scalars: scalars)
        }
    }

    // MARK: - Degree Bound Proof Generation

    /// Generate a degree bound proof for a polynomial.
    ///
    /// Proves that deg(f) <= degreeBound by computing the shifted commitment
    /// C_shifted = [s^(D - d - 1) * f(s)]_1 and returning it alongside the
    /// original commitment.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array of f(x).
    ///   - degreeBound: Claimed degree bound d (deg(f) <= d).
    /// - Returns: A DegreeBoundProof containing both commitments.
    /// - Throws: If polynomial degree exceeds the degree bound or SRS size.
    public func proveDegreeBound(polynomial: [Fr], degreeBound: Int) throws -> DegreeBoundProof {
        let commitment = try kzgEngine.commit(polynomial)
        let shiftedCommitment = try computeShiftedCommitment(polynomial, degreeBound: degreeBound)

        return DegreeBoundProof(
            commitment: commitment,
            shiftedCommitment: shiftedCommitment,
            degreeBound: degreeBound,
            maxDegree: maxDegree
        )
    }

    /// Generate a combined degree bound + evaluation proof.
    ///
    /// Proves both that deg(f) <= d and that f(z) = y, combining a degree
    /// bound proof with a standard KZG opening.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array of f(x).
    ///   - degreeBound: Claimed degree bound d.
    ///   - point: Evaluation point z.
    /// - Returns: A combined opening + degree bound proof.
    public func proveDegreeBoundOpening(
        polynomial: [Fr],
        degreeBound: Int,
        point: Fr
    ) throws -> DegreeBoundOpeningProof {
        let dbProof = try proveDegreeBound(polynomial: polynomial, degreeBound: degreeBound)
        let kzgProof = try kzgEngine.open(polynomial, at: point)

        return DegreeBoundOpeningProof(
            degreeBoundProof: dbProof,
            point: point,
            evaluation: kzgProof.evaluation,
            witness: kzgProof.witness
        )
    }

    /// Generate degree bound proofs for multiple polynomials in a batch.
    ///
    /// Each polynomial may have a different degree bound. The commitments are
    /// computed using batch MSM for efficiency.
    ///
    /// - Parameters:
    ///   - polynomials: Array of coefficient arrays.
    ///   - degreeBounds: Per-polynomial degree bounds.
    /// - Returns: Array of individual DegreeBoundProof for each polynomial.
    public func batchProveDegreeBound(
        polynomials: [[Fr]],
        degreeBounds: [Int]
    ) throws -> [DegreeBoundProof] {
        guard polynomials.count == degreeBounds.count else {
            throw MSMError.invalidInput
        }

        let n = polynomials.count
        let commitments = try kzgEngine.batchCommit(polynomials)

        var proofs = [DegreeBoundProof]()
        proofs.reserveCapacity(n)

        for i in 0..<n {
            let shifted = try computeShiftedCommitment(polynomials[i], degreeBound: degreeBounds[i])
            proofs.append(DegreeBoundProof(
                commitment: commitments[i],
                shiftedCommitment: shifted,
                degreeBound: degreeBounds[i],
                maxDegree: maxDegree
            ))
        }

        return proofs
    }

    // MARK: - Degree Bound Verification

    /// Verify a single degree bound proof using the SRS secret.
    ///
    /// Checks: C_shifted == [s^(D-d-1)] * C
    ///
    /// This verifies that the shifted commitment is consistent with the original
    /// commitment under the shift factor s^(D-d-1). If the polynomial had degree > d,
    /// the shifted polynomial would have degree > D-1, and the commitment would be
    /// inconsistent with the D-point SRS.
    ///
    /// - Parameters:
    ///   - proof: The degree bound proof to verify.
    ///   - srsSecret: The SRS toxic waste scalar s (testing mode only).
    /// - Returns: true if the degree bound proof is valid.
    public func verifyDegreeBound(proof: DegreeBoundProof, srsSecret: Fr) -> Bool {
        let D = proof.maxDegree
        let d = proof.degreeBound

        guard d >= 0, D > d else { return false }

        // Compute shift factor s^(D - d - 1)
        let shift = D - d - 1
        let shiftFactor = computePower(srsSecret, exponent: shift)

        // Expected: C_shifted == shiftFactor * C
        let expected = cPointScalarMul(proof.commitment, shiftFactor)

        return pointsEqual(proof.shiftedCommitment, expected)
    }

    /// Verify a combined degree bound + evaluation proof.
    ///
    /// Checks both:
    ///   1. Degree bound: C_shifted == [s^(D-d-1)] * C
    ///   2. Evaluation: C - [y]*G == [s - z] * witness
    ///
    /// - Parameters:
    ///   - proof: The combined proof.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if both checks pass.
    public func verifyDegreeBoundOpening(
        proof: DegreeBoundOpeningProof,
        srsSecret: Fr
    ) -> Bool {
        // Check 1: Degree bound
        guard verifyDegreeBound(proof: proof.degreeBoundProof, srsSecret: srsSecret) else {
            return false
        }

        // Check 2: Evaluation opening
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, proof.evaluation)
        let lhs = pointAdd(proof.degreeBoundProof.commitment, pointNeg(yG))

        let sMz = frSub(srsSecret, proof.point)
        let rhs = cPointScalarMul(proof.witness, sMz)

        return pointsEqual(lhs, rhs)
    }

    /// Batch verify multiple degree bound proofs using random linear combination.
    ///
    /// Derives a batching challenge r from a Fiat-Shamir transcript and checks:
    ///   sum_i r^i * (C_shifted_i - [s^(D-d_i-1)] * C_i) == O  (point at infinity)
    ///
    /// This reduces N individual checks to two MSMs + one comparison.
    ///
    /// - Parameters:
    ///   - proofs: Array of degree bound proofs.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all degree bound proofs are valid.
    public func batchVerifyDegreeBound(
        proofs: [DegreeBoundProof],
        srsSecret: Fr
    ) throws -> Bool {
        guard !proofs.isEmpty else { return true }

        let n = proofs.count
        if n == 1 {
            return verifyDegreeBound(proof: proofs[0], srsSecret: srsSecret)
        }

        // Derive batching challenge
        let r = deriveBatchChallenge(proofs: proofs)

        // Compute powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        // Compute: sum_i r^i * C_shifted_i
        var lhsAccum = pointIdentity()
        for i in 0..<n {
            lhsAccum = pointAdd(lhsAccum, cPointScalarMul(proofs[i].shiftedCommitment, rPowers[i]))
        }

        // Compute: sum_i r^i * [s^(D-d_i-1)] * C_i
        var rhsAccum = pointIdentity()
        for i in 0..<n {
            let shift = proofs[i].maxDegree - proofs[i].degreeBound - 1
            let shiftFactor = computePower(srsSecret, exponent: shift)
            let scaledCommitment = cPointScalarMul(proofs[i].commitment, shiftFactor)
            rhsAccum = pointAdd(rhsAccum, cPointScalarMul(scaledCommitment, rPowers[i]))
        }

        return pointsEqual(lhsAccum, rhsAccum)
    }

    /// Batch verify degree bounds using GPU MSM for large batches.
    ///
    /// When the number of claims exceeds the GPU threshold, the two accumulations
    /// (shifted commitments and expected commitments) are computed via GPU MSM.
    ///
    /// - Parameters:
    ///   - batch: Batch degree bound claim.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all claims pass.
    public func batchVerifyDegreeBoundClaim(
        batch: BatchDegreeBoundClaim,
        srsSecret: Fr
    ) throws -> Bool {
        guard batch.isWellFormed else { return false }
        guard batch.count > 0 else { return true }

        let n = batch.count

        // Single claim: direct check
        if n == 1 {
            let proof = DegreeBoundProof(
                commitment: batch.commitments[0],
                shiftedCommitment: batch.shiftedCommitments[0],
                degreeBound: batch.degreeBounds[0],
                maxDegree: batch.maxDegree
            )
            return verifyDegreeBound(proof: proof, srsSecret: srsSecret)
        }

        // Derive batching challenge
        let proofs = (0..<n).map { i in
            DegreeBoundProof(
                commitment: batch.commitments[i],
                shiftedCommitment: batch.shiftedCommitments[i],
                degreeBound: batch.degreeBounds[i],
                maxDegree: batch.maxDegree
            )
        }
        let r = deriveBatchChallenge(proofs: proofs)

        // Powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        // Use GPU MSM for the shifted commitment accumulation when batch is large
        let useGPU = n >= GPUKZGDegreeBoundEngine.GPU_THRESHOLD

        // LHS: sum_i r^i * C_shifted_i
        let lhs: PointProjective
        if useGPU {
            let shiftedAffs = batch.shiftedCommitments.map { batchToAffine([$0])[0] }
            lhs = try msmEngine.msm(points: shiftedAffs, scalars: batchFrToLimbs(rPowers))
        } else {
            lhs = cpuMSMProj(points: batch.shiftedCommitments, scalars: rPowers)
        }

        // RHS: sum_i r^i * [s^(D-d_i-1)] * C_i
        // Combine the shift factor and r^i into a single scalar per claim
        var combinedScalars = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            let shift = batch.maxDegree - batch.degreeBounds[i] - 1
            let shiftFactor = computePower(srsSecret, exponent: shift)
            combinedScalars[i] = frMul(rPowers[i], shiftFactor)
        }

        let rhs: PointProjective
        if useGPU {
            let commitAffs = batch.commitments.map { batchToAffine([$0])[0] }
            rhs = try msmEngine.msm(points: commitAffs, scalars: batchFrToLimbs(combinedScalars))
        } else {
            rhs = cpuMSMProj(points: batch.commitments, scalars: combinedScalars)
        }

        return pointsEqual(lhs, rhs)
    }

    /// Batch verify with detailed diagnostics.
    public func batchVerifyDetailed(
        proofs: [DegreeBoundProof],
        srsSecret: Fr
    ) throws -> DegreeBoundVerifyResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let n = proofs.count
        let useGPU = n >= GPUKZGDegreeBoundEngine.GPU_THRESHOLD
        let valid = try batchVerifyDegreeBound(proofs: proofs, srsSecret: srsSecret)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let summary = "Degree bound: verified \(n) claim(s) in " +
            "\(String(format: "%.3f", elapsed * 1000))ms [GPU=\(useGPU), valid=\(valid)]"

        return DegreeBoundVerifyResult(
            isValid: valid,
            claimCount: n,
            usedGPU: useGPU,
            elapsedSeconds: elapsed,
            summary: summary
        )
    }

    // MARK: - Multi-Point Opening with Degree Bounds

    /// Generate a multi-point opening proof with degree bound checks.
    ///
    /// For each polynomial f_i opened at point z_i with degree bound d_i:
    ///   1. Compute degree bound proof (shifted commitment)
    ///   2. Compute evaluation y_i = f_i(z_i)
    ///   3. Compute KZG witness for the opening
    ///   4. Combine all into a batch proof via random linear combination
    ///
    /// - Parameters:
    ///   - polynomials: Array of coefficient arrays.
    ///   - points: Evaluation points (one per polynomial).
    ///   - degreeBounds: Degree bounds (one per polynomial).
    /// - Returns: A MultiPointDegreeBoundProof.
    public func proveMultiPointWithDegreeBound(
        polynomials: [[Fr]],
        points: [Fr],
        degreeBounds: [Int]
    ) throws -> MultiPointDegreeBoundProof {
        let n = polynomials.count
        guard n == points.count, n == degreeBounds.count, n > 0 else {
            throw MSMError.invalidInput
        }

        // Step 1: Generate degree bound proofs
        let dbProofs = try batchProveDegreeBound(polynomials: polynomials, degreeBounds: degreeBounds)

        // Step 2: Compute evaluations and KZG witnesses
        var evaluations = [Fr]()
        var witnesses = [PointProjective]()
        evaluations.reserveCapacity(n)
        witnesses.reserveCapacity(n)

        for i in 0..<n {
            let kzgProof = try kzgEngine.open(polynomials[i], at: points[i])
            evaluations.append(kzgProof.evaluation)
            witnesses.append(kzgProof.witness)
        }

        // Step 3: Derive gamma from transcript
        let transcript = Transcript(label: "degree-bound-multi-open", backend: .poseidon2)
        for i in 0..<n {
            absorbPoint(dbProofs[i].commitment, into: transcript)
            absorbPoint(dbProofs[i].shiftedCommitment, into: transcript)
            transcript.absorb(points[i])
            transcript.absorb(evaluations[i])
        }
        let gamma = transcript.squeeze()

        // Step 4: Compute combined batch proof
        // batchProof = sum_i gamma^i * witness_i
        var batchProof = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            batchProof = pointAdd(batchProof, cPointScalarMul(witnesses[i], gammaPow))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, gamma)
            }
        }

        return MultiPointDegreeBoundProof(
            degreeBoundProofs: dbProofs,
            witnesses: witnesses,
            points: points,
            evaluations: evaluations,
            batchProof: batchProof,
            gamma: gamma
        )
    }

    /// Verify a multi-point opening proof with degree bound checks.
    ///
    /// Checks both degree bounds and evaluation openings for all polynomials.
    ///
    /// - Parameters:
    ///   - proof: The multi-point degree bound proof.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all degree bounds and openings are valid.
    public func verifyMultiPointWithDegreeBound(
        proof: MultiPointDegreeBoundProof,
        srsSecret: Fr
    ) -> Bool {
        let n = proof.degreeBoundProofs.count
        guard n == proof.witnesses.count,
              n == proof.points.count,
              n == proof.evaluations.count else {
            return false
        }

        // Check all degree bounds
        for dbp in proof.degreeBoundProofs {
            if !verifyDegreeBound(proof: dbp, srsSecret: srsSecret) {
                return false
            }
        }

        // Check all openings
        let g1 = pointFromAffine(srs[0])
        for i in 0..<n {
            let yG = cPointScalarMul(g1, proof.evaluations[i])
            let lhs = pointAdd(proof.degreeBoundProofs[i].commitment, pointNeg(yG))
            let sMz = frSub(srsSecret, proof.points[i])
            let rhs = cPointScalarMul(proof.witnesses[i], sMz)
            if !pointsEqual(lhs, rhs) {
                return false
            }
        }

        // Verify batch proof consistency: batchProof == sum_i gamma^i * witness_i
        var expectedBatch = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            expectedBatch = pointAdd(expectedBatch, cPointScalarMul(proof.witnesses[i], gammaPow))
            if i < n - 1 {
                gammaPow = frMul(gammaPow, proof.gamma)
            }
        }

        return pointsEqual(proof.batchProof, expectedBatch)
    }

    // MARK: - SRS Trimming

    /// Trim the SRS to support polynomials up to a given degree.
    ///
    /// Extracts the first (degree + 1) SRS points and optionally precomputes
    /// shift factors for common degree bound shifts.
    ///
    /// - Parameters:
    ///   - degree: Maximum polynomial degree to support.
    ///   - precomputeShifts: Array of shift amounts to precompute (optional).
    ///   - srsSecret: SRS secret for shift factor precomputation (optional).
    /// - Returns: A TrimmedSRS, or nil if degree exceeds SRS size.
    public func trimSRS(
        degree: Int,
        precomputeShifts: [Int] = [],
        srsSecret: Fr? = nil
    ) -> TrimmedSRS? {
        let neededPoints = degree + 1
        guard neededPoints <= srs.count else { return nil }

        let trimmedPoints = Array(srs.prefix(neededPoints))

        var shiftFactors = [Int: Fr]()
        if let secret = srsSecret {
            for shift in precomputeShifts {
                if shift >= 0 && shift < maxDegree {
                    shiftFactors[shift] = computePower(secret, exponent: shift)
                }
            }
        }

        return TrimmedSRS(
            g1Points: trimmedPoints,
            maxDegree: degree,
            fullDegree: maxDegree,
            shiftFactors: shiftFactors
        )
    }

    /// Extract a sub-SRS suitable for a specific circuit with degree bound constraints.
    ///
    /// Computes the minimum SRS size needed given both the maximum polynomial degree
    /// and the maximum shift required for degree bound proofs.
    ///
    /// - Parameters:
    ///   - maxPolyDegree: Maximum degree of any polynomial in the circuit.
    ///   - maxDegreeBound: Maximum degree bound to prove.
    /// - Returns: A TrimmedSRS, or nil if the SRS is too small.
    public func trimSRSForCircuit(
        maxPolyDegree: Int,
        maxDegreeBound: Int
    ) -> TrimmedSRS? {
        // We need enough SRS points to support both:
        //   - Direct commitment: needs maxPolyDegree + 1 points
        //   - Shifted commitment: needs shift + maxPolyDegree + 1 points
        //     where shift = D - maxDegreeBound - 1
        // Since we use the same SRS for both, D must be >= maxPolyDegree + 1
        // AND D must be >= maxDegreeBound + 1

        let minDegree = max(maxPolyDegree, maxDegreeBound)
        guard minDegree + 1 <= srs.count else { return nil }

        return trimSRS(degree: minDegree)
    }

    // MARK: - Degree Bound Accumulator

    /// Accumulator for incremental degree bound proof collection and batch verification.
    public final class DegreeBoundAccumulator {
        private var proofs: [DegreeBoundProof] = []

        public init() {}

        /// Add a degree bound proof to the accumulator.
        public func addProof(_ proof: DegreeBoundProof) {
            proofs.append(proof)
        }

        /// Add a proof from components.
        public func addProof(commitment: PointProjective, shiftedCommitment: PointProjective,
                             degreeBound: Int, maxDegree: Int) {
            proofs.append(DegreeBoundProof(
                commitment: commitment,
                shiftedCommitment: shiftedCommitment,
                degreeBound: degreeBound,
                maxDegree: maxDegree
            ))
        }

        /// Number of accumulated proofs.
        public var count: Int { proofs.count }

        /// Clear all accumulated proofs.
        public func clear() { proofs.removeAll() }

        /// Retrieve all accumulated proofs.
        public func getProofs() -> [DegreeBoundProof] { proofs }
    }

    /// Create a new degree bound accumulator.
    public func createAccumulator() -> DegreeBoundAccumulator {
        return DegreeBoundAccumulator()
    }

    /// Verify all proofs in an accumulator.
    public func verifyAccumulator(
        _ accumulator: DegreeBoundAccumulator,
        srsSecret: Fr
    ) throws -> Bool {
        return try batchVerifyDegreeBound(proofs: accumulator.getProofs(), srsSecret: srsSecret)
    }

    // MARK: - Combined Degree Bound + KZG Batch Verification

    /// Verify a batch of combined degree bound + opening proofs.
    ///
    /// Each claim provides a polynomial commitment, shifted commitment, evaluation
    /// point, evaluation value, and KZG witness. The engine verifies both the degree
    /// bound and the evaluation in a single batch operation.
    ///
    /// - Parameters:
    ///   - proofs: Array of combined proofs.
    ///   - srsSecret: The SRS secret s.
    /// - Returns: true if all proofs are valid.
    public func batchVerifyDegreeBoundOpenings(
        proofs: [DegreeBoundOpeningProof],
        srsSecret: Fr
    ) throws -> Bool {
        guard !proofs.isEmpty else { return true }

        let n = proofs.count

        // Phase 1: Batch verify all degree bounds
        let dbProofs = proofs.map { $0.degreeBoundProof }
        guard try batchVerifyDegreeBound(proofs: dbProofs, srsSecret: srsSecret) else {
            return false
        }

        // Phase 2: Batch verify all openings
        let r = deriveOpeningBatchChallenge(proofs: proofs)
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // LHS = sum_i r^i * (C_i - [y_i]*G)
        var lhs = pointIdentity()
        for i in 0..<n {
            let yG = cPointScalarMul(g1, proofs[i].evaluation)
            let diff = pointAdd(proofs[i].degreeBoundProof.commitment, pointNeg(yG))
            lhs = pointAdd(lhs, cPointScalarMul(diff, rPowers[i]))
        }

        // RHS = sum_i r^i * (s - z_i) * witness_i
        var rhs = pointIdentity()
        for i in 0..<n {
            let sMz = frSub(srsSecret, proofs[i].point)
            let combined = frMul(rPowers[i], sMz)
            rhs = pointAdd(rhs, cPointScalarMul(proofs[i].witness, combined))
        }

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Polynomial Degree Utilities

    /// Compute the actual degree of a polynomial (index of highest non-zero coefficient).
    ///
    /// Returns -1 for the zero polynomial.
    public func polynomialDegree(_ coeffs: [Fr]) -> Int {
        let zero = frToInt(Fr.zero)
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            if frToInt(coeffs[i]) != zero {
                return i
            }
        }
        return -1
    }

    /// Check whether a polynomial's actual degree is within a claimed bound.
    ///
    /// This is a local check (no cryptographic proof); useful for prover-side validation
    /// before generating a degree bound proof.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array.
    ///   - degreeBound: Claimed maximum degree.
    /// - Returns: true if the actual degree is <= degreeBound.
    public func checkDegreeBound(_ polynomial: [Fr], degreeBound: Int) -> Bool {
        let actualDeg = polynomialDegree(polynomial)
        return actualDeg <= degreeBound
    }

    /// Pad a polynomial to exactly (degreeBound + 1) coefficients by appending zeros.
    ///
    /// Useful for normalizing polynomial representations before batch operations.
    ///
    /// - Parameters:
    ///   - polynomial: Coefficient array.
    ///   - degreeBound: Target degree bound.
    /// - Returns: Padded coefficient array, or nil if actual degree exceeds bound.
    public func padToDegree(_ polynomial: [Fr], degreeBound: Int) -> [Fr]? {
        guard checkDegreeBound(polynomial, degreeBound: degreeBound) else { return nil }
        let targetLen = degreeBound + 1
        if polynomial.count >= targetLen {
            return Array(polynomial.prefix(targetLen))
        }
        var padded = polynomial
        padded.append(contentsOf: [Fr](repeating: Fr.zero, count: targetLen - polynomial.count))
        return padded
    }

    /// Strip trailing zero coefficients from a polynomial.
    ///
    /// - Parameter polynomial: Coefficient array.
    /// - Returns: Trimmed coefficient array (at least [Fr.zero] for the zero polynomial).
    public func trimPolynomial(_ polynomial: [Fr]) -> [Fr] {
        let zero = frToInt(Fr.zero)
        var lastNonZero = polynomial.count - 1
        while lastNonZero > 0 && frToInt(polynomial[lastNonZero]) == zero {
            lastNonZero -= 1
        }
        return Array(polynomial.prefix(lastNonZero + 1))
    }

    // MARK: - Shift Factor Computation

    /// Compute the shift factor s^k for degree bound proofs.
    ///
    /// Given the SRS secret s and a shift amount k, returns s^k in Fr.
    /// Uses square-and-multiply for efficiency.
    ///
    /// - Parameters:
    ///   - srsSecret: The SRS secret s.
    ///   - degreeBound: The degree bound d (shift = maxDegree - d - 1).
    /// - Returns: The shift factor s^(maxDegree - d - 1).
    public func computeShiftFactor(srsSecret: Fr, degreeBound: Int) -> Fr {
        let shift = maxDegree - degreeBound - 1
        guard shift >= 0 else { return Fr.one }
        return computePower(srsSecret, exponent: shift)
    }

    /// Precompute shift factors for a set of degree bounds.
    ///
    /// Returns a dictionary mapping degree bound -> shift factor.
    /// Useful when verifying multiple proofs with the same set of degree bounds.
    ///
    /// - Parameters:
    ///   - srsSecret: The SRS secret.
    ///   - degreeBounds: Set of degree bounds to precompute.
    /// - Returns: Dictionary mapping degreeBound -> s^(D-d-1).
    public func precomputeShiftFactors(srsSecret: Fr, degreeBounds: [Int]) -> [Int: Fr] {
        var result = [Int: Fr]()
        let uniqueBounds = Set(degreeBounds)
        for d in uniqueBounds {
            let shift = maxDegree - d - 1
            if shift >= 0 {
                result[d] = computePower(srsSecret, exponent: shift)
            }
        }
        return result
    }

    /// Batch verify degree bounds with precomputed shift factors for speed.
    ///
    /// - Parameters:
    ///   - proofs: Degree bound proofs.
    ///   - srsSecret: The SRS secret.
    ///   - shiftFactors: Precomputed shift factors (degreeBound -> s^(D-d-1)).
    /// - Returns: true if all proofs pass.
    public func batchVerifyWithPrecomputedShifts(
        proofs: [DegreeBoundProof],
        srsSecret: Fr,
        shiftFactors: [Int: Fr]
    ) throws -> Bool {
        guard !proofs.isEmpty else { return true }

        let n = proofs.count
        if n == 1 {
            guard let sf = shiftFactors[proofs[0].degreeBound] else {
                return verifyDegreeBound(proof: proofs[0], srsSecret: srsSecret)
            }
            let expected = cPointScalarMul(proofs[0].commitment, sf)
            return pointsEqual(proofs[0].shiftedCommitment, expected)
        }

        let r = deriveBatchChallenge(proofs: proofs)
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        var lhsAccum = pointIdentity()
        var rhsAccum = pointIdentity()

        for i in 0..<n {
            lhsAccum = pointAdd(lhsAccum, cPointScalarMul(proofs[i].shiftedCommitment, rPowers[i]))

            let sf = shiftFactors[proofs[i].degreeBound] ?? computePower(srsSecret, exponent: proofs[i].maxDegree - proofs[i].degreeBound - 1)
            let expected = cPointScalarMul(proofs[i].commitment, sf)
            rhsAccum = pointAdd(rhsAccum, cPointScalarMul(expected, rPowers[i]))
        }

        return pointsEqual(lhsAccum, rhsAccum)
    }

    // MARK: - Internal Helpers

    /// Compute base^exponent in Fr using square-and-multiply.
    private func computePower(_ base: Fr, exponent: Int) -> Fr {
        guard exponent > 0 else { return Fr.one }
        if exponent == 1 { return base }

        var result = Fr.one
        var current = base
        var exp = exponent

        while exp > 0 {
            if exp & 1 == 1 {
                result = frMul(result, current)
            }
            current = frMul(current, current)
            exp >>= 1
        }
        return result
    }

    /// Convert Fr array to flat [UInt32] limbs.
    private func batchFrToFlatLimbs(_ coeffs: [Fr]) -> [UInt32] {
        let n = coeffs.count
        var limbs = [UInt32](repeating: 0, count: n * 8)
        coeffs.withUnsafeBytes { src in
            limbs.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        return limbs
    }

    /// Convert Fr array to [[UInt32]] limbs for GPU MSM.
    private func batchFrToLimbs(_ scalars: [Fr]) -> [[UInt32]] {
        let flat = batchFrToFlatLimbs(scalars)
        let n = scalars.count
        var result = [[UInt32]]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(flat[i * 8..<(i + 1) * 8]))
        }
        return result
    }

    /// CPU MSM fallback for projective points.
    private func cpuMSMProj(points: [PointProjective], scalars: [Fr]) -> PointProjective {
        var result = pointIdentity()
        for i in 0..<points.count {
            result = pointAdd(result, cPointScalarMul(points[i], scalars[i]))
        }
        return result
    }

    /// Compare two projective points for equality via affine conversion.
    private func pointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Absorb a projective point into a transcript.
    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            let xInt = fpToInt(aff[0].x)
            let yInt = fpToInt(aff[0].y)
            transcript.absorb(Fr.from64(xInt))
            transcript.absorb(Fr.from64(yInt))
        }
    }

    /// Derive a batching challenge from degree bound proofs.
    private func deriveBatchChallenge(proofs: [DegreeBoundProof]) -> Fr {
        let transcript = Transcript(label: "degree-bound-batch-verify", backend: .poseidon2)
        for proof in proofs {
            absorbPoint(proof.commitment, into: transcript)
            absorbPoint(proof.shiftedCommitment, into: transcript)
            let dbFr = frFromInt(UInt64(proof.degreeBound))
            transcript.absorb(dbFr)
        }
        return transcript.squeeze()
    }

    /// Derive a batching challenge from combined opening proofs.
    private func deriveOpeningBatchChallenge(proofs: [DegreeBoundOpeningProof]) -> Fr {
        let transcript = Transcript(label: "degree-bound-opening-batch", backend: .poseidon2)
        for proof in proofs {
            absorbPoint(proof.degreeBoundProof.commitment, into: transcript)
            absorbPoint(proof.degreeBoundProof.shiftedCommitment, into: transcript)
            transcript.absorb(proof.point)
            transcript.absorb(proof.evaluation)
            absorbPoint(proof.witness, into: transcript)
        }
        return transcript.squeeze()
    }
}
