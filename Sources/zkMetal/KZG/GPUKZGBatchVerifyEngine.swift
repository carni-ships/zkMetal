// GPUKZGBatchVerifyEngine — GPU-accelerated KZG batch verification engine
//
// Batch verification of multiple KZG polynomial commitment openings using
// random linear combination to reduce N pairing checks to a single aggregated check.
//
// Key algorithms:
//   1. Random linear combination: Given N claims {(C_i, z_i, y_i, pi_i)},
//      derive random challenge r and verify a single combined equation:
//        sum_i r^i * (C_i - [y_i]*G - z_i * pi_i)  == [s] * sum_i r^i * pi_i
//
//   2. Multi-point batch verification: Different evaluation points per commitment.
//      Each claim specifies its own (z_i, y_i) and proof pi_i. The random
//      linear combination collapses all checks into one.
//
//   3. Aggregated pairing computation:
//      Production mode (no SRS secret):
//        e(sum_i r^i * (C_i - [y_i]*G - z_i * pi_i), [1]_2) == e(sum_i r^i * pi_i, [s]_2)
//      Two pairings total, regardless of N.
//
//   4. GPU-accelerated MSM: The two large multi-scalar multiplications are
//      dispatched to Metal GPU when N is above threshold, providing significant
//      speedup for Plonk-scale batch verification (20+ commitments).
//
//   5. Single-polynomial mode: Verify a single opening without batching overhead.
//
//   6. Multi-polynomial batch mode: Verify openings across different polynomials
//      at different points in one shot.
//
// Supports both BN254 ate pairing (production) and algebraic SRS-secret verification
// (testing). The Fiat-Shamir transcript ensures non-interactivity.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Batch Verification Claim

/// A single KZG opening claim for batch verification.
/// Represents the statement: "polynomial with commitment C evaluates to y at point z,
/// with witness proof pi."
public struct BatchVerifyClaim {
    /// Commitment to the polynomial: C = [f(s)]_1
    public let commitment: PointProjective
    /// Evaluation point z
    public let point: Fr
    /// Claimed evaluation value y = f(z)
    public let value: Fr
    /// KZG witness proof: pi = [q(s)]_1 where q(x) = (f(x) - y) / (x - z)
    public let proof: PointProjective

    public init(commitment: PointProjective, point: Fr, value: Fr, proof: PointProjective) {
        self.commitment = commitment
        self.point = point
        self.value = value
        self.proof = proof
    }
}

// MARK: - Batch Verification Result

/// Detailed result of a batch verification, with diagnostics.
public struct BatchVerifyResult {
    /// Whether all claims were verified successfully.
    public let isValid: Bool
    /// Number of claims that were verified.
    public let claimCount: Int
    /// Whether GPU MSM was used for the batch combination.
    public let usedGPU: Bool
    /// Time taken for the batch verification (seconds).
    public let elapsedSeconds: Double
    /// Human-readable description of the verification.
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

// MARK: - Multi-Opening Batch Claim

/// A batch of claims where each polynomial may be opened at a different point.
/// Used for multi-polynomial batch verification.
public struct MultiOpenBatchClaim {
    /// Polynomial commitments [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Evaluation points [z_0, ..., z_{N-1}] (one per commitment)
    public let points: [Fr]
    /// Claimed evaluation values [y_0, ..., y_{N-1}]
    public let values: [Fr]
    /// KZG witness proofs [pi_0, ..., pi_{N-1}]
    public let proofs: [PointProjective]

    public init(commitments: [PointProjective], points: [Fr],
                values: [Fr], proofs: [PointProjective]) {
        self.commitments = commitments
        self.points = points
        self.values = values
        self.proofs = proofs
    }

    /// Number of claims in this batch.
    public var count: Int { commitments.count }

    /// Validate that all arrays have the same length.
    public var isWellFormed: Bool {
        let n = commitments.count
        return n == points.count && n == values.count && n == proofs.count
    }
}

// MARK: - Same-Point Batch Claim

/// Optimized batch claim where all polynomials are opened at the same point.
/// Common in Plonk where 20+ polynomials are opened at zeta.
public struct SamePointBatchClaim {
    /// Polynomial commitments [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Single evaluation point z (shared by all)
    public let point: Fr
    /// Claimed evaluation values [y_0, ..., y_{N-1}]
    public let values: [Fr]
    /// KZG witness proofs [pi_0, ..., pi_{N-1}]
    public let proofs: [PointProjective]

    public init(commitments: [PointProjective], point: Fr,
                values: [Fr], proofs: [PointProjective]) {
        self.commitments = commitments
        self.point = point
        self.values = values
        self.proofs = proofs
    }

    /// Number of claims in this batch.
    public var count: Int { commitments.count }
}

// MARK: - GPU KZG Batch Verify Engine

/// GPU-accelerated KZG batch verification engine.
///
/// Provides:
///   - Single opening verification (fast path, no GPU overhead)
///   - Multi-point batch verification (random linear combination + GPU MSM)
///   - Same-point batch verification (Plonk-optimized)
///   - Multi-polynomial batch mode (different polynomials at different points)
///   - Aggregated pairing computation for production verification
///   - Fiat-Shamir transcript for non-interactive challenge derivation
///   - Detailed verification results with timing and GPU usage info
///
/// Usage:
///   ```swift
///   let engine = try GPUKZGBatchVerifyEngine(srs: srs)
///   let claims = [BatchVerifyClaim(commitment: c, point: z, value: y, proof: pi)]
///   let result = try engine.batchVerify(claims: claims, srsSecret: secret)
///   ```
public final class GPUKZGBatchVerifyEngine {
    public static let version = Versions.gpuKZG

    /// GPU MSM threshold: below this count, CPU scalar-mul is faster.
    private static let GPU_BATCH_THRESHOLD = 4

    /// The Metal MSM engine for GPU-accelerated multi-scalar multiplication.
    public let msmEngine: MetalMSM

    /// SRS points for commitment re-derivation and generator access.
    public let srs: [PointAffine]

    // MARK: - Initialization

    /// Create a batch verification engine with an SRS.
    ///
    /// - Parameter srs: Structured reference string [G, sG, s^2 G, ...] in affine form.
    /// - Throws: If Metal device or MSM pipeline initialization fails.
    public init(srs: [PointAffine]) throws {
        self.msmEngine = try MetalMSM()
        self.srs = srs
    }

    /// Create a batch verification engine with an existing MSM engine.
    ///
    /// Avoids GPU pipeline re-creation when the caller already has a MetalMSM.
    ///
    /// - Parameters:
    ///   - srs: Structured reference string.
    ///   - msmEngine: Pre-existing Metal MSM engine.
    public init(srs: [PointAffine], msmEngine: MetalMSM) {
        self.msmEngine = msmEngine
        self.srs = srs
    }

    // MARK: - Single Claim Verification

    /// Verify a single KZG opening claim.
    ///
    /// Checks: C - [y]*G == [s - z] * pi
    ///
    /// This is the fast path with no batching overhead, used when only one
    /// claim needs verification or as a fallback from the batch path.
    ///
    /// - Parameters:
    ///   - claim: The opening claim to verify.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if the claim is valid.
    public func verifySingle(claim: BatchVerifyClaim, srsSecret: Fr) -> Bool {
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, claim.value)
        let lhs = pointAdd(claim.commitment, pointNeg(yG))

        let sMz = frSub(srsSecret, claim.point)
        let rhs = cPointScalarMul(claim.proof, sMz)

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Batch Verification (Core)

    /// Batch verify N KZG opening claims using random linear combination + GPU MSM.
    ///
    /// Algorithm:
    ///   1. Derive batching challenge r from Fiat-Shamir transcript.
    ///   2. Compute LHS = sum_i r^i * (C_i - [y_i]*G) via GPU MSM.
    ///   3. Compute RHS = sum_i r^i * (s - z_i) * pi_i via GPU MSM.
    ///   4. Check LHS == RHS.
    ///
    /// For N == 1, falls back to verifySingle (no GPU overhead).
    /// For N >= GPU_BATCH_THRESHOLD, uses GPU MSM for both accumulations.
    ///
    /// - Parameters:
    ///   - claims: Array of N opening claims.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all N openings are valid.
    public func batchVerify(claims: [BatchVerifyClaim], srsSecret: Fr) throws -> Bool {
        guard !claims.isEmpty else { return true }
        let n = claims.count

        if n == 1 {
            return verifySingle(claim: claims[0], srsSecret: srsSecret)
        }

        // Step 1: Derive batching challenge
        let r = deriveBatchChallenge(claims: claims)

        // Step 2: Compute powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // Step 3: Build LHS = sum_i r^i * (C_i - [y_i]*G)
        var lhsPoints = [PointAffine]()
        lhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let yG = cPointScalarMul(g1, claims[i].value)
            let diff = pointAdd(claims[i].commitment, pointNeg(yG))
            lhsPoints.append(batchToAffine([diff])[0])
        }

        let lhs: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            lhs = try msmEngine.msm(points: lhsPoints, scalars: batchFrToLimbs(rPowers))
        } else {
            lhs = cpuMSM(points: lhsPoints, scalars: rPowers)
        }

        // Step 4: Build RHS = sum_i r^i * (s - z_i) * pi_i
        var rhsScalarsFr = [Fr](repeating: Fr.zero, count: n)
        var rhsPoints = [PointAffine]()
        rhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let sMz = frSub(srsSecret, claims[i].point)
            rhsScalarsFr[i] = frMul(rPowers[i], sMz)
            rhsPoints.append(batchToAffine([claims[i].proof])[0])
        }

        let rhs: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            rhs = try msmEngine.msm(points: rhsPoints, scalars: batchFrToLimbs(rhsScalarsFr))
        } else {
            rhs = cpuMSM(points: rhsPoints, scalars: rhsScalarsFr)
        }

        // Step 5: Check LHS == RHS
        return pointsEqual(lhs, rhs)
    }

    // MARK: - Batch Verification with Detailed Result

    /// Batch verify with detailed diagnostics (timing, GPU usage, summary).
    ///
    /// - Parameters:
    ///   - claims: Array of N opening claims.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: BatchVerifyResult with isValid, timing, and diagnostics.
    public func batchVerifyDetailed(claims: [BatchVerifyClaim], srsSecret: Fr) throws -> BatchVerifyResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let n = claims.count
        let useGPU = n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD
        let valid = try batchVerify(claims: claims, srsSecret: srsSecret)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let summary = "Batch verified \(n) claim(s) in \(String(format: "%.3f", elapsed * 1000))ms" +
                       " [GPU=\(useGPU), valid=\(valid)]"

        return BatchVerifyResult(
            isValid: valid,
            claimCount: n,
            usedGPU: useGPU,
            elapsedSeconds: elapsed,
            summary: summary
        )
    }

    // MARK: - Multi-Point Batch Verification

    /// Batch verify claims from a MultiOpenBatchClaim (different points per commitment).
    ///
    /// Converts the multi-open batch into individual BatchVerifyClaim entries
    /// and delegates to the core batch verification.
    ///
    /// - Parameters:
    ///   - batch: The multi-open batch claim.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all claims in the batch are valid.
    public func batchVerifyMultiOpen(batch: MultiOpenBatchClaim, srsSecret: Fr) throws -> Bool {
        guard batch.isWellFormed else { return false }
        guard batch.count > 0 else { return true }

        let claims = (0..<batch.count).map { i in
            BatchVerifyClaim(
                commitment: batch.commitments[i],
                point: batch.points[i],
                value: batch.values[i],
                proof: batch.proofs[i]
            )
        }
        return try batchVerify(claims: claims, srsSecret: srsSecret)
    }

    // MARK: - Same-Point Batch Verification

    /// Optimized batch verification when all polynomials are opened at the same point.
    ///
    /// Exploits the shared evaluation point to compute (s - z) once and factor it out:
    ///   LHS = sum_i r^i * (C_i - [y_i]*G)
    ///   RHS = (s - z) * sum_i r^i * pi_i
    ///
    /// This avoids N separate (s - z_i) computations.
    ///
    /// - Parameters:
    ///   - batch: Same-point batch claim.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all claims are valid.
    public func batchVerifySamePoint(batch: SamePointBatchClaim, srsSecret: Fr) throws -> Bool {
        let n = batch.count
        guard n > 0, n == batch.values.count, n == batch.proofs.count else {
            return false
        }

        if n == 1 {
            let claim = BatchVerifyClaim(
                commitment: batch.commitments[0], point: batch.point,
                value: batch.values[0], proof: batch.proofs[0])
            return verifySingle(claim: claim, srsSecret: srsSecret)
        }

        // Derive challenge from individual claims
        let claims = (0..<n).map { i in
            BatchVerifyClaim(
                commitment: batch.commitments[i], point: batch.point,
                value: batch.values[i], proof: batch.proofs[i])
        }
        let r = deriveBatchChallenge(claims: claims)

        // Powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // LHS: sum_i r^i * (C_i - [y_i]*G)
        var lhsPoints = [PointAffine]()
        lhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let yG = cPointScalarMul(g1, batch.values[i])
            let diff = pointAdd(batch.commitments[i], pointNeg(yG))
            lhsPoints.append(batchToAffine([diff])[0])
        }

        let lhs: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            lhs = try msmEngine.msm(points: lhsPoints, scalars: batchFrToLimbs(rPowers))
        } else {
            lhs = cpuMSM(points: lhsPoints, scalars: rPowers)
        }

        // RHS: (s - z) * sum_i r^i * pi_i
        var proofAffs = [PointAffine]()
        proofAffs.reserveCapacity(n)
        for i in 0..<n {
            proofAffs.append(batchToAffine([batch.proofs[i]])[0])
        }

        let combinedProof: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            combinedProof = try msmEngine.msm(points: proofAffs, scalars: batchFrToLimbs(rPowers))
        } else {
            combinedProof = cpuMSM(points: proofAffs, scalars: rPowers)
        }

        let sMz = frSub(srsSecret, batch.point)
        let rhs = cPointScalarMul(combinedProof, sMz)

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Array-Based Convenience APIs

    /// Batch verify KZG openings given parallel arrays.
    ///
    /// - Parameters:
    ///   - commitments: [C_0, ..., C_{N-1}] polynomial commitments
    ///   - points: [z_0, ..., z_{N-1}] evaluation points
    ///   - values: [y_0, ..., y_{N-1}] claimed evaluations
    ///   - proofs: [pi_0, ..., pi_{N-1}] KZG witness proofs
    ///   - srsSecret: toxic waste s (testing only)
    /// - Returns: true if all openings are valid
    public func batchVerifyArrays(
        commitments: [PointProjective],
        points: [Fr],
        values: [Fr],
        proofs: [PointProjective],
        srsSecret: Fr
    ) throws -> Bool {
        let n = commitments.count
        guard n == points.count, n == values.count, n == proofs.count else {
            return false
        }
        let claims = (0..<n).map { i in
            BatchVerifyClaim(
                commitment: commitments[i], point: points[i],
                value: values[i], proof: proofs[i])
        }
        return try batchVerify(claims: claims, srsSecret: srsSecret)
    }

    /// Same-point batch verify given parallel arrays.
    ///
    /// - Parameters:
    ///   - commitments: [C_0, ..., C_{N-1}] polynomial commitments
    ///   - point: shared evaluation point z
    ///   - values: [y_0, ..., y_{N-1}] claimed evaluations
    ///   - proofs: [pi_0, ..., pi_{N-1}] KZG witness proofs
    ///   - srsSecret: toxic waste s (testing only)
    /// - Returns: true if all openings are valid
    public func batchVerifySamePointArrays(
        commitments: [PointProjective],
        point: Fr,
        values: [Fr],
        proofs: [PointProjective],
        srsSecret: Fr
    ) throws -> Bool {
        let batch = SamePointBatchClaim(
            commitments: commitments, point: point,
            values: values, proofs: proofs)
        return try batchVerifySamePoint(batch: batch, srsSecret: srsSecret)
    }

    // MARK: - Aggregated Pairing Verification (Production)

    /// Verify batch via aggregated pairing check (no SRS secret needed).
    ///
    /// Production-mode verification using BN254 ate pairing:
    ///   e(LHS_combined, [1]_2) == e(RHS_combined, [s]_2)
    ///
    /// Where:
    ///   LHS_combined = sum_i r^i * (C_i - [y_i]*G + [z_i] * pi_i)
    ///   RHS_combined = sum_i r^i * pi_i
    ///
    /// This reduces N pairing checks to exactly 2 pairings.
    ///
    /// - Parameters:
    ///   - claims: Array of N opening claims.
    ///   - g2Gen: G2 generator point [1]_2
    ///   - g2Tau: G2 SRS point [s]_2
    /// - Returns: true if the aggregated pairing check passes.
    public func batchVerifyPairing(
        claims: [BatchVerifyClaim],
        g2Gen: G2AffinePoint,
        g2Tau: G2AffinePoint
    ) throws -> Bool {
        guard !claims.isEmpty else { return true }
        let n = claims.count

        // Derive batching challenge
        let r = deriveBatchChallenge(claims: claims)

        // Compute powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // LHS_combined = sum_i r^i * (C_i - [y_i]*G + [z_i] * pi_i)
        var lhsCombined = pointIdentity()
        for i in 0..<n {
            let yG = cPointScalarMul(g1, claims[i].value)
            let zPi = cPointScalarMul(claims[i].proof, claims[i].point)
            var term = pointAdd(claims[i].commitment, pointNeg(yG))
            term = pointAdd(term, zPi)
            lhsCombined = pointAdd(lhsCombined, cPointScalarMul(term, rPowers[i]))
        }

        // RHS_combined = sum_i r^i * pi_i
        var rhsCombined = pointIdentity()
        for i in 0..<n {
            rhsCombined = pointAdd(rhsCombined, cPointScalarMul(claims[i].proof, rPowers[i]))
        }

        // Pairing check: e(LHS_combined, [1]_2) == e(RHS_combined, [s]_2)
        // Equivalent to: e(LHS_combined, [1]_2) * e(-RHS_combined, [s]_2) == 1
        let lhsAff = batchToAffine([lhsCombined])[0]
        let rhsAff = batchToAffine([rhsCombined])[0]
        let negRhsAff = PointAffine(x: rhsAff.x, y: fpNeg(rhsAff.y))

        return cBN254PairingCheck([(lhsAff, g2Gen), (negRhsAff, g2Tau)])
    }

    // MARK: - Incremental Batch Verification

    /// Accumulator for incremental batch verification.
    /// Add claims one by one, then finalize with a single batch check.
    public final class BatchAccumulator {
        private var claims: [BatchVerifyClaim] = []

        public init() {}

        /// Add a single claim to the accumulator.
        public func addClaim(_ claim: BatchVerifyClaim) {
            claims.append(claim)
        }

        /// Add a claim from components.
        public func addClaim(commitment: PointProjective, point: Fr, value: Fr, proof: PointProjective) {
            claims.append(BatchVerifyClaim(
                commitment: commitment, point: point,
                value: value, proof: proof))
        }

        /// Number of accumulated claims.
        public var count: Int { claims.count }

        /// Clear all accumulated claims.
        public func clear() { claims.removeAll() }

        /// Retrieve all accumulated claims for verification.
        public func getClaims() -> [BatchVerifyClaim] { claims }
    }

    /// Create a new batch accumulator.
    public func createAccumulator() -> BatchAccumulator {
        return BatchAccumulator()
    }

    /// Verify all claims in an accumulator.
    ///
    /// - Parameters:
    ///   - accumulator: The batch accumulator with collected claims.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all accumulated claims are valid.
    public func verifyAccumulator(_ accumulator: BatchAccumulator, srsSecret: Fr) throws -> Bool {
        return try batchVerify(claims: accumulator.getClaims(), srsSecret: srsSecret)
    }

    // MARK: - Cross-Polynomial Batch Verification

    /// Verify openings from multiple polynomials at multiple distinct points,
    /// where each polynomial may have different evaluation points.
    ///
    /// This is the most general batch verification mode. Each entry in `polyOpenings`
    /// specifies a polynomial's commitment and its set of opening claims.
    ///
    /// - Parameters:
    ///   - polyOpenings: Array of (commitment, [(point, value, proof)]) per polynomial.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all openings across all polynomials are valid.
    public func batchVerifyCrossPolynomial(
        polyOpenings: [(commitment: PointProjective, openings: [(point: Fr, value: Fr, proof: PointProjective)])],
        srsSecret: Fr
    ) throws -> Bool {
        var allClaims = [BatchVerifyClaim]()
        for entry in polyOpenings {
            for opening in entry.openings {
                allClaims.append(BatchVerifyClaim(
                    commitment: entry.commitment,
                    point: opening.point,
                    value: opening.value,
                    proof: opening.proof))
            }
        }
        return try batchVerify(claims: allClaims, srsSecret: srsSecret)
    }

    // MARK: - Transcript-Based Batch Verification

    /// Batch verify with an external Fiat-Shamir transcript.
    ///
    /// The caller provides a transcript that may already contain prior protocol
    /// messages. The engine absorbs claims and derives the batching challenge
    /// from this transcript, ensuring binding to the larger protocol context.
    ///
    /// - Parameters:
    ///   - claims: Array of N opening claims.
    ///   - transcript: External Fiat-Shamir transcript.
    ///   - srsSecret: The SRS secret s (testing mode).
    /// - Returns: true if all N openings are valid.
    public func batchVerifyWithTranscript(
        claims: [BatchVerifyClaim],
        transcript: Transcript,
        srsSecret: Fr
    ) throws -> Bool {
        guard !claims.isEmpty else { return true }
        let n = claims.count

        if n == 1 {
            return verifySingle(claim: claims[0], srsSecret: srsSecret)
        }

        // Absorb all claims into the provided transcript
        for claim in claims {
            absorbPoint(claim.commitment, into: transcript)
            transcript.absorb(claim.point)
            transcript.absorb(claim.value)
            absorbPoint(claim.proof, into: transcript)
        }
        let r = transcript.squeeze()

        // Compute powers of r
        var rPowers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            rPowers[i] = frMul(rPowers[i - 1], r)
        }

        let g1 = pointFromAffine(srs[0])

        // LHS = sum_i r^i * (C_i - [y_i]*G)
        var lhsPoints = [PointAffine]()
        lhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let yG = cPointScalarMul(g1, claims[i].value)
            let diff = pointAdd(claims[i].commitment, pointNeg(yG))
            lhsPoints.append(batchToAffine([diff])[0])
        }

        let lhs: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            lhs = try msmEngine.msm(points: lhsPoints, scalars: batchFrToLimbs(rPowers))
        } else {
            lhs = cpuMSM(points: lhsPoints, scalars: rPowers)
        }

        // RHS = sum_i r^i * (s - z_i) * pi_i
        var rhsScalarsFr = [Fr](repeating: Fr.zero, count: n)
        var rhsPoints = [PointAffine]()
        rhsPoints.reserveCapacity(n)
        for i in 0..<n {
            let sMz = frSub(srsSecret, claims[i].point)
            rhsScalarsFr[i] = frMul(rPowers[i], sMz)
            rhsPoints.append(batchToAffine([claims[i].proof])[0])
        }

        let rhs: PointProjective
        if n >= GPUKZGBatchVerifyEngine.GPU_BATCH_THRESHOLD {
            rhs = try msmEngine.msm(points: rhsPoints, scalars: batchFrToLimbs(rhsScalarsFr))
        } else {
            rhs = cpuMSM(points: rhsPoints, scalars: rhsScalarsFr)
        }

        return pointsEqual(lhs, rhs)
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    /// Derive a batching challenge r by absorbing all claims into a fresh transcript.
    private func deriveBatchChallenge(claims: [BatchVerifyClaim]) -> Fr {
        let transcript = Transcript(label: "gpu-kzg-batch-verify", backend: .poseidon2)
        for claim in claims {
            absorbPoint(claim.commitment, into: transcript)
            transcript.absorb(claim.point)
            transcript.absorb(claim.value)
            absorbPoint(claim.proof, into: transcript)
        }
        return transcript.squeeze()
    }

    // MARK: - CPU MSM Fallback

    /// CPU multi-scalar multiplication for small batches.
    private func cpuMSM(points: [PointAffine], scalars: [Fr]) -> PointProjective {
        var result = pointIdentity()
        for i in 0..<points.count {
            let prod = cPointScalarMul(pointFromAffine(points[i]), scalars[i])
            result = pointAdd(result, prod)
        }
        return result
    }

    // MARK: - Scalar Conversion

    /// Convert Fr to [UInt32] limbs for GPU MSM.
    private func frToLimbs(_ scalar: Fr) -> [UInt32] {
        var limbs = [UInt32](repeating: 0, count: 8)
        withUnsafeBytes(of: scalar) { src in
            limbs.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    1
                )
            }
        }
        return limbs
    }

    /// Batch convert [Fr] to [[UInt32]] limbs for GPU MSM.
    private func batchFrToLimbs(_ scalars: [Fr]) -> [[UInt32]] {
        let n = scalars.count
        var flat = [UInt32](repeating: 0, count: n * 8)
        scalars.withUnsafeBytes { src in
            flat.withUnsafeMutableBufferPointer { dst in
                bn254_fr_batch_to_limbs(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dst.baseAddress!,
                    Int32(n)
                )
            }
        }
        var result = [[UInt32]]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(flat[i * 8 ..< (i + 1) * 8]))
        }
        return result
    }

    // MARK: - Helpers

    /// Absorb a projective point into the transcript.
    private func absorbPoint(_ p: PointProjective, into transcript: Transcript) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
        } else {
            let aff = batchToAffine([p])
            transcript.absorb(fpToFr(aff[0].x))
            transcript.absorb(fpToFr(aff[0].y))
        }
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

    /// Check if an Fr value is zero.
    private func frIsZero(_ a: Fr) -> Bool {
        return frToInt(a) == frToInt(Fr.zero)
    }
}
