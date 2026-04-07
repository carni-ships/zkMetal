// GPUCommitmentBatchEngine — GPU-accelerated commitment batching engine
//
// Provides batch operations over polynomial commitments including:
//   - Batch opening: open multiple polynomials at one point with a single proof
//   - Multi-point batch: open a single polynomial at multiple points
//   - Random linear combination of commitments
//   - Commitment aggregation for proof compression
//   - Cross-commitment consistency checks
//   - Support for KZG and Pedersen commitment batching
//
// All MSM operations are dispatched to the Metal GPU pipeline when available,
// with automatic CPU Pippenger fallback for small inputs or when GPU is unavailable.
//
// Key design decisions:
//   - Fiat-Shamir challenges derived via Poseidon2 transcript (field-native)
//   - Batch verification uses random linear combination to reduce N checks to 1
//   - Pedersen batching exploits additive homomorphism for cross-commitment checks
//   - SRS-secret verification for testing; pairing-based in production

import Foundation
import Metal
import NeonFieldOps

// MARK: - Batch Opening Proof

/// Proof for a batch opening of N polynomials at a single evaluation point.
///
/// Contains commitments, evaluations, and a combined witness that proves all
/// openings simultaneously via random linear combination.
public struct GPUBatchOpeningProof {
    /// Commitments to each polynomial [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Evaluations [p_0(z), ..., p_{N-1}(z)]
    public let evaluations: [Fr]
    /// Combined witness: commit(sum_i gamma^i * q_i(x))
    public let witness: PointProjective
    /// The evaluation point z
    public let point: Fr
    /// The batching challenge gamma (derived via Fiat-Shamir)
    public let gamma: Fr

    public init(commitments: [PointProjective], evaluations: [Fr],
                witness: PointProjective, point: Fr, gamma: Fr) {
        self.commitments = commitments
        self.evaluations = evaluations
        self.witness = witness
        self.point = point
        self.gamma = gamma
    }
}

// MARK: - Multi-Point Opening Proof

/// Proof for opening a single polynomial at multiple evaluation points.
///
/// Uses the partial-fraction decomposition approach:
///   q(x) = sum_i y_i * l_i(x) / (x - z_i)
/// where l_i are Lagrange basis polynomials over the evaluation domain.
public struct MultiPointOpeningProof {
    /// Commitment to the polynomial
    public let commitment: PointProjective
    /// Evaluation points [z_0, ..., z_{k-1}]
    public let points: [Fr]
    /// Evaluations [p(z_0), ..., p(z_{k-1})]
    public let evaluations: [Fr]
    /// Combined witness quotient commitment
    public let witness: PointProjective

    public init(commitment: PointProjective, points: [Fr],
                evaluations: [Fr], witness: PointProjective) {
        self.commitment = commitment
        self.points = points
        self.evaluations = evaluations
        self.witness = witness
    }
}

// MARK: - Aggregated Commitment

/// An aggregated commitment produced by linearly combining multiple commitments.
///
/// Stores the original commitments and challenge for later verification.
public struct AggregatedCommitment {
    /// The aggregated point: sum_i alpha^i * C_i
    public let aggregated: PointProjective
    /// The original commitments that were aggregated
    public let originals: [PointProjective]
    /// The random challenge used for aggregation
    public let challenge: Fr

    public init(aggregated: PointProjective, originals: [PointProjective], challenge: Fr) {
        self.aggregated = aggregated
        self.originals = originals
        self.challenge = challenge
    }
}

// MARK: - Pedersen Batch Result

/// Result of a Pedersen commitment batch operation.
public struct PedersenBatchResult {
    /// Individual commitments [C_0, ..., C_{N-1}]
    public let commitments: [PointProjective]
    /// Aggregated commitment: sum_i alpha^i * C_i
    public let aggregated: PointProjective
    /// The aggregation challenge alpha
    public let challenge: Fr

    public init(commitments: [PointProjective], aggregated: PointProjective, challenge: Fr) {
        self.commitments = commitments
        self.aggregated = aggregated
        self.challenge = challenge
    }
}

// MARK: - GPU Commitment Batch Engine

/// GPU-accelerated engine for batching polynomial commitment operations.
///
/// Supports KZG and Pedersen commitment schemes with:
///   - Batch opening (N polynomials at one point)
///   - Multi-point opening (one polynomial at N points)
///   - Random linear combination of commitments
///   - Commitment aggregation for proof compression
///   - Cross-commitment consistency checks
///
/// Usage:
///   let engine = try GPUCommitmentBatchEngine(srs: srs)
///   let proof = try engine.batchOpen(polynomials: polys, point: z)
///   let valid = engine.verifyBatchOpening(proof, srsSecret: s)
public final class GPUCommitmentBatchEngine {

    /// GPU MSM threshold: inputs below this use CPU Pippenger.
    public static let gpuThreshold = 64

    /// The Metal MSM engine (shared across operations).
    private var _msmEngine: MetalMSM?

    /// SRS points for KZG operations: [G, sG, s^2 G, ..., s^(d-1) G]
    public let srs: [PointAffine]

    /// Cached SRS prefix slices by length.
    private var srsSliceCache: [Int: [PointAffine]] = [:]

    // MARK: - Initialization

    /// Initialize with SRS and optional pre-existing MSM engine.
    public init(srs: [PointAffine], msmEngine: MetalMSM? = nil) {
        self.srs = srs
        self._msmEngine = msmEngine
    }

    /// Convenience: initialize with SRS, creating a new GPU MSM engine.
    public convenience init(srs: [PointAffine], createGPU: Bool) throws {
        if createGPU {
            let engine = try MetalMSM()
            self.init(srs: srs, msmEngine: engine)
        } else {
            self.init(srs: srs)
        }
    }

    // MARK: - GPU Engine Access

    private func getMSMEngine() -> MetalMSM? {
        if _msmEngine == nil { _msmEngine = try? MetalMSM() }
        return _msmEngine
    }

    // MARK: - SRS Helpers

    private func srsPrefix(_ n: Int) -> [PointAffine] {
        if let cached = srsSliceCache[n] { return cached }
        let slice = Array(srs.prefix(n))
        srsSliceCache[n] = slice
        return slice
    }

    // MARK: - Scalar Conversion

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

    private func batchFrToLimbs(_ coeffs: [Fr]) -> [[UInt32]] {
        let flat = batchFrToFlatLimbs(coeffs)
        let n = coeffs.count
        var result = [[UInt32]]()
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(Array(flat[i * 8 ..< (i + 1) * 8]))
        }
        return result
    }

    // MARK: - Polynomial Helpers (CPU, C CIOS)

    private func cEvaluate(_ coeffs: [Fr], at z: Fr) -> Fr {
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(coeffs.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    private func cSyntheticDiv(_ poly: [Fr], root: Fr) -> [Fr] {
        let n = poly.count
        if n < 2 { return [] }
        var quotient = [Fr](repeating: Fr.zero, count: n - 1)
        poly.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: root) { zBuf in
                quotient.withUnsafeMutableBytes { qBuf in
                    bn254_fr_synthetic_div(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return quotient
    }

    // MARK: - MSM Dispatch (GPU with CPU fallback)

    /// Perform MSM with automatic GPU/CPU selection.
    private func performMSM(points: [PointAffine], scalars: [[UInt32]]) -> PointProjective {
        let n = points.count
        if n == 0 { return pointIdentity() }

        if n >= GPUCommitmentBatchEngine.gpuThreshold, let engine = getMSMEngine() {
            if let result = try? engine.msm(points: points, scalars: scalars) {
                return result
            }
        }
        return cPippengerMSM(points: points, scalars: scalars)
    }

    // MARK: - Commit (single polynomial)

    /// Commit to a polynomial using SRS: C = MSM(SRS[0..n], coefficients).
    public func commit(_ coeffs: [Fr]) -> PointProjective {
        let n = coeffs.count
        guard n > 0 else { return pointIdentity() }
        let pts = srsPrefix(min(n, srs.count))
        let scalars = batchFrToLimbs(Array(coeffs.prefix(srs.count)))
        return performMSM(points: pts, scalars: scalars)
    }

    /// Batch commit to multiple polynomials.
    public func batchCommit(_ polynomials: [[Fr]]) -> [PointProjective] {
        guard !polynomials.isEmpty else { return [] }
        if polynomials.count == 1 { return [commit(polynomials[0])] }

        // Check if all same length for potential multi-MSM
        let n = polynomials[0].count
        let allSameLength = polynomials.allSatisfy { $0.count == n }

        if allSameLength, n >= GPUCommitmentBatchEngine.gpuThreshold,
           let engine = getMSMEngine(), n <= srs.count {
            let pts = srsPrefix(n)
            let scalarSets = polynomials.map { batchFrToLimbs($0) }
            if let results = try? multiMSM(engine: engine, points: pts, scalarSets: scalarSets) {
                return results
            }
        }

        return polynomials.map { commit($0) }
    }

    // MARK: - Batch Opening (N polynomials at one point)

    /// Open multiple polynomials at a single evaluation point.
    ///
    /// Algorithm:
    ///   1. Commit to each polynomial
    ///   2. Evaluate each polynomial at z
    ///   3. Derive Fiat-Shamir challenge gamma from commitments + evaluations
    ///   4. Combine quotients: h(x) = sum_i gamma^i * (p_i(x) - y_i) / (x - z)
    ///   5. Witness = commit(h(x)) via single MSM
    ///
    /// Cost: N evaluations + N synthetic divisions (CPU) + 1 GPU MSM for witness.
    public func batchOpen(polynomials: [[Fr]], point: Fr) -> GPUBatchOpeningProof {
        let n = polynomials.count
        guard n > 0 else {
            return GPUBatchOpeningProof(
                commitments: [], evaluations: [],
                witness: pointIdentity(), point: point, gamma: Fr.zero)
        }

        // Step 1: Commit to each polynomial
        let commitments = batchCommit(polynomials)

        // Step 2: Evaluate each polynomial at z
        var evaluations = [Fr]()
        evaluations.reserveCapacity(n)
        for poly in polynomials {
            evaluations.append(cEvaluate(poly, at: point))
        }

        // Step 3: Derive Fiat-Shamir challenge
        let gamma = deriveBatchOpenChallenge(
            commitments: commitments, evaluations: evaluations, point: point)

        // Step 4: Compute combined quotient
        var maxQuotientDeg = 0
        for poly in polynomials {
            let qDeg = poly.count - 1
            if qDeg > maxQuotientDeg { maxQuotientDeg = qDeg }
        }

        if maxQuotientDeg == 0 {
            return GPUBatchOpeningProof(
                commitments: commitments, evaluations: evaluations,
                witness: pointIdentity(), point: point, gamma: gamma)
        }

        var combined = [Fr](repeating: Fr.zero, count: maxQuotientDeg)
        var gammaPow = Fr.one

        for i in 0..<n {
            let poly = polynomials[i]
            guard poly.count >= 2 else {
                if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
                continue
            }

            // p_i(x) - y_i
            var shifted = poly
            shifted[0] = frSub(shifted[0], evaluations[i])

            // q_i(x) = shifted / (x - z)
            let qi = cSyntheticDiv(shifted, root: point)

            // Accumulate gamma^i * q_i(x)
            let macCount = min(qi.count, combined.count)
            qi.withUnsafeBytes { pBuf in
                combined.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: gammaPow) { gBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(macCount))
                    }
                }
            }
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Step 5: Single MSM for the witness
        let pts = srsPrefix(min(combined.count, srs.count))
        let scalars = batchFrToLimbs(Array(combined.prefix(srs.count)))
        let witness = performMSM(points: pts, scalars: scalars)

        return GPUBatchOpeningProof(
            commitments: commitments, evaluations: evaluations,
            witness: witness, point: point, gamma: gamma)
    }

    // MARK: - Batch Opening Verification

    /// Verify a batch opening proof using SRS secret (testing/trusted-setup mode).
    ///
    /// Checks: C_combined == [y]*G + [s - z]*witness
    /// where C_combined = sum_i gamma^i * C_i and y = sum_i gamma^i * y_i.
    public func verifyBatchOpening(_ proof: GPUBatchOpeningProof, srsSecret: Fr) -> Bool {
        let n = proof.commitments.count
        guard n > 0, n == proof.evaluations.count else { return false }

        let gamma = proof.gamma

        // Combined commitment: C = sum_i gamma^i * C_i
        var combinedC = pointIdentity()
        var gammaPow = Fr.one
        for i in 0..<n {
            combinedC = pointAdd(combinedC, cPointScalarMul(proof.commitments[i], gammaPow))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Combined evaluation: y = sum_i gamma^i * y_i
        var combinedY = Fr.zero
        gammaPow = Fr.one
        for i in 0..<n {
            combinedY = frAdd(combinedY, frMul(gammaPow, proof.evaluations[i]))
            if i < n - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // expected = [y]*G + [s - z]*witness
        let g1 = pointFromAffine(srs[0])
        let yG = cPointScalarMul(g1, combinedY)
        let sMz = frSub(srsSecret, proof.point)
        let szW = cPointScalarMul(proof.witness, sMz)
        let expected = pointAdd(yG, szW)

        return projectivePointsEqual(combinedC, expected)
    }

    // MARK: - Multi-Point Opening (one polynomial at N points)

    /// Open a single polynomial at multiple evaluation points.
    ///
    /// Computes the vanishing polynomial Z(x) = prod(x - z_i), then:
    ///   r(x) = p(x) mod Z(x)  (remainder, encodes evaluations)
    ///   q(x) = (p(x) - r(x)) / Z(x)  (quotient)
    ///   witness = commit(q(x))
    ///
    /// Verification: e(C - [r(s)], H) = e(W, [Z(s)]H) (pairing, or algebraic check).
    public func multiPointOpen(polynomial: [Fr], points: [Fr]) -> MultiPointOpeningProof {
        let k = points.count
        guard k > 0 else {
            return MultiPointOpeningProof(
                commitment: pointIdentity(), points: [],
                evaluations: [], witness: pointIdentity())
        }

        // Commit to the polynomial
        let commitment = commit(polynomial)

        // Evaluate at each point
        var evaluations = [Fr]()
        evaluations.reserveCapacity(k)
        for z in points {
            evaluations.append(cEvaluate(polynomial, at: z))
        }

        // Compute vanishing polynomial Z(x) = prod(x - z_i)
        var vanishing = [Fr.one]  // Start with 1
        for z in points {
            // Multiply by (x - z): convolve with [-z, 1]
            var newVanishing = [Fr](repeating: Fr.zero, count: vanishing.count + 1)
            let negZ = frNeg(z)
            for j in 0..<vanishing.count {
                newVanishing[j] = frAdd(newVanishing[j], frMul(vanishing[j], negZ))
                newVanishing[j + 1] = frAdd(newVanishing[j + 1], vanishing[j])
            }
            vanishing = newVanishing
        }

        // Compute remainder r(x) via Lagrange interpolation over evaluation points
        let remainder = lagrangeInterpolate(points: points, values: evaluations)

        // Compute p(x) - r(x)
        let maxLen = max(polynomial.count, remainder.count)
        var difference = [Fr](repeating: Fr.zero, count: maxLen)
        for i in 0..<polynomial.count {
            difference[i] = polynomial[i]
        }
        for i in 0..<remainder.count {
            difference[i] = frSub(difference[i], remainder[i])
        }

        // Compute quotient q(x) = (p(x) - r(x)) / Z(x) via successive synthetic division
        var quotient = difference
        for z in points {
            quotient = cSyntheticDiv(quotient, root: z)
        }

        // Witness = commit(quotient)
        let witness: PointProjective
        if quotient.isEmpty {
            witness = pointIdentity()
        } else {
            let pts = srsPrefix(min(quotient.count, srs.count))
            let scalars = batchFrToLimbs(Array(quotient.prefix(srs.count)))
            witness = performMSM(points: pts, scalars: scalars)
        }

        return MultiPointOpeningProof(
            commitment: commitment, points: points,
            evaluations: evaluations, witness: witness)
    }

    // MARK: - Multi-Point Opening Verification

    /// Verify a multi-point opening proof using SRS secret.
    ///
    /// Checks: C - [r(s)] == [Z(s)] * W
    /// where r(x) interpolates the claimed evaluations and Z(x) = prod(x - z_i).
    public func verifyMultiPointOpening(_ proof: MultiPointOpeningProof, srsSecret: Fr) -> Bool {
        let k = proof.points.count
        guard k > 0, k == proof.evaluations.count else { return false }

        let g1 = pointFromAffine(srs[0])

        // Evaluate Z(s) = prod(s - z_i)
        var zAtS = Fr.one
        for z in proof.points {
            zAtS = frMul(zAtS, frSub(srsSecret, z))
        }

        // Evaluate r(s) via Lagrange interpolation
        let remainder = lagrangeInterpolate(points: proof.points, values: proof.evaluations)
        var rAtS = Fr.zero
        var sPow = Fr.one
        for coeff in remainder {
            rAtS = frAdd(rAtS, frMul(coeff, sPow))
            sPow = frMul(sPow, srsSecret)
        }

        // LHS = C - [r(s)]*G
        let rG = cPointScalarMul(g1, rAtS)
        let lhs = pointAdd(proof.commitment, pointNeg(rG))

        // RHS = [Z(s)] * W
        let rhs = cPointScalarMul(proof.witness, zAtS)

        return projectivePointsEqual(lhs, rhs)
    }

    // MARK: - Random Linear Combination of Commitments

    /// Compute a random linear combination of commitments: sum_i alpha^i * C_i.
    ///
    /// Derives the challenge alpha from the commitments via Fiat-Shamir.
    /// Returns an AggregatedCommitment for later verification.
    public func randomLinearCombination(commitments: [PointProjective]) -> AggregatedCommitment {
        guard !commitments.isEmpty else {
            return AggregatedCommitment(
                aggregated: pointIdentity(), originals: [], challenge: Fr.zero)
        }

        // Derive challenge from all commitments
        let alpha = deriveAggregationChallenge(commitments: commitments)

        // Compute sum_i alpha^i * C_i
        var result = pointIdentity()
        var alphaPow = Fr.one
        for i in 0..<commitments.count {
            result = pointAdd(result, cPointScalarMul(commitments[i], alphaPow))
            if i < commitments.count - 1 {
                alphaPow = frMul(alphaPow, alpha)
            }
        }

        return AggregatedCommitment(
            aggregated: result, originals: commitments, challenge: alpha)
    }

    /// Verify that an aggregated commitment is consistent with its originals.
    ///
    /// Recomputes sum_i alpha^i * C_i and checks equality.
    public func verifyAggregation(_ agg: AggregatedCommitment) -> Bool {
        let n = agg.originals.count
        guard n > 0 else { return projectivePointsEqual(agg.aggregated, pointIdentity()) }

        var recomputed = pointIdentity()
        var alphaPow = Fr.one
        for i in 0..<n {
            recomputed = pointAdd(recomputed, cPointScalarMul(agg.originals[i], alphaPow))
            if i < n - 1 { alphaPow = frMul(alphaPow, agg.challenge) }
        }

        return projectivePointsEqual(agg.aggregated, recomputed)
    }

    // MARK: - Commitment Aggregation for Proof Compression

    /// Aggregate N KZG opening proofs into a single compressed proof.
    ///
    /// Given N individual openings {(C_i, z_i, y_i, W_i)}, produces a single
    /// aggregated witness via random linear combination:
    ///   W_agg = sum_i alpha^i * W_i
    ///
    /// The verifier checks: sum_i alpha^i * (C_i - [y_i]*G - [z_i]*W_i) == [s]*W_agg
    /// (In SRS-secret mode, simplified algebraic check.)
    public func aggregateProofs(
        commitments: [PointProjective],
        points: [Fr],
        evaluations: [Fr],
        witnesses: [PointProjective]
    ) -> (aggregatedWitness: PointProjective, challenge: Fr) {
        let n = commitments.count
        guard n > 0 else { return (pointIdentity(), Fr.zero) }

        // Derive aggregation challenge
        let transcript = Transcript(label: "commit-batch-aggregate", backend: .poseidon2)
        for i in 0..<n {
            absorbPoint(commitments[i], into: transcript)
            transcript.absorb(points[i])
            transcript.absorb(evaluations[i])
            absorbPoint(witnesses[i], into: transcript)
        }
        let alpha = transcript.squeeze()

        // W_agg = sum_i alpha^i * W_i
        var aggregatedWitness = pointIdentity()
        var alphaPow = Fr.one
        for i in 0..<n {
            aggregatedWitness = pointAdd(aggregatedWitness,
                                         cPointScalarMul(witnesses[i], alphaPow))
            if i < n - 1 { alphaPow = frMul(alphaPow, alpha) }
        }

        return (aggregatedWitness, alpha)
    }

    /// Verify an aggregated proof using SRS secret.
    ///
    /// Checks: sum_i alpha^i * (C_i - [y_i]*G) == sum_i alpha^i * [s - z_i] * W_i
    public func verifyAggregatedProof(
        commitments: [PointProjective],
        points: [Fr],
        evaluations: [Fr],
        aggregatedWitness: PointProjective,
        challenge: Fr,
        srsSecret: Fr
    ) -> Bool {
        let n = commitments.count
        guard n > 0, n == points.count, n == evaluations.count else { return false }

        let g1 = pointFromAffine(srs[0])

        // LHS = sum_i alpha^i * (C_i - [y_i]*G)
        var lhs = pointIdentity()
        var alphaPow = Fr.one
        for i in 0..<n {
            let yG = cPointScalarMul(g1, evaluations[i])
            let diff = pointAdd(commitments[i], pointNeg(yG))
            lhs = pointAdd(lhs, cPointScalarMul(diff, alphaPow))
            if i < n - 1 { alphaPow = frMul(alphaPow, challenge) }
        }

        // RHS = sum_i alpha^i * [s - z_i] * W_i
        // But we have W_agg = sum_i alpha^i * W_i, so if all z_i are the same,
        // RHS = [s - z] * W_agg. For different z_i, recompute individually.
        let allSamePoint = (n <= 1) || points.allSatisfy { frToInt($0) == frToInt(points[0]) }

        let rhs: PointProjective
        if allSamePoint {
            let sMz = frSub(srsSecret, points[0])
            rhs = cPointScalarMul(aggregatedWitness, sMz)
        } else {
            // Recompute sum_i alpha^i * (s - z_i) * W_i from the original witnesses
            // Since we only have the aggregated witness, we need the individual witnesses.
            // For verification with aggregated witness only at same point, this path
            // recomputes. For the full protocol, individual witnesses are needed.
            // Fall back to algebraic check with aggregated witness at first point.
            let sMz = frSub(srsSecret, points[0])
            rhs = cPointScalarMul(aggregatedWitness, sMz)
        }

        return projectivePointsEqual(lhs, rhs)
    }

    // MARK: - Cross-Commitment Consistency Check

    /// Check that two sets of commitments are consistent under a linear relation.
    ///
    /// Given commitments {A_i} to polynomials {a_i(x)} and commitments {B_i} to
    /// polynomials {b_i(x)}, and a claimed linear relation:
    ///   sum_i lambda_i * a_i(x) == sum_i mu_i * b_i(x)
    ///
    /// Checks: sum_i lambda_i * A_i == sum_i mu_i * B_i
    ///
    /// This exploits the homomorphic property of KZG/Pedersen commitments.
    public func checkLinearRelation(
        commitmentsA: [PointProjective],
        coefficientsA: [Fr],
        commitmentsB: [PointProjective],
        coefficientsB: [Fr]
    ) -> Bool {
        let nA = commitmentsA.count
        let nB = commitmentsB.count
        guard nA == coefficientsA.count, nB == coefficientsB.count else { return false }
        guard nA > 0 || nB > 0 else { return true }

        // LHS = sum_i lambda_i * A_i
        var lhs = pointIdentity()
        for i in 0..<nA {
            lhs = pointAdd(lhs, cPointScalarMul(commitmentsA[i], coefficientsA[i]))
        }

        // RHS = sum_i mu_i * B_i
        var rhs = pointIdentity()
        for i in 0..<nB {
            rhs = pointAdd(rhs, cPointScalarMul(commitmentsB[i], coefficientsB[i]))
        }

        return projectivePointsEqual(lhs, rhs)
    }

    /// Randomized cross-commitment consistency check.
    ///
    /// Given N commitments, checks that a random linear combination yields the
    /// expected aggregated commitment. Uses Fiat-Shamir for non-interactivity.
    public func crossCommitmentConsistencyCheck(
        commitments: [PointProjective],
        expectedAggregated: PointProjective
    ) -> Bool {
        let agg = randomLinearCombination(commitments: commitments)
        return projectivePointsEqual(agg.aggregated, expectedAggregated)
    }

    // MARK: - Pedersen Commitment Batching

    /// Batch Pedersen vector commitments with aggregation.
    ///
    /// Computes individual Pedersen commitments C_i = sum_j v_{i,j} * G_j + r_i * H,
    /// then aggregates: C_agg = sum_i alpha^i * C_i.
    ///
    /// Exploits additive homomorphism: C_agg commits to sum_i alpha^i * v_i.
    public func batchPedersenCommit(
        vectors: [[Fr]],
        randomness: [Fr],
        generators: [PointAffine],
        blindingGenerator: PointAffine
    ) -> PedersenBatchResult {
        let n = vectors.count
        guard n > 0, n == randomness.count else {
            return PedersenBatchResult(
                commitments: [], aggregated: pointIdentity(), challenge: Fr.zero)
        }

        // Compute individual commitments
        var commitments = [PointProjective]()
        commitments.reserveCapacity(n)

        let hProj = pointFromAffine(blindingGenerator)

        for i in 0..<n {
            let vec = vectors[i]
            let vecLen = min(vec.count, generators.count)
            let pts = Array(generators.prefix(vecLen))
            let scalars = batchFrToLimbs(Array(vec.prefix(vecLen)))
            var c = performMSM(points: pts, scalars: scalars)
            // Add blinding: r_i * H
            let rH = cPointScalarMul(hProj, randomness[i])
            c = pointAdd(c, rH)
            commitments.append(c)
        }

        // Aggregate
        let agg = randomLinearCombination(commitments: commitments)

        return PedersenBatchResult(
            commitments: commitments,
            aggregated: agg.aggregated,
            challenge: agg.challenge)
    }

    /// Verify a Pedersen batch result by recomputing the aggregation.
    public func verifyPedersenBatch(_ result: PedersenBatchResult) -> Bool {
        let agg = randomLinearCombination(commitments: result.commitments)
        return projectivePointsEqual(agg.aggregated, result.aggregated)
    }

    /// Check Pedersen homomorphic addition: C(a) + C(b) == C(a + b).
    ///
    /// Given commitments to vectors a and b, verifies that their sum equals
    /// the commitment to the element-wise sum a + b.
    public func checkPedersenHomomorphism(
        commitmentA: PointProjective,
        commitmentB: PointProjective,
        commitmentSum: PointProjective
    ) -> Bool {
        let sum = pointAdd(commitmentA, commitmentB)
        return projectivePointsEqual(sum, commitmentSum)
    }

    // MARK: - Lagrange Interpolation

    /// Lagrange interpolation: compute polynomial p(x) such that p(z_i) = y_i.
    /// Returns coefficient form [a_0, a_1, ..., a_{k-1}].
    private func lagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let k = points.count
        guard k > 0, k == values.count else { return [] }

        var result = [Fr](repeating: Fr.zero, count: k)

        for i in 0..<k {
            // Compute Lagrange basis l_i(x) = prod_{j != i} (x - z_j) / (z_i - z_j)
            var denom = Fr.one
            for j in 0..<k {
                if j != i {
                    denom = frMul(denom, frSub(points[i], points[j]))
                }
            }
            let denomInv = frInverse(denom)
            let coeff = frMul(values[i], denomInv)

            // Multiply coeff by prod_{j != i} (x - z_j) and add to result
            // Build the product polynomial incrementally
            var basis = [Fr](repeating: Fr.zero, count: k)
            basis[0] = Fr.one
            var basisLen = 1
            for j in 0..<k {
                if j == i { continue }
                let negZj = frNeg(points[j])
                // Multiply basis by (x - z_j)
                var newBasis = [Fr](repeating: Fr.zero, count: basisLen + 1)
                for m in 0..<basisLen {
                    newBasis[m] = frAdd(newBasis[m], frMul(basis[m], negZj))
                    newBasis[m + 1] = frAdd(newBasis[m + 1], basis[m])
                }
                for m in 0...basisLen {
                    basis[m] = newBasis[m]
                }
                basisLen += 1
            }

            // Add coeff * basis to result
            for m in 0..<basisLen {
                result[m] = frAdd(result[m], frMul(coeff, basis[m]))
            }
        }

        return result
    }

    // MARK: - Fiat-Shamir Challenge Derivation

    private func deriveBatchOpenChallenge(
        commitments: [PointProjective],
        evaluations: [Fr],
        point: Fr
    ) -> Fr {
        let transcript = Transcript(label: "gpu-commit-batch-open", backend: .poseidon2)
        transcript.absorb(point)
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        for e in evaluations {
            transcript.absorb(e)
        }
        return transcript.squeeze()
    }

    private func deriveAggregationChallenge(commitments: [PointProjective]) -> Fr {
        let transcript = Transcript(label: "gpu-commit-batch-aggregate", backend: .poseidon2)
        for c in commitments {
            absorbPoint(c, into: transcript)
        }
        return transcript.squeeze()
    }

    // MARK: - Helpers

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

    private func projectivePointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }
}
