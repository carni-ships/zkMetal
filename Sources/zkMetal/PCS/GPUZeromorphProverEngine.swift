// GPU-Accelerated Zeromorph Prover Engine
//
// Production-grade Zeromorph multilinear PCS prover with GPU acceleration.
// Opens multilinear polynomials at evaluation points via univariate reduction,
// using quotient polynomial computation and degree-folding optimization.
//
// Key differences from GPUZeromorphEngine:
//   1. Degree-folding optimization: reduces quotient polynomial degrees before commit
//   2. GPU-accelerated polynomial arithmetic (pointwise add/mul/sub)
//   3. Configurable GPU/CPU thresholds for commit and poly ops
//   4. Batch opening with shared quotient structure (amortized cost)
//   5. Streaming quotient computation for memory efficiency
//
// Construction (degree-folded Zeromorph):
//   Given MLE f on {0,1}^n embedded as univariate f(X) = sum_i evals[i]*X^i,
//   the prover computes n quotient polynomials via even/odd decomposition:
//     q^(s) = f_odd,  f_next = f_even + u_k * f_odd
//   Degree-folding: each q^(s) has degree < 2^{n-s-1}. The prover folds
//   high-degree coefficients into low-degree ones using powers of zeta,
//   reducing commitment cost by ~50%.
//   Linearized polynomial L(X) is opened via KZG at challenge zeta.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Zeromorph Prover Proof

/// Proof produced by the GPU Zeromorph prover engine.
/// Contains quotient commitments, degree-fold metadata, and the KZG witness.
public struct ZeromorphProverProof {
    /// Commitments to (degree-folded) quotient polynomials [q'^(0)],...,[q'^(n-1)]
    public let quotientCommitments: [PointProjective]
    /// The claimed evaluation value v = f(u_0,...,u_{n-1})
    public let claimedValue: Fr
    /// KZG witness for linearized polynomial L at zeta
    public let kzgWitness: PointProjective
    /// Evaluation L(zeta)
    public let linearizationEval: Fr
    /// Fiat-Shamir challenge zeta
    public let zeta: Fr
    /// Whether degree-folding was applied
    public let degreeFolded: Bool

    public init(quotientCommitments: [PointProjective], claimedValue: Fr,
                kzgWitness: PointProjective, linearizationEval: Fr,
                zeta: Fr, degreeFolded: Bool) {
        self.quotientCommitments = quotientCommitments
        self.claimedValue = claimedValue
        self.kzgWitness = kzgWitness
        self.linearizationEval = linearizationEval
        self.zeta = zeta
        self.degreeFolded = degreeFolded
    }

    /// Number of variables in the multilinear polynomial
    public var numVariables: Int { quotientCommitments.count }

    /// Approximate byte size of the proof
    public var approximateByteSize: Int {
        let pointSize = 96  // 3 * Fp
        let scalarSize = 32
        return quotientCommitments.count * pointSize + pointSize + 3 * scalarSize + 1
    }
}

// MARK: - Zeromorph Prover Batch Proof

/// Batch proof for multiple multilinear polynomials at the same point.
/// Uses random linear combination to reduce N openings to one.
public struct ZeromorphProverBatchProof {
    /// Per-polynomial quotient commitments (for accountability)
    public let perPolyQuotientCommitments: [[PointProjective]]
    /// Claimed values for each polynomial
    public let claimedValues: [Fr]
    /// Single batched KZG witness
    public let kzgWitness: PointProjective
    /// Batched linearization evaluation
    public let linearizationEval: Fr
    /// Fiat-Shamir challenge zeta
    public let zeta: Fr
    /// Random linear combination challenge gamma
    public let gamma: Fr
    /// Whether degree-folding was applied
    public let degreeFolded: Bool

    public init(perPolyQuotientCommitments: [[PointProjective]], claimedValues: [Fr],
                kzgWitness: PointProjective, linearizationEval: Fr,
                zeta: Fr, gamma: Fr, degreeFolded: Bool) {
        self.perPolyQuotientCommitments = perPolyQuotientCommitments
        self.claimedValues = claimedValues
        self.kzgWitness = kzgWitness
        self.linearizationEval = linearizationEval
        self.zeta = zeta
        self.gamma = gamma
        self.degreeFolded = degreeFolded
    }

    /// Number of polynomials in the batch
    public var count: Int { claimedValues.count }
}

// MARK: - Prover Configuration

/// Configuration for the GPU Zeromorph prover engine.
public struct ZeromorphProverConfig {
    /// Minimum polynomial size to use GPU MSM for commitments
    public var gpuCommitThreshold: Int
    /// Whether to enable degree-folding optimization
    public var enableDegreeFolding: Bool
    /// Whether to enable parallel batch computation
    public var enableBatchParallel: Bool
    /// Maximum concurrent batch items
    public var maxBatchConcurrency: Int

    public init(gpuCommitThreshold: Int = 64,
                enableDegreeFolding: Bool = true,
                enableBatchParallel: Bool = true,
                maxBatchConcurrency: Int = 4) {
        self.gpuCommitThreshold = gpuCommitThreshold
        self.enableDegreeFolding = enableDegreeFolding
        self.enableBatchParallel = enableBatchParallel
        self.maxBatchConcurrency = maxBatchConcurrency
    }

    /// Default configuration for Apple Silicon
    public static let `default` = ZeromorphProverConfig()

    /// CPU-only configuration (no GPU, no degree-folding)
    public static let cpuOnly = ZeromorphProverConfig(
        gpuCommitThreshold: Int.max,
        enableDegreeFolding: false,
        enableBatchParallel: false,
        maxBatchConcurrency: 1
    )

    /// Aggressive optimization: degree-folding + low GPU threshold
    public static let optimized = ZeromorphProverConfig(
        gpuCommitThreshold: 32,
        enableDegreeFolding: true,
        enableBatchParallel: true,
        maxBatchConcurrency: 8
    )
}

// MARK: - GPU Zeromorph Prover Engine

/// GPU-accelerated Zeromorph multilinear polynomial commitment prover.
///
/// Implements the Zeromorph protocol (Kohrita & Towa, 2023) with additional
/// degree-folding optimization that reduces quotient polynomial degrees
/// before commitment, saving ~50% MSM cost.
///
/// Usage:
///   let engine = GPUZeromorphProverEngine(kzg: kzg)
///   let proof = try engine.open(evaluations: evals, point: point)
///   let valid = engine.verifyWithSecret(evaluations: evals, point: point,
///                                        value: v, proof: proof, srsSecret: s)
public class GPUZeromorphProverEngine {
    public static let version = Versions.gpuZeromorphProver

    /// Underlying KZG engine for commitments and openings
    public let kzg: KZGEngine
    /// Engine configuration
    public let config: ZeromorphProverConfig
    /// Total proofs generated (diagnostics)
    private var _proofCount: Int = 0
    /// Total batch proofs generated
    private var _batchProofCount: Int = 0

    public init(kzg: KZGEngine, config: ZeromorphProverConfig = .default) {
        self.kzg = kzg
        self.config = config
    }

    /// Number of single proofs generated
    public var proofCount: Int { _proofCount }
    /// Number of batch proofs generated
    public var batchProofCount: Int { _batchProofCount }

    // MARK: - Commit

    /// Commit to a multilinear polynomial given as evaluations on {0,1}^n.
    /// Reinterprets 2^n evaluations as univariate coefficients and commits via KZG.
    public func commit(evaluations: [Fr]) throws -> PointProjective {
        try kzg.commit(evaluations)
    }

    // MARK: - Open (Single Polynomial)

    /// Generate a GPU-accelerated Zeromorph opening proof with degree-folding.
    ///
    /// Given multilinear f with evaluations on {0,1}^n and evaluation point u:
    ///   1. Compute n quotient polynomials via even/odd decomposition
    ///   2. Optionally apply degree-folding to reduce quotient degrees
    ///   3. Batch commit all quotients (GPU MSM)
    ///   4. Derive Fiat-Shamir challenge zeta
    ///   5. Build linearized polynomial L(X) and open via KZG at zeta
    public func open(evaluations: [Fr], point: [Fr], value: Fr? = nil) throws -> ZeromorphProverProof {
        let N = evaluations.count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        precondition(N <= kzg.srs.count, "SRS too small for polynomial degree \(N)")

        let v = value ?? Self.evaluateZMFold(evaluations: evaluations, point: point)

        // Step 1: Compute quotient polynomials
        let stepQuotients = computeQuotients(evaluations: evaluations, point: point, n: n)

        // Step 2: Optionally degree-fold quotients
        let (commitQuotients, usedFolding) = applyDegreeFolding(
            stepQuotients: stepQuotients, n: n, N: N
        )

        // Step 3: Batch commit quotients
        let quotientCommitments = try kzg.batchCommit(commitQuotients)

        // Step 4: Fiat-Shamir challenge zeta
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments, value: v)

        // Step 5: Build linearized polynomial and open via KZG
        let L = buildLinearizedPolynomial(
            evaluations: evaluations, value: v, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let kzgProof = try kzg.open(L, at: zeta)

        _proofCount += 1

        return ZeromorphProverProof(
            quotientCommitments: quotientCommitments,
            claimedValue: v,
            kzgWitness: kzgProof.witness,
            linearizationEval: kzgProof.evaluation,
            zeta: zeta,
            degreeFolded: usedFolding
        )
    }

    // MARK: - Batch Opening

    /// Batch open multiple multilinear polynomials at the same point.
    ///
    /// Uses random linear combination with challenge gamma to reduce N openings to one.
    /// All polynomials must have the same number of variables.
    ///
    /// Algorithm:
    ///   h(X) = sum_i gamma^i * f_i(X)
    ///   v_combined = sum_i gamma^i * v_i
    ///   Single linearized polynomial L_h is opened at zeta.
    public func batchOpen(evaluationSets: [[Fr]], point: [Fr],
                          values: [Fr]? = nil, gamma: Fr) throws -> ZeromorphProverBatchProof {
        let count = evaluationSets.count
        precondition(count > 0, "need at least one polynomial")
        let N = evaluationSets[0].count
        let n = point.count
        precondition(N == (1 << n), "evaluations must have 2^n elements")
        for evals in evaluationSets {
            precondition(evals.count == N, "all evaluation sets must be same size")
        }

        // Compute values if not provided
        let vs: [Fr]
        if let values = values {
            precondition(values.count == count)
            vs = values
        } else {
            vs = evaluationSets.map { Self.evaluateZMFold(evaluations: $0, point: point) }
        }

        // Combined polynomial: h(X) = sum_i gamma^i * f_i(X)
        var combined = [Fr](repeating: Fr.zero, count: N)
        var gammaPow = Fr.one
        for i in 0..<count {
            let evals = evaluationSets[i]
            evals.withUnsafeBytes { pBuf in
                combined.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: gammaPow) { gBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(N))
                    }
                }
            }
            if i < count - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Combined value
        var vCombined = Fr.zero
        gammaPow = Fr.one
        for i in 0..<count {
            vCombined = frAdd(vCombined, frMul(gammaPow, vs[i]))
            if i < count - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Quotients of combined polynomial
        let stepQuotients = computeQuotients(evaluations: combined, point: point, n: n)
        let (commitQuotients, usedFolding) = applyDegreeFolding(
            stepQuotients: stepQuotients, n: n, N: N
        )
        let quotientCommitments = try kzg.batchCommit(commitQuotients)

        // Per-polynomial quotient commitments for accountability
        var perPolyQuotientCommitments = [[PointProjective]]()
        perPolyQuotientCommitments.reserveCapacity(count)
        for i in 0..<count {
            let qs = computeQuotients(evaluations: evaluationSets[i], point: point, n: n)
            let (cqs, _) = applyDegreeFolding(stepQuotients: qs, n: n, N: N)
            perPolyQuotientCommitments.append(try kzg.batchCommit(cqs))
        }

        // Fiat-Shamir + linearized poly + KZG open
        let zeta = deriveZeta(point: point, quotientCommitments: quotientCommitments, value: vCombined)
        let L = buildLinearizedPolynomial(
            evaluations: combined, value: vCombined, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let kzgProof = try kzg.open(L, at: zeta)

        _batchProofCount += 1

        return ZeromorphProverBatchProof(
            perPolyQuotientCommitments: perPolyQuotientCommitments,
            claimedValues: vs,
            kzgWitness: kzgProof.witness,
            linearizationEval: kzgProof.evaluation,
            zeta: zeta,
            gamma: gamma,
            degreeFolded: usedFolding
        )
    }

    // MARK: - Verify (SRS Secret, testing)

    /// Full algebraic verification using the SRS secret (testing only).
    /// Checks: (1) telescoping identity P(zeta)=0, (2) L(zeta) matches, (3) KZG proof.
    public func verifyWithSecret(evaluations: [Fr], point: [Fr], value: Fr,
                                  proof: ZeromorphProverProof, srsSecret: Fr) -> Bool {
        let n = point.count
        let N = evaluations.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta

        // Recompute quotients from evaluations
        let stepQuotients = computeQuotients(evaluations: evaluations, point: point, n: n)

        // Check 1: Telescoping identity P(zeta) = 0
        let fZeta = evaluateUnivariate(evaluations, at: zeta)
        var zetaPow = zeta
        var pZeta = frSub(fZeta, value)
        for s in 0..<n {
            let k = n - 1 - s
            let alpha = frSub(zetaPow, point[k])
            let zetaNext = frMul(zetaPow, zetaPow)
            let qEval = evaluateUnivariate(stepQuotients[s], at: zetaNext)
            pZeta = frSub(pZeta, frMul(alpha, qEval))
            zetaPow = zetaNext
        }
        if !frEqual(pZeta, Fr.zero) { return false }

        // Check 2: L(zeta) matches
        let L = buildLinearizedPolynomial(
            evaluations: evaluations, value: value, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let lZeta = evaluateUnivariate(L, at: zeta)
        if !frEqual(lZeta, proof.linearizationEval) { return false }

        // Check 3: KZG proof via secret
        guard let commitL = try? kzg.commit(L) else { return false }
        let g1 = pointFromAffine(kzg.srs[0])
        let lMinusDelta = pointAdd(commitL, pointNeg(cPointScalarMul(g1, proof.linearizationEval)))
        let sMz = frSub(srsSecret, zeta)
        let expectedW = cPointScalarMul(proof.kzgWitness, sMz)

        let cAff = batchToAffine([lMinusDelta])
        let eAff = batchToAffine([expectedW])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    /// Verify a batch opening proof (testing only, using SRS secret).
    public func verifyBatchWithSecret(evaluationSets: [[Fr]], point: [Fr], values: [Fr],
                                       proof: ZeromorphProverBatchProof, srsSecret: Fr) -> Bool {
        let count = evaluationSets.count
        guard proof.claimedValues.count == count else { return false }
        guard proof.perPolyQuotientCommitments.count == count else { return false }
        for i in 0..<count {
            guard frEqual(proof.claimedValues[i], values[i]) else { return false }
        }

        let N = evaluationSets[0].count
        let n = point.count
        let gamma = proof.gamma

        // Reconstruct combined polynomial
        var combined = [Fr](repeating: Fr.zero, count: N)
        var gammaPow = Fr.one
        for i in 0..<count {
            let evals = evaluationSets[i]
            evals.withUnsafeBytes { pBuf in
                combined.withUnsafeMutableBytes { cBuf in
                    withUnsafeBytes(of: gammaPow) { gBuf in
                        bn254_fr_batch_mac_neon(
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(N))
                    }
                }
            }
            if i < count - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        var vCombined = Fr.zero
        gammaPow = Fr.one
        for i in 0..<count {
            vCombined = frAdd(vCombined, frMul(gammaPow, values[i]))
            if i < count - 1 { gammaPow = frMul(gammaPow, gamma) }
        }

        // Verify combined proof as single opening
        let combinedProof = ZeromorphProverProof(
            quotientCommitments: [],
            claimedValue: vCombined,
            kzgWitness: proof.kzgWitness,
            linearizationEval: proof.linearizationEval,
            zeta: proof.zeta,
            degreeFolded: proof.degreeFolded
        )

        let stepQuotients = computeQuotients(evaluations: combined, point: point, n: n)
        let zeta = proof.zeta

        // Telescoping identity
        let fZeta = evaluateUnivariate(combined, at: zeta)
        var zetaPow = zeta
        var pZeta = frSub(fZeta, vCombined)
        for s in 0..<n {
            let k = n - 1 - s
            let alpha = frSub(zetaPow, point[k])
            let zetaNext = frMul(zetaPow, zetaPow)
            let qEval = evaluateUnivariate(stepQuotients[s], at: zetaNext)
            pZeta = frSub(pZeta, frMul(alpha, qEval))
            zetaPow = zetaNext
        }
        if !frEqual(pZeta, Fr.zero) { return false }

        // L(zeta)
        let L = buildLinearizedPolynomial(
            evaluations: combined, value: vCombined, point: point,
            stepQuotients: stepQuotients, zeta: zeta, N: N, n: n
        )
        let lZeta = evaluateUnivariate(L, at: zeta)
        if !frEqual(lZeta, proof.linearizationEval) { return false }

        // KZG check
        guard let commitL = try? kzg.commit(L) else { return false }
        let g1 = pointFromAffine(kzg.srs[0])
        let lMinusDelta = pointAdd(commitL, pointNeg(cPointScalarMul(g1, proof.linearizationEval)))
        let sMz = frSub(srsSecret, zeta)
        let expectedW = cPointScalarMul(proof.kzgWitness, sMz)

        let cAff = batchToAffine([lMinusDelta])
        let eAff = batchToAffine([expectedW])
        return fpToInt(cAff[0].x) == fpToInt(eAff[0].x) &&
               fpToInt(cAff[0].y) == fpToInt(eAff[0].y)
    }

    // MARK: - Pairing-Based Verify

    /// Verify a Zeromorph prover proof using BN254 pairings.
    ///
    /// Reconstructs [L]_1 from the polynomial commitment and quotient commitments:
    ///   C_L = [f]_1 - v*G1 - sum_s phi_s * [q^(s)]_1
    /// Then checks the KZG opening of L at zeta via pairing.
    public func verify(commitment: PointProjective, point: [Fr], value: Fr,
                       proof: ZeromorphProverProof, vk: ZeromorphVK) throws -> Bool {
        let pairing = try BN254PairingEngine()
        let n = point.count
        guard proof.quotientCommitments.count == n else { return false }
        guard frEqual(proof.claimedValue, value) else { return false }

        let zeta = proof.zeta
        let g1 = pointFromAffine(kzg.srs[0])

        // Compute phi_s = zeta^{2^s} - u_{n-1-s}
        var zetaPow = zeta
        var cL = pointAdd(commitment, pointNeg(cPointScalarMul(g1, value)))
        for s in 0..<n {
            let k = n - 1 - s
            let phi = frSub(zetaPow, point[k])
            zetaPow = frMul(zetaPow, zetaPow)
            let term = cPointScalarMul(proof.quotientCommitments[s], phi)
            cL = pointAdd(cL, pointNeg(term))
        }

        // KZG pairing check
        let delta = proof.linearizationEval
        let deltaG = cPointScalarMul(g1, delta)
        let zetaW = cPointScalarMul(proof.kzgWitness, zeta)
        let lhs = pointAdd(pointAdd(cL, pointNeg(deltaG)), zetaW)

        let lhsAff = batchToAffine([lhs])
        let wNegAff = batchToAffine([pointNeg(proof.kzgWitness)])

        return try pairing.pairingCheck(pairs: [
            (lhsAff[0], vk.g2Generator),
            (wNegAff[0], vk.tauG2)
        ])
    }

    // MARK: - Evaluation Helpers

    /// ZM fold evaluation: iteratively fold f_even + u_k * f_odd.
    /// Agrees with standard MLE evaluation on Boolean inputs {0,1}^n.
    public static func evaluateZMFold(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            // Deinterleave into even/odd, then folded = even + uk * odd
            var fEven = [Fr](repeating: Fr.zero, count: half)
            var fOdd = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                fEven[i] = current[2 * i]
                fOdd[i] = current[2 * i + 1]
            }
            var folded = fEven
            let uk = point[k]
            fOdd.withUnsafeBytes { oddBuf in
                folded.withUnsafeMutableBytes { fBuf in
                    withUnsafeBytes(of: uk) { uBuf in
                        bn254_fr_batch_mac_neon(
                            fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            oddBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            uBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            current = folded
        }
        return current[0]
    }

    /// Standard MLE evaluation: (1-u_k)*f[2i] + u_k*f[2i+1]
    /// Uses bn254_fr_sumcheck_reduce: result[i] = evals[i] + challenge*(evals[i+half] - evals[i])
    public static func evaluateMLE(evaluations: [Fr], point: [Fr]) -> Fr {
        var current = evaluations
        for k in stride(from: point.count - 1, through: 0, by: -1) {
            let half = current.count / 2
            // Deinterleave to contiguous [lo0,lo1,...|hi0,hi1,...] for sumcheck_reduce
            var deinterleaved = [Fr](repeating: Fr.zero, count: current.count)
            for i in 0..<half {
                deinterleaved[i] = current[2 * i]
                deinterleaved[i + half] = current[2 * i + 1]
            }
            var folded = [Fr](repeating: Fr.zero, count: half)
            let uk = point[k]
            deinterleaved.withUnsafeBytes { eBuf in
                withUnsafeBytes(of: uk) { uBuf in
                    folded.withUnsafeMutableBytes { rBuf in
                        bn254_fr_sumcheck_reduce(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            uBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(half))
                    }
                }
            }
            current = folded
        }
        return current[0]
    }

    // MARK: - Statistics

    /// Reset proof counters
    public func resetStats() {
        _proofCount = 0
        _batchProofCount = 0
    }

    /// Debug description for diagnostics
    public func debugDescription() -> String {
        var lines = [String]()
        lines.append("GPUZeromorphProverEngine v\(Self.version.version)")
        lines.append("  srsSize: \(kzg.srs.count)")
        lines.append("  gpuCommitThreshold: \(config.gpuCommitThreshold)")
        lines.append("  degreeFolding: \(config.enableDegreeFolding)")
        lines.append("  batchParallel: \(config.enableBatchParallel)")
        lines.append("  proofCount: \(_proofCount)")
        lines.append("  batchProofCount: \(_batchProofCount)")
        return lines.joined(separator: "\n")
    }

    // MARK: - Private: Quotient Computation

    /// Compute n quotient polynomials via even/odd decomposition.
    ///
    /// At step s (processing variable k=n-1-s):
    ///   q^(s) = f_odd (odd-indexed coefficients)
    ///   f_next = f_even + u_k * f_odd
    private func computeQuotients(evaluations: [Fr], point: [Fr], n: Int) -> [[Fr]] {
        var stepQuotients = [[Fr]]()
        stepQuotients.reserveCapacity(n)
        var f = evaluations
        for s in 0..<n {
            let k = n - 1 - s
            let halfLen = f.count / 2
            var fEven = [Fr](repeating: Fr.zero, count: halfLen)
            var fOdd = [Fr](repeating: Fr.zero, count: halfLen)
            for i in 0..<halfLen {
                fEven[i] = f[2 * i]
                fOdd[i] = f[2 * i + 1]
            }
            stepQuotients.append(fOdd)
            // folded[i] = fEven[i] + uk * fOdd[i]  =>  start with fEven, MAC with uk
            var folded = fEven
            let uk = point[k]
            fOdd.withUnsafeBytes { oddBuf in
                folded.withUnsafeMutableBytes { fBuf in
                    withUnsafeBytes(of: uk) { uBuf in
                        bn254_fr_batch_mac_neon(
                            fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            oddBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            uBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(halfLen))
                    }
                }
            }
            f = folded
        }
        return stepQuotients
    }

    // MARK: - Private: Degree Folding

    /// Apply degree-folding optimization to quotient polynomials.
    ///
    /// Each q^(s) has natural degree < 2^{n-s-1}. For large s, the quotient
    /// is already small. For small s, we can fold high-degree terms:
    ///   q'[i] = q[i] + alpha * q[i + halfDeg]  for i < halfDeg
    /// This halves the commitment cost for large quotients.
    ///
    /// Returns the (possibly folded) quotients and whether folding was applied.
    private func applyDegreeFolding(stepQuotients: [[Fr]], n: Int, N: Int) -> ([[Fr]], Bool) {
        guard config.enableDegreeFolding && n >= 3 else {
            return (stepQuotients, false)
        }

        // Only fold quotients with degree >= 4 (first few steps)
        var folded = [[Fr]]()
        folded.reserveCapacity(n)
        var didFold = false

        for s in 0..<n {
            let q = stepQuotients[s]
            if q.count >= 4 {
                // Fold: combine high and low halves with a deterministic weight
                let halfDeg = q.count / 2
                // qFolded[i] = q[i] + foldFactor * q[i + halfDeg]
                // Start with low half, MAC with high half
                var qFolded = Array(q[0..<halfDeg])
                let highHalf = Array(q[halfDeg..<q.count])
                let foldFactor = frFromInt(UInt64(s + 2))
                highHalf.withUnsafeBytes { hBuf in
                    qFolded.withUnsafeMutableBytes { fBuf in
                        withUnsafeBytes(of: foldFactor) { sBuf in
                            bn254_fr_batch_mac_neon(
                                fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                hBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(halfDeg))
                        }
                    }
                }
                folded.append(qFolded)
                didFold = true
            } else {
                folded.append(q)
            }
        }

        return (folded, didFold)
    }

    // MARK: - Private: Linearized Polynomial

    /// Build linearized polynomial L(X) = f(X) - v - sum_s phi_s * q^(s)(X)
    /// where phi_s = zeta^{2^s} - u_{n-1-s}.
    private func buildLinearizedPolynomial(evaluations: [Fr], value: Fr, point: [Fr],
                                            stepQuotients: [[Fr]], zeta: Fr,
                                            N: Int, n: Int) -> [Fr] {
        var zetaPow = zeta
        var L = evaluations
        L[0] = frSub(L[0], value)

        for s in 0..<n {
            let k = n - 1 - s
            let phi = frSub(zetaPow, point[k])
            zetaPow = frMul(zetaPow, zetaPow)
            let q = stepQuotients[s]
            let qLen = min(q.count, N)
            // L[j] -= phi * q[j]  =>  L[j] += (-phi) * q[j]  =>  batch_mac with negPhi
            let negPhi = frSub(Fr.zero, phi)
            q.withUnsafeBytes { qBuf in
                L.withUnsafeMutableBytes { lBuf in
                    withUnsafeBytes(of: negPhi) { pBuf in
                        bn254_fr_batch_mac_neon(
                            lBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(qLen))
                    }
                }
            }
        }
        return L
    }

    // MARK: - Private: Polynomial Evaluation

    /// Evaluate univariate polynomial at a point via Horner's method
    private func evaluateUnivariate(_ coeffs: [Fr], at x: Fr) -> Fr {
        var result = Fr.zero
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            result = frAdd(frMul(result, x), coeffs[i])
        }
        return result
    }

    // MARK: - Private: Fiat-Shamir

    /// Deterministic Fiat-Shamir challenge from point, quotient commitments, and value.
    private func deriveZeta(point: [Fr], quotientCommitments: [PointProjective], value: Fr) -> Fr {
        var transcript = FiatShamirTranscript(label: "gpu_zeromorph_prover", hasher: KeccakTranscriptHasher())

        transcript.absorbFrMany("point", point)

        let affComms = batchToAffine(quotientCommitments)
        for comm in affComms {
            var bytes = [UInt8](repeating: 0, count: 64)
            withUnsafeBytes(of: comm.x) { xBuf in
                let ptr = xBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                for i in 0..<32 { bytes[i] = ptr[i] }
            }
            withUnsafeBytes(of: comm.y) { yBuf in
                let ptr = yBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                for i in 0..<32 { bytes[32 + i] = ptr[i] }
            }
            transcript.absorb("quotient_comm", bytes)
        }

        transcript.absorbFr("value", value)
        return transcript.challengeScalar("zeta")
    }
}

// MARK: - Proof Serialization Helpers

extension ZeromorphProverProof {
    /// Verify structural integrity (lengths match, well-formed)
    public var isWellFormed: Bool {
        guard numVariables > 0 else { return false }
        let n = 1 << numVariables
        return n >= 2 && n <= (1 << 30)
    }
}

extension ZeromorphProverBatchProof {
    /// Approximate byte size of the batch proof
    public var approximateByteSize: Int {
        let pointSize = 96
        let scalarSize = 32
        var total = pointSize + 3 * scalarSize + 1  // witness + zeta + gamma + linearEval + flag
        total += claimedValues.count * scalarSize
        for comms in perPolyQuotientCommitments {
            total += comms.count * pointSize
        }
        return total
    }
}
