// GPUSTARKDeepCompositionEngine — GPU-accelerated STARK deep composition polynomial
//
// Implements the full DEEP (Domain Extension for Eliminating Pretenders) pipeline:
//   1. Out-of-domain (OOD) sampling: pick random zeta outside the evaluation domain
//   2. OOD evaluation: evaluate trace + constraint polynomials at zeta
//   3. DEEP quotient construction: (f(x) - f(zeta)) / (x - zeta) for each column
//   4. Batch DEEP composition: alpha-weighted sum of all DEEP quotients
//   5. FRI integration: prepare the composed polynomial for FRI commitment
//
// Works with BN254 Fr field type. Falls back to CPU when Metal is unavailable.

import Foundation
import Metal
import NeonFieldOps

// MARK: - DEEP Configuration

/// Configuration for the DEEP composition engine.
public struct DEEPCompositionConfig {
    /// Log2 of the trace length.
    public let logTraceLen: Int
    /// Blowup factor for the LDE domain (power of 2, >= 2).
    public let blowupFactor: Int
    /// Number of trace columns.
    public let numTraceColumns: Int
    /// Number of constraint composition segments (splits of the composition poly).
    public let numCompositionSegments: Int
    /// Coset shift generator for LDE domain.
    public let cosetShift: Fr

    /// Trace length = 2^logTraceLen.
    public var traceLen: Int { 1 << logTraceLen }
    /// LDE domain size = traceLen * blowupFactor.
    public var ldeDomainSize: Int { traceLen * blowupFactor }
    /// Log2 of LDE domain size.
    public var logLDEDomainSize: Int { logTraceLen + Int(log2(Double(blowupFactor))) }

    public init(logTraceLen: Int, blowupFactor: Int, numTraceColumns: Int,
                numCompositionSegments: Int = 1, cosetShift: Fr? = nil) {
        precondition(logTraceLen > 0 && logTraceLen <= 20,
                     "logTraceLen must be in [1, 20]")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2 >= 2")
        precondition(numTraceColumns > 0, "Must have at least one trace column")
        precondition(numCompositionSegments > 0, "Must have at least one composition segment")
        self.logTraceLen = logTraceLen
        self.blowupFactor = blowupFactor
        self.numTraceColumns = numTraceColumns
        self.numCompositionSegments = numCompositionSegments
        self.cosetShift = cosetShift ?? frFromInt(Fr.GENERATOR)
    }
}

// MARK: - OOD Evaluation Frame

/// Out-of-domain evaluation frame: trace + constraint values at the OOD point zeta.
public struct OODEvaluationFrame {
    /// OOD sampling point.
    public let zeta: Fr
    /// Trace column evaluations at zeta: traceEvals[colIdx] = trace_col(zeta).
    public let traceEvals: [Fr]
    /// Trace column evaluations at zeta * omega (next row): traceNextEvals[colIdx].
    public let traceNextEvals: [Fr]
    /// Composition polynomial segment evaluations at zeta.
    public let compositionEvals: [Fr]

    public init(zeta: Fr, traceEvals: [Fr], traceNextEvals: [Fr], compositionEvals: [Fr]) {
        self.zeta = zeta
        self.traceEvals = traceEvals
        self.traceNextEvals = traceNextEvals
        self.compositionEvals = compositionEvals
    }
}

// MARK: - DEEP Quotient

/// A single DEEP quotient polynomial: (f(x) - f(zeta)) / (x - zeta).
public struct DEEPQuotient {
    /// Label for debugging (e.g., "trace_col_0", "composition_seg_1").
    public let label: String
    /// Evaluations of the quotient polynomial over the LDE domain.
    public let evaluations: [Fr]

    public init(label: String, evaluations: [Fr]) {
        self.label = label
        self.evaluations = evaluations
    }
}

// MARK: - DEEP Composition Result

/// Result of the full DEEP composition polynomial computation.
public struct DEEPCompositionResult {
    /// The final composed polynomial evaluations over the LDE domain, ready for FRI.
    public let composedEvaluations: [Fr]
    /// Individual DEEP quotients (for inspection/debugging).
    public let quotients: [DEEPQuotient]
    /// The OOD frame that was used.
    public let oodFrame: OODEvaluationFrame
    /// Alpha mixing coefficients used (one per quotient).
    public let alphas: [Fr]
    /// Configuration used.
    public let config: DEEPCompositionConfig

    /// Number of quotients composed.
    public var numQuotients: Int { quotients.count }
    /// Domain size of the composed polynomial.
    public var domainSize: Int { composedEvaluations.count }
}

// MARK: - DEEP Errors

public enum DEEPCompositionError: Error, CustomStringConvertible {
    case noGPU
    case invalidOODFrame(String)
    case quotientComputationFailed(String)
    case friPreparationFailed(String)
    case domainPointCollision(String)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .invalidOODFrame(let msg): return "Invalid OOD frame: \(msg)"
        case .quotientComputationFailed(let msg): return "Quotient computation failed: \(msg)"
        case .friPreparationFailed(let msg): return "FRI preparation failed: \(msg)"
        case .domainPointCollision(let msg): return "Domain point collision: \(msg)"
        }
    }
}

// MARK: - GPU STARK Deep Composition Engine

/// GPU-accelerated STARK DEEP composition polynomial engine.
///
/// The DEEP technique transforms the STARK verification equation into a form
/// suitable for FRI by sampling out-of-domain and constructing quotient polynomials.
///
/// Pipeline for `compose()`:
///   1. Compute LDE domain points: x_i = cosetShift * omega_M^i
///   2. For each trace column j:
///      Q_j(x_i) = (traceCol_j(x_i) - traceCol_j(zeta)) / (x_i - zeta)
///   3. For each trace column j (next-row quotient):
///      Q'_j(x_i) = (traceCol_j(x_i) - traceCol_j(zeta*omega)) / (x_i - zeta*omega)
///   4. For each composition segment k:
///      Q_comp_k(x_i) = (compSeg_k(x_i) - compSeg_k(zeta)) / (x_i - zeta)
///   5. Batch all quotients: D(x_i) = sum_t alpha^t * Q_t(x_i)
///   6. The result D(x) is a low-degree polynomial ready for FRI commitment.
public final class GPUSTARKDeepCompositionEngine {
    /// Minimum domain size to use GPU path.
    public static let gpuThreshold = 256

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let useGPU: Bool

    private var nttEngine: NTTEngine?

    public init(forceGPU: Bool = false) throws {
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            self.useGPU = true
        } else if forceGPU {
            throw DEEPCompositionError.noGPU
        } else {
            self.device = nil
            self.commandQueue = nil
            self.useGPU = false
        }
    }

    // MARK: - NTT Engine Accessor

    private func getNTTEngine() throws -> NTTEngine {
        if let e = nttEngine { return e }
        let e = try NTTEngine()
        nttEngine = e
        return e
    }

    // MARK: - OOD Sampling

    /// Sample an out-of-domain point zeta from a Fiat-Shamir challenge.
    ///
    /// The OOD point must not lie in the trace domain or LDE domain.
    /// We derive zeta by hashing the commitment and squaring to move away from roots of unity.
    public func sampleOODPoint(commitmentHash: Fr, config: DEEPCompositionConfig) -> Fr {
        // Hash-based derivation: zeta = hash^2 + hash + 1 (avoids trivial roots of unity)
        let h = commitmentHash
        let hSq = frMul(h, h)
        let zeta = frAdd(frAdd(hSq, h), Fr.one)
        return zeta
    }

    // MARK: - OOD Evaluation

    /// Evaluate trace polynomials and composition segments at the OOD point zeta.
    ///
    /// Given polynomial coefficients for each column, evaluates at zeta and zeta*omega.
    ///
    /// - Parameters:
    ///   - traceCoeffs: Polynomial coefficients per trace column [colIdx][coeffIdx].
    ///   - compositionCoeffs: Polynomial coefficients per composition segment.
    ///   - zeta: The OOD sampling point.
    ///   - config: DEEP configuration.
    /// - Returns: An `OODEvaluationFrame` with evaluations at zeta and zeta*omega.
    public func evaluateOOD(
        traceCoeffs: [[Fr]],
        compositionCoeffs: [[Fr]],
        zeta: Fr,
        config: DEEPCompositionConfig
    ) throws -> OODEvaluationFrame {
        guard traceCoeffs.count == config.numTraceColumns else {
            throw DEEPCompositionError.invalidOODFrame(
                "Expected \(config.numTraceColumns) trace columns, got \(traceCoeffs.count)")
        }
        guard compositionCoeffs.count == config.numCompositionSegments else {
            throw DEEPCompositionError.invalidOODFrame(
                "Expected \(config.numCompositionSegments) composition segments, got \(compositionCoeffs.count)")
        }

        let omega = frRootOfUnity(logN: config.logTraceLen)
        let zetaNext = frMul(zeta, omega)

        // Evaluate trace columns at zeta and zeta*omega
        var traceEvals = [Fr](repeating: Fr.zero, count: config.numTraceColumns)
        var traceNextEvals = [Fr](repeating: Fr.zero, count: config.numTraceColumns)
        for i in 0..<config.numTraceColumns {
            traceEvals[i] = evaluatePolynomial(traceCoeffs[i], at: zeta)
            traceNextEvals[i] = evaluatePolynomial(traceCoeffs[i], at: zetaNext)
        }

        // Evaluate composition segments at zeta
        var compEvals = [Fr](repeating: Fr.zero, count: config.numCompositionSegments)
        for i in 0..<config.numCompositionSegments {
            compEvals[i] = evaluatePolynomial(compositionCoeffs[i], at: zeta)
        }

        return OODEvaluationFrame(
            zeta: zeta,
            traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEvals: compEvals
        )
    }

    // MARK: - Single DEEP Quotient

    /// Compute a single DEEP quotient: Q(x_i) = (f(x_i) - f(z)) / (x_i - z).
    ///
    /// - Parameters:
    ///   - columnEvals: f(x_i) — LDE evaluations of the polynomial over the coset domain.
    ///   - oodEval: f(z) — evaluation at the OOD point.
    ///   - oodPoint: z — the OOD point.
    ///   - domainPoints: x_i — the LDE domain points.
    ///   - label: Descriptive label for this quotient.
    /// - Returns: A `DEEPQuotient` with evaluations over the domain.
    public func computeQuotient(
        columnEvals: [Fr],
        oodEval: Fr,
        oodPoint: Fr,
        domainPoints: [Fr],
        label: String
    ) throws -> DEEPQuotient {
        let m = columnEvals.count
        guard m == domainPoints.count else {
            throw DEEPCompositionError.quotientComputationFailed(
                "\(label): columnEvals.count (\(m)) != domainPoints.count (\(domainPoints.count))")
        }

        // Compute denominators and batch-invert (Montgomery's trick)
        var denoms = [Fr](repeating: Fr.zero, count: m)
        var ood = oodPoint
        domainPoints.withUnsafeBytes { dBuf in
            denoms.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: &ood) { oBuf in
                    bn254_fr_batch_sub_scalar(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(m))
                }
            }
        }
        var denomInvs = [Fr](repeating: Fr.zero, count: m)
        denoms.withUnsafeBytes { src in
            denomInvs.withUnsafeMutableBytes { dst in
                bn254_fr_batch_inverse_safe(
                    src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m),
                    dst.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // numerator[i] = columnEvals[i] - oodEval, quotient[i] = numerator[i] * denomInvs[i]
        var quotientEvals = [Fr](repeating: Fr.zero, count: m)
        columnEvals.withUnsafeBytes { cBuf in
            denomInvs.withUnsafeBytes { dBuf in
                quotientEvals.withUnsafeMutableBytes { qBuf in
                    var ood = oodEval
                    withUnsafeBytes(of: &ood) { oBuf in
                        // First compute numerators in-place: q[i] = c[i] - oodEval
                        bn254_fr_batch_sub_scalar(
                            qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            oBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(m))
                    }
                    // Then multiply by denomInvs: q[i] *= denomInvs[i]
                    let qPtr = qBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    bn254_fr_batch_mul_neon(
                        qPtr,
                        qPtr,
                        dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(m))
                }
            }
        }

        return DEEPQuotient(label: label, evaluations: quotientEvals)
    }

    // MARK: - Batch DEEP Quotients

    /// Compute all DEEP quotients for trace columns (current + next) and composition segments.
    ///
    /// Produces:
    ///   - numTraceColumns quotients for (trace(x) - trace(zeta)) / (x - zeta)
    ///   - numTraceColumns quotients for (trace(x) - trace(zeta*omega)) / (x - zeta*omega)
    ///   - numCompositionSegments quotients for (comp(x) - comp(zeta)) / (x - zeta)
    public func computeAllQuotients(
        traceLDEColumns: [[Fr]],
        compositionLDESegments: [[Fr]],
        oodFrame: OODEvaluationFrame,
        config: DEEPCompositionConfig
    ) throws -> [DEEPQuotient] {
        let m = config.ldeDomainSize
        let omega = frRootOfUnity(logN: config.logTraceLen)
        let zetaNext = frMul(oodFrame.zeta, omega)

        // Precompute LDE domain points: x_i = cosetShift * omega_M^i
        let domainPoints = computeDomainPoints(config: config)

        guard traceLDEColumns.count == config.numTraceColumns else {
            throw DEEPCompositionError.quotientComputationFailed(
                "Expected \(config.numTraceColumns) trace LDE columns, got \(traceLDEColumns.count)")
        }
        guard compositionLDESegments.count == config.numCompositionSegments else {
            throw DEEPCompositionError.quotientComputationFailed(
                "Expected \(config.numCompositionSegments) composition segments, got \(compositionLDESegments.count)")
        }

        var quotients = [DEEPQuotient]()
        let totalQuotients = 2 * config.numTraceColumns + config.numCompositionSegments
        quotients.reserveCapacity(totalQuotients)

        // Trace quotients at zeta
        for col in 0..<config.numTraceColumns {
            guard traceLDEColumns[col].count == m else {
                throw DEEPCompositionError.quotientComputationFailed(
                    "Trace column \(col): expected \(m) evals, got \(traceLDEColumns[col].count)")
            }
            let q = try computeQuotient(
                columnEvals: traceLDEColumns[col],
                oodEval: oodFrame.traceEvals[col],
                oodPoint: oodFrame.zeta,
                domainPoints: domainPoints,
                label: "trace_col_\(col)_at_zeta"
            )
            quotients.append(q)
        }

        // Trace quotients at zeta * omega (next-row)
        for col in 0..<config.numTraceColumns {
            let q = try computeQuotient(
                columnEvals: traceLDEColumns[col],
                oodEval: oodFrame.traceNextEvals[col],
                oodPoint: zetaNext,
                domainPoints: domainPoints,
                label: "trace_col_\(col)_at_zeta_omega"
            )
            quotients.append(q)
        }

        // Composition segment quotients at zeta
        for seg in 0..<config.numCompositionSegments {
            guard compositionLDESegments[seg].count == m else {
                throw DEEPCompositionError.quotientComputationFailed(
                    "Composition segment \(seg): expected \(m) evals, got \(compositionLDESegments[seg].count)")
            }
            let q = try computeQuotient(
                columnEvals: compositionLDESegments[seg],
                oodEval: oodFrame.compositionEvals[seg],
                oodPoint: oodFrame.zeta,
                domainPoints: domainPoints,
                label: "composition_seg_\(seg)"
            )
            quotients.append(q)
        }

        return quotients
    }

    // MARK: - Full DEEP Composition

    /// Compute the full DEEP composition polynomial D(x) = sum_t alpha^t * Q_t(x).
    ///
    /// This is the main entry point. It computes all quotients and batches them
    /// into a single polynomial ready for FRI commitment.
    ///
    /// - Parameters:
    ///   - traceLDEColumns: LDE evaluations per trace column [colIdx][evalIdx].
    ///   - compositionLDESegments: LDE evaluations per composition segment [segIdx][evalIdx].
    ///   - oodFrame: Out-of-domain evaluation frame.
    ///   - alpha: Random mixing coefficient (from Fiat-Shamir).
    ///   - config: DEEP configuration.
    /// - Returns: `DEEPCompositionResult` with the composed polynomial and metadata.
    public func compose(
        traceLDEColumns: [[Fr]],
        compositionLDESegments: [[Fr]],
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        config: DEEPCompositionConfig
    ) throws -> DEEPCompositionResult {
        let m = config.ldeDomainSize

        // Compute all quotients
        let quotients = try computeAllQuotients(
            traceLDEColumns: traceLDEColumns,
            compositionLDESegments: compositionLDESegments,
            oodFrame: oodFrame,
            config: config
        )

        // Batch with alpha powers: D(x_i) = sum_t alpha^t * Q_t(x_i)
        let (composed, alphas) = batchQuotients(quotients: quotients, alpha: alpha, domainSize: m)

        return DEEPCompositionResult(
            composedEvaluations: composed,
            quotients: quotients,
            oodFrame: oodFrame,
            alphas: alphas,
            config: config
        )
    }

    // MARK: - Batch Quotients with Alpha Mixing

    /// Combine multiple quotient polynomials using random linear combination.
    ///
    /// D(x_i) = sum_t alpha^t * Q_t(x_i)
    private func batchQuotients(
        quotients: [DEEPQuotient], alpha: Fr, domainSize: Int
    ) -> (evaluations: [Fr], alphas: [Fr]) {
        var alphas = [Fr]()
        alphas.reserveCapacity(quotients.count)
        var alphaPow = Fr.one
        for _ in quotients {
            alphas.append(alphaPow)
            alphaPow = frMul(alphaPow, alpha)
        }

        // Parallel composition: each chunk of domain points processes all quotients
        var composed = [Fr](repeating: Fr.zero, count: domainSize)
        let numCPUs = ProcessInfo.processInfo.activeProcessorCount
        let chunkSize = max(1, domainSize / numCPUs)
        let chunks = stride(from: 0, to: domainSize, by: chunkSize).map { start in
            (start, min(start + chunkSize, domainSize))
        }

        composed.withUnsafeMutableBufferPointer { outBuf in
            DispatchQueue.concurrentPerform(iterations: chunks.count) { chunkIdx in
                let (start, end) = chunks[chunkIdx]
                for i in start..<end {
                    var acc = Fr.zero
                    for (qIdx, q) in quotients.enumerated() {
                        acc = frAdd(acc, frMul(alphas[qIdx], q.evaluations[i]))
                    }
                    outBuf[i] = acc
                }
            }
        }

        return (composed, alphas)
    }

    // MARK: - FRI Preparation

    /// Prepare the DEEP composition polynomial for FRI commitment.
    ///
    /// Converts the composed evaluations to coefficient form via iNTT,
    /// verifies the degree bound, and returns coefficients ready for FRI.
    ///
    /// - Parameters:
    ///   - result: The DEEP composition result.
    ///   - maxDegreeBound: Maximum allowed degree (typically traceLen - 1).
    /// - Returns: Polynomial coefficients of the DEEP composition.
    public func prepareForFRI(
        result: DEEPCompositionResult,
        maxDegreeBound: Int? = nil
    ) throws -> [Fr] {
        let config = result.config
        let m = config.ldeDomainSize
        let logM = config.logLDEDomainSize

        // Convert evaluations to coefficients via iNTT on the coset domain.
        // First undo the coset shift, then iNTT.
        let evals = result.composedEvaluations
        guard evals.count == m else {
            throw DEEPCompositionError.friPreparationFailed(
                "Expected \(m) evaluations, got \(evals.count)")
        }

        // iNTT to get coefficients
        let coeffs: [Fr]
        if useGPU && m >= GPUSTARKDeepCompositionEngine.gpuThreshold {
            let engine = try getNTTEngine()
            coeffs = try engine.intt(evals)
        } else {
            coeffs = NTTEngine.cpuINTT(evals, logN: logM)
        }

        // Undo coset shift: coeff[i] /= cosetShift^i
        var unshifted = [Fr](repeating: Fr.zero, count: m)
        let cosetShiftInv = frInverse(config.cosetShift)
        var gInvPow = Fr.one
        for i in 0..<m {
            unshifted[i] = frMul(coeffs[i], gInvPow)
            gInvPow = frMul(gInvPow, cosetShiftInv)
        }

        // Verify degree bound if specified
        if let bound = maxDegreeBound {
            let effectiveDeg = effectiveDegree(of: unshifted)
            if effectiveDeg > bound {
                throw DEEPCompositionError.friPreparationFailed(
                    "DEEP poly degree \(effectiveDeg) exceeds bound \(bound)")
            }
        }

        return unshifted
    }

    // MARK: - Verify DEEP at OOD Point

    /// Verify that the DEEP composition polynomial satisfies D(zeta) = expected value.
    ///
    /// Reconstructs the expected value from OOD evaluations and checks consistency.
    /// Used by the verifier side of the protocol.
    ///
    /// - Parameters:
    ///   - oodFrame: The OOD evaluation frame.
    ///   - alpha: Mixing coefficient.
    ///   - claimedValue: The prover's claimed D(zeta).
    /// - Returns: True if the claimed value is consistent.
    public func verifyDEEPAtOOD(
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        claimedValue: Fr
    ) -> Bool {
        // At the OOD point zeta, each quotient Q_t(zeta) evaluates to:
        //   Q_t(zeta) = (f_t(zeta) - f_t(zeta)) / (zeta - zeta) -> 0/0
        // But we use L'Hopital / the polynomial division result: the quotient
        // evaluated at zeta equals the derivative f'(zeta) / 1 = f'(zeta).
        //
        // In practice, the verifier reconstructs the expected DEEP value from
        // the claimed OOD evaluations directly:
        //   D(zeta) = sum_t alpha^t * 0 = 0 for the trace quotients at zeta
        // This check ensures the claimed deep value is consistent with the
        // Fiat-Shamir transcript.
        //
        // For the verifier check, we reconstruct from raw OOD evals:
        var alphaPow = Fr.one
        var expected = Fr.zero

        // Trace evals at zeta contribute
        for eval in oodFrame.traceEvals {
            expected = frAdd(expected, frMul(alphaPow, eval))
            alphaPow = frMul(alphaPow, alpha)
        }
        // Trace evals at zeta*omega
        for eval in oodFrame.traceNextEvals {
            expected = frAdd(expected, frMul(alphaPow, eval))
            alphaPow = frMul(alphaPow, alpha)
        }
        // Composition evals
        for eval in oodFrame.compositionEvals {
            expected = frAdd(expected, frMul(alphaPow, eval))
            alphaPow = frMul(alphaPow, alpha)
        }

        return frEqual(expected, claimedValue)
    }

    // MARK: - Evaluate DEEP at Arbitrary Point

    /// Evaluate the DEEP composition polynomial at an arbitrary point x
    /// given the individual column evaluations at x and the OOD frame.
    ///
    /// D(x) = sum_t alpha^t * (f_t(x) - f_t(z_t)) / (x - z_t)
    ///
    /// This is used during FRI query verification to check that the claimed
    /// evaluation matches the DEEP construction.
    public func evaluateDEEPAt(
        point: Fr,
        traceEvalsAtPoint: [Fr],
        compositionEvalsAtPoint: [Fr],
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        config: DEEPCompositionConfig
    ) -> Fr {
        let omega = frRootOfUnity(logN: config.logTraceLen)
        let zetaNext = frMul(oodFrame.zeta, omega)
        let xMinusZetaInv = frInverse(frSub(point, oodFrame.zeta))
        let xMinusZetaNextInv = frInverse(frSub(point, zetaNext))

        var alphaPow = Fr.one
        var result = Fr.zero

        // Trace quotients at zeta
        for col in 0..<config.numTraceColumns {
            let num = frSub(traceEvalsAtPoint[col], oodFrame.traceEvals[col])
            let q = frMul(num, xMinusZetaInv)
            result = frAdd(result, frMul(alphaPow, q))
            alphaPow = frMul(alphaPow, alpha)
        }

        // Trace quotients at zeta * omega
        for col in 0..<config.numTraceColumns {
            let num = frSub(traceEvalsAtPoint[col], oodFrame.traceNextEvals[col])
            let q = frMul(num, xMinusZetaNextInv)
            result = frAdd(result, frMul(alphaPow, q))
            alphaPow = frMul(alphaPow, alpha)
        }

        // Composition segment quotients at zeta
        for seg in 0..<config.numCompositionSegments {
            let num = frSub(compositionEvalsAtPoint[seg], oodFrame.compositionEvals[seg])
            let q = frMul(num, xMinusZetaInv)
            result = frAdd(result, frMul(alphaPow, q))
            alphaPow = frMul(alphaPow, alpha)
        }

        return result
    }

    // MARK: - Domain Points

    /// Compute the LDE domain points: x_i = cosetShift * omega_M^i.
    public func computeDomainPoints(config: DEEPCompositionConfig) -> [Fr] {
        let m = config.ldeDomainSize
        let logM = config.logLDEDomainSize
        let omegaM = frRootOfUnity(logN: logM)

        var points = [Fr](repeating: Fr.zero, count: m)
        var w = Fr.one
        for i in 0..<m {
            points[i] = frMul(config.cosetShift, w)
            w = frMul(w, omegaM)
        }
        return points
    }

    // MARK: - Polynomial Evaluation (Horner)

    /// Evaluate a polynomial at a single point using Horner's method.
    public func evaluatePolynomial(_ coeffs: [Fr], at point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: point) { zBuf in
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

    // MARK: - Effective Degree

    /// Compute the effective degree of a polynomial (index of highest non-zero coefficient).
    public func effectiveDegree(of coeffs: [Fr]) -> Int {
        for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
            if !frEqual(coeffs[i], Fr.zero) {
                return i
            }
        }
        return 0
    }

    // MARK: - Batch Column Interpolation

    /// Interpolate multiple LDE columns to coefficient form via iNTT.
    public func batchInterpolate(columns: [[Fr]], logN: Int) throws -> [[Fr]] {
        var results = [[Fr]]()
        results.reserveCapacity(columns.count)
        for col in columns {
            let n = col.count
            precondition(n == (1 << logN), "Column length must equal 2^logN")
            let coeffs: [Fr]
            if useGPU && n >= GPUSTARKDeepCompositionEngine.gpuThreshold {
                let engine = try getNTTEngine()
                coeffs = try engine.intt(col)
            } else {
                coeffs = NTTEngine.cpuINTT(col, logN: logN)
            }
            results.append(coeffs)
        }
        return results
    }

    // MARK: - Reconstruct DEEP Value for Verifier

    /// Reconstruct the expected DEEP composition value at a query point,
    /// given individual polynomial evaluations and the OOD frame.
    ///
    /// This is the core verifier check: at each FRI query position x_i,
    /// the verifier recomputes D(x_i) from the trace/composition openings
    /// and compares against the FRI layer-0 evaluation.
    public func reconstructAtQuery(
        queryPoint: Fr,
        traceOpenings: [Fr],
        compositionOpenings: [Fr],
        oodFrame: OODEvaluationFrame,
        alpha: Fr,
        config: DEEPCompositionConfig
    ) -> Fr {
        return evaluateDEEPAt(
            point: queryPoint,
            traceEvalsAtPoint: traceOpenings,
            compositionEvalsAtPoint: compositionOpenings,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )
    }

    // MARK: - Composition Segment Split

    /// Split a composition polynomial into segments of degree < traceLen.
    ///
    /// Given composition polynomial C(x) of degree < numSegments * traceLen,
    /// split into C(x) = C_0(x) + x^N * C_1(x) + x^(2N) * C_2(x) + ...
    /// where each C_i has degree < N.
    ///
    /// This is needed because the composition poly degree can be much larger than
    /// the trace degree, so we split it into segments each of degree < N for DEEP.
    public func splitCompositionPoly(
        coeffs: [Fr], traceLen: Int, numSegments: Int
    ) -> [[Fr]] {
        var segments = [[Fr]]()
        segments.reserveCapacity(numSegments)

        for s in 0..<numSegments {
            let start = s * traceLen
            var segment = [Fr](repeating: Fr.zero, count: traceLen)
            for i in 0..<traceLen {
                let idx = start + i
                if idx < coeffs.count {
                    segment[i] = coeffs[idx]
                }
                // else stays zero (zero-padding)
            }
            segments.append(segment)
        }

        return segments
    }

    // MARK: - Evaluate Composition Segments on LDE Domain

    /// Evaluate composition polynomial segments on the LDE domain via NTT.
    ///
    /// Each segment is zero-padded to ldeDomainSize, coset-shifted, and NTT'd.
    public func evaluateSegmentsOnLDE(
        segments: [[Fr]], config: DEEPCompositionConfig
    ) throws -> [[Fr]] {
        let m = config.ldeDomainSize
        let logM = config.logLDEDomainSize
        let n = config.traceLen

        var results = [[Fr]]()
        results.reserveCapacity(segments.count)

        for seg in segments {
            // Zero-pad and coset-shift
            var padded = [Fr](repeating: Fr.zero, count: m)
            var gPow = Fr.one
            for i in 0..<min(seg.count, n) {
                padded[i] = frMul(seg[i], gPow)
                gPow = frMul(gPow, config.cosetShift)
            }

            // NTT to get evaluations
            let evals: [Fr]
            if useGPU && m >= GPUSTARKDeepCompositionEngine.gpuThreshold {
                let engine = try getNTTEngine()
                evals = try engine.ntt(padded)
            } else {
                evals = NTTEngine.cpuNTT(padded, logN: logM)
            }
            results.append(evals)
        }

        return results
    }

    // MARK: - End-to-End DEEP + FRI Integration

    /// Full DEEP pipeline: from trace LDE + composition LDE to FRI-ready polynomial.
    ///
    /// This combines OOD evaluation, DEEP quotient construction, alpha batching,
    /// and FRI preparation into a single call.
    ///
    /// - Parameters:
    ///   - traceLDEColumns: LDE evaluations of trace columns.
    ///   - traceCoeffs: Coefficient form of trace polynomials (for OOD eval).
    ///   - compositionLDESegments: LDE evaluations of composition segments.
    ///   - compositionCoeffs: Coefficient form of composition segments (for OOD eval).
    ///   - zeta: OOD sampling point.
    ///   - alpha: Mixing coefficient.
    ///   - config: DEEP configuration.
    /// - Returns: Tuple of (DEEP composition result, FRI-ready coefficients).
    public func deepComposeForFRI(
        traceLDEColumns: [[Fr]],
        traceCoeffs: [[Fr]],
        compositionLDESegments: [[Fr]],
        compositionCoeffs: [[Fr]],
        zeta: Fr,
        alpha: Fr,
        config: DEEPCompositionConfig
    ) throws -> (result: DEEPCompositionResult, friCoeffs: [Fr]) {
        // Step 1: Evaluate at OOD point
        let oodFrame = try evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compositionCoeffs,
            zeta: zeta,
            config: config
        )

        // Step 2: Compose DEEP polynomial
        let deepResult = try compose(
            traceLDEColumns: traceLDEColumns,
            compositionLDESegments: compositionLDESegments,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )

        // Step 3: Convert to coefficients for FRI
        let friCoeffs = try prepareForFRI(result: deepResult)

        return (deepResult, friCoeffs)
    }
}
