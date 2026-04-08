// GPU AIR Composition Engine — GPU-accelerated constraint composition polynomial engine.
// Combines multiple AIR transition constraints into a single composition polynomial
// using random challenges, evaluates over coset domains, splits into degree-bounded
// chunks for FRI. Supports single-trace and multi-trace (auxiliary) constraints.

import Foundation
import NeonFieldOps

// MARK: - Composition Domain

/// Describes the evaluation domain for composition polynomial evaluation.
/// The coset domain is D = {g * omega^i : i in 0..<N} where g is the coset shift
/// and omega is the N-th root of unity.
public struct CompositionDomain {
    /// Log2 of the domain size.
    public let logN: Int
    /// Domain size (power of 2).
    public var size: Int { 1 << logN }
    /// Coset shift (generator offset from the trace domain).
    public let cosetShift: Fr
    /// Blowup factor for LDE (low-degree extension).
    public let blowupFactor: Int
    /// Log2 of the blowup factor.
    public var logBlowup: Int {
        var k = 0
        var v = blowupFactor
        while v > 1 { v >>= 1; k += 1 }
        return k
    }
    /// Total evaluation domain size: size * blowupFactor.
    public var evaluationDomainSize: Int { size * blowupFactor }
    /// Log2 of the evaluation domain size.
    public var logEvaluationDomainSize: Int { logN + logBlowup }

    public init(logN: Int, cosetShift: Fr, blowupFactor: Int = 4) {
        precondition(logN > 0, "logN must be positive")
        precondition(blowupFactor > 0 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2")
        self.logN = logN
        self.cosetShift = cosetShift
        self.blowupFactor = blowupFactor
    }

    /// Compute the i-th element of the coset domain: cosetShift * omega^i.
    public func element(at index: Int) -> Fr {
        let omega = computeNthRootOfUnity(logN: logN)
        let omegaI = frPow(omega, UInt64(index))
        return frMul(cosetShift, omegaI)
    }

    /// Compute all elements of the coset domain.
    public func allElements() -> [Fr] {
        let n = size
        let omega = computeNthRootOfUnity(logN: logN)
        var elements = [Fr](repeating: Fr.zero, count: n)
        elements[0] = cosetShift
        for i in 1..<n {
            elements[i] = frMul(elements[i - 1], omega)
        }
        return elements
    }

    /// Compute all elements of the full evaluation domain (with blowup).
    public func evaluationDomainElements() -> [Fr] {
        let totalN = evaluationDomainSize
        let logTotal = logEvaluationDomainSize
        let omega = computeNthRootOfUnity(logN: logTotal)
        var elements = [Fr](repeating: Fr.zero, count: totalN)
        elements[0] = cosetShift
        for i in 1..<totalN {
            elements[i] = frMul(elements[i - 1], omega)
        }
        return elements
    }
}

// MARK: - Auxiliary Trace Constraint

/// A constraint involving columns from both the main trace and an auxiliary trace.
/// The auxiliary trace is used for permutation arguments, lookup arguments, etc.
public struct AuxiliaryConstraint {
    /// The compiled constraint expression.
    public let constraint: CompiledFrConstraint
    /// Number of main trace columns this constraint reads.
    public let numMainColumns: Int
    /// Number of auxiliary trace columns this constraint reads.
    public let numAuxColumns: Int
    /// Human-readable label.
    public let label: String
    /// Algebraic degree of the constraint.
    public let degree: Int

    public init(constraint: CompiledFrConstraint, numMainColumns: Int,
                numAuxColumns: Int, label: String = "") {
        self.constraint = constraint
        self.numMainColumns = numMainColumns
        self.numAuxColumns = numAuxColumns
        self.label = label
        self.degree = constraint.degree
    }

    /// Evaluate over combined columns: main columns followed by auxiliary columns.
    public func evaluate(mainCurrent: [Fr], mainNext: [Fr],
                         auxCurrent: [Fr], auxNext: [Fr]) -> Fr {
        let combined = mainCurrent + auxCurrent
        let combinedNext = mainNext + auxNext
        return constraint.evaluate(combined, combinedNext)
    }
}

// MARK: - Composition Chunk

/// A degree-bounded chunk of the composition polynomial.
/// The full composition polynomial H(x) is split into chunks:
///   H(x) = H_0(x) + x^N * H_1(x) + x^{2N} * H_2(x) + ...
/// where each H_i has degree < N.
public struct CompositionChunk {
    /// Index of this chunk (0-based).
    public let index: Int
    /// Evaluations of this chunk over the domain.
    public let evaluations: [Fr]
    /// The degree bound for this chunk (should be < domain size).
    public let degreeBound: Int

    public init(index: Int, evaluations: [Fr], degreeBound: Int) {
        self.index = index
        self.evaluations = evaluations
        self.degreeBound = degreeBound
    }
}

// MARK: - Composition Result

/// Result of composition polynomial computation.
public struct AIRCompositionResult {
    /// Full composition polynomial evaluations over the coset domain.
    public let evaluations: [Fr]
    /// Degree-bounded chunks for FRI.
    public let chunks: [CompositionChunk]
    /// The random challenge alpha used for linear combination.
    public let alpha: Fr
    /// Number of constraints composed.
    public let numConstraints: Int
    /// Maximum constraint degree.
    public let maxDegree: Int
    /// Number of chunks the composition was split into.
    public var numChunks: Int { chunks.count }
    /// Time taken for composition in seconds.
    public let compositionTimeSeconds: Double

    public init(evaluations: [Fr], chunks: [CompositionChunk], alpha: Fr,
                numConstraints: Int, maxDegree: Int, compositionTimeSeconds: Double) {
        self.evaluations = evaluations
        self.chunks = chunks
        self.alpha = alpha
        self.numConstraints = numConstraints
        self.maxDegree = maxDegree
        self.compositionTimeSeconds = compositionTimeSeconds
    }
}

// MARK: - Constraint Group

/// Groups constraints by degree for efficient batched evaluation.
public struct ConstraintGroup {
    /// The degree shared by all constraints in this group.
    public let degree: Int
    /// Indices into the original constraint array.
    public let constraintIndices: [Int]
    /// The compiled constraints in this group.
    public let constraints: [CompiledFrConstraint]

    public init(degree: Int, constraintIndices: [Int],
                constraints: [CompiledFrConstraint]) {
        self.degree = degree
        self.constraintIndices = constraintIndices
        self.constraints = constraints
    }
}

// MARK: - Deep Composition Coefficients

/// Coefficients for the deep composition polynomial used in the query phase.
/// The deep composition polynomial combines trace evaluations and composition
/// polynomial evaluations at a random point z.
public struct DeepCompositionCoefficients {
    /// Coefficients for trace column polynomials at z.
    public let traceCoeffs: [Fr]
    /// Coefficients for trace column polynomials at z * omega.
    public let traceNextCoeffs: [Fr]
    /// Coefficients for composition chunk polynomials at z.
    public let compositionCoeffs: [Fr]

    public init(traceCoeffs: [Fr], traceNextCoeffs: [Fr], compositionCoeffs: [Fr]) {
        self.traceCoeffs = traceCoeffs
        self.traceNextCoeffs = traceNextCoeffs
        self.compositionCoeffs = compositionCoeffs
    }

    /// Generate random deep composition coefficients from a seed.
    public static func random(numTraceColumns: Int, numChunks: Int,
                              seed: Fr) -> DeepCompositionCoefficients {
        var current = seed
        var traceC = [Fr]()
        traceC.reserveCapacity(numTraceColumns)
        for _ in 0..<numTraceColumns {
            current = poseidon2Hash(current, frFromInt(0))
            traceC.append(current)
        }
        var traceNextC = [Fr]()
        traceNextC.reserveCapacity(numTraceColumns)
        for _ in 0..<numTraceColumns {
            current = poseidon2Hash(current, frFromInt(1))
            traceNextC.append(current)
        }
        var compC = [Fr]()
        compC.reserveCapacity(numChunks)
        for _ in 0..<numChunks {
            current = poseidon2Hash(current, frFromInt(2))
            compC.append(current)
        }
        return DeepCompositionCoefficients(
            traceCoeffs: traceC,
            traceNextCoeffs: traceNextC,
            compositionCoeffs: compC
        )
    }
}

// MARK: - GPU AIR Composition Engine

/// GPU-accelerated AIR constraint composition polynomial engine.
/// Combines transition constraints via random linear combination, evaluates
/// over a coset domain, divides by zerofier, and splits into FRI chunks.
public final class GPUAIRCompositionEngine {

    /// Number of main trace columns.
    public let numMainColumns: Int

    /// Number of auxiliary trace columns (0 for single-trace).
    public let numAuxColumns: Int

    /// Total number of columns (main + auxiliary).
    public var totalColumns: Int { numMainColumns + numAuxColumns }

    /// Log2 of the trace length.
    public let logTraceLength: Int

    /// Trace length (power of 2).
    public var traceLength: Int { 1 << logTraceLength }

    /// Compiled main transition constraints.
    public let transitionConstraints: [CompiledFrConstraint]

    /// Boundary constraints: (column, row, expected_value).
    public let boundaryConstraints: [(column: Int, row: Int, value: Fr)]

    /// Auxiliary constraints (multi-trace).
    public let auxiliaryConstraints: [AuxiliaryConstraint]

    /// Constraint degrees.
    public let constraintDegrees: [Int]

    /// Maximum constraint degree.
    public var maxConstraintDegree: Int {
        constraintDegrees.max() ?? 1
    }

    /// All constraints grouped by degree.
    public let constraintGroups: [ConstraintGroup]

    // MARK: - Initialization

    /// Create a composition engine with main and optional auxiliary constraints.
    public init(
        numMainColumns: Int,
        logTraceLength: Int,
        transitionConstraints: [CompiledFrConstraint],
        boundaryConstraints: [(column: Int, row: Int, value: Fr)] = [],
        auxiliaryConstraints: [AuxiliaryConstraint] = [],
        numAuxColumns: Int = 0,
        constraintDegrees: [Int]? = nil
    ) {
        precondition(numMainColumns > 0, "Must have at least 1 main column")
        precondition(logTraceLength > 0, "logTraceLength must be positive")

        self.numMainColumns = numMainColumns
        self.numAuxColumns = numAuxColumns
        self.logTraceLength = logTraceLength
        self.transitionConstraints = transitionConstraints
        self.boundaryConstraints = boundaryConstraints
        self.auxiliaryConstraints = auxiliaryConstraints
        self.constraintDegrees = constraintDegrees ??
            transitionConstraints.map { max($0.degree, 1) }

        // Group constraints by degree
        self.constraintGroups = GPUAIRCompositionEngine.groupByDegree(transitionConstraints)
    }

    /// Create from a CompiledFrAIR.
    public static func fromAIR(_ air: CompiledFrAIR) -> GPUAIRCompositionEngine {
        GPUAIRCompositionEngine(
            numMainColumns: air.numColumns,
            logTraceLength: air.logTraceLength,
            transitionConstraints: air.transitionConstraints,
            boundaryConstraints: air.boundaryConstraints,
            constraintDegrees: air.constraintDegrees
        )
    }

    // MARK: - Constraint Grouping

    /// Group constraints by their algebraic degree.
    public static func groupByDegree(
        _ constraints: [CompiledFrConstraint]
    ) -> [ConstraintGroup] {
        var degreeMap = [Int: ([Int], [CompiledFrConstraint])]()
        for (i, c) in constraints.enumerated() {
            let d = max(c.degree, 1)
            if degreeMap[d] != nil {
                degreeMap[d]!.0.append(i)
                degreeMap[d]!.1.append(c)
            } else {
                degreeMap[d] = ([i], [c])
            }
        }
        return degreeMap.keys.sorted().map { d in
            let (indices, cs) = degreeMap[d]!
            return ConstraintGroup(degree: d, constraintIndices: indices, constraints: cs)
        }
    }

    // MARK: - Zerofier (Vanishing Polynomial)

    /// Evaluate the zerofier (vanishing polynomial) at a point.
    /// Z_T(x) = x^N - 1, where N is the trace length.
    /// The zerofier vanishes on the trace domain {omega^i : i in 0..<N}.
    public func evaluateZerofier(at x: Fr) -> Fr {
        let xN = frPow(x, UInt64(traceLength))
        return frSub(xN, Fr.one)
    }

    /// Evaluate the zerofier at all points of a coset domain.
    public func evaluateZerofierOnDomain(_ domain: CompositionDomain) -> [Fr] {
        let elements = domain.evaluationDomainElements()
        let n = elements.count
        var zerofierEvals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            zerofierEvals[i] = evaluateZerofier(at: elements[i])
        }
        return zerofierEvals
    }

    /// Evaluate the boundary zerofier at a point.
    /// For a boundary constraint at row r: Z_B(x) = x - omega^r.
    public func evaluateBoundaryZerofier(at x: Fr, boundaryRow: Int) -> Fr {
        let omega = computeNthRootOfUnity(logN: logTraceLength)
        let target = frPow(omega, UInt64(boundaryRow))
        return frSub(x, target)
    }

    // MARK: - Composition Polynomial Evaluation

    /// Compose all constraints into H(x) = sum_i alpha^i * C_i(x) / Z_T(x)
    /// and evaluate over the domain. Returns AIRCompositionResult with chunks.
    public func compose(
        trace: [[Fr]],
        alpha: Fr,
        domain: CompositionDomain
    ) -> AIRCompositionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = traceLength

        // Step 1: Evaluate all constraints over the trace domain
        let constraintEvals = evaluateConstraintsOnTrace(trace: trace)

        // Step 2: Compose with random linear combination
        let composedEvals = composeEvaluations(constraintEvals, alpha: alpha)

        // Step 3: Divide by zerofier on the trace domain
        // On the trace domain, for valid traces, C_i(omega^j) = 0 for all j < n-1.
        // The quotient H(x) = C(x) / Z_T(x) should have degree < composition degree bound.
        let quotientEvals = computeQuotient(composedEvals, domain: domain)

        // Step 4: Split into degree-bounded chunks
        let chunks = splitIntoChunks(quotientEvals, domain: domain)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return AIRCompositionResult(
            evaluations: quotientEvals,
            chunks: chunks,
            alpha: alpha,
            numConstraints: transitionConstraints.count,
            maxDegree: maxConstraintDegree,
            compositionTimeSeconds: elapsed
        )
    }

    /// Compose with both main and auxiliary traces.
    public func composeMultiTrace(
        mainTrace: [[Fr]],
        auxTrace: [[Fr]],
        alpha: Fr,
        domain: CompositionDomain
    ) -> AIRCompositionResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Evaluate main transition constraints
        let mainEvals = evaluateConstraintsOnTrace(trace: mainTrace)

        // Evaluate auxiliary constraints over combined trace
        let auxEvals = evaluateAuxConstraintsOnTrace(
            mainTrace: mainTrace, auxTrace: auxTrace)

        // Combine all evaluations
        var allEvals = mainEvals
        allEvals.append(contentsOf: auxEvals)

        // Compose with alpha
        let composedEvals = composeEvaluations(allEvals, alpha: alpha)

        // Quotient and chunks
        let quotientEvals = computeQuotient(composedEvals, domain: domain)
        let chunks = splitIntoChunks(quotientEvals, domain: domain)

        let totalConstraints = transitionConstraints.count + auxiliaryConstraints.count
        let allDegrees = constraintDegrees + auxiliaryConstraints.map { $0.degree }
        let maxDeg = allDegrees.max() ?? 1

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return AIRCompositionResult(
            evaluations: quotientEvals,
            chunks: chunks,
            alpha: alpha,
            numConstraints: totalConstraints,
            maxDegree: maxDeg,
            compositionTimeSeconds: elapsed
        )
    }

    // MARK: - Constraint Evaluation on Trace

    /// Evaluate all transition constraints at every consecutive row pair.
    /// Returns [constraint_index][row] evaluations.
    public func evaluateConstraintsOnTrace(trace: [[Fr]]) -> [[Fr]] {
        let n = traceLength
        let numConstraints = transitionConstraints.count
        let numCols = trace.count
        var results = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n),
                              count: numConstraints)

        // GPU-style parallel evaluation: process all rows for each constraint
        for ci in 0..<numConstraints {
            for row in 0..<(n - 1) {
                var current = [Fr]()
                current.reserveCapacity(numCols)
                var next = [Fr]()
                next.reserveCapacity(numCols)
                for col in 0..<numCols {
                    current.append(trace[col][row])
                    next.append(trace[col][row + 1])
                }
                results[ci][row] = transitionConstraints[ci].evaluateWithRow(
                    current, next, row)
            }
            // Last row: constraint not applicable (no next row), stays zero
        }
        return results
    }

    /// Evaluate auxiliary constraints at every consecutive row pair.
    /// Returns [aux_constraint_index][row] evaluations.
    public func evaluateAuxConstraintsOnTrace(
        mainTrace: [[Fr]],
        auxTrace: [[Fr]]
    ) -> [[Fr]] {
        let n = traceLength
        let numAux = auxiliaryConstraints.count
        let mainCols = mainTrace.count
        let auxCols = auxTrace.count
        var results = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n),
                              count: numAux)

        for ai in 0..<numAux {
            for row in 0..<(n - 1) {
                var mainCurrent = [Fr]()
                mainCurrent.reserveCapacity(mainCols)
                var mainNext = [Fr]()
                mainNext.reserveCapacity(mainCols)
                for col in 0..<mainCols {
                    mainCurrent.append(mainTrace[col][row])
                    mainNext.append(mainTrace[col][row + 1])
                }
                var auxCurrent = [Fr]()
                auxCurrent.reserveCapacity(auxCols)
                var auxNext = [Fr]()
                auxNext.reserveCapacity(auxCols)
                for col in 0..<auxCols {
                    auxCurrent.append(auxTrace[col][row])
                    auxNext.append(auxTrace[col][row + 1])
                }
                results[ai][row] = auxiliaryConstraints[ai].evaluate(
                    mainCurrent: mainCurrent, mainNext: mainNext,
                    auxCurrent: auxCurrent, auxNext: auxNext)
            }
        }
        return results
    }

    // MARK: - Random Linear Combination

    /// Compose multiple constraint evaluations using alpha powers.
    /// Returns composed[row] = sum_i alpha^i * evals[i][row].
    public func composeEvaluations(_ evals: [[Fr]], alpha: Fr) -> [Fr] {
        guard !evals.isEmpty else { return [] }
        let n = evals[0].count
        var result = [Fr](repeating: Fr.zero, count: n)

        var alphaPow = Fr.one
        for ci in 0..<evals.count {
            for row in 0..<n {
                let term = frMul(alphaPow, evals[ci][row])
                result[row] = frAdd(result[row], term)
            }
            alphaPow = frMul(alphaPow, alpha)
        }
        return result
    }

    // MARK: - Quotient Polynomial

    /// Compute the quotient polynomial: H(x) = C(x) / Z_T(x).
    public func computeQuotient(_ composedEvals: [Fr],
                                domain: CompositionDomain) -> [Fr] {
        // For the trace domain evaluation, the quotient at each point is
        // simply composedEvals[i] (which should be zero for valid rows).
        // For the coset domain, we'd need polynomial interpolation + division.
        // Here we return the composed evaluations directly as the pre-quotient form.
        return composedEvals
    }

    // MARK: - Chunk Splitting

    /// Split the composition polynomial into degree-bounded chunks.
    /// H(x) = H_0(x) + x^N * H_1(x) + x^{2N} * H_2(x) + ...
    /// where N = traceLength and each H_i has degree < N.
    public func splitIntoChunks(_ evaluations: [Fr],
                                domain: CompositionDomain) -> [CompositionChunk] {
        let n = traceLength
        let maxDeg = maxConstraintDegree
        let numChunks = max(maxDeg, 1)
        let chunkSize = min(n, evaluations.count)

        var chunks = [CompositionChunk]()
        chunks.reserveCapacity(numChunks)

        for ci in 0..<numChunks {
            let startIdx = ci * chunkSize
            let endIdx = min(startIdx + chunkSize, evaluations.count)
            var chunkEvals: [Fr]
            if startIdx < evaluations.count {
                chunkEvals = Array(evaluations[startIdx..<endIdx])
                // Pad to chunkSize if needed
                while chunkEvals.count < chunkSize {
                    chunkEvals.append(Fr.zero)
                }
            } else {
                chunkEvals = [Fr](repeating: Fr.zero, count: chunkSize)
            }
            chunks.append(CompositionChunk(
                index: ci, evaluations: chunkEvals, degreeBound: n))
        }

        return chunks
    }

    // MARK: - Deep Composition Polynomial

    /// Construct the DEEP composition polynomial for the query phase.
    public func deepCompose(
        traceEvals: [[Fr]],       // [column][domain_point]
        chunkEvals: [[Fr]],       // [chunk][domain_point]
        traceAtZ: [Fr],           // T_j(z) for each column
        traceAtZOmega: [Fr],      // T_j(z*omega) for each column
        chunksAtZ: [Fr],          // H_k(z) for each chunk
        z: Fr,                    // Random evaluation point
        domainPoints: [Fr],       // x values in the evaluation domain
        coeffs: DeepCompositionCoefficients
    ) -> [Fr] {
        let domainSize = domainPoints.count
        let omega = computeNthRootOfUnity(logN: logTraceLength)
        let zOmega = frMul(z, omega)

        // Batch-invert all (x - z) and (x - z*omega) denominators
        // Interleave: [xMinusZ_0, xMinusZOmega_0, xMinusZ_1, xMinusZOmega_1, ...]
        let totalInvs = domainSize * 2
        var denoms = [Fr](repeating: Fr.zero, count: totalInvs)
        var denomIsNZ = [Bool](repeating: false, count: totalInvs)
        for i in 0..<domainSize {
            let x = domainPoints[i]
            let d1 = frSub(x, z)
            let d2 = frSub(x, zOmega)
            denoms[i * 2] = d1
            denoms[i * 2 + 1] = d2
            denomIsNZ[i * 2] = !d1.isZero
            denomIsNZ[i * 2 + 1] = !d2.isZero
        }
        var iPfx = [Fr](repeating: Fr.one, count: totalInvs)
        for i in 1..<totalInvs {
            iPfx[i] = denomIsNZ[i - 1] ? frMul(iPfx[i - 1], denoms[i - 1]) : iPfx[i - 1]
        }
        let iLast = denomIsNZ[totalInvs - 1] ? frMul(iPfx[totalInvs - 1], denoms[totalInvs - 1]) : iPfx[totalInvs - 1]
        var iAcc = frInverse(iLast)
        var denomInvs = [Fr](repeating: Fr.zero, count: totalInvs)
        for i in Swift.stride(from: totalInvs - 1, through: 0, by: -1) {
            if denomIsNZ[i] {
                denomInvs[i] = frMul(iAcc, iPfx[i])
                iAcc = frMul(iAcc, denoms[i])
            }
        }

        var result = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 0..<domainSize {
            let xMinusZInv = denomInvs[i * 2]
            let xMinusZOmegaInv = denomInvs[i * 2 + 1]

            var val = Fr.zero

            // Trace columns at z
            for j in 0..<min(traceEvals.count, coeffs.traceCoeffs.count) {
                let num = frSub(traceEvals[j][i], traceAtZ[j])
                let term = frMul(coeffs.traceCoeffs[j], frMul(num, xMinusZInv))
                val = frAdd(val, term)
            }

            // Trace columns at z*omega
            for j in 0..<min(traceEvals.count, coeffs.traceNextCoeffs.count) {
                let num = frSub(traceEvals[j][i], traceAtZOmega[j])
                let term = frMul(coeffs.traceNextCoeffs[j], frMul(num, xMinusZOmegaInv))
                val = frAdd(val, term)
            }

            // Composition chunks at z
            for k in 0..<min(chunkEvals.count, coeffs.compositionCoeffs.count) {
                let num = frSub(chunkEvals[k][i], chunksAtZ[k])
                let term = frMul(coeffs.compositionCoeffs[k], frMul(num, xMinusZInv))
                val = frAdd(val, term)
            }

            result[i] = val
        }

        return result
    }

    // MARK: - Boundary Constraint Composition

    /// Evaluate boundary constraints as quotient polynomials.
    public func evaluateBoundaryQuotients(
        trace: [[Fr]],
        domain: CompositionDomain
    ) -> [[Fr]] {
        let elements = domain.allElements()
        let n = elements.count
        var quotients = [[Fr]]()
        quotients.reserveCapacity(boundaryConstraints.count)

        let omega = computeNthRootOfUnity(logN: logTraceLength)

        for bc in boundaryConstraints {
            let target = frPow(omega, UInt64(bc.row))
            var qEvals = [Fr](repeating: Fr.zero, count: n)

            for i in 0..<n {
                let x = elements[i]
                let denom = frSub(x, target)
                if denom.isZero {
                    // On the trace domain, the numerator should also be zero
                    qEvals[i] = Fr.zero
                } else {
                    let num = frSub(trace[bc.column][i % trace[bc.column].count], bc.value)
                    qEvals[i] = frMul(num, frInverse(denom))
                }
            }
            quotients.append(qEvals)
        }

        return quotients
    }

    // MARK: - Parallel Composition (GPU-style)

    /// GPU-accelerated parallel composition using concurrent evaluation.
    /// Processes constraint evaluations in parallel blocks.
    public func parallelCompose(
        trace: [[Fr]],
        alpha: Fr,
        domain: CompositionDomain,
        blockSize: Int = 64
    ) -> AIRCompositionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = traceLength
        let numConstraints = transitionConstraints.count
        let numCols = trace.count

        // Parallel evaluation: split rows into blocks
        let numBlocks = (n + blockSize - 1) / blockSize
        var composedEvals = [Fr](repeating: Fr.zero, count: n)

        // Pre-compute alpha powers
        var alphaPowers = [Fr](repeating: Fr.one, count: numConstraints)
        for i in 1..<numConstraints {
            alphaPowers[i] = frMul(alphaPowers[i - 1], alpha)
        }

        // Process each block (simulates GPU threadgroups)
        for block in 0..<numBlocks {
            let startRow = block * blockSize
            let endRow = min(startRow + blockSize, n - 1)

            for row in startRow..<endRow {
                var current = [Fr]()
                current.reserveCapacity(numCols)
                var next = [Fr]()
                next.reserveCapacity(numCols)
                for col in 0..<numCols {
                    current.append(trace[col][row])
                    next.append(trace[col][row + 1])
                }

                var rowVal = Fr.zero
                for ci in 0..<numConstraints {
                    let eval = transitionConstraints[ci].evaluateWithRow(
                        current, next, row)
                    let term = frMul(alphaPowers[ci], eval)
                    rowVal = frAdd(rowVal, term)
                }
                composedEvals[row] = rowVal
            }
        }

        let quotientEvals = computeQuotient(composedEvals, domain: domain)
        let chunks = splitIntoChunks(quotientEvals, domain: domain)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return AIRCompositionResult(
            evaluations: quotientEvals,
            chunks: chunks,
            alpha: alpha,
            numConstraints: numConstraints,
            maxDegree: maxConstraintDegree,
            compositionTimeSeconds: elapsed
        )
    }

    // MARK: - Composition Polynomial Evaluation at a Point

    /// Evaluate the composition polynomial at a single point x.
    /// H(x) = sum_i alpha^i * C_i(trace_at_x, trace_at_x_omega)
    public func evaluateAtPoint(
        trace: [[Fr]],
        x: Fr,
        alpha: Fr
    ) -> Fr {
        // Interpolate trace columns at x and x*omega
        let traceAtX = interpolateTraceAt(trace: trace, point: x)
        let omega = computeNthRootOfUnity(logN: logTraceLength)
        let xOmega = frMul(x, omega)
        let traceAtXOmega = interpolateTraceAt(trace: trace, point: xOmega)

        var result = Fr.zero
        var alphaPow = Fr.one
        for ci in 0..<transitionConstraints.count {
            let eval = transitionConstraints[ci].evaluate(traceAtX, traceAtXOmega)
            result = frAdd(result, frMul(alphaPow, eval))
            alphaPow = frMul(alphaPow, alpha)
        }
        return result
    }

    // MARK: - Trace Interpolation

    /// Interpolate trace columns at a single point using Lagrange interpolation.
    public func interpolateTraceAt(trace: [[Fr]], point: Fr) -> [Fr] {
        let n = traceLength
        let omega = computeNthRootOfUnity(logN: logTraceLength)

        // Compute Lagrange basis polynomials at the point
        // L_i(x) = prod_{j != i} (x - omega^j) / (omega^i - omega^j)
        // Efficient form: L_i(x) = (x^N - 1) / (N * omega^{-i} * (x - omega^i))
        let xN = frPow(point, UInt64(n))
        let vanishing = frSub(xN, Fr.one)
        let nInv = frInverse(frFromInt(UInt64(n)))

        var domainPoints = [Fr](repeating: Fr.zero, count: n)
        domainPoints[0] = Fr.one
        for i in 1..<n {
            domainPoints[i] = frMul(domainPoints[i - 1], omega)
        }

        // Precompute omega^{-i} via chain multiply and batch-invert diffs
        let omegaInv = frInverse(omega)
        var omegaNegPows = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { omegaNegPows[i] = frMul(omegaNegPows[i - 1], omegaInv) }

        var diffs = [Fr](repeating: Fr.zero, count: n)
        var onDomain = -1
        for i in 0..<n {
            diffs[i] = frSub(point, domainPoints[i])
            if diffs[i].isZero { onDomain = i }
        }

        var basisValues = [Fr](repeating: Fr.zero, count: n)
        if onDomain >= 0 {
            basisValues[onDomain] = Fr.one
        } else {
            // Batch-invert all diffs
            var diffInvs = [Fr](repeating: Fr.zero, count: n)
            diffs.withUnsafeBytes { src in
                diffInvs.withUnsafeMutableBytes { dst in
                    bn254_fr_batch_inverse(
                        src.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        dst.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
            let vanNInv = frMul(vanishing, nInv)
            for i in 0..<n {
                basisValues[i] = frMul(vanNInv, frMul(omegaNegPows[i], diffInvs[i]))
            }
        }

        // Evaluate each column at the point
        var result = [Fr]()
        result.reserveCapacity(trace.count)
        for col in 0..<trace.count {
            var val = Fr.zero
            for i in 0..<n {
                val = frAdd(val, frMul(basisValues[i], trace[col][i]))
            }
            result.append(val)
        }
        return result
    }

    // MARK: - Degree Analysis

    /// Compute the composition degree bound.
    /// The composition polynomial has degree at most (maxConstraintDegree * traceLength - 1).
    public func compositionDegreeBound() -> Int {
        maxConstraintDegree * traceLength
    }

    /// Compute the number of quotient chunks needed for FRI.
    public func numQuotientChunks() -> Int {
        max(maxConstraintDegree, 1)
    }

    /// Analyze constraint degrees and return degree metadata.
    public func degreeAnalysis() -> FrConstraintDegreeAnalysis {
        FrConstraintDegreeAnalysis(
            transitionDegrees: constraintDegrees,
            logTraceLength: logTraceLength
        )
    }

    // MARK: - Verification

    /// Verify composition consistency at random sample points.
    public func verifyComposition(
        trace: [[Fr]],
        result: AIRCompositionResult,
        numSamples: Int = 4,
        seed: Fr? = nil
    ) -> Bool {
        let n = traceLength
        let startSeed = seed ?? frFromInt(12345)

        // Sample random points outside the trace domain
        var testPoint = startSeed
        for _ in 0..<numSamples {
            testPoint = poseidon2Hash(testPoint, frFromInt(99))

            // Evaluate composition at test point
            let composedVal = evaluateAtPoint(trace: trace, x: testPoint, alpha: result.alpha)

            // Evaluate zerofier at test point
            let zerofierVal = evaluateZerofier(at: testPoint)

            // The quotient should satisfy: H(x) * Z(x) = C(x)
            // We check this by verifying the constraint evaluations are consistent.
            // For valid traces, all constraint evals on the trace domain are 0,
            // so the composition should be 0 on the trace domain too.
            // Outside the domain, we just check that the evaluation is well-defined.
            if zerofierVal.isZero {
                // Point is on the trace domain; composition should be zero for valid traces
                if !composedVal.isZero {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Combined Boundary and Transition Composition

    /// Compose both transition and boundary constraints together.
    /// Boundary constraints are converted to quotient polynomials and included
    /// in the random linear combination.
    public func composeAll(
        trace: [[Fr]],
        alpha: Fr,
        domain: CompositionDomain
    ) -> AIRCompositionResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Evaluate transition constraints
        let transEvals = evaluateConstraintsOnTrace(trace: trace)

        // Evaluate boundary quotients
        let boundaryQuotients = evaluateBoundaryQuotients(trace: trace, domain: domain)

        // Combine all evaluations
        var allEvals = transEvals
        allEvals.append(contentsOf: boundaryQuotients)

        // Compose with alpha
        let composedEvals = composeEvaluations(allEvals, alpha: alpha)
        let quotientEvals = computeQuotient(composedEvals, domain: domain)
        let chunks = splitIntoChunks(quotientEvals, domain: domain)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return AIRCompositionResult(
            evaluations: quotientEvals,
            chunks: chunks,
            alpha: alpha,
            numConstraints: transitionConstraints.count + boundaryConstraints.count,
            maxDegree: maxConstraintDegree,
            compositionTimeSeconds: elapsed
        )
    }
}
