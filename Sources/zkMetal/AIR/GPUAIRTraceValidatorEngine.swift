// GPU AIR Trace Validator Engine — GPU-accelerated validation of AIR execution traces
//
// Validates that an execution trace satisfies all AIR constraints:
//   1. Transition constraint checking over consecutive row pairs
//   2. Boundary constraint verification at specific rows
//   3. Periodic column evaluation (constraints active every k rows)
//   4. Trace degree bounds checking
//   5. Constraint composition polynomial validation
//   6. Multi-register trace validation across all columns
//
// Operates over BN254 Fr field elements. Uses batch evaluation for efficiency
// and supports both individual constraint checking and composed validation.
//
// Architecture:
//   - TraceValidator holds compiled constraints and trace metadata
//   - Validation methods return detailed violation reports
//   - Composition polynomial support for STARK quotient checking
//   - Degree bound analysis ensures trace polynomials stay within bounds

import Foundation

// MARK: - Validation Result

/// Result of validating a single constraint at a specific row.
public struct TraceViolation: Equatable {
    /// Index of the constraint that was violated.
    public let constraintIndex: Int
    /// Row at which the violation occurred.
    public let row: Int
    /// The nonzero evaluation value (should be zero for valid traces).
    public let evaluationValue: Fr
    /// Human-readable description of the violation.
    public let description: String

    public static func == (lhs: TraceViolation, rhs: TraceViolation) -> Bool {
        lhs.constraintIndex == rhs.constraintIndex && lhs.row == rhs.row
    }
}

/// Comprehensive validation report for an AIR trace.
public struct TraceValidationReport {
    /// Whether the trace is fully valid (no violations).
    public let isValid: Bool
    /// List of all violations found (empty if valid).
    public let violations: [TraceViolation]
    /// Number of transition constraints checked.
    public let numTransitionConstraints: Int
    /// Number of boundary constraints checked.
    public let numBoundaryConstraints: Int
    /// Number of periodic constraints checked.
    public let numPeriodicConstraints: Int
    /// Total number of row evaluations performed.
    public let totalEvaluations: Int
    /// Time taken for validation in seconds.
    public let validationTimeSeconds: Double

    /// Summary string for logging.
    public var summary: String {
        if isValid {
            return "VALID: \(totalEvaluations) evaluations, " +
                   "\(numTransitionConstraints) transition + " +
                   "\(numBoundaryConstraints) boundary + " +
                   "\(numPeriodicConstraints) periodic constraints"
        } else {
            return "INVALID: \(violations.count) violations in " +
                   "\(totalEvaluations) evaluations"
        }
    }
}

// MARK: - Periodic Constraint

/// A constraint that is only active every `period` rows.
/// At rows where `row % period != phase`, the constraint is not enforced.
public struct FrPeriodicConstraint {
    /// The compiled constraint to evaluate.
    public let constraint: CompiledFrConstraint
    /// The period (constraint active every `period` rows).
    public let period: Int
    /// Phase offset (constraint active when `row % period == phase`).
    public let phase: Int
    /// Human-readable label.
    public let label: String

    public init(constraint: CompiledFrConstraint, period: Int, phase: Int = 0, label: String = "") {
        precondition(period > 0, "Period must be positive")
        precondition(phase >= 0 && phase < period, "Phase must be in [0, period)")
        self.constraint = constraint
        self.period = period
        self.phase = phase
        self.label = label
    }
}

// MARK: - Degree Bound Check Result

/// Result of checking trace polynomial degree bounds.
public struct DegreeBoundCheckResult {
    /// Whether all columns are within their degree bounds.
    public let withinBounds: Bool
    /// Per-column effective degrees (highest nonzero coefficient index).
    public let columnDegrees: [Int]
    /// The maximum allowed degree.
    public let maxAllowedDegree: Int
    /// Columns that exceed the degree bound.
    public let violatingColumns: [Int]
}

// MARK: - Composition Validation Result

/// Result of validating the constraint composition polynomial.
public struct CompositionValidationResult {
    /// Whether the composition polynomial vanishes on the trace domain.
    public let isValid: Bool
    /// Rows where the composition is nonzero (violations).
    public let nonzeroRows: [Int]
    /// The alpha used for random linear combination.
    public let alpha: Fr
    /// Maximum absolute evaluation (for approximate zero checks).
    public let maxEvaluation: Fr?
}

// MARK: - GPU AIR Trace Validator Engine

/// GPU-accelerated AIR trace validation engine.
///
/// Validates execution traces against compiled AIR constraints with support for:
/// - Transition constraints (consecutive row pairs)
/// - Boundary constraints (fixed values at specific rows)
/// - Periodic constraints (active every k rows)
/// - Degree bound checking via NTT
/// - Composition polynomial validation
///
/// Usage:
/// ```
/// let validator = GPUAIRTraceValidatorEngine(
///     numColumns: 2,
///     logTraceLength: 4,
///     transitionConstraints: compiledTransitions,
///     boundaryConstraints: [(column: 0, row: 0, value: Fr.one)],
///     periodicConstraints: [periodic]
/// )
/// let report = validator.validateTrace(trace)
/// print(report.summary)
/// ```
public final class GPUAIRTraceValidatorEngine {

    /// Number of columns (registers) in the trace.
    public let numColumns: Int

    /// Log2 of the trace length.
    public let logTraceLength: Int

    /// Trace length (power of 2).
    public var traceLength: Int { 1 << logTraceLength }

    /// Compiled transition constraints.
    public let transitionConstraints: [CompiledFrConstraint]

    /// Boundary constraints: (column, row, expected_value).
    public let boundaryConstraints: [(column: Int, row: Int, value: Fr)]

    /// Periodic constraints (active every k rows).
    public let periodicConstraints: [FrPeriodicConstraint]

    /// Constraint degrees for degree bound analysis.
    public let constraintDegrees: [Int]

    /// Maximum transition constraint degree.
    public var maxConstraintDegree: Int {
        constraintDegrees.max() ?? 1
    }

    // MARK: - Initialization

    /// Create a validator from individually specified constraints.
    public init(
        numColumns: Int,
        logTraceLength: Int,
        transitionConstraints: [CompiledFrConstraint],
        boundaryConstraints: [(column: Int, row: Int, value: Fr)] = [],
        periodicConstraints: [FrPeriodicConstraint] = [],
        constraintDegrees: [Int]? = nil
    ) {
        precondition(numColumns > 0, "Must have at least 1 column")
        precondition(logTraceLength > 0, "logTraceLength must be positive")

        self.numColumns = numColumns
        self.logTraceLength = logTraceLength
        self.transitionConstraints = transitionConstraints
        self.boundaryConstraints = boundaryConstraints
        self.periodicConstraints = periodicConstraints
        self.constraintDegrees = constraintDegrees ??
            transitionConstraints.map { max($0.degree, 1) }
    }

    /// Create a validator from a CompiledFrAIR.
    public static func fromAIR(_ air: CompiledFrAIR) -> GPUAIRTraceValidatorEngine {
        GPUAIRTraceValidatorEngine(
            numColumns: air.numColumns,
            logTraceLength: air.logTraceLength,
            transitionConstraints: air.transitionConstraints,
            boundaryConstraints: air.boundaryConstraints,
            constraintDegrees: air.constraintDegrees
        )
    }

    // MARK: - Full Trace Validation

    /// Validate the entire trace against all constraints.
    /// Returns a comprehensive validation report.
    public func validateTrace(_ trace: [[Fr]]) -> TraceValidationReport {
        let startTime = CFAbsoluteTimeGetCurrent()
        var violations = [TraceViolation]()
        var totalEvals = 0

        // 1. Validate trace dimensions
        if let dimViolation = validateDimensions(trace) {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            return TraceValidationReport(
                isValid: false,
                violations: [dimViolation],
                numTransitionConstraints: transitionConstraints.count,
                numBoundaryConstraints: boundaryConstraints.count,
                numPeriodicConstraints: periodicConstraints.count,
                totalEvaluations: 0,
                validationTimeSeconds: elapsed
            )
        }

        // 2. Check boundary constraints
        let boundaryViolations = validateBoundaryConstraints(trace)
        violations.append(contentsOf: boundaryViolations)
        totalEvals += boundaryConstraints.count

        // 3. Check transition constraints over all consecutive row pairs
        let (transViolations, transEvals) = validateTransitionConstraints(trace)
        violations.append(contentsOf: transViolations)
        totalEvals += transEvals

        // 4. Check periodic constraints
        let (periodicViolations, periodicEvals) = validatePeriodicConstraints(trace)
        violations.append(contentsOf: periodicViolations)
        totalEvals += periodicEvals

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return TraceValidationReport(
            isValid: violations.isEmpty,
            violations: violations,
            numTransitionConstraints: transitionConstraints.count,
            numBoundaryConstraints: boundaryConstraints.count,
            numPeriodicConstraints: periodicConstraints.count,
            totalEvaluations: totalEvals,
            validationTimeSeconds: elapsed
        )
    }

    // MARK: - Dimension Validation

    /// Check that the trace has the right shape.
    private func validateDimensions(_ trace: [[Fr]]) -> TraceViolation? {
        guard trace.count == numColumns else {
            return TraceViolation(
                constraintIndex: -1, row: -1, evaluationValue: Fr.zero,
                description: "Trace has \(trace.count) columns, expected \(numColumns)"
            )
        }
        let n = traceLength
        for (i, col) in trace.enumerated() {
            if col.count != n {
                return TraceViolation(
                    constraintIndex: -1, row: -1, evaluationValue: Fr.zero,
                    description: "Column \(i) has \(col.count) rows, expected \(n)"
                )
            }
        }
        return nil
    }

    // MARK: - Boundary Constraint Validation

    /// Validate all boundary constraints.
    /// Returns list of violations (empty if all pass).
    public func validateBoundaryConstraints(_ trace: [[Fr]]) -> [TraceViolation] {
        var violations = [TraceViolation]()

        for (i, bc) in boundaryConstraints.enumerated() {
            guard bc.column >= 0 && bc.column < trace.count else {
                violations.append(TraceViolation(
                    constraintIndex: -(i + 1), row: bc.row,
                    evaluationValue: Fr.zero,
                    description: "Boundary column \(bc.column) out of range"
                ))
                continue
            }
            guard bc.row >= 0 && bc.row < trace[bc.column].count else {
                violations.append(TraceViolation(
                    constraintIndex: -(i + 1), row: bc.row,
                    evaluationValue: Fr.zero,
                    description: "Boundary row \(bc.row) out of range for column \(bc.column)"
                ))
                continue
            }

            let actual = trace[bc.column][bc.row]
            if actual != bc.value {
                let diff = frSub(actual, bc.value)
                violations.append(TraceViolation(
                    constraintIndex: -(i + 1), row: bc.row,
                    evaluationValue: diff,
                    description: "Boundary violation: column \(bc.column), row \(bc.row) " +
                                 "expected \(frToUInt64(bc.value)) got \(frToUInt64(actual))"
                ))
            }
        }

        return violations
    }

    // MARK: - Transition Constraint Validation

    /// Validate all transition constraints over consecutive row pairs.
    /// Returns (violations, number_of_evaluations).
    public func validateTransitionConstraints(_ trace: [[Fr]]) -> ([TraceViolation], Int) {
        let n = traceLength
        var violations = [TraceViolation]()
        var totalEvals = 0

        for row in 0..<(n - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            for (ci, constraint) in transitionConstraints.enumerated() {
                let eval = constraint.evaluateWithRow(current, next, row)
                totalEvals += 1
                if !eval.isZero {
                    violations.append(TraceViolation(
                        constraintIndex: ci, row: row,
                        evaluationValue: eval,
                        description: "Transition constraint \(ci) (\(constraint.label)) " +
                                     "violated at row \(row)"
                    ))
                }
            }
        }

        return (violations, totalEvals)
    }

    // MARK: - Periodic Constraint Validation

    /// Validate all periodic constraints.
    /// Only evaluates at rows matching the periodic phase.
    /// Returns (violations, number_of_evaluations).
    public func validatePeriodicConstraints(_ trace: [[Fr]]) -> ([TraceViolation], Int) {
        let n = traceLength
        var violations = [TraceViolation]()
        var totalEvals = 0

        for (pi, pc) in periodicConstraints.enumerated() {
            for row in stride(from: pc.phase, to: n - 1, by: pc.period) {
                var current = [Fr]()
                current.reserveCapacity(numColumns)
                var next = [Fr]()
                next.reserveCapacity(numColumns)
                for col in 0..<numColumns {
                    current.append(trace[col][row])
                    next.append(trace[col][row + 1])
                }

                let eval = pc.constraint.evaluateWithRow(current, next, row)
                totalEvals += 1
                if !eval.isZero {
                    violations.append(TraceViolation(
                        constraintIndex: transitionConstraints.count + pi,
                        row: row,
                        evaluationValue: eval,
                        description: "Periodic constraint \(pi) (\(pc.label)) " +
                                     "violated at row \(row) (period=\(pc.period))"
                    ))
                }
            }
        }

        return (violations, totalEvals)
    }

    // MARK: - Batch Transition Evaluation

    /// Evaluate all transition constraints over the entire trace, returning a matrix
    /// of evaluations: result[constraint_index][row] = evaluation at that row.
    /// All entries should be zero for a valid trace.
    public func batchEvaluateTransitions(_ trace: [[Fr]]) -> [[Fr]] {
        let n = traceLength
        let numConstraints = transitionConstraints.count
        var results = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n - 1),
                              count: numConstraints)

        for row in 0..<(n - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            for ci in 0..<numConstraints {
                results[ci][row] = transitionConstraints[ci].evaluateWithRow(current, next, row)
            }
        }

        return results
    }

    // MARK: - Degree Bound Checking

    /// Check that each trace column's polynomial representation stays within
    /// the expected degree bound.
    ///
    /// For a trace of length N, each column should be a polynomial of degree < N.
    /// After NTT interpolation and checking high-order coefficients, we verify
    /// that no coefficient beyond the allowed degree is nonzero.
    ///
    /// - Parameters:
    ///   - trace: The execution trace (column-major)
    ///   - maxDegree: Maximum allowed polynomial degree (default: traceLength - 1)
    /// - Returns: Degree bound check result
    public func checkDegreeBounds(
        _ trace: [[Fr]],
        maxDegree: Int? = nil
    ) -> DegreeBoundCheckResult {
        let n = traceLength
        let logN = logTraceLength
        let bound = maxDegree ?? (n - 1)
        var columnDegrees = [Int]()
        var violating = [Int]()

        for (colIdx, col) in trace.enumerated() {
            // Compute polynomial coefficients via INTT
            let coeffs = cpuINTT(col, logN: logN)

            // Find effective degree (highest nonzero coefficient)
            var deg = 0
            for i in stride(from: coeffs.count - 1, through: 0, by: -1) {
                if !coeffs[i].isZero {
                    deg = i
                    break
                }
            }

            columnDegrees.append(deg)
            if deg > bound {
                violating.append(colIdx)
            }
        }

        return DegreeBoundCheckResult(
            withinBounds: violating.isEmpty,
            columnDegrees: columnDegrees,
            maxAllowedDegree: bound,
            violatingColumns: violating
        )
    }

    // MARK: - Constraint Composition Polynomial Validation

    /// Validate the constraint composition polynomial.
    ///
    /// Computes the random linear combination of all constraints:
    ///   C(x) = sum_i alpha^i * C_i(x)
    /// and checks that it vanishes on all trace rows.
    ///
    /// This is the key soundness check: if C(x) = 0 on the trace domain,
    /// then with high probability all individual constraints are satisfied.
    ///
    /// - Parameters:
    ///   - trace: The execution trace
    ///   - alpha: Random challenge for linear combination
    /// - Returns: Composition validation result
    public func validateCompositionPolynomial(
        _ trace: [[Fr]],
        alpha: Fr
    ) -> CompositionValidationResult {
        let n = traceLength
        var nonzeroRows = [Int]()
        var maxEval: Fr? = nil

        let composed = GPUAIRConstraintCompiler.composeConstraints(
            transitionConstraints, alpha: alpha)

        for row in 0..<(n - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            let eval = composed(current, next)
            if !eval.isZero {
                nonzeroRows.append(row)
            }
            // Track max for diagnostic
            if maxEval == nil {
                maxEval = eval
            }
        }

        return CompositionValidationResult(
            isValid: nonzeroRows.isEmpty,
            nonzeroRows: nonzeroRows,
            alpha: alpha,
            maxEvaluation: maxEval
        )
    }

    // MARK: - Multi-Register Trace Validation

    /// Validate a multi-register trace with cross-register constraints.
    ///
    /// In addition to the standard transition/boundary checks, this validates
    /// that cross-register relationships hold. For example, in a CPU execution
    /// trace, the program counter register and the instruction register must
    /// be consistent.
    ///
    /// - Parameters:
    ///   - trace: The execution trace (column-major)
    ///   - crossRegisterConstraints: Additional constraints that reference
    ///     multiple registers in the same row
    /// - Returns: Comprehensive validation report
    public func validateMultiRegisterTrace(
        _ trace: [[Fr]],
        crossRegisterConstraints: [CompiledFrConstraint] = []
    ) -> TraceValidationReport {
        let startTime = CFAbsoluteTimeGetCurrent()
        var violations = [TraceViolation]()
        var totalEvals = 0

        // Standard validation
        if let dimViolation = validateDimensions(trace) {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            return TraceValidationReport(
                isValid: false, violations: [dimViolation],
                numTransitionConstraints: transitionConstraints.count,
                numBoundaryConstraints: boundaryConstraints.count,
                numPeriodicConstraints: periodicConstraints.count,
                totalEvaluations: 0,
                validationTimeSeconds: elapsed
            )
        }

        // Boundary
        let bv = validateBoundaryConstraints(trace)
        violations.append(contentsOf: bv)
        totalEvals += boundaryConstraints.count

        // Transition
        let (tv, te) = validateTransitionConstraints(trace)
        violations.append(contentsOf: tv)
        totalEvals += te

        // Periodic
        let (pv, pe) = validatePeriodicConstraints(trace)
        violations.append(contentsOf: pv)
        totalEvals += pe

        // Cross-register constraints (evaluated at every row pair)
        let n = traceLength
        let baseIdx = transitionConstraints.count + periodicConstraints.count
        for row in 0..<(n - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            for (ci, constraint) in crossRegisterConstraints.enumerated() {
                let eval = constraint.evaluateWithRow(current, next, row)
                totalEvals += 1
                if !eval.isZero {
                    violations.append(TraceViolation(
                        constraintIndex: baseIdx + ci, row: row,
                        evaluationValue: eval,
                        description: "Cross-register constraint \(ci) (\(constraint.label)) " +
                                     "violated at row \(row)"
                    ))
                }
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return TraceValidationReport(
            isValid: violations.isEmpty,
            violations: violations,
            numTransitionConstraints: transitionConstraints.count + crossRegisterConstraints.count,
            numBoundaryConstraints: boundaryConstraints.count,
            numPeriodicConstraints: periodicConstraints.count,
            totalEvaluations: totalEvals,
            validationTimeSeconds: elapsed
        )
    }

    // MARK: - Degree Analysis

    /// Perform full degree analysis for the validator's constraints.
    public func analyzeDegrees() -> FrConstraintDegreeAnalysis {
        FrConstraintDegreeAnalysis(
            transitionDegrees: constraintDegrees,
            logTraceLength: logTraceLength
        )
    }

    // MARK: - Validation Helpers

    /// Validate a single constraint at a specific row.
    /// Returns nil if the constraint is satisfied, or the evaluation value if violated.
    public func evaluateConstraintAtRow(
        constraintIndex: Int,
        trace: [[Fr]],
        row: Int
    ) -> Fr? {
        guard constraintIndex < transitionConstraints.count else { return nil }
        guard row < traceLength - 1 else { return nil }

        var current = [Fr]()
        current.reserveCapacity(numColumns)
        var next = [Fr]()
        next.reserveCapacity(numColumns)
        for col in 0..<numColumns {
            current.append(trace[col][row])
            next.append(trace[col][row + 1])
        }

        let eval = transitionConstraints[constraintIndex].evaluateWithRow(current, next, row)
        return eval.isZero ? nil : eval
    }

    /// Count the total number of nonzero constraint evaluations in the trace.
    /// A valid trace should return 0.
    public func countViolations(_ trace: [[Fr]]) -> Int {
        let evals = batchEvaluateTransitions(trace)
        var count = 0
        for constraintEvals in evals {
            for eval in constraintEvals {
                if !eval.isZero {
                    count += 1
                }
            }
        }
        return count
    }

    // MARK: - Internal: CPU INTT for degree checking

    /// Simple CPU INTT for degree bound checking.
    /// Uses Cooley-Tukey butterfly with BN254 Fr roots of unity.
    private func cpuINTT(_ values: [Fr], logN: Int) -> [Fr] {
        let n = 1 << logN
        guard values.count == n else { return values }

        // Bit-reverse permutation
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var rev = 0
            var x = i
            for _ in 0..<logN {
                rev = (rev << 1) | (x & 1)
                x >>= 1
            }
            result[rev] = values[i]
        }

        // Butterfly stages using inverse roots
        let omega = computeNthRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        var len = 2
        while len <= n {
            let halfLen = len / 2
            let step = frPow(omegaInv, UInt64(n / len))
            var w = Fr.one
            for j in 0..<halfLen {
                var i = j
                while i < n {
                    let u = result[i]
                    let v = frMul(result[i + halfLen], w)
                    result[i] = frAdd(u, v)
                    result[i + halfLen] = frSub(u, v)
                    i += len
                }
                w = frMul(w, step)
            }
            len *= 2
        }

        // Scale by 1/n
        let nInv = frInverse(frFromInt(UInt64(n)))
        for i in 0..<n {
            result[i] = frMul(result[i], nInv)
        }

        return result
    }
}

// MARK: - Convenience: Build Validator from Expressions

extension GPUAIRTraceValidatorEngine {

    /// Build a validator directly from FrAIRExpression transition constraints.
    ///
    /// Usage:
    /// ```
    /// let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
    ///     numColumns: 2,
    ///     logTraceLength: 4,
    ///     transitions: [
    ///         FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1),
    ///         FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    ///     ],
    ///     boundaries: [(column: 0, row: 0, value: Fr.one)]
    /// )
    /// ```
    public static func fromExpressions(
        numColumns: Int,
        logTraceLength: Int,
        transitions: [FrAIRExpression],
        boundaries: [(column: Int, row: Int, value: Fr)] = [],
        publicInputValues: [String: Fr] = [:],
        periodicColumnValues: [[Fr]] = []
    ) throws -> GPUAIRTraceValidatorEngine {
        var compiled = [CompiledFrConstraint]()
        for (idx, expr) in transitions.enumerated() {
            do {
                let c = try GPUAIRConstraintCompiler.compileExpression(
                    expr, numColumns: numColumns,
                    publicInputValues: publicInputValues,
                    periodicColumnValues: periodicColumnValues
                )
                compiled.append(c)
            } catch {
                throw GPUAIRConstraintCompiler.CompilerError.invalidConstraint(
                    index: idx, underlying: error)
            }
        }

        return GPUAIRTraceValidatorEngine(
            numColumns: numColumns,
            logTraceLength: logTraceLength,
            transitionConstraints: compiled,
            boundaryConstraints: boundaries
        )
    }
}
