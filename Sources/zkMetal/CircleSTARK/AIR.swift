// Circle STARK AIR — Algebraic Intermediate Representation for Circle STARKs over M31
// Defines the interface for arithmetic constraint systems evaluated on circle domains.
//
// Supports:
// - Transition constraints (row-to-row relations)
// - Boundary constraints (specific row/column values)
// - Periodic constraints (repeating patterns)
// - Extension field (QM31) for challenge randomness and soundness

import Foundation

// MARK: - Core AIR Protocol

/// Algebraic Intermediate Representation for Circle STARKs.
/// An AIR defines a computation trace and constraints that must hold on every row.
public protocol CircleAIR {
    /// Number of trace columns
    var numColumns: Int { get }

    /// Log2 of trace length (trace has 2^logTraceLength rows)
    var logTraceLength: Int { get }

    /// Number of trace rows (must be power of 2)
    var traceLength: Int { get }

    /// Number of constraints
    var numConstraints: Int { get }

    /// Maximum degree of each constraint (for quotient degree calculation).
    /// A transition constraint of degree d means the numerator polynomial has degree d * traceLength.
    var constraintDegrees: [Int] { get }

    /// Generate the execution trace: [column][row] of M31 elements.
    /// Each column has exactly `traceLength` elements.
    func generateTrace() -> [[M31]]

    /// Evaluate all transition constraints at a single row.
    /// `current[col]` = trace value at current row, `next[col]` = trace value at next row.
    /// Returns array of constraint evaluations; all should be zero on a valid trace.
    func evaluateConstraints(current: [M31], next: [M31]) -> [M31]

    /// Evaluate boundary constraints.
    /// Returns pairs of (column, row, expected_value) that must hold.
    var boundaryConstraints: [(column: Int, row: Int, value: M31)] { get }
}

extension CircleAIR {
    public var traceLength: Int { 1 << logTraceLength }

    /// Verify that a trace satisfies all AIR constraints (CPU check).
    /// Returns nil if valid, or an error description if invalid.
    public func verifyTrace(_ trace: [[M31]]) -> String? {
        let n = traceLength
        guard trace.count == numColumns else {
            return "Expected \(numColumns) columns, got \(trace.count)"
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == n else {
                return "Column \(ci): expected \(n) rows, got \(col.count)"
            }
        }

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column < numColumns && bc.row < n else {
                return "Boundary constraint out of range: col=\(bc.column), row=\(bc.row)"
            }
            if trace[bc.column][bc.row].v != bc.value.v {
                return "Boundary constraint failed: col=\(bc.column), row=\(bc.row), expected=\(bc.value.v), got=\(trace[bc.column][bc.row].v)"
            }
        }

        // Check transition constraints on all rows except the last
        for i in 0..<(n - 1) {
            let current = (0..<numColumns).map { trace[$0][i] }
            let next = (0..<numColumns).map { trace[$0][i + 1] }
            let evals = evaluateConstraints(current: current, next: next)
            for (ci, ev) in evals.enumerated() {
                if ev.v != 0 {
                    return "Transition constraint \(ci) failed at row \(i): eval=\(ev.v)"
                }
            }
        }

        return nil
    }
}

// MARK: - Constraint Types

/// A boundary constraint: trace[column][row] == value
public struct BoundaryConstraint {
    public let column: Int
    public let row: Int
    public let value: M31

    public init(column: Int, row: Int, value: M31) {
        self.column = column
        self.row = row
        self.value = value
    }
}

/// A periodic constraint with a repeating pattern over the trace.
/// The constraint c(x) should be zero at every row where the period selector is 1.
public struct PeriodicConstraint {
    /// Period in rows (must be power of 2)
    public let period: Int
    /// Offset within the period where the constraint is active
    public let offset: Int
    /// Constraint evaluation function: given current row values, returns constraint value
    public let evaluate: ([M31]) -> M31

    public init(period: Int, offset: Int = 0, evaluate: @escaping ([M31]) -> M31) {
        precondition(period > 0 && (period & (period - 1)) == 0, "Period must be power of 2")
        self.period = period
        self.offset = offset
        self.evaluate = evaluate
    }
}

// MARK: - Generic AIR (declarative constraint definition)

/// A generic AIR defined by closures, useful for testing and simple circuits.
public struct GenericAIR: CircleAIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let numConstraints: Int
    public let constraintDegrees: [Int]
    public let boundaryConstraints: [(column: Int, row: Int, value: M31)]

    private let _generateTrace: () -> [[M31]]
    private let _evaluateConstraints: ([M31], [M31]) -> [M31]

    public init(numColumns: Int, logTraceLength: Int,
                constraintDegrees: [Int],
                boundaryConstraints: [(column: Int, row: Int, value: M31)],
                generateTrace: @escaping () -> [[M31]],
                evaluateConstraints: @escaping ([M31], [M31]) -> [M31]) {
        self.numColumns = numColumns
        self.logTraceLength = logTraceLength
        self.numConstraints = constraintDegrees.count
        self.constraintDegrees = constraintDegrees
        self.boundaryConstraints = boundaryConstraints
        self._generateTrace = generateTrace
        self._evaluateConstraints = evaluateConstraints
    }

    public func generateTrace() -> [[M31]] { _generateTrace() }
    public func evaluateConstraints(current: [M31], next: [M31]) -> [M31] {
        _evaluateConstraints(current, next)
    }
}

// MARK: - Extension Field AIR Evaluation

/// Evaluate AIR constraints over QM31 extension field (for soundness).
/// This lifts M31 constraint evaluations to QM31 by embedding M31 -> QM31
/// and evaluating with QM31 challenge values.
public func evaluateConstraintsQM31<A: CircleAIR>(
    air: A, traceEvals: [[M31]], row: Int,
    nextRow: Int, alpha: QM31
) -> QM31 {
    let current = (0..<air.numColumns).map { traceEvals[$0][row] }
    let next = (0..<air.numColumns).map { traceEvals[$0][nextRow] }
    let constraintVals = air.evaluateConstraints(current: current, next: next)

    // Random linear combination: sum_i alpha^i * c_i(x)
    var result = QM31.zero
    var alphaPow = QM31.one
    for cv in constraintVals {
        let lifted = QM31.from(cv)
        result = qm31Add(result, qm31Mul(alphaPow, lifted))
        alphaPow = qm31Mul(alphaPow, alpha)
    }
    return result
}
