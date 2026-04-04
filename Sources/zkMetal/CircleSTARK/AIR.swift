// Circle STARK AIR — Algebraic Intermediate Representation for Circle STARKs over M31
// Defines the interface for arithmetic constraint systems evaluated on circle domains.

import Foundation

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
}
