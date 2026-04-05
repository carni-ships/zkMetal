// STARK Trace Column Analyzer — detect patterns and suggest compression strategies
//
// Analyzes execution trace columns to find:
// - Constant columns (same value in every row)
// - Periodic columns (repeating pattern with period dividing trace length)
// - Linear dependencies between columns
//
// Uses these analyses to recommend the best compression strategy and
// eliminate redundant data before commitment.

import Foundation

// MARK: - Column Analysis Results

/// Analysis result for a single column.
public enum ColumnKind: Equatable {
    /// Column has the same value in every row
    case constant(M31)
    /// Column repeats a pattern of the given period
    case periodic(period: Int, values: [M31])
    /// Column has no detected pattern (general witness data)
    case general
}

/// Full analysis of all columns in a trace.
public struct ColumnAnalysis {
    /// Per-column analysis results, indexed by column index
    public let columns: [ColumnKind]
    /// Indices of constant columns
    public let constantColumns: [Int]
    /// Indices of periodic columns (not including constants, which are period-1)
    public let periodicColumns: [Int]
    /// Indices of general (non-constant, non-periodic) columns
    public let generalColumns: [Int]
    /// Detected linear dependency groups: each group is a set of column indices
    /// where one column can be expressed as a linear combination of the others
    public let linearDependencies: [[Int]]

    public init(columns: [ColumnKind], constantColumns: [Int], periodicColumns: [Int],
                generalColumns: [Int], linearDependencies: [[Int]]) {
        self.columns = columns
        self.constantColumns = constantColumns
        self.periodicColumns = periodicColumns
        self.generalColumns = generalColumns
        self.linearDependencies = linearDependencies
    }
}

// MARK: - Compression Plan

/// Recommended compression plan based on column analysis.
public struct CompressionPlan {
    /// Columns to eliminate (constant columns stored separately)
    public let eliminateConstant: [Int]
    /// Columns to factor out as periodic (stored as short pattern + period)
    public let eliminatePeriodic: [(column: Int, period: Int)]
    /// Suggested algebraic compression groups for remaining columns
    public let algebraicGroups: [[Int]]
    /// Whether zero-knowledge blinding is recommended
    public let addBlinding: Bool
    /// Estimated compression ratio (compressed size / original size)
    public let estimatedRatio: Double

    public init(eliminateConstant: [Int], eliminatePeriodic: [(column: Int, period: Int)],
                algebraicGroups: [[Int]], addBlinding: Bool, estimatedRatio: Double) {
        self.eliminateConstant = eliminateConstant
        self.eliminatePeriodic = eliminatePeriodic
        self.algebraicGroups = algebraicGroups
        self.addBlinding = addBlinding
        self.estimatedRatio = estimatedRatio
    }
}

// MARK: - Column Analysis

/// Analyze all columns of an execution trace to detect patterns.
///
/// - Parameter trace: Column-major trace [column][row] of M31 elements
/// - Returns: Full analysis including constant, periodic, and general columns
public func analyzeColumns(trace: [[M31]]) -> ColumnAnalysis {
    guard !trace.isEmpty else {
        return ColumnAnalysis(columns: [], constantColumns: [], periodicColumns: [],
                              generalColumns: [], linearDependencies: [])
    }

    let numCols = trace.count
    let numRows = trace[0].count
    var kinds = [ColumnKind](repeating: .general, count: numCols)
    var constants = [Int]()
    var periodics = [Int]()
    var generals = [Int]()

    for col in 0..<numCols {
        let column = trace[col]

        // Check for constant column
        if isConstantColumn(column) {
            kinds[col] = .constant(column[0])
            constants.append(col)
            continue
        }

        // Check for periodic column (try periods 2, 4, 8, ... up to numRows/2)
        if let (period, pattern) = detectPeriod(column, maxPeriod: min(numRows / 2, 256)) {
            kinds[col] = .periodic(period: period, values: pattern)
            periodics.append(col)
            continue
        }

        generals.append(col)
    }

    // Detect linear dependencies among general columns
    let deps = detectLinearDependencies(trace: trace, generalColumns: generals, numRows: numRows)

    return ColumnAnalysis(
        columns: kinds,
        constantColumns: constants,
        periodicColumns: periodics,
        generalColumns: generals,
        linearDependencies: deps
    )
}

// MARK: - Compression Suggestion

/// Recommend the best compression strategy based on column analysis.
///
/// - Parameter analysis: Result from `analyzeColumns`
/// - Returns: A compression plan with estimated savings
public func suggestCompression(analysis: ColumnAnalysis) -> CompressionPlan {
    let totalCols = analysis.columns.count
    guard totalCols > 0 else {
        return CompressionPlan(eliminateConstant: [], eliminatePeriodic: [],
                               algebraicGroups: [], addBlinding: false, estimatedRatio: 1.0)
    }

    // Collect periodic eliminations
    var periodicElim = [(column: Int, period: Int)]()
    for col in analysis.periodicColumns {
        if case .periodic(let period, _) = analysis.columns[col] {
            periodicElim.append((column: col, period: period))
        }
    }

    // Group remaining general columns for algebraic compression
    // Use linear dependency groups if available, otherwise pair adjacent columns
    var algebraicGroups = [[Int]]()
    var usedInDeps = Set<Int>()

    for depGroup in analysis.linearDependencies {
        algebraicGroups.append(depGroup)
        for col in depGroup {
            usedInDeps.insert(col)
        }
    }

    // Pair up remaining general columns not in dependency groups
    let unpaired = analysis.generalColumns.filter { !usedInDeps.contains($0) }
    var i = 0
    while i + 1 < unpaired.count {
        algebraicGroups.append([unpaired[i], unpaired[i + 1]])
        i += 2
    }

    // Estimate compression ratio
    let eliminatedCols = analysis.constantColumns.count + analysis.periodicColumns.count
    let pairedCols = algebraicGroups.reduce(0) { $0 + $1.count }
    let compressedAlgebraic = algebraicGroups.count
    let remainingUnpaired = analysis.generalColumns.count - (pairedCols - usedInDeps.count * 0)
    let remainingGeneral = max(0, unpaired.count - (i > 0 ? i : 0))
    let compressedTotal = compressedAlgebraic + remainingGeneral
    let ratio = totalCols > 0 ? Double(compressedTotal) / Double(totalCols) : 1.0

    return CompressionPlan(
        eliminateConstant: analysis.constantColumns,
        eliminatePeriodic: periodicElim,
        algebraicGroups: algebraicGroups,
        addBlinding: false,
        estimatedRatio: min(ratio, 1.0)
    )
}

// MARK: - Constant Column Elimination

/// Remove constant columns from the trace, returning the reduced trace
/// and a dictionary of the constant values.
///
/// - Parameter trace: Column-major trace [column][row]
/// - Returns: Reduced trace (without constant columns) and constant values keyed by original index
public func eliminateConstantColumns(trace: [[M31]]) -> (trace: [[M31]], constants: [Int: M31]) {
    guard !trace.isEmpty else { return (trace: [], constants: [:]) }

    var reduced = [[M31]]()
    var constants = [Int: M31]()

    for (col, column) in trace.enumerated() {
        if isConstantColumn(column) {
            constants[col] = column[0]
        } else {
            reduced.append(column)
        }
    }

    return (trace: reduced, constants: constants)
}

// MARK: - Periodic Column Elimination

/// Factor out periodic columns from the trace. A periodic column with period k
/// is replaced by storing only k values (the repeating pattern), reducing
/// commitment cost from numRows to k elements.
///
/// - Parameters:
///   - trace: Column-major trace [column][row]
///   - period: Maximum period to detect (columns with period <= this are factored out)
/// - Returns: Reduced trace and the periodic values keyed by original column index
public func eliminatePeriodicColumns(trace: [[M31]], period: Int) -> (trace: [[M31]], periodicValues: [Int: [M31]]) {
    guard !trace.isEmpty else { return (trace: [], periodicValues: [:]) }

    var reduced = [[M31]]()
    var periodicVals = [Int: [M31]]()

    for (col, column) in trace.enumerated() {
        if let (detectedPeriod, pattern) = detectPeriod(column, maxPeriod: period) {
            periodicVals[col] = pattern
            _ = detectedPeriod  // stored implicitly as pattern.count
        } else {
            reduced.append(column)
        }
    }

    return (trace: reduced, periodicValues: periodicVals)
}

// MARK: - Internal Helpers

/// Check if all elements of a column are identical.
private func isConstantColumn(_ column: [M31]) -> Bool {
    guard let first = column.first else { return true }
    for i in 1..<column.count {
        if column[i].v != first.v { return false }
    }
    return true
}

/// Detect the smallest period of a column, up to maxPeriod.
/// Returns (period, pattern) if periodic, nil if not periodic within the bound.
private func detectPeriod(_ column: [M31], maxPeriod: Int) -> (Int, [M31])? {
    let n = column.count
    guard n >= 2 else { return nil }

    // Try periods 1 (constant, already handled), 2, 4, 8, ...
    var period = 2
    while period <= maxPeriod && period <= n / 2 {
        // Check if column repeats with this period
        if n % period == 0 {
            var matches = true
            let pattern = Array(column[0..<period])
            for row in period..<n {
                if column[row].v != pattern[row % period].v {
                    matches = false
                    break
                }
            }
            if matches {
                return (period, pattern)
            }
        }
        period *= 2
    }
    return nil
}

/// Detect linear dependencies among general columns by sampling rows.
/// Two columns are considered linearly dependent if col_a = c * col_b for some constant c,
/// or more generally if col_a + c * col_b = 0 for all rows.
///
/// Uses a sampling approach: check a few rows to find candidate relationships,
/// then verify across all rows.
private func detectLinearDependencies(trace: [[M31]], generalColumns: [Int], numRows: Int) -> [[Int]] {
    guard generalColumns.count >= 2, numRows >= 2 else { return [] }

    var groups = [[Int]]()
    var used = Set<Int>()

    for i in 0..<generalColumns.count {
        let colI = generalColumns[i]
        if used.contains(colI) { continue }

        for j in (i + 1)..<generalColumns.count {
            let colJ = generalColumns[j]
            if used.contains(colJ) { continue }

            // Check if trace[colI] = c * trace[colJ] for some constant c
            // Find c from the first non-zero pair
            if let ratio = findRatio(trace[colI], trace[colJ], numRows: numRows) {
                // Verify the ratio holds for all rows
                var isDependent = true
                for row in 0..<numRows {
                    let expected = m31Mul(ratio, trace[colJ][row])
                    if trace[colI][row].v != expected.v {
                        isDependent = false
                        break
                    }
                }
                if isDependent {
                    groups.append([colI, colJ])
                    used.insert(colI)
                    used.insert(colJ)
                    break
                }
            }
        }
    }

    return groups
}

/// Find the ratio a/b from two columns, using the first row where b is non-zero.
private func findRatio(_ colA: [M31], _ colB: [M31], numRows: Int) -> M31? {
    for row in 0..<numRows {
        if colB[row].v != 0 {
            // ratio = colA[row] / colB[row]
            let inv = m31Inverse(colB[row])
            return m31Mul(colA[row], inv)
        }
    }
    return nil
}
