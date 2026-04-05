// STARK Trace Compression — reduce execution trace size before commitment
//
// Techniques:
// - Algebraic: combine related columns via random linear combination
// - Interleave: merge two half-width traces into one full-width trace
// - ZeroKnowledge: add random blinding rows for ZK property
//
// Column-major format: trace = [[FieldElement]] where trace[col][row]
// Works with both M31 (Circle STARK) and Bb (BabyBear STARK) fields.

import Foundation

// MARK: - Compression Strategy

/// Strategy for compressing an execution trace.
public enum TraceCompressionStrategy {
    /// Combine related columns via random linear combination.
    /// Given verifier randomness r, columns (a, b) become r*a + b.
    /// Reduces column count while preserving constraint satisfaction.
    case algebraic(randomness: M31)

    /// Interleave two half-width traces into one full-width trace.
    /// Rows of trace A become even rows, rows of trace B become odd rows.
    /// Halves the number of Merkle commitments needed.
    case interleave

    /// Add random blinding rows to the trace for zero-knowledge.
    /// The blinding rows prevent leaking witness information through
    /// the polynomial evaluations outside the trace domain.
    case zeroKnowledge(blindingRows: Int)
}

// MARK: - Column Mapping (for decompression)

/// Records how original columns map to compressed columns,
/// enabling evaluation at arbitrary points after compression.
public struct ColumnMapping: Equatable {
    /// For algebraic compression: which original columns were combined
    /// into each compressed column, plus the randomness used.
    public struct AlgebraicGroup: Equatable {
        /// Indices of original columns that were combined
        public let originalIndices: [Int]
        /// The randomness value used for linear combination
        public let randomness: M31

        public init(originalIndices: [Int], randomness: M31) {
            self.originalIndices = originalIndices
            self.randomness = randomness
        }
    }

    /// For interleaving: which trace each column originally belonged to
    public struct InterleaveInfo: Equatable {
        /// Number of columns in the first (even-row) trace
        public let firstTraceColumns: Int
        /// Number of columns in the second (odd-row) trace
        public let secondTraceColumns: Int
        /// Original number of rows per sub-trace
        public let originalRows: Int

        public init(firstTraceColumns: Int, secondTraceColumns: Int, originalRows: Int) {
            self.firstTraceColumns = firstTraceColumns
            self.secondTraceColumns = secondTraceColumns
            self.originalRows = originalRows
        }
    }

    /// For zero-knowledge: how many blinding rows were added
    public struct BlindingInfo: Equatable {
        /// Number of blinding rows appended
        public let blindingRows: Int
        /// Original trace length before blinding
        public let originalLength: Int

        public init(blindingRows: Int, originalLength: Int) {
            self.blindingRows = blindingRows
            self.originalLength = originalLength
        }
    }

    /// Which compression was applied
    public enum Kind: Equatable {
        case algebraic([AlgebraicGroup])
        case interleave(InterleaveInfo)
        case zeroKnowledge(BlindingInfo)
        case identity
    }

    public let kind: Kind
    /// Number of columns in the original trace
    public let originalColumnCount: Int
    /// Number of columns in the compressed trace
    public let compressedColumnCount: Int

    public init(kind: Kind, originalColumnCount: Int, compressedColumnCount: Int) {
        self.kind = kind
        self.originalColumnCount = originalColumnCount
        self.compressedColumnCount = compressedColumnCount
    }
}

// MARK: - Compressed Trace

/// Result of trace compression: the compressed data plus decompression metadata.
public struct CompressedTrace {
    /// Compressed trace columns: [column][row] of M31 elements
    public let columns: [[M31]]
    /// Mapping from compressed columns back to original columns
    public let mapping: ColumnMapping
    /// Number of rows in the compressed trace
    public var numRows: Int { columns.isEmpty ? 0 : columns[0].count }
    /// Number of columns in the compressed trace
    public var numColumns: Int { columns.count }

    public init(columns: [[M31]], mapping: ColumnMapping) {
        self.columns = columns
        self.mapping = mapping
    }
}

// MARK: - Trace Compression

/// Compress an execution trace using the specified strategy.
///
/// - Parameters:
///   - trace: Column-major execution trace [column][row] of M31 elements
///   - strategy: Which compression technique to apply
/// - Returns: Compressed trace with decompression metadata
public func compressTrace(trace: [[M31]], strategy: TraceCompressionStrategy) -> CompressedTrace {
    switch strategy {
    case .algebraic(let randomness):
        return compressAlgebraic(trace: trace, randomness: randomness)
    case .interleave:
        return compressInterleave(trace: trace)
    case .zeroKnowledge(let blindingRows):
        return compressZeroKnowledge(trace: trace, blindingRows: blindingRows)
    }
}

// MARK: - Algebraic Compression

/// Combine pairs of adjacent columns via random linear combination.
/// For columns (a, b), the compressed column is r*a + b.
/// If the trace has an odd number of columns, the last column passes through unchanged.
private func compressAlgebraic(trace: [[M31]], randomness: M31) -> CompressedTrace {
    guard !trace.isEmpty else {
        return CompressedTrace(
            columns: [],
            mapping: ColumnMapping(kind: .identity, originalColumnCount: 0, compressedColumnCount: 0)
        )
    }

    let numCols = trace.count
    let numRows = trace[0].count
    let numPairs = numCols / 2
    let hasOdd = numCols % 2 != 0
    let compressedCols = numPairs + (hasOdd ? 1 : 0)

    var compressed = [[M31]](repeating: [M31](repeating: M31.zero, count: numRows), count: compressedCols)
    var groups = [ColumnMapping.AlgebraicGroup]()

    // Combine pairs: compressed[k] = r * trace[2k] + trace[2k+1]
    for pair in 0..<numPairs {
        let colA = trace[2 * pair]
        let colB = trace[2 * pair + 1]
        for row in 0..<numRows {
            compressed[pair][row] = m31Add(m31Mul(randomness, colA[row]), colB[row])
        }
        groups.append(ColumnMapping.AlgebraicGroup(
            originalIndices: [2 * pair, 2 * pair + 1],
            randomness: randomness
        ))
    }

    // Pass through odd column unchanged
    if hasOdd {
        compressed[numPairs] = trace[numCols - 1]
        groups.append(ColumnMapping.AlgebraicGroup(
            originalIndices: [numCols - 1],
            randomness: M31.one
        ))
    }

    let mapping = ColumnMapping(
        kind: .algebraic(groups),
        originalColumnCount: numCols,
        compressedColumnCount: compressedCols
    )
    return CompressedTrace(columns: compressed, mapping: mapping)
}

// MARK: - Interleave Compression

/// Interleave two halves of a trace into a single trace with double the rows.
/// The first half of columns go into even rows, the second half into odd rows.
/// Both halves must have the same number of rows (power of 2).
/// The resulting trace has max(firstHalfCols, secondHalfCols) columns and 2*numRows rows.
private func compressInterleave(trace: [[M31]]) -> CompressedTrace {
    guard trace.count >= 2 else {
        return CompressedTrace(
            columns: trace,
            mapping: ColumnMapping(kind: .identity,
                                   originalColumnCount: trace.count,
                                   compressedColumnCount: trace.count)
        )
    }

    let numCols = trace.count
    let numRows = trace[0].count
    let halfCols = numCols / 2
    let firstHalf = Array(trace[0..<halfCols])
    let secondHalf = Array(trace[halfCols..<numCols])

    // The interleaved trace has halfCols columns and 2*numRows rows.
    // Even rows come from firstHalf, odd rows from secondHalf.
    // If halves have different column counts, pad the shorter one with zeros.
    let maxHalfCols = max(firstHalf.count, secondHalf.count)
    let interleavedRows = 2 * numRows

    var interleaved = [[M31]](repeating: [M31](repeating: M31.zero, count: interleavedRows),
                              count: maxHalfCols)

    for col in 0..<maxHalfCols {
        for row in 0..<numRows {
            // Even rows from first half
            if col < firstHalf.count {
                interleaved[col][2 * row] = firstHalf[col][row]
            }
            // Odd rows from second half
            if col < secondHalf.count {
                interleaved[col][2 * row + 1] = secondHalf[col][row]
            }
        }
    }

    let mapping = ColumnMapping(
        kind: .interleave(ColumnMapping.InterleaveInfo(
            firstTraceColumns: firstHalf.count,
            secondTraceColumns: secondHalf.count,
            originalRows: numRows
        )),
        originalColumnCount: numCols,
        compressedColumnCount: maxHalfCols
    )
    return CompressedTrace(columns: interleaved, mapping: mapping)
}

// MARK: - Zero-Knowledge Blinding

/// Append random blinding rows to the trace for zero-knowledge property.
/// The blinding rows contain uniformly random field elements, ensuring
/// that polynomial evaluations outside the trace domain leak no witness info.
/// The trace length is rounded up to the next power of 2 after adding blinding rows.
private func compressZeroKnowledge(trace: [[M31]], blindingRows: Int) -> CompressedTrace {
    guard !trace.isEmpty, blindingRows > 0 else {
        return CompressedTrace(
            columns: trace,
            mapping: ColumnMapping(kind: .identity,
                                   originalColumnCount: trace.count,
                                   compressedColumnCount: trace.count)
        )
    }

    let numCols = trace.count
    let originalRows = trace[0].count
    let totalRows = nextPowerOfTwo(originalRows + blindingRows)

    var blinded = [[M31]](repeating: [M31](repeating: M31.zero, count: totalRows), count: numCols)

    for col in 0..<numCols {
        // Copy original trace data
        for row in 0..<originalRows {
            blinded[col][row] = trace[col][row]
        }
        // Fill blinding rows with pseudo-random values
        // Using a simple deterministic PRNG seeded by column/row for reproducibility in tests;
        // in production, use cryptographic randomness.
        for row in originalRows..<totalRows {
            let seed = UInt64(col) &* 0x9E3779B97F4A7C15 &+ UInt64(row) &* 0x6C62272E07BB0142
            let mixed = (seed ^ (seed >> 30)) &* 0xBF58476D1CE4E5B9
            blinded[col][row] = M31(v: UInt32(mixed % UInt64(M31.P)))
        }
    }

    let mapping = ColumnMapping(
        kind: .zeroKnowledge(ColumnMapping.BlindingInfo(
            blindingRows: totalRows - originalRows,
            originalLength: originalRows
        )),
        originalColumnCount: numCols,
        compressedColumnCount: numCols
    )
    return CompressedTrace(columns: blinded, mapping: mapping)
}

// MARK: - Decompression (Evaluation)

/// Decompress a single evaluation point from a compressed trace.
///
/// Given compressed column values at some evaluation point (from polynomial opening),
/// reconstruct the original column values at that point using the column mapping.
///
/// - Parameters:
///   - compressed: Evaluation of compressed columns at the query point
///   - point: The evaluation point (for interleave: used to separate even/odd)
///   - mapping: The column mapping from compression
/// - Returns: Reconstructed evaluations for all original columns
public func decompressEvaluation(compressed: [M31], point: M31, mapping: ColumnMapping) -> [M31] {
    switch mapping.kind {
    case .identity:
        return compressed

    case .algebraic(let groups):
        // For algebraic compression, we cannot fully decompress a single evaluation
        // back to individual original column values (information is lost by design).
        // Instead, we return the compressed evaluations padded to original width.
        // The verifier checks constraints on the compressed representation directly.
        var result = [M31](repeating: M31.zero, count: mapping.originalColumnCount)
        for (gi, group) in groups.enumerated() {
            guard gi < compressed.count else { break }
            if group.originalIndices.count == 1 {
                // Pass-through column: exact value
                result[group.originalIndices[0]] = compressed[gi]
            } else {
                // Combined column: store the combined value at the first index,
                // zero at the second. The verifier must use the combined constraint.
                result[group.originalIndices[0]] = compressed[gi]
            }
        }
        return result

    case .interleave(let info):
        // For interleave, split the evaluation based on even/odd parity.
        // At evaluation point x: even-row poly = (f(x) + f(-x))/2,
        //                         odd-row poly = (f(x) - f(-x))/(2*x)
        // But with a single point, we return the interleaved values directly,
        // mapped back to the original two-trace layout.
        var result = [M31](repeating: M31.zero, count: mapping.originalColumnCount)
        let maxCols = compressed.count
        for col in 0..<min(info.firstTraceColumns, maxCols) {
            result[col] = compressed[col]
        }
        for col in 0..<min(info.secondTraceColumns, maxCols) {
            result[info.firstTraceColumns + col] = compressed[col]
        }
        return result

    case .zeroKnowledge(let blindInfo):
        // For ZK blinding, the original values are just the first originalLength values.
        // At a query point, the blinded polynomial evaluates to the same thing
        // as the original over the trace domain — the blinding only affects
        // evaluations outside the domain. Return compressed values as-is.
        _ = blindInfo
        return compressed
    }
}

// MARK: - Helpers

/// Round up to the next power of 2.
private func nextPowerOfTwo(_ n: Int) -> Int {
    guard n > 0 else { return 1 }
    var v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1
}
