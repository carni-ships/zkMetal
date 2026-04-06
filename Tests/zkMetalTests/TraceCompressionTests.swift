// Tests for STARK trace compression and column analysis

import Foundation
import zkMetal

func runTraceCompressionTests() {
    suite("TraceCompression")

    testAlgebraicCompression()
    testAlgebraicCompressionOddColumns()
    testInterleaveCompression()
    testZeroKnowledgeBlinding()
    testColumnAnalyzerConstant()
    testColumnAnalyzerPeriodic()
    testColumnAnalyzerLinearDep()
    testEliminateConstantColumns()
    testEliminatePeriodicColumns()
    testCompressionSuggestion()
    testDecompressEvaluationIdentity()
    testDecompressEvaluationAlgebraic()
    testEmptyTrace()
}

// MARK: - Algebraic Compression Tests

private func testAlgebraicCompression() {
    // 4 columns, 8 rows
    let n = 8
    var trace = [[M31]](repeating: [M31](repeating: M31.zero, count: n), count: 4)
    for col in 0..<4 {
        for row in 0..<n {
            trace[col][row] = M31(v: UInt32(col * 10 + row))
        }
    }

    let r = M31(v: 7)
    let compressed = compressTrace(trace: trace, strategy: .algebraic(randomness: r))

    // 4 columns -> 2 compressed columns
    expectEqual(compressed.numColumns, 2, "algebraic: 4 cols -> 2")
    expectEqual(compressed.numRows, n, "algebraic: row count preserved")

    // Verify: compressed[0][row] = r * trace[0][row] + trace[1][row]
    for row in 0..<n {
        let expected = m31Add(m31Mul(r, trace[0][row]), trace[1][row])
        expectEqual(compressed.columns[0][row], expected,
                    "algebraic: col0 row\(row)")
    }

    // Verify: compressed[1][row] = r * trace[2][row] + trace[3][row]
    for row in 0..<n {
        let expected = m31Add(m31Mul(r, trace[2][row]), trace[3][row])
        expectEqual(compressed.columns[1][row], expected,
                    "algebraic: col1 row\(row)")
    }

    // Check mapping
    if case .algebraic(let groups) = compressed.mapping.kind {
        expectEqual(groups.count, 2, "algebraic: 2 groups")
        expectEqual(groups[0].originalIndices, [0, 1], "algebraic: group0 indices")
        expectEqual(groups[1].originalIndices, [2, 3], "algebraic: group1 indices")
    } else {
        expect(false, "algebraic: wrong mapping kind")
    }
}

private func testAlgebraicCompressionOddColumns() {
    // 3 columns: first pair combined, last column passes through
    let n = 4
    var trace = [[M31]](repeating: [M31](repeating: M31.zero, count: n), count: 3)
    for col in 0..<3 {
        for row in 0..<n {
            trace[col][row] = M31(v: UInt32(col + row * 3 + 1))
        }
    }

    let r = M31(v: 5)
    let compressed = compressTrace(trace: trace, strategy: .algebraic(randomness: r))

    expectEqual(compressed.numColumns, 2, "algebraic odd: 3 cols -> 2")

    // Last column passes through unchanged
    for row in 0..<n {
        expectEqual(compressed.columns[1][row], trace[2][row],
                    "algebraic odd: passthrough row\(row)")
    }
}

// MARK: - Interleave Compression Tests

private func testInterleaveCompression() {
    // 4 columns, 8 rows -> interleave: first 2 cols (even rows), last 2 cols (odd rows)
    let n = 8
    var trace = [[M31]](repeating: [M31](repeating: M31.zero, count: n), count: 4)
    for col in 0..<4 {
        for row in 0..<n {
            trace[col][row] = M31(v: UInt32(col * 100 + row))
        }
    }

    let compressed = compressTrace(trace: trace, strategy: .interleave)

    // 4 cols split into 2+2, interleaved into 2 columns with 16 rows
    expectEqual(compressed.numColumns, 2, "interleave: 4->2 cols")
    expectEqual(compressed.numRows, 16, "interleave: 8->16 rows")

    // Even rows = first half columns
    for col in 0..<2 {
        for row in 0..<n {
            expectEqual(compressed.columns[col][2 * row], trace[col][row],
                        "interleave: even row\(row) col\(col)")
        }
    }

    // Odd rows = second half columns
    for col in 0..<2 {
        for row in 0..<n {
            expectEqual(compressed.columns[col][2 * row + 1], trace[2 + col][row],
                        "interleave: odd row\(row) col\(col)")
        }
    }

    // Check mapping
    if case .interleave(let info) = compressed.mapping.kind {
        expectEqual(info.firstTraceColumns, 2, "interleave: first half cols")
        expectEqual(info.secondTraceColumns, 2, "interleave: second half cols")
        expectEqual(info.originalRows, n, "interleave: original rows")
    } else {
        expect(false, "interleave: wrong mapping kind")
    }
}

// MARK: - Zero-Knowledge Blinding Tests

private func testZeroKnowledgeBlinding() {
    let n = 8
    var trace = [[M31]](repeating: [M31](repeating: M31.zero, count: n), count: 2)
    for col in 0..<2 {
        for row in 0..<n {
            trace[col][row] = M31(v: UInt32(col * 10 + row + 1))
        }
    }

    let blindingRows = 3
    let compressed = compressTrace(trace: trace, strategy: .zeroKnowledge(blindingRows: blindingRows))

    // 8 original + 3 blinding -> next power of 2 = 16
    expectEqual(compressed.numRows, 16, "zk: padded to 16 rows")
    expectEqual(compressed.numColumns, 2, "zk: column count preserved")

    // Original data preserved
    for col in 0..<2 {
        for row in 0..<n {
            expectEqual(compressed.columns[col][row], trace[col][row],
                        "zk: original data col\(col) row\(row)")
        }
    }

    // Blinding rows are non-zero (with high probability)
    var hasNonZeroBlinding = false
    for col in 0..<2 {
        for row in n..<16 {
            if compressed.columns[col][row].v != 0 {
                hasNonZeroBlinding = true
                break
            }
        }
    }
    expect(hasNonZeroBlinding, "zk: blinding rows contain non-zero values")

    // Check mapping
    if case .zeroKnowledge(let info) = compressed.mapping.kind {
        expectEqual(info.originalLength, n, "zk: original length")
        expectEqual(info.blindingRows, 8, "zk: actual blinding rows (padded)")
    } else {
        expect(false, "zk: wrong mapping kind")
    }
}

// MARK: - Column Analyzer Tests

private func testColumnAnalyzerConstant() {
    let n = 16
    // Column 0: constant 42
    // Column 1: varying
    let col0 = [M31](repeating: M31(v: 42), count: n)
    var col1 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col1[i] = M31(v: UInt32(i)) }

    let analysis = analyzeColumns(trace: [col0, col1])

    expectEqual(analysis.constantColumns, [0], "analyzer: constant col detected")
    expectEqual(analysis.generalColumns, [1], "analyzer: general col detected")

    if case .constant(let val) = analysis.columns[0] {
        expectEqual(val, M31(v: 42), "analyzer: constant value")
    } else {
        expect(false, "analyzer: expected constant kind")
    }
}

private func testColumnAnalyzerPeriodic() {
    let n = 16
    // Column 0: periodic with period 4: [1, 2, 3, 4, 1, 2, 3, 4, ...]
    var col0 = [M31](repeating: M31.zero, count: n)
    let pattern: [UInt32] = [1, 2, 3, 4]
    for i in 0..<n { col0[i] = M31(v: pattern[i % 4]) }

    // Column 1: periodic with period 2: [10, 20, 10, 20, ...]
    var col1 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col1[i] = M31(v: i % 2 == 0 ? 10 : 20) }

    // Column 2: general
    var col2 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col2[i] = M31(v: UInt32(i * i)) }

    let analysis = analyzeColumns(trace: [col0, col1, col2])

    expectEqual(analysis.periodicColumns.count, 2, "analyzer: 2 periodic cols")
    expect(analysis.periodicColumns.contains(0), "analyzer: col0 is periodic")
    expect(analysis.periodicColumns.contains(1), "analyzer: col1 is periodic")
    expectEqual(analysis.generalColumns, [2], "analyzer: col2 is general")

    if case .periodic(let period, let values) = analysis.columns[0] {
        expectEqual(period, 4, "analyzer: col0 period")
        expectEqual(values.count, 4, "analyzer: col0 pattern length")
    } else {
        expect(false, "analyzer: expected periodic kind for col0")
    }

    if case .periodic(let period, _) = analysis.columns[1] {
        expectEqual(period, 2, "analyzer: col1 period")
    } else {
        expect(false, "analyzer: expected periodic kind for col1")
    }
}

private func testColumnAnalyzerLinearDep() {
    let n = 8
    // col0: [1, 2, 3, 4, 5, 6, 7, 8]
    // col1: [3, 6, 9, 12, 15, 18, 21, 24]  (= 3 * col0)
    // col2: [10, 20, 30, 40, 50, 60, 70, 80] (= 10 * col0)... different ratio
    var col0 = [M31](repeating: M31.zero, count: n)
    var col1 = [M31](repeating: M31.zero, count: n)
    var col2 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n {
        col0[i] = M31(v: UInt32(i + 1))
        col1[i] = M31(v: UInt32((i + 1) * 3))
        col2[i] = M31(v: UInt32(i * i + 1))  // independent
    }

    let analysis = analyzeColumns(trace: [col0, col1, col2])

    // col0 and col1 are linearly dependent (col1 = 3 * col0)
    expectEqual(analysis.linearDependencies.count, 1, "linear dep: 1 group")
    if !analysis.linearDependencies.isEmpty {
        let group = analysis.linearDependencies[0]
        expect(group.contains(0) && group.contains(1),
               "linear dep: cols 0 and 1 grouped")
    }
}

// MARK: - Elimination Tests

private func testEliminateConstantColumns() {
    let n = 8
    let col0 = [M31](repeating: M31(v: 42), count: n)
    var col1 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col1[i] = M31(v: UInt32(i)) }
    let col2 = [M31](repeating: M31(v: 99), count: n)

    let result = eliminateConstantColumns(trace: [col0, col1, col2])

    expectEqual(result.trace.count, 1, "eliminate const: 1 col remaining")
    expectEqual(result.constants.count, 2, "eliminate const: 2 constants found")
    expectEqual(result.constants[0], M31(v: 42), "eliminate const: col0 value")
    expectEqual(result.constants[2], M31(v: 99), "eliminate const: col2 value")

    // Remaining column should be col1
    for i in 0..<n {
        expectEqual(result.trace[0][i], M31(v: UInt32(i)),
                    "eliminate const: remaining col data")
    }
}

private func testEliminatePeriodicColumns() {
    let n = 16
    // Periodic column with period 2
    var col0 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col0[i] = M31(v: i % 2 == 0 ? 5 : 10) }

    // General column
    var col1 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { col1[i] = M31(v: UInt32(i * 7 + 3)) }

    let result = eliminatePeriodicColumns(trace: [col0, col1], period: 8)

    expectEqual(result.trace.count, 1, "eliminate periodic: 1 col remaining")
    expectEqual(result.periodicValues.count, 1, "eliminate periodic: 1 periodic found")

    if let pattern = result.periodicValues[0] {
        expectEqual(pattern.count, 2, "eliminate periodic: period 2")
        expectEqual(pattern[0], M31(v: 5), "eliminate periodic: pattern[0]")
        expectEqual(pattern[1], M31(v: 10), "eliminate periodic: pattern[1]")
    } else {
        expect(false, "eliminate periodic: col0 should be periodic")
    }
}

// MARK: - Compression Suggestion Tests

private func testCompressionSuggestion() {
    let n = 16

    // Build a trace with mixed column types
    let constCol = [M31](repeating: M31(v: 1), count: n)
    var periodicCol = [M31](repeating: M31.zero, count: n)
    for i in 0..<n { periodicCol[i] = M31(v: i % 2 == 0 ? 1 : 0) }
    var genCol0 = [M31](repeating: M31.zero, count: n)
    var genCol1 = [M31](repeating: M31.zero, count: n)
    for i in 0..<n {
        genCol0[i] = M31(v: UInt32(i * 3 + 1))
        genCol1[i] = M31(v: UInt32(i * 7 + 2))
    }

    let analysis = analyzeColumns(trace: [constCol, periodicCol, genCol0, genCol1])
    let plan = suggestCompression(analysis: analysis)

    expectEqual(plan.eliminateConstant.count, 1, "plan: 1 constant to eliminate")
    expectEqual(plan.eliminatePeriodic.count, 1, "plan: 1 periodic to eliminate")
    expect(plan.estimatedRatio <= 1.0, "plan: ratio <= 1.0")
}

// MARK: - Decompression Tests

private func testDecompressEvaluationIdentity() {
    let vals = [M31(v: 1), M31(v: 2), M31(v: 3)]
    let mapping = ColumnMapping(kind: .identity, originalColumnCount: 3, compressedColumnCount: 3)
    let result = decompressEvaluation(compressed: vals, point: M31.zero, mapping: mapping)
    expectEqual(result.count, 3, "decompress identity: count")
    for i in 0..<3 {
        expectEqual(result[i], vals[i], "decompress identity: val\(i)")
    }
}

private func testDecompressEvaluationAlgebraic() {
    // 4 original cols compressed to 2 via algebraic
    let groups = [
        ColumnMapping.AlgebraicGroup(originalIndices: [0, 1], randomness: M31(v: 7)),
        ColumnMapping.AlgebraicGroup(originalIndices: [2, 3], randomness: M31(v: 7)),
    ]
    let mapping = ColumnMapping(kind: .algebraic(groups),
                                originalColumnCount: 4, compressedColumnCount: 2)
    let compressed = [M31(v: 100), M31(v: 200)]
    let result = decompressEvaluation(compressed: compressed, point: M31.zero, mapping: mapping)

    expectEqual(result.count, 4, "decompress algebraic: 4 original cols")
    // Combined values stored at first index of each group
    expectEqual(result[0], M31(v: 100), "decompress algebraic: group0")
    expectEqual(result[2], M31(v: 200), "decompress algebraic: group1")
}

// MARK: - Edge Cases

private func testEmptyTrace() {
    let empty: [[M31]] = []

    // Algebraic on empty
    let a = compressTrace(trace: empty, strategy: .algebraic(randomness: M31(v: 1)))
    expectEqual(a.numColumns, 0, "empty algebraic: 0 cols")

    // Interleave on empty
    let b = compressTrace(trace: empty, strategy: .interleave)
    expectEqual(b.numColumns, 0, "empty interleave: 0 cols")

    // ZK on empty
    let c = compressTrace(trace: empty, strategy: .zeroKnowledge(blindingRows: 4))
    expectEqual(c.numColumns, 0, "empty zk: 0 cols")

    // Analyzer on empty
    let analysis = analyzeColumns(trace: empty)
    expectEqual(analysis.columns.count, 0, "empty analysis: 0 cols")
}
