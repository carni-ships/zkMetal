// LookupTable -- Prebuilt lookup tables and table combinators for zkMetal
//
// Provides commonly-used lookup tables (range checks, bitwise ops, byte decomposition)
// with compressed representations for large sparse tables.
//
// These tables can be used with any lookup strategy (LogUp, Lasso, cq) via the
// UnifiedLookup engine.

import Foundation

// MARK: - Prebuilt Table Definitions

/// A reusable lookup table with metadata for strategy selection.
/// Wraps raw field-element values with size, structure, and sparsity info
/// so UnifiedLookup can auto-select the best proof strategy.
public struct PrebuiltLookupTable {
    /// Human-readable name for debugging/profiling
    public let name: String
    /// Full table values as field elements (materialized from generator if needed)
    public let values: [Fr]
    /// Whether this table has tensor-decomposable structure (eligible for Lasso)
    public let isStructured: Bool
    /// If structured, the corresponding LassoTable definition
    public let lassoTable: LassoTable?
    /// Number of columns (1 for single-column, >1 for cross-product tables)
    public let numColumns: Int
    /// Original bit-width (for range/bitwise tables)
    public let bits: Int

    public var count: Int { values.count }

    public init(name: String, values: [Fr], isStructured: Bool = false,
                lassoTable: LassoTable? = nil, numColumns: Int = 1, bits: Int = 0) {
        self.name = name
        self.values = values
        self.isStructured = isStructured
        self.lassoTable = lassoTable
        self.numColumns = numColumns
        self.bits = bits
    }
}

// MARK: - Range Table

/// Build a range table [0, 1, ..., 2^bits - 1].
/// For bits <= 16, returns a flat table suitable for LogUp or cq.
/// For bits > 16, also returns a LassoTable for decomposed proving.
public func RangeTable(bits: Int) -> PrebuiltLookupTable {
    precondition(bits > 0 && bits <= 64, "Range table bits must be in 1...64")

    let size = 1 << bits
    let values: [Fr] = (0..<size).map { frFromInt(UInt64($0)) }

    if bits <= 16 {
        // Small enough for flat table -- no decomposition needed
        return PrebuiltLookupTable(
            name: "Range(\(bits))",
            values: values,
            isStructured: bits >= 8,
            lassoTable: bits >= 8 ? LassoTable.rangeCheck(bits: bits, chunks: bits / 8) : nil,
            bits: bits
        )
    } else {
        // Large range: always provide Lasso decomposition
        let chunks = (bits + 7) / 8  // ceil(bits/8) byte-level chunks
        let adjustedBits = chunks * 8
        let lasso = LassoTable.rangeCheck(bits: adjustedBits, chunks: chunks)
        return PrebuiltLookupTable(
            name: "Range(\(bits))",
            values: values,
            isStructured: true,
            lassoTable: lasso,
            bits: bits
        )
    }
}

// MARK: - XOR Table

/// Build a XOR table for `bits`-wide operands.
/// Table entry at index (a * 2^bits + b) = a XOR b, for a,b in [0, 2^bits).
/// Total size: 2^(2*bits).
///
/// For bits <= 8, returns a flat table. For bits > 8, provides Lasso decomposition.
public func XORTable(bits: Int) -> PrebuiltLookupTable {
    precondition(bits > 0 && bits <= 32, "XOR table bits must be in 1...32")
    precondition(bits % 2 == 0, "XOR table bits must be even")

    let range = 1 << bits
    let tableSize = range * range
    let values: [Fr] = (0..<tableSize).map { idx in
        let a = idx / range
        let b = idx % range
        return frFromInt(UInt64(a ^ b))
    }

    let lasso: LassoTable? = bits > 4 ? LassoTable.xor(bits: bits) : nil

    return PrebuiltLookupTable(
        name: "XOR(\(bits))",
        values: values,
        isStructured: bits > 4,
        lassoTable: lasso,
        numColumns: 2,
        bits: bits
    )
}

// MARK: - AND Table

/// Build an AND table for `bits`-wide operands.
/// Table entry at index (a * 2^bits + b) = a AND b.
/// Total size: 2^(2*bits).
public func ANDTable(bits: Int) -> PrebuiltLookupTable {
    precondition(bits > 0 && bits <= 32, "AND table bits must be in 1...32")
    precondition(bits % 2 == 0, "AND table bits must be even")

    let range = 1 << bits
    let tableSize = range * range
    let values: [Fr] = (0..<tableSize).map { idx in
        let a = idx / range
        let b = idx % range
        return frFromInt(UInt64(a & b))
    }

    let lasso: LassoTable? = bits > 4 ? LassoTable.and(bits: bits) : nil

    return PrebuiltLookupTable(
        name: "AND(\(bits))",
        values: values,
        isStructured: bits > 4,
        lassoTable: lasso,
        numColumns: 2,
        bits: bits
    )
}

// MARK: - Byte Decomposition Table

/// Build a byte decomposition table: values [0, 1, ..., 255].
/// Used for proving that a value decomposes into valid bytes.
/// This is the building block for Lasso-style range checks.
public let ByteDecompTable: PrebuiltLookupTable = {
    let values: [Fr] = (0..<256).map { frFromInt(UInt64($0)) }
    return PrebuiltLookupTable(
        name: "ByteDecomp",
        values: values,
        isStructured: true,
        lassoTable: LassoTable.rangeCheck(bits: 8, chunks: 1),
        bits: 8
    )
}()

// MARK: - Table Combination (Cross-Product)

/// Combine two single-column tables into a multi-column cross-product table.
///
/// Given tables A = [a0, a1, ...] and B = [b0, b1, ...], produces a table
/// with entries [a_i * shift + b_j] for all (i,j), where shift = |B|.
/// This encodes the pair (a_i, b_j) as a single field element.
///
/// Useful for multi-column lookups: e.g., proving (x, y) in Table_A x Table_B.
///
/// - Parameters:
///   - tableA: First table
///   - tableB: Second table
///   - name: Optional name for the combined table
/// - Returns: Cross-product table of size |A| * |B|
public func crossProductTable(_ tableA: PrebuiltLookupTable,
                               _ tableB: PrebuiltLookupTable,
                               name: String? = nil) -> PrebuiltLookupTable {
    let sizeA = tableA.count
    let sizeB = tableB.count
    let shift = frFromInt(UInt64(sizeB))

    var values = [Fr]()
    values.reserveCapacity(sizeA * sizeB)
    for i in 0..<sizeA {
        let aShifted = frMul(tableA.values[i], shift)
        for j in 0..<sizeB {
            values.append(frAdd(aShifted, tableB.values[j]))
        }
    }

    let combinedName = name ?? "\(tableA.name) x \(tableB.name)"
    return PrebuiltLookupTable(
        name: combinedName,
        values: values,
        isStructured: false,
        lassoTable: nil,
        numColumns: tableA.numColumns + tableB.numColumns,
        bits: tableA.bits + tableB.bits
    )
}

// MARK: - Compressed Sparse Table

/// A compressed representation for large sparse lookup tables.
///
/// Instead of materializing the full table (which might have 2^32 entries),
/// stores only the entries that are actually present. The full table is
/// reconstructed on demand or used directly with cq (which only needs
/// the entries with non-zero multiplicity).
public struct CompressedLookupTable {
    /// The non-default entries: maps index -> value
    public let entries: [(index: UInt64, value: Fr)]
    /// The default value for entries not in `entries` (typically zero)
    public let defaultValue: Fr
    /// Logical size of the full table (may be much larger than entries.count)
    public let logicalSize: Int
    /// Name for debugging
    public let name: String

    public init(name: String, entries: [(index: UInt64, value: Fr)],
                defaultValue: Fr = Fr.zero, logicalSize: Int) {
        self.name = name
        self.entries = entries
        self.defaultValue = defaultValue
        self.logicalSize = logicalSize
    }

    /// Density: fraction of non-default entries
    public var density: Double {
        Double(entries.count) / Double(logicalSize)
    }

    /// Materialize into a full flat table (for LogUp/Lasso).
    /// Warning: only call this if logicalSize is manageable (e.g., <= 2^20).
    public func materialize() -> [Fr] {
        precondition(logicalSize <= (1 << 24),
                     "Table too large to materialize: \(logicalSize) entries")
        var table = [Fr](repeating: defaultValue, count: logicalSize)
        for (idx, val) in entries {
            table[Int(idx)] = val
        }
        return table
    }

    /// Convert to a PrebuiltLookupTable by materializing.
    public func toPrebuilt() -> PrebuiltLookupTable {
        let values = materialize()
        return PrebuiltLookupTable(
            name: name,
            values: values,
            isStructured: false,
            lassoTable: nil,
            bits: Int(log2(Double(logicalSize)))
        )
    }

    /// Extract only the entries relevant to a set of lookups.
    /// Returns a minimal flat table containing exactly the looked-up values
    /// plus padding to the next power of 2.
    ///
    /// This is useful for cq, where the prover cost depends on the number
    /// of distinct looked-up values, not the full table size.
    public func extractRelevant(lookups: [Fr]) -> [Fr] {
        var seen = Set<FrKey>()
        var relevant = [Fr]()
        for v in lookups {
            let key = FrKey(v)
            if !seen.contains(key) {
                seen.insert(key)
                relevant.append(v)
            }
        }
        // Pad to power of 2
        var padded = relevant.count
        var p = 1
        while p < padded { p <<= 1 }
        while relevant.count < p {
            relevant.append(relevant.last ?? Fr.zero)
        }
        return relevant
    }
}

// MARK: - Convenience Builders

/// Build a custom single-column table from a closure.
/// Generates values by calling `generator(i)` for i in 0..<size.
public func customTable(name: String, size: Int, generator: (Int) -> Fr) -> PrebuiltLookupTable {
    precondition(size > 0, "Table size must be positive")
    let values = (0..<size).map { generator($0) }
    return PrebuiltLookupTable(
        name: name,
        values: values,
        isStructured: false,
        lassoTable: nil,
        bits: Int(log2(Double(size)))
    )
}

/// Build a table from an explicit list of UInt64 values.
public func tableFromValues(name: String, _ vals: [UInt64]) -> PrebuiltLookupTable {
    var values = vals.map { frFromInt($0) }
    // Pad to power of 2
    var p = 1
    while p < values.count { p <<= 1 }
    while values.count < p {
        values.append(values.last ?? Fr.zero)
    }
    return PrebuiltLookupTable(
        name: name,
        values: values,
        isStructured: false,
        lassoTable: nil,
        bits: Int(log2(Double(p)))
    )
}
