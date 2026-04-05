// Halo2Columns — Column types and cell references for Halo2 permutation argument
//
// Provides the column/cell abstraction layer used by Halo2PermutationAssembly.
// These types model Halo2's column system where cells are addressed by (column, row)
// and columns have types (advice, fixed, instance) that determine their role in
// the constraint system.
//
// Re-exports and extends the Halo2Column/Halo2ColumnType types from Halo2Circuit.swift
// with additional types needed specifically for the permutation argument:
//   - ColumnType: lightweight enum for permutation column classification
//   - Column: typed column with index
//   - Cell: (column, row) coordinate in the execution trace
//   - Halo2Region extensions for permutation-aware cell assignment

import Foundation
import NeonFieldOps

// MARK: - Column Type (Permutation-specific)

/// Lightweight column type enum for the permutation argument.
/// Mirrors Halo2's Column<T> type parameter.
public enum ColumnType: Equatable, Hashable, Sendable {
    /// Witness column — prover-supplied values, not public.
    case advice
    /// Preprocessed column — fixed at key generation time.
    case fixed
    /// Public input column — known to both prover and verifier.
    case instance
}

// MARK: - Column

/// A typed column reference for the permutation argument.
/// Combines a column type with a zero-based index within that type.
public struct Column: Equatable, Hashable, Sendable {
    /// The type of this column (advice, fixed, or instance).
    public let type: ColumnType
    /// Zero-based index within columns of this type.
    public let index: Int

    public init(type: ColumnType, index: Int) {
        self.type = type
        self.index = index
    }

    /// Convenience constructors matching Halo2's Column<Advice>, etc.
    public static func advice(_ index: Int) -> Column {
        Column(type: .advice, index: index)
    }

    public static func fixed(_ index: Int) -> Column {
        Column(type: .fixed, index: index)
    }

    public static func instance(_ index: Int) -> Column {
        Column(type: .instance, index: index)
    }
}

// MARK: - Cell

/// A cell in the execution trace, identified by its column and row.
/// This is the fundamental unit of the permutation argument: equality constraints
/// operate on pairs of cells.
public struct Cell: Equatable, Hashable, Sendable {
    /// The column this cell belongs to.
    public let column: Column
    /// The row index in the execution trace.
    public let row: Int

    public init(column: Column, row: Int) {
        self.column = column
        self.row = row
    }

    /// Convenience initializer from raw (col, row) indices.
    /// Defaults to advice column type.
    public init(col: Int, row: Int) {
        self.column = .advice(col)
        self.row = row
    }
}

// MARK: - Permutation Column Set

/// Tracks which columns participate in the permutation argument.
/// In Halo2, only columns with `enable_equality()` called on them can
/// have copy constraints. This type maintains the ordered set of such columns
/// and provides a flat index mapping used by the permutation polynomial.
public struct PermutationColumns {
    /// Ordered list of columns participating in the permutation.
    /// The order determines the flat index: column i occupies positions [i*n, (i+1)*n).
    public private(set) var columns: [Column] = []

    /// Fast lookup from column to its position in the ordered list.
    private var columnIndex: [Column: Int] = [:]

    public init() {}

    /// Add a column to the permutation. No-op if already present.
    @discardableResult
    public mutating func add(_ column: Column) -> Int {
        if let idx = columnIndex[column] {
            return idx
        }
        let idx = columns.count
        columns.append(column)
        columnIndex[column] = idx
        return idx
    }

    /// Look up the flat index of a column, or nil if not in the permutation.
    public func index(of column: Column) -> Int? {
        return columnIndex[column]
    }

    /// Number of columns in the permutation.
    public var count: Int { columns.count }

    /// Convert a cell to a flat position index: flatCol * n + row.
    /// Returns nil if the cell's column is not in the permutation.
    public func flatIndex(cell: Cell, domainSize n: Int) -> Int? {
        guard let colIdx = columnIndex[cell.column] else { return nil }
        return colIdx * n + cell.row
    }
}

// MARK: - Halo2 Region Cell Tracking

/// Extension point for region-based cell tracking in the permutation argument.
/// When a region assigns a cell, it records the assignment here so that
/// the permutation assembly can later resolve equality constraints.
public struct RegionCellTracker {
    /// All cells assigned in this region, keyed by (column, offset).
    public var assignedCells: [Cell: Fr] = [:]

    public init() {}

    /// Record a cell assignment.
    public mutating func record(cell: Cell, value: Fr) {
        assignedCells[cell] = value
    }

    /// Look up the value assigned to a cell, if any.
    public func value(at cell: Cell) -> Fr? {
        return assignedCells[cell]
    }
}
