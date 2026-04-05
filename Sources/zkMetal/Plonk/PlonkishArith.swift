// PlonkishArith — Plonkish arithmetization engine
//
// Converts high-level circuit descriptions into Plonk constraint systems with custom gates.
// Provides a Halo2-style API with:
//   - Typed columns (advice, fixed, instance)
//   - Region-based cell assignment via a layouter
//   - Arithmetic, lookup, and custom gate support
//   - WitnessAssignment: maps columns to evaluation vectors
//   - ConstraintSystem: collects gates, permutation, lookup arguments
//   - compile() -> produces PlonkCircuit + witness polynomials

import Foundation
import NeonFieldOps

// MARK: - Column Reference

/// A column reference in the Plonkish constraint system.
/// Wraps the existing Column type with additional metadata for the arithmetization.
public struct PlonkishColumn: Equatable, Hashable {
    public let column: Column
    /// Column index within its type (advice/fixed/instance)
    public var index: Int { column.index }
    public var type: ColumnType { column.type }

    public init(_ column: Column) {
        self.column = column
    }

    public static func advice(_ idx: Int) -> PlonkishColumn {
        PlonkishColumn(Column.advice(idx))
    }

    public static func fixed(_ idx: Int) -> PlonkishColumn {
        PlonkishColumn(Column.fixed(idx))
    }

    public static func instance(_ idx: Int) -> PlonkishColumn {
        PlonkishColumn(Column.instance(idx))
    }
}

// MARK: - Cell Reference

/// A cell in the Plonkish trace, addressed by column + absolute row.
public struct PlonkishCell: Equatable, Hashable {
    public let column: PlonkishColumn
    public let row: Int

    public init(column: PlonkishColumn, row: Int) {
        self.column = column
        self.row = row
    }
}

// MARK: - Assigned Cell

/// A cell that has been assigned a value during region layout.
/// Carries the cell coordinate and the assigned value for later constraint checking.
public struct PlonkishAssignedCell {
    public let cell: PlonkishCell
    public let value: Fr

    public init(cell: PlonkishCell, value: Fr) {
        self.cell = cell
        self.value = value
    }
}

// MARK: - Region

/// A region in the Plonkish circuit, similar to Halo2's Region.
/// Regions provide relative addressing: offsets within the region map to absolute rows.
/// The layouter assigns a starting row for each region.
public class PlonkishRegion {
    /// Starting absolute row of this region in the trace.
    public let startRow: Int
    /// Reference back to the circuit for recording assignments.
    private weak var circuit: PlonkishCircuit?
    /// Number of rows used so far in this region.
    public private(set) var rowsUsed: Int = 0

    init(startRow: Int, circuit: PlonkishCircuit) {
        self.startRow = startRow
        self.circuit = circuit
    }

    /// Assign a value to an advice cell at (column, offset) within this region.
    @discardableResult
    public func assignAdvice(column: Int, offset: Int, value: Fr) -> PlonkishAssignedCell {
        let col = PlonkishColumn.advice(column)
        let absRow = startRow + offset
        rowsUsed = max(rowsUsed, offset + 1)
        circuit?.assignCell(column: col, row: absRow, value: value)
        return PlonkishAssignedCell(cell: PlonkishCell(column: col, row: absRow), value: value)
    }

    /// Assign a value to a fixed cell at (column, offset) within this region.
    @discardableResult
    public func assignFixed(column: Int, offset: Int, value: Fr) -> PlonkishAssignedCell {
        let col = PlonkishColumn.fixed(column)
        let absRow = startRow + offset
        rowsUsed = max(rowsUsed, offset + 1)
        circuit?.assignCell(column: col, row: absRow, value: value)
        return PlonkishAssignedCell(cell: PlonkishCell(column: col, row: absRow), value: value)
    }

    /// Constrain two cells to be equal (copy constraint / permutation argument).
    public func constrainEqual(_ a: PlonkishAssignedCell, _ b: PlonkishAssignedCell) {
        circuit?.addCopyConstraint(a.cell, b.cell)
    }

    /// Assign an instance value and constrain it equal to an advice cell.
    @discardableResult
    public func assignInstance(column: Int, offset: Int, value: Fr,
                               constrainTo: PlonkishAssignedCell? = nil) -> PlonkishAssignedCell {
        let col = PlonkishColumn.instance(column)
        let absRow = startRow + offset
        rowsUsed = max(rowsUsed, offset + 1)
        circuit?.assignCell(column: col, row: absRow, value: value)
        let assigned = PlonkishAssignedCell(cell: PlonkishCell(column: col, row: absRow), value: value)
        if let target = constrainTo {
            constrainEqual(assigned, target)
        }
        return assigned
    }
}

// MARK: - Gate Definition

/// A named gate constraint in the Plonkish system.
/// Each gate defines a polynomial expression over column evaluations that must vanish.
public struct PlonkishGateDefinition {
    public let name: String
    /// The constraint evaluation closure.
    /// Takes a row evaluator and returns the constraint value (must be zero for valid witness).
    public let evaluate: (PlonkishRowEvaluator) -> Fr

    public init(name: String, evaluate: @escaping (PlonkishRowEvaluator) -> Fr) {
        self.name = name
        self.evaluate = evaluate
    }
}

// MARK: - Row Evaluator

/// Provides access to column values at a given row (with rotation support).
/// Used by gate definitions to evaluate constraints.
public struct PlonkishRowEvaluator {
    private let adviceCols: [[Fr]]
    private let fixedCols: [[Fr]]
    private let instanceCols: [[Fr]]
    private let row: Int
    private let numRows: Int

    init(adviceCols: [[Fr]], fixedCols: [[Fr]], instanceCols: [[Fr]],
         row: Int, numRows: Int) {
        self.adviceCols = adviceCols
        self.fixedCols = fixedCols
        self.instanceCols = instanceCols
        self.row = row
        self.numRows = numRows
    }

    /// Query an advice column at the current row + rotation.
    public func queryAdvice(column: Int, rotation: Int = 0) -> Fr {
        let r = (row + rotation + numRows) % numRows
        guard column < adviceCols.count, r < adviceCols[column].count else { return Fr.zero }
        return adviceCols[column][r]
    }

    /// Query a fixed column at the current row + rotation.
    public func queryFixed(column: Int, rotation: Int = 0) -> Fr {
        let r = (row + rotation + numRows) % numRows
        guard column < fixedCols.count, r < fixedCols[column].count else { return Fr.zero }
        return fixedCols[column][r]
    }

    /// Query an instance column at the current row + rotation.
    public func queryInstance(column: Int, rotation: Int = 0) -> Fr {
        let r = (row + rotation + numRows) % numRows
        guard column < instanceCols.count, r < instanceCols[column].count else { return Fr.zero }
        return instanceCols[column][r]
    }
}

// MARK: - Lookup Argument Definition

/// A lookup argument: the input expressions must be contained in the table expressions.
public struct PlonkishLookupDefinition {
    public let name: String
    /// Evaluates the input expression at a given row.
    public let inputExpr: (PlonkishRowEvaluator) -> Fr
    /// Evaluates the table expression at a given row.
    public let tableExpr: (PlonkishRowEvaluator) -> Fr

    public init(name: String,
                inputExpr: @escaping (PlonkishRowEvaluator) -> Fr,
                tableExpr: @escaping (PlonkishRowEvaluator) -> Fr) {
        self.name = name
        self.inputExpr = inputExpr
        self.tableExpr = tableExpr
    }
}

// MARK: - Constraint System

/// Collects gates, permutation columns, and lookup arguments.
/// Analogous to Halo2's ConstraintSystem<F>.
public class PlonkishConstraintSystem {
    /// Number of advice columns
    public private(set) var numAdviceCols: Int = 0
    /// Number of fixed columns
    public private(set) var numFixedCols: Int = 0
    /// Number of instance columns
    public private(set) var numInstanceCols: Int = 0

    /// Gate definitions
    public private(set) var gates: [PlonkishGateDefinition] = []
    /// Lookup argument definitions
    public private(set) var lookups: [PlonkishLookupDefinition] = []
    /// Columns enabled for equality (permutation argument)
    public private(set) var equalityColumns: [PlonkishColumn] = []
    /// Custom gate templates
    public private(set) var customGateTemplates: [(any CustomGateTemplate, [PlonkishColumn])] = []

    public init() {}

    /// Allocate advice columns.
    public func adviceColumn() -> PlonkishColumn {
        let col = PlonkishColumn.advice(numAdviceCols)
        numAdviceCols += 1
        return col
    }

    /// Allocate a fixed column.
    public func fixedColumn() -> PlonkishColumn {
        let col = PlonkishColumn.fixed(numFixedCols)
        numFixedCols += 1
        return col
    }

    /// Allocate an instance (public input) column.
    public func instanceColumn() -> PlonkishColumn {
        let col = PlonkishColumn.instance(numInstanceCols)
        numInstanceCols += 1
        return col
    }

    /// Enable equality on a column (required for copy constraints).
    public func enableEquality(_ col: PlonkishColumn) {
        if !equalityColumns.contains(col) {
            equalityColumns.append(col)
        }
    }

    /// Register a gate constraint.
    public func createGate(name: String, evaluate: @escaping (PlonkishRowEvaluator) -> Fr) {
        gates.append(PlonkishGateDefinition(name: name, evaluate: evaluate))
    }

    /// Register a lookup argument.
    public func createLookup(name: String,
                             inputExpr: @escaping (PlonkishRowEvaluator) -> Fr,
                             tableExpr: @escaping (PlonkishRowEvaluator) -> Fr) {
        lookups.append(PlonkishLookupDefinition(
            name: name, inputExpr: inputExpr, tableExpr: tableExpr))
    }

    /// Register a custom gate template on specific columns.
    public func addCustomGate(_ template: any CustomGateTemplate, columns: [PlonkishColumn]) {
        customGateTemplates.append((template, columns))
    }
}

// MARK: - Witness Assignment

/// Maps columns to their evaluation vectors (the execution trace).
public struct WitnessAssignment {
    /// advice[colIdx][row] = value
    public var advice: [[Fr]]
    /// fixed[colIdx][row] = value
    public var fixed: [[Fr]]
    /// instance[colIdx][row] = value
    public var instance: [[Fr]]
    /// Number of rows in the trace
    public let numRows: Int

    public init(numAdvice: Int, numFixed: Int, numInstance: Int, numRows: Int) {
        self.numRows = numRows
        self.advice = (0..<numAdvice).map { _ in [Fr](repeating: Fr.zero, count: numRows) }
        self.fixed = (0..<numFixed).map { _ in [Fr](repeating: Fr.zero, count: numRows) }
        self.instance = (0..<numInstance).map { _ in [Fr](repeating: Fr.zero, count: numRows) }
    }

    /// Set a cell value.
    public mutating func set(column: PlonkishColumn, row: Int, value: Fr) {
        switch column.type {
        case .advice:
            if column.index < advice.count && row < numRows {
                advice[column.index][row] = value
            }
        case .fixed:
            if column.index < fixed.count && row < numRows {
                fixed[column.index][row] = value
            }
        case .instance:
            if column.index < instance.count && row < numRows {
                instance[column.index][row] = value
            }
        }
    }

    /// Get a cell value.
    public func get(column: PlonkishColumn, row: Int) -> Fr {
        switch column.type {
        case .advice:
            guard column.index < advice.count, row < numRows else { return Fr.zero }
            return advice[column.index][row]
        case .fixed:
            guard column.index < fixed.count, row < numRows else { return Fr.zero }
            return fixed[column.index][row]
        case .instance:
            guard column.index < instance.count, row < numRows else { return Fr.zero }
            return instance[column.index][row]
        }
    }
}

// MARK: - Compiled Result

/// Result of compiling a PlonkishCircuit into the low-level PlonkCircuit representation.
public struct PlonkishCompilationResult {
    /// The low-level Plonk circuit (gates + wires + copy constraints)
    public let circuit: PlonkCircuit
    /// Witness polynomial evaluations: flat array of Fr values for all wires
    public let witnessPolynomials: [[Fr]]
    /// The witness assignment (column-oriented view)
    public let witness: WitnessAssignment
    /// Number of rows used
    public let numRows: Int
}

// MARK: - Plonkish Circuit

/// High-level Plonkish circuit builder with region-based assignment.
/// Supports advice/fixed/instance columns, custom gates, and lookup arguments.
///
/// Usage:
///   1. Configure the constraint system (allocate columns, define gates)
///   2. Lay out regions (assign witness values)
///   3. compile() to produce a PlonkCircuit + witness
public class PlonkishCircuit {
    /// The constraint system definition.
    public let cs: PlonkishConstraintSystem
    /// Current next row for region allocation.
    private var nextRow: Int = 0
    /// All cell assignments: (column, row) -> value
    private var cellValues: [PlonkishCell: Fr] = [:]
    /// Copy constraints: pairs of cells that must be equal.
    private var copyConstraints: [(PlonkishCell, PlonkishCell)] = []
    /// Regions allocated so far.
    private var regions: [PlonkishRegion] = []
    /// Lookup table values for the lookup argument.
    private var lookupTableValues: [[Fr]] = []
    /// Fixed column values set outside regions.
    private var fixedValues: [PlonkishCell: Fr] = [:]

    public init(cs: PlonkishConstraintSystem) {
        self.cs = cs
    }

    // MARK: - Cell Assignment (internal)

    func assignCell(column: PlonkishColumn, row: Int, value: Fr) {
        let cell = PlonkishCell(column: column, row: row)
        cellValues[cell] = value
        if column.type == .fixed {
            fixedValues[cell] = value
        }
    }

    func addCopyConstraint(_ a: PlonkishCell, _ b: PlonkishCell) {
        copyConstraints.append((a, b))
    }

    // MARK: - Region Layout

    /// Allocate a new region and execute the layout closure.
    /// Returns whatever the closure returns (typically an array of AssignedCell).
    @discardableResult
    public func layoutRegion<T>(name: String = "",
                                numRows: Int = 0,
                                _ layout: (PlonkishRegion) -> T) -> T {
        let region = PlonkishRegion(startRow: nextRow, circuit: self)
        regions.append(region)
        let result = layout(region)
        // Advance nextRow by the number of rows actually used
        let used = max(region.rowsUsed, numRows)
        nextRow += max(used, 1)
        return result
    }

    // MARK: - Fixed Column Setup

    /// Assign a fixed column value at an absolute row (outside any region).
    public func assignFixedGlobal(column: Int, row: Int, value: Fr) {
        let col = PlonkishColumn.fixed(column)
        assignCell(column: col, row: row, value: value)
    }

    // MARK: - Lookup Table

    /// Register a lookup table. Returns the table index.
    @discardableResult
    public func addLookupTable(values: [Fr]) -> Int {
        let idx = lookupTableValues.count
        lookupTableValues.append(values)
        return idx
    }

    // MARK: - Compile

    /// Compile the Plonkish circuit into a low-level PlonkCircuit and witness.
    ///
    /// Produces:
    ///   - PlonkGate array with selector polynomials derived from fixed columns + gate definitions
    ///   - Wire assignments mapping to a flat variable namespace
    ///   - Copy constraints translated to variable-pair form
    ///   - Witness polynomials for all advice columns
    public func compile() -> PlonkishCompilationResult {
        // Determine trace size (next power of 2, min 4)
        var numRows = max(nextRow, 4)
        var logN = 2
        while (1 << logN) < numRows { logN += 1 }
        numRows = 1 << logN

        // Build witness assignment
        var witness = WitnessAssignment(
            numAdvice: cs.numAdviceCols,
            numFixed: cs.numFixedCols,
            numInstance: cs.numInstanceCols,
            numRows: numRows)

        for (cell, value) in cellValues {
            if cell.row < numRows {
                witness.set(column: cell.column, row: cell.row, value: value)
            }
        }

        // Map (column, row) -> variable index for wire assignments
        // Variable layout: advice cols first, then fixed, then instance
        // Variable index = colOffset + row
        let adviceOffset = 0
        let fixedOffset = cs.numAdviceCols * numRows
        let instanceOffset = fixedOffset + cs.numFixedCols * numRows

        func varIndex(col: PlonkishColumn, row: Int) -> Int {
            switch col.type {
            case .advice: return adviceOffset + col.index * numRows + row
            case .fixed: return fixedOffset + col.index * numRows + row
            case .instance: return instanceOffset + col.index * numRows + row
            }
        }

        // Build PlonkGates: one gate per row
        // The standard Plonkish arithmetic gate:
        //   qL * a + qR * b + qO * c + qM * a*b + qC = 0
        // Selectors come from fixed columns if available, otherwise from gate evaluation.

        var plonkGates: [PlonkGate] = []
        var wireAssignments: [[Int]] = []

        // Determine wire columns: use first 3 advice columns (a, b, c)
        // If fewer than 3 advice columns, pad with zeros
        let aCol = cs.numAdviceCols > 0 ? 0 : -1
        let bCol = cs.numAdviceCols > 1 ? 1 : -1
        let cCol = cs.numAdviceCols > 2 ? 2 : -1

        // Build evaluator for each row to check gate constraints
        let totalVars = (cs.numAdviceCols + cs.numFixedCols + cs.numInstanceCols) * numRows

        for row in 0..<numRows {
            // Default selector values from fixed columns
            // Convention: fixed col 0 = qL, 1 = qR, 2 = qO, 3 = qM, 4 = qC
            let qL = cs.numFixedCols > 0 ? witness.get(column: .fixed(0), row: row) : Fr.zero
            let qR = cs.numFixedCols > 1 ? witness.get(column: .fixed(1), row: row) : Fr.zero
            let qO = cs.numFixedCols > 2 ? witness.get(column: .fixed(2), row: row) : Fr.zero
            let qM = cs.numFixedCols > 3 ? witness.get(column: .fixed(3), row: row) : Fr.zero
            let qC = cs.numFixedCols > 4 ? witness.get(column: .fixed(4), row: row) : Fr.zero

            let gate = PlonkGate(qL: qL, qR: qR, qO: qO, qM: qM, qC: qC)
            plonkGates.append(gate)

            // Wire assignments: a, b, c from advice columns 0, 1, 2
            let aVar = aCol >= 0 ? varIndex(col: .advice(aCol), row: row) : row
            let bVar = bCol >= 0 ? varIndex(col: .advice(bCol), row: row) : row
            let cVar = cCol >= 0 ? varIndex(col: .advice(cCol), row: row) : row
            wireAssignments.append([aVar, bVar, cVar])
        }

        // Translate copy constraints to variable-index pairs
        var plonkCopyConstraints: [(Int, Int)] = []
        for (cellA, cellB) in copyConstraints {
            let vA = varIndex(col: cellA.column, row: cellA.row)
            let vB = varIndex(col: cellB.column, row: cellB.row)
            plonkCopyConstraints.append((vA, vB))
        }

        // Translate lookup tables
        var plonkLookupTables: [PlonkLookupTable] = []
        for (i, values) in lookupTableValues.enumerated() {
            plonkLookupTables.append(PlonkLookupTable(id: i, values: values))
        }

        // Build flat witness: variable index -> value
        var flatWitness = [Fr](repeating: Fr.zero, count: totalVars)
        for colIdx in 0..<cs.numAdviceCols {
            for row in 0..<numRows {
                flatWitness[adviceOffset + colIdx * numRows + row] = witness.advice[colIdx][row]
            }
        }
        for colIdx in 0..<cs.numFixedCols {
            for row in 0..<numRows {
                flatWitness[fixedOffset + colIdx * numRows + row] = witness.fixed[colIdx][row]
            }
        }
        for colIdx in 0..<cs.numInstanceCols {
            for row in 0..<numRows {
                flatWitness[instanceOffset + colIdx * numRows + row] = witness.instance[colIdx][row]
            }
        }

        // Instance columns as public inputs
        var publicInputIndices: [Int] = []
        for colIdx in 0..<cs.numInstanceCols {
            for row in 0..<numRows {
                let val = witness.instance[colIdx][row]
                if !frEqual(val, Fr.zero) {
                    publicInputIndices.append(varIndex(col: .instance(colIdx), row: row))
                }
            }
        }

        // Build witness polynomials (one per advice column)
        let witnessPolynomials = witness.advice

        let plonkCircuit = PlonkCircuit(
            gates: plonkGates,
            copyConstraints: plonkCopyConstraints,
            wireAssignments: wireAssignments,
            lookupTables: plonkLookupTables,
            publicInputIndices: publicInputIndices
        )

        return PlonkishCompilationResult(
            circuit: plonkCircuit,
            witnessPolynomials: witnessPolynomials,
            witness: witness,
            numRows: numRows
        )
    }

    // MARK: - Constraint Verification

    /// Verify that all gate constraints are satisfied by the current witness.
    /// Returns true if every gate evaluates to zero at every used row.
    public func verify() -> Bool {
        let numRows = max(nextRow, 4)

        // Build witness
        var witness = WitnessAssignment(
            numAdvice: cs.numAdviceCols,
            numFixed: cs.numFixedCols,
            numInstance: cs.numInstanceCols,
            numRows: numRows)

        for (cell, value) in cellValues {
            if cell.row < numRows {
                witness.set(column: cell.column, row: cell.row, value: value)
            }
        }

        // Check each gate at each used row
        for row in 0..<nextRow {
            let evaluator = PlonkishRowEvaluator(
                adviceCols: witness.advice,
                fixedCols: witness.fixed,
                instanceCols: witness.instance,
                row: row,
                numRows: numRows)

            for gate in cs.gates {
                let result = gate.evaluate(evaluator)
                if !frEqual(result, Fr.zero) {
                    return false
                }
            }
        }

        // Check copy constraints
        for (cellA, cellB) in copyConstraints {
            let valA = witness.get(column: cellA.column, row: cellA.row)
            let valB = witness.get(column: cellB.column, row: cellB.row)
            if !frEqual(valA, valB) {
                return false
            }
        }

        return true
    }

    /// Number of rows used so far.
    public var usedRows: Int { nextRow }
}
