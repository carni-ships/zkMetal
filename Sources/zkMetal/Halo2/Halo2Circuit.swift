// Halo2Circuit — Halo2-compatible circuit API for zkMetal
//
// Provides a faithful translation of Halo2's ConstraintSystem, Expression,
// Region, and Circuit abstractions so that existing Halo2 circuit definitions
// can target zkMetal's GPU-accelerated Plonk backend.
//
// Mapping to Halo2 Rust types:
//   ConstraintSystem<F>  -> Halo2ConstraintSystem
//   Expression<F>        -> Halo2Expression
//   Region<F>            -> Halo2Region
//   Circuit<F>           -> Halo2Circuit protocol
//   Column<Advice>       -> Halo2Column(.advice, index)
//   Column<Fixed>        -> Halo2Column(.fixed, index)
//   Column<Instance>     -> Halo2Column(.instance, index)
//   Selector             -> Halo2Selector

import Foundation
import NeonFieldOps

// MARK: - Column Types

/// Column type matching Halo2's Advice / Fixed / Instance distinction.
public enum Halo2ColumnType: Equatable, Hashable, Sendable {
    case advice
    case fixed
    case instance
}

/// A typed column reference, analogous to halo2::plonk::Column<T>.
public struct Halo2Column: Equatable, Hashable, Sendable {
    public let columnType: Halo2ColumnType
    public let index: Int

    public init(_ columnType: Halo2ColumnType, _ index: Int) {
        self.columnType = columnType
        self.index = index
    }
}

// MARK: - Selector

/// Simple selector, analogous to halo2::plonk::Selector.
/// Selectors activate gates at specific rows in the execution trace.
public struct Halo2Selector: Equatable, Hashable, Sendable {
    public let index: Int

    public init(_ index: Int) {
        self.index = index
    }
}

// MARK: - Expression

/// Symbolic expression type matching Halo2's Expression<F>.
///
/// An expression tree that represents polynomial constraints over columns
/// with rotations. The prover evaluates these symbolically over the domain
/// to build the quotient polynomial.
public indirect enum Halo2Expression {
    /// A field constant.
    case constant(Fr)
    /// A selector query — evaluates to 0 or 1 at each row.
    case selector(Halo2Selector)
    /// Query a fixed column at a rotation relative to the current row.
    case fixed(Halo2Column, Rotation)
    /// Query an advice column at a rotation relative to the current row.
    case advice(Halo2Column, Rotation)
    /// Query an instance column at a rotation relative to the current row.
    case instance(Halo2Column, Rotation)
    /// Negation: -expr
    case negated(Halo2Expression)
    /// Addition: lhs + rhs
    case sum(Halo2Expression, Halo2Expression)
    /// Multiplication: lhs * rhs
    case product(Halo2Expression, Halo2Expression)
    /// Scalar multiplication: scalar * expr
    case scaled(Halo2Expression, Fr)
}

// MARK: - Expression arithmetic operators

extension Halo2Expression {
    public static func + (lhs: Halo2Expression, rhs: Halo2Expression) -> Halo2Expression {
        .sum(lhs, rhs)
    }

    public static func * (lhs: Halo2Expression, rhs: Halo2Expression) -> Halo2Expression {
        .product(lhs, rhs)
    }

    public static prefix func - (expr: Halo2Expression) -> Halo2Expression {
        .negated(expr)
    }

    public static func - (lhs: Halo2Expression, rhs: Halo2Expression) -> Halo2Expression {
        .sum(lhs, .negated(rhs))
    }

    /// Evaluate the expression given concrete values for all column queries.
    ///
    /// - Parameters:
    ///   - selectorValues: Maps selector index -> Fr value at the current row.
    ///   - fixedValues: Maps (column index, rotation offset) -> Fr value.
    ///   - adviceValues: Maps (column index, rotation offset) -> Fr value.
    ///   - instanceValues: Maps (column index, rotation offset) -> Fr value.
    public func evaluate(
        selectorValues: [Int: Fr],
        fixedValues: [Int: [Int: Fr]],
        adviceValues: [Int: [Int: Fr]],
        instanceValues: [Int: [Int: Fr]]
    ) -> Fr {
        switch self {
        case .constant(let c):
            return c
        case .selector(let sel):
            return selectorValues[sel.index] ?? Fr.zero
        case .fixed(let col, let rot):
            return fixedValues[col.index]?[rot.value] ?? Fr.zero
        case .advice(let col, let rot):
            return adviceValues[col.index]?[rot.value] ?? Fr.zero
        case .instance(let col, let rot):
            return instanceValues[col.index]?[rot.value] ?? Fr.zero
        case .negated(let e):
            return frSub(Fr.zero, e.evaluate(selectorValues: selectorValues,
                                             fixedValues: fixedValues,
                                             adviceValues: adviceValues,
                                             instanceValues: instanceValues))
        case .sum(let a, let b):
            return frAdd(a.evaluate(selectorValues: selectorValues,
                                    fixedValues: fixedValues,
                                    adviceValues: adviceValues,
                                    instanceValues: instanceValues),
                         b.evaluate(selectorValues: selectorValues,
                                    fixedValues: fixedValues,
                                    adviceValues: adviceValues,
                                    instanceValues: instanceValues))
        case .product(let a, let b):
            return frMul(a.evaluate(selectorValues: selectorValues,
                                    fixedValues: fixedValues,
                                    adviceValues: adviceValues,
                                    instanceValues: instanceValues),
                         b.evaluate(selectorValues: selectorValues,
                                    fixedValues: fixedValues,
                                    adviceValues: adviceValues,
                                    instanceValues: instanceValues))
        case .scaled(let e, let s):
            return frMul(s, e.evaluate(selectorValues: selectorValues,
                                       fixedValues: fixedValues,
                                       adviceValues: adviceValues,
                                       instanceValues: instanceValues))
        }
    }
}

// MARK: - Gate Definition

/// A named gate with its polynomial constraint expressions.
/// Mirrors halo2::plonk::Gate.
public struct Halo2Gate {
    public let name: String
    /// Each expression in the list must evaluate to zero for valid witnesses.
    public let polys: [Halo2Expression]

    public init(name: String, polys: [Halo2Expression]) {
        self.name = name
        self.polys = polys
    }
}

// MARK: - Lookup Constraint

/// A lookup constraint: the input expressions must be found in the table expressions.
/// Mirrors halo2::plonk::lookup::Argument.
public struct Halo2Lookup {
    public let name: String
    /// (input_expression, table_expression) pairs.
    public let inputExpressions: [Halo2Expression]
    public let tableExpressions: [Halo2Expression]

    public init(name: String,
                inputExpressions: [Halo2Expression],
                tableExpressions: [Halo2Expression]) {
        self.name = name
        self.inputExpressions = inputExpressions
        self.tableExpressions = tableExpressions
    }
}

// MARK: - Constraint System

/// Halo2-compatible constraint system for circuit configuration.
///
/// Usage mirrors the Halo2 Rust API:
/// ```
/// let a = cs.adviceColumn()
/// let b = cs.adviceColumn()
/// let s = cs.selector()
/// cs.createGate("mul") { |virtual_cells| ... }
/// ```
public class Halo2ConstraintSystem {

    // Column allocations
    public private(set) var adviceColumns: [Halo2Column] = []
    public private(set) var fixedColumns: [Halo2Column] = []
    public private(set) var instanceColumns: [Halo2Column] = []

    // Selector allocations (each backed by a fixed column under the hood)
    public private(set) var selectors: [Halo2Selector] = []

    // Equality-enabled columns (for copy constraints / permutation argument)
    public private(set) var equalityEnabledColumns: Set<Halo2Column> = []

    // Registered gates
    public private(set) var gates: [Halo2Gate] = []

    // Registered lookup arguments
    public private(set) var lookups: [Halo2Lookup] = []

    // Minimum number of rows (may be increased by rotations)
    public private(set) var minimumRows: Int = 0

    public init() {}

    // MARK: - Column allocation

    /// Allocate a new advice column. Returns the column reference.
    @discardableResult
    public func adviceColumn() -> Halo2Column {
        let col = Halo2Column(.advice, adviceColumns.count)
        adviceColumns.append(col)
        return col
    }

    /// Allocate a new fixed column. Returns the column reference.
    @discardableResult
    public func fixedColumn() -> Halo2Column {
        let col = Halo2Column(.fixed, fixedColumns.count)
        fixedColumns.append(col)
        return col
    }

    /// Allocate a new instance (public input) column. Returns the column reference.
    @discardableResult
    public func instanceColumn() -> Halo2Column {
        let col = Halo2Column(.instance, instanceColumns.count)
        instanceColumns.append(col)
        return col
    }

    /// Allocate a simple selector. Returns the selector reference.
    @discardableResult
    public func selector() -> Halo2Selector {
        let sel = Halo2Selector(selectors.count)
        selectors.append(sel)
        return sel
    }

    // MARK: - Copy constraints

    /// Enable equality constraints (permutation argument) on a column.
    /// This allows `region.copy()` to constrain cells in this column to be equal.
    public func enableEquality(_ column: Halo2Column) {
        equalityEnabledColumns.insert(column)
    }

    // MARK: - Gates

    /// Create a custom gate with the given name.
    ///
    /// The closure receives a `VirtualCells` helper for querying column values
    /// at rotations, and returns a list of constraint expressions that must
    /// all evaluate to zero.
    ///
    /// Mirrors: `meta.create_gate("name", |virtual_cells| { ... })` in Halo2.
    public func createGate(_ name: String, _ constraints: (VirtualCells) -> [Halo2Expression]) {
        let vc = VirtualCells(cs: self)
        let polys = constraints(vc)
        gates.append(Halo2Gate(name: name, polys: polys))

        // Update minimum rows based on rotations used
        let maxRot = vc.maxRotationUsed
        if maxRot + 1 > minimumRows {
            minimumRows = maxRot + 1
        }
    }

    // MARK: - Lookups

    /// Register a lookup argument.
    ///
    /// The closure returns pairs of (input_expression, table_expression).
    /// For each row where the gate is active, the input values must appear
    /// in the corresponding table column.
    ///
    /// Mirrors: `meta.lookup("name", |virtual_cells| { ... })` in Halo2.
    public func lookup(_ name: String,
                       _ tableMap: (VirtualCells) -> [(Halo2Expression, Halo2Expression)]) {
        let vc = VirtualCells(cs: self)
        let pairs = tableMap(vc)
        let inputs = pairs.map { $0.0 }
        let tables = pairs.map { $0.1 }
        lookups.append(Halo2Lookup(name: name,
                                   inputExpressions: inputs,
                                   tableExpressions: tables))
    }
}

// MARK: - VirtualCells

/// Helper for querying cells during gate/lookup definition.
/// Mirrors halo2::plonk::VirtualCells.
public class VirtualCells {
    private weak var cs: Halo2ConstraintSystem?
    public private(set) var maxRotationUsed: Int = 0

    init(cs: Halo2ConstraintSystem) {
        self.cs = cs
    }

    /// Query a selector at the current row.
    public func querySelector(_ sel: Halo2Selector) -> Halo2Expression {
        return .selector(sel)
    }

    /// Query a fixed column at the given rotation.
    public func queryFixed(_ col: Halo2Column, at rotation: Rotation = .cur) -> Halo2Expression {
        trackRotation(rotation)
        return .fixed(col, rotation)
    }

    /// Query an advice column at the given rotation.
    public func queryAdvice(_ col: Halo2Column, at rotation: Rotation = .cur) -> Halo2Expression {
        trackRotation(rotation)
        return .advice(col, rotation)
    }

    /// Query an instance column at the given rotation.
    public func queryInstance(_ col: Halo2Column, at rotation: Rotation = .cur) -> Halo2Expression {
        trackRotation(rotation)
        return .instance(col, rotation)
    }

    private func trackRotation(_ rot: Rotation) {
        let absVal = abs(rot.value)
        if absVal > maxRotationUsed {
            maxRotationUsed = absVal
        }
    }
}

// MARK: - Region (Assignment)

/// A region for assigning values during circuit synthesis.
/// Mirrors halo2::circuit::Region.
///
/// The layouter allocates a contiguous block of rows for each region.
/// Assignments within the region use offsets relative to the region start.
public class Halo2Region {
    /// Region start row in the global trace.
    public let startRow: Int
    /// Reference to the assignment target.
    private let assignment: Halo2Assignment

    init(startRow: Int, assignment: Halo2Assignment) {
        self.startRow = startRow
        self.assignment = assignment
    }

    /// Assign a value to an advice cell.
    ///
    /// - Parameters:
    ///   - column: The advice column.
    ///   - offset: Row offset within this region.
    ///   - value: Closure returning the field value.
    /// - Returns: An `AssignedCell` handle for use in copy constraints.
    @discardableResult
    public func assignAdvice(
        column: Halo2Column,
        offset: Int,
        to value: () -> Fr
    ) -> AssignedCell {
        let row = startRow + offset
        let val = value()
        assignment.setAdvice(column: column.index, row: row, value: val)
        return AssignedCell(column: column, row: row, value: val)
    }

    /// Assign a value to a fixed cell.
    @discardableResult
    public func assignFixed(
        column: Halo2Column,
        offset: Int,
        to value: () -> Fr
    ) -> AssignedCell {
        let row = startRow + offset
        let val = value()
        assignment.setFixed(column: column.index, row: row, value: val)
        return AssignedCell(column: column, row: row, value: val)
    }

    /// Enable a selector at the given offset within this region.
    public func enableSelector(_ sel: Halo2Selector, offset: Int) {
        let row = startRow + offset
        assignment.setSelector(index: sel.index, row: row, value: Fr.one)
    }

    /// Copy constraint: constrain two cells to be equal.
    /// Both columns must have had `enableEquality()` called on them.
    public func constrainEqual(_ a: AssignedCell, _ b: AssignedCell) {
        assignment.addCopyConstraint(
            lhsColumn: a.column, lhsRow: a.row,
            rhsColumn: b.column, rhsRow: b.row
        )
    }

    /// Constrain a cell to a constant value.
    public func constrainConstant(_ cell: AssignedCell, constant: Fr) {
        assignment.addConstantConstraint(column: cell.column, row: cell.row, value: constant)
    }
}

// MARK: - AssignedCell

/// Handle to an assigned cell, used for copy constraints.
/// Mirrors halo2::circuit::AssignedCell.
public struct AssignedCell {
    public let column: Halo2Column
    public let row: Int
    public let value: Fr

    public init(column: Halo2Column, row: Int, value: Fr) {
        self.column = column
        self.row = row
        self.value = value
    }
}

// MARK: - Assignment Storage

/// Internal storage for cell assignments during synthesis.
/// Collects all advice/fixed/instance values plus copy constraints.
public class Halo2Assignment {
    /// advice[columnIndex][row] = Fr value
    public var advice: [[Fr?]] = []
    /// fixed[columnIndex][row] = Fr value
    public var fixed: [[Fr?]] = []
    /// instance[columnIndex][row] = Fr value
    public var instance: [[Fr?]] = []
    /// selector[selectorIndex][row] = Fr value (0 or 1)
    public var selectorValues: [[Fr?]] = []

    /// Copy constraints: (lhsColumn, lhsRow, rhsColumn, rhsRow)
    public var copyConstraints: [(Halo2Column, Int, Halo2Column, Int)] = []

    /// Constant constraints: (column, row, value)
    public var constantConstraints: [(Halo2Column, Int, Fr)] = []

    /// Number of rows in the trace
    public var numRows: Int

    public init(numAdvice: Int, numFixed: Int, numInstance: Int,
                numSelectors: Int, numRows: Int) {
        self.numRows = numRows
        self.advice = Array(repeating: Array(repeating: nil, count: numRows), count: numAdvice)
        self.fixed = Array(repeating: Array(repeating: nil, count: numRows), count: numFixed)
        self.instance = Array(repeating: Array(repeating: nil, count: numRows), count: numInstance)
        self.selectorValues = Array(repeating: Array(repeating: nil, count: numRows), count: numSelectors)
    }

    func ensureRows(_ needed: Int) {
        if needed <= numRows { return }
        let oldRows = numRows
        numRows = needed
        for i in 0..<advice.count {
            advice[i].append(contentsOf: Array(repeating: nil, count: needed - oldRows))
        }
        for i in 0..<fixed.count {
            fixed[i].append(contentsOf: Array(repeating: nil, count: needed - oldRows))
        }
        for i in 0..<instance.count {
            instance[i].append(contentsOf: Array(repeating: nil, count: needed - oldRows))
        }
        for i in 0..<selectorValues.count {
            selectorValues[i].append(contentsOf: Array(repeating: nil, count: needed - oldRows))
        }
    }

    func setAdvice(column: Int, row: Int, value: Fr) {
        ensureRows(row + 1)
        advice[column][row] = value
    }

    func setFixed(column: Int, row: Int, value: Fr) {
        ensureRows(row + 1)
        fixed[column][row] = value
    }

    func setSelector(index: Int, row: Int, value: Fr) {
        ensureRows(row + 1)
        selectorValues[index][row] = value
    }

    func addCopyConstraint(lhsColumn: Halo2Column, lhsRow: Int,
                           rhsColumn: Halo2Column, rhsRow: Int) {
        copyConstraints.append((lhsColumn, lhsRow, rhsColumn, rhsRow))
    }

    func addConstantConstraint(column: Halo2Column, row: Int, value: Fr) {
        constantConstraints.append((column, row, value))
    }
}

// MARK: - Layouter

/// Simple single-pass layouter that allocates regions sequentially.
/// Mirrors halo2::circuit::SimpleFloorPlanner.
public class Halo2Layouter {
    private let assignment: Halo2Assignment
    private var nextRow: Int = 0

    init(assignment: Halo2Assignment) {
        self.assignment = assignment
    }

    /// Allocate a region and run the closure to populate it.
    ///
    /// Mirrors: `layouter.assign_region(|| "name", |mut region| { ... })` in Halo2.
    public func assignRegion(_ name: String, _ assign: (Halo2Region) throws -> Void) throws {
        let region = Halo2Region(startRow: nextRow, assignment: assignment)
        try assign(region)
        // Advance past the rows used (conservatively: at least 1 row per region)
        let maxUsed = findMaxRowUsed()
        nextRow = maxUsed + 1
    }

    /// Set instance column values (public inputs) directly.
    public func setInstance(column: Halo2Column, row: Int, value: Fr) {
        assignment.ensureRows(row + 1)
        if column.index < assignment.instance.count {
            assignment.instance[column.index][row] = value
        }
    }

    private func findMaxRowUsed() -> Int {
        var maxRow = nextRow
        for col in assignment.advice {
            for (row, val) in col.enumerated() {
                if val != nil && row > maxRow { maxRow = row }
            }
        }
        for col in assignment.fixed {
            for (row, val) in col.enumerated() {
                if val != nil && row > maxRow { maxRow = row }
            }
        }
        return maxRow
    }
}

// MARK: - Circuit Protocol

/// Protocol matching Halo2's `Circuit<F>` trait.
///
/// Conformers define:
///   1. `configure()` — allocate columns, define gates and lookups
///   2. `synthesize()` — assign witness values using the layouter
///
/// Usage:
/// ```swift
/// struct MyCircuit: Halo2Circuit {
///     typealias Config = MyConfig
///
///     static func configure(cs: Halo2ConstraintSystem) -> MyConfig {
///         let a = cs.adviceColumn()
///         let s = cs.selector()
///         cs.createGate("add") { vc in
///             let lhs = vc.queryAdvice(a, at: .cur)
///             let rhs = vc.queryAdvice(a, at: .next)
///             let sel = vc.querySelector(s)
///             return [sel * (lhs + rhs)]
///         }
///         return MyConfig(a: a, s: s)
///     }
///
///     func synthesize(config: MyConfig, layouter: Halo2Layouter) throws {
///         try layouter.assignRegion("main") { region in
///             region.enableSelector(config.s, offset: 0)
///             region.assignAdvice(column: config.a, offset: 0) { Fr.one }
///         }
///     }
/// }
/// ```
public protocol Halo2Circuit {
    associatedtype Config

    /// Configure the constraint system: allocate columns, define gates.
    /// Called once during key generation.
    static func configure(cs: Halo2ConstraintSystem) -> Config

    /// Synthesize the circuit: assign all advice/fixed values.
    /// Called for each proof.
    func synthesize(config: Config, layouter: Halo2Layouter) throws
}
