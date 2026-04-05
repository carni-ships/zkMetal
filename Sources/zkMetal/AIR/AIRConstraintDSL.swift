// AIR Constraint DSL — high-level builder for defining AIR constraints
// that compile down to the existing Circle STARK prover infrastructure.
//
// Usage:
//   let air = AIRBuilder(logTraceLength: 4)
//   air.addColumn(name: "a")
//   air.addColumn(name: "b")
//   air.addPublicInput(name: "init_a", value: M31.one)
//   air.boundary { ctx in
//       ctx.col("a") - ctx.pub("init_a")  // a[0] == init_a
//   }
//   air.transition { ctx in
//       ctx.next("a") - ctx.col("b")      // a' = b
//   }
//   let compiled = try air.compile()
//
// The compiled result conforms to CircleAIR and can be passed directly
// to CircleSTARKProver for GPU-accelerated proving.

import Foundation

// MARK: - AIR Expression Tree

/// Symbolic expression over trace columns, public inputs, and constants.
/// These form the AST that gets flattened during compilation into concrete
/// M31 constraint evaluators.
public indirect enum AIRExpression {
    /// Value of a column at the current row
    case column(String)
    /// Value of a column at the next row (rotation +1)
    case nextColumn(String)
    /// Value of a column at the previous row (rotation -1)
    case prevColumn(String)
    /// A constant M31 value
    case constant(M31)
    /// Reference to a named public input
    case publicInput(String)
    /// Reference to a periodic column at the current row
    case periodicColumn(String)
    /// Addition of two expressions
    case add(AIRExpression, AIRExpression)
    /// Subtraction (a - b)
    case sub(AIRExpression, AIRExpression)
    /// Multiplication of two expressions
    case mul(AIRExpression, AIRExpression)
    /// Negation
    case neg(AIRExpression)
    /// Exponentiation by a constant power
    case pow(AIRExpression, UInt32)

    /// The maximum constraint degree of this expression tree.
    /// Columns and constants have degree 1, mul multiplies degrees, pow scales.
    public var degree: Int {
        switch self {
        case .column, .nextColumn, .prevColumn, .periodicColumn:
            return 1
        case .constant, .publicInput:
            return 0
        case .add(let a, let b), .sub(let a, let b):
            return max(a.degree, b.degree)
        case .mul(let a, let b):
            return a.degree + b.degree
        case .neg(let a):
            return a.degree
        case .pow(let base, let exp):
            return base.degree * Int(exp)
        }
    }

    /// Whether this expression references next-row values (needs next row access).
    public var usesNextRow: Bool {
        switch self {
        case .nextColumn:
            return true
        case .column, .prevColumn, .constant, .publicInput, .periodicColumn:
            return false
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            return a.usesNextRow || b.usesNextRow
        case .neg(let a), .pow(let a, _):
            return a.usesNextRow
        }
    }

    /// Whether this expression references prev-row values.
    public var usesPrevRow: Bool {
        switch self {
        case .prevColumn:
            return true
        case .column, .nextColumn, .constant, .publicInput, .periodicColumn:
            return false
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            return a.usesPrevRow || b.usesPrevRow
        case .neg(let a), .pow(let a, _):
            return a.usesPrevRow
        }
    }
}

// MARK: - Arithmetic Operators on AIRExpression

public func + (lhs: AIRExpression, rhs: AIRExpression) -> AIRExpression {
    .add(lhs, rhs)
}

public func - (lhs: AIRExpression, rhs: AIRExpression) -> AIRExpression {
    .sub(lhs, rhs)
}

public func * (lhs: AIRExpression, rhs: AIRExpression) -> AIRExpression {
    .mul(lhs, rhs)
}

public prefix func - (expr: AIRExpression) -> AIRExpression {
    .neg(expr)
}

/// Convenience: expression + integer constant
public func + (lhs: AIRExpression, rhs: UInt32) -> AIRExpression {
    .add(lhs, .constant(M31(v: rhs)))
}

public func + (lhs: UInt32, rhs: AIRExpression) -> AIRExpression {
    .add(.constant(M31(v: lhs)), rhs)
}

/// Convenience: expression - integer constant
public func - (lhs: AIRExpression, rhs: UInt32) -> AIRExpression {
    .sub(lhs, .constant(M31(v: rhs)))
}

/// Convenience: expression * integer constant
public func * (lhs: AIRExpression, rhs: UInt32) -> AIRExpression {
    .mul(lhs, .constant(M31(v: rhs)))
}

public func * (lhs: UInt32, rhs: AIRExpression) -> AIRExpression {
    .mul(.constant(M31(v: lhs)), rhs)
}

/// Power operator
public func ** (lhs: AIRExpression, rhs: UInt32) -> AIRExpression {
    .pow(lhs, rhs)
}

// Define the precedence for ** (exponentiation, higher than multiplication)
precedencegroup ExponentiationPrecedence {
    higherThan: MultiplicationPrecedence
    associativity: right
}

infix operator ** : ExponentiationPrecedence

// MARK: - AIR Context (provided to constraint closures)

/// Context object passed to transition and boundary constraint closures.
/// Provides named access to columns, public inputs, and periodic columns.
public class AIRContext {
    let columnNames: [String]
    let publicInputNames: [String]
    let periodicColumnNames: [String]

    init(columnNames: [String], publicInputNames: [String], periodicColumnNames: [String]) {
        self.columnNames = columnNames
        self.publicInputNames = publicInputNames
        self.periodicColumnNames = periodicColumnNames
    }

    /// Reference the current-row value of a named column.
    public func col(_ name: String) -> AIRExpression {
        precondition(columnNames.contains(name), "Unknown column '\(name)'. Available: \(columnNames)")
        return .column(name)
    }

    /// Reference the next-row value of a named column (rotation +1).
    public func next(_ name: String) -> AIRExpression {
        precondition(columnNames.contains(name), "Unknown column '\(name)'. Available: \(columnNames)")
        return .nextColumn(name)
    }

    /// Reference the previous-row value of a named column (rotation -1).
    public func prev(_ name: String) -> AIRExpression {
        precondition(columnNames.contains(name), "Unknown column '\(name)'. Available: \(columnNames)")
        return .prevColumn(name)
    }

    /// Reference a named public input.
    public func pub(_ name: String) -> AIRExpression {
        precondition(publicInputNames.contains(name), "Unknown public input '\(name)'. Available: \(publicInputNames)")
        return .publicInput(name)
    }

    /// Reference a periodic column at the current row.
    public func periodic(_ name: String) -> AIRExpression {
        precondition(periodicColumnNames.contains(name), "Unknown periodic column '\(name)'. Available: \(periodicColumnNames)")
        return .periodicColumn(name)
    }

    /// Create a constant expression from an M31 value.
    public func constant(_ value: M31) -> AIRExpression {
        .constant(value)
    }

    /// Create a constant expression from a UInt32.
    public func constant(_ value: UInt32) -> AIRExpression {
        .constant(M31(v: value))
    }
}

// MARK: - Boundary Constraint Specification

/// Where a boundary constraint applies.
public enum BoundaryLocation {
    /// First row (row 0)
    case first
    /// Last row (row n-1)
    case last
    /// Specific row index
    case row(Int)
}

/// A named boundary constraint: at a given row, the expression must equal zero.
struct DSLBoundaryConstraint {
    let location: BoundaryLocation
    let expression: AIRExpression
}

// MARK: - Periodic Column Definition

/// A periodic column repeats a fixed pattern of values over the trace.
/// The values array length must be a power of 2, and divides the trace length.
struct DSLPeriodicColumn {
    let name: String
    let values: [M31]
}

// MARK: - AIR Builder (fluent API)

/// Fluent builder for defining AIR constraints.
///
/// Example — Fibonacci:
/// ```
/// let builder = AIRBuilder(logTraceLength: 4)
/// builder.addColumn(name: "a")
/// builder.addColumn(name: "b")
/// builder.addPublicInput(name: "a0", value: M31.one)
/// builder.addPublicInput(name: "b0", value: M31.one)
/// builder.boundary(at: .first) { ctx in ctx.col("a") - ctx.pub("a0") }
/// builder.boundary(at: .first) { ctx in ctx.col("b") - ctx.pub("b0") }
/// builder.transition { ctx in ctx.next("a") - ctx.col("b") }
/// builder.transition { ctx in ctx.next("b") - (ctx.col("a") + ctx.col("b")) }
/// let air = try builder.compile()
/// ```
public class AIRBuilder {
    /// Log2 of trace length
    public let logTraceLength: Int

    /// Ordered column names
    private var columns: [String] = []
    /// Column name -> index mapping
    private var columnIndex: [String: Int] = [:]

    /// Public inputs: name -> value
    private var publicInputs: [(name: String, value: M31)] = []
    private var publicInputIndex: [String: Int] = [:]

    /// Periodic columns
    private var periodicColumns: [DSLPeriodicColumn] = []
    private var periodicColumnIndex: [String: Int] = [:]

    /// Transition constraints (must hold on rows 0..n-2)
    private var transitionConstraints: [AIRExpression] = []

    /// Boundary constraints
    private var boundaryConstraintDefs: [DSLBoundaryConstraint] = []

    /// Optional trace generator closure: given public inputs dict, returns [column][row] trace
    private var traceGenerator: (([String: M31]) -> [[M31]])?

    public init(logTraceLength: Int) {
        precondition(logTraceLength >= 1, "Need at least 2 rows")
        self.logTraceLength = logTraceLength
    }

    /// Trace length (2^logTraceLength).
    public var traceLength: Int { 1 << logTraceLength }

    // MARK: - Column Definition

    /// Add a trace column with the given name.
    @discardableResult
    public func addColumn(name: String) -> AIRBuilder {
        precondition(!columnIndex.keys.contains(name), "Duplicate column name '\(name)'")
        columnIndex[name] = columns.count
        columns.append(name)
        return self
    }

    // MARK: - Public Input Definition

    /// Add a named public input with its value.
    @discardableResult
    public func addPublicInput(name: String, value: M31) -> AIRBuilder {
        precondition(!publicInputIndex.keys.contains(name), "Duplicate public input '\(name)'")
        publicInputIndex[name] = publicInputs.count
        publicInputs.append((name: name, value: value))
        return self
    }

    /// Convenience: add a public input from a UInt32.
    @discardableResult
    public func addPublicInput(name: String, value: UInt32) -> AIRBuilder {
        return addPublicInput(name: name, value: M31(v: value))
    }

    // MARK: - Periodic Column Definition

    /// Add a periodic column. Values repeat cyclically over the trace.
    /// The values array length must be a power of 2 and divide traceLength.
    @discardableResult
    public func addPeriodicColumn(name: String, values: [M31]) -> AIRBuilder {
        precondition(!periodicColumnIndex.keys.contains(name), "Duplicate periodic column '\(name)'")
        let len = values.count
        precondition(len > 0 && (len & (len - 1)) == 0, "Periodic column length must be power of 2")
        precondition(traceLength % len == 0, "Periodic column length must divide trace length")
        periodicColumnIndex[name] = periodicColumns.count
        periodicColumns.append(DSLPeriodicColumn(name: name, values: values))
        return self
    }

    // MARK: - Transition Constraints

    /// Add a transition constraint. The expression must evaluate to zero on every
    /// row pair (current, next) for rows 0..n-2.
    @discardableResult
    public func transition(_ build: (AIRContext) -> AIRExpression) -> AIRBuilder {
        let ctx = AIRContext(
            columnNames: columns,
            publicInputNames: publicInputs.map(\.name),
            periodicColumnNames: periodicColumns.map(\.name)
        )
        let expr = build(ctx)
        transitionConstraints.append(expr)
        return self
    }

    // MARK: - Boundary Constraints

    /// Add a boundary constraint at the given location.
    /// The expression must evaluate to zero at that row.
    @discardableResult
    public func boundary(at location: BoundaryLocation = .first,
                         _ build: (AIRContext) -> AIRExpression) -> AIRBuilder {
        let ctx = AIRContext(
            columnNames: columns,
            publicInputNames: publicInputs.map(\.name),
            periodicColumnNames: periodicColumns.map(\.name)
        )
        let expr = build(ctx)
        boundaryConstraintDefs.append(DSLBoundaryConstraint(location: location, expression: expr))
        return self
    }

    // MARK: - Trace Generator

    /// Set the trace generation function.
    /// The closure receives a dictionary of public input name -> value and must return
    /// [column][row] trace data matching the declared columns and trace length.
    @discardableResult
    public func traceGen(_ generator: @escaping ([String: M31]) -> [[M31]]) -> AIRBuilder {
        self.traceGenerator = generator
        return self
    }

    // MARK: - Compilation

    /// Compilation errors.
    public enum CompileError: Error, CustomStringConvertible {
        case noColumns
        case noConstraints
        case unknownColumn(String)
        case unknownPublicInput(String)
        case unknownPeriodicColumn(String)
        case prevRowInTransition
        case noTraceGenerator

        public var description: String {
            switch self {
            case .noColumns: return "AIR has no columns defined"
            case .noConstraints: return "AIR has no constraints defined"
            case .unknownColumn(let n): return "Unknown column '\(n)' in constraint"
            case .unknownPublicInput(let n): return "Unknown public input '\(n)' in constraint"
            case .unknownPeriodicColumn(let n): return "Unknown periodic column '\(n)' in constraint"
            case .prevRowInTransition: return "prev() not supported in transition constraints (use next() instead)"
            case .noTraceGenerator: return "No trace generator set; call traceGen() before compile()"
            }
        }
    }

    /// Compile the builder into a CompiledAIR that conforms to CircleAIR.
    /// Validates all column references, computes constraint degrees, and builds
    /// the evaluator closures.
    public func compile() throws -> CompiledAIR {
        guard !columns.isEmpty else { throw CompileError.noColumns }
        guard !transitionConstraints.isEmpty || !boundaryConstraintDefs.isEmpty else {
            throw CompileError.noConstraints
        }
        guard traceGenerator != nil else { throw CompileError.noTraceGenerator }

        // Validate all expressions
        for expr in transitionConstraints {
            try validateExpression(expr)
            if expr.usesPrevRow {
                throw CompileError.prevRowInTransition
            }
        }
        for bc in boundaryConstraintDefs {
            try validateExpression(bc.expression)
        }

        // Compute constraint degrees
        let degrees = transitionConstraints.map { max($0.degree, 1) }

        // Build column index for evaluation
        let colIdx = columnIndex
        let pubIdx = publicInputIndex
        let pubValues = Dictionary(uniqueKeysWithValues: publicInputs.map { ($0.name, $0.value) })
        let perIdx = periodicColumnIndex
        let perCols = periodicColumns

        // Build evaluator closures for transition constraints
        let transitionEvaluators: [([M31], [M31], Int) -> M31] = transitionConstraints.map { expr in
            return { current, next, row in
                self.evaluate(expr, current: current, next: next, row: row,
                              colIdx: colIdx, pubValues: pubValues,
                              perIdx: perIdx, perCols: perCols)
            }
        }

        // Convert boundary constraints to (column, row, value) triples
        // by resolving simple patterns, or keep as expression evaluators
        var simpleBoundaries: [(column: Int, row: Int, value: M31)] = []
        var complexBoundaryEvaluators: [(row: Int, evaluator: ([M31]) -> M31)] = []

        let n = traceLength
        for bc in boundaryConstraintDefs {
            let row: Int
            switch bc.location {
            case .first: row = 0
            case .last: row = n - 1
            case .row(let r): row = r
            }

            // Try to resolve as simple "col(X) - constant/pub(Y)" pattern
            if let simple = resolveSimpleBoundary(bc.expression, pubValues: pubValues) {
                guard let ci = colIdx[simple.column] else {
                    throw CompileError.unknownColumn(simple.column)
                }
                simpleBoundaries.append((column: ci, row: row, value: simple.value))
            } else {
                // Complex boundary: evaluate the full expression at the given row
                let expr = bc.expression
                complexBoundaryEvaluators.append((row: row, evaluator: { current in
                    self.evaluate(expr, current: current, next: current, row: row,
                                  colIdx: colIdx, pubValues: pubValues,
                                  perIdx: perIdx, perCols: perCols)
                }))
            }
        }

        let gen = traceGenerator!
        let pubInputsCopy = publicInputs

        return CompiledAIR(
            numColumns: columns.count,
            logTraceLength: logTraceLength,
            numConstraints: transitionConstraints.count,
            constraintDegrees: degrees,
            simpleBoundaryConstraints: simpleBoundaries,
            complexBoundaryEvaluators: complexBoundaryEvaluators,
            transitionEvaluators: transitionEvaluators,
            traceGenerator: {
                let pubDict = Dictionary(uniqueKeysWithValues: pubInputsCopy.map { ($0.name, $0.value) })
                return gen(pubDict)
            }
        )
    }

    // MARK: - Internal Helpers

    /// Validate that all column/input references in an expression are valid.
    private func validateExpression(_ expr: AIRExpression) throws {
        switch expr {
        case .column(let name), .nextColumn(let name), .prevColumn(let name):
            guard columnIndex[name] != nil else { throw CompileError.unknownColumn(name) }
        case .publicInput(let name):
            guard publicInputIndex[name] != nil else { throw CompileError.unknownPublicInput(name) }
        case .periodicColumn(let name):
            guard periodicColumnIndex[name] != nil else { throw CompileError.unknownPeriodicColumn(name) }
        case .constant:
            break
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            try validateExpression(a)
            try validateExpression(b)
        case .neg(let a), .pow(let a, _):
            try validateExpression(a)
        }
    }

    /// Evaluate an AIRExpression given current/next row values.
    private func evaluate(_ expr: AIRExpression,
                          current: [M31], next: [M31], row: Int,
                          colIdx: [String: Int],
                          pubValues: [String: M31],
                          perIdx: [String: Int],
                          perCols: [DSLPeriodicColumn]) -> M31 {
        switch expr {
        case .column(let name):
            return current[colIdx[name]!]
        case .nextColumn(let name):
            return next[colIdx[name]!]
        case .prevColumn(let name):
            // For prev, the caller must arrange the values. In compiled form,
            // we handle this differently. For now, return current (boundary only).
            return current[colIdx[name]!]
        case .constant(let v):
            return v
        case .publicInput(let name):
            return pubValues[name]!
        case .periodicColumn(let name):
            let pc = perCols[perIdx[name]!]
            return pc.values[row % pc.values.count]
        case .add(let a, let b):
            return m31Add(evaluate(a, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols),
                          evaluate(b, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols))
        case .sub(let a, let b):
            return m31Sub(evaluate(a, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols),
                          evaluate(b, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols))
        case .mul(let a, let b):
            return m31Mul(evaluate(a, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols),
                          evaluate(b, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols))
        case .neg(let a):
            return m31Neg(evaluate(a, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols))
        case .pow(let base, let exp):
            let baseVal = evaluate(base, current: current, next: next, row: row,
                                   colIdx: colIdx, pubValues: pubValues,
                                   perIdx: perIdx, perCols: perCols)
            return m31Pow(baseVal, exp)
        }
    }

    /// Try to resolve a boundary expression as a simple "col(X) - value" pattern.
    /// Returns (column_name, expected_value) if it matches, nil otherwise.
    private func resolveSimpleBoundary(_ expr: AIRExpression, pubValues: [String: M31]) -> (column: String, value: M31)? {
        switch expr {
        // col("x") - constant(v)  => col x must equal v
        case .sub(.column(let name), .constant(let v)):
            return (name, v)
        // col("x") - pub("y")  => col x must equal pub y's value
        case .sub(.column(let name), .publicInput(let pubName)):
            if let v = pubValues[pubName] {
                return (name, v)
            }
            return nil
        // constant(v) - col("x") would mean col = -v; less common but handle it
        default:
            return nil
        }
    }
}

// MARK: - Compiled AIR (conforms to CircleAIR)

/// The result of compiling an AIRBuilder. Conforms to CircleAIR and can be
/// passed directly to CircleSTARKProver.
public struct CompiledAIR: CircleAIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let numConstraints: Int
    public let constraintDegrees: [Int]

    /// Simple boundary constraints that map directly to CircleAIR's format
    let simpleBoundaryConstraints: [(column: Int, row: Int, value: M31)]

    /// Complex boundary constraints that need expression evaluation
    let complexBoundaryEvaluators: [(row: Int, evaluator: ([M31]) -> M31)]

    /// Transition constraint evaluators: (current, next, rowIndex) -> M31
    let transitionEvaluators: [([M31], [M31], Int) -> M31]

    /// Trace generator closure
    let traceGenerator: () -> [[M31]]

    public var boundaryConstraints: [(column: Int, row: Int, value: M31)] {
        simpleBoundaryConstraints
    }

    public func generateTrace() -> [[M31]] {
        traceGenerator()
    }

    public func evaluateConstraints(current: [M31], next: [M31]) -> [M31] {
        // Row index is not available in the CircleAIR protocol, so we pass 0
        // for periodic column evaluation. For full periodic column support,
        // use evaluateConstraintsAtRow instead.
        return transitionEvaluators.map { eval in eval(current, next, 0) }
    }

    /// Evaluate transition constraints at a specific row (supports periodic columns).
    public func evaluateConstraintsAtRow(current: [M31], next: [M31], row: Int) -> [M31] {
        return transitionEvaluators.map { eval in eval(current, next, row) }
    }

    /// Verify the full trace including complex boundary constraints.
    /// This is a more complete verification than CircleAIR's default verifyTrace,
    /// as it also checks complex boundary expressions and periodic columns.
    public func verifyTraceFull(_ trace: [[M31]]) -> String? {
        let n = traceLength
        guard trace.count == numColumns else {
            return "Expected \(numColumns) columns, got \(trace.count)"
        }
        for (ci, col) in trace.enumerated() {
            guard col.count == n else {
                return "Column \(ci): expected \(n) rows, got \(col.count)"
            }
        }

        // Check simple boundary constraints
        for bc in simpleBoundaryConstraints {
            guard bc.column < numColumns && bc.row < n else {
                return "Boundary constraint out of range: col=\(bc.column), row=\(bc.row)"
            }
            if trace[bc.column][bc.row].v != bc.value.v {
                return "Boundary constraint failed: col=\(bc.column), row=\(bc.row), expected=\(bc.value.v), got=\(trace[bc.column][bc.row].v)"
            }
        }

        // Check complex boundary constraints
        for cbc in complexBoundaryEvaluators {
            let rowVals = (0..<numColumns).map { trace[$0][cbc.row] }
            let eval = cbc.evaluator(rowVals)
            if eval.v != 0 {
                return "Complex boundary constraint failed at row \(cbc.row): eval=\(eval.v)"
            }
        }

        // Check transition constraints on all rows except the last, with row index
        for i in 0..<(n - 1) {
            let current = (0..<numColumns).map { trace[$0][i] }
            let next = (0..<numColumns).map { trace[$0][i + 1] }
            let evals = evaluateConstraintsAtRow(current: current, next: next, row: i)
            for (ci, ev) in evals.enumerated() {
                if ev.v != 0 {
                    return "Transition constraint \(ci) failed at row \(i): eval=\(ev.v)"
                }
            }
        }

        return nil
    }
}
