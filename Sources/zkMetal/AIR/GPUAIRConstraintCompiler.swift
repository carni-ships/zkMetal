// GPU AIR Constraint Compiler — compiles high-level AIR constraint descriptions
// into optimized evaluation functions over the BN254 Fr scalar field.
//
// This compiler operates on Fr (BN254) elements rather than M31, enabling
// constraint evaluation for SNARKs and STARKs over large prime fields.
//
// Key features:
//   1. Parse constraint expressions (polynomial, periodic, boundary)
//   2. Compile to optimized evaluation closures (no AST traversal at eval time)
//   3. Support for transition constraints and boundary constraints
//   4. Constraint degree analysis and composition
//   5. Batch constraint evaluation over entire traces
//   6. Constant folding and expression optimization

import Foundation

// MARK: - Fr-Based Expression AST

/// Algebraic expression over BN254 Fr field elements.
/// This is the IR (intermediate representation) for the GPU AIR constraint compiler.
public indirect enum FrAIRExpression {
    case column(Int)                          // Column i at current row
    case nextColumn(Int)                      // Column i at next row
    case constant(Fr)                         // Field constant
    case publicInput(String)                  // Named public input
    case periodicColumn(Int, period: Int)     // Periodic column with given period
    case add(FrAIRExpression, FrAIRExpression)
    case sub(FrAIRExpression, FrAIRExpression)
    case mul(FrAIRExpression, FrAIRExpression)
    case neg(FrAIRExpression)
    case pow(FrAIRExpression, UInt64)

    /// Algebraic degree of this expression.
    public var degree: Int {
        switch self {
        case .column, .nextColumn:
            return 1
        case .constant, .publicInput:
            return 0
        case .periodicColumn:
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

    /// Whether this expression references next-row columns.
    public var usesNextRow: Bool {
        switch self {
        case .column, .constant, .publicInput, .periodicColumn:
            return false
        case .nextColumn:
            return true
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            return a.usesNextRow || b.usesNextRow
        case .neg(let a), .pow(let a, _):
            return a.usesNextRow
        }
    }

    /// Maximum column index referenced.
    public var maxColumnIndex: Int {
        switch self {
        case .column(let i), .nextColumn(let i):
            return i
        case .constant, .publicInput, .periodicColumn:
            return -1
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            return max(a.maxColumnIndex, b.maxColumnIndex)
        case .neg(let a), .pow(let a, _):
            return a.maxColumnIndex
        }
    }
}

// MARK: - Operator Overloads for FrAIRExpression

public func + (_ lhs: FrAIRExpression, _ rhs: FrAIRExpression) -> FrAIRExpression {
    .add(lhs, rhs)
}

public func - (_ lhs: FrAIRExpression, _ rhs: FrAIRExpression) -> FrAIRExpression {
    .sub(lhs, rhs)
}

public func * (_ lhs: FrAIRExpression, _ rhs: FrAIRExpression) -> FrAIRExpression {
    .mul(lhs, rhs)
}

public prefix func - (_ expr: FrAIRExpression) -> FrAIRExpression {
    .neg(expr)
}

// MARK: - Fr Expression Builder (positional API)

/// Lightweight builder for creating Fr-based AIR constraint expressions.
public struct FrAIRExprBuilder {
    /// Reference column i at the current row.
    public static func col(_ i: Int) -> FrAIRExpression {
        .column(i)
    }

    /// Reference column i at the next row.
    public static func next(_ i: Int) -> FrAIRExpression {
        .nextColumn(i)
    }

    /// A field constant from a small integer.
    public static func constant(_ v: UInt64) -> FrAIRExpression {
        .constant(frFromInt(v))
    }

    /// A field constant from an Fr value.
    public static func constant(_ v: Fr) -> FrAIRExpression {
        .constant(v)
    }

    /// A named public input.
    public static func publicInput(_ name: String) -> FrAIRExpression {
        .publicInput(name)
    }

    /// A periodic column (repeats every `period` rows).
    public static func periodic(_ index: Int, period: Int) -> FrAIRExpression {
        .periodicColumn(index, period: period)
    }
}

// MARK: - Compiled Fr Constraint

/// A compiled constraint: the AST has been flattened into a native Swift closure
/// that evaluates over Fr column arrays with no dictionary lookups or recursion.
public struct CompiledFrConstraint {
    /// Evaluate given current and next row values (indexed by column).
    /// Returns the constraint evaluation (zero on valid trace).
    public let evaluate: (_ current: [Fr], _ next: [Fr]) -> Fr

    /// Evaluate with row index for periodic column support.
    public let evaluateWithRow: (_ current: [Fr], _ next: [Fr], _ row: Int) -> Fr

    /// The algebraic degree of this constraint.
    public let degree: Int

    /// Human-readable description.
    public let label: String
}

// MARK: - Fr Constraint Kind

/// Describes what kind of constraint this is.
public enum FrConstraintKind {
    /// Transition constraint: applies to consecutive row pairs (row i, row i+1).
    case transition
    /// Boundary constraint: fixes a column value at a specific row.
    case boundary(column: Int, row: Int, value: Fr)
    /// Periodic constraint: active every `period` rows.
    case periodic(period: Int)
}

// MARK: - Fr Degree Analysis

/// Degree analysis for quotient polynomial bounds in Fr-based STARKs.
public struct FrConstraintDegreeAnalysis {
    /// Degree of each transition constraint.
    public let transitionDegrees: [Int]
    /// Maximum transition constraint degree.
    public let maxTransitionDegree: Int
    /// The composition degree bound.
    public let compositionDegreeBound: Int
    /// Number of quotient polynomial chunks for FRI.
    public let numQuotientChunks: Int

    public init(transitionDegrees: [Int], logTraceLength: Int) {
        self.transitionDegrees = transitionDegrees
        self.maxTransitionDegree = transitionDegrees.max() ?? 1
        let traceLength = 1 << logTraceLength
        self.compositionDegreeBound = maxTransitionDegree * traceLength
        self.numQuotientChunks = maxTransitionDegree
    }
}

// MARK: - Compiled Fr AIR

/// A fully compiled AIR over Fr, ready for proving.
public struct CompiledFrAIR {
    public let numColumns: Int
    public let logTraceLength: Int
    public let transitionConstraints: [CompiledFrConstraint]
    public let boundaryConstraints: [(column: Int, row: Int, value: Fr)]
    public let constraintDegrees: [Int]
    public let traceGenerator: () -> [[Fr]]

    public var traceLength: Int { 1 << logTraceLength }
    public var numConstraints: Int { transitionConstraints.count }

    /// Generate the execution trace.
    public func generateTrace() -> [[Fr]] {
        traceGenerator()
    }

    /// Verify the trace satisfies all constraints.
    /// Returns nil on success, or a description of the first violation.
    public func verifyTrace(_ trace: [[Fr]]) -> String? {
        let n = traceLength

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column < trace.count else {
                return "Boundary column \(bc.column) out of range"
            }
            guard bc.row < trace[bc.column].count else {
                return "Boundary row \(bc.row) out of range for column \(bc.column)"
            }
            if trace[bc.column][bc.row] != bc.value {
                return "Boundary violation: column \(bc.column), row \(bc.row)"
            }
        }

        // Check transition constraints on all consecutive row pairs
        for row in 0..<(n - 1) {
            var current = [Fr]()
            var next = [Fr]()
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            for (ci, constraint) in transitionConstraints.enumerated() {
                let eval = constraint.evaluateWithRow(current, next, row)
                if !eval.isZero {
                    return "Transition constraint \(ci) violated at row \(row)"
                }
            }
        }

        return nil
    }
}

// MARK: - GPU AIR Constraint Compiler

/// Compiles Fr-based AIR constraint descriptions into efficient evaluation code.
///
/// Takes FrAIRExpression ASTs and produces:
/// - Native Swift closures for fast CPU evaluation
/// - Degree analysis for quotient polynomial bounds
/// - Random linear combination (composition) functions
/// - Batch evaluation over entire traces
public struct GPUAIRConstraintCompiler {

    // MARK: - Expression Compilation

    /// Compile a single FrAIRExpression into an efficient evaluation closure.
    public static func compileExpression(
        _ expr: FrAIRExpression,
        numColumns: Int,
        publicInputValues: [String: Fr] = [:],
        periodicColumnValues: [[Fr]] = []
    ) throws -> CompiledFrConstraint {
        // Validate references
        try validateExpression(expr, numColumns: numColumns,
                               publicInputValues: publicInputValues,
                               numPeriodicColumns: periodicColumnValues.count)

        let degree = expr.degree
        let label = describeExpression(expr)

        // Build the evaluator closure (flattened, no AST traversal at eval time)
        let evaluator = buildEvaluator(expr,
                                        publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)

        return CompiledFrConstraint(
            evaluate: { current, next in evaluator(current, next, 0) },
            evaluateWithRow: evaluator,
            degree: degree,
            label: label
        )
    }

    // MARK: - Full AIR Compilation

    /// Compile a complete AIR from expressions, boundaries, and a trace generator.
    ///
    /// Usage:
    /// ```
    /// let air = try GPUAIRConstraintCompiler.compile(
    ///     numColumns: 2,
    ///     logTraceLength: 4,
    ///     transitions: [
    ///         FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1),
    ///         FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    ///     ],
    ///     boundaries: [(column: 0, row: 0, value: Fr.one)],
    ///     traceGenerator: { ... }
    /// )
    /// ```
    public static func compile(
        numColumns: Int,
        logTraceLength: Int,
        transitions: [FrAIRExpression],
        boundaries: [(column: Int, row: Int, value: Fr)],
        publicInputValues: [String: Fr] = [:],
        periodicColumnValues: [[Fr]] = [],
        traceGenerator: @escaping () -> [[Fr]]
    ) throws -> CompiledFrAIR {
        guard numColumns > 0 else {
            throw CompilerError.invalidColumnCount(numColumns)
        }
        guard !transitions.isEmpty || !boundaries.isEmpty else {
            throw CompilerError.noConstraints
        }

        var compiledTransitions: [CompiledFrConstraint] = []
        for (idx, expr) in transitions.enumerated() {
            do {
                let compiled = try compileExpression(
                    expr, numColumns: numColumns,
                    publicInputValues: publicInputValues,
                    periodicColumnValues: periodicColumnValues)
                compiledTransitions.append(compiled)
            } catch {
                throw CompilerError.invalidConstraint(index: idx, underlying: error)
            }
        }

        let degrees = compiledTransitions.map { max($0.degree, 1) }

        return CompiledFrAIR(
            numColumns: numColumns,
            logTraceLength: logTraceLength,
            transitionConstraints: compiledTransitions,
            boundaryConstraints: boundaries,
            constraintDegrees: degrees.isEmpty ? [1] : degrees,
            traceGenerator: traceGenerator
        )
    }

    // MARK: - Degree Analysis

    /// Analyze constraint degrees from compiled constraints.
    public static func analyzeDegrees(
        _ constraints: [CompiledFrConstraint],
        logTraceLength: Int
    ) -> FrConstraintDegreeAnalysis {
        let degrees = constraints.map { max($0.degree, 1) }
        return FrConstraintDegreeAnalysis(transitionDegrees: degrees,
                                           logTraceLength: logTraceLength)
    }

    /// Analyze degrees from raw expressions.
    public static func analyzeDegrees(
        _ expressions: [FrAIRExpression],
        logTraceLength: Int
    ) -> FrConstraintDegreeAnalysis {
        let degrees = expressions.map { max($0.degree, 1) }
        return FrConstraintDegreeAnalysis(transitionDegrees: degrees,
                                           logTraceLength: logTraceLength)
    }

    // MARK: - Composition (Random Linear Combination)

    /// Compose multiple constraints into a single evaluation using
    /// random linear combination: sum_i alpha^i * C_i(x).
    public static func composeConstraints(
        _ constraints: [CompiledFrConstraint],
        alpha: Fr
    ) -> (_ current: [Fr], _ next: [Fr]) -> Fr {
        return { current, next in
            var result = Fr.zero
            var alphaPow = Fr.one
            for constraint in constraints {
                let eval = constraint.evaluate(current, next)
                result = frAdd(result, frMul(alphaPow, eval))
                alphaPow = frMul(alphaPow, alpha)
            }
            return result
        }
    }

    // MARK: - Batch Evaluation

    /// Evaluate all transition constraints over the entire trace at once.
    /// Returns a matrix: [constraint_index][row] -> Fr evaluation.
    public static func batchEvaluate(
        constraints: [CompiledFrConstraint],
        trace: [[Fr]],
        numRows: Int
    ) -> [[Fr]] {
        let numConstraints = constraints.count
        var results = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows - 1),
                              count: numConstraints)

        let numColumns = trace.count
        for row in 0..<(numRows - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }

            for ci in 0..<numConstraints {
                results[ci][row] = constraints[ci].evaluateWithRow(current, next, row)
            }
        }
        return results
    }

    /// Evaluate the composed constraint (with alpha) over the entire trace.
    /// Returns a single array: [row] -> Fr evaluation of the composed constraint.
    public static func batchEvaluateComposed(
        constraints: [CompiledFrConstraint],
        trace: [[Fr]],
        numRows: Int,
        alpha: Fr
    ) -> [Fr] {
        let composed = composeConstraints(constraints, alpha: alpha)
        var results = [Fr](repeating: Fr.zero, count: numRows - 1)

        let numColumns = trace.count
        for row in 0..<(numRows - 1) {
            var current = [Fr]()
            current.reserveCapacity(numColumns)
            var next = [Fr]()
            next.reserveCapacity(numColumns)
            for col in 0..<numColumns {
                current.append(trace[col][row])
                next.append(trace[col][row + 1])
            }
            results[row] = composed(current, next)
        }
        return results
    }

    // MARK: - Expression Optimization (Constant Folding)

    /// Optimize an expression by folding constants at compile time.
    public static func optimizeExpression(_ expr: FrAIRExpression) -> FrAIRExpression {
        switch expr {
        case .column, .nextColumn, .publicInput, .periodicColumn:
            return expr

        case .constant:
            return expr

        case .add(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(frAdd(va, vb))
            }
            if case .constant(let v) = oa, v.isZero { return ob }
            if case .constant(let v) = ob, v.isZero { return oa }
            return .add(oa, ob)

        case .sub(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(frSub(va, vb))
            }
            if case .constant(let v) = ob, v.isZero { return oa }
            return .sub(oa, ob)

        case .mul(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(frMul(va, vb))
            }
            if case .constant(let v) = oa, v.isZero { return .constant(Fr.zero) }
            if case .constant(let v) = ob, v.isZero { return .constant(Fr.zero) }
            if case .constant(let v) = oa, v == Fr.one { return ob }
            if case .constant(let v) = ob, v == Fr.one { return oa }
            return .mul(oa, ob)

        case .neg(let a):
            let oa = optimizeExpression(a)
            if case .constant(let v) = oa {
                return .constant(frNeg(v))
            }
            return .neg(oa)

        case .pow(let base, let exp):
            let ob = optimizeExpression(base)
            if case .constant(let v) = ob {
                return .constant(frPow(v, exp))
            }
            if exp == 0 { return .constant(Fr.one) }
            if exp == 1 { return ob }
            return .pow(ob, exp)
        }
    }

    // MARK: - Compiler Errors

    public enum CompilerError: Error, CustomStringConvertible {
        case invalidColumnCount(Int)
        case noConstraints
        case columnOutOfRange(Int, max: Int)
        case unknownPublicInput(String)
        case periodicColumnOutOfRange(Int, max: Int)
        case invalidConstraint(index: Int, underlying: Error)

        public var description: String {
            switch self {
            case .invalidColumnCount(let n):
                return "Invalid column count: \(n)"
            case .noConstraints:
                return "No constraints provided"
            case .columnOutOfRange(let i, let m):
                return "Column \(i) out of range [0, \(m))"
            case .unknownPublicInput(let name):
                return "Unknown public input '\(name)'"
            case .periodicColumnOutOfRange(let i, let m):
                return "Periodic column \(i) out of range [0, \(m))"
            case .invalidConstraint(let i, let e):
                return "Constraint \(i): \(e)"
            }
        }
    }

    // MARK: - Internal: Validation

    private static func validateExpression(
        _ expr: FrAIRExpression,
        numColumns: Int,
        publicInputValues: [String: Fr],
        numPeriodicColumns: Int
    ) throws {
        switch expr {
        case .column(let i), .nextColumn(let i):
            guard i >= 0 && i < numColumns else {
                throw CompilerError.columnOutOfRange(i, max: numColumns)
            }
        case .constant:
            break
        case .publicInput(let name):
            guard publicInputValues[name] != nil else {
                throw CompilerError.unknownPublicInput(name)
            }
        case .periodicColumn(let i, _):
            guard i >= 0 && i < numPeriodicColumns else {
                throw CompilerError.periodicColumnOutOfRange(i, max: numPeriodicColumns)
            }
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            try validateExpression(a, numColumns: numColumns,
                                   publicInputValues: publicInputValues,
                                   numPeriodicColumns: numPeriodicColumns)
            try validateExpression(b, numColumns: numColumns,
                                   publicInputValues: publicInputValues,
                                   numPeriodicColumns: numPeriodicColumns)
        case .neg(let a), .pow(let a, _):
            try validateExpression(a, numColumns: numColumns,
                                   publicInputValues: publicInputValues,
                                   numPeriodicColumns: numPeriodicColumns)
        }
    }

    // MARK: - Internal: Build Evaluator Closure

    private static func buildEvaluator(
        _ expr: FrAIRExpression,
        publicInputValues: [String: Fr],
        periodicColumnValues: [[Fr]]
    ) -> (_ current: [Fr], _ next: [Fr], _ row: Int) -> Fr {
        switch expr {
        case .column(let i):
            return { current, _, _ in current[i] }

        case .nextColumn(let i):
            return { _, next, _ in next[i] }

        case .constant(let v):
            return { _, _, _ in v }

        case .publicInput(let name):
            let v = publicInputValues[name]!
            return { _, _, _ in v }

        case .periodicColumn(let i, let period):
            let values = periodicColumnValues[i]
            return { _, _, row in values[row % period] }

        case .add(let a, let b):
            let evalA = buildEvaluator(a, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            let evalB = buildEvaluator(b, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            return { c, n, r in frAdd(evalA(c, n, r), evalB(c, n, r)) }

        case .sub(let a, let b):
            let evalA = buildEvaluator(a, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            let evalB = buildEvaluator(b, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            return { c, n, r in frSub(evalA(c, n, r), evalB(c, n, r)) }

        case .mul(let a, let b):
            let evalA = buildEvaluator(a, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            let evalB = buildEvaluator(b, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            return { c, n, r in frMul(evalA(c, n, r), evalB(c, n, r)) }

        case .neg(let a):
            let evalA = buildEvaluator(a, publicInputValues: publicInputValues,
                                        periodicColumnValues: periodicColumnValues)
            return { c, n, r in frNeg(evalA(c, n, r)) }

        case .pow(let base, let exp):
            let evalBase = buildEvaluator(base, publicInputValues: publicInputValues,
                                           periodicColumnValues: periodicColumnValues)
            return { c, n, r in frPow(evalBase(c, n, r), exp) }
        }
    }

    // MARK: - Internal: Expression Description

    private static func describeExpression(_ expr: FrAIRExpression) -> String {
        switch expr {
        case .column(let i):
            return "col(\(i))"
        case .nextColumn(let i):
            return "next(\(i))"
        case .constant:
            return "const"
        case .publicInput(let name):
            return "pub(\(name))"
        case .periodicColumn(let i, let period):
            return "periodic(\(i),\(period))"
        case .add(let a, let b):
            return "(\(describeExpression(a)) + \(describeExpression(b)))"
        case .sub(let a, let b):
            return "(\(describeExpression(a)) - \(describeExpression(b)))"
        case .mul(let a, let b):
            return "(\(describeExpression(a)) * \(describeExpression(b)))"
        case .neg(let a):
            return "(-\(describeExpression(a)))"
        case .pow(let base, let exp):
            return "(\(describeExpression(base))^\(exp))"
        }
    }
}

// MARK: - Standard Fr AIR Library

/// Common AIR patterns over Fr (BN254).
public struct FrAIRStandardLibrary {

    // MARK: - Fibonacci AIR

    /// Fibonacci AIR over Fr: next(0) = col(1), next(1) = col(0) + col(1).
    public static func fibonacci(
        logTraceLength: Int,
        a0: Fr = Fr.one,
        b0: Fr = Fr.one
    ) throws -> CompiledFrAIR {
        let col = FrAIRExprBuilder.self

        let capturedA0 = a0
        let capturedB0 = b0

        return try GPUAIRConstraintCompiler.compile(
            numColumns: 2,
            logTraceLength: logTraceLength,
            transitions: [
                col.next(0) - col.col(1),
                col.next(1) - (col.col(0) + col.col(1))
            ],
            boundaries: [
                (column: 0, row: 0, value: a0),
                (column: 1, row: 0, value: b0)
            ],
            traceGenerator: {
                let n = 1 << logTraceLength
                var colA = [Fr](repeating: Fr.zero, count: n)
                var colB = [Fr](repeating: Fr.zero, count: n)
                colA[0] = capturedA0
                colB[0] = capturedB0
                for i in 1..<n {
                    colA[i] = colB[i - 1]
                    colB[i] = frAdd(colA[i - 1], colB[i - 1])
                }
                return [colA, colB]
            }
        )
    }

    // MARK: - Range Check AIR

    /// Range check: transition constraint diff * (bound - diff) = 0 on sorted values.
    public static func rangeCheck(
        logTraceLength: Int,
        values: [Fr],
        bound: UInt64 = 65536
    ) throws -> CompiledFrAIR {
        let n = 1 << logTraceLength
        precondition(values.count <= n, "Too many values for trace length")

        let col = FrAIRExprBuilder.self
        let diff = col.next(0) - col.col(0)
        let boundExpr = col.constant(bound)
        let constraint = diff * (boundExpr - diff)

        // Sort values for the trace
        let capturedValues = values

        return try GPUAIRConstraintCompiler.compile(
            numColumns: 1,
            logTraceLength: logTraceLength,
            transitions: [constraint],
            boundaries: [],
            traceGenerator: {
                let traceLen = 1 << logTraceLength
                // Sort by extracting integer values
                var sortable: [(UInt64, Fr)] = capturedValues.map { (frToUInt64($0), $0) }
                sortable.sort { $0.0 < $1.0 }
                var result = sortable.map { $0.1 }
                while result.count < traceLen {
                    result.append(result.last ?? Fr.zero)
                }
                return [result]
            }
        )
    }

    // MARK: - Permutation Check AIR

    /// Permutation check using grand product argument over Fr.
    public static func permutationCheck(
        logTraceLength: Int,
        valuesA: [Fr],
        valuesB: [Fr],
        gamma: Fr
    ) throws -> CompiledFrAIR {
        let n = 1 << logTraceLength
        precondition(valuesA.count == n)
        precondition(valuesB.count == n)

        let col = FrAIRExprBuilder.self
        let gammaExpr = col.constant(gamma)

        // next(2) * (col(1) + gamma) - col(2) * (col(0) + gamma) = 0
        let constraint = col.next(2) * (col.col(1) + gammaExpr) -
                          col.col(2) * (col.col(0) + gammaExpr)

        let capturedA = valuesA
        let capturedB = valuesB
        let capturedGamma = gamma

        return try GPUAIRConstraintCompiler.compile(
            numColumns: 3,
            logTraceLength: logTraceLength,
            transitions: [constraint],
            boundaries: [
                (column: 2, row: 0, value: Fr.one)
            ],
            traceGenerator: {
                let traceLen = 1 << logTraceLength
                var colA = capturedA
                var colB = capturedB

                var acc = [Fr](repeating: Fr.zero, count: traceLen)
                acc[0] = Fr.one
                for i in 1..<traceLen {
                    let num = frMul(acc[i - 1], frAdd(colA[i - 1], capturedGamma))
                    let den = frAdd(colB[i - 1], capturedGamma)
                    acc[i] = frMul(num, frInverse(den))
                }
                return [colA, colB, acc]
            }
        )
    }
}
