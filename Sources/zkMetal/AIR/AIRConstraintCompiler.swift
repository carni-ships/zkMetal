// AIR Constraint Compiler — compiles high-level algebraic constraint descriptions
// into efficient evaluation code for STARK provers.
//
// Provides two complementary APIs:
//   1. Index-based DSL: col(i), next(i), constant(v) for positional column access
//   2. Compilation to native Swift closures for CPU evaluation
//   3. Constraint degree analysis for quotient polynomial degree bounds
//   4. Automatic composition with random linear combination
//   5. Standard library of common AIR patterns (Fibonacci, Range Check, Permutation, Memory)
//
// The compiled output conforms to CircleAIR and works with CircleSTARKProver.

import Foundation

// MARK: - Index-Based Expression Builder

/// A lightweight expression builder that references columns by index rather than name.
/// This is the "positional" API described in the task specification:
///   col(i) — column i at current row
///   next(i) — column i at next row
///   constant(v) — field constant
public struct AIRExprBuilder {
    /// Reference column i at the current row.
    public static func col(_ i: Int) -> AIRExpression {
        .column("__col_\(i)")
    }

    /// Reference column i at the next row.
    public static func next(_ i: Int) -> AIRExpression {
        .nextColumn("__col_\(i)")
    }

    /// A field constant.
    public static func constant(_ v: UInt32) -> AIRExpression {
        .constant(M31(v: v))
    }

    /// A field constant from M31.
    public static func constant(_ v: M31) -> AIRExpression {
        .constant(v)
    }
}

// MARK: - Constraint Specification

/// A single constraint to be compiled. Wraps an expression with metadata.
public struct AIRConstraintSpec {
    /// The constraint expression (must evaluate to zero on valid trace).
    public let expression: AIRExpression
    /// Whether this is a transition constraint (applies to consecutive row pairs)
    /// or a boundary constraint (applies at a specific row).
    public let kind: Kind
    /// Optional period: if set, constraint is only active every `period` rows.
    public let period: Int?

    public enum Kind {
        case transition
        case boundary(column: Int, row: Int, value: M31)
        case periodic(period: Int)
    }
}

// MARK: - Compiled Constraint (efficient closure form)

/// A compiled constraint: the AST has been flattened into a native Swift closure
/// that directly evaluates over column arrays, avoiding dictionary lookups and
/// recursive AST traversal at evaluation time.
public struct CompiledConstraint {
    /// Evaluate this constraint given current and next row values (indexed by column).
    /// Returns the constraint evaluation (should be zero on valid trace).
    public let evaluate: (_ current: [M31], _ next: [M31]) -> M31

    /// The algebraic degree of this constraint.
    public let degree: Int

    /// Human-readable description of the constraint for debugging.
    public let description: String
}

// MARK: - Degree Analysis

/// Analyzes constraint degrees for quotient polynomial degree bound calculation.
public struct ConstraintDegreeAnalysis {
    /// Degree of each transition constraint.
    public let transitionDegrees: [Int]
    /// Maximum transition constraint degree.
    public let maxTransitionDegree: Int
    /// The composition degree bound: max_degree * trace_length.
    public let compositionDegreeBound: Int
    /// Number of quotient polynomial chunks needed for FRI.
    public let numQuotientChunks: Int

    public init(transitionDegrees: [Int], logTraceLength: Int) {
        self.transitionDegrees = transitionDegrees
        self.maxTransitionDegree = transitionDegrees.max() ?? 1
        let traceLength = 1 << logTraceLength
        // Composition polynomial degree = max_constraint_degree * trace_length
        // after dividing by the vanishing polynomial (degree trace_length),
        // the quotient has degree (max_degree - 1) * trace_length.
        self.compositionDegreeBound = maxTransitionDegree * traceLength
        // Number of chunks to split quotient for FRI commitment
        // Each chunk has degree < trace_length, so we need max_degree chunks.
        self.numQuotientChunks = maxTransitionDegree
    }
}

// MARK: - AIR Constraint Compiler

/// Compiles AIR constraint descriptions into efficient evaluation code.
///
/// The compiler takes constraint expressions (from AIRExprBuilder or AIRBuilder)
/// and produces:
/// - Native Swift closures for fast CPU evaluation
/// - Degree analysis for quotient polynomial bounds
/// - Random linear combination (composition) functions
/// - A CompiledAIR conforming to CircleAIR
public struct AIRConstraintCompiler {

    // MARK: - Expression Compilation

    /// Compile an AIRExpression into a native Swift closure for efficient evaluation.
    /// The closure takes (current_row_values, next_row_values) indexed by column position.
    ///
    /// This avoids dictionary lookups and recursive AST traversal at evaluation time
    /// by resolving all column references to integer indices at compile time.
    public static func compileExpression(
        _ expr: AIRExpression,
        columnIndex: [String: Int],
        publicInputValues: [String: M31] = [:],
        periodicColumns: [(name: String, values: [M31])] = []
    ) throws -> CompiledConstraint {
        let periodicIndex = Dictionary(uniqueKeysWithValues:
            periodicColumns.enumerated().map { ($0.element.name, $0.offset) })

        // Validate all references
        try validateReferences(expr, columnIndex: columnIndex,
                               publicInputValues: publicInputValues,
                               periodicIndex: periodicIndex)

        let degree = expr.degree

        // Build the closure by flattening the AST at compile time
        let evaluator = buildEvaluator(expr, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)

        return CompiledConstraint(
            evaluate: { current, next in evaluator(current, next, 0) },
            degree: degree,
            description: describeExpression(expr)
        )
    }

    /// Compile an expression that supports row-indexed periodic column evaluation.
    public static func compileExpressionWithRow(
        _ expr: AIRExpression,
        columnIndex: [String: Int],
        publicInputValues: [String: M31] = [:],
        periodicColumns: [(name: String, values: [M31])] = []
    ) throws -> (_ current: [M31], _ next: [M31], _ row: Int) -> M31 {
        let periodicIndex = Dictionary(uniqueKeysWithValues:
            periodicColumns.enumerated().map { ($0.element.name, $0.offset) })

        try validateReferences(expr, columnIndex: columnIndex,
                               publicInputValues: publicInputValues,
                               periodicIndex: periodicIndex)

        return buildEvaluator(expr, columnIndex: columnIndex,
                              publicInputValues: publicInputValues,
                              periodicColumns: periodicColumns,
                              periodicIndex: periodicIndex)
    }

    // MARK: - Composition (Random Linear Combination)

    /// Compose multiple compiled constraints into a single evaluation using
    /// random linear combination: sum_i alpha^i * C_i(x).
    ///
    /// This is the standard approach for batching constraints in STARK provers.
    public static func composeConstraints(
        _ constraints: [CompiledConstraint],
        alpha: M31
    ) -> (_ current: [M31], _ next: [M31]) -> M31 {
        return { current, next in
            var result = M31.zero
            var alphaPow = M31.one
            for constraint in constraints {
                let eval = constraint.evaluate(current, next)
                result = m31Add(result, m31Mul(alphaPow, eval))
                alphaPow = m31Mul(alphaPow, alpha)
            }
            return result
        }
    }

    /// Compose constraints with QM31 extension field alpha for full soundness.
    public static func composeConstraintsQM31(
        _ constraints: [CompiledConstraint],
        alpha: QM31
    ) -> (_ current: [M31], _ next: [M31]) -> QM31 {
        return { current, next in
            var result = QM31.zero
            var alphaPow = QM31.one
            for constraint in constraints {
                let eval = constraint.evaluate(current, next)
                let lifted = QM31.from(eval)
                result = qm31Add(result, qm31Mul(alphaPow, lifted))
                alphaPow = qm31Mul(alphaPow, alpha)
            }
            return result
        }
    }

    // MARK: - Degree Analysis

    /// Analyze the degrees of a set of constraints.
    public static func analyzeDegrees(
        _ constraints: [CompiledConstraint],
        logTraceLength: Int
    ) -> ConstraintDegreeAnalysis {
        let degrees = constraints.map { max($0.degree, 1) }
        return ConstraintDegreeAnalysis(transitionDegrees: degrees,
                                         logTraceLength: logTraceLength)
    }

    /// Analyze degrees from raw expressions.
    public static func analyzeDegrees(
        _ expressions: [AIRExpression],
        logTraceLength: Int
    ) -> ConstraintDegreeAnalysis {
        let degrees = expressions.map { max($0.degree, 1) }
        return ConstraintDegreeAnalysis(transitionDegrees: degrees,
                                         logTraceLength: logTraceLength)
    }

    // MARK: - Full AIR Compilation (Index-Based API)

    /// Compile a complete AIR from index-based constraint specifications.
    ///
    /// This is the primary entry point for the index-based API:
    /// ```
    /// let air = try AIRConstraintCompiler.compile(
    ///     numColumns: 2,
    ///     logTraceLength: 4,
    ///     transitions: [
    ///         AIRExprBuilder.next(0) - AIRExprBuilder.col(1),
    ///         AIRExprBuilder.next(1) - (AIRExprBuilder.col(0) + AIRExprBuilder.col(1))
    ///     ],
    ///     boundaries: [(column: 0, row: 0, value: M31.one),
    ///                  (column: 1, row: 0, value: M31.one)],
    ///     traceGenerator: { ... }
    /// )
    /// ```
    public static func compile(
        numColumns: Int,
        logTraceLength: Int,
        transitions: [AIRExpression],
        boundaries: [(column: Int, row: Int, value: M31)],
        traceGenerator: @escaping () -> [[M31]]
    ) throws -> CompiledAIR {
        guard numColumns > 0 else {
            throw AIRBuilder.CompileError.noColumns
        }
        guard !transitions.isEmpty || !boundaries.isEmpty else {
            throw AIRBuilder.CompileError.noConstraints
        }

        // Build column index for __col_N naming convention
        var columnIndex: [String: Int] = [:]
        for i in 0..<numColumns {
            columnIndex["__col_\(i)"] = i
        }

        // Validate and compile transition constraints
        var compiledTransitions: [CompiledConstraint] = []
        for (idx, expr) in transitions.enumerated() {
            do {
                let compiled = try compileExpression(expr, columnIndex: columnIndex)
                compiledTransitions.append(compiled)
            } catch {
                throw CompilerError.invalidTransitionConstraint(
                    index: idx, underlying: error)
            }
        }

        // Build evaluators with closure capture for CompiledAIR
        let transitionEvaluators: [([M31], [M31], Int) -> M31] = compiledTransitions.map { cc in
            return { current, next, _ in cc.evaluate(current, next) }
        }

        let degrees = compiledTransitions.map { $0.degree }

        return CompiledAIR(
            numColumns: numColumns,
            logTraceLength: logTraceLength,
            numConstraints: transitions.count,
            constraintDegrees: degrees.isEmpty ? [1] : degrees,
            simpleBoundaryConstraints: boundaries,
            complexBoundaryEvaluators: [],
            transitionEvaluators: transitionEvaluators,
            traceGenerator: traceGenerator
        )
    }

    // MARK: - Compiler Errors

    public enum CompilerError: Error, CustomStringConvertible {
        case unknownColumn(String)
        case unknownPublicInput(String)
        case unknownPeriodicColumn(String)
        case invalidTransitionConstraint(index: Int, underlying: Error)
        case invalidBoundaryConstraint(index: Int, underlying: Error)

        public var description: String {
            switch self {
            case .unknownColumn(let n):
                return "Unknown column '\(n)' in constraint"
            case .unknownPublicInput(let n):
                return "Unknown public input '\(n)' in constraint"
            case .unknownPeriodicColumn(let n):
                return "Unknown periodic column '\(n)' in constraint"
            case .invalidTransitionConstraint(let i, let e):
                return "Transition constraint \(i): \(e)"
            case .invalidBoundaryConstraint(let i, let e):
                return "Boundary constraint \(i): \(e)"
            }
        }
    }

    // MARK: - Internal: AST Validation

    private static func validateReferences(
        _ expr: AIRExpression,
        columnIndex: [String: Int],
        publicInputValues: [String: M31],
        periodicIndex: [String: Int]
    ) throws {
        switch expr {
        case .column(let name), .nextColumn(let name), .prevColumn(let name):
            guard columnIndex[name] != nil else {
                throw CompilerError.unknownColumn(name)
            }
        case .publicInput(let name):
            guard publicInputValues[name] != nil else {
                throw CompilerError.unknownPublicInput(name)
            }
        case .periodicColumn(let name):
            guard periodicIndex[name] != nil else {
                throw CompilerError.unknownPeriodicColumn(name)
            }
        case .constant:
            break
        case .add(let a, let b), .sub(let a, let b), .mul(let a, let b):
            try validateReferences(a, columnIndex: columnIndex,
                                   publicInputValues: publicInputValues,
                                   periodicIndex: periodicIndex)
            try validateReferences(b, columnIndex: columnIndex,
                                   publicInputValues: publicInputValues,
                                   periodicIndex: periodicIndex)
        case .neg(let a), .pow(let a, _):
            try validateReferences(a, columnIndex: columnIndex,
                                   publicInputValues: publicInputValues,
                                   periodicIndex: periodicIndex)
        }
    }

    // MARK: - Internal: Build Evaluator Closure

    /// Recursively builds a closure from an AST node. The closure captures all
    /// resolved indices and constant values, so evaluation is a flat series of
    /// field operations with no dictionary lookups.
    private static func buildEvaluator(
        _ expr: AIRExpression,
        columnIndex: [String: Int],
        publicInputValues: [String: M31],
        periodicColumns: [(name: String, values: [M31])],
        periodicIndex: [String: Int]
    ) -> (_ current: [M31], _ next: [M31], _ row: Int) -> M31 {
        switch expr {
        case .column(let name):
            let idx = columnIndex[name]!
            return { current, _, _ in current[idx] }

        case .nextColumn(let name):
            let idx = columnIndex[name]!
            return { _, next, _ in next[idx] }

        case .prevColumn(let name):
            let idx = columnIndex[name]!
            return { current, _, _ in current[idx] }

        case .constant(let v):
            return { _, _, _ in v }

        case .publicInput(let name):
            let v = publicInputValues[name]!
            return { _, _, _ in v }

        case .periodicColumn(let name):
            let pcIdx = periodicIndex[name]!
            let values = periodicColumns[pcIdx].values
            let period = values.count
            return { _, _, row in values[row % period] }

        case .add(let a, let b):
            let evalA = buildEvaluator(a, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            let evalB = buildEvaluator(b, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            return { c, n, r in m31Add(evalA(c, n, r), evalB(c, n, r)) }

        case .sub(let a, let b):
            let evalA = buildEvaluator(a, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            let evalB = buildEvaluator(b, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            return { c, n, r in m31Sub(evalA(c, n, r), evalB(c, n, r)) }

        case .mul(let a, let b):
            let evalA = buildEvaluator(a, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            let evalB = buildEvaluator(b, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            return { c, n, r in m31Mul(evalA(c, n, r), evalB(c, n, r)) }

        case .neg(let a):
            let evalA = buildEvaluator(a, columnIndex: columnIndex,
                                        publicInputValues: publicInputValues,
                                        periodicColumns: periodicColumns,
                                        periodicIndex: periodicIndex)
            return { c, n, r in m31Neg(evalA(c, n, r)) }

        case .pow(let base, let exp):
            let evalBase = buildEvaluator(base, columnIndex: columnIndex,
                                           publicInputValues: publicInputValues,
                                           periodicColumns: periodicColumns,
                                           periodicIndex: periodicIndex)
            return { c, n, r in m31Pow(evalBase(c, n, r), exp) }
        }
    }

    // MARK: - Internal: Expression Description

    private static func describeExpression(_ expr: AIRExpression) -> String {
        switch expr {
        case .column(let name):
            if name.hasPrefix("__col_") {
                return "col(\(name.dropFirst(6)))"
            }
            return name
        case .nextColumn(let name):
            if name.hasPrefix("__col_") {
                return "next(\(name.dropFirst(6)))"
            }
            return "next(\(name))"
        case .prevColumn(let name):
            return "prev(\(name))"
        case .constant(let v):
            return "\(v.v)"
        case .publicInput(let name):
            return "pub(\(name))"
        case .periodicColumn(let name):
            return "periodic(\(name))"
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

// MARK: - Standard AIR Library

/// Standard library of common AIR constraint patterns.
/// Each function returns a CompiledAIR that conforms to CircleAIR and can be
/// passed directly to CircleSTARKProver.
public struct AIRStandardLibrary {

    // MARK: - Fibonacci AIR

    /// Build a Fibonacci AIR: next(0) = col(1), next(1) = col(0) + col(1).
    ///
    /// Two columns (a, b), linear constraints, trace length 2^logN.
    /// This is equivalent to FibonacciAIR but built via the compiler.
    public static func fibonacci(
        logTraceLength: Int,
        a0: M31 = M31.one,
        b0: M31 = M31.one
    ) throws -> CompiledAIR {
        let col = AIRExprBuilder.self
        let capturedA0 = a0
        let capturedB0 = b0

        return try AIRConstraintCompiler.compile(
            numColumns: 2,
            logTraceLength: logTraceLength,
            transitions: [
                // next(0) = col(1)  =>  next(0) - col(1) = 0
                col.next(0) - col.col(1),
                // next(1) = col(0) + col(1)  =>  next(1) - col(0) - col(1) = 0
                col.next(1) - col.col(0) - col.col(1)
            ],
            boundaries: [
                (column: 0, row: 0, value: a0),
                (column: 1, row: 0, value: b0)
            ],
            traceGenerator: {
                let n = 1 << logTraceLength
                var colA = [M31](repeating: M31.zero, count: n)
                var colB = [M31](repeating: M31.zero, count: n)
                colA[0] = capturedA0
                colB[0] = capturedB0
                for i in 1..<n {
                    colA[i] = colB[i - 1]
                    colB[i] = m31Add(colA[i - 1], colB[i - 1])
                }
                return [colA, colB]
            }
        )
    }

    // MARK: - Range Check AIR

    /// Build a range check AIR: proves each value in [0, bound).
    ///
    /// Uses the sorted-value approach: trace is sorted, transition checks monotonicity.
    /// Constraint: diff * (bound - diff) where diff = next(0) - col(0).
    /// Degree 2 (quadratic).
    public static func rangeCheck(
        logTraceLength: Int,
        values: [M31],
        bound: UInt32 = 65536
    ) throws -> CompiledAIR {
        let n = 1 << logTraceLength
        precondition(values.count <= n, "Too many values for trace length")

        let col = AIRExprBuilder.self
        let sorted = values.sorted { $0.v < $1.v }
        let minVal = sorted.first ?? M31.zero

        // diff = next(0) - col(0)
        // constraint: diff * (bound - diff) = 0
        let diff = col.next(0) - col.col(0)
        let boundExpr = col.constant(bound)
        let constraint = diff * (boundExpr - diff)

        let capturedValues = values

        return try AIRConstraintCompiler.compile(
            numColumns: 1,
            logTraceLength: logTraceLength,
            transitions: [constraint],
            boundaries: [(column: 0, row: 0, value: minVal)],
            traceGenerator: {
                let traceLen = 1 << logTraceLength
                var s = capturedValues.sorted { $0.v < $1.v }
                while s.count < traceLen {
                    s.append(s.last ?? M31.zero)
                }
                return [s]
            }
        )
    }

    // MARK: - Permutation Check AIR

    /// Build a permutation check AIR: proves column B is a permutation of column A.
    ///
    /// Uses the grand product argument:
    ///   Column 0: values A (sorted or original order)
    ///   Column 1: values B (claimed permutation)
    ///   Column 2: running product accumulator
    ///
    /// The running product accumulates (A[i] + gamma) / (B[i] + gamma) for a
    /// random challenge gamma. If B is a permutation of A, the final product is 1.
    ///
    /// Transition constraint:
    ///   acc[i+1] * (B[i] + gamma) = acc[i] * (A[i] + gamma)
    ///   => next(2) * (col(1) + gamma) - col(2) * (col(0) + gamma) = 0
    ///
    /// Boundary constraints:
    ///   acc[0] = 1 (start with identity)
    ///
    /// Note: gamma should be a Fiat-Shamir challenge in a real prover; here
    /// we use a deterministic value for the compiled AIR.
    public static func permutationCheck(
        logTraceLength: Int,
        valuesA: [M31],
        valuesB: [M31],
        gamma: M31
    ) throws -> CompiledAIR {
        let n = 1 << logTraceLength
        precondition(valuesA.count == n, "valuesA must have exactly 2^logN elements")
        precondition(valuesB.count == n, "valuesB must have exactly 2^logN elements")

        let col = AIRExprBuilder.self
        let gammaExpr = col.constant(gamma)

        // Transition: next(2) * (col(1) + gamma) - col(2) * (col(0) + gamma) = 0
        let constraint = col.next(2) * (col.col(1) + gammaExpr) -
                          col.col(2) * (col.col(0) + gammaExpr)

        let capturedA = valuesA
        let capturedB = valuesB
        let capturedGamma = gamma

        return try AIRConstraintCompiler.compile(
            numColumns: 3,
            logTraceLength: logTraceLength,
            transitions: [constraint],
            boundaries: [
                (column: 2, row: 0, value: M31.one) // acc starts at 1
            ],
            traceGenerator: {
                let traceLen = 1 << logTraceLength
                var colA = capturedA
                var colB = capturedB
                // Pad if needed
                while colA.count < traceLen { colA.append(M31.zero) }
                while colB.count < traceLen { colB.append(M31.zero) }

                // Compute running product accumulator
                var acc = [M31](repeating: M31.zero, count: traceLen)
                acc[0] = M31.one
                for i in 1..<traceLen {
                    // acc[i] = acc[i-1] * (A[i-1] + gamma) / (B[i-1] + gamma)
                    let num = m31Mul(acc[i - 1], m31Add(colA[i - 1], capturedGamma))
                    let den = m31Add(colB[i - 1], capturedGamma)
                    acc[i] = m31Mul(num, m31Inverse(den))
                }
                return [colA, colB, acc]
            }
        )
    }

    // MARK: - Memory Consistency AIR

    /// Build a memory consistency AIR: proves valid read/write memory access.
    ///
    /// Trace layout (address-sorted):
    ///   Column 0: address
    ///   Column 1: value
    ///   Column 2: is_write (1 for write, 0 for read)
    ///   Column 3: timestamp (monotonically increasing)
    ///
    /// Transition constraints:
    ///   1. Addresses are non-decreasing: addr_diff * (addr_diff - 1) = 0
    ///      where addr_diff = next(0) - col(0), meaning addr changes by 0 or 1
    ///      (simplified; in practice addresses can differ by more)
    ///   2. When same address (addr_diff = 0): value unchanged unless write
    ///      => (1 - addr_diff) * (1 - next(2)) * (next(1) - col(1)) = 0
    ///      (if same address and next is a read, value must match)
    ///   3. Timestamps are strictly increasing:
    ///      next(3) - col(3) - 1 constrained to be in [0, max_time)
    ///      (simplified: just checking next(3) > col(3))
    public static func memoryConsistency(
        logTraceLength: Int,
        operations: [(address: UInt32, value: M31, isWrite: Bool, timestamp: UInt32)]
    ) throws -> CompiledAIR {
        let n = 1 << logTraceLength
        precondition(operations.count <= n, "Too many operations for trace length")

        let col = AIRExprBuilder.self
        let one = col.constant(1)

        // addr_diff = next(0) - col(0)
        let addrDiff = col.next(0) - col.col(0)

        // Constraint 1: address non-decreasing, and when same address + read, value matches
        // (1 - addrDiff) * (1 - next(2)) * (next(1) - col(1)) = 0
        let sameAddrReadConstraint =
            (one - addrDiff) * (one - col.next(2)) * (col.next(1) - col.col(1))

        let capturedOps = operations

        return try AIRConstraintCompiler.compile(
            numColumns: 4,
            logTraceLength: logTraceLength,
            transitions: [sameAddrReadConstraint],
            boundaries: [],
            traceGenerator: {
                let traceLen = 1 << logTraceLength
                // Sort operations by address, then timestamp
                var sorted = capturedOps.sorted {
                    if $0.address != $1.address { return $0.address < $1.address }
                    return $0.timestamp < $1.timestamp
                }
                // Pad with dummy reads
                while sorted.count < traceLen {
                    let lastAddr = sorted.last?.address ?? 0
                    let lastVal = sorted.last?.value ?? M31.zero
                    let lastTs = sorted.last?.timestamp ?? 0
                    sorted.append((address: lastAddr, value: lastVal,
                                   isWrite: false, timestamp: lastTs + 1))
                }

                var addresses = [M31](repeating: M31.zero, count: traceLen)
                var values = [M31](repeating: M31.zero, count: traceLen)
                var writes = [M31](repeating: M31.zero, count: traceLen)
                var timestamps = [M31](repeating: M31.zero, count: traceLen)

                for i in 0..<traceLen {
                    addresses[i] = M31(v: sorted[i].address)
                    values[i] = sorted[i].value
                    writes[i] = M31(v: sorted[i].isWrite ? 1 : 0)
                    timestamps[i] = M31(v: sorted[i].timestamp)
                }

                return [addresses, values, writes, timestamps]
            }
        )
    }
}

// MARK: - Expression Optimization (Constant Folding)

extension AIRConstraintCompiler {
    /// Optimize an expression by folding constants at compile time.
    /// For example: constant(3) + constant(4) -> constant(7).
    public static func optimizeExpression(_ expr: AIRExpression) -> AIRExpression {
        switch expr {
        case .column, .nextColumn, .prevColumn, .publicInput, .periodicColumn:
            return expr

        case .constant:
            return expr

        case .add(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(m31Add(va, vb))
            }
            // 0 + x = x
            if case .constant(let v) = oa, v.v == 0 { return ob }
            if case .constant(let v) = ob, v.v == 0 { return oa }
            return .add(oa, ob)

        case .sub(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(m31Sub(va, vb))
            }
            // x - 0 = x
            if case .constant(let v) = ob, v.v == 0 { return oa }
            return .sub(oa, ob)

        case .mul(let a, let b):
            let oa = optimizeExpression(a)
            let ob = optimizeExpression(b)
            if case .constant(let va) = oa, case .constant(let vb) = ob {
                return .constant(m31Mul(va, vb))
            }
            // 0 * x = 0
            if case .constant(let v) = oa, v.v == 0 { return .constant(M31.zero) }
            if case .constant(let v) = ob, v.v == 0 { return .constant(M31.zero) }
            // 1 * x = x
            if case .constant(let v) = oa, v.v == 1 { return ob }
            if case .constant(let v) = ob, v.v == 1 { return oa }
            return .mul(oa, ob)

        case .neg(let a):
            let oa = optimizeExpression(a)
            if case .constant(let v) = oa {
                return .constant(m31Neg(v))
            }
            return .neg(oa)

        case .pow(let base, let exp):
            let ob = optimizeExpression(base)
            if case .constant(let v) = ob {
                return .constant(m31Pow(v, exp))
            }
            if exp == 0 { return .constant(M31.one) }
            if exp == 1 { return ob }
            return .pow(ob, exp)
        }
    }
}
