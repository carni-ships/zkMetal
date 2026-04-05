// R1CSOptimizer — R1CS-specific constraint optimizations
//
// R1CS constraints have the form: A * z . B * z = C * z
// where . is Hadamard (element-wise) product and z = [1, public_inputs, witness].
//
// This module provides R1CS-specific optimizations that exploit the bilinear structure:
//   1. Linear constraint elimination — substitute variables appearing in exactly one
//      linear (degree-1) constraint
//   2. Witness size reduction — back-substitute intermediate variables to reduce
//      the witness vector length
//
// All operations use BN254 Fr arithmetic via NeonFieldOps CIOS Montgomery.

import Foundation
import NeonFieldOps

// MARK: - R1CS System

/// An R1CS instance: A * z . B * z = C * z
/// where z = [1, x_1, ..., x_l, w_1, ..., w_m]
///   - l = numPublicInputs
///   - m = numWitness
///   - total variables n = 1 + l + m
public struct R1CSSystem {
    /// Left matrix A (m_constraints x n_variables)
    public var A: SparseMatrix
    /// Right matrix B
    public var B: SparseMatrix
    /// Output matrix C
    public var C: SparseMatrix
    /// Number of public input variables (excludes the leading 1)
    public let numPublicInputs: Int
    /// Total number of variables (including the leading 1)
    public var numVariables: Int { A.cols }
    /// Number of constraints
    public var numConstraints: Int { A.rows }
    /// Number of witness (private) variables
    public var numWitness: Int { numVariables - 1 - numPublicInputs }

    public init(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix, numPublicInputs: Int) {
        precondition(A.rows == B.rows && B.rows == C.rows, "Row count mismatch")
        precondition(A.cols == B.cols && B.cols == C.cols, "Column count mismatch")
        self.A = A
        self.B = B
        self.C = C
        self.numPublicInputs = numPublicInputs
    }

    /// Check satisfaction: for each row i, (A*z)[i] * (B*z)[i] == (C*z)[i]
    public func isSatisfied(z: [Fr]) -> Bool {
        precondition(z.count == numVariables)
        let az = A.mulVec(z)
        let bz = B.mulVec(z)
        let cz = C.mulVec(z)
        for i in 0..<numConstraints {
            let lhs = frMul(az[i], bz[i])
            if lhs != cz[i] { return false }
        }
        return true
    }

    /// Convert to CCS for use with folding schemes.
    public func toCCS() -> CCSInstance {
        CCSInstance.fromR1CS(A: A, B: B, C: C, numPublicInputs: numPublicInputs)
    }

    /// Summary string
    public var summary: String {
        let nnz = A.nnz + B.nnz + C.nnz
        return "R1CS: \(numConstraints) constraints, \(numVariables) vars (\(numPublicInputs) public, \(numWitness) witness), \(nnz) nnz"
    }
}

// MARK: - R1CS Optimizer

/// R1CS-specific optimization passes that exploit bilinear structure.
public enum R1CSOptimizer {

    // MARK: - Linear Constraint Elimination

    /// Eliminate variables that appear in exactly one linear (degree-1) constraint.
    ///
    /// A constraint is "linear" if one of A or B is the identity row (all-ones column 0).
    /// Specifically, if row i has A[i,:] = e_0 (selects z[0]=1) or B[i,:] = e_0,
    /// then the constraint degenerates to: 1 * (B*z)[i] = (C*z)[i], i.e., a linear
    /// relation among variables.
    ///
    /// When a witness variable w_j appears in only one such linear constraint,
    /// we can solve for w_j in terms of other variables and substitute it everywhere,
    /// eliminating both the constraint and the variable.
    ///
    /// Protected variables (column 0 = constant 1, and public input columns) are never eliminated.
    public static func eliminateLinearConstraints(r1cs: R1CSSystem) -> R1CSSystem {
        let m = r1cs.numConstraints
        let n = r1cs.numVariables
        let protectedCols = 1 + r1cs.numPublicInputs  // columns 0..protectedCols-1 are protected

        // Find linear constraints: row i where A[i,:] has exactly one nonzero at col 0 with value 1
        // (meaning A*z = z[0] = 1, so constraint becomes B*z = C*z, a linear relation)
        var linearRows = Set<Int>()
        for i in 0..<m {
            if isUnitRow(r1cs.A, row: i, col: 0) || isUnitRow(r1cs.B, row: i, col: 0) {
                linearRows.insert(i)
            }
        }

        if linearRows.isEmpty { return r1cs }

        // For each linear constraint, find witness variables that appear in only that constraint
        // across ALL of A, B, C
        var varConstraintCount = [Int: Set<Int>]()  // variable -> set of constraint rows using it
        countVariableUsage(r1cs.A, into: &varConstraintCount)
        countVariableUsage(r1cs.B, into: &varConstraintCount)
        countVariableUsage(r1cs.C, into: &varConstraintCount)

        // Find eliminatable pairs: (constraint_row, variable_col) where:
        //   - constraint is linear
        //   - variable is a witness variable (col >= protectedCols)
        //   - variable appears in exactly one constraint
        var eliminatedRows = Set<Int>()
        var eliminatedCols = Set<Int>()

        for row in linearRows {
            // Get variables in this constraint
            let varsInRow = variablesInRow(r1cs, row: row)
            for col in varsInRow {
                if col < protectedCols { continue }  // protected
                if eliminatedCols.contains(col) { continue }
                guard let usageSet = varConstraintCount[col] else { continue }
                if usageSet.count == 1 && usageSet.contains(row) {
                    // This variable appears only in this linear constraint: eliminate
                    eliminatedRows.insert(row)
                    eliminatedCols.insert(col)
                    break  // one elimination per constraint
                }
            }
        }

        if eliminatedRows.isEmpty { return r1cs }

        // Build new matrices without eliminated rows
        let keepRows = (0..<m).filter { !eliminatedRows.contains($0) }
        let newA = extractRows(r1cs.A, rows: keepRows)
        let newB = extractRows(r1cs.B, rows: keepRows)
        let newC = extractRows(r1cs.C, rows: keepRows)

        return R1CSSystem(A: newA, B: newB, C: newC, numPublicInputs: r1cs.numPublicInputs)
    }

    // MARK: - Witness Size Reduction

    /// Reduce witness size by back-substituting intermediate variables.
    ///
    /// Identifies witness variables that are fully determined by a single constraint
    /// (they appear in exactly one row of C with coefficient 1, and that row's
    /// A and B entries don't reference the same variable). These intermediate
    /// variables can be recomputed from other variables, so they don't need to
    /// be in the witness.
    ///
    /// This pass removes the corresponding constraints and returns a system with
    /// fewer constraints. The caller can recompute eliminated witness values
    /// from the remaining witness using the substitution map.
    public static func reduceWitnessSize(r1cs: R1CSSystem) -> (R1CSSystem, substitutions: [Int: SubstitutionRule]) {
        let m = r1cs.numConstraints
        let protectedCols = 1 + r1cs.numPublicInputs

        // Find variables that appear as sole output in exactly one C row
        var cColToRows = [Int: [Int]]()  // col -> list of rows where it appears in C
        for i in 0..<m {
            let start = r1cs.C.rowPtr[i]
            let end = r1cs.C.rowPtr[i + 1]
            for j in start..<end {
                let col = r1cs.C.colIdx[j]
                cColToRows[col, default: []].append(i)
            }
        }

        var substitutions = [Int: SubstitutionRule]()
        var eliminatedRows = Set<Int>()

        for (col, rows) in cColToRows {
            if col < protectedCols { continue }  // don't eliminate public vars
            if rows.count != 1 { continue }  // must appear in exactly one C row
            let row = rows[0]
            if eliminatedRows.contains(row) { continue }

            // Check that C[row, col] is the only nonzero in this row of C
            let cStart = r1cs.C.rowPtr[row]
            let cEnd = r1cs.C.rowPtr[row + 1]
            if cEnd - cStart != 1 { continue }  // must be sole entry
            let cVal = r1cs.C.values[cStart]
            if cVal != Fr.one { continue }  // coefficient must be 1

            // Check that this variable doesn't appear in A or B for this row
            let aVars = variablesInMatrixRow(r1cs.A, row: row)
            let bVars = variablesInMatrixRow(r1cs.B, row: row)
            if aVars.contains(col) || bVars.contains(col) { continue }

            // This variable is determined: z[col] = (A[row,:]*z) * (B[row,:]*z)
            // Record the substitution rule
            let aTerms = extractRowTerms(r1cs.A, row: row)
            let bTerms = extractRowTerms(r1cs.B, row: row)
            substitutions[col] = SubstitutionRule(
                eliminatedVar: col,
                aTerms: aTerms,
                bTerms: bTerms
            )
            eliminatedRows.insert(row)
        }

        if eliminatedRows.isEmpty {
            return (r1cs, substitutions: [:])
        }

        let keepRows = (0..<m).filter { !eliminatedRows.contains($0) }
        let newA = extractRows(r1cs.A, rows: keepRows)
        let newB = extractRows(r1cs.B, rows: keepRows)
        let newC = extractRows(r1cs.C, rows: keepRows)

        let reduced = R1CSSystem(A: newA, B: newB, C: newC, numPublicInputs: r1cs.numPublicInputs)
        return (reduced, substitutions: substitutions)
    }

    // MARK: - Internal Helpers

    /// Check if a sparse matrix row has exactly one nonzero entry at the given column, with value 1.
    private static func isUnitRow(_ mat: SparseMatrix, row: Int, col: Int) -> Bool {
        let start = mat.rowPtr[row]
        let end = mat.rowPtr[row + 1]
        if end - start != 1 { return false }
        return mat.colIdx[start] == col && mat.values[start] == Fr.one
    }

    /// Count which constraints each variable appears in.
    private static func countVariableUsage(_ mat: SparseMatrix, into counts: inout [Int: Set<Int>]) {
        for row in 0..<mat.rows {
            let start = mat.rowPtr[row]
            let end = mat.rowPtr[row + 1]
            for j in start..<end {
                let col = mat.colIdx[j]
                counts[col, default: Set<Int>()].insert(row)
            }
        }
    }

    /// Get all variable columns referenced by a given row across A, B, C.
    private static func variablesInRow(_ r1cs: R1CSSystem, row: Int) -> Set<Int> {
        var vars = Set<Int>()
        for mat in [r1cs.A, r1cs.B, r1cs.C] {
            let start = mat.rowPtr[row]
            let end = mat.rowPtr[row + 1]
            for j in start..<end {
                vars.insert(mat.colIdx[j])
            }
        }
        return vars
    }

    /// Get variable columns in a single matrix row.
    private static func variablesInMatrixRow(_ mat: SparseMatrix, row: Int) -> Set<Int> {
        var vars = Set<Int>()
        let start = mat.rowPtr[row]
        let end = mat.rowPtr[row + 1]
        for j in start..<end {
            vars.insert(mat.colIdx[j])
        }
        return vars
    }

    /// Extract (column, value) pairs for a matrix row.
    private static func extractRowTerms(_ mat: SparseMatrix, row: Int) -> [(col: Int, value: Fr)] {
        let start = mat.rowPtr[row]
        let end = mat.rowPtr[row + 1]
        var terms = [(col: Int, value: Fr)]()
        terms.reserveCapacity(end - start)
        for j in start..<end {
            terms.append((col: mat.colIdx[j], value: mat.values[j]))
        }
        return terms
    }

    /// Extract a subset of rows from a sparse matrix, producing a new matrix.
    private static func extractRows(_ mat: SparseMatrix, rows: [Int]) -> SparseMatrix {
        let newRows = rows.count
        var builder = SparseMatrixBuilder(rows: newRows, cols: mat.cols)
        for (newRow, oldRow) in rows.enumerated() {
            let start = mat.rowPtr[oldRow]
            let end = mat.rowPtr[oldRow + 1]
            for j in start..<end {
                builder.set(row: newRow, col: mat.colIdx[j], value: mat.values[j])
            }
        }
        return builder.build()
    }
}

// MARK: - Substitution Rule

/// Describes how an eliminated variable can be recomputed:
///   z[eliminatedVar] = (sum of aTerms[i].value * z[aTerms[i].col])
///                    * (sum of bTerms[i].value * z[bTerms[i].col])
public struct SubstitutionRule {
    /// The variable column index that was eliminated
    public let eliminatedVar: Int
    /// Terms from the A row: z[eliminatedVar] = (A_terms dot z) * (B_terms dot z)
    public let aTerms: [(col: Int, value: Fr)]
    /// Terms from the B row
    public let bTerms: [(col: Int, value: Fr)]

    /// Recompute the eliminated variable's value from a partial witness.
    public func evaluate(z: [Fr]) -> Fr {
        var aSum = Fr.zero
        for (col, val) in aTerms {
            aSum = frAdd(aSum, frMul(val, z[col]))
        }
        var bSum = Fr.zero
        for (col, val) in bTerms {
            bSum = frAdd(bSum, frMul(val, z[col]))
        }
        return frMul(aSum, bSum)
    }
}

// MARK: - R1CS from ConstraintSystem

extension R1CSSystem {
    /// Build an R1CS system from a ConstraintSystem that uses only multiplication constraints.
    /// Each constraint of the form `a * b - c == 0` maps to one R1CS row.
    ///
    /// For general constraint systems, use `CCSInstance.fromConstraintSystem` instead.
    public static func fromConstraintSystem(_ cs: ConstraintSystem, numPublicInputs: Int = 0) -> R1CSSystem? {
        let numCons = cs.constraints.count
        let numVars = 1 + cs.numWires  // +1 for the constant column z[0]=1

        var aBuilder = SparseMatrixBuilder(rows: numCons, cols: numVars)
        var bBuilder = SparseMatrixBuilder(rows: numCons, cols: numVars)
        var cBuilder = SparseMatrixBuilder(rows: numCons, cols: numVars)

        for (i, constraint) in cs.constraints.enumerated() {
            let expr = constraint.expr.constantFolded()

            // Try to decompose as a*b - c = 0
            guard let (aTerms, bTerms, cTerms) = decomposeR1CS(expr) else {
                return nil  // Not decomposable to R1CS
            }

            for (col, val) in aTerms {
                aBuilder.set(row: i, col: col, value: val)
            }
            for (col, val) in bTerms {
                bBuilder.set(row: i, col: col, value: val)
            }
            for (col, val) in cTerms {
                cBuilder.set(row: i, col: col, value: val)
            }
        }

        return R1CSSystem(
            A: aBuilder.build(),
            B: bBuilder.build(),
            C: cBuilder.build(),
            numPublicInputs: numPublicInputs
        )
    }

    /// Try to decompose an expression into R1CS form: (A_terms) * (B_terms) = C_terms
    /// Returns (A_entries, B_entries, C_entries) where each entry is (column_index, value).
    /// Column 0 is the constant column (z[0]=1), columns 1..numWires are wire values.
    private static func decomposeR1CS(_ expr: Expr) -> ([(Int, Fr)], [(Int, Fr)], [(Int, Fr)])? {
        // Handle: mul(a_expr, b_expr) - c_wire = 0
        // i.e., add(mul(a,b), neg(wire(c)))
        switch expr {
        case .add(.mul(let a, let b), .neg(.wire(let c))):
            guard let aTerms = linearTerms(a), let bTerms = linearTerms(b) else { return nil }
            let cTerms = [(1 + c.index, Fr.one)]
            return (aTerms, bTerms, cTerms)

        case .mul(let a, let b):
            // mul(a, b) = 0, meaning A*B = 0 => C = 0
            guard let aTerms = linearTerms(a), let bTerms = linearTerms(b) else { return nil }
            return (aTerms, bTerms, [])

        default:
            // Try as linear constraint: expr = 0 => 1 * expr = 0
            // A = [1 at col 0], B = linear terms, C = 0
            guard let terms = linearTerms(expr) else { return nil }
            return ([(0, Fr.one)], terms, [])
        }
    }

    /// Extract linear terms from an expression: returns [(column, coefficient)] or nil if non-linear.
    /// Column 0 = constant, column 1+i = wire i.
    private static func linearTerms(_ expr: Expr) -> [(Int, Fr)]? {
        switch expr {
        case .wire(let w):
            return [(1 + w.index, Fr.one)]
        case .constant(let c):
            return [(0, c)]
        case .add(let a, let b):
            guard let ta = linearTerms(a), let tb = linearTerms(b) else { return nil }
            return ta + tb
        case .neg(let a):
            guard let ta = linearTerms(a) else { return nil }
            return ta.map { ($0.0, frNeg($0.1)) }
        case .mul(.constant(let c), let inner):
            guard let ti = linearTerms(inner) else { return nil }
            return ti.map { ($0.0, frMul(c, $0.1)) }
        case .mul(let inner, .constant(let c)):
            guard let ti = linearTerms(inner) else { return nil }
            return ti.map { ($0.0, frMul(c, $0.1)) }
        default:
            return nil  // non-linear
        }
    }
}
