// CCSConstraintSystem — Unified CCS adapter for R1CS, Plonkish, and AIR
//
// Customizable Constraint Systems (CCS) generalize R1CS, Plonk, and AIR into:
//   sum_i c_i * prod_{j in S_i} M_j * z = 0
//
// This file provides:
//   - CCSConstraint: a single term (coefficient + matrix index set)
//   - CCSWitness: private witness vector
//   - Conversions from R1CS, Plonk, and AIR to CCS
//   - Satisfaction checking via C-accelerated sparse matvec
//
// All arithmetic is over BN254 Fr using CIOS Montgomery (NeonFieldOps).

import Foundation
import NeonFieldOps

// MARK: - CCS Constraint (single term specification)

/// A single CCS term: c * prod_{j in matrixIndices} (M_j * z)
/// The full CCS relation is: sum of these terms (Hadamard product per row) = 0
public struct CCSConstraint {
    /// Scalar coefficient for this term
    public let coefficient: Fr
    /// Indices into the CCSInstance.matrices array; the Hadamard product of
    /// these matvec results is scaled by `coefficient` and summed.
    public let matrixIndices: [Int]

    public init(coefficient: Fr, matrixIndices: [Int]) {
        self.coefficient = coefficient
        self.matrixIndices = matrixIndices
    }
}

// MARK: - CCS Witness

/// Private witness vector for a CCS instance.
/// The full assignment is z = [1, publicInput, witness].
public struct CCSWitness {
    /// Private portion of the witness (excludes the leading 1 and public inputs)
    public let w: [Fr]

    public init(_ w: [Fr]) {
        self.w = w
    }

    /// Build the full assignment vector z = [1, publicInput, w].
    public func buildZ(publicInput: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + w.count)
        z.append(Fr.one)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: w)
        return z
    }

    /// Build a relaxed assignment z = [u, publicInput, w] (for folded instances).
    public func buildRelaxedZ(u: Fr, publicInput: [Fr]) -> [Fr] {
        var z = [Fr]()
        z.reserveCapacity(1 + publicInput.count + w.count)
        z.append(u)
        z.append(contentsOf: publicInput)
        z.append(contentsOf: w)
        return z
    }
}

// MARK: - Satisfaction Checking

extension CCSInstance {
    /// Check CCS satisfaction given public input and private witness separately.
    /// Builds z = [1, publicInput, witness.w] and delegates to isSatisfied(z:).
    public func isSatisfied(publicInput: [Fr], witness: CCSWitness) -> Bool {
        let z = witness.buildZ(publicInput: publicInput)
        return isSatisfied(z: z)
    }

    /// Extract CCSConstraint descriptors from this instance.
    public var constraints: [CCSConstraint] {
        (0..<q).map { j in
            CCSConstraint(coefficient: coefficients[j], matrixIndices: multisets[j])
        }
    }

    /// Degree of the constraint system (max multiset size).
    public var degree: Int { d }

    /// Number of terms (q).
    public var numTerms: Int { q }
}

// MARK: - Plonk to CCS Conversion

extension CCSInstance {
    /// Convert a Plonk arithmetization to CCS.
    ///
    /// Standard Plonk gate:
    ///   qL*a + qR*b + qO*c + qM*a*b + qC = 0
    ///
    /// CCS encoding with 8 matrices, 5 terms:
    ///   Variables z = [1, a_0..a_{n-1}, b_0..b_{n-1}, c_0..c_{n-1}]
    ///   where n = number of gates.
    ///
    ///   M_qL: selector matrix for qL (diagonal with qL values, applied to a-column)
    ///   M_qR: selector matrix for qR (applied to b-column)
    ///   M_qO: selector matrix for qO (applied to c-column)
    ///   M_qM: selector matrix for qM
    ///   M_qC: selector matrix for qC (applied to constant column = z[0])
    ///   M_a:  picks out the a-wire values
    ///   M_b:  picks out the b-wire values
    ///   M_c:  picks out the c-wire values
    ///
    ///   Terms:
    ///     +1 * (M_qL * z) . (M_a * z)    — qL*a
    ///     +1 * (M_qR * z) . (M_b * z)    — qR*b
    ///     +1 * (M_qO * z) . (M_c * z)    — qO*c
    ///     +1 * (M_qM * z) . (M_a * z) . (M_b * z)  — qM*a*b
    ///     +1 * (M_qC * z)                 — qC
    ///
    /// - Parameters:
    ///   - numGates: number of Plonk gates
    ///   - qL: left selector values (length = numGates)
    ///   - qR: right selector values
    ///   - qO: output selector values
    ///   - qM: multiplication selector values
    ///   - qC: constant selector values
    ///   - numPublicInputs: number of public input wires
    /// - Returns: CCSInstance representing the Plonk circuit
    public static func fromPlonk(
        numGates: Int,
        qL: [Fr], qR: [Fr], qO: [Fr], qM: [Fr], qC: [Fr],
        numPublicInputs: Int = 0
    ) -> CCSInstance {
        precondition(qL.count == numGates)
        precondition(qR.count == numGates)
        precondition(qO.count == numGates)
        precondition(qM.count == numGates)
        precondition(qC.count == numGates)

        let n = numGates
        // z = [1, a_0..a_{n-1}, b_0..b_{n-1}, c_0..c_{n-1}]
        let numVars = 1 + 3 * n

        // Wire extraction matrices: M_a picks z[1..n], M_b picks z[n+1..2n], M_c picks z[2n+1..3n]
        let matA = buildWirePickMatrix(rows: n, cols: numVars, colOffset: 1)
        let matB = buildWirePickMatrix(rows: n, cols: numVars, colOffset: 1 + n)
        let matC = buildWirePickMatrix(rows: n, cols: numVars, colOffset: 1 + 2 * n)

        // Selector matrices: diagonal with selector values, selecting the constant column z[0]=1
        let matQL = buildSelectorMatrix(rows: n, cols: numVars, selectorVals: qL, wireOffset: 1)
        let matQR = buildSelectorMatrix(rows: n, cols: numVars, selectorVals: qR, wireOffset: 1 + n)
        let matQO = buildSelectorMatrix(rows: n, cols: numVars, selectorVals: qO, wireOffset: 1 + 2 * n)
        let matQM = buildSelectorDiag(rows: n, cols: numVars, selectorVals: qM)
        let matQC = buildConstantSelectorMatrix(rows: n, cols: numVars, selectorVals: qC)

        // Matrix indices: 0=M_a, 1=M_b, 2=M_c, 3=M_qL, 4=M_qR, 5=M_qO, 6=M_qM, 7=M_qC
        let matrices = [matA, matB, matC, matQL, matQR, matQO, matQM, matQC]

        // Terms: qL*a + qR*b + qO*c + qM*a*b + qC = 0
        // Term 0: +1 * M_qL . M_a  (Hadamard → qL[i]*a[i] per row)
        // Term 1: +1 * M_qR . M_b
        // Term 2: +1 * M_qO . M_c
        // Term 3: +1 * M_qM . M_a . M_b  (degree-3 Hadamard → qM[i]*a[i]*b[i])
        // Term 4: +1 * M_qC
        let multisets: [[Int]] = [
            [3, 0],     // qL * a
            [4, 1],     // qR * b
            [5, 2],     // qO * c
            [6, 0, 1],  // qM * a * b
            [7],        // qC
        ]
        let coefficients = [Fr.one, Fr.one, Fr.one, Fr.one, Fr.one]

        return CCSInstance(
            m: n, n: numVars,
            matrices: matrices,
            multisets: multisets,
            coefficients: coefficients,
            numPublicInputs: numPublicInputs
        )
    }
}

// MARK: - AIR to CCS Conversion

extension CCSInstance {
    /// Convert an Algebraic Intermediate Representation (AIR) to CCS.
    ///
    /// AIR constraints are of the form:
    ///   sum_k alpha_k * prod_{(col,rowOff) in P_k} trace[row+rowOff][col] = 0
    ///
    /// for each row in the execution trace.
    ///
    /// This encodes the trace as a flattened witness and builds sparse matrices
    /// that pick out the appropriate trace cells. Transition constraints reference
    /// both current-row and next-row values.
    ///
    /// - Parameters:
    ///   - traceWidth: number of columns in the execution trace
    ///   - traceLength: number of rows in the execution trace
    ///   - transitionConstraints: array of AIR constraints, each is a list of
    ///     (coefficient, [(column, rowOffset)]) terms
    ///   - numPublicInputs: number of public inputs
    /// - Returns: CCSInstance
    public static func fromAIR(
        traceWidth: Int,
        traceLength: Int,
        transitionConstraints: [(coefficient: Fr, wireSets: [(col: Int, rowOffset: Int)])],
        numPublicInputs: Int = 0
    ) -> CCSInstance {
        // z = [1, trace flattened row-major: trace[0][0], trace[0][1], ..., trace[T-1][W-1]]
        let numVars = 1 + traceWidth * traceLength
        // Transition constraints apply to rows 0..(traceLength-2) (since next-row refs)
        let m = traceLength - 1

        // Build one matrix per unique (col, rowOffset) pair
        var pairToMatIdx: [Int: Int] = [:]  // encoded pair -> matrix index
        var matrices: [SparseMatrix] = []

        func encodePair(_ col: Int, _ rowOff: Int) -> Int {
            return col * 1000 + (rowOff + 500)  // unique key assuming col<1000, |rowOff|<500
        }

        func getOrCreateMatrix(col: Int, rowOffset: Int) -> Int {
            let key = encodePair(col, rowOffset)
            if let idx = pairToMatIdx[key] { return idx }
            // Build matrix: row i maps to trace[(i+rowOffset)][col]
            var builder = SparseMatrixBuilder(rows: m, cols: numVars)
            for row in 0..<m {
                let traceRow = row + rowOffset
                guard traceRow >= 0 && traceRow < traceLength else { continue }
                let varIdx = 1 + traceRow * traceWidth + col
                builder.set(row: row, col: varIdx, value: Fr.one)
            }
            let idx = matrices.count
            matrices.append(builder.build())
            pairToMatIdx[key] = idx
            return idx
        }

        // Build multisets and coefficients from transition constraints
        var multisets: [[Int]] = []
        var coefficients: [Fr] = []

        for term in transitionConstraints {
            var matIndices: [Int] = []
            for (col, rowOff) in term.wireSets {
                matIndices.append(getOrCreateMatrix(col: col, rowOffset: rowOff))
            }
            multisets.append(matIndices)
            coefficients.append(term.coefficient)
        }

        return CCSInstance(
            m: m, n: numVars,
            matrices: matrices,
            multisets: multisets,
            coefficients: coefficients,
            numPublicInputs: numPublicInputs
        )
    }
}

// MARK: - ConstraintSystem to CCS Conversion

extension CCSInstance {
    /// Convert a ConstraintSystem (from ConstraintIR) to CCS.
    ///
    /// Each constraint expr == 0 is decomposed into a sum of multilinear terms.
    /// The expression tree is flattened: additions become separate CCS terms,
    /// multiplications increase the Hadamard degree.
    ///
    /// Only supports current-row wire references (row == 0). For cross-row
    /// constraints (AIR-style), use fromAIR instead.
    ///
    /// - Parameters:
    ///   - system: the ConstraintSystem to convert
    ///   - numRows: number of evaluation rows (trace length)
    ///   - numPublicInputs: number of public input wires
    /// - Returns: CCSInstance
    public static func fromConstraintSystem(
        _ system: ConstraintSystem,
        numRows: Int,
        numPublicInputs: Int = 0
    ) -> CCSInstance {
        precondition(!system.hasCrossRowConstraints,
                     "Use fromAIR for cross-row constraints")

        // z = [1, w_0, w_1, ..., w_{numWires-1}] per row
        let numVars = 1 + system.numWires
        let m = system.constraints.count * numRows

        // Flatten each constraint expression into sum-of-products form.
        // Each product term becomes a CCS multiset.
        var matrices: [SparseMatrix] = []
        var multisets: [[Int]] = []
        var coefficients: [Fr] = []

        // Wire extraction matrix cache: wireIdx -> matrix index
        var wireMatIdx: [Int: Int] = [:]

        func getWireMatrix(_ wireIdx: Int) -> Int {
            if let idx = wireMatIdx[wireIdx] { return idx }
            var builder = SparseMatrixBuilder(rows: m, cols: numVars)
            for row in 0..<m {
                builder.set(row: row, col: 1 + wireIdx, value: Fr.one)
            }
            let idx = matrices.count
            matrices.append(builder.build())
            wireMatIdx[wireIdx] = idx
            return idx
        }

        // Constant matrix (picks z[0] = 1)
        var constMatIdx: Int? = nil
        func getConstMatrix() -> Int {
            if let idx = constMatIdx { return idx }
            var builder = SparseMatrixBuilder(rows: m, cols: numVars)
            for row in 0..<m {
                builder.set(row: row, col: 0, value: Fr.one)
            }
            let idx = matrices.count
            matrices.append(builder.build())
            constMatIdx = idx
            return idx
        }

        // Recursively decompose expr into sum-of-products.
        // Returns array of (coefficient, [wireIndex]) terms.
        func decompose(_ expr: Expr) -> [(Fr, [Int])] {
            switch expr {
            case .wire(let w):
                return [(Fr.one, [w.index])]

            case .constant(let c):
                // c * z[0] where z[0] = 1
                return [(c, [])]

            case .add(let a, let b):
                return decompose(a) + decompose(b)

            case .neg(let a):
                return decompose(a).map { (frNeg($0.0), $0.1) }

            case .mul(let a, let b):
                let ta = decompose(a)
                let tb = decompose(b)
                var result: [(Fr, [Int])] = []
                for (ca, wa) in ta {
                    for (cb, wb) in tb {
                        result.append((frMul(ca, cb), wa + wb))
                    }
                }
                return result

            case .pow(let base, let n):
                var acc = decompose(base)
                for _ in 1..<n {
                    let base_terms = decompose(base)
                    var result: [(Fr, [Int])] = []
                    for (ca, wa) in acc {
                        for (cb, wb) in base_terms {
                            result.append((frMul(ca, cb), wa + wb))
                        }
                    }
                    acc = result
                }
                return acc
            }
        }

        for constraint in system.constraints {
            let folded = constraint.expr.constantFolded()
            let terms = decompose(folded)

            for (coeff, wires) in terms {
                if coeff.isZero { continue }

                if wires.isEmpty {
                    // Pure constant term: coeff * M_const * z
                    multisets.append([getConstMatrix()])
                    coefficients.append(coeff)
                } else {
                    var matIndices: [Int] = []
                    for wireIdx in wires {
                        matIndices.append(getWireMatrix(wireIdx))
                    }
                    multisets.append(matIndices)
                    coefficients.append(coeff)
                }
            }
        }

        return CCSInstance(
            m: m, n: numVars,
            matrices: matrices,
            multisets: multisets,
            coefficients: coefficients,
            numPublicInputs: numPublicInputs
        )
    }
}

// MARK: - Sparse Matrix Builders (internal)

/// Build an identity-like "wire pick" matrix: row i selects z[colOffset + i].
func buildWirePickMatrix(rows: Int, cols: Int, colOffset: Int) -> SparseMatrix {
    var builder = SparseMatrixBuilder(rows: rows, cols: cols)
    for i in 0..<rows {
        builder.set(row: i, col: colOffset + i, value: Fr.one)
    }
    return builder.build()
}

/// Build a selector matrix: row i has selectorVals[i] at column wireOffset+i.
/// When multiplied by z and Hadamard'd with the wire-pick result, this yields
/// selectorVals[i] * wire[i].
func buildSelectorMatrix(rows: Int, cols: Int, selectorVals: [Fr], wireOffset: Int) -> SparseMatrix {
    var builder = SparseMatrixBuilder(rows: rows, cols: cols)
    for i in 0..<rows {
        if !selectorVals[i].isZero {
            builder.set(row: i, col: wireOffset + i, value: selectorVals[i])
        }
    }
    return builder.build()
}

/// Build a diagonal selector matrix (not tied to a specific wire column).
/// Used for qM where the Hadamard product handles wire extraction.
func buildSelectorDiag(rows: Int, cols: Int, selectorVals: [Fr]) -> SparseMatrix {
    // This matrix, when mulVec'd with z, should produce selectorVals[i] for row i.
    // We achieve this by picking z[0] = 1 and scaling by selectorVals[i].
    var builder = SparseMatrixBuilder(rows: rows, cols: cols)
    for i in 0..<rows {
        if !selectorVals[i].isZero {
            builder.set(row: i, col: 0, value: selectorVals[i])
        }
    }
    return builder.build()
}

/// Build a constant selector matrix: row i picks z[0]*qC[i].
func buildConstantSelectorMatrix(rows: Int, cols: Int, selectorVals: [Fr]) -> SparseMatrix {
    var builder = SparseMatrixBuilder(rows: rows, cols: cols)
    for i in 0..<rows {
        if !selectorVals[i].isZero {
            builder.set(row: i, col: 0, value: selectorVals[i])
        }
    }
    return builder.build()
}

// MARK: - CCS Metadata

extension CCSInstance {
    /// Human-readable description of the CCS structure.
    public var summary: String {
        """
        CCS: m=\(m) constraints, n=\(n) variables, t=\(t) matrices, \
        q=\(q) terms, degree=\(d), nnz=\(matrices.map(\.nnz).reduce(0, +))
        """
    }

    /// Check structural validity (matrix dimensions, index bounds).
    public var isWellFormed: Bool {
        guard multisets.count == coefficients.count else { return false }
        for mat in matrices {
            guard mat.rows == m && mat.cols == n else { return false }
        }
        for sj in multisets {
            for idx in sj {
                guard idx >= 0 && idx < t else { return false }
            }
        }
        return true
    }
}
