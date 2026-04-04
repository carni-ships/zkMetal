// Customizable Constraint System (CCS) — generalization of R1CS, Plonkish, AIR
// CCS instance: (M_1,...,M_t, q, d, S_1,...,S_q, c_1,...,c_q)
// Satisfying witness z: sum_j c_j * (prod_{i in S_j} M_i * z) = 0

import Foundation

// MARK: - Sparse Matrix (CSR format)

/// Sparse matrix in Compressed Sparse Row format over BN254 Fr.
public struct SparseMatrix {
    public let rows: Int
    public let cols: Int
    public let rowPtr: [Int]    // length = rows + 1
    public let colIdx: [Int]    // length = nnz
    public let values: [Fr]     // length = nnz

    public var nnz: Int { values.count }

    public init(rows: Int, cols: Int, rowPtr: [Int], colIdx: [Int], values: [Fr]) {
        precondition(rowPtr.count == rows + 1)
        precondition(colIdx.count == values.count)
        self.rows = rows
        self.cols = cols
        self.rowPtr = rowPtr
        self.colIdx = colIdx
        self.values = values
    }

    /// Matrix-vector multiply: result = M * z
    public func mulVec(_ z: [Fr]) -> [Fr] {
        precondition(z.count == cols, "Vector length \(z.count) != matrix cols \(cols)")
        var result = [Fr](repeating: .zero, count: rows)
        for i in 0..<rows {
            var acc = Fr.zero
            for k in rowPtr[i]..<rowPtr[i + 1] {
                acc = frAdd(acc, frMul(values[k], z[colIdx[k]]))
            }
            result[i] = acc
        }
        return result
    }

    /// Create an identity matrix of given size.
    public static func identity(_ n: Int) -> SparseMatrix {
        var rowPtr = [Int](repeating: 0, count: n + 1)
        var colIdx = [Int](repeating: 0, count: n)
        var values = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            rowPtr[i] = i
            colIdx[i] = i
        }
        rowPtr[n] = n
        return SparseMatrix(rows: n, cols: n, rowPtr: rowPtr, colIdx: colIdx, values: values)
    }

    /// Create a zero matrix.
    public static func zeros(rows: Int, cols: Int) -> SparseMatrix {
        SparseMatrix(rows: rows, cols: cols,
                     rowPtr: [Int](repeating: 0, count: rows + 1),
                     colIdx: [], values: [])
    }
}

// MARK: - Sparse Matrix Builder

/// Incremental builder for CSR sparse matrices.
public struct SparseMatrixBuilder {
    public let rows: Int
    public let cols: Int
    private var entries: [(row: Int, col: Int, val: Fr)]

    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.entries = []
    }

    public mutating func set(row: Int, col: Int, value: Fr) {
        precondition(row < rows && col < cols)
        entries.append((row, col, value))
    }

    public func build() -> SparseMatrix {
        // Sort by row then col
        let sorted = entries.sorted { a, b in
            a.row != b.row ? a.row < b.row : a.col < b.col
        }
        var rowPtr = [Int](repeating: 0, count: rows + 1)
        var colIdx = [Int]()
        var values = [Fr]()
        colIdx.reserveCapacity(sorted.count)
        values.reserveCapacity(sorted.count)

        var idx = 0
        for r in 0..<rows {
            rowPtr[r] = colIdx.count
            while idx < sorted.count && sorted[idx].row == r {
                colIdx.append(sorted[idx].col)
                values.append(sorted[idx].val)
                idx += 1
            }
        }
        rowPtr[rows] = colIdx.count
        return SparseMatrix(rows: rows, cols: cols, rowPtr: rowPtr, colIdx: colIdx, values: values)
    }
}

// MARK: - CCS Instance

/// Customizable Constraint System.
///
/// A CCS instance is satisfied by witness z if:
///   sum_{j=0}^{q-1} c_j * hadamard(M_{S_j[0]} * z, M_{S_j[1]} * z, ...) = 0
///
/// where hadamard is element-wise product.
public struct CCSInstance {
    public let m: Int                   // number of constraints
    public let n: Int                   // number of variables (1 + public + witness)
    public let t: Int                   // number of matrices
    public let matrices: [SparseMatrix] // M_1, ..., M_t
    public let multisets: [[Int]]       // S_1, ..., S_q (indices into matrices, 0-based)
    public let coefficients: [Fr]       // c_1, ..., c_q
    public let numPublicInputs: Int     // number of public input elements

    public var q: Int { multisets.count }
    public var d: Int { multisets.map(\.count).max() ?? 0 }

    public init(m: Int, n: Int, matrices: [SparseMatrix], multisets: [[Int]],
                coefficients: [Fr], numPublicInputs: Int = 0) {
        precondition(multisets.count == coefficients.count)
        precondition(matrices.allSatisfy { $0.rows == m && $0.cols == n })
        self.m = m
        self.n = n
        self.t = matrices.count
        self.matrices = matrices
        self.multisets = multisets
        self.coefficients = coefficients
        self.numPublicInputs = numPublicInputs
    }

    /// Check whether z satisfies the CCS constraints.
    /// z = [1, x_1, ..., x_l, w_1, ..., w_{n-l-1}]
    public func isSatisfied(z: [Fr]) -> Bool {
        precondition(z.count == n, "z length \(z.count) != n \(n)")

        // Accumulator: sum of c_j * hadamard(M_{S_j} * z)
        var acc = [Fr](repeating: .zero, count: m)

        for j in 0..<q {
            let sj = multisets[j]
            // Compute hadamard product of M_i * z for i in S_j
            var hadamard = [Fr](repeating: Fr.one, count: m)
            for matIdx in sj {
                let mv = matrices[matIdx].mulVec(z)
                for i in 0..<m {
                    hadamard[i] = frMul(hadamard[i], mv[i])
                }
            }
            // acc += c_j * hadamard
            for i in 0..<m {
                acc[i] = frAdd(acc[i], frMul(coefficients[j], hadamard[i]))
            }
        }

        return acc.allSatisfy(\.isZero)
    }

    /// Compute the j-th CCS term: c_j * hadamard(M_{S_j} * z)
    public func computeTerm(j: Int, z: [Fr]) -> [Fr] {
        let sj = multisets[j]
        var hadamard = [Fr](repeating: Fr.one, count: m)
        for matIdx in sj {
            let mv = matrices[matIdx].mulVec(z)
            for i in 0..<m {
                hadamard[i] = frMul(hadamard[i], mv[i])
            }
        }
        return hadamard.map { frMul(coefficients[j], $0) }
    }
}

// MARK: - R1CS to CCS Conversion

extension CCSInstance {
    /// Convert R1CS (A * z . B * z = C * z) to CCS.
    ///
    /// CCS with t=3, q=2:
    ///   S_0 = {0, 1} (A, B), c_0 = 1
    ///   S_1 = {2}     (C),    c_1 = -1
    ///   Constraint: 1*(A*z . B*z) + (-1)*(C*z) = 0
    public static func fromR1CS(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix,
                                numPublicInputs: Int = 0) -> CCSInstance {
        precondition(A.rows == B.rows && B.rows == C.rows)
        precondition(A.cols == B.cols && B.cols == C.cols)

        let frMinusOne = frSub(Fr.zero, Fr.one)

        return CCSInstance(
            m: A.rows,
            n: A.cols,
            matrices: [A, B, C],
            multisets: [[0, 1], [2]],
            coefficients: [Fr.one, frMinusOne],
            numPublicInputs: numPublicInputs
        )
    }
}

// MARK: - Fr helpers

/// Negate a field element: -a mod r
public func frNeg(_ a: Fr) -> Fr {
    if a.isZero { return a }
    return frSub(Fr.zero, a)
}

/// Double a field element
public func frDouble(_ a: Fr) -> Fr {
    frAdd(a, a)
}

/// Equality check for Fr
public func frEq(_ a: Fr, _ b: Fr) -> Bool {
    let al = a.to64(), bl = b.to64()
    return al[0] == bl[0] && al[1] == bl[1] && al[2] == bl[2] && al[3] == bl[3]
}
