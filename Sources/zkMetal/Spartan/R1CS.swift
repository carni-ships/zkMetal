// R1CS — Rank-1 Constraint System representation for Spartan
//
// An R1CS instance encodes: A*z . B*z = C*z (Hadamard product)
// where z = (1, public_inputs, witness) and A,B,C are sparse matrices.

import Foundation

// MARK: - Sparse Matrix Types

/// A single nonzero entry in a sparse matrix.
public struct SparseEntry {
    public let row: Int
    public let col: Int
    public let value: Fr

    public init(row: Int, col: Int, value: Fr) {
        self.row = row
        self.col = col
        self.value = value
    }
}

/// R1CS instance: A*z . B*z = C*z where . is Hadamard (entry-wise) product.
///
/// Variable layout in z: [1, public_inputs..., witness...]
public struct R1CSInstance {
    public let numConstraints: Int
    public let numVariables: Int
    public let numPublic: Int
    public let A: [SparseEntry]
    public let B: [SparseEntry]
    public let C: [SparseEntry]

    public init(numConstraints: Int, numVariables: Int, numPublic: Int,
                A: [SparseEntry], B: [SparseEntry], C: [SparseEntry]) {
        self.numConstraints = numConstraints
        self.numVariables = numVariables
        self.numPublic = numPublic
        self.A = A
        self.B = B
        self.C = C
    }

    /// Check if z satisfies A*z . B*z = C*z.
    public func isSatisfied(z: [Fr]) -> Bool {
        precondition(z.count == numVariables)
        let az = sparseMatVecSpartan(A, z: z, numRows: numConstraints)
        let bz = sparseMatVecSpartan(B, z: z, numRows: numConstraints)
        let cz = sparseMatVecSpartan(C, z: z, numRows: numConstraints)
        for i in 0..<numConstraints {
            let prod = frMul(az[i], bz[i])
            if frToInt(prod) != frToInt(cz[i]) { return false }
        }
        return true
    }

    /// Build the full z vector: z = [1, publicInputs..., witness...]
    public static func buildZ(publicInputs: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr.one]
        z.append(contentsOf: publicInputs)
        z.append(contentsOf: witness)
        return z
    }

    // MARK: - Multilinear Extension Helpers

    public var logM: Int { spartanCeilLog2(numConstraints) }
    public var logN: Int { spartanCeilLog2(numVariables) }
    public var paddedM: Int { 1 << logM }
    public var paddedN: Int { 1 << logN }

    /// Sparse inner product: sum_{(i,j) in M} M[i,j] * eqTau[i] * z[j]
    public static func sparseInnerProduct(matrix: [SparseEntry], eqTau: [Fr], z: [Fr]) -> Fr {
        var result = Fr.zero
        for entry in matrix {
            guard entry.row < eqTau.count && entry.col < z.count else { continue }
            let term = frMul(entry.value, frMul(eqTau[entry.row], z[entry.col]))
            result = frAdd(result, term)
        }
        return result
    }

    /// Build z_tilde: pad z to 2^logN for MLE over boolean hypercube.
    public func buildZTilde(z: [Fr]) -> [Fr] {
        var zPadded = z
        while zPadded.count < paddedN { zPadded.append(Fr.zero) }
        return zPadded
    }
}

// MARK: - R1CS Builder

public class R1CSBuilder {
    public var numPublic: Int = 0
    public var nextVariable: Int = 1  // 0 reserved for constant 1
    public var constraintsA: [SparseEntry] = []
    public var constraintsB: [SparseEntry] = []
    public var constraintsC: [SparseEntry] = []
    public var numConstraints: Int = 0

    public init() {}

    public func addPublicInput() -> Int {
        let idx = nextVariable; nextVariable += 1; numPublic += 1; return idx
    }

    public func addWitness() -> Int {
        let idx = nextVariable; nextVariable += 1; return idx
    }

    /// Add constraint: (sum a_i*z[i]) * (sum b_j*z[j]) = (sum c_k*z[k])
    public func addConstraint(a: [(Int, Fr)], b: [(Int, Fr)], c: [(Int, Fr)]) {
        let row = numConstraints
        for (col, val) in a { constraintsA.append(SparseEntry(row: row, col: col, value: val)) }
        for (col, val) in b { constraintsB.append(SparseEntry(row: row, col: col, value: val)) }
        for (col, val) in c { constraintsC.append(SparseEntry(row: row, col: col, value: val)) }
        numConstraints += 1
    }

    /// out = a * b
    public func mulGate(a: Int, b: Int, out: Int) {
        addConstraint(a: [(a, Fr.one)], b: [(b, Fr.one)], c: [(out, Fr.one)])
    }

    /// out = a + b  (expressed as (a+b)*1 = out)
    public func addGate(a: Int, b: Int, out: Int) {
        addConstraint(a: [(a, Fr.one), (b, Fr.one)], b: [(0, Fr.one)], c: [(out, Fr.one)])
    }

    /// out = a + constant
    public func addConstGate(a: Int, constant: Fr, out: Int) {
        addConstraint(a: [(a, Fr.one), (0, constant)], b: [(0, Fr.one)], c: [(out, Fr.one)])
    }

    public func build() -> R1CSInstance {
        R1CSInstance(numConstraints: numConstraints, numVariables: nextVariable,
                     numPublic: numPublic, A: constraintsA, B: constraintsB, C: constraintsC)
    }
}

// MARK: - Example Circuits

extension R1CSBuilder {
    /// x^2 + x + 5 = y
    public static func exampleQuadratic() -> (R1CSInstance, (Fr) -> (publicInputs: [Fr], witness: [Fr])) {
        let b = R1CSBuilder()
        let y = b.addPublicInput()
        let x = b.addWitness()
        let v1 = b.addWitness()
        let v2 = b.addWitness()
        b.mulGate(a: x, b: x, out: v1)
        b.addGate(a: v1, b: x, out: v2)
        b.addConstGate(a: v2, constant: frFromInt(5), out: y)
        let instance = b.build()
        let gen: (Fr) -> (publicInputs: [Fr], witness: [Fr]) = { xVal in
            let v1Val = frMul(xVal, xVal)
            let v2Val = frAdd(v1Val, xVal)
            let yVal = frAdd(v2Val, frFromInt(5))
            return ([yVal], [xVal, v1Val, v2Val])
        }
        return (instance, gen)
    }

    /// Synthetic R1CS with n multiply gates for benchmarking.
    public static func syntheticR1CS(numConstraints n: Int) -> (R1CSInstance, [Fr], [Fr]) {
        let b = R1CSBuilder()
        let y = b.addPublicInput()
        var vars = [Int]()
        let x = b.addWitness()
        vars.append(x)
        for _ in 0..<n {
            let out = b.addWitness()
            b.mulGate(a: vars.last!, b: vars.last!, out: out)
            vars.append(out)
        }
        b.addConstraint(a: [(vars.last!, Fr.one)], b: [(0, Fr.one)], c: [(y, Fr.one)])
        let instance = b.build()

        let xVal = frFromInt(2)
        var values = [Fr]()
        values.append(xVal)
        for i in 0..<n { values.append(frMul(values[i], values[i])) }
        return (instance, [values.last!], values)
    }
}

// MARK: - Helpers (prefixed to avoid conflicts)

func sparseMatVecSpartan(_ matrix: [SparseEntry], z: [Fr], numRows: Int) -> [Fr] {
    var result = [Fr](repeating: Fr.zero, count: numRows)
    for entry in matrix {
        guard entry.row < numRows && entry.col < z.count else { continue }
        result[entry.row] = frAdd(result[entry.row], frMul(entry.value, z[entry.col]))
    }
    return result
}

func spartanCeilLog2(_ n: Int) -> Int {
    if n <= 1 { return 0 }
    var k = 0; var v = 1
    while v < n { v <<= 1; k += 1 }
    return k
}
