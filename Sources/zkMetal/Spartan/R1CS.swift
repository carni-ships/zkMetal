// R1CS — Rank-1 Constraint System representation for Spartan
//
// An R1CS instance encodes: A*z . B*z = C*z (Hadamard product)
// where z = (1, public_inputs, witness) and A,B,C are sparse matrices.
//
// The sparse representation stores only nonzero entries for efficiency.

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
///   - index 0: constant 1
///   - indices 1..numPublic: public inputs
///   - indices numPublic+1..numVariables-1: private witness
public struct R1CSInstance {
    public let numConstraints: Int   // m (rows of A,B,C)
    public let numVariables: Int     // n (columns = 1 + public + witness)
    public let numPublic: Int        // number of public input variables
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
        precondition(z.count == numVariables, "z must have \(numVariables) entries")

        let az = sparseMatVec(A, z: z, numRows: numConstraints)
        let bz = sparseMatVec(B, z: z, numRows: numConstraints)
        let cz = sparseMatVec(C, z: z, numRows: numConstraints)

        for i in 0..<numConstraints {
            let prod = frMul(az[i], bz[i])
            if frToInt(prod) != frToInt(cz[i]) {
                return false
            }
        }
        return true
    }

    /// Build the full z vector from public inputs and witness.
    /// z = [1, publicInputs..., witness...]
    public static func buildZ(publicInputs: [Fr], witness: [Fr]) -> [Fr] {
        var z = [Fr.one]
        z.append(contentsOf: publicInputs)
        z.append(contentsOf: witness)
        return z
    }

    // MARK: - Multilinear Extensions

    /// Pad constraint/variable counts to next power of two.
    public var logM: Int { ceilLog2(numConstraints) }
    public var logN: Int { ceilLog2(numVariables) }
    public var paddedM: Int { 1 << logM }
    public var paddedN: Int { 1 << logN }

    /// Compute the multilinear extension of a sparse matrix evaluated at (tau, x).
    /// M_tilde(tau, x) = sum_{i,j} M[i,j] * eq(tau, bin(i)) * eq(x, bin(j))
    /// But we compute: sum_x M_tilde(tau, x) * z_tilde(x) via sparse matrix-vector.
    ///
    /// For Spartan, we need: val_a = sum_x A_tilde(tau,x) * z_tilde(x)
    /// = sum_{i,j} A[i,j] * eq(tau, bin(i)) * z[j]
    /// = sum_{(i,j) in A} A[i,j] * eq_tau[i] * z[j]
    public static func sparseInnerProduct(matrix: [SparseEntry], eqTau: [Fr], z: [Fr]) -> Fr {
        var result = Fr.zero
        for entry in matrix {
            guard entry.row < eqTau.count && entry.col < z.count else { continue }
            let term = frMul(entry.value, frMul(eqTau[entry.row], z[entry.col]))
            result = frAdd(result, term)
        }
        return result
    }

    /// Build the evaluations of z_tilde over the boolean hypercube {0,1}^logN.
    /// z_tilde is the multilinear extension of z (padded to 2^logN).
    public func buildZTilde(z: [Fr]) -> [Fr] {
        let paddedSize = paddedN
        var zPadded = z
        while zPadded.count < paddedSize {
            zPadded.append(Fr.zero)
        }
        return zPadded
    }
}

// MARK: - R1CS Builder Helpers

/// Helper to build R1CS instances from simple arithmetic constraints.
public class R1CSBuilder {
    public var numPublic: Int = 0
    public var nextVariable: Int = 1  // 0 is reserved for constant 1
    public var constraintsA: [SparseEntry] = []
    public var constraintsB: [SparseEntry] = []
    public var constraintsC: [SparseEntry] = []
    public var numConstraints: Int = 0

    public init() {}

    /// Allocate a public input variable.
    public func addPublicInput() -> Int {
        let idx = nextVariable
        nextVariable += 1
        numPublic += 1
        return idx
    }

    /// Allocate a private witness variable.
    public func addWitness() -> Int {
        let idx = nextVariable
        nextVariable += 1
        return idx
    }

    /// Add constraint: (sum_i a_i * z[a_idx_i]) * (sum_j b_j * z[b_idx_j]) = (sum_k c_k * z[c_idx_k])
    public func addConstraint(a: [(Int, Fr)], b: [(Int, Fr)], c: [(Int, Fr)]) {
        let row = numConstraints
        for (col, val) in a {
            constraintsA.append(SparseEntry(row: row, col: col, value: val))
        }
        for (col, val) in b {
            constraintsB.append(SparseEntry(row: row, col: col, value: val))
        }
        for (col, val) in c {
            constraintsC.append(SparseEntry(row: row, col: col, value: val))
        }
        numConstraints += 1
    }

    /// Convenience: constrain out = a * b
    public func mulGate(a: Int, b: Int, out: Int) {
        addConstraint(
            a: [(a, Fr.one)],
            b: [(b, Fr.one)],
            c: [(out, Fr.one)]
        )
    }

    /// Convenience: constrain out = a + b (expressed as (a + b) * 1 = out)
    public func addGate(a: Int, b: Int, out: Int) {
        addConstraint(
            a: [(a, Fr.one), (b, Fr.one)],
            b: [(0, Fr.one)],   // constant 1
            c: [(out, Fr.one)]
        )
    }

    /// Convenience: constrain out = a + constant (expressed as (a + c*1) * 1 = out)
    public func addConstGate(a: Int, constant: Fr, out: Int) {
        addConstraint(
            a: [(a, Fr.one), (0, constant)],
            b: [(0, Fr.one)],
            c: [(out, Fr.one)]
        )
    }

    /// Build the R1CS instance.
    public func build() -> R1CSInstance {
        R1CSInstance(
            numConstraints: numConstraints,
            numVariables: nextVariable,
            numPublic: numPublic,
            A: constraintsA,
            B: constraintsB,
            C: constraintsC
        )
    }
}

// MARK: - Example Circuits

extension R1CSBuilder {
    /// Build R1CS for: x^2 + x + 5 = y
    /// Variables: z = [1, y, x, v1, v2]
    ///   v1 = x * x
    ///   v2 = v1 + x  (= x^2 + x)
    ///   y = v2 + 5
    public static func exampleQuadratic() -> (R1CSInstance, (Fr) -> (publicInputs: [Fr], witness: [Fr])) {
        let builder = R1CSBuilder()
        let y = builder.addPublicInput()   // index 1
        let x = builder.addWitness()       // index 2
        let v1 = builder.addWitness()      // index 3: x*x
        let v2 = builder.addWitness()      // index 4: x^2 + x

        // v1 = x * x
        builder.mulGate(a: x, b: x, out: v1)

        // v2 = v1 + x
        builder.addGate(a: v1, b: x, out: v2)

        // y = v2 + 5  =>  (v2 + 5*1) * 1 = y
        builder.addConstGate(a: v2, constant: frFromInt(5), out: y)

        let instance = builder.build()

        let witnessGen: (Fr) -> (publicInputs: [Fr], witness: [Fr]) = { xVal in
            let v1Val = frMul(xVal, xVal)
            let v2Val = frAdd(v1Val, xVal)
            let yVal = frAdd(v2Val, frFromInt(5))
            return (publicInputs: [yVal], witness: [xVal, v1Val, v2Val])
        }

        return (instance, witnessGen)
    }

    /// Build a synthetic R1CS with `numConstraints` multiply gates.
    /// Useful for benchmarking at scale.
    public static func syntheticR1CS(numConstraints n: Int) -> (R1CSInstance, [Fr], [Fr]) {
        let builder = R1CSBuilder()
        let y = builder.addPublicInput()  // final output

        // Chain of multiplications: v[i] = v[i-1] * v[i-1]
        var vars = [Int]()
        let x = builder.addWitness()
        vars.append(x)

        for _ in 0..<n {
            let out = builder.addWitness()
            let prev = vars.last!
            builder.mulGate(a: prev, b: prev, out: out)
            vars.append(out)
        }

        // Final constraint: last var = y
        let last = vars.last!
        builder.addConstraint(
            a: [(last, Fr.one)],
            b: [(0, Fr.one)],
            c: [(y, Fr.one)]
        )

        let instance = builder.build()

        // Generate witness for x = 2
        let xVal = frFromInt(2)
        var values = [Fr]()
        values.append(xVal)
        for i in 0..<n {
            values.append(frMul(values[i], values[i]))
        }
        let yVal = values.last!

        // witness = [x, v1, v2, ..., vn]
        let publicInputs = [yVal]
        let witness = values

        return (instance, publicInputs, witness)
    }
}

// MARK: - Helpers

func sparseMatVec(_ matrix: [SparseEntry], z: [Fr], numRows: Int) -> [Fr] {
    var result = [Fr](repeating: Fr.zero, count: numRows)
    for entry in matrix {
        guard entry.row < numRows && entry.col < z.count else { continue }
        let term = frMul(entry.value, z[entry.col])
        result[entry.row] = frAdd(result[entry.row], term)
    }
    return result
}

func ceilLog2(_ n: Int) -> Int {
    if n <= 1 { return 0 }
    return Int((Double(n - 1)).logBase2().rounded(.up))
}

private extension Double {
    func logBase2() -> Double {
        return log2(self)
    }
}
