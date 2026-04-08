// R1CS — Rank-1 Constraint System representation for Spartan
//
// An R1CS instance encodes: A*z . B*z = C*z (Hadamard product)
// where z = (1, public_inputs, witness) and A,B,C are sparse matrices.

import Foundation
import NeonFieldOps

// MARK: - Sparse Matrix Types

/// A single nonzero entry in a sparse matrix.
public struct SpartanEntry {
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
public struct SpartanR1CS {
    public let numConstraints: Int
    public let numVariables: Int
    public let numPublic: Int
    public let A: [SpartanEntry]
    public let B: [SpartanEntry]
    public let C: [SpartanEntry]

    public init(numConstraints: Int, numVariables: Int, numPublic: Int,
                A: [SpartanEntry], B: [SpartanEntry], C: [SpartanEntry]) {
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
    public static func sparseInnerProduct(matrix: [SpartanEntry], eqTau: [Fr], z: [Fr]) -> Fr {
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

public class SpartanR1CSBuilder {
    public var numPublic: Int = 0
    public var nextVariable: Int = 1  // 0 reserved for constant 1
    public var constraintsA: [SpartanEntry] = []
    public var constraintsB: [SpartanEntry] = []
    public var constraintsC: [SpartanEntry] = []
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
        for (col, val) in a { constraintsA.append(SpartanEntry(row: row, col: col, value: val)) }
        for (col, val) in b { constraintsB.append(SpartanEntry(row: row, col: col, value: val)) }
        for (col, val) in c { constraintsC.append(SpartanEntry(row: row, col: col, value: val)) }
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

    public func build() -> SpartanR1CS {
        SpartanR1CS(numConstraints: numConstraints, numVariables: nextVariable,
                     numPublic: numPublic, A: constraintsA, B: constraintsB, C: constraintsC)
    }
}

// MARK: - Example Circuits

extension SpartanR1CSBuilder {
    /// x^2 + x + 5 = y
    public static func exampleQuadratic() -> (SpartanR1CS, (Fr) -> (publicInputs: [Fr], witness: [Fr])) {
        let b = SpartanR1CSBuilder()
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
    public static func syntheticR1CS(numConstraints n: Int) -> (SpartanR1CS, [Fr], [Fr]) {
        let b = SpartanR1CSBuilder()
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

func sparseMatVecSpartan(_ matrix: [SpartanEntry], z: [Fr], numRows: Int) -> [Fr] {
    var result = [Fr](repeating: Fr.zero, count: numRows)
    let n = matrix.count
    if n < 8192 {
        for entry in matrix {
            guard entry.row < numRows && entry.col < z.count else { continue }
            result[entry.row] = frAdd(result[entry.row], frMul(entry.value, z[entry.col]))
        }
        return result
    }
    let nThreads = min(8, ProcessInfo.processInfo.activeProcessorCount)
    let chunkSize = (n + nThreads - 1) / nThreads
    var partials = [[Fr]](repeating: [Fr](repeating: .zero, count: numRows), count: nThreads)
    DispatchQueue.concurrentPerform(iterations: nThreads) { t in
        let start = t * chunkSize
        let end = min(start + chunkSize, n)
        for idx in start..<end {
            let entry = matrix[idx]
            guard entry.row < numRows && entry.col < z.count else { continue }
            partials[t][entry.row] = frAdd(partials[t][entry.row], frMul(entry.value, z[entry.col]))
        }
    }
    for t in 0..<nThreads {
        partials[t].withUnsafeBytes { pBuf in
            result.withUnsafeMutableBytes { rBuf in
                bn254_fr_batch_add_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(numRows))
            }
        }
    }
    return result
}

func spartanCeilLog2(_ n: Int) -> Int {
    if n <= 1 { return 0 }
    var k = 0; var v = 1
    while v < n { v <<= 1; k += 1 }
    return k
}
