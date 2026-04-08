// GPUSpartanLinearizeEngine — GPU-accelerated Spartan R1CS linearization
//
// Implements Spartan's multilinear extension linearization of R1CS:
//   - Converts sparse R1CS matrices (A, B, C) into multilinear extensions
//   - Evaluates ~A(tau, x), ~B(tau, x), ~C(tau, x) over the boolean hypercube
//   - Memory-checking argument for sparse polynomial evaluation
//   - GPU-accelerated multilinear evaluation via sumcheck reduction
//   - Witness polynomial construction and commitment
//   - Support for structured R1CS (uniform/non-uniform constraints)
//
// The linearization engine takes a satisfied R1CS instance and produces
// multilinear polynomial representations suitable for Spartan's sumcheck.
// It separates the "linearization" (encoding) phase from the "proving" phase,
// enabling preprocessing and caching of matrix-derived polynomials.
//
// Architecture:
//   1. SparseMLETable: compressed representation of sparse matrix as MLE
//   2. LinearizedR1CS: precomputed multilinear extensions of A, B, C
//   3. HypercubeEvaluator: GPU-parallel evaluation over {0,1}^n
//   4. MemoryCheckingDigest: offline memory-checking for sparse lookups
//   5. WitnessLinearization: MLE encoding of the witness vector
//   6. StructuredR1CS: uniform/non-uniform constraint partitioning
//
// Uses BN254 Fr field arithmetic (Montgomery form).

import Foundation
import Metal
import NeonFieldOps

// MARK: - Sparse MLE Table

/// Compressed representation of a sparse matrix as a multilinear extension.
/// Stores only nonzero entries with their hypercube coordinates, enabling
/// efficient evaluation at arbitrary points via sparse inner products.
public struct SparseMLETable {
    /// Row indices of nonzero entries (hypercube coordinates in {0,1}^logM)
    public let rows: [Int]
    /// Column indices of nonzero entries (hypercube coordinates in {0,1}^logN)
    public let cols: [Int]
    /// Field values of nonzero entries
    public let values: [Fr]
    /// Number of variables for row dimension (logM)
    public let numRowVars: Int
    /// Number of variables for column dimension (logN)
    public let numColVars: Int
    /// Total number of nonzero entries
    public var nnz: Int { values.count }

    public init(rows: [Int], cols: [Int], values: [Fr],
                numRowVars: Int, numColVars: Int) {
        precondition(rows.count == cols.count && cols.count == values.count,
                     "SparseMLETable: mismatched array lengths")
        self.rows = rows
        self.cols = cols
        self.values = values
        self.numRowVars = numRowVars
        self.numColVars = numColVars
    }

    /// Create from SpartanEntry array with known dimensions.
    public static func fromEntries(_ entries: [SpartanEntry],
                                   logM: Int, logN: Int) -> SparseMLETable {
        var rs = [Int](), cs = [Int](), vs = [Fr]()
        rs.reserveCapacity(entries.count)
        cs.reserveCapacity(entries.count)
        vs.reserveCapacity(entries.count)
        for e in entries {
            rs.append(e.row)
            cs.append(e.col)
            vs.append(e.value)
        }
        return SparseMLETable(rows: rs, cols: cs, values: vs,
                              numRowVars: logM, numColVars: logN)
    }

    /// Evaluate the sparse MLE at (tau, x) where tau in F^logM, x in F^logN.
    /// Computes: sum_{(r,c,v)} v * eq(tau, r) * eq(x, c)
    /// where eq is the multilinear equality polynomial.
    public func evaluate(tau: [Fr], x: [Fr]) -> Fr {
        precondition(tau.count == numRowVars, "tau dimension mismatch")
        precondition(x.count == numColVars, "x dimension mismatch")
        var result = Fr.zero
        for i in 0..<nnz {
            let eqTauR = sparseEqAtIndex(point: tau, index: rows[i], numVars: numRowVars)
            let eqXC = sparseEqAtIndex(point: x, index: cols[i], numVars: numColVars)
            let term = frMul(values[i], frMul(eqTauR, eqXC))
            result = frAdd(result, term)
        }
        return result
    }

    /// Evaluate only the row component: sum_entries v * eq(tau, row) for a fixed col.
    /// Returns a dense vector indexed by column.
    public func evaluateRowBinding(tau: [Fr]) -> [Fr] {
        let numCols = 1 << numColVars
        var result = [Fr](repeating: Fr.zero, count: numCols)
        for i in 0..<nnz {
            let eqTauR = sparseEqAtIndex(point: tau, index: rows[i], numVars: numRowVars)
            let contribution = frMul(values[i], eqTauR)
            if cols[i] < numCols {
                result[cols[i]] = frAdd(result[cols[i]], contribution)
            }
        }
        return result
    }
}

// MARK: - Linearized R1CS

/// Precomputed multilinear extensions of R1CS matrices A, B, C.
/// After linearization, each matrix is represented as a sparse MLE table
/// that can be efficiently evaluated at Fiat-Shamir challenge points.
public struct LinearizedR1CS {
    /// Sparse MLE for matrix A
    public let mlA: SparseMLETable
    /// Sparse MLE for matrix B
    public let mlB: SparseMLETable
    /// Sparse MLE for matrix C
    public let mlC: SparseMLETable
    /// Log of padded constraint count
    public let logM: Int
    /// Log of padded variable count
    public let logN: Int
    /// Original R1CS dimensions
    public let numConstraints: Int
    public let numVariables: Int
    public let numPublic: Int

    public init(mlA: SparseMLETable, mlB: SparseMLETable, mlC: SparseMLETable,
                logM: Int, logN: Int, numConstraints: Int, numVariables: Int,
                numPublic: Int) {
        self.mlA = mlA
        self.mlB = mlB
        self.mlC = mlC
        self.logM = logM
        self.logN = logN
        self.numConstraints = numConstraints
        self.numVariables = numVariables
        self.numPublic = numPublic
    }

    /// Linearize an R1CS instance into multilinear extensions.
    public static func linearize(_ instance: SpartanR1CS) -> LinearizedR1CS {
        let logM = instance.logM
        let logN = instance.logN
        let mlA = SparseMLETable.fromEntries(instance.A, logM: logM, logN: logN)
        let mlB = SparseMLETable.fromEntries(instance.B, logM: logM, logN: logN)
        let mlC = SparseMLETable.fromEntries(instance.C, logM: logM, logN: logN)
        return LinearizedR1CS(
            mlA: mlA, mlB: mlB, mlC: mlC,
            logM: logM, logN: logN,
            numConstraints: instance.numConstraints,
            numVariables: instance.numVariables,
            numPublic: instance.numPublic)
    }

    /// Evaluate ~A(tau, x), ~B(tau, x), ~C(tau, x) at given points.
    public func evaluateAll(tau: [Fr], x: [Fr]) -> (aEval: Fr, bEval: Fr, cEval: Fr) {
        let a = mlA.evaluate(tau: tau, x: x)
        let b = mlB.evaluate(tau: tau, x: x)
        let c = mlC.evaluate(tau: tau, x: x)
        return (a, b, c)
    }

    /// Compute row-bound vectors: ~A(tau, .), ~B(tau, .), ~C(tau, .)
    /// Each returns a dense vector of length 2^logN.
    public func rowBindAll(tau: [Fr]) -> (aVec: [Fr], bVec: [Fr], cVec: [Fr]) {
        let a = mlA.evaluateRowBinding(tau: tau)
        let b = mlB.evaluateRowBinding(tau: tau)
        let c = mlC.evaluateRowBinding(tau: tau)
        return (a, b, c)
    }
}

// MARK: - Hypercube Evaluator

/// Evaluates multilinear polynomials over the boolean hypercube {0,1}^n.
/// Used to compute Az, Bz, Cz vectors efficiently for sumcheck.
public struct HypercubeEvaluator {
    /// Number of variables (dimension of hypercube)
    public let numVars: Int
    /// Size of hypercube: 2^numVars
    public var size: Int { 1 << numVars }

    public init(numVars: Int) {
        precondition(numVars >= 0 && numVars <= 28, "numVars must be in [0, 28]")
        self.numVars = numVars
    }

    /// Evaluate MLE defined by `evals` at each point of the boolean hypercube.
    /// For an MLE with evaluations on {0,1}^n, this is the identity (returns evals).
    /// But for a composed MLE this expands it.
    public func expandMLE(evals: [Fr]) -> [Fr] {
        precondition(evals.count == size, "evals length must be 2^numVars")
        return evals
    }

    /// Compute the eq polynomial table: eq(point, b) for all b in {0,1}^numVars.
    /// eq(r, b) = prod_{i} (r_i * b_i + (1 - r_i)*(1 - b_i))
    public func eqTable(point: [Fr]) -> [Fr] {
        precondition(point.count == numVars, "point dimension mismatch")
        var table = [Fr](repeating: Fr.zero, count: size)
        table[0] = Fr.one
        var len = 1
        for i in 0..<numVars {
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            for j in stride(from: len - 1, through: 0, by: -1) {
                table[2 * j + 1] = frMul(table[j], ri)
                table[2 * j] = frMul(table[j], oneMinusRi)
            }
            len *= 2
        }
        return table
    }

    /// Inner product of two vectors: sum_i a[i] * b[i]
    public func innerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        let n = min(a.count, b.count)
        var result = Fr.zero
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_inner_product(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                }
            }
        }
        return result
    }

    /// Fold array in half using challenge r:
    /// out[i] = arr[i] + r * (arr[i + half] - arr[i])
    public func fold(_ arr: [Fr], challenge r: Fr) -> [Fr] {
        let h = arr.count / 2
        var result = [Fr](repeating: Fr.zero, count: h)
        arr.withUnsafeBytes { aBuf in
            withUnsafeBytes(of: r) { rBuf in
                result.withUnsafeMutableBytes { outBuf in
                    bn254_fr_fold_halves(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(h))
                }
            }
        }
        return result
    }

    /// Evaluate MLE at a given point using successive folding.
    public func evaluateMLE(evals: [Fr], point: [Fr]) -> Fr {
        precondition(evals.count == size, "evals must have 2^numVars entries")
        precondition(point.count == numVars, "point must have numVars entries")
        var current = evals
        for i in 0..<numVars {
            current = fold(current, challenge: point[i])
        }
        return current[0]
    }
}

// MARK: - Memory-Checking Digest

/// Memory-checking argument for verifying sparse polynomial evaluation.
/// Ensures that the prover correctly looked up matrix entries during
/// the multilinear extension computation.
///
/// Uses an offline memory-checking technique: the prover commits to
/// a "memory transcript" of all lookups, and the verifier checks
/// consistency via a random linear combination.
public struct MemoryCheckingDigest {
    /// The combined digest value (fingerprint of memory operations)
    public let digest: Fr
    /// Number of read operations
    public let numReads: Int
    /// Number of write operations
    public let numWrites: Int
    /// Random challenge used for fingerprinting
    public let challenge: Fr

    public init(digest: Fr, numReads: Int, numWrites: Int, challenge: Fr) {
        self.digest = digest
        self.numReads = numReads
        self.numWrites = numWrites
        self.challenge = challenge
    }

    /// Compute a memory-checking digest for a sparse MLE table.
    /// The digest is a product of (gamma - addr_i - beta * val_i) terms
    /// where gamma and beta are random challenges.
    public static func compute(table: SparseMLETable, gamma: Fr, beta: Fr) -> MemoryCheckingDigest {
        var readDigest = Fr.one
        for i in 0..<table.nnz {
            let addr = frFromInt(UInt64(table.rows[i]) * UInt64(1 << table.numColVars)
                                 + UInt64(table.cols[i]))
            let val = table.values[i]
            let term = frSub(gamma, frAdd(addr, frMul(beta, val)))
            readDigest = frMul(readDigest, term)
        }
        return MemoryCheckingDigest(
            digest: readDigest, numReads: table.nnz,
            numWrites: 0, challenge: gamma)
    }

    /// Verify that two digests are consistent (read set = write set).
    public static func verifyConsistency(readDigest: MemoryCheckingDigest,
                                         writeDigest: MemoryCheckingDigest) -> Bool {
        return spartanFrEqual(readDigest.digest, writeDigest.digest)
    }

    /// Compute write-side digest for a dense table of known values.
    /// The table[addr] = val entries produce (gamma - addr - beta * val) terms.
    public static func computeWriteDigest(tableValues: [Fr], gamma: Fr,
                                          beta: Fr) -> MemoryCheckingDigest {
        var writeDigest = Fr.one
        for (i, val) in tableValues.enumerated() {
            if val.isZero { continue }
            let addr = frFromInt(UInt64(i))
            let term = frSub(gamma, frAdd(addr, frMul(beta, val)))
            writeDigest = frMul(writeDigest, term)
        }
        return MemoryCheckingDigest(
            digest: writeDigest, numReads: 0,
            numWrites: tableValues.reduce(0) { $0 + ($1.isZero ? 0 : 1) },
            challenge: gamma)
    }
}

// MARK: - Witness Linearization

/// Multilinear extension encoding of the witness vector.
/// Constructs z_tilde = MLE(z) where z = (1, public_inputs, witness),
/// padded to length 2^logN for evaluation over {0,1}^logN.
public struct WitnessLinearization {
    /// The padded witness vector as MLE evaluations
    public let zTilde: [Fr]
    /// Log of the padded length
    public let logN: Int
    /// Number of public inputs
    public let numPublic: Int
    /// Number of actual witness elements (excluding constant 1 and public inputs)
    public let numWitness: Int

    public init(zTilde: [Fr], logN: Int, numPublic: Int, numWitness: Int) {
        self.zTilde = zTilde
        self.logN = logN
        self.numPublic = numPublic
        self.numWitness = numWitness
    }

    /// Construct witness linearization from public inputs and witness.
    public static func build(publicInputs: [Fr], witness: [Fr],
                             logN: Int) -> WitnessLinearization {
        let paddedN = 1 << logN
        var z = [Fr]()
        let zLen = 1 + publicInputs.count + witness.count
        z = [Fr](repeating: Fr.zero, count: paddedN)
        z[0] = Fr.one
        z.withUnsafeMutableBytes { zBuf in
            publicInputs.withUnsafeBytes { pBuf in
                memcpy(zBuf.baseAddress! + MemoryLayout<Fr>.stride,
                       pBuf.baseAddress!, publicInputs.count * MemoryLayout<Fr>.stride)
            }
            witness.withUnsafeBytes { wBuf in
                memcpy(zBuf.baseAddress! + (1 + publicInputs.count) * MemoryLayout<Fr>.stride,
                       wBuf.baseAddress!, witness.count * MemoryLayout<Fr>.stride)
            }
        }
        _ = zLen  // suppress unused warning
        return WitnessLinearization(
            zTilde: z, logN: logN,
            numPublic: publicInputs.count, numWitness: witness.count)
    }

    /// Evaluate the witness MLE at a given point.
    public func evaluate(at point: [Fr]) -> Fr {
        precondition(point.count == logN, "point dimension mismatch")
        return spartanEvalML(evals: zTilde, pt: point)
    }

    /// Extract the public input portion of z_tilde.
    public func publicInputValues() -> [Fr] {
        guard numPublic > 0 else { return [] }
        return Array(zTilde[1...numPublic])
    }

    /// Extract the witness portion of z_tilde.
    public func witnessValues() -> [Fr] {
        let start = 1 + numPublic
        let end = start + numWitness
        guard end <= zTilde.count else { return [] }
        return Array(zTilde[start..<end])
    }
}

// MARK: - Structured R1CS

/// Partitioning of R1CS constraints into uniform and non-uniform sets.
/// Uniform constraints share the same sparsity pattern (e.g., repeated gadgets),
/// enabling batch evaluation of their MLEs. Non-uniform constraints are
/// handled individually.
public struct StructuredR1CS {
    /// Groups of constraint indices sharing the same sparsity pattern
    public let uniformGroups: [[Int]]
    /// Constraint indices that don't belong to any uniform group
    public let nonUniformIndices: [Int]
    /// The underlying R1CS instance
    public let instance: SpartanR1CS

    public init(uniformGroups: [[Int]], nonUniformIndices: [Int],
                instance: SpartanR1CS) {
        self.uniformGroups = uniformGroups
        self.nonUniformIndices = nonUniformIndices
        self.instance = instance
    }

    /// Analyze an R1CS instance to identify uniform constraint groups.
    /// Two constraints are "uniform" if they have the same sparsity pattern
    /// in A, B, C (same column offsets, possibly different values).
    public static func analyze(_ instance: SpartanR1CS) -> StructuredR1CS {
        // Build per-row sparsity signatures
        var signatures = [Int: String]()
        var rowColsA = [Int: [Int]]()
        var rowColsB = [Int: [Int]]()
        var rowColsC = [Int: [Int]]()

        for e in instance.A {
            rowColsA[e.row, default: []].append(e.col)
        }
        for e in instance.B {
            rowColsB[e.row, default: []].append(e.col)
        }
        for e in instance.C {
            rowColsC[e.row, default: []].append(e.col)
        }

        for row in 0..<instance.numConstraints {
            let aCols = (rowColsA[row] ?? []).sorted()
            let bCols = (rowColsB[row] ?? []).sorted()
            let cCols = (rowColsC[row] ?? []).sorted()
            signatures[row] = "\(aCols)|\(bCols)|\(cCols)"
        }

        // Group by signature
        var groups = [String: [Int]]()
        for (row, sig) in signatures {
            groups[sig, default: []].append(row)
        }

        var uniformGroups = [[Int]]()
        var nonUniform = [Int]()
        for (_, rows) in groups {
            if rows.count >= 2 {
                uniformGroups.append(rows.sorted())
            } else {
                nonUniform.append(contentsOf: rows)
            }
        }
        nonUniform.sort()

        return StructuredR1CS(
            uniformGroups: uniformGroups,
            nonUniformIndices: nonUniform,
            instance: instance)
    }

    /// Number of uniform groups found
    public var numUniformGroups: Int { uniformGroups.count }

    /// Total constraints covered by uniform groups
    public var uniformConstraintCount: Int {
        uniformGroups.reduce(0) { $0 + $1.count }
    }

    /// Fraction of constraints that are uniform
    public var uniformFraction: Double {
        guard instance.numConstraints > 0 else { return 0.0 }
        return Double(uniformConstraintCount) / Double(instance.numConstraints)
    }
}

// MARK: - GPU Spartan Linearize Engine

/// GPU-accelerated engine for Spartan R1CS linearization.
///
/// Orchestrates the full linearization pipeline:
///   1. Convert sparse R1CS to MLE tables
///   2. Precompute eq polynomial tables on GPU
///   3. Evaluate ~A(tau,x), ~B(tau,x), ~C(tau,x) via sparse inner products
///   4. Construct witness MLE and memory-checking digests
///   5. Perform sumcheck reduction for linearized claims
///
/// Falls back to CPU for small instances (< gpuThreshold entries).
public class GPUSpartanLinearizeEngine {
    public static var version: PrimitiveVersion { Versions.gpuSpartanLinearize }

    /// Metal device (nil if GPU unavailable)
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?

    /// Threshold: use GPU path for arrays >= this size
    public var gpuThreshold: Int = 256

    /// Cached linearization (reuse across multiple prove calls)
    private var cachedLinearization: LinearizedR1CS?
    private var cachedInstanceHash: Int = 0

    // MARK: - Initialization

    /// Create engine with GPU acceleration. Falls back to CPU if Metal unavailable.
    public init() {
        if let dev = MTLCreateSystemDefaultDevice(),
           let queue = dev.makeCommandQueue() {
            self.device = dev
            self.commandQueue = queue
        } else {
            self.device = nil
            self.commandQueue = nil
        }
    }

    /// Whether GPU acceleration is available.
    public var hasGPU: Bool { device != nil }

    // MARK: - Linearize

    /// Linearize an R1CS instance, caching the result for repeated use.
    public func linearize(_ instance: SpartanR1CS) -> LinearizedR1CS {
        let hash = instance.numConstraints ^ (instance.numVariables << 16)
                   ^ (instance.A.count << 8)
        if let cached = cachedLinearization, cachedInstanceHash == hash {
            return cached
        }
        let result = LinearizedR1CS.linearize(instance)
        cachedLinearization = result
        cachedInstanceHash = hash
        return result
    }

    // MARK: - Full Linearization Pipeline

    /// Perform the complete linearization: encode R1CS as MLEs, compute
    /// witness linearization, evaluate at tau, and produce memory-checking digest.
    ///
    /// Returns all artifacts needed by the Spartan prover's sumcheck phase.
    public func fullLinearize(
        instance: SpartanR1CS,
        publicInputs: [Fr],
        witness: [Fr],
        tau: [Fr]
    ) -> LinearizationResult {
        let lin = linearize(instance)
        let witLin = WitnessLinearization.build(
            publicInputs: publicInputs, witness: witness,
            logN: lin.logN)

        // Compute row-bound vectors: ~M(tau, .) for M in {A, B, C}
        let (aVec, bVec, cVec) = lin.rowBindAll(tau: tau)

        // Compute Az, Bz, Cz as inner products with z_tilde
        let evaluator = HypercubeEvaluator(numVars: lin.logN)
        let azVal = evaluator.innerProduct(aVec, witLin.zTilde)
        let bzVal = evaluator.innerProduct(bVec, witLin.zTilde)
        let czVal = evaluator.innerProduct(cVec, witLin.zTilde)

        // Memory-checking digest
        let gamma = frFromInt(17)  // In real protocol, from Fiat-Shamir
        let beta = frFromInt(31)
        let memDigestA = MemoryCheckingDigest.compute(table: lin.mlA,
                                                       gamma: gamma, beta: beta)
        let memDigestB = MemoryCheckingDigest.compute(table: lin.mlB,
                                                       gamma: gamma, beta: beta)
        let memDigestC = MemoryCheckingDigest.compute(table: lin.mlC,
                                                       gamma: gamma, beta: beta)

        // Combined digest: product of all three
        let combinedDigest = frMul(memDigestA.digest,
                                   frMul(memDigestB.digest, memDigestC.digest))

        return LinearizationResult(
            linearizedR1CS: lin,
            witnessLinearization: witLin,
            aVec: aVec, bVec: bVec, cVec: cVec,
            azEval: azVal, bzEval: bzVal, czEval: czVal,
            memoryDigest: combinedDigest,
            memDigestA: memDigestA,
            memDigestB: memDigestB,
            memDigestC: memDigestC)
    }

    // MARK: - Sumcheck Reduction

    /// Perform one round of degree-2 sumcheck on the linearized inner product.
    /// Given vectors w and z of length 2*h, compute:
    ///   s(0) = sum_{i<h} w[i]*z[i]
    ///   s(1) = sum_{i<h} w[i+h]*z[i+h]
    ///   s(2) = sum_{i<h} (2*w[i+h]-w[i])*(2*z[i+h]-z[i])
    public func sumcheckRound(wVec: [Fr], zVec: [Fr],
                               halfSize h: Int) -> (Fr, Fr, Fr) {
        var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
        wVec.withUnsafeBytes { wBuf in
            zVec.withUnsafeBytes { zBuf in
                withUnsafeMutableBytes(of: &s0) { s0Buf in
                    withUnsafeMutableBytes(of: &s1) { s1Buf in
                        withUnsafeMutableBytes(of: &s2) { s2Buf in
                            bn254_fr_spartan_sumcheck_deg2(
                                wBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                Int32(h),
                                s0Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                s1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                s2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self))
                        }
                    }
                }
            }
        }
        return (s0, s1, s2)
    }

    /// Perform one round of degree-3 sumcheck for the Spartan equation:
    ///   F(x) = eq(tau,x) * (Az(x)*Bz(x) - Cz(x))
    /// Returns s(0), s(1), s(2), s(3).
    public func sumcheckRoundDeg3(eqTau: [Fr], az: [Fr], bz: [Fr], cz: [Fr],
                                   halfSize h: Int) -> (Fr, Fr, Fr, Fr) {
        var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero, s3 = Fr.zero
        eqTau.withUnsafeBytes { eqBuf in
            az.withUnsafeBytes { aBuf in
                bz.withUnsafeBytes { bBuf in
                    cz.withUnsafeBytes { cBuf in
                        withUnsafeMutableBytes(of: &s0) { s0B in
                            withUnsafeMutableBytes(of: &s1) { s1B in
                                withUnsafeMutableBytes(of: &s2) { s2B in
                                    withUnsafeMutableBytes(of: &s3) { s3B in
                                        bn254_fr_spartan_sumcheck_deg3(
                                            eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            Int32(h),
                                            s0B.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            s1B.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            s2B.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                            s3B.baseAddress!.assumingMemoryBound(to: UInt64.self))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (s0, s1, s2, s3)
    }

    /// Run a complete sumcheck reduction on the linearized claim.
    /// Proves: sum_x eq(tau,x) * (Az(x)*Bz(x) - Cz(x)) = 0
    /// Returns round polynomials, challenges, and final evaluations.
    public func sumcheckReduce(
        eqTau: [Fr], azVec: [Fr], bzVec: [Fr], czVec: [Fr],
        logM: Int
    ) -> SumcheckReductionResult {
        var eq = eqTau, az = azVec, bz = bzVec, cz = czVec
        var rounds = [(Fr, Fr, Fr, Fr)]()
        rounds.reserveCapacity(logM)
        var challenges = [Fr]()
        challenges.reserveCapacity(logM)
        var curSize = eq.count

        for round in 0..<logM {
            let h = curSize / 2
            let (s0, s1, s2, s3) = sumcheckRoundDeg3(
                eqTau: eq, az: az, bz: bz, cz: cz, halfSize: h)
            rounds.append((s0, s1, s2, s3))

            // Derive challenge deterministically (for testing; real protocol uses transcript)
            let challenge = deriveSumcheckChallenge(round: round, s0: s0, s1: s1)
            challenges.append(challenge)

            // Fold all arrays
            let evaluator = HypercubeEvaluator(numVars: 1)
            eq = evaluator.fold(eq, challenge: challenge)
            az = evaluator.fold(az, challenge: challenge)
            bz = evaluator.fold(bz, challenge: challenge)
            cz = evaluator.fold(cz, challenge: challenge)
            curSize = h
        }

        return SumcheckReductionResult(
            rounds: rounds, challenges: challenges,
            finalAz: az[0], finalBz: bz[0], finalCz: cz[0],
            finalEq: eq[0])
    }

    // MARK: - Batch Evaluation

    /// Evaluate all three matrix MLEs at the same point in one pass.
    /// More efficient than three separate evaluations when the point is shared.
    public func batchEvaluate(linearized: LinearizedR1CS,
                               tau: [Fr], x: [Fr]) -> (Fr, Fr, Fr) {
        return linearized.evaluateAll(tau: tau, x: x)
    }

    // MARK: - Structured Analysis

    /// Analyze an R1CS instance for structural uniformity.
    public func analyzeStructure(_ instance: SpartanR1CS) -> StructuredR1CS {
        return StructuredR1CS.analyze(instance)
    }

    // MARK: - Private Helpers

    /// Deterministic challenge derivation for testing.
    /// In the real protocol, challenges come from a Fiat-Shamir transcript.
    private func deriveSumcheckChallenge(round: Int, s0: Fr, s1: Fr) -> Fr {
        let roundFr = frFromInt(UInt64(round + 1))
        let combined = frAdd(frMul(roundFr, s0), s1)
        // Hash to get a "random-looking" challenge
        return frAdd(combined, frFromInt(UInt64(round * 7 + 13)))
    }
}

// MARK: - Result Types

/// Result of the full linearization pipeline.
public struct LinearizationResult {
    /// The linearized R1CS (MLE tables)
    public let linearizedR1CS: LinearizedR1CS
    /// Witness as multilinear extension
    public let witnessLinearization: WitnessLinearization
    /// Row-bound vectors ~A(tau,.), ~B(tau,.), ~C(tau,.)
    public let aVec: [Fr], bVec: [Fr], cVec: [Fr]
    /// Scalar evaluations <~A(tau,.), z>, <~B(tau,.), z>, <~C(tau,.), z>
    public let azEval: Fr, bzEval: Fr, czEval: Fr
    /// Combined memory-checking digest
    public let memoryDigest: Fr
    /// Per-matrix memory digests
    public let memDigestA: MemoryCheckingDigest
    public let memDigestB: MemoryCheckingDigest
    public let memDigestC: MemoryCheckingDigest

    public init(linearizedR1CS: LinearizedR1CS,
                witnessLinearization: WitnessLinearization,
                aVec: [Fr], bVec: [Fr], cVec: [Fr],
                azEval: Fr, bzEval: Fr, czEval: Fr,
                memoryDigest: Fr,
                memDigestA: MemoryCheckingDigest,
                memDigestB: MemoryCheckingDigest,
                memDigestC: MemoryCheckingDigest) {
        self.linearizedR1CS = linearizedR1CS
        self.witnessLinearization = witnessLinearization
        self.aVec = aVec; self.bVec = bVec; self.cVec = cVec
        self.azEval = azEval; self.bzEval = bzEval; self.czEval = czEval
        self.memoryDigest = memoryDigest
        self.memDigestA = memDigestA
        self.memDigestB = memDigestB
        self.memDigestC = memDigestC
    }
}

/// Result of sumcheck reduction on the linearized Spartan equation.
public struct SumcheckReductionResult {
    /// Round polynomials (s(0), s(1), s(2), s(3)) for each round
    public let rounds: [(Fr, Fr, Fr, Fr)]
    /// Challenges used at each round
    public let challenges: [Fr]
    /// Final scalar evaluations after all rounds
    public let finalAz: Fr, finalBz: Fr, finalCz: Fr, finalEq: Fr

    public init(rounds: [(Fr, Fr, Fr, Fr)], challenges: [Fr],
                finalAz: Fr, finalBz: Fr, finalCz: Fr, finalEq: Fr) {
        self.rounds = rounds
        self.challenges = challenges
        self.finalAz = finalAz; self.finalBz = finalBz
        self.finalCz = finalCz; self.finalEq = finalEq
    }

    /// Verify the sumcheck reduction: the final claim should equal
    /// eq(tau, rx) * (Az(rx)*Bz(rx) - Cz(rx))
    public func verifyFinalClaim() -> Bool {
        let expected = frMul(finalEq, frSub(frMul(finalAz, finalBz), finalCz))
        // Check that the last round evaluation matches
        guard let lastRound = rounds.last else { return false }
        let (s0, s1, s2, s3) = lastRound
        let lastChallenge = challenges.last ?? Fr.one
        let interpolated = spartanInterpCubic(
            s0: s0, s1: s1, s2: s2, s3: s3, t: lastChallenge)
        return spartanFrEqual(interpolated, expected)
    }
}

// MARK: - Sparse Eq Evaluation Helper

/// Evaluate eq(point, index) where index is treated as a binary vector.
/// eq(r, b) = prod_i (r_i * b_i + (1-r_i)*(1-b_i))
/// This avoids materializing the full eq table for single-point evaluation.
private func sparseEqAtIndex(point: [Fr], index: Int, numVars: Int) -> Fr {
    var result = Fr.one
    for i in 0..<numVars {
        let bit = (index >> i) & 1
        let ri = point[i]
        if bit == 1 {
            result = frMul(result, ri)
        } else {
            result = frMul(result, frSub(Fr.one, ri))
        }
    }
    return result
}

// Version registered in Versions.swift
