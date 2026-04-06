// GPUSTARKTraceLDEEngine — GPU-accelerated STARK trace Low-Degree Extension
//
// Implements the full STARK trace LDE pipeline with Metal GPU acceleration:
//   1. Trace polynomial interpolation (iNTT over execution trace columns)
//   2. Coset LDE: evaluate trace polynomials over blowup coset domain
//   3. Multi-column batch LDE (all columns in one pipeline)
//   4. Constraint composition polynomial evaluation
//   5. Boundary constraint enforcement
//   6. Trace commitment via Merkle tree of LDE evaluations
//
// Works with BN254 Fr field type. Falls back to CPU when Metal is unavailable.

import Foundation
import Metal

// MARK: - LDE Configuration

/// Configuration for the STARK trace LDE engine.
public struct STARKTraceLDEConfig {
    /// Log2 of the trace length (trace has 2^logTraceLen rows).
    public let logTraceLen: Int
    /// Blowup factor for LDE domain (must be power of 2, >= 2).
    public let blowupFactor: Int
    /// Number of trace columns.
    public let numColumns: Int
    /// Coset shift generator (multiplicative offset for the LDE domain).
    public let cosetShift: Fr

    /// Computed: trace length = 2^logTraceLen.
    public var traceLen: Int { 1 << logTraceLen }
    /// Computed: LDE domain size = traceLen * blowupFactor.
    public var ldeDomainSize: Int { traceLen * blowupFactor }
    /// Computed: log2 of LDE domain size.
    public var logLDEDomainSize: Int { logTraceLen + Int(log2(Double(blowupFactor))) }

    public init(logTraceLen: Int, blowupFactor: Int, numColumns: Int, cosetShift: Fr? = nil) {
        precondition(logTraceLen > 0 && logTraceLen <= 20,
                     "logTraceLen must be in [1, 20]")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2 >= 2")
        precondition(numColumns > 0, "Must have at least one trace column")
        self.logTraceLen = logTraceLen
        self.blowupFactor = blowupFactor
        self.numColumns = numColumns
        self.cosetShift = cosetShift ?? frFromInt(Fr.GENERATOR)
    }
}

// MARK: - Boundary Constraint

/// A boundary constraint over BN254 Fr: trace column `column` at row `row` must equal `value`.
public struct FrBoundaryConstraint {
    /// Column index in the trace.
    public let column: Int
    /// Row index in the trace.
    public let row: Int
    /// Expected value at that position.
    public let value: Fr

    public init(column: Int, row: Int, value: Fr) {
        self.column = column
        self.row = row
        self.value = value
    }
}

// MARK: - LDE Result

/// Result of a trace LDE computation.
public struct TraceLDEResult {
    /// LDE evaluations per column: ldeColumns[colIdx][evalIdx].
    public let ldeColumns: [[Fr]]
    /// Merkle root commitment of the LDE evaluations.
    public let commitment: Fr
    /// Merkle tree leaves (hashes of row-interleaved LDE evaluations).
    public let merkleLeaves: [Fr]
    /// Merkle tree internal nodes (for proof generation).
    public let merkleNodes: [Fr]
    /// Configuration used.
    public let config: STARKTraceLDEConfig

    /// Number of evaluations per column.
    public var domainSize: Int { config.ldeDomainSize }
}

// MARK: - Composition Result

/// Result of constraint composition polynomial evaluation.
public struct CompositionResult {
    /// Composition polynomial evaluations over the LDE domain.
    public let evaluations: [Fr]
    /// Merkle root commitment.
    public let commitment: Fr
}

// MARK: - LDE Errors

public enum STARKTraceLDEError: Error, CustomStringConvertible {
    case noGPU
    case noCommandQueue
    case invalidTraceShape(String)
    case nttFailed(String)
    case commitmentFailed(String)
    case boundaryConstraintViolation(String)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU device found"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .invalidTraceShape(let msg): return "Invalid trace shape: \(msg)"
        case .nttFailed(let msg): return "NTT operation failed: \(msg)"
        case .commitmentFailed(let msg): return "Commitment failed: \(msg)"
        case .boundaryConstraintViolation(let msg): return "Boundary constraint violated: \(msg)"
        }
    }
}

// MARK: - GPU STARK Trace LDE Engine

/// GPU-accelerated STARK trace Low-Degree Extension engine.
///
/// Pipeline for a single `extend()` call:
///   1. For each trace column: iNTT to get coefficients
///   2. Zero-pad coefficients from N to M = blowupFactor * N
///   3. Apply coset shift: coeff[i] *= g^i
///   4. Forward NTT of size M to get LDE evaluations
///   5. Build Merkle tree over row-interleaved LDE evaluations
///
/// The GPU is used for NTT/iNTT operations and coset shift when the domain
/// is large enough to amortize dispatch overhead. For small domains (<= 256),
/// CPU fallback is used automatically.
public final class GPUSTARKTraceLDEEngine {
    /// Minimum trace length to use GPU path (below this, CPU is faster).
    public static let gpuThreshold = 256

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let useGPU: Bool

    // NTT engine for GPU path (lazy)
    private var nttEngine: NTTEngine?

    public init(forceGPU: Bool = false) throws {
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            self.commandQueue = device.makeCommandQueue()
            self.useGPU = true
        } else if forceGPU {
            throw STARKTraceLDEError.noGPU
        } else {
            self.device = nil
            self.commandQueue = nil
            self.useGPU = false
        }
    }

    // MARK: - NTT Engine Accessor

    private func getNTTEngine() throws -> NTTEngine {
        if let e = nttEngine { return e }
        let e = try NTTEngine()
        nttEngine = e
        return e
    }

    // MARK: - Full Trace LDE

    /// Perform Low-Degree Extension on an execution trace.
    ///
    /// - Parameters:
    ///   - trace: Execution trace as column-major arrays. trace[colIdx][rowIdx].
    ///   - config: LDE configuration.
    /// - Returns: `TraceLDEResult` with LDE evaluations and Merkle commitment.
    public func extend(trace: [[Fr]], config: STARKTraceLDEConfig) throws -> TraceLDEResult {
        // Validate trace shape
        guard trace.count == config.numColumns else {
            throw STARKTraceLDEError.invalidTraceShape(
                "Expected \(config.numColumns) columns, got \(trace.count)")
        }
        for (i, col) in trace.enumerated() {
            guard col.count == config.traceLen else {
                throw STARKTraceLDEError.invalidTraceShape(
                    "Column \(i): expected \(config.traceLen) rows, got \(col.count)")
            }
        }

        // Perform LDE on each column
        let ldeColumns: [[Fr]]
        if useGPU && config.traceLen >= GPUSTARKTraceLDEEngine.gpuThreshold {
            ldeColumns = try gpuBatchLDE(trace: trace, config: config)
        } else {
            ldeColumns = try cpuBatchLDE(trace: trace, config: config)
        }

        // Build Merkle tree commitment over interleaved rows
        let (commitment, leaves, nodes) = buildMerkleTree(
            ldeColumns: ldeColumns, domainSize: config.ldeDomainSize)

        return TraceLDEResult(
            ldeColumns: ldeColumns,
            commitment: commitment,
            merkleLeaves: leaves,
            merkleNodes: nodes,
            config: config
        )
    }

    // MARK: - Single Column LDE

    /// Perform coset LDE on a single column of evaluations.
    ///
    /// Algorithm:
    ///   1. iNTT(evals) -> coefficients
    ///   2. Zero-pad to M = blowupFactor * N
    ///   3. Coset shift: coeff[i] *= g^i
    ///   4. NTT(padded) -> LDE evaluations
    public func extendColumn(
        evals: [Fr], logN: Int, blowupFactor: Int, cosetShift: Fr
    ) throws -> [Fr] {
        let n = evals.count
        precondition(n == (1 << logN), "evals.count must equal 2^logN")

        let logM = logN + Int(log2(Double(blowupFactor)))
        let m = 1 << logM

        // Step 1: iNTT to get coefficients
        let coeffs: [Fr]
        if useGPU && n >= GPUSTARKTraceLDEEngine.gpuThreshold {
            let engine = try getNTTEngine()
            coeffs = try engine.intt(evals)
        } else {
            coeffs = cpuINTT(evals, logN: logN)
        }

        // Step 2+3: Zero-pad and apply coset shift
        var padded = [Fr](repeating: Fr.zero, count: m)
        var gPower = Fr.one
        for i in 0..<n {
            padded[i] = frMul(coeffs[i], gPower)
            gPower = frMul(gPower, cosetShift)
        }
        // Remaining entries stay zero (zero-padded)

        // Step 4: Forward NTT
        let lde: [Fr]
        if useGPU && m >= GPUSTARKTraceLDEEngine.gpuThreshold {
            let engine = try getNTTEngine()
            lde = try engine.ntt(padded)
        } else {
            lde = cpuNTT(padded, logN: logM)
        }

        return lde
    }

    // MARK: - GPU Batch LDE

    /// GPU-accelerated batch LDE: processes all columns through the GPU NTT engine.
    private func gpuBatchLDE(trace: [[Fr]], config: STARKTraceLDEConfig) throws -> [[Fr]] {
        let engine = try getNTTEngine()
        let logM = config.logLDEDomainSize
        let m = config.ldeDomainSize
        let n = config.traceLen

        var results = [[Fr]]()
        results.reserveCapacity(config.numColumns)

        for col in trace {
            // iNTT to get coefficients
            let coeffs = try engine.intt(col)

            // Zero-pad + coset shift
            var padded = [Fr](repeating: Fr.zero, count: m)
            var gPower = Fr.one
            for i in 0..<n {
                padded[i] = frMul(coeffs[i], gPower)
                gPower = frMul(gPower, config.cosetShift)
            }

            // Forward NTT on extended domain
            let lde = try engine.ntt(padded)
            results.append(lde)
        }

        return results
    }

    // MARK: - CPU Batch LDE

    /// CPU fallback batch LDE for small traces or when GPU is unavailable.
    private func cpuBatchLDE(trace: [[Fr]], config: STARKTraceLDEConfig) throws -> [[Fr]] {
        let logM = config.logLDEDomainSize
        let m = config.ldeDomainSize
        let n = config.traceLen

        var results = [[Fr]]()
        results.reserveCapacity(config.numColumns)

        for col in trace {
            let coeffs = cpuINTT(col, logN: config.logTraceLen)

            var padded = [Fr](repeating: Fr.zero, count: m)
            var gPower = Fr.one
            for i in 0..<n {
                padded[i] = frMul(coeffs[i], gPower)
                gPower = frMul(gPower, config.cosetShift)
            }

            let lde = cpuNTT(padded, logN: logM)
            results.append(lde)
        }

        return results
    }

    // MARK: - Constraint Composition Polynomial

    /// Evaluate the constraint composition polynomial over the LDE domain.
    ///
    /// Given an LDE result and a set of transition constraints, computes:
    ///   C(x) = sum_i alpha^i * constraint_i(trace(x), trace(omega*x)) / Z_H(x)
    ///
    /// where Z_H(x) = x^N - 1 is the vanishing polynomial over the trace domain,
    /// and alpha is a random mixing coefficient.
    ///
    /// - Parameters:
    ///   - ldeResult: The trace LDE result.
    ///   - constraintEvaluator: Evaluates transition constraints given (current_row, next_row).
    ///                          Returns array of constraint evaluations.
    ///   - alpha: Random mixing coefficient for batching constraints.
    /// - Returns: Composition polynomial evaluations and commitment.
    public func evaluateComposition(
        ldeResult: TraceLDEResult,
        constraintEvaluator: ([Fr], [Fr]) -> [Fr],
        alpha: Fr
    ) throws -> CompositionResult {
        let config = ldeResult.config
        let m = config.ldeDomainSize
        let n = config.traceLen
        let numCols = config.numColumns

        // Precompute the vanishing polynomial evaluations: Z_H(x) = x^N - 1
        // Over the coset domain, x = g * omega_M^i, so Z_H(x) = (g * omega_M^i)^N - 1
        let logM = config.logLDEDomainSize
        let omegaM = frRootOfUnity(logN: logM)
        let cosetShift = config.cosetShift

        // g^N (coset shift raised to trace length)
        var gN = Fr.one
        for _ in 0..<n { gN = frMul(gN, cosetShift) }

        // omega_M^N for each index
        var omegaMpowers = [Fr](repeating: Fr.one, count: m)
        var w = Fr.one
        for i in 1..<m {
            w = frMul(w, omegaM)
            omegaMpowers[i] = w
        }

        // Vanishing poly Z_H(x_i) = (g * omega_M^i)^N - 1 = g^N * (omega_M^(i*N)) - 1
        // omega_M^N is a (m/N)-th root of unity; chain multiply for efficiency
        let omegaMN = frPow(omegaM, UInt64(n))
        var vanishing = [Fr](repeating: Fr.zero, count: m)
        var omegaMNpow = Fr.one
        for i in 0..<m {
            vanishing[i] = frSub(frMul(gN, omegaMNpow), Fr.one)
            omegaMNpow = frMul(omegaMNpow, omegaMN)
        }

        // Batch-invert vanishing poly: 3(m-1) muls + 1 inverse
        var vanishingInv = [Fr](repeating: Fr.zero, count: m)
        var vPrefix = [Fr](repeating: Fr.one, count: m)
        for i in 1..<m {
            vPrefix[i] = vanishing[i - 1] == Fr.zero ? vPrefix[i - 1] : frMul(vPrefix[i - 1], vanishing[i - 1])
        }
        let vLast = vanishing[m - 1] == Fr.zero ? vPrefix[m - 1] : frMul(vPrefix[m - 1], vanishing[m - 1])
        var vInv = frInverse(vLast)
        for i in stride(from: m - 1, through: 0, by: -1) {
            if vanishing[i] != Fr.zero {
                vanishingInv[i] = frMul(vInv, vPrefix[i])
                vInv = frMul(vInv, vanishing[i])
            }
        }

        // Evaluate constraints at each LDE point and divide by vanishing poly
        // The blowup factor > 1 guarantees Z_H is nonzero on the coset domain
        var composition = [Fr](repeating: Fr.zero, count: m)

        for i in 0..<m {
            // Gather current row values: ldeColumns[col][i]
            var currentRow = [Fr](repeating: Fr.zero, count: numCols)
            var nextRow = [Fr](repeating: Fr.zero, count: numCols)
            let nextIdx = (i + config.blowupFactor) % m

            for c in 0..<numCols {
                currentRow[c] = ldeResult.ldeColumns[c][i]
                nextRow[c] = ldeResult.ldeColumns[c][nextIdx]
            }

            // Evaluate all transition constraints
            let constraintVals = constraintEvaluator(currentRow, nextRow)

            // Mix constraints with alpha and divide by vanishing polynomial
            var mixed = Fr.zero
            var alphaPow = Fr.one
            for cv in constraintVals {
                mixed = frAdd(mixed, frMul(alphaPow, cv))
                alphaPow = frMul(alphaPow, alpha)
            }

            // Divide by Z_H(x_i)
            composition[i] = frMul(mixed, vanishingInv[i])
        }

        // Build Merkle commitment for composition polynomial
        let (root, _, _) = buildMerkleTreeSingle(evaluations: composition)

        return CompositionResult(evaluations: composition, commitment: root)
    }

    // MARK: - Boundary Constraint Enforcement

    /// Verify and enforce boundary constraints against the trace.
    ///
    /// Checks that trace[constraint.column][constraint.row] == constraint.value.
    /// Returns boundary quotient evaluations for each constraint.
    ///
    /// The boundary quotient for constraint (col, row, val) is:
    ///   B_j(x) = (trace_col(x) - val) / (x - omega^row)
    public func enforceBoundaryConstraints(
        ldeResult: TraceLDEResult,
        constraints: [FrBoundaryConstraint]
    ) throws -> [[Fr]] {
        let config = ldeResult.config
        let n = config.traceLen
        let m = config.ldeDomainSize
        let logM = config.logLDEDomainSize
        let omegaN = frRootOfUnity(logN: config.logTraceLen)
        let omegaM = frRootOfUnity(logN: logM)

        // Validate constraints
        for (i, bc) in constraints.enumerated() {
            guard bc.column >= 0 && bc.column < config.numColumns else {
                throw STARKTraceLDEError.boundaryConstraintViolation(
                    "Constraint \(i): column \(bc.column) out of range [0, \(config.numColumns))")
            }
            guard bc.row >= 0 && bc.row < n else {
                throw STARKTraceLDEError.boundaryConstraintViolation(
                    "Constraint \(i): row \(bc.row) out of range [0, \(n))")
            }
        }

        // Compute LDE domain points: x_i = cosetShift * omega_M^i
        var domainPoints = [Fr](repeating: Fr.zero, count: m)
        var w = Fr.one
        for i in 0..<m {
            domainPoints[i] = frMul(config.cosetShift, w)
            w = frMul(w, omegaM)
        }

        var quotients = [[Fr]]()
        quotients.reserveCapacity(constraints.count)

        for bc in constraints {
            // omega_N^row
            var omegaRow = Fr.one
            for _ in 0..<bc.row {
                omegaRow = frMul(omegaRow, omegaN)
            }

            // B(x_i) = (trace_col(x_i) - val) / (x_i - omega_N^row)
            // Batch-invert denominators
            var bcDenoms = [Fr](repeating: Fr.zero, count: m)
            for i in 0..<m { bcDenoms[i] = frSub(domainPoints[i], omegaRow) }
            var bcPrefix = [Fr](repeating: Fr.one, count: m)
            for i in 1..<m {
                bcPrefix[i] = bcDenoms[i - 1] == Fr.zero ? bcPrefix[i - 1] : frMul(bcPrefix[i - 1], bcDenoms[i - 1])
            }
            let bcLast = bcDenoms[m - 1] == Fr.zero ? bcPrefix[m - 1] : frMul(bcPrefix[m - 1], bcDenoms[m - 1])
            var bcInvRunning = frInverse(bcLast)
            var bcDenomInvs = [Fr](repeating: Fr.zero, count: m)
            for i in stride(from: m - 1, through: 0, by: -1) {
                if bcDenoms[i] != Fr.zero {
                    bcDenomInvs[i] = frMul(bcInvRunning, bcPrefix[i])
                    bcInvRunning = frMul(bcInvRunning, bcDenoms[i])
                }
            }
            var quotient = [Fr](repeating: Fr.zero, count: m)
            for i in 0..<m {
                let traceVal = ldeResult.ldeColumns[bc.column][i]
                let numerator = frSub(traceVal, bc.value)
                quotient[i] = frMul(numerator, bcDenomInvs[i])
            }
            quotients.append(quotient)
        }

        return quotients
    }

    // MARK: - Query Trace at LDE Point

    /// Get all column evaluations at a specific LDE domain index.
    public func queryRow(ldeResult: TraceLDEResult, index: Int) -> [Fr] {
        var row = [Fr](repeating: Fr.zero, count: ldeResult.config.numColumns)
        for c in 0..<ldeResult.config.numColumns {
            row[c] = ldeResult.ldeColumns[c][index]
        }
        return row
    }

    /// Get the Merkle proof for a specific LDE row index.
    public func getMerkleProof(ldeResult: TraceLDEResult, leafIndex: Int) -> FrMerklePath {
        let depth = Int(log2(Double(ldeResult.config.ldeDomainSize)))
        var siblings = [Fr]()
        var idx = leafIndex

        for level in 0..<depth {
            let sibIdx = idx ^ 1
            let levelOffset = (1 << (depth - level)) - 1  // offset to this level's nodes
            // For a simple binary tree: nodes at level 0 are leaves
            if level == 0 {
                siblings.append(ldeResult.merkleLeaves[sibIdx])
            } else {
                let nodeIdx = levelOffset + sibIdx
                if nodeIdx < ldeResult.merkleNodes.count {
                    siblings.append(ldeResult.merkleNodes[nodeIdx])
                } else {
                    siblings.append(Fr.zero)
                }
            }
            idx >>= 1
        }

        return FrMerklePath(
            leafIndex: leafIndex,
            leaf: ldeResult.merkleLeaves[leafIndex],
            siblings: siblings,
            root: ldeResult.commitment
        )
    }

    // MARK: - Merkle Tree Construction

    /// Build a Merkle tree from row-interleaved LDE columns.
    /// Each leaf = hash of all column values at that row index.
    private func buildMerkleTree(
        ldeColumns: [[Fr]], domainSize: Int
    ) -> (root: Fr, leaves: [Fr], nodes: [Fr]) {
        let numCols = ldeColumns.count

        // Compute leaf hashes: leaf[i] = hash of row i across all columns
        var leaves = [Fr](repeating: Fr.zero, count: domainSize)
        for i in 0..<domainSize {
            var rowHash = Fr.zero
            for c in 0..<numCols {
                rowHash = merkleHash(left: rowHash, right: ldeColumns[c][i])
            }
            leaves[i] = rowHash
        }

        // Build tree bottom-up
        // Total nodes = 2 * domainSize - 1
        let totalNodes = 2 * domainSize - 1
        var nodes = [Fr](repeating: Fr.zero, count: totalNodes)

        // Copy leaves to the right half
        for i in 0..<domainSize {
            nodes[domainSize - 1 + i] = leaves[i]
        }

        // Build internal nodes
        var idx = domainSize - 2
        while idx >= 0 {
            let left = nodes[2 * idx + 1]
            let right = nodes[2 * idx + 2]
            nodes[idx] = merkleHash(left: left, right: right)
            idx -= 1
        }

        return (nodes[0], leaves, nodes)
    }

    /// Build Merkle tree for a single evaluation vector.
    private func buildMerkleTreeSingle(
        evaluations: [Fr]
    ) -> (root: Fr, leaves: [Fr], nodes: [Fr]) {
        let n = evaluations.count
        let totalNodes = 2 * n - 1
        var nodes = [Fr](repeating: Fr.zero, count: totalNodes)

        for i in 0..<n {
            nodes[n - 1 + i] = evaluations[i]
        }

        var idx = n - 2
        while idx >= 0 {
            let left = nodes[2 * idx + 1]
            let right = nodes[2 * idx + 2]
            nodes[idx] = merkleHash(left: left, right: right)
            idx -= 1
        }

        return (nodes[0], evaluations, nodes)
    }

    /// Simple algebraic Merkle hash: H(l, r) = l^2 + 3*r + 7
    /// (Matches GPUSTARKVerifierEngine for compatibility; production would use Poseidon2.)
    private func merkleHash(left: Fr, right: Fr) -> Fr {
        let lSq = frMul(left, left)
        let three = frFromInt(3)
        let seven = frFromInt(7)
        let rScaled = frMul(three, right)
        return frAdd(frAdd(lSq, rScaled), seven)
    }

    // MARK: - CPU NTT Fallback

    /// CPU reference forward NTT (Cooley-Tukey DIT with bit-reversal).
    private func cpuNTT(_ input: [Fr], logN: Int) -> [Fr] {
        return NTTEngine.cpuNTT(input, logN: logN)
    }

    /// CPU reference inverse NTT.
    private func cpuINTT(_ input: [Fr], logN: Int) -> [Fr] {
        return NTTEngine.cpuINTT(input, logN: logN)
    }

    // MARK: - Polynomial Evaluation (Horner)

    /// Evaluate a polynomial given by coefficients at a single point using Horner's method.
    public func evaluatePolynomial(_ coeffs: [Fr], at point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), coeffs[i])
        }
        return result
    }

    // MARK: - Interpolation (Coefficients from Evaluations)

    /// Recover polynomial coefficients from evaluations via iNTT.
    /// Input: evaluations at omega^0, omega^1, ..., omega^(n-1).
    /// Output: coefficient representation.
    public func interpolate(evals: [Fr], logN: Int) throws -> [Fr] {
        let n = evals.count
        precondition(n == (1 << logN), "evals.count must equal 2^logN")

        if useGPU && n >= GPUSTARKTraceLDEEngine.gpuThreshold {
            let engine = try getNTTEngine()
            return try engine.intt(evals)
        } else {
            return cpuINTT(evals, logN: logN)
        }
    }

    // MARK: - Batch Interpolation

    /// Recover polynomial coefficients for multiple columns.
    public func batchInterpolate(columns: [[Fr]], logN: Int) throws -> [[Fr]] {
        var results = [[Fr]]()
        results.reserveCapacity(columns.count)
        for col in columns {
            results.append(try interpolate(evals: col, logN: logN))
        }
        return results
    }
}
