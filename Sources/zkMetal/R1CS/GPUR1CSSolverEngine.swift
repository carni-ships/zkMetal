// GPUR1CSSolverEngine — GPU-accelerated R1CS constraint satisfaction and solving
//
// Implements GPU-parallel sparse matrix-vector multiplication for R1CS systems
// of the form A*w . B*w = C*w (Hadamard product), with CPU fallback for small
// systems. Supports:
//   - Sparse matrix-vector multiplication (GPU or CPU)
//   - Witness augmentation (compute auxiliary variables from public inputs)
//   - Constraint satisfaction checking (batch Hadamard comparison)
//   - Circom-compatible R1CS format (via R1CSParser / R1CSFileConstraint)
//
// Wire layout (Circom convention):
//   w[0] = 1 (constant "one" wire)
//   w[1..numPublic] = public inputs (outputs + public signals)
//   w[numPublic+1..] = private witness (private inputs + intermediate wires)
//
// Public API:
//   checkSatisfaction(r1cs:witness:)        — verify A*w . B*w == C*w for all constraints
//   sparseMatVec(entries:witness:numRows:)   — GPU sparse matrix-vector product
//   augmentWitness(r1cs:publicInputs:)       — compute full witness from public inputs
//   residuals(r1cs:witness:)                 — per-constraint residual A*w . B*w - C*w
//   fromCircom(file:)                        — import from parsed Circom R1CS file

import Foundation
import Metal

// MARK: - R1CS Solver Result

/// Result of GPU R1CS satisfaction check.
public struct R1CSSatisfactionResult {
    /// Whether all constraints are satisfied.
    public let satisfied: Bool
    /// Number of constraints checked.
    public let numConstraints: Int
    /// Indices of unsatisfied constraints (empty if satisfied).
    public let unsatisfiedIndices: [Int]
    /// Per-constraint residuals: A*w . B*w - C*w (all zero if satisfied).
    public let residuals: [Fr]
}

/// Result of GPU witness augmentation.
public struct R1CSAugmentResult {
    /// The full witness vector [1, pub_1..pub_n, priv_1..priv_m].
    public let witness: [Fr]
    /// Whether all witness variables were resolved.
    public let isComplete: Bool
    /// Number of solver iterations used.
    public let iterations: Int
    /// Indices of unresolved variables.
    public let unresolvedIndices: [Int]
}

// MARK: - GPU R1CS Solver Engine

public class GPUR1CSSolverEngine {

    /// Metal device (nil if GPU unavailable, uses CPU fallback).
    public let device: MTLDevice?
    public let commandQueue: MTLCommandQueue?

    private var spmvPipeline: MTLComputePipelineState?
    private var hadamardCheckPipeline: MTLComputePipelineState?

    private let threadgroupSize: Int
    private let pool: GPUBufferPool?

    /// Arrays smaller than this threshold are computed on CPU.
    public var cpuThreshold: Int = 256

    /// Maximum iterations for witness augmentation.
    public var maxAugmentIterations: Int = 1000

    // MARK: - Initialization

    /// Create engine. Falls back to CPU if GPU is unavailable.
    public init(threadgroupSize: Int = 256) {
        self.threadgroupSize = threadgroupSize

        if let device = MTLCreateSystemDefaultDevice(),
           let queue = device.makeCommandQueue() {
            self.device = device
            self.commandQueue = queue
            self.pool = GPUBufferPool(device: device)

            do {
                let lib = try GPUR1CSSolverEngine.compileShaders(device: device)
                if let spmvFn = lib.makeFunction(name: "r1cs_sparse_matvec"),
                   let hadFn = lib.makeFunction(name: "r1cs_hadamard_check") {
                    self.spmvPipeline = try device.makeComputePipelineState(function: spmvFn)
                    self.hadamardCheckPipeline = try device.makeComputePipelineState(function: hadFn)
                }
            } catch {
                // GPU shader compilation failed; fall back to CPU
                self.spmvPipeline = nil
                self.hadamardCheckPipeline = nil
            }
        } else {
            self.device = nil
            self.commandQueue = nil
            self.pool = nil
        }
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        \(cleanFr)

        // Sparse matrix entry: (row, col, val[8]) = 10 x uint32
        struct SparseEntry {
            uint row;
            uint col;
            uint val[8]; // Fr in Montgomery form as 8 x uint32
        };

        // Sparse matrix-vector product: result[row] += val * witness[col]
        // Each thread processes one sparse entry and atomically accumulates.
        // We use a two-pass approach: first compute products, then reduce by row.
        kernel void r1cs_sparse_matvec(
            device const SparseEntry* entries [[buffer(0)]],
            device const uint* witness       [[buffer(1)]],  // flattened Fr[numVars] as uint32[numVars*8]
            device uint* output              [[buffer(2)]],   // flattened Fr[numRows] as uint32[numRows*8]
            device const uint& numEntries    [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= numEntries) return;

            SparseEntry e = entries[tid];

            // Load val
            uint va[8], wb[8];
            for (int i = 0; i < 8; i++) {
                va[i] = e.val[i];
                wb[i] = witness[e.col * 8 + i];
            }

            // Multiply: product = val * witness[col] in Montgomery form
            uint prod[8];
            fr_mul(va, wb, prod);

            // Accumulate into output[row] using atomic add (field add)
            // Since atomic Fr add is complex, we use a simple serial approach
            // by writing products to a staging buffer indexed by entry id.
            // The host will reduce. For now, store product at entry position.
            for (int i = 0; i < 8; i++) {
                output[tid * 8 + i] = prod[i];
            }
        }

        // Hadamard check: verify az[i] * bz[i] == cz[i] for each constraint
        // Output: flags[i] = 1 if satisfied, 0 if not
        kernel void r1cs_hadamard_check(
            device const uint* az  [[buffer(0)]],   // Fr[numConstraints]
            device const uint* bz  [[buffer(1)]],   // Fr[numConstraints]
            device const uint* cz  [[buffer(2)]],   // Fr[numConstraints]
            device uint* flags     [[buffer(3)]],    // uint[numConstraints]
            device const uint& n   [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= n) return;

            uint a[8], b[8], c[8], prod[8];
            for (int i = 0; i < 8; i++) {
                a[i] = az[tid * 8 + i];
                b[i] = bz[tid * 8 + i];
                c[i] = cz[tid * 8 + i];
            }

            fr_mul(a, b, prod);

            // Check prod == c
            uint eq = 1;
            for (int i = 0; i < 8; i++) {
                if (prod[i] != c[i]) { eq = 0; break; }
            }
            flags[tid] = eq;
        }
        """

        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        return try device.makeLibrary(source: kernelSource, options: opts)
    }

    // MARK: - Sparse Matrix-Vector Product (CPU)

    /// CPU sparse matrix-vector multiply: result[row] += val * witness[col]
    public func cpuSparseMatVec(_ entries: [R1CSEntry], _ witness: [Fr], numRows: Int) -> [Fr] {
        var result = [Fr](repeating: .zero, count: numRows)
        for e in entries {
            guard e.col < witness.count else { continue }
            result[e.row] = frAdd(result[e.row], frMul(e.val, witness[e.col]))
        }
        return result
    }

    // MARK: - Sparse Matrix-Vector Product (GPU)

    /// GPU sparse matrix-vector multiply with CPU fallback.
    public func sparseMatVec(entries: [R1CSEntry], witness: [Fr], numRows: Int) -> [Fr] {
        // CPU fallback for small systems or no GPU
        if entries.count < cpuThreshold || spmvPipeline == nil || device == nil {
            return cpuSparseMatVec(entries, witness, numRows: numRows)
        }

        guard let device = device, let queue = commandQueue, let pipeline = spmvPipeline else {
            return cpuSparseMatVec(entries, witness, numRows: numRows)
        }

        // Pack entries into GPU-friendly format: (row: UInt32, col: UInt32, val: 8xUInt32)
        let entryStride = 10  // 2 + 8 uint32s per entry
        var packedEntries = [UInt32](repeating: 0, count: entries.count * entryStride)
        for (i, e) in entries.enumerated() {
            let base = i * entryStride
            packedEntries[base] = UInt32(e.row)
            packedEntries[base + 1] = UInt32(e.col)
            let limbs = e.val.to64()
            for j in 0..<4 {
                packedEntries[base + 2 + j * 2] = UInt32(limbs[j] & 0xFFFFFFFF)
                packedEntries[base + 2 + j * 2 + 1] = UInt32(limbs[j] >> 32)
            }
        }

        // Pack witness
        var packedWitness = [UInt32](repeating: 0, count: witness.count * 8)
        for (i, w) in witness.enumerated() {
            let limbs = w.to64()
            for j in 0..<4 {
                packedWitness[i * 8 + j * 2] = UInt32(limbs[j] & 0xFFFFFFFF)
                packedWitness[i * 8 + j * 2 + 1] = UInt32(limbs[j] >> 32)
            }
        }

        let entryBuf = device.makeBuffer(bytes: packedEntries,
                                          length: packedEntries.count * 4, options: .storageModeShared)!
        let witnessBuf = device.makeBuffer(bytes: packedWitness,
                                            length: packedWitness.count * 4, options: .storageModeShared)!
        // Output: one Fr per entry (will reduce on CPU by row)
        let outputBuf = device.makeBuffer(length: entries.count * 8 * 4, options: .storageModeShared)!
        var numEnt = UInt32(entries.count)
        let numBuf = device.makeBuffer(bytes: &numEnt, length: 4, options: .storageModeShared)!

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return cpuSparseMatVec(entries, witness, numRows: numRows)
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(entryBuf, offset: 0, index: 0)
        enc.setBuffer(witnessBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBuffer(numBuf, offset: 0, index: 3)

        let gridSize = MTLSize(width: entries.count, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, entries.count), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back and reduce by row
        let outPtr = outputBuf.contents().bindMemory(to: UInt32.self, capacity: entries.count * 8)
        var result = [Fr](repeating: .zero, count: numRows)

        for (i, e) in entries.enumerated() {
            var limbs = [UInt64](repeating: 0, count: 4)
            for j in 0..<4 {
                let lo = UInt64(outPtr[i * 8 + j * 2])
                let hi = UInt64(outPtr[i * 8 + j * 2 + 1])
                limbs[j] = lo | (hi << 32)
            }
            let product = Fr.from64(limbs)
            result[e.row] = frAdd(result[e.row], product)
        }

        return result
    }

    // MARK: - Constraint Satisfaction Check

    /// Check whether witness satisfies all R1CS constraints: A*w . B*w == C*w.
    public func checkSatisfaction(r1cs: R1CSInstance, witness: [Fr]) -> R1CSSatisfactionResult {
        let n = r1cs.numConstraints
        guard witness.count >= r1cs.numVars else {
            return R1CSSatisfactionResult(satisfied: false, numConstraints: n,
                                           unsatisfiedIndices: Array(0..<n),
                                           residuals: [Fr](repeating: .zero, count: n))
        }

        let az = sparseMatVec(entries: r1cs.aEntries, witness: witness, numRows: n)
        let bz = sparseMatVec(entries: r1cs.bEntries, witness: witness, numRows: n)
        let cz = sparseMatVec(entries: r1cs.cEntries, witness: witness, numRows: n)

        var residuals = [Fr](repeating: .zero, count: n)
        var unsatisfied = [Int]()

        for i in 0..<n {
            let ab = frMul(az[i], bz[i])
            residuals[i] = frSub(ab, cz[i])
            if !residuals[i].isZero {
                unsatisfied.append(i)
            }
        }

        return R1CSSatisfactionResult(satisfied: unsatisfied.isEmpty,
                                       numConstraints: n,
                                       unsatisfiedIndices: unsatisfied,
                                       residuals: residuals)
    }

    // MARK: - Residuals

    /// Compute per-constraint residuals: A*w . B*w - C*w.
    public func residuals(r1cs: R1CSInstance, witness: [Fr]) -> [Fr] {
        let n = r1cs.numConstraints
        let az = sparseMatVec(entries: r1cs.aEntries, witness: witness, numRows: n)
        let bz = sparseMatVec(entries: r1cs.bEntries, witness: witness, numRows: n)
        let cz = sparseMatVec(entries: r1cs.cEntries, witness: witness, numRows: n)

        var res = [Fr](repeating: .zero, count: n)
        for i in 0..<n {
            res[i] = frSub(frMul(az[i], bz[i]), cz[i])
        }
        return res
    }

    // MARK: - Witness Augmentation

    /// Compute full witness from R1CS constraints and public inputs.
    /// Uses iterative forward/backward propagation (delegates to WitnessSolver).
    public func augmentWitness(r1cs: R1CSInstance, publicInputs: [Fr],
                               privateHints: [Int: Fr] = [:]) -> R1CSAugmentResult {
        let solver = WitnessSolver(maxIterations: maxAugmentIterations)
        let result = solver.solve(r1cs: r1cs, publicInputs: publicInputs, privateHints: privateHints)

        return R1CSAugmentResult(witness: result.witness,
                                  isComplete: result.isFullySolved,
                                  iterations: result.iterations,
                                  unresolvedIndices: result.unsolvedIndices)
    }

    // MARK: - Circom Import

    /// Import R1CS from a parsed Circom R1CS file.
    public static func fromCircom(file: R1CSFile) -> R1CSInstance {
        return R1CSParser.toR1CSInstance(file)
    }

    /// Import R1CS and convert to per-constraint sparse format.
    public static func constraintSetFromCircom(file: R1CSFile) -> R1CSConstraintSet {
        let numPublic = Int(file.header.nOutputs + file.header.nPubInputs)
        let numVars = Int(file.header.nWires)
        return R1CSConstraintSet(from: file.constraints, numVars: numVars, numPublic: numPublic)
    }

    // MARK: - Batch Satisfaction Check

    /// Check multiple witness vectors against the same R1CS system.
    /// Returns array of satisfaction results, one per witness.
    public func batchCheckSatisfaction(r1cs: R1CSInstance, witnesses: [[Fr]]) -> [R1CSSatisfactionResult] {
        return witnesses.map { checkSatisfaction(r1cs: r1cs, witness: $0) }
    }

    // MARK: - Utility

    /// Build a simple R1CS constraint: coeff_a[i]*w[col_a[i]] * coeff_b[j]*w[col_b[j]] = coeff_c[k]*w[col_c[k]]
    /// Convenience for building test circuits.
    public static func makeConstraint(
        a: [(col: Int, val: Fr)],
        b: [(col: Int, val: Fr)],
        c: [(col: Int, val: Fr)],
        row: Int
    ) -> (aEntries: [R1CSEntry], bEntries: [R1CSEntry], cEntries: [R1CSEntry]) {
        let aE = a.map { R1CSEntry(row: row, col: $0.col, val: $0.val) }
        let bE = b.map { R1CSEntry(row: row, col: $0.col, val: $0.val) }
        let cE = c.map { R1CSEntry(row: row, col: $0.col, val: $0.val) }
        return (aE, bE, cE)
    }
}
