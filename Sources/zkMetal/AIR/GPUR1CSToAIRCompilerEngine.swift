// GPUR1CSToAIRCompilerEngine — GPU-accelerated R1CS to AIR compiler
//
// Converts R1CS (Rank-1 Constraint System) constraints into AIR (Algebraic
// Intermediate Representation) trace format suitable for STARK proving.
//
// R1CS operates on a witness vector w and three sparse matrices A, B, C such that
// for each constraint i: (A_i . w) * (B_i . w) = (C_i . w). The AIR representation
// encodes the same relation as:
//   - An execution trace with columns for each wire assignment
//   - Transition constraints that enforce the multiplication gate relation
//   - Boundary constraints from public inputs
//
// Key operations:
//   1. Flatten R1CS matrices A, B, C into trace columns
//   2. Generate transition constraints from R1CS structure
//   3. Handle boundary constraints from public inputs
//   4. Optimize trace layout for cache efficiency (column reordering)
//   5. GPU-accelerated trace generation and constraint evaluation
//
// Supports both uniform (single step function) and non-uniform R1CS:
//   - Uniform: all constraints share the same structure, encoded as periodic AIR
//   - Non-uniform: constraints differ per row, encoded with selector columns
//
// Wire layout (Circom convention, preserved in trace):
//   w[0] = 1 (constant "one" wire)
//   w[1..numPublic] = public inputs
//   w[numPublic+1..] = private witness
//
// Trace layout (column-major):
//   Columns 0..2: a_val, b_val, c_val (evaluated constraint values per row)
//   Columns 3..3+numVars-1: wire assignment columns (one per variable)
//   Additional selector columns for non-uniform R1CS

import Foundation
import Metal

// MARK: - AIR Trace Layout Configuration

/// Describes how R1CS variables and constraints map to AIR trace columns.
public struct AIRTraceLayout {
    /// Total number of columns in the execution trace.
    public let numColumns: Int
    /// Log2 of the number of rows (padded to power of 2).
    public let logNumRows: Int
    /// Number of rows (power of 2).
    public var numRows: Int { 1 << logNumRows }
    /// Column index where wire assignment columns begin.
    public let wireColumnOffset: Int
    /// Number of wire columns (= numVars from R1CS).
    public let numWireColumns: Int
    /// Column indices for the A, B, C evaluated values.
    public let aValColumn: Int
    public let bValColumn: Int
    public let cValColumn: Int
    /// Column index for selector columns (non-uniform R1CS only).
    public let selectorColumnOffset: Int
    /// Number of selector columns (0 for uniform R1CS).
    public let numSelectorColumns: Int
    /// Mapping from original wire index to reordered column index.
    /// Used for cache-efficient layout (frequently accessed wires grouped together).
    public let wireToColumn: [Int]
    /// Inverse mapping: column index -> original wire index.
    public let columnToWire: [Int]

    public init(numColumns: Int, logNumRows: Int, wireColumnOffset: Int,
                numWireColumns: Int, aValColumn: Int, bValColumn: Int,
                cValColumn: Int, selectorColumnOffset: Int,
                numSelectorColumns: Int, wireToColumn: [Int], columnToWire: [Int]) {
        self.numColumns = numColumns
        self.logNumRows = logNumRows
        self.wireColumnOffset = wireColumnOffset
        self.numWireColumns = numWireColumns
        self.aValColumn = aValColumn
        self.bValColumn = bValColumn
        self.cValColumn = cValColumn
        self.selectorColumnOffset = selectorColumnOffset
        self.numSelectorColumns = numSelectorColumns
        self.wireToColumn = wireToColumn
        self.columnToWire = columnToWire
    }
}

// MARK: - AIR Transition Constraint (from R1CS)

/// A single AIR transition constraint derived from an R1CS constraint.
/// Encodes: A_row . w * B_row . w = C_row . w as a polynomial identity
/// over the trace columns.
public struct R1CSAIRTransitionConstraint {
    /// Index of the original R1CS constraint this came from.
    public let constraintIndex: Int
    /// Sparse entries for the A matrix row (wire_index, coefficient).
    public let aTerms: [(wireIndex: Int, coeff: Fr)]
    /// Sparse entries for the B matrix row.
    public let bTerms: [(wireIndex: Int, coeff: Fr)]
    /// Sparse entries for the C matrix row.
    public let cTerms: [(wireIndex: Int, coeff: Fr)]
    /// Algebraic degree of this constraint (always 2 for R1CS: deg(A*B - C) = 2).
    public let degree: Int
    /// Whether this constraint is active only on specific rows (non-uniform).
    public let selectorIndex: Int?

    public init(constraintIndex: Int,
                aTerms: [(wireIndex: Int, coeff: Fr)],
                bTerms: [(wireIndex: Int, coeff: Fr)],
                cTerms: [(wireIndex: Int, coeff: Fr)],
                degree: Int = 2,
                selectorIndex: Int? = nil) {
        self.constraintIndex = constraintIndex
        self.aTerms = aTerms
        self.bTerms = bTerms
        self.cTerms = cTerms
        self.degree = degree
        self.selectorIndex = selectorIndex
    }
}

// MARK: - AIR Boundary Constraint (from public inputs)

/// A boundary constraint derived from R1CS public inputs.
/// Fixes wire values at row 0 of the trace.
public struct R1CSAIRBoundaryConstraint {
    /// Column index in the trace.
    public let column: Int
    /// Row index (typically 0 for initial values).
    public let row: Int
    /// Expected field value.
    public let value: Fr
    /// Label for debugging.
    public let label: String

    public init(column: Int, row: Int, value: Fr, label: String) {
        self.column = column
        self.row = row
        self.value = value
        self.label = label
    }
}

// MARK: - Compiled AIR from R1CS

/// The result of compiling an R1CS system into an AIR representation.
/// Contains all information needed for STARK proving.
public struct CompiledR1CSAIR {
    /// The trace layout describing column assignments.
    public let layout: AIRTraceLayout
    /// Transition constraints (one per R1CS constraint, or fewer if uniform).
    public let transitionConstraints: [R1CSAIRTransitionConstraint]
    /// Boundary constraints from public inputs.
    public let boundaryConstraints: [R1CSAIRBoundaryConstraint]
    /// The execution trace (column-major).
    public let trace: [[Fr]]
    /// Whether this is a uniform AIR (all rows have the same constraint).
    public let isUniform: Bool
    /// Maximum constraint degree.
    public let maxDegree: Int
    /// Number of original R1CS constraints.
    public let numR1CSConstraints: Int
    /// Number of original R1CS variables.
    public let numR1CSVariables: Int
    /// Number of public inputs.
    public let numPublicInputs: Int
    /// Wire access frequency counts (for layout optimization analysis).
    public let wireAccessCounts: [Int]

    public init(layout: AIRTraceLayout,
                transitionConstraints: [R1CSAIRTransitionConstraint],
                boundaryConstraints: [R1CSAIRBoundaryConstraint],
                trace: [[Fr]], isUniform: Bool, maxDegree: Int,
                numR1CSConstraints: Int, numR1CSVariables: Int,
                numPublicInputs: Int, wireAccessCounts: [Int]) {
        self.layout = layout
        self.transitionConstraints = transitionConstraints
        self.boundaryConstraints = boundaryConstraints
        self.trace = trace
        self.isUniform = isUniform
        self.maxDegree = maxDegree
        self.numR1CSConstraints = numR1CSConstraints
        self.numR1CSVariables = numR1CSVariables
        self.numPublicInputs = numPublicInputs
        self.wireAccessCounts = wireAccessCounts
    }

    /// Verify that the trace satisfies all constraints. Returns nil on success,
    /// or a description of the first violation found.
    public func verify() -> String? {
        let n = layout.numRows

        // Check boundary constraints
        for bc in boundaryConstraints {
            guard bc.column < trace.count else {
                return "Boundary column \(bc.column) out of range (have \(trace.count) cols)"
            }
            guard bc.row < trace[bc.column].count else {
                return "Boundary row \(bc.row) out of range for column \(bc.column)"
            }
            if !frEqual(trace[bc.column][bc.row], bc.value) {
                return "Boundary violation at column \(bc.column), row \(bc.row): \(bc.label)"
            }
        }

        // Check transition constraints
        let numActive = min(numR1CSConstraints, n)
        for row in 0..<numActive {
            for tc in transitionConstraints {
                // If this constraint uses a selector, check it
                if let selIdx = tc.selectorIndex {
                    let selCol = layout.selectorColumnOffset + selIdx
                    if selCol < trace.count && row < trace[selCol].count {
                        if trace[selCol][row].isZero {
                            continue // Selector is 0, constraint inactive on this row
                        }
                    }
                }

                // Evaluate A . w, B . w, C . w at this row
                let aVal = evaluateLinearCombination(tc.aTerms, atRow: row)
                let bVal = evaluateLinearCombination(tc.bTerms, atRow: row)
                let cVal = evaluateLinearCombination(tc.cTerms, atRow: row)

                // Check A*B = C
                let product = frMul(aVal, bVal)
                let residual = frSub(product, cVal)
                if !residual.isZero {
                    return "Constraint \(tc.constraintIndex) violated at row \(row)"
                }
            }
        }

        return nil
    }

    /// Evaluate a linear combination sum(coeff_i * wire_i) at a given trace row.
    private func evaluateLinearCombination(
        _ terms: [(wireIndex: Int, coeff: Fr)], atRow row: Int
    ) -> Fr {
        var sum = Fr.zero
        for (wireIdx, coeff) in terms {
            let col = layout.wireColumnOffset + wireIdx
            guard col < trace.count && row < trace[col].count else { continue }
            sum = frAdd(sum, frMul(coeff, trace[col][row]))
        }
        return sum
    }
}

// MARK: - Wire Access Analysis

/// Analyzes wire access patterns in R1CS for cache-efficient trace layout.
public struct WireAccessAnalysis {
    /// Number of times each wire is referenced across all constraints.
    public let accessCounts: [Int]
    /// Wire indices sorted by descending access frequency.
    public let sortedByFrequency: [Int]
    /// Wires that appear in the same constraint (co-occurrence edges).
    public let coOccurrenceCount: Int
    /// Average number of non-zero entries per constraint row.
    public let avgDensity: Double

    public init(accessCounts: [Int], sortedByFrequency: [Int],
                coOccurrenceCount: Int, avgDensity: Double) {
        self.accessCounts = accessCounts
        self.sortedByFrequency = sortedByFrequency
        self.coOccurrenceCount = coOccurrenceCount
        self.avgDensity = avgDensity
    }
}

// MARK: - R1CS Uniformity Analysis

/// Result of analyzing whether an R1CS system is uniform.
public struct R1CSUniformityAnalysis {
    /// Whether the R1CS is uniform (all constraints have the same structure).
    public let isUniform: Bool
    /// Number of distinct constraint structures found.
    public let numDistinctStructures: Int
    /// For each constraint, the structure index it belongs to.
    public let constraintStructureMap: [Int]
    /// The canonical structure for each group: (numATerms, numBTerms, numCTerms).
    public let structures: [(numATerms: Int, numBTerms: Int, numCTerms: Int)]

    public init(isUniform: Bool, numDistinctStructures: Int,
                constraintStructureMap: [Int],
                structures: [(numATerms: Int, numBTerms: Int, numCTerms: Int)]) {
        self.isUniform = isUniform
        self.numDistinctStructures = numDistinctStructures
        self.constraintStructureMap = constraintStructureMap
        self.structures = structures
    }
}

// MARK: - Constraint Evaluation Result

/// Result of batch constraint evaluation over the entire trace.
public struct ConstraintEvaluationResult {
    /// Whether all constraints are satisfied at all active rows.
    public let allSatisfied: Bool
    /// Number of constraint-row pairs checked.
    public let numChecked: Int
    /// Indices of (constraintIndex, row) pairs that failed.
    public let violations: [(constraintIndex: Int, row: Int)]
    /// Per-constraint residuals at each row (only populated if requested).
    public let residuals: [[Fr]]
    /// Wall-clock time for evaluation in seconds.
    public let evaluationTimeSeconds: Double

    public init(allSatisfied: Bool, numChecked: Int,
                violations: [(constraintIndex: Int, row: Int)],
                residuals: [[Fr]], evaluationTimeSeconds: Double) {
        self.allSatisfied = allSatisfied
        self.numChecked = numChecked
        self.violations = violations
        self.residuals = residuals
        self.evaluationTimeSeconds = evaluationTimeSeconds
    }
}

// MARK: - GPU R1CS-to-AIR Compiler Engine

/// GPU-accelerated engine for compiling R1CS constraint systems into AIR format.
///
/// The compilation pipeline:
/// 1. Analyze R1CS structure (uniformity, wire access patterns)
/// 2. Compute optimal trace layout (column reordering for cache efficiency)
/// 3. Build transition constraints from R1CS A, B, C matrices
/// 4. Generate boundary constraints from public inputs
/// 5. Fill execution trace from witness
/// 6. Optionally verify trace satisfies all constraints (GPU-accelerated)
///
/// Usage:
/// ```swift
/// let engine = GPUR1CSToAIRCompilerEngine()
/// let r1cs = R1CSInstance(numConstraints: 3, numVars: 5, numPublic: 1,
///                         aEntries: ..., bEntries: ..., cEntries: ...)
/// let witness = [Fr.one, x, y, z, w]
/// let air = try engine.compile(r1cs: r1cs, witness: witness)
/// assert(air.verify() == nil) // All constraints satisfied
/// ```
public final class GPUR1CSToAIRCompilerEngine {

    /// Metal device (nil if GPU unavailable).
    public let device: MTLDevice?
    public let commandQueue: MTLCommandQueue?

    private var traceGenPipeline: MTLComputePipelineState?
    private var constraintEvalPipeline: MTLComputePipelineState?

    private let threadgroupSize: Int
    private let pool: GPUBufferPool?

    /// Arrays smaller than this threshold are processed on CPU.
    public var cpuThreshold: Int = 256

    /// Whether to optimize trace layout for cache efficiency.
    public var enableLayoutOptimization: Bool = true

    /// Whether to pad trace to power of 2.
    public var padToPowerOfTwo: Bool = true

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
                let lib = try GPUR1CSToAIRCompilerEngine.compileShaders(device: device)
                if let traceGenFn = lib.makeFunction(name: "r1cs_air_trace_gen"),
                   let evalFn = lib.makeFunction(name: "r1cs_air_constraint_eval") {
                    self.traceGenPipeline = try device.makeComputePipelineState(function: traceGenFn)
                    self.constraintEvalPipeline = try device.makeComputePipelineState(function: evalFn)
                }
            } catch {
                self.traceGenPipeline = nil
                self.constraintEvalPipeline = nil
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

        // Sparse entry for trace generation: (row, wireCol, val[8])
        struct TraceGenEntry {
            uint row;
            uint wireCol;
            uint val[8];
        };

        // GPU kernel: evaluate linear combination for each constraint row
        // For each entry, compute val * witness[wireCol] and store product
        // at entry index. Host reduces by (row, matrix) to get A.w, B.w, C.w.
        kernel void r1cs_air_trace_gen(
            device const TraceGenEntry* entries [[buffer(0)]],
            device const uint* witness          [[buffer(1)]],  // Fr[numVars] as uint32[numVars*8]
            device uint* output                 [[buffer(2)]],   // Fr[numEntries] as uint32[numEntries*8]
            device const uint& numEntries       [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= numEntries) return;

            TraceGenEntry e = entries[tid];

            uint va[8], wb[8];
            for (int i = 0; i < 8; i++) {
                va[i] = e.val[i];
                wb[i] = witness[e.wireCol * 8 + i];
            }

            uint prod[8];
            fr_mul(va, wb, prod);

            for (int i = 0; i < 8; i++) {
                output[tid * 8 + i] = prod[i];
            }
        }

        // GPU kernel: batch evaluate constraint residuals
        // For each row: residual = A_val * B_val - C_val
        // Input: a_vals[row], b_vals[row], c_vals[row] (pre-computed linear combos)
        // Output: flags[row] = 1 if residual == 0, else 0
        kernel void r1cs_air_constraint_eval(
            device const uint* a_vals  [[buffer(0)]],  // Fr[numRows]
            device const uint* b_vals  [[buffer(1)]],  // Fr[numRows]
            device const uint* c_vals  [[buffer(2)]],  // Fr[numRows]
            device uint* flags         [[buffer(3)]],   // uint[numRows]
            device const uint& n       [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= n) return;

            uint a[8], b[8], c[8], prod[8];
            for (int i = 0; i < 8; i++) {
                a[i] = a_vals[tid * 8 + i];
                b[i] = b_vals[tid * 8 + i];
                c[i] = c_vals[tid * 8 + i];
            }

            fr_mul(a, b, prod);

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

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(
                    atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        let candidates = [
            execDir + "/Shaders",
            execDir + "/../share/zkMetal/Shaders",
            "./Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/bn254_fr.metal") {
                return c
            }
        }
        return execDir + "/Shaders"
    }

    // MARK: - Wire Access Analysis

    /// Analyze wire access patterns across all R1CS constraints.
    /// Used to determine optimal trace column ordering.
    public func analyzeWireAccess(r1cs: R1CSInstance) -> WireAccessAnalysis {
        var counts = [Int](repeating: 0, count: r1cs.numVars)

        // Count references across all three matrices
        for e in r1cs.aEntries {
            if e.col < r1cs.numVars { counts[e.col] += 1 }
        }
        for e in r1cs.bEntries {
            if e.col < r1cs.numVars { counts[e.col] += 1 }
        }
        for e in r1cs.cEntries {
            if e.col < r1cs.numVars { counts[e.col] += 1 }
        }

        // Sort wire indices by descending frequency
        let sorted = (0..<r1cs.numVars).sorted { counts[$0] > counts[$1] }

        // Count co-occurrences (wires appearing in same constraint)
        var coOccurrences = 0
        var rowWires = [Int: Set<Int>]()
        for e in r1cs.aEntries { rowWires[e.row, default: Set()].insert(e.col) }
        for e in r1cs.bEntries { rowWires[e.row, default: Set()].insert(e.col) }
        for e in r1cs.cEntries { rowWires[e.row, default: Set()].insert(e.col) }
        for (_, wires) in rowWires {
            let n = wires.count
            coOccurrences += n * (n - 1) / 2
        }

        // Average density
        let totalEntries = r1cs.aEntries.count + r1cs.bEntries.count + r1cs.cEntries.count
        let avgDensity = r1cs.numConstraints > 0
            ? Double(totalEntries) / Double(r1cs.numConstraints)
            : 0.0

        return WireAccessAnalysis(
            accessCounts: counts,
            sortedByFrequency: sorted,
            coOccurrenceCount: coOccurrences,
            avgDensity: avgDensity
        )
    }

    // MARK: - Uniformity Analysis

    /// Analyze whether an R1CS system has uniform constraint structure.
    /// Uniform R1CS can use a more compact AIR representation with periodic columns.
    public func analyzeUniformity(r1cs: R1CSInstance) -> R1CSUniformityAnalysis {
        // Build per-constraint structure signatures
        // A constraint's structure is defined by the pattern of non-zero entries
        // (which wire indices appear, not the actual coefficient values).
        var aByRow = [Int: [Int]]()
        var bByRow = [Int: [Int]]()
        var cByRow = [Int: [Int]]()

        for e in r1cs.aEntries { aByRow[e.row, default: []].append(e.col) }
        for e in r1cs.bEntries { bByRow[e.row, default: []].append(e.col) }
        for e in r1cs.cEntries { cByRow[e.row, default: []].append(e.col) }

        // Compute structural signature: (numA, numB, numC) per constraint
        var signatures = [(Int, Int, Int)]()
        for i in 0..<r1cs.numConstraints {
            let na = aByRow[i]?.count ?? 0
            let nb = bByRow[i]?.count ?? 0
            let nc = cByRow[i]?.count ?? 0
            signatures.append((na, nb, nc))
        }

        // Group constraints by signature
        var signatureToIndex = [String: Int]()
        var distinctStructures = [(numATerms: Int, numBTerms: Int, numCTerms: Int)]()
        var constraintMap = [Int]()

        for sig in signatures {
            let key = "\(sig.0),\(sig.1),\(sig.2)"
            if let idx = signatureToIndex[key] {
                constraintMap.append(idx)
            } else {
                let idx = distinctStructures.count
                signatureToIndex[key] = idx
                distinctStructures.append((numATerms: sig.0, numBTerms: sig.1, numCTerms: sig.2))
                constraintMap.append(idx)
            }
        }

        let isUniform = distinctStructures.count <= 1

        return R1CSUniformityAnalysis(
            isUniform: isUniform,
            numDistinctStructures: distinctStructures.count,
            constraintStructureMap: constraintMap,
            structures: distinctStructures
        )
    }

    // MARK: - Trace Layout Computation

    /// Compute optimal trace layout for an R1CS system.
    /// Reorders wire columns by access frequency for cache efficiency.
    public func computeLayout(
        r1cs: R1CSInstance,
        wireAnalysis: WireAccessAnalysis,
        uniformityAnalysis: R1CSUniformityAnalysis
    ) -> AIRTraceLayout {
        let numVars = r1cs.numVars
        let numConstraints = r1cs.numConstraints

        // Compute padded row count (power of 2 >= numConstraints)
        var logRows = 0
        if padToPowerOfTwo {
            var rows = 1
            while rows < numConstraints { rows <<= 1; logRows += 1 }
            if numConstraints == 0 { logRows = 1 } // Minimum 2 rows
        } else {
            logRows = 1
            while (1 << logRows) < numConstraints { logRows += 1 }
        }
        if logRows == 0 { logRows = 1 }

        // First 3 columns: a_val, b_val, c_val
        let aValCol = 0
        let bValCol = 1
        let cValCol = 2
        let wireOffset = 3

        // Build wire-to-column mapping
        var wireToCol = [Int](repeating: 0, count: numVars)
        var colToWire = [Int](repeating: 0, count: numVars)

        if enableLayoutOptimization && numVars > 0 {
            // Reorder by access frequency: most-accessed wires first
            for (newIdx, wireIdx) in wireAnalysis.sortedByFrequency.enumerated() {
                wireToCol[wireIdx] = wireOffset + newIdx
                colToWire[newIdx] = wireIdx
            }
        } else {
            // Identity mapping
            for i in 0..<numVars {
                wireToCol[i] = wireOffset + i
                colToWire[i] = i
            }
        }

        // Selector columns for non-uniform R1CS
        let numSelectors = uniformityAnalysis.isUniform
            ? 0 : uniformityAnalysis.numDistinctStructures
        let selectorOffset = wireOffset + numVars

        let totalCols = wireOffset + numVars + numSelectors

        return AIRTraceLayout(
            numColumns: totalCols,
            logNumRows: logRows,
            wireColumnOffset: wireOffset,
            numWireColumns: numVars,
            aValColumn: aValCol,
            bValColumn: bValCol,
            cValColumn: cValCol,
            selectorColumnOffset: selectorOffset,
            numSelectorColumns: numSelectors,
            wireToColumn: wireToCol,
            columnToWire: colToWire
        )
    }

    // MARK: - Transition Constraint Generation

    /// Build AIR transition constraints from R1CS sparse matrices.
    /// Each R1CS constraint i: (A_i . w) * (B_i . w) = (C_i . w)
    /// becomes an AIR transition constraint checking the same relation
    /// at trace row i.
    public func buildTransitionConstraints(
        r1cs: R1CSInstance,
        uniformityAnalysis: R1CSUniformityAnalysis
    ) -> [R1CSAIRTransitionConstraint] {
        // Group entries by constraint (row)
        var aByRow = [Int: [(wireIndex: Int, coeff: Fr)]]()
        var bByRow = [Int: [(wireIndex: Int, coeff: Fr)]]()
        var cByRow = [Int: [(wireIndex: Int, coeff: Fr)]]()

        for e in r1cs.aEntries {
            aByRow[e.row, default: []].append((wireIndex: e.col, coeff: e.val))
        }
        for e in r1cs.bEntries {
            bByRow[e.row, default: []].append((wireIndex: e.col, coeff: e.val))
        }
        for e in r1cs.cEntries {
            cByRow[e.row, default: []].append((wireIndex: e.col, coeff: e.val))
        }

        var constraints = [R1CSAIRTransitionConstraint]()

        if uniformityAnalysis.isUniform && r1cs.numConstraints > 0 {
            // Uniform: all constraints share the same structure.
            // We emit one constraint that applies to all rows.
            // Use the first constraint's structure as the template.
            let aTerms = aByRow[0] ?? []
            let bTerms = bByRow[0] ?? []
            let cTerms = cByRow[0] ?? []

            constraints.append(R1CSAIRTransitionConstraint(
                constraintIndex: 0,
                aTerms: aTerms,
                bTerms: bTerms,
                cTerms: cTerms,
                degree: 2,
                selectorIndex: nil
            ))
        } else {
            // Non-uniform: each constraint gets its own transition constraint
            // with a selector column to activate it on the correct row.
            for i in 0..<r1cs.numConstraints {
                let aTerms = aByRow[i] ?? []
                let bTerms = bByRow[i] ?? []
                let cTerms = cByRow[i] ?? []

                let selectorIdx = uniformityAnalysis.numDistinctStructures > 1
                    ? uniformityAnalysis.constraintStructureMap[i]
                    : nil

                constraints.append(R1CSAIRTransitionConstraint(
                    constraintIndex: i,
                    aTerms: aTerms,
                    bTerms: bTerms,
                    cTerms: cTerms,
                    degree: 2,
                    selectorIndex: selectorIdx
                ))
            }
        }

        return constraints
    }

    // MARK: - Boundary Constraint Generation

    /// Generate boundary constraints from R1CS public inputs.
    /// Public inputs are fixed at row 0 of the trace.
    public func buildBoundaryConstraints(
        r1cs: R1CSInstance,
        witness: [Fr],
        layout: AIRTraceLayout
    ) -> [R1CSAIRBoundaryConstraint] {
        var boundaries = [R1CSAIRBoundaryConstraint]()

        // Wire 0 is always 1 (constant one wire)
        boundaries.append(R1CSAIRBoundaryConstraint(
            column: layout.wireColumnOffset + 0,
            row: 0,
            value: Fr.one,
            label: "constant_one_wire"
        ))

        // Public inputs (wires 1..numPublic)
        for i in 1...r1cs.numPublic {
            guard i < witness.count else { continue }
            boundaries.append(R1CSAIRBoundaryConstraint(
                column: layout.wireColumnOffset + i,
                row: 0,
                value: witness[i],
                label: "public_input_\(i)"
            ))
        }

        return boundaries
    }

    // MARK: - Trace Generation (CPU)

    /// Generate the execution trace on CPU from R1CS and witness.
    /// Fills all columns: wire assignments, A/B/C evaluated values, selectors.
    public func generateTraceCPU(
        r1cs: R1CSInstance,
        witness: [Fr],
        layout: AIRTraceLayout,
        transitionConstraints: [R1CSAIRTransitionConstraint],
        uniformityAnalysis: R1CSUniformityAnalysis
    ) -> [[Fr]] {
        let numCols = layout.numColumns
        let numRows = layout.numRows

        // Initialize trace with zeros
        var trace = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows),
                           count: numCols)

        // Fill wire assignment columns
        // Every row gets the same wire values (the witness is "broadcast")
        // because R1CS is a single-step system. Each constraint row reads
        // from the same witness vector.
        for wireIdx in 0..<min(r1cs.numVars, witness.count) {
            let col = layout.wireColumnOffset + wireIdx
            guard col < numCols else { continue }
            for row in 0..<numRows {
                trace[col][row] = witness[wireIdx]
            }
        }

        // Compute A.w, B.w, C.w for each constraint row
        // Group entries by row
        var aByRow = [Int: [(col: Int, val: Fr)]]()
        var bByRow = [Int: [(col: Int, val: Fr)]]()
        var cByRow = [Int: [(col: Int, val: Fr)]]()

        for e in r1cs.aEntries { aByRow[e.row, default: []].append((col: e.col, val: e.val)) }
        for e in r1cs.bEntries { bByRow[e.row, default: []].append((col: e.col, val: e.val)) }
        for e in r1cs.cEntries { cByRow[e.row, default: []].append((col: e.col, val: e.val)) }

        for row in 0..<min(r1cs.numConstraints, numRows) {
            // A.w
            var aVal = Fr.zero
            if let entries = aByRow[row] {
                for e in entries {
                    guard e.col < witness.count else { continue }
                    aVal = frAdd(aVal, frMul(e.val, witness[e.col]))
                }
            }
            trace[layout.aValColumn][row] = aVal

            // B.w
            var bVal = Fr.zero
            if let entries = bByRow[row] {
                for e in entries {
                    guard e.col < witness.count else { continue }
                    bVal = frAdd(bVal, frMul(e.val, witness[e.col]))
                }
            }
            trace[layout.bValColumn][row] = bVal

            // C.w
            var cVal = Fr.zero
            if let entries = cByRow[row] {
                for e in entries {
                    guard e.col < witness.count else { continue }
                    cVal = frAdd(cVal, frMul(e.val, witness[e.col]))
                }
            }
            trace[layout.cValColumn][row] = cVal
        }

        // Fill selector columns for non-uniform R1CS
        if !uniformityAnalysis.isUniform && layout.numSelectorColumns > 0 {
            for row in 0..<min(r1cs.numConstraints, numRows) {
                let structIdx = uniformityAnalysis.constraintStructureMap[row]
                let selCol = layout.selectorColumnOffset + structIdx
                if selCol < numCols {
                    trace[selCol][row] = Fr.one
                }
            }
        }

        return trace
    }

    // MARK: - Trace Generation (GPU)

    /// Generate the execution trace using GPU for the linear combination step.
    /// Falls back to CPU for small systems or if GPU is unavailable.
    public func generateTraceGPU(
        r1cs: R1CSInstance,
        witness: [Fr],
        layout: AIRTraceLayout,
        transitionConstraints: [R1CSAIRTransitionConstraint],
        uniformityAnalysis: R1CSUniformityAnalysis
    ) -> [[Fr]] {
        let totalEntries = r1cs.aEntries.count + r1cs.bEntries.count + r1cs.cEntries.count

        // CPU fallback for small systems or no GPU
        guard totalEntries >= cpuThreshold,
              let device = device,
              let queue = commandQueue,
              let pipeline = traceGenPipeline else {
            return generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                    transitionConstraints: transitionConstraints,
                                    uniformityAnalysis: uniformityAnalysis)
        }

        // Pack all entries (A, B, C) with matrix tag for reduction
        struct TaggedEntry {
            let row: Int
            let wireCol: Int
            let val: Fr
            let matrix: Int  // 0=A, 1=B, 2=C
        }

        var allEntries = [TaggedEntry]()
        allEntries.reserveCapacity(totalEntries)
        for e in r1cs.aEntries { allEntries.append(TaggedEntry(row: e.row, wireCol: e.col, val: e.val, matrix: 0)) }
        for e in r1cs.bEntries { allEntries.append(TaggedEntry(row: e.row, wireCol: e.col, val: e.val, matrix: 1)) }
        for e in r1cs.cEntries { allEntries.append(TaggedEntry(row: e.row, wireCol: e.col, val: e.val, matrix: 2)) }

        // Pack entries for GPU: (row: UInt32, wireCol: UInt32, val: 8xUInt32) = 10 uint32s
        let entryStride = 10
        var packed = [UInt32](repeating: 0, count: allEntries.count * entryStride)
        for (i, e) in allEntries.enumerated() {
            let base = i * entryStride
            packed[base] = UInt32(e.row)
            packed[base + 1] = UInt32(e.wireCol)
            let limbs = e.val.to64()
            for j in 0..<4 {
                packed[base + 2 + j * 2] = UInt32(limbs[j] & 0xFFFFFFFF)
                packed[base + 2 + j * 2 + 1] = UInt32(limbs[j] >> 32)
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

        let entryBuf = device.makeBuffer(bytes: packed,
                                          length: packed.count * 4,
                                          options: .storageModeShared)!
        let witnessBuf = device.makeBuffer(bytes: packedWitness,
                                            length: packedWitness.count * 4,
                                            options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: allEntries.count * 8 * 4,
                                           options: .storageModeShared)!
        var numEnt = UInt32(allEntries.count)
        let numBuf = device.makeBuffer(bytes: &numEnt, length: 4, options: .storageModeShared)!

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                    transitionConstraints: transitionConstraints,
                                    uniformityAnalysis: uniformityAnalysis)
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(entryBuf, offset: 0, index: 0)
        enc.setBuffer(witnessBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBuffer(numBuf, offset: 0, index: 3)

        let gridSize = MTLSize(width: allEntries.count, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, allEntries.count), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back products and reduce by (row, matrix)
        let outPtr = outputBuf.contents().bindMemory(to: UInt32.self,
                                                      capacity: allEntries.count * 8)

        let numRows = layout.numRows
        let numCols = layout.numColumns

        var trace = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows),
                           count: numCols)

        // Fill wire columns (broadcast witness)
        for wireIdx in 0..<min(r1cs.numVars, witness.count) {
            let col = layout.wireColumnOffset + wireIdx
            guard col < numCols else { continue }
            for row in 0..<numRows {
                trace[col][row] = witness[wireIdx]
            }
        }

        // Reduce GPU products by (row, matrix)
        for (i, e) in allEntries.enumerated() {
            var limbs = [UInt64](repeating: 0, count: 4)
            for j in 0..<4 {
                let lo = UInt64(outPtr[i * 8 + j * 2])
                let hi = UInt64(outPtr[i * 8 + j * 2 + 1])
                limbs[j] = lo | (hi << 32)
            }
            let product = Fr.from64(limbs)

            let targetCol: Int
            switch e.matrix {
            case 0: targetCol = layout.aValColumn
            case 1: targetCol = layout.bValColumn
            default: targetCol = layout.cValColumn
            }

            if e.row < numRows {
                trace[targetCol][e.row] = frAdd(trace[targetCol][e.row], product)
            }
        }

        // Fill selector columns
        if !uniformityAnalysis.isUniform && layout.numSelectorColumns > 0 {
            for row in 0..<min(r1cs.numConstraints, numRows) {
                let structIdx = uniformityAnalysis.constraintStructureMap[row]
                let selCol = layout.selectorColumnOffset + structIdx
                if selCol < numCols {
                    trace[selCol][row] = Fr.one
                }
            }
        }

        return trace
    }

    // MARK: - Batch Constraint Evaluation

    /// Evaluate all constraints over the trace in batch.
    /// Uses GPU when available for large traces.
    public func evaluateConstraints(
        trace: [[Fr]],
        layout: AIRTraceLayout,
        transitionConstraints: [R1CSAIRTransitionConstraint],
        numActiveRows: Int,
        collectResiduals: Bool = false
    ) -> ConstraintEvaluationResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        var violations = [(constraintIndex: Int, row: Int)]()
        var residuals = [[Fr]]()
        var numChecked = 0

        if collectResiduals {
            residuals = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numActiveRows),
                               count: transitionConstraints.count)
        }

        // Try GPU path for large evaluations
        if numActiveRows >= cpuThreshold,
           let device = device,
           let queue = commandQueue,
           let pipeline = constraintEvalPipeline,
           transitionConstraints.count == 1 || !collectResiduals {
            // GPU batch evaluation: only for the case where we have pre-computed
            // A, B, C values in the trace (columns 0, 1, 2)
            let aVals = trace[layout.aValColumn]
            let bVals = trace[layout.bValColumn]
            let cVals = trace[layout.cValColumn]

            let n = numActiveRows

            var packedA = [UInt32](repeating: 0, count: n * 8)
            var packedB = [UInt32](repeating: 0, count: n * 8)
            var packedC = [UInt32](repeating: 0, count: n * 8)

            for i in 0..<n {
                let aLimbs = aVals[i].to64()
                let bLimbs = bVals[i].to64()
                let cLimbs = cVals[i].to64()
                for j in 0..<4 {
                    packedA[i * 8 + j * 2] = UInt32(aLimbs[j] & 0xFFFFFFFF)
                    packedA[i * 8 + j * 2 + 1] = UInt32(aLimbs[j] >> 32)
                    packedB[i * 8 + j * 2] = UInt32(bLimbs[j] & 0xFFFFFFFF)
                    packedB[i * 8 + j * 2 + 1] = UInt32(bLimbs[j] >> 32)
                    packedC[i * 8 + j * 2] = UInt32(cLimbs[j] & 0xFFFFFFFF)
                    packedC[i * 8 + j * 2 + 1] = UInt32(cLimbs[j] >> 32)
                }
            }

            let aBuf = device.makeBuffer(bytes: packedA, length: packedA.count * 4,
                                          options: .storageModeShared)!
            let bBuf = device.makeBuffer(bytes: packedB, length: packedB.count * 4,
                                          options: .storageModeShared)!
            let cBuf = device.makeBuffer(bytes: packedC, length: packedC.count * 4,
                                          options: .storageModeShared)!
            let flagsBuf = device.makeBuffer(length: n * 4, options: .storageModeShared)!
            var numRows32 = UInt32(n)
            let nBuf = device.makeBuffer(bytes: &numRows32, length: 4,
                                          options: .storageModeShared)!

            if let cmdBuf = queue.makeCommandBuffer(),
               let enc = cmdBuf.makeComputeCommandEncoder() {
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(aBuf, offset: 0, index: 0)
                enc.setBuffer(bBuf, offset: 0, index: 1)
                enc.setBuffer(cBuf, offset: 0, index: 2)
                enc.setBuffer(flagsBuf, offset: 0, index: 3)
                enc.setBuffer(nBuf, offset: 0, index: 4)

                let gridSize = MTLSize(width: n, height: 1, depth: 1)
                let tgSize = MTLSize(width: min(threadgroupSize, n), height: 1, depth: 1)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                let flagsPtr = flagsBuf.contents().bindMemory(to: UInt32.self, capacity: n)
                numChecked = n
                for i in 0..<n {
                    if flagsPtr[i] == 0 {
                        violations.append((constraintIndex: 0, row: i))
                    }
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                return ConstraintEvaluationResult(
                    allSatisfied: violations.isEmpty,
                    numChecked: numChecked,
                    violations: violations,
                    residuals: residuals,
                    evaluationTimeSeconds: elapsed
                )
            }
        }

        // CPU fallback: evaluate each constraint at each row
        for row in 0..<numActiveRows {
            for (ci, tc) in transitionConstraints.enumerated() {
                // Evaluate A.w
                var aVal = Fr.zero
                for (wireIdx, coeff) in tc.aTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    aVal = frAdd(aVal, frMul(coeff, trace[col][row]))
                }

                // Evaluate B.w
                var bVal = Fr.zero
                for (wireIdx, coeff) in tc.bTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    bVal = frAdd(bVal, frMul(coeff, trace[col][row]))
                }

                // Evaluate C.w
                var cVal = Fr.zero
                for (wireIdx, coeff) in tc.cTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    cVal = frAdd(cVal, frMul(coeff, trace[col][row]))
                }

                let product = frMul(aVal, bVal)
                let residual = frSub(product, cVal)
                numChecked += 1

                if collectResiduals {
                    residuals[ci][row] = residual
                }

                if !residual.isZero {
                    violations.append((constraintIndex: ci, row: row))
                }
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return ConstraintEvaluationResult(
            allSatisfied: violations.isEmpty,
            numChecked: numChecked,
            violations: violations,
            residuals: residuals,
            evaluationTimeSeconds: elapsed
        )
    }

    // MARK: - Full Compilation Pipeline

    /// Compile an R1CS system and witness into a complete AIR representation.
    ///
    /// Pipeline:
    /// 1. Analyze wire access patterns and uniformity
    /// 2. Compute optimal trace layout
    /// 3. Build transition and boundary constraints
    /// 4. Generate execution trace (GPU or CPU)
    ///
    /// - Parameters:
    ///   - r1cs: The R1CS constraint system.
    ///   - witness: The full witness vector (including the leading 1).
    ///   - useGPU: Whether to use GPU for trace generation.
    /// - Returns: A compiled AIR with trace and constraints.
    /// - Throws: If the witness is too small for the R1CS system.
    public func compile(
        r1cs: R1CSInstance,
        witness: [Fr],
        useGPU: Bool = true
    ) throws -> CompiledR1CSAIR {
        // Validate witness size
        guard witness.count >= r1cs.numVars else {
            throw R1CSAIRCompilerError.insufficientWitness(
                needed: r1cs.numVars, provided: witness.count)
        }

        // Step 1: Analyze R1CS structure
        let wireAnalysis = analyzeWireAccess(r1cs: r1cs)
        let uniformityAnalysis = analyzeUniformity(r1cs: r1cs)

        // Step 2: Compute layout
        let layout = computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                   uniformityAnalysis: uniformityAnalysis)

        // Step 3: Build constraints
        let transitions = buildTransitionConstraints(
            r1cs: r1cs, uniformityAnalysis: uniformityAnalysis)
        let boundaries = buildBoundaryConstraints(
            r1cs: r1cs, witness: witness, layout: layout)

        // Step 4: Generate trace
        let trace: [[Fr]]
        if useGPU {
            trace = generateTraceGPU(r1cs: r1cs, witness: witness, layout: layout,
                                     transitionConstraints: transitions,
                                     uniformityAnalysis: uniformityAnalysis)
        } else {
            trace = generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                     transitionConstraints: transitions,
                                     uniformityAnalysis: uniformityAnalysis)
        }

        let maxDegree = transitions.map { $0.degree }.max() ?? 0

        return CompiledR1CSAIR(
            layout: layout,
            transitionConstraints: transitions,
            boundaryConstraints: boundaries,
            trace: trace,
            isUniform: uniformityAnalysis.isUniform,
            maxDegree: maxDegree,
            numR1CSConstraints: r1cs.numConstraints,
            numR1CSVariables: r1cs.numVars,
            numPublicInputs: r1cs.numPublic,
            wireAccessCounts: wireAnalysis.accessCounts
        )
    }

    // MARK: - FrAIRExpression Conversion

    /// Convert the transition constraints of a CompiledR1CSAIR into FrAIRExpression form,
    /// suitable for use with GPUAIRConstraintCompiler for STARK proving.
    ///
    /// Each R1CS constraint becomes: A_expr * B_expr - C_expr = 0
    /// where A_expr = sum(a_coeff_j * col(wireOffset + j)).
    public func toFrAIRExpressions(
        air: CompiledR1CSAIR
    ) -> [FrAIRExpression] {
        var expressions = [FrAIRExpression]()

        for tc in air.transitionConstraints {
            // Build A linear combination expression
            let aExpr = buildLinearCombinationExpression(
                tc.aTerms, wireOffset: air.layout.wireColumnOffset)
            // Build B linear combination expression
            let bExpr = buildLinearCombinationExpression(
                tc.bTerms, wireOffset: air.layout.wireColumnOffset)
            // Build C linear combination expression
            let cExpr = buildLinearCombinationExpression(
                tc.cTerms, wireOffset: air.layout.wireColumnOffset)

            // Constraint: A * B - C = 0
            let abProduct = FrAIRExpression.mul(aExpr, bExpr)
            let constraint = FrAIRExpression.sub(abProduct, cExpr)

            expressions.append(constraint)
        }

        return expressions
    }

    /// Build a FrAIRExpression for a linear combination: sum(coeff_i * col(wireOffset + i)).
    private func buildLinearCombinationExpression(
        _ terms: [(wireIndex: Int, coeff: Fr)],
        wireOffset: Int
    ) -> FrAIRExpression {
        guard !terms.isEmpty else {
            return .constant(Fr.zero)
        }

        var result: FrAIRExpression?
        for (wireIdx, coeff) in terms {
            let colExpr = FrAIRExpression.column(wireOffset + wireIdx)
            let term: FrAIRExpression
            if frEqual(coeff, Fr.one) {
                term = colExpr
            } else {
                term = .mul(.constant(coeff), colExpr)
            }

            if let prev = result {
                result = .add(prev, term)
            } else {
                result = term
            }
        }

        return result ?? .constant(Fr.zero)
    }

    // MARK: - Trace Density Analysis

    /// Compute sparsity statistics of the R1CS matrices and resulting trace.
    /// Useful for deciding between GPU and CPU evaluation paths.
    public func computeDensityStats(r1cs: R1CSInstance) -> (
        aDensity: Double, bDensity: Double, cDensity: Double,
        totalNonZeros: Int, avgEntriesPerRow: Double
    ) {
        let n = r1cs.numConstraints
        let m = r1cs.numVars
        let maxEntries = n * m
        guard maxEntries > 0 else {
            return (0, 0, 0, 0, 0)
        }

        let aCount = r1cs.aEntries.count
        let bCount = r1cs.bEntries.count
        let cCount = r1cs.cEntries.count
        let total = aCount + bCount + cCount

        return (
            aDensity: Double(aCount) / Double(maxEntries),
            bDensity: Double(bCount) / Double(maxEntries),
            cDensity: Double(cCount) / Double(maxEntries),
            totalNonZeros: total,
            avgEntriesPerRow: n > 0 ? Double(total) / Double(n) : 0
        )
    }

    // MARK: - Trace Column Reorder

    /// Reorder trace columns using a custom permutation.
    /// This is a post-processing step for cache-aware layout optimization.
    public func reorderTraceColumns(
        _ trace: [[Fr]], permutation: [Int]
    ) -> [[Fr]] {
        guard !trace.isEmpty && !permutation.isEmpty else { return trace }
        let numCols = trace.count
        guard permutation.count == numCols else { return trace }

        var result = [[Fr]](repeating: [], count: numCols)
        for (newIdx, oldIdx) in permutation.enumerated() {
            guard oldIdx < numCols else { continue }
            result[newIdx] = trace[oldIdx]
        }
        return result
    }

    // MARK: - Constraint Composition

    /// Compose multiple transition constraints into a single polynomial
    /// using random challenges (for STARK deep composition).
    /// result = sum(alpha^i * constraint_i) evaluated at each row.
    public func composeConstraints(
        trace: [[Fr]],
        layout: AIRTraceLayout,
        transitionConstraints: [R1CSAIRTransitionConstraint],
        alpha: Fr,
        numActiveRows: Int
    ) -> [Fr] {
        var composed = [Fr](repeating: Fr.zero, count: numActiveRows)

        var alphaPow = Fr.one
        for tc in transitionConstraints {
            for row in 0..<numActiveRows {
                // Evaluate A.w
                var aVal = Fr.zero
                for (wireIdx, coeff) in tc.aTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    aVal = frAdd(aVal, frMul(coeff, trace[col][row]))
                }

                // Evaluate B.w
                var bVal = Fr.zero
                for (wireIdx, coeff) in tc.bTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    bVal = frAdd(bVal, frMul(coeff, trace[col][row]))
                }

                // Evaluate C.w
                var cVal = Fr.zero
                for (wireIdx, coeff) in tc.cTerms {
                    let col = layout.wireColumnOffset + wireIdx
                    guard col < trace.count && row < trace[col].count else { continue }
                    cVal = frAdd(cVal, frMul(coeff, trace[col][row]))
                }

                let residual = frSub(frMul(aVal, bVal), cVal)
                composed[row] = frAdd(composed[row], frMul(alphaPow, residual))
            }
            alphaPow = frMul(alphaPow, alpha)
        }

        return composed
    }

    // MARK: - Vanishing Polynomial Support

    /// Compute the vanishing polynomial evaluation at a point x:
    /// Z_H(x) = x^n - 1, where n is the trace length.
    public func evaluateVanishingPoly(at x: Fr, logTraceLength: Int) -> Fr {
        let n = UInt64(1 << logTraceLength)
        let xn = frPow(x, n)
        return frSub(xn, Fr.one)
    }

    /// Compute the quotient of a constraint composition by the vanishing polynomial
    /// at a point: q(x) = composed(x) / Z_H(x).
    /// Requires that composed(x) is divisible by Z_H(x) (i.e., constraints are satisfied).
    public func evaluateQuotientAt(composedValue: Fr, vanishingValue: Fr) -> Fr {
        if vanishingValue.isZero {
            return Fr.zero // On the trace domain, the quotient is defined via L'Hopital
        }
        return frMul(composedValue, frInverse(vanishingValue))
    }

    // MARK: - Degree Analysis

    /// Compute constraint degree bounds for quotient polynomial sizing.
    public func analyzeDegrees(
        transitionConstraints: [R1CSAIRTransitionConstraint],
        logTraceLength: Int
    ) -> FrConstraintDegreeAnalysis {
        let degrees = transitionConstraints.map { $0.degree }
        return FrConstraintDegreeAnalysis(
            transitionDegrees: degrees,
            logTraceLength: logTraceLength
        )
    }

    // MARK: - Non-Uniform Support

    /// For non-uniform R1CS, build selector polynomials that activate each
    /// constraint structure on the appropriate rows.
    /// Returns one polynomial per distinct constraint structure.
    public func buildSelectorPolynomials(
        uniformityAnalysis: R1CSUniformityAnalysis,
        logTraceLength: Int
    ) -> [[Fr]] {
        let n = 1 << logTraceLength
        let numStructures = uniformityAnalysis.numDistinctStructures

        var selectors = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n),
                               count: numStructures)

        for (row, structIdx) in uniformityAnalysis.constraintStructureMap.enumerated() {
            if row < n && structIdx < numStructures {
                selectors[structIdx][row] = Fr.one
            }
        }

        return selectors
    }

    // MARK: - Trace Validation Helpers

    /// Quick check: verify that A.w * B.w = C.w at all constraint rows in the trace.
    /// This is a lightweight verification that skips building full constraint objects.
    public func quickVerify(
        trace: [[Fr]], layout: AIRTraceLayout, numConstraints: Int
    ) -> Bool {
        let numActive = min(numConstraints, layout.numRows)
        for row in 0..<numActive {
            let aVal = trace[layout.aValColumn][row]
            let bVal = trace[layout.bValColumn][row]
            let cVal = trace[layout.cValColumn][row]

            let product = frMul(aVal, bVal)
            if !frEqual(product, cVal) {
                return false
            }
        }
        return true
    }

    /// Extract the public input values from a compiled AIR trace.
    public func extractPublicInputs(
        air: CompiledR1CSAIR
    ) -> [Fr] {
        var publicInputs = [Fr]()
        for bc in air.boundaryConstraints {
            if bc.label.hasPrefix("public_input_") {
                publicInputs.append(bc.value)
            }
        }
        return publicInputs
    }
}

// MARK: - Errors

/// Errors that can occur during R1CS-to-AIR compilation.
public enum R1CSAIRCompilerError: Error {
    case insufficientWitness(needed: Int, provided: Int)
    case invalidR1CS(String)
    case gpuError(String)
    case traceVerificationFailed(String)
}

// Note: frEqual is defined in LookupEngine.swift and available module-wide.
