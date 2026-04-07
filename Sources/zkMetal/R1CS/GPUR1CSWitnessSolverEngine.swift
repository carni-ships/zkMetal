// GPUR1CSWitnessSolverEngine — GPU-accelerated R1CS witness solver with dependency analysis
//
// Solves R1CS systems: given A, B, C matrices and public inputs, compute full witness.
// Features:
//   - Topological sort of constraint dependencies for optimal solve order
//   - Forward propagation through linear constraints
//   - Hint-based solving for non-linear constraints (division, comparison)
//   - GPU-accelerated batch field operations during witness computation
//   - Incremental witness extension (add constraints, re-solve)
//   - Witness validation: verify A*w . B*w = C*w
//
// Wire layout (Circom convention):
//   w[0] = 1 (constant "one" wire)
//   w[1..numPublic] = public inputs
//   w[numPublic+1..] = private witness
//
// Public API:
//   solve(system:publicInputs:hints:)         — full witness solve with dependency ordering
//   solveIncremental(state:newConstraints:)    — extend existing witness with new constraints
//   validate(system:witness:)                  — verify A*w . B*w = C*w
//   topologicalOrder(system:)                  — compute optimal constraint evaluation order
//   batchFieldMul(a:b:)                        — GPU-accelerated batch field multiply
//   batchFieldAdd(a:b:)                        — GPU-accelerated batch field add

import Foundation
import Metal
import NeonFieldOps

// MARK: - Witness Solver Types

/// A sparse linear combination: sum of (coefficient * variable_index) terms.
public struct WitnessLinearCombination {
    public let terms: [(varIdx: Int, coeff: Fr)]

    public init(terms: [(varIdx: Int, coeff: Fr)]) {
        self.terms = terms
    }

    /// All variable indices referenced by this linear combination.
    public var variableIndices: [Int] {
        terms.map { $0.varIdx }
    }

    /// Whether this LC is a single term (monomial).
    public var isSingleTerm: Bool { terms.count == 1 }

    /// Whether this LC is empty (zero).
    public var isEmpty: Bool { terms.isEmpty }
}

/// An R1CS constraint in structured form: A . w * B . w = C . w
public struct WitnessConstraint {
    public let a: WitnessLinearCombination
    public let b: WitnessLinearCombination
    public let c: WitnessLinearCombination
    public let index: Int

    public init(a: WitnessLinearCombination, b: WitnessLinearCombination,
                c: WitnessLinearCombination, index: Int) {
        self.a = a
        self.b = b
        self.c = c
        self.index = index
    }

    /// All variable indices referenced by this constraint.
    public var allVariables: Set<Int> {
        var s = Set<Int>()
        for t in a.terms { s.insert(t.varIdx) }
        for t in b.terms { s.insert(t.varIdx) }
        for t in c.terms { s.insert(t.varIdx) }
        return s
    }

    /// Variables that could potentially be solved by this constraint.
    /// For A*B=C: variables appearing only in C are candidates if A and B are fully known.
    public var outputCandidates: Set<Int> {
        let aVars = Set(a.variableIndices)
        let bVars = Set(b.variableIndices)
        let cVars = Set(c.variableIndices)
        // C-only variables are primary outputs; also consider single-unknown in A or B
        return cVars.subtracting(aVars).subtracting(bVars)
    }
}

/// A hint function for solving non-linear constraints.
/// Given the current (partial) witness, returns a value for a specific variable.
public struct WitnessHint {
    /// The variable index this hint provides.
    public let targetVar: Int
    /// The computation: given current witness values, produce the target value.
    public let compute: ([Fr]) -> Fr

    public init(targetVar: Int, compute: @escaping ([Fr]) -> Fr) {
        self.targetVar = targetVar
        self.compute = compute
    }
}

/// Complete R1CS system for witness solving.
public struct WitnessSolverSystem {
    public let constraints: [WitnessConstraint]
    public let numVars: Int
    public let numPublic: Int  // not counting wire 0

    public init(constraints: [WitnessConstraint], numVars: Int, numPublic: Int) {
        self.constraints = constraints
        self.numVars = numVars
        self.numPublic = numPublic
    }

    /// Build from an R1CSInstance (flat sparse entries).
    public init(from r1cs: R1CSInstance) {
        self.numVars = r1cs.numVars
        self.numPublic = r1cs.numPublic
        let n = r1cs.numConstraints

        // Group entries by row for each matrix
        var aByRow = [Int: [(varIdx: Int, coeff: Fr)]]()
        var bByRow = [Int: [(varIdx: Int, coeff: Fr)]]()
        var cByRow = [Int: [(varIdx: Int, coeff: Fr)]]()

        for e in r1cs.aEntries {
            aByRow[e.row, default: []].append((varIdx: e.col, coeff: e.val))
        }
        for e in r1cs.bEntries {
            bByRow[e.row, default: []].append((varIdx: e.col, coeff: e.val))
        }
        for e in r1cs.cEntries {
            cByRow[e.row, default: []].append((varIdx: e.col, coeff: e.val))
        }

        var result = [WitnessConstraint]()
        result.reserveCapacity(n)
        for i in 0..<n {
            let aLC = WitnessLinearCombination(terms: aByRow[i] ?? [])
            let bLC = WitnessLinearCombination(terms: bByRow[i] ?? [])
            let cLC = WitnessLinearCombination(terms: cByRow[i] ?? [])
            result.append(WitnessConstraint(a: aLC, b: bLC, c: cLC, index: i))
        }
        self.constraints = result
    }
}

/// Result of witness solving.
public struct WitnessSolverEngineResult {
    /// Full witness vector: [1, pub_1, ..., pub_n, priv_1, ..., priv_m].
    public let witness: [Fr]
    /// Whether every variable was resolved.
    public let isComplete: Bool
    /// Total solver iterations.
    public let iterations: Int
    /// Indices of variables that remain unsolved.
    public let unresolvedIndices: [Int]
    /// Constraint evaluation order used (topological sort).
    public let evaluationOrder: [Int]
    /// Number of variables resolved by hints.
    public let hintSolvedCount: Int

    public init(witness: [Fr], isComplete: Bool, iterations: Int,
                unresolvedIndices: [Int], evaluationOrder: [Int], hintSolvedCount: Int) {
        self.witness = witness
        self.isComplete = isComplete
        self.iterations = iterations
        self.unresolvedIndices = unresolvedIndices
        self.evaluationOrder = evaluationOrder
        self.hintSolvedCount = hintSolvedCount
    }
}

/// Result of witness validation.
public struct WitnessValidationResult {
    /// Whether all constraints are satisfied.
    public let valid: Bool
    /// Number of constraints checked.
    public let numConstraints: Int
    /// Indices of violated constraints.
    public let violatedConstraints: [Int]
    /// Per-constraint residual values.
    public let residuals: [Fr]
    /// Maximum residual norm (0 if all satisfied).
    public let maxResidualIndex: Int

    public init(valid: Bool, numConstraints: Int, violatedConstraints: [Int],
                residuals: [Fr], maxResidualIndex: Int) {
        self.valid = valid
        self.numConstraints = numConstraints
        self.violatedConstraints = violatedConstraints
        self.residuals = residuals
        self.maxResidualIndex = maxResidualIndex
    }
}

/// Mutable state for incremental witness solving.
public struct IncrementalWitnessState {
    /// Current witness values.
    public var witness: [Fr]
    /// Which variables are resolved.
    public var solved: [Bool]
    /// Active constraints.
    public var constraints: [WitnessConstraint]
    /// Active hints.
    public var hints: [WitnessHint]
    /// Total variables.
    public var numVars: Int
    /// Number of public inputs.
    public var numPublic: Int

    public init(numVars: Int, numPublic: Int) {
        self.numVars = numVars
        self.numPublic = numPublic
        self.witness = [Fr](repeating: .zero, count: numVars)
        self.solved = [Bool](repeating: false, count: numVars)
        self.constraints = []
        self.hints = []
        // Wire 0 = 1
        self.witness[0] = Fr.one
        self.solved[0] = true
    }
}

/// Dependency graph node for topological sorting.
private struct DepNode {
    var constraintIdx: Int
    var dependsOn: Set<Int>     // variable indices needed as input
    var produces: Set<Int>      // variable indices this constraint can solve
    var inDegree: Int = 0       // number of unresolved dependencies
}

// MARK: - GPU R1CS Witness Solver Engine

public final class GPUR1CSWitnessSolverEngine {

    /// Metal device (nil if GPU unavailable).
    public let device: MTLDevice?
    public let commandQueue: MTLCommandQueue?

    private var batchMulPipeline: MTLComputePipelineState?
    private var batchAddPipeline: MTLComputePipelineState?
    private var batchSubPipeline: MTLComputePipelineState?

    private let threadgroupSize: Int

    /// Threshold below which CPU is used instead of GPU for batch ops.
    public var gpuThreshold: Int = 512

    /// Maximum solver iterations.
    public var maxIterations: Int = 2000

    // MARK: - Initialization

    public init(threadgroupSize: Int = 256) {
        self.threadgroupSize = threadgroupSize

        if let device = MTLCreateSystemDefaultDevice(),
           let queue = device.makeCommandQueue() {
            self.device = device
            self.commandQueue = queue

            do {
                let lib = try GPUR1CSWitnessSolverEngine.compileShaders(device: device)
                if let mulFn = lib.makeFunction(name: "witness_batch_mul"),
                   let addFn = lib.makeFunction(name: "witness_batch_add"),
                   let subFn = lib.makeFunction(name: "witness_batch_sub") {
                    self.batchMulPipeline = try device.makeComputePipelineState(function: mulFn)
                    self.batchAddPipeline = try device.makeComputePipelineState(function: addFn)
                    self.batchSubPipeline = try device.makeComputePipelineState(function: subFn)
                }
            } catch {
                self.batchMulPipeline = nil
                self.batchAddPipeline = nil
                self.batchSubPipeline = nil
            }
        } else {
            self.device = nil
            self.commandQueue = nil
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

        // Batch element-wise field multiply: out[i] = a[i] * b[i]
        kernel void witness_batch_mul(
            device const uint* a  [[buffer(0)]],
            device const uint* b  [[buffer(1)]],
            device uint* out      [[buffer(2)]],
            device const uint& n  [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= n) return;
            uint va[8], vb[8], vr[8];
            for (int i = 0; i < 8; i++) {
                va[i] = a[tid * 8 + i];
                vb[i] = b[tid * 8 + i];
            }
            fr_mul(va, vb, vr);
            for (int i = 0; i < 8; i++) {
                out[tid * 8 + i] = vr[i];
            }
        }

        // Batch element-wise field add: out[i] = a[i] + b[i]
        kernel void witness_batch_add(
            device const uint* a  [[buffer(0)]],
            device const uint* b  [[buffer(1)]],
            device uint* out      [[buffer(2)]],
            device const uint& n  [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= n) return;
            uint va[8], vb[8], vr[8];
            for (int i = 0; i < 8; i++) {
                va[i] = a[tid * 8 + i];
                vb[i] = b[tid * 8 + i];
            }
            fr_add(va, vb, vr);
            for (int i = 0; i < 8; i++) {
                out[tid * 8 + i] = vr[i];
            }
        }

        // Batch element-wise field subtract: out[i] = a[i] - b[i]
        kernel void witness_batch_sub(
            device const uint* a  [[buffer(0)]],
            device const uint* b  [[buffer(1)]],
            device uint* out      [[buffer(2)]],
            device const uint& n  [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= n) return;
            uint va[8], vb[8], vr[8];
            for (int i = 0; i < 8; i++) {
                va[i] = a[tid * 8 + i];
                vb[i] = b[tid * 8 + i];
            }
            fr_sub(va, vb, vr);
            for (int i = 0; i < 8; i++) {
                out[tid * 8 + i] = vr[i];
            }
        }
        """

        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        return try device.makeLibrary(source: kernelSource, options: opts)
    }

    // MARK: - Topological Sort

    /// Compute optimal constraint evaluation order via topological sort.
    /// Constraints whose inputs are already known (public inputs, wire 0) come first.
    public func topologicalOrder(system: WitnessSolverSystem,
                                 knownVars: Set<Int>) -> [Int] {
        let n = system.constraints.count
        if n == 0 { return [] }

        // Build dependency nodes
        var nodes = [DepNode]()
        nodes.reserveCapacity(n)

        // Track which variables each constraint can produce
        // A constraint can solve a variable in C if A and B are fully known
        // Or a variable in A if B and C are fully known, etc.
        for constraint in system.constraints {
            let aVars = Set(constraint.a.variableIndices)
            let bVars = Set(constraint.b.variableIndices)
            let cVars = Set(constraint.c.variableIndices)
            let allVars = aVars.union(bVars).union(cVars)

            // Dependencies: all variables this constraint reads
            let deps = allVars.subtracting(knownVars)

            // Outputs: variables that could be solved
            // Primary: C-only variables (when A, B are fully known)
            var produces = Set<Int>()
            for v in cVars {
                if !aVars.contains(v) && !bVars.contains(v) {
                    produces.insert(v)
                }
            }
            // Secondary: single-unknown in A or B
            if aVars.count == 1, let v = aVars.first, !knownVars.contains(v) {
                produces.insert(v)
            }
            if bVars.count == 1, let v = bVars.first, !knownVars.contains(v) {
                produces.insert(v)
            }

            nodes.append(DepNode(constraintIdx: constraint.index,
                                 dependsOn: deps,
                                 produces: produces))
        }

        // Kahn's algorithm: constraints with no unsolved dependencies go first
        var inDegree = [Int](repeating: 0, count: n)
        var resolved = knownVars
        var queue = [Int]()
        var order = [Int]()
        order.reserveCapacity(n)

        // Compute initial in-degrees
        for i in 0..<n {
            let unresolved = nodes[i].dependsOn.subtracting(resolved)
            inDegree[i] = unresolved.count
            if inDegree[i] == 0 {
                queue.append(i)
            }
        }

        var head = 0
        while head < queue.count {
            let idx = queue[head]
            head += 1
            order.append(nodes[idx].constraintIdx)

            // Mark outputs as resolved
            let newlyResolved = nodes[idx].produces
            resolved.formUnion(newlyResolved)

            // Update in-degrees
            for i in 0..<n {
                if inDegree[i] > 0 {
                    let remaining = nodes[i].dependsOn.subtracting(resolved)
                    let newDeg = remaining.count
                    if newDeg == 0 && inDegree[i] > 0 {
                        queue.append(i)
                    }
                    inDegree[i] = newDeg
                }
            }
        }

        // Append any constraints not yet ordered (cyclic dependencies)
        if order.count < n {
            let ordered = Set(order)
            for i in 0..<n {
                if !ordered.contains(nodes[i].constraintIdx) {
                    order.append(nodes[i].constraintIdx)
                }
            }
        }

        return order
    }

    // MARK: - Core Solver

    /// Solve witness for an R1CS system with dependency-ordered propagation.
    public func solve(system: WitnessSolverSystem,
                      publicInputs: [Fr],
                      hints: [WitnessHint] = []) -> WitnessSolverEngineResult {
        let numVars = system.numVars
        var witness = [Fr](repeating: .zero, count: numVars)
        var solved = [Bool](repeating: false, count: numVars)

        // Initialize wire 0 = 1
        witness[0] = Fr.one
        solved[0] = true

        // Set public inputs (wires 1..numPublic)
        for i in 0..<min(publicInputs.count, system.numPublic) {
            witness[1 + i] = publicInputs[i]
            solved[1 + i] = true
        }

        // Apply hints
        var hintSolved = 0
        for hint in hints {
            if hint.targetVar >= 0 && hint.targetVar < numVars && !solved[hint.targetVar] {
                witness[hint.targetVar] = hint.compute(witness)
                solved[hint.targetVar] = true
                hintSolved += 1
            }
        }

        // Compute known set for topological ordering
        var knownVars = Set<Int>()
        for i in 0..<numVars where solved[i] {
            knownVars.insert(i)
        }

        // Get optimal evaluation order
        let evalOrder = topologicalOrder(system: system, knownVars: knownVars)

        // Build constraint lookup by index
        var constraintByIdx = [Int: WitnessConstraint]()
        for c in system.constraints {
            constraintByIdx[c.index] = c
        }

        // Iterative solver: follow topological order, repeat until no progress
        var iterations = 0
        var progress = true

        while progress && iterations < maxIterations {
            progress = false
            iterations += 1

            for cIdx in evalOrder {
                guard let constraint = constraintByIdx[cIdx] else { continue }

                if solveConstraint(constraint, witness: &witness, solved: &solved) {
                    progress = true
                }
            }

            // Re-apply hints with updated witness (hints may depend on solved vars)
            for hint in hints {
                if hint.targetVar >= 0 && hint.targetVar < numVars && !solved[hint.targetVar] {
                    witness[hint.targetVar] = hint.compute(witness)
                    solved[hint.targetVar] = true
                    hintSolved += 1
                    progress = true
                }
            }
        }

        // Collect unsolved
        var unsolved = [Int]()
        for i in 0..<numVars where !solved[i] {
            unsolved.append(i)
        }

        return WitnessSolverEngineResult(
            witness: witness,
            isComplete: unsolved.isEmpty,
            iterations: iterations,
            unresolvedIndices: unsolved,
            evaluationOrder: evalOrder,
            hintSolvedCount: hintSolved
        )
    }

    /// Solve a single R1CS constraint for an unknown variable.
    /// Returns true if a new variable was resolved.
    private func solveConstraint(_ constraint: WitnessConstraint,
                                 witness: inout [Fr],
                                 solved: inout [Bool]) -> Bool {
        let evalA = evaluateLC(constraint.a, witness: witness, solved: solved)
        let evalB = evaluateLC(constraint.b, witness: witness, solved: solved)
        let evalC = evaluateLC(constraint.c, witness: witness, solved: solved)

        // Case 1: A and B fully known, C has exactly 1 unknown
        // a_val * b_val = c_known + coeff * x  =>  x = (a*b - c_known) / coeff
        if evalA.unknownCount == 0 && evalB.unknownCount == 0 && evalC.unknownCount == 1 {
            let ab = frMul(evalA.value, evalB.value)
            let diff = frSub(ab, evalC.value)
            if !evalC.unknownCoeff.isZero {
                let inv = frInverse(evalC.unknownCoeff)
                witness[evalC.unknownIdx] = frMul(diff, inv)
                solved[evalC.unknownIdx] = true
                return true
            }
        }

        // Case 2: A has 1 unknown, B and C fully known, B nonzero
        // (a_known + coeff*x) * b_val = c_val  =>  x = (c_val - a_known*b_val) / (coeff*b_val)
        if evalA.unknownCount == 1 && evalB.unknownCount == 0 && evalC.unknownCount == 0 {
            let bVal = evalB.value
            if !bVal.isZero {
                let rhs = frSub(evalC.value, frMul(evalA.value, bVal))
                let denom = frMul(evalA.unknownCoeff, bVal)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalA.unknownIdx] = frMul(rhs, inv)
                    solved[evalA.unknownIdx] = true
                    return true
                }
            }
        }

        // Case 3: B has 1 unknown, A and C fully known, A nonzero
        // a_val * (b_known + coeff*x) = c_val  =>  x = (c_val - a_val*b_known) / (coeff*a_val)
        if evalB.unknownCount == 1 && evalA.unknownCount == 0 && evalC.unknownCount == 0 {
            let aVal = evalA.value
            if !aVal.isZero {
                let rhs = frSub(evalC.value, frMul(aVal, evalB.value))
                let denom = frMul(evalB.unknownCoeff, aVal)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalB.unknownIdx] = frMul(rhs, inv)
                    solved[evalB.unknownIdx] = true
                    return true
                }
            }
        }

        // Case 4: A and B each have 1 unknown (same variable), C fully known
        // Handles x*x = c type constraints where A and B reference the same var
        if evalA.unknownCount == 1 && evalB.unknownCount == 1 &&
           evalA.unknownIdx == evalB.unknownIdx && evalC.unknownCount == 0 {
            let aKnown = evalA.value
            let bKnown = evalB.value

            // If one side's known part is zero, it becomes linear:
            // coeff*x * b_known = c_val  =>  x = c_val / (coeff * b_known)
            if aKnown.isZero && !bKnown.isZero {
                let denom = frMul(evalA.unknownCoeff, bKnown)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalA.unknownIdx] = frMul(evalC.value, inv)
                    solved[evalA.unknownIdx] = true
                    return true
                }
            } else if bKnown.isZero && !aKnown.isZero {
                let denom = frMul(evalB.unknownCoeff, aKnown)
                if !denom.isZero {
                    let inv = frInverse(denom)
                    witness[evalB.unknownIdx] = frMul(evalC.value, inv)
                    solved[evalB.unknownIdx] = true
                    return true
                }
            }
        }

        return false
    }

    /// Evaluate a linear combination with the current witness.
    private func evaluateLC(_ lc: WitnessLinearCombination,
                            witness: [Fr],
                            solved: [Bool])
        -> (value: Fr, unknownCount: Int, unknownIdx: Int, unknownCoeff: Fr) {
        var value = Fr.zero
        var unknownCount = 0
        var unknownIdx = -1
        var unknownCoeff = Fr.zero

        for term in lc.terms {
            guard term.varIdx < witness.count else { continue }
            if solved[term.varIdx] {
                value = frAdd(value, frMul(term.coeff, witness[term.varIdx]))
            } else {
                unknownCount += 1
                unknownIdx = term.varIdx
                unknownCoeff = term.coeff
            }
        }
        return (value, unknownCount, unknownIdx, unknownCoeff)
    }

    // MARK: - Incremental Solving

    /// Create an initial incremental state from public inputs.
    public func makeIncrementalState(numVars: Int, numPublic: Int,
                                     publicInputs: [Fr]) -> IncrementalWitnessState {
        var state = IncrementalWitnessState(numVars: numVars, numPublic: numPublic)
        for i in 0..<min(publicInputs.count, numPublic) {
            state.witness[1 + i] = publicInputs[i]
            state.solved[1 + i] = true
        }
        return state
    }

    /// Add new constraints to an incremental state and re-solve.
    /// Returns the updated result.
    public func solveIncremental(state: inout IncrementalWitnessState,
                                 newConstraints: [WitnessConstraint]) -> WitnessSolverEngineResult {
        state.constraints.append(contentsOf: newConstraints)

        let system = WitnessSolverSystem(constraints: state.constraints,
                                          numVars: state.numVars,
                                          numPublic: state.numPublic)

        // Build partial known set from already-solved variables
        var pub = [Fr]()
        for i in 0..<state.numPublic {
            pub.append(state.witness[1 + i])
        }

        // Create hints from already-solved private variables
        var hints = state.hints
        for i in (1 + state.numPublic)..<state.numVars {
            if state.solved[i] {
                let val = state.witness[i]
                hints.append(WitnessHint(targetVar: i, compute: { _ in val }))
            }
        }

        let result = solve(system: system, publicInputs: pub, hints: hints)

        // Update state from result
        for i in 0..<state.numVars {
            if !state.solved[i] {
                state.witness[i] = result.witness[i]
                state.solved[i] = (i < result.witness.count) &&
                    !result.unresolvedIndices.contains(i)
            }
        }

        return result
    }

    // MARK: - Witness Validation

    /// Validate that a witness satisfies all constraints: A*w . B*w = C*w.
    public func validate(system: WitnessSolverSystem,
                         witness: [Fr]) -> WitnessValidationResult {
        let n = system.constraints.count
        guard witness.count >= system.numVars else {
            return WitnessValidationResult(valid: false, numConstraints: n,
                                            violatedConstraints: Array(0..<n),
                                            residuals: [Fr](repeating: .zero, count: n),
                                            maxResidualIndex: 0)
        }

        var residuals = [Fr](repeating: .zero, count: n)
        var violated = [Int]()
        var maxIdx = 0

        for (i, constraint) in system.constraints.enumerated() {
            let aVal = evaluateLCFull(constraint.a, witness: witness)
            let bVal = evaluateLCFull(constraint.b, witness: witness)
            let cVal = evaluateLCFull(constraint.c, witness: witness)
            let ab = frMul(aVal, bVal)
            residuals[i] = frSub(ab, cVal)
            if !residuals[i].isZero {
                violated.append(i)
                maxIdx = i
            }
        }

        return WitnessValidationResult(valid: violated.isEmpty,
                                        numConstraints: n,
                                        violatedConstraints: violated,
                                        residuals: residuals,
                                        maxResidualIndex: violated.isEmpty ? -1 : maxIdx)
    }

    /// Evaluate a linear combination fully (all variables must be known).
    private func evaluateLCFull(_ lc: WitnessLinearCombination, witness: [Fr]) -> Fr {
        var value = Fr.zero
        for term in lc.terms {
            guard term.varIdx < witness.count else { continue }
            value = frAdd(value, frMul(term.coeff, witness[term.varIdx]))
        }
        return value
    }

    // MARK: - Validate from R1CSInstance

    /// Convenience: validate using an R1CSInstance directly.
    public func validateR1CS(r1cs: R1CSInstance, witness: [Fr]) -> WitnessValidationResult {
        let system = WitnessSolverSystem(from: r1cs)
        return validate(system: system, witness: witness)
    }

    // MARK: - GPU Batch Field Operations

    /// GPU-accelerated batch field multiplication: out[i] = a[i] * b[i].
    public func batchFieldMul(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        if n < gpuThreshold { return cpuBatchMul(a, b, count: n) }
        guard let result = gpuBatchOp(a, b, count: n, pipeline: batchMulPipeline) else {
            return cpuBatchMul(a, b, count: n)
        }
        return result
    }

    /// GPU-accelerated batch field addition: out[i] = a[i] + b[i].
    public func batchFieldAdd(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        if n < gpuThreshold { return cpuBatchAdd(a, b, count: n) }
        guard let result = gpuBatchOp(a, b, count: n, pipeline: batchAddPipeline) else {
            return cpuBatchAdd(a, b, count: n)
        }
        return result
    }

    /// GPU-accelerated batch field subtraction: out[i] = a[i] - b[i].
    public func batchFieldSub(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let n = min(a.count, b.count)
        if n == 0 { return [] }
        if n < gpuThreshold { return cpuBatchSub(a, b, count: n) }
        guard let result = gpuBatchOp(a, b, count: n, pipeline: batchSubPipeline) else {
            return cpuBatchSub(a, b, count: n)
        }
        return result
    }

    // MARK: - GPU Dispatch

    private func gpuBatchOp(_ a: [Fr], _ b: [Fr], count n: Int,
                            pipeline: MTLComputePipelineState?) -> [Fr]? {
        guard let device = device, let queue = commandQueue, let pipeline = pipeline else {
            return nil
        }

        // Pack Fr arrays to UInt32 buffers
        let packedA = packFrArray(a, count: n)
        let packedB = packFrArray(b, count: n)

        guard let bufA = device.makeBuffer(bytes: packedA, length: n * 8 * 4, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: packedB, length: n * 8 * 4, options: .storageModeShared),
              let bufOut = device.makeBuffer(length: n * 8 * 4, options: .storageModeShared) else {
            return nil
        }

        var count = UInt32(n)
        guard let bufN = device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared),
              let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.setBuffer(bufN, offset: 0, index: 3)

        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(threadgroupSize, n), height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return unpackFrArray(bufOut, count: n)
    }

    // MARK: - Packing Utilities

    private func packFrArray(_ arr: [Fr], count n: Int) -> [UInt32] {
        var packed = [UInt32](repeating: 0, count: n * 8)
        for i in 0..<n {
            let limbs = arr[i].to64()
            for j in 0..<4 {
                packed[i * 8 + j * 2] = UInt32(limbs[j] & 0xFFFFFFFF)
                packed[i * 8 + j * 2 + 1] = UInt32(limbs[j] >> 32)
            }
        }
        return packed
    }

    private func unpackFrArray(_ buffer: MTLBuffer, count n: Int) -> [Fr] {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: n * 8)
        var result = [Fr](repeating: .zero, count: n)
        for i in 0..<n {
            var limbs = [UInt64](repeating: 0, count: 4)
            for j in 0..<4 {
                let lo = UInt64(ptr[i * 8 + j * 2])
                let hi = UInt64(ptr[i * 8 + j * 2 + 1])
                limbs[j] = lo | (hi << 32)
            }
            result[i] = Fr.from64(limbs)
        }
        return result
    }

    // MARK: - CPU Fallback Batch Operations

    private func cpuBatchMul(_ a: [Fr], _ b: [Fr], count n: Int) -> [Fr] {
        var result = [Fr](repeating: .zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_parallel(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    private func cpuBatchAdd(_ a: [Fr], _ b: [Fr], count n: Int) -> [Fr] {
        var result = [Fr](repeating: .zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_add_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    private func cpuBatchSub(_ a: [Fr], _ b: [Fr], count n: Int) -> [Fr] {
        var result = [Fr](repeating: .zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_sub_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    // MARK: - Convenience: Solve from R1CSInstance

    /// Convenience: solve from an R1CSInstance directly.
    public func solveFromR1CS(r1cs: R1CSInstance,
                              publicInputs: [Fr],
                              hints: [WitnessHint] = []) -> WitnessSolverEngineResult {
        let system = WitnessSolverSystem(from: r1cs)
        return solve(system: system, publicInputs: publicInputs, hints: hints)
    }

    // MARK: - Dependency Analysis

    /// Analyze which variables each constraint depends on and produces.
    /// Returns (constraintIndex, inputVars, outputVars) for each constraint.
    public func analyzeDependencies(system: WitnessSolverSystem)
        -> [(constraintIdx: Int, inputs: Set<Int>, outputs: Set<Int>)] {
        var result = [(constraintIdx: Int, inputs: Set<Int>, outputs: Set<Int>)]()
        result.reserveCapacity(system.constraints.count)

        for constraint in system.constraints {
            let aVars = Set(constraint.a.variableIndices)
            let bVars = Set(constraint.b.variableIndices)
            let cVars = Set(constraint.c.variableIndices)
            let inputs = aVars.union(bVars)
            let outputs = cVars.subtracting(inputs)
            result.append((constraintIdx: constraint.index, inputs: inputs, outputs: outputs))
        }
        return result
    }

    // MARK: - Helper: Build Constraint

    /// Build a WitnessConstraint from sparse entry tuples.
    public static func makeConstraint(
        a: [(col: Int, val: Fr)],
        b: [(col: Int, val: Fr)],
        c: [(col: Int, val: Fr)],
        index: Int
    ) -> WitnessConstraint {
        let aLC = WitnessLinearCombination(terms: a.map { (varIdx: $0.col, coeff: $0.val) })
        let bLC = WitnessLinearCombination(terms: b.map { (varIdx: $0.col, coeff: $0.val) })
        let cLC = WitnessLinearCombination(terms: c.map { (varIdx: $0.col, coeff: $0.val) })
        return WitnessConstraint(a: aLC, b: bLC, c: cLC, index: index)
    }

    /// Build a WitnessSolverSystem from constraint tuples.
    public static func makeSystem(
        constraints: [WitnessConstraint],
        numVars: Int,
        numPublic: Int
    ) -> WitnessSolverSystem {
        return WitnessSolverSystem(constraints: constraints, numVars: numVars,
                                    numPublic: numPublic)
    }
}
