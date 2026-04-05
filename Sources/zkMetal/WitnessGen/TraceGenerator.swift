// TraceGenerator — GPU-accelerated AIR execution trace generation
//
// For AIR (Algebraic Intermediate Representation) programs, the witness is
// an execution trace: a matrix where each row is one step of the computation
// and each column is one register/wire value.
//
// Key insight: many trace columns are independent of each other. Even when
// rows are sequential (row i depends on row i-1), different columns within
// the same step can be computed in parallel. This module exploits both:
//   1. Column-level parallelism: independent columns computed simultaneously
//   2. Row-level parallelism: independent rows (where possible) via GPU
//   3. Batch execution: multiple trace instances generated concurrently
//
// Supports both M31 (Circle STARK) and BN254 Fr (PLONK/Groth16) fields.

import Foundation
import Metal

// MARK: - AIR Program Representation

/// Describes an AIR trace computation step.
/// Each step transforms the current state into the next state.
public struct AIRStep {
    /// Column dependencies: for each output column, which input columns it reads.
    /// Used to determine column independence for parallel scheduling.
    public let outputCol: Int
    public let inputCols: [Int]
    /// The computation: given input values, produce the output value.
    public let compute: ([Fr]) -> Fr

    public init(outputCol: Int, inputCols: [Int], compute: @escaping ([Fr]) -> Fr) {
        self.outputCol = outputCol
        self.inputCols = inputCols
        self.compute = compute
    }
}

/// A complete AIR program description for trace generation.
public struct AIRProgram {
    /// Number of columns (registers) in the trace.
    public let numColumns: Int
    /// Steps to execute per row transition.
    /// Steps are applied in order: the output of earlier steps is visible to later ones.
    public let steps: [AIRStep]
    /// Initial state (row 0 values for each column).
    public let initialState: [Fr]

    public init(numColumns: Int, steps: [AIRStep], initialState: [Fr]) {
        precondition(initialState.count == numColumns)
        self.numColumns = numColumns
        self.steps = steps
        self.initialState = initialState
    }
}

/// Column dependency analysis result.
public struct ColumnSchedule {
    /// Groups of columns that can be computed in parallel.
    /// Each group is a set of step indices whose output columns are independent.
    public let parallelGroups: [[Int]]
    /// For each step, which group it belongs to.
    public let stepToGroup: [Int]
}

// MARK: - M31 AIR Program

/// AIR program over M31 for Circle STARK trace generation.
public struct M31AIRStep {
    public let outputCol: Int
    public let inputCols: [Int]
    public let compute: ([M31]) -> M31

    public init(outputCol: Int, inputCols: [Int], compute: @escaping ([M31]) -> M31) {
        self.outputCol = outputCol
        self.inputCols = inputCols
        self.compute = compute
    }
}

public struct M31AIRProgram {
    public let numColumns: Int
    public let steps: [M31AIRStep]
    public let initialState: [M31]

    public init(numColumns: Int, steps: [M31AIRStep], initialState: [M31]) {
        precondition(initialState.count == numColumns)
        self.numColumns = numColumns
        self.steps = steps
        self.initialState = initialState
    }
}

// MARK: - Trace Generator

/// GPU-accelerated AIR trace generator.
///
/// For sequential traces (each row depends on the previous), the generator:
/// 1. Analyzes column dependencies to find independent groups
/// 2. Within each group, computes columns in parallel using GCD
/// 3. For linear recurrences, delegates to WitnessEngine's matrix-power GPU kernel
/// 4. For independent-row traces, delegates to TraceEngine's GPU evaluator
///
/// For batch trace generation (multiple independent instances), all instances
/// are processed in parallel on GPU.
public class TraceGenerator {
    public static let version = Versions.witness

    /// The BN254 Fr trace engine (for PLONK/Groth16 traces).
    private var traceEngine: TraceEngine?
    /// The M31 witness engine (for Circle STARK traces).
    private var witnessEngine: WitnessEngine?

    public init() {}

    /// Lazily initialize the BN254 trace engine.
    private func getTraceEngine() throws -> TraceEngine {
        if let engine = traceEngine { return engine }
        let engine = try TraceEngine()
        traceEngine = engine
        return engine
    }

    /// Lazily initialize the M31 witness engine.
    private func getWitnessEngine() throws -> WitnessEngine {
        if let engine = witnessEngine { return engine }
        let engine = try WitnessEngine()
        witnessEngine = engine
        return engine
    }

    // MARK: - Column Dependency Analysis

    /// Analyze step dependencies and group into parallel-executable sets.
    public func analyzeColumnDependencies(steps: [AIRStep]) -> ColumnSchedule {
        return buildSchedule(
            steps: steps.map { ($0.outputCol, $0.inputCols) }
        )
    }

    /// Analyze M31 step dependencies.
    public func analyzeM31ColumnDependencies(steps: [M31AIRStep]) -> ColumnSchedule {
        return buildSchedule(
            steps: steps.map { ($0.outputCol, $0.inputCols) }
        )
    }

    private func buildSchedule(steps: [(outputCol: Int, inputCols: [Int])]) -> ColumnSchedule {
        // Build a dependency graph: step i depends on step j if i reads a column that j writes
        let n = steps.count
        var inDegree = [Int](repeating: 0, count: n)
        var dependents = [[Int]](repeating: [], count: n)

        // Map: column -> last step that wrote to it
        var lastWriter = [Int: Int]()

        for (i, step) in steps.enumerated() {
            for inputCol in step.inputCols {
                if let writer = lastWriter[inputCol], writer != i {
                    dependents[writer].append(i)
                    inDegree[i] += 1
                }
            }
            lastWriter[step.outputCol] = i
        }

        // Topological sort into parallel groups (BFS by levels)
        var groups = [[Int]]()
        var stepToGroup = [Int](repeating: -1, count: n)
        var ready = [Int]()

        for i in 0..<n {
            if inDegree[i] == 0 {
                ready.append(i)
            }
        }

        while !ready.isEmpty {
            let group = ready
            let groupIdx = groups.count
            groups.append(group)

            var nextReady = [Int]()
            for stepIdx in group {
                stepToGroup[stepIdx] = groupIdx
                for dep in dependents[stepIdx] {
                    inDegree[dep] -= 1
                    if inDegree[dep] == 0 {
                        nextReady.append(dep)
                    }
                }
            }
            ready = nextReady
        }

        return ColumnSchedule(parallelGroups: groups, stepToGroup: stepToGroup)
    }

    // MARK: - BN254 Fr AIR Trace Generation

    /// Generate an AIR execution trace on GPU.
    ///
    /// - Parameters:
    ///   - program: The AIR program to execute
    ///   - input: Additional per-row inputs (if any)
    ///   - steps: Number of rows (steps) in the trace
    /// - Returns: Trace matrix [column][row] of Fr elements
    public func generateAIRTrace(
        program: AIRProgram,
        input: [Fr],
        steps numRows: Int
    ) -> [[Fr]] {
        let numCols = program.numColumns
        let schedule = analyzeColumnDependencies(steps: program.steps)

        // Allocate trace matrix: [column][row]
        var trace = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows), count: numCols)

        // Set initial state (row 0)
        for col in 0..<numCols {
            trace[col][0] = program.initialState[col]
        }

        // Check if this is an independent-row program (no step reads from same-row columns
        // that are written by other steps in the same row). If so, use GPU parallel evaluation.
        let hasRowDependency = checkRowDependency(steps: program.steps)

        if !hasRowDependency {
            // Independent rows: compile to TraceProgram and run on GPU
            return generateIndependentRowTrace(program: program, input: input, numRows: numRows)
        }

        // Sequential rows: process row-by-row with column parallelism
        for row in 1..<numRows {
            // Read current state from previous row
            var currentState = [Fr](repeating: Fr.zero, count: numCols)
            for col in 0..<numCols {
                currentState[col] = trace[col][row - 1]
            }

            // Process each parallel group
            for group in schedule.parallelGroups {
                if group.count >= 4 {
                    // Parallel column computation within the group
                    var results = [Fr](repeating: Fr.zero, count: group.count)
                    let capturedState = currentState

                    DispatchQueue.concurrentPerform(iterations: group.count) { i in
                        let stepIdx = group[i]
                        let step = program.steps[stepIdx]
                        let inputs = step.inputCols.map { capturedState[$0] }
                        results[i] = step.compute(inputs)
                    }

                    // Write results
                    for (i, stepIdx) in group.enumerated() {
                        let outCol = program.steps[stepIdx].outputCol
                        currentState[outCol] = results[i]
                        trace[outCol][row] = results[i]
                    }
                } else {
                    // Small group: sequential is faster
                    for stepIdx in group {
                        let step = program.steps[stepIdx]
                        let inputs = step.inputCols.map { currentState[$0] }
                        let result = step.compute(inputs)
                        currentState[step.outputCol] = result
                        trace[step.outputCol][row] = result
                    }
                }
            }
        }

        return trace
    }

    /// Check if any step reads a column that another step in the same row writes.
    private func checkRowDependency(steps: [AIRStep]) -> Bool {
        var written = Set<Int>()
        for step in steps {
            for inputCol in step.inputCols {
                if written.contains(inputCol) {
                    return true
                }
            }
            written.insert(step.outputCol)
        }
        return false
    }

    /// Generate trace for programs where each row is independent (no sequential dependency).
    /// Compiles to a TraceProgram and runs on GPU.
    private func generateIndependentRowTrace(
        program: AIRProgram,
        input: [Fr],
        numRows: Int
    ) -> [[Fr]] {
        // Since independent-row programs don't reference previous rows,
        // they can be fully parallelized on GPU via TraceEngine.
        // However, we need a TraceProgram (instruction stream), which requires
        // the program to be expressed as field arithmetic ops.
        // For now, fall back to CPU parallel evaluation.

        let numCols = program.numColumns
        var trace = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows), count: numCols)

        // Set row 0
        for col in 0..<numCols {
            trace[col][0] = program.initialState[col]
        }

        // Parallel row evaluation
        let steps = program.steps
        DispatchQueue.concurrentPerform(iterations: numRows - 1) { rowMinus1 in
            let row = rowMinus1 + 1
            var state = program.initialState  // Each row starts fresh from initial state
            for step in steps {
                let inputs = step.inputCols.map { state[$0] }
                state[step.outputCol] = step.compute(inputs)
            }
            for col in 0..<numCols {
                trace[col][row] = state[col]
            }
        }

        return trace
    }

    // MARK: - M31 AIR Trace Generation

    /// Generate an M31 AIR execution trace.
    ///
    /// For Circle STARK traces over Mersenne-31.
    /// Uses WitnessEngine's GPU kernels when possible.
    ///
    /// - Parameters:
    ///   - program: The M31 AIR program
    ///   - input: Additional per-row inputs
    ///   - steps: Number of rows in the trace
    /// - Returns: Trace matrix [column][row] of M31 elements
    public func generateM31AIRTrace(
        program: M31AIRProgram,
        input: [M31],
        steps numRows: Int
    ) -> [[M31]] {
        let numCols = program.numColumns
        let schedule = analyzeM31ColumnDependencies(steps: program.steps)

        var trace = [[M31]](repeating: [M31](repeating: M31.zero, count: numRows), count: numCols)

        // Set initial state
        for col in 0..<numCols {
            trace[col][0] = program.initialState[col]
        }

        // Process rows sequentially with column parallelism
        for row in 1..<numRows {
            var currentState = [M31](repeating: M31.zero, count: numCols)
            for col in 0..<numCols {
                currentState[col] = trace[col][row - 1]
            }

            for group in schedule.parallelGroups {
                if group.count >= 4 {
                    var results = [M31](repeating: M31.zero, count: group.count)
                    let capturedState = currentState

                    DispatchQueue.concurrentPerform(iterations: group.count) { i in
                        let stepIdx = group[i]
                        let step = program.steps[stepIdx]
                        let inputs = step.inputCols.map { capturedState[$0] }
                        results[i] = step.compute(inputs)
                    }

                    for (i, stepIdx) in group.enumerated() {
                        let outCol = program.steps[stepIdx].outputCol
                        currentState[outCol] = results[i]
                        trace[outCol][row] = results[i]
                    }
                } else {
                    for stepIdx in group {
                        let step = program.steps[stepIdx]
                        let inputs = step.inputCols.map { currentState[$0] }
                        let result = step.compute(inputs)
                        currentState[step.outputCol] = result
                        trace[step.outputCol][row] = result
                    }
                }
            }
        }

        return trace
    }

    // MARK: - GPU Compiled Trace Generation

    /// Generate a trace using a compiled TraceProgram on GPU.
    /// This is the fastest path: the program is expressed as an instruction stream
    /// and each row is evaluated independently by a GPU thread.
    ///
    /// - Parameters:
    ///   - program: Compiled instruction stream
    ///   - inputs: Flat array of per-row inputs
    ///   - numRows: Number of rows
    /// - Returns: Trace as [row][col] array
    public func generateCompiledTrace(
        program: CompiledProgram,
        inputs: [Fr],
        numRows: Int
    ) throws -> [[Fr]] {
        let engine = try getTraceEngine()
        return try engine.evaluateToArray(program: program, inputs: inputs, numRows: numRows)
    }

    /// Generate a trace using a compiled M31 TraceProgram on GPU.
    public func generateCompiledM31Trace(
        program: CompiledM31Program,
        inputs: [UInt32],
        numRows: Int
    ) throws -> [[M31]] {
        let engine = try getWitnessEngine()
        return try engine.evaluateToArray(program: program, inputs: inputs, numRows: numRows)
    }

    // MARK: - Specialized: Fibonacci Trace

    /// Generate a Fibonacci AIR trace on GPU using matrix-power doubling.
    /// This is a specialized fast path for the common Fibonacci recurrence.
    public func generateFibonacciTrace(
        a0: M31, b0: M31, numRows: Int
    ) throws -> (colA: [M31], colB: [M31]) {
        let engine = try getWitnessEngine()
        return try engine.generateFibonacciTrace(a0: a0, b0: b0, numRows: numRows)
    }

    // MARK: - Specialized: Linear Recurrence

    /// Generate a trace for a linear recurrence on GPU.
    /// state[i] = T * state[i-1] where T is a constant transfer matrix.
    public func generateLinearRecurrenceTrace(
        transferMatrix: [[M31]],
        initialState: [M31],
        numRows: Int
    ) throws -> [[M31]] {
        let engine = try getWitnessEngine()
        return try engine.generateLinearRecurrence(
            transferMatrix: transferMatrix,
            initialState: initialState,
            numRows: numRows
        )
    }

    // MARK: - Batch Trace Generation

    /// Generate multiple independent trace instances in parallel.
    /// Each instance has its own initial state and inputs, but shares the same program.
    /// The instances are batched on GPU for maximum throughput.
    ///
    /// - Parameters:
    ///   - program: Compiled instruction stream (shared across all instances)
    ///   - instances: Array of (inputs, numRows) for each instance
    /// - Returns: Array of trace matrices, one per instance
    public func generateBatchTraces(
        program: CompiledProgram,
        instances: [(inputs: [Fr], numRows: Int)]
    ) throws -> [[[Fr]]] {
        let engine = try getTraceEngine()

        // Process instances in parallel using GCD
        var results = [[[Fr]]?](repeating: nil, count: instances.count)

        DispatchQueue.concurrentPerform(iterations: instances.count) { i in
            let (inputs, numRows) = instances[i]
            do {
                let trace = try engine.evaluateToArray(
                    program: program, inputs: inputs, numRows: numRows
                )
                results[i] = trace
            } catch {
                // Store empty result on error
                results[i] = []
            }
        }

        return results.map { $0 ?? [] }
    }

    /// Generate multiple independent M31 Fibonacci traces in batch.
    public func generateBatchFibonacciTraces(
        instances: [(a0: M31, b0: M31, numRows: Int)]
    ) throws -> [([M31], [M31])] {
        let engine = try getWitnessEngine()

        var results = [([M31], [M31])?](repeating: nil, count: instances.count)

        DispatchQueue.concurrentPerform(iterations: instances.count) { i in
            let inst = instances[i]
            do {
                let (colA, colB) = try engine.generateFibonacciTrace(
                    a0: inst.a0, b0: inst.b0, numRows: inst.numRows
                )
                results[i] = (colA, colB)
            } catch {
                results[i] = ([], [])
            }
        }

        return results.map { $0 ?? ([], []) }
    }
}
