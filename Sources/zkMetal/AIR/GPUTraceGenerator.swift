// GPUTraceGenerator — GPU-accelerated execution trace generator for STARK provers
//
// An execution trace is a matrix where each row is a state and each column is a
// register. Generating the trace involves repeated computation of the transition
// function, followed by low-degree extension (LDE) for STARK soundness, and
// column-wise polynomial commitment via MSM.
//
// Pipeline:
//   1. generateTrace: run transition function to fill trace matrix (GPU or CPU)
//   2. extendTrace: LDE via coset NTT (each column extended independently on GPU)
//   3. commitTrace: commit each column via GPU MSM
//
// Supported AIR patterns:
//   - Hash chains (Poseidon2 state columns across rounds)
//   - Fibonacci sequences (linear recurrence)
//   - State machines (arbitrary transition functions)
//   - Range check decompositions
//
// Field support: BN254 Fr and BabyBear (Bb).

import Foundation
import Metal

// MARK: - Trace Matrix

/// Column-major trace matrix over BN254 Fr.
/// `columns[col][row]` gives the field element at a specific column and row.
/// All columns have the same number of rows (padded to power of 2).
public struct TraceMatrix {
    /// Column-major storage: columns[i] is the i-th register's values across all rows.
    public var columns: [[Fr]]

    /// Number of columns (registers).
    public var numColumns: Int { columns.count }

    /// Number of rows (steps). All columns have the same length.
    public var numRows: Int { columns.first?.count ?? 0 }

    public init(columns: [[Fr]]) {
        precondition(!columns.isEmpty, "Trace must have at least one column")
        let n = columns[0].count
        for col in columns {
            precondition(col.count == n, "All columns must have the same length")
        }
        self.columns = columns
    }

    /// Access element at (row, col).
    public subscript(row: Int, col: Int) -> Fr {
        get { columns[col][row] }
        set { columns[col][row] = newValue }
    }
}

/// Column-major trace matrix over BabyBear.
public struct BbTraceMatrix {
    public var columns: [[Bb]]

    public var numColumns: Int { columns.count }
    public var numRows: Int { columns.first?.count ?? 0 }

    public init(columns: [[Bb]]) {
        precondition(!columns.isEmpty, "Trace must have at least one column")
        let n = columns[0].count
        for col in columns {
            precondition(col.count == n, "All columns must have the same length")
        }
        self.columns = columns
    }

    public subscript(row: Int, col: Int) -> Bb {
        get { columns[col][row] }
        set { columns[col][row] = newValue }
    }
}

// MARK: - AIR Transition

/// Describes how one state transitions to the next in an AIR program.
/// The transition function receives the current row state and returns the next row state.
public struct AIRTransition {
    /// Number of columns (registers) in the state.
    public let numColumns: Int

    /// Transition function: given current state, produce next state.
    /// Both input and output are arrays of `numColumns` field elements.
    public let apply: ([Fr]) -> [Fr]

    public init(numColumns: Int, apply: @escaping ([Fr]) -> [Fr]) {
        self.numColumns = numColumns
        self.apply = apply
    }
}

/// BabyBear version of AIR transition.
public struct BbAIRTransition {
    public let numColumns: Int
    public let apply: ([Bb]) -> [Bb]

    public init(numColumns: Int, apply: @escaping ([Bb]) -> [Bb]) {
        self.numColumns = numColumns
        self.apply = apply
    }
}

// MARK: - Common AIR Patterns

/// Factory for common AIR transition functions.
public enum AIRPatterns {

    // MARK: Fibonacci

    /// Fibonacci AIR: 2 columns (a, b), transition a' = b, b' = a + b.
    public static func fibonacci() -> AIRTransition {
        AIRTransition(numColumns: 2) { state in
            [state[1], frAdd(state[0], state[1])]
        }
    }

    /// BabyBear Fibonacci AIR.
    public static func fibonacciBb() -> BbAIRTransition {
        BbAIRTransition(numColumns: 2) { state in
            [state[1], bbAdd(state[0], state[1])]
        }
    }

    // MARK: Hash Chain

    /// Hash chain AIR: stateWidth columns, transition applies hashFn to current state.
    /// The hash function should take and return `stateWidth` elements.
    public static func hashChain(stateWidth: Int, hashFn: @escaping ([Fr]) -> [Fr]) -> AIRTransition {
        AIRTransition(numColumns: stateWidth) { state in
            hashFn(state)
        }
    }

    /// BabyBear hash chain.
    public static func hashChainBb(stateWidth: Int, hashFn: @escaping ([Bb]) -> [Bb]) -> BbAIRTransition {
        BbAIRTransition(numColumns: stateWidth) { state in
            hashFn(state)
        }
    }

    // MARK: State Machine

    /// Generic state machine: arbitrary transition function with named registers.
    /// The closure receives the full state vector and returns the next state.
    public static func stateMachine(numRegisters: Int, transition: @escaping ([Fr]) -> [Fr]) -> AIRTransition {
        AIRTransition(numColumns: numRegisters, apply: transition)
    }

    /// BabyBear state machine.
    public static func stateMachineBb(numRegisters: Int, transition: @escaping ([Bb]) -> [Bb]) -> BbAIRTransition {
        BbAIRTransition(numColumns: numRegisters, apply: transition)
    }
}

// MARK: - GPU Trace Generator

/// GPU-accelerated execution trace generator for STARK provers.
///
/// Generates, extends, and commits execution traces using Metal GPU acceleration
/// where profitable. Falls back to CPU for small traces or when GPU is unavailable.
///
/// Usage:
/// ```
/// let gen = try GPUTraceGenerator()
/// let trace = gen.generateTrace(
///     initialState: [Fr.one, Fr.one],
///     transitionFn: AIRPatterns.fibonacci(),
///     numSteps: 1024
/// )
/// let extended = try gen.extendTrace(trace: trace, blowupFactor: 4)
/// let commitments = try gen.commitTrace(trace: extended)
/// ```
public class GPUTraceGenerator {
    public static let version = Versions.gpuTraceGen

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Lazy-initialized sub-engines
    private var traceEngine: TraceEngine?
    private var ldeEngine: GPUCosetLDEEngine?
    private var msmEngine: MetalMSM?
    private var nttEngine: NTTEngine?
    private var bbNttEngine: BabyBearNTTEngine?

    // GPU pipeline states for trace extension
    private var extendBatchFr: MTLComputePipelineState?
    private var extendBatchBb: MTLComputePipelineState?
    private var transposeFr: MTLComputePipelineState?
    private var transposeBb: MTLComputePipelineState?

    // Coset power cache
    private var frCosetPowersCache: [String: MTLBuffer] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        try compileShaders()
    }

    // MARK: - Shader Compilation

    private func compileShaders() throws {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let extendSource = try String(contentsOfFile: shaderDir + "/witness/trace_extend.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { line in
                    if line.contains("#include") || line.contains("#ifndef") || line.contains("#endif") { return false }
                    if line.contains("#define") {
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        let parts = trimmed.split(separator: " ", maxSplits: 3)
                        return parts.count >= 3
                    }
                    return true
                }
                .joined(separator: "\n")
        }

        let combined = clean(frSource) + "\n" + clean(bbSource) + "\n" + clean(extendSource)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        if let fn = library.makeFunction(name: "trace_extend_batch_fr") {
            extendBatchFr = try device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "trace_extend_batch_bb") {
            extendBatchBb = try device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "trace_transpose_fr") {
            transposeFr = try device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "trace_transpose_bb") {
            transposeBb = try device.makeComputePipelineState(function: fn)
        }
    }

    // MARK: - Lazy Sub-Engine Access

    private func getTraceEngine() throws -> TraceEngine {
        if let e = traceEngine { return e }
        let e = try TraceEngine()
        traceEngine = e
        return e
    }

    private func getLDEEngine() throws -> GPUCosetLDEEngine {
        if let e = ldeEngine { return e }
        let e = try GPUCosetLDEEngine()
        ldeEngine = e
        return e
    }

    private func getMSMEngine() throws -> MetalMSM {
        if let e = msmEngine { return e }
        let e = try MetalMSM()
        msmEngine = e
        return e
    }

    private func getNTTEngine() throws -> NTTEngine {
        if let e = nttEngine { return e }
        let e = try NTTEngine()
        nttEngine = e
        return e
    }

    private func getBbNTTEngine() throws -> BabyBearNTTEngine {
        if let e = bbNttEngine { return e }
        let e = try BabyBearNTTEngine()
        bbNttEngine = e
        return e
    }

    // MARK: - Trace Generation (BN254 Fr)

    /// Generate an execution trace by repeatedly applying the transition function.
    ///
    /// Starting from `initialState`, applies `transitionFn` for `numSteps` iterations,
    /// producing a trace matrix with `numSteps` rows (padded to next power of 2).
    ///
    /// For sequential traces (row i depends on row i-1), computation is done on CPU
    /// since each row depends on the previous. For large traces with independent columns,
    /// column-level parallelism is exploited via GCD.
    ///
    /// - Parameters:
    ///   - initialState: Starting values for each register (one per column)
    ///   - transitionFn: How the state evolves each step
    ///   - numSteps: Number of steps (rows before padding)
    /// - Returns: Column-major trace matrix, padded to power-of-2 rows
    public func generateTrace(
        initialState: [Fr],
        transitionFn: AIRTransition,
        numSteps: Int
    ) -> TraceMatrix {
        precondition(initialState.count == transitionFn.numColumns,
                     "Initial state size must match transition function column count")
        precondition(numSteps >= 1, "Need at least 1 step")

        let numCols = transitionFn.numColumns
        let paddedRows = nextPowerOfTwo(numSteps)

        // Allocate column-major storage
        var columns = (0..<numCols).map { col in
            var column = [Fr](repeating: Fr.zero, count: paddedRows)
            column[0] = initialState[col]
            return column
        }

        // Execute transition function row by row
        var currentState = initialState
        for row in 1..<numSteps {
            let nextState = transitionFn.apply(currentState)
            for col in 0..<numCols {
                columns[col][row] = nextState[col]
            }
            currentState = nextState
        }

        // Pad remaining rows with last valid state
        if paddedRows > numSteps {
            for col in 0..<numCols {
                let lastVal = columns[col][numSteps - 1]
                for row in numSteps..<paddedRows {
                    columns[col][row] = lastVal
                }
            }
        }

        return TraceMatrix(columns: columns)
    }

    /// Generate trace using a compiled TraceProgram on GPU.
    /// For programs that can be expressed as an instruction stream, this is faster
    /// for large traces since all rows are evaluated in parallel.
    ///
    /// - Parameters:
    ///   - program: Compiled instruction stream
    ///   - inputs: Per-row input values (flattened row-major)
    ///   - numRows: Number of rows
    /// - Returns: Column-major trace matrix
    public func generateTraceGPU(
        program: CompiledProgram,
        inputs: [Fr],
        numRows: Int
    ) throws -> TraceMatrix {
        let engine = try getTraceEngine()
        let rows = try engine.evaluateToArray(program: program, inputs: inputs, numRows: numRows)

        // Convert from row-major to column-major
        let numCols = program.numCols
        var columns = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: numRows), count: numCols)
        for row in 0..<numRows {
            for col in 0..<numCols {
                columns[col][row] = rows[row][col]
            }
        }

        return TraceMatrix(columns: columns)
    }

    // MARK: - Trace Extension (LDE)

    /// Extend a trace via coset NTT (Low-Degree Extension).
    ///
    /// Each column is independently extended from domain H (size N) to coset domain
    /// g*H' (size blowupFactor * N). This is the standard STARK trace extension step
    /// that ensures soundness of the polynomial IOP.
    ///
    /// Algorithm per column:
    ///   1. INTT to get polynomial coefficients
    ///   2. Zero-pad coefficients to extended size
    ///   3. Multiply by coset powers (g^i)
    ///   4. Forward NTT over extended domain
    ///
    /// Steps 2-3 are fused into a single GPU kernel for all columns simultaneously.
    ///
    /// - Parameters:
    ///   - trace: Original execution trace
    ///   - blowupFactor: Extension factor (must be power of 2, typically 2, 4, or 8)
    /// - Returns: Extended trace matrix with blowupFactor * numRows rows
    public func extendTrace(trace: TraceMatrix, blowupFactor: Int) throws -> TraceMatrix {
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2")

        let n = trace.numRows
        precondition(n > 0 && (n & (n - 1)) == 0, "Trace rows must be a power of 2")

        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        precondition(logM <= Fr.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        let m = 1 << logM
        let numCols = trace.numColumns
        let cosetShift = frFromInt(Fr.GENERATOR)

        // For small traces, use CPU path
        if n <= 64 || numCols == 0 {
            return try cpuExtendTrace(trace: trace, blowupFactor: blowupFactor, cosetShift: cosetShift)
        }

        // GPU batch extension path
        let engine = try getNTTEngine()

        // Step 1: INTT each column to get coefficients
        var allCoeffs = [[Fr]]()
        allCoeffs.reserveCapacity(numCols)
        for col in trace.columns {
            allCoeffs.append(try engine.intt(col))
        }

        // Step 2+3: Fused zero-pad + coset shift on GPU (all columns in single dispatch)
        let cosetPowers = getCosetPowers(logM: logM, cosetShift: cosetShift)

        // Pack all coefficient columns contiguously
        var packed = [Fr]()
        packed.reserveCapacity(n * numCols)
        for col in allCoeffs {
            packed.append(contentsOf: col)
        }

        guard let inputBuf = device.makeBuffer(
            bytes: packed, length: n * numCols * MemoryLayout<Fr>.stride,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate trace extend input buffer")
        }

        guard let outputBuf = device.makeBuffer(
            length: m * numCols * MemoryLayout<Fr>.stride,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate trace extend output buffer")
        }

        guard let pipeline = extendBatchFr else {
            throw MSMError.missingKernel
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        var numColsVal = UInt32(numCols)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        enc.setBytes(&numColsVal, length: 4, index: 5)

        let totalThreads = m * numCols
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: totalThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError("Trace extend GPU error: \(error.localizedDescription)")
        }

        // Step 4: Forward NTT each column
        var extendedColumns = [[Fr]]()
        extendedColumns.reserveCapacity(numCols)
        let outPtr = outputBuf.contents().bindMemory(to: Fr.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            extendedColumns.append(try engine.ntt(colSlice))
        }

        return TraceMatrix(columns: extendedColumns)
    }

    /// CPU fallback for trace extension.
    private func cpuExtendTrace(trace: TraceMatrix, blowupFactor: Int, cosetShift: Fr) throws -> TraceMatrix {
        let n = trace.numRows
        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        var extendedColumns = [[Fr]]()
        extendedColumns.reserveCapacity(trace.numColumns)

        for col in trace.columns {
            // INTT -> coefficients
            let coeffs = NTTEngine.cpuINTT(col, logN: logN)

            // Zero-pad + coset shift
            var padded = [Fr](repeating: Fr.zero, count: m)
            var gPow = Fr.one
            for i in 0..<m {
                let c = (i < n) ? coeffs[i] : Fr.zero
                padded[i] = frMul(c, gPow)
                gPow = frMul(gPow, cosetShift)
            }

            // Forward NTT
            extendedColumns.append(NTTEngine.cpuNTT(padded, logN: logM))
        }

        return TraceMatrix(columns: extendedColumns)
    }

    // MARK: - Trace Commitment

    /// Commit each column of the trace via GPU MSM.
    ///
    /// Each column is treated as a vector of scalars, and the commitment is
    /// computed as MSM(SRS, column) where SRS is the structured reference string.
    /// This produces one G1 point per column.
    ///
    /// - Parameters:
    ///   - trace: The (possibly extended) trace matrix
    ///   - srs: Structured reference string (G1 points); must have at least numRows elements
    /// - Returns: One commitment point per column
    public func commitTrace(
        trace: TraceMatrix,
        srs: [PointAffine]
    ) throws -> [PointProjective] {
        let n = trace.numRows
        precondition(srs.count >= n, "SRS must have at least as many points as trace rows")
        let srsSlice = Array(srs.prefix(n))

        let engine = try getMSMEngine()
        var commitments = [PointProjective]()
        commitments.reserveCapacity(trace.numColumns)

        for col in trace.columns {
            // Convert Fr elements to scalar limbs for MSM
            let scalars = col.map { fr -> [UInt32] in
                [fr.v.0, fr.v.1, fr.v.2, fr.v.3,
                 fr.v.4, fr.v.5, fr.v.6, fr.v.7]
            }
            let commitment = try engine.msm(points: srsSlice, scalars: scalars)
            commitments.append(commitment)
        }

        return commitments
    }

    // MARK: - BabyBear Trace Generation

    /// Generate a BabyBear execution trace.
    public func generateTraceBb(
        initialState: [Bb],
        transitionFn: BbAIRTransition,
        numSteps: Int
    ) -> BbTraceMatrix {
        precondition(initialState.count == transitionFn.numColumns)
        precondition(numSteps >= 1)

        let numCols = transitionFn.numColumns
        let paddedRows = nextPowerOfTwo(numSteps)

        var columns = (0..<numCols).map { col in
            var column = [Bb](repeating: Bb.zero, count: paddedRows)
            column[0] = initialState[col]
            return column
        }

        var currentState = initialState
        for row in 1..<numSteps {
            let nextState = transitionFn.apply(currentState)
            for col in 0..<numCols {
                columns[col][row] = nextState[col]
            }
            currentState = nextState
        }

        if paddedRows > numSteps {
            for col in 0..<numCols {
                let lastVal = columns[col][numSteps - 1]
                for row in numSteps..<paddedRows {
                    columns[col][row] = lastVal
                }
            }
        }

        return BbTraceMatrix(columns: columns)
    }

    /// Extend a BabyBear trace via coset NTT.
    public func extendTraceBb(trace: BbTraceMatrix, blowupFactor: Int) throws -> BbTraceMatrix {
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0)

        let n = trace.numRows
        precondition(n > 0 && (n & (n - 1)) == 0)

        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        precondition(logM <= Bb.TWO_ADICITY)

        let cosetShift = Bb(v: Bb.GENERATOR)

        // Use GPUCosetLDEEngine's batch extend for BabyBear
        let lde = try getLDEEngine()
        let extended = try lde.batchExtend(
            columns: trace.columns,
            logN: logN,
            blowupFactor: blowupFactor,
            cosetShift: cosetShift
        )

        return BbTraceMatrix(columns: extended)
    }

    // MARK: - Coset Power Cache

    private func getCosetPowers(logM: Int, cosetShift: Fr) -> MTLBuffer {
        let key = "\(logM)_\(cosetShift.v.0)_\(cosetShift.v.1)_\(cosetShift.v.2)_\(cosetShift.v.3)"
        if let cached = frCosetPowersCache[key] { return cached }

        let m = 1 << logM
        var powers = [Fr](repeating: Fr.one, count: m)
        for i in 1..<m {
            powers[i] = frMul(powers[i - 1], cosetShift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        frCosetPowersCache[key] = buf
        return buf
    }

    // MARK: - Utilities

    private func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v |= v >> 32
        return v + 1
    }
}
