// GPU Constraint Evaluation Engine — Plonk/AIR gate-aware constraint evaluation
//
// Evaluates constraint polynomials over a domain on GPU. Each point in the evaluation
// domain is computed independently (embarrassingly parallel).
//
// Supports common gate types via a generic GateDescriptor encoding:
//   - Arithmetic: qL*a + qR*b + qO*c + qM*a*b + qC = 0
//   - Multiplication: a * b - c = 0
//   - Boolean: a * (1 - a) = 0
//   - Addition: a + b - c = 0
//   - Poseidon2 full/partial rounds
//   - Range decomposition
//
// Also supports the existing ConstraintSystem/Expr IR via the ConstraintEngine path.

import Foundation
import Metal

// MARK: - Gate Type Encoding

/// Gate types matching the Metal shader opcodes
public enum ConstraintGateType: UInt32 {
    case arithmetic       = 0  // qL*a + qR*b + qO*c + qM*a*b + qC = 0
    case mul              = 1  // a * b - c = 0
    case bool             = 2  // a * (1 - a) = 0
    case poseidon2Full    = 3  // Full Poseidon2 round
    case poseidon2Partial = 4  // Partial Poseidon2 round
    case add              = 5  // a + b - c = 0
    case rangeDecomp      = 6  // sum(bit_i * 2^i) - value = 0
}

/// A gate descriptor that encodes one constraint for GPU evaluation.
/// Mirrors the Metal GateDescriptor struct layout (32 bytes).
public struct GateDescriptor {
    public var type: UInt32
    public var colA: UInt32
    public var colB: UInt32
    public var colC: UInt32
    public var aux0: UInt32
    public var aux1: UInt32
    public var aux2: UInt32
    public var selIdx: UInt32   // 0xFFFFFFFF = no selector (always active)

    public init(type: ConstraintGateType, colA: UInt32, colB: UInt32 = 0, colC: UInt32 = 0,
                aux0: UInt32 = 0, aux1: UInt32 = 0, aux2: UInt32 = 0,
                selIdx: UInt32 = 0xFFFFFFFF) {
        self.type = type.rawValue
        self.colA = colA
        self.colB = colB
        self.colC = colC
        self.aux0 = aux0
        self.aux1 = aux1
        self.aux2 = aux2
        self.selIdx = selIdx
    }

    // Convenience factories

    /// Arithmetic gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
    /// constantsBaseIdx points to 5 consecutive constants [qL, qR, qO, qM, qC] in the pool.
    public static func arithmetic(colA: UInt32, colB: UInt32, colC: UInt32,
                                  constantsBaseIdx: UInt32,
                                  selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .arithmetic, colA: colA, colB: colB, colC: colC,
                       aux0: constantsBaseIdx, selIdx: selIdx)
    }

    /// Multiplication gate: a * b - c = 0
    public static func mul(colA: UInt32, colB: UInt32, colC: UInt32,
                          selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .mul, colA: colA, colB: colB, colC: colC, selIdx: selIdx)
    }

    /// Boolean gate: a * (1 - a) = 0
    public static func bool(col: UInt32, selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .bool, colA: col, selIdx: selIdx)
    }

    /// Addition gate: a + b - c = 0
    public static func add(colA: UInt32, colB: UInt32, colC: UInt32,
                          selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .add, colA: colA, colB: colB, colC: colC, selIdx: selIdx)
    }

    /// Poseidon2 full round gate
    /// colA = first column of state, width = state width,
    /// rowOffset = row offset for state_out (typically 1),
    /// constantsBaseIdx = index into constants pool for round constants then MDS
    public static func poseidon2Full(colA: UInt32, width: UInt32, rowOffset: UInt32 = 1,
                                     constantsBaseIdx: UInt32,
                                     selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .poseidon2Full, colA: colA,
                       aux0: width, aux1: rowOffset, aux2: constantsBaseIdx,
                       selIdx: selIdx)
    }

    /// Poseidon2 partial round gate
    public static func poseidon2Partial(colA: UInt32, width: UInt32, rowOffset: UInt32 = 1,
                                        constantsBaseIdx: UInt32,
                                        selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .poseidon2Partial, colA: colA,
                       aux0: width, aux1: rowOffset, aux2: constantsBaseIdx,
                       selIdx: selIdx)
    }

    /// Range decomposition gate: sum(bit_i * 2^i) - value = 0
    /// colA = value column, colB = first bit column, aux0 = num_bits
    public static func rangeDecomp(valueCol: UInt32, firstBitCol: UInt32,
                                   numBits: UInt32,
                                   selIdx: UInt32 = 0xFFFFFFFF) -> GateDescriptor {
        GateDescriptor(type: .rangeDecomp, colA: valueCol, colB: firstBitCol,
                       aux0: numBits, selIdx: selIdx)
    }
}

// MARK: - Constraint Params (matches Metal struct)

/// Parameters struct matching the Metal ConstraintParams layout
struct ConstraintParams {
    var numCols: UInt32
    var domainSize: UInt32
    var numGates: UInt32
    var numSelectors: UInt32
    var numConstants: UInt32
}

// MARK: - GPU Constraint Eval Engine

/// GPU-accelerated constraint evaluation engine for Plonk/AIR circuits.
///
/// Provides two APIs:
/// 1. Gate-descriptor based: Define gates as GateDescriptor structs with typed opcodes
/// 2. ConstraintSystem based: Use the existing Expr-based IR (delegates to ConstraintEngine)
///
/// The gate-descriptor API is lower-level but supports all common gate types natively
/// without runtime Metal compilation. The constraint IR API is more flexible for
/// arbitrary polynomial expressions.
public class GPUConstraintEvalEngine {
    public static let version = Versions.gpuConstraintEval

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let tuning: TuningConfig

    // Precompiled pipelines from constraint_eval.metal
    private let evalPipeline: MTLComputePipelineState
    private let quotientPipeline: MTLComputePipelineState
    private let fusedQuotientPipeline: MTLComputePipelineState

    // For ConstraintSystem/Expr IR path
    private let constraintEngine: ConstraintEngine

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)

        // Compile the constraint_eval.metal shader
        let library = try GPUConstraintEvalEngine.compileShaders(device: device)

        guard let evalFn = library.makeFunction(name: "constraint_eval_kernel") else {
            throw MSMError.missingKernel
        }
        self.evalPipeline = try device.makeComputePipelineState(function: evalFn)

        guard let quotientFn = library.makeFunction(name: "compute_quotient_kernel") else {
            throw MSMError.missingKernel
        }
        self.quotientPipeline = try device.makeComputePipelineState(function: quotientFn)

        guard let fusedFn = library.makeFunction(name: "fused_constraint_quotient_kernel") else {
            throw MSMError.missingKernel
        }
        self.fusedQuotientPipeline = try device.makeComputePipelineState(function: fusedFn)

        self.constraintEngine = try ConstraintEngine()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let constraintSource = try String(contentsOfFile: shaderDir + "/constraint/constraint_eval.metal", encoding: .utf8)

        // Strip #include from constraint source (already have fr source inlined)
        let cleanConstraint = constraintSource.split(separator: "\n")
            .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
            .joined(separator: "\n")

        let fullSource = frSource + "\n" + cleanConstraint

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        do {
            return try device.makeLibrary(source: fullSource, options: options)
        } catch {
            fputs("GPUConstraintEvalEngine: Metal compilation failed\n", stderr)
            fputs("Error: \(error.localizedDescription)\n", stderr)
            throw MSMError.gpuError("Metal compile error: \(error.localizedDescription)")
        }
    }

    // MARK: - Evaluate Constraints (Gate Descriptor API)

    /// Evaluate constraints defined by gate descriptors over a trace.
    ///
    /// - Parameters:
    ///   - trace: Array of MTLBuffers, one per column (each has domainSize Fr values)
    ///   - gates: Array of gate descriptors defining the constraints
    ///   - constants: Constants pool (Fr values referenced by gate descriptors)
    ///   - selectors: Optional selector polynomial buffers (one per selector column)
    ///   - domainSize: Number of domain points
    /// - Returns: MTLBuffer with domainSize * numGates Fr values
    public func evaluateConstraints(
        trace: [MTLBuffer],
        gates: [GateDescriptor],
        constants: [Fr],
        selectors: [MTLBuffer] = [],
        domainSize: Int
    ) throws -> MTLBuffer {
        let numCols = trace.count
        let numGates = gates.count

        // Pack trace columns into a single column-major buffer
        let traceBuf = try packTraceColumns(trace, domainSize: domainSize)

        // Pack gate descriptors
        let gatesBuf = try packGateDescriptors(gates)

        // Pack selectors into column-major buffer
        let selectorsBuf = try packSelectorColumns(selectors, domainSize: domainSize)

        // Pack constants
        let constantsBuf = try packConstants(constants)

        // Params
        var params = ConstraintParams(
            numCols: UInt32(numCols),
            domainSize: UInt32(domainSize),
            numGates: UInt32(numGates),
            numSelectors: UInt32(selectors.count),
            numConstants: UInt32(constants.count)
        )

        // Output buffer
        let outputSize = domainSize * numGates * MemoryLayout<Fr>.stride
        guard let outputBuf = device.makeBuffer(length: max(outputSize, 32),
                                                 options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate constraint output buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(evalPipeline)
        enc.setBuffer(traceBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(gatesBuf, offset: 0, index: 2)
        enc.setBuffer(selectorsBuf, offset: 0, index: 3)
        enc.setBuffer(constantsBuf, offset: 0, index: 4)
        enc.setBytes(&params, length: MemoryLayout<ConstraintParams>.stride, index: 5)

        let tg = min(256, Int(evalPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Constraint eval GPU error: \(error.localizedDescription)")
        }

        return outputBuf
    }

    // MARK: - Compute Quotient Polynomial

    /// Compute the quotient polynomial from constraint evaluations.
    /// quotient[row] = sum_i (alpha^i * constraintEvals[row * numGates + i]) / Z_H(row)
    ///
    /// - Parameters:
    ///   - constraintEvals: Buffer from evaluateConstraints (domainSize * numGates Fr values)
    ///   - vanishingPoly: Precomputed 1/Z_H(x) for each domain point (domainSize Fr values)
    ///   - alphaPowers: Alpha challenge powers [alpha^0, alpha^1, ..., alpha^(numGates-1)]
    ///   - domainSize: Number of domain points
    ///   - numGates: Number of constraints/gates
    /// - Returns: MTLBuffer with domainSize Fr values (the quotient polynomial evaluations)
    public func computeQuotient(
        constraintEvals: MTLBuffer,
        vanishingPoly: MTLBuffer,
        alphaPowers: MTLBuffer,
        domainSize: Int,
        numGates: Int
    ) throws -> MTLBuffer {
        let outputSize = domainSize * MemoryLayout<Fr>.stride
        guard let quotientBuf = device.makeBuffer(length: max(outputSize, 32),
                                                    options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate quotient buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var ds = UInt32(domainSize)
        var ng = UInt32(numGates)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(quotientPipeline)
        enc.setBuffer(constraintEvals, offset: 0, index: 0)
        enc.setBuffer(quotientBuf, offset: 0, index: 1)
        enc.setBuffer(alphaPowers, offset: 0, index: 2)
        enc.setBuffer(vanishingPoly, offset: 0, index: 3)
        enc.setBytes(&ds, length: 4, index: 4)
        enc.setBytes(&ng, length: 4, index: 5)

        let tg = min(256, Int(quotientPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Quotient eval GPU error: \(error.localizedDescription)")
        }

        return quotientBuf
    }

    // MARK: - Fused Constraint + Quotient

    /// Fused constraint evaluation + quotient computation in one GPU pass.
    /// Avoids writing the intermediate constraint evaluations to global memory.
    public func evaluateAndComputeQuotient(
        trace: [MTLBuffer],
        gates: [GateDescriptor],
        constants: [Fr],
        selectors: [MTLBuffer] = [],
        vanishingPoly: MTLBuffer,
        alphaPowers: MTLBuffer,
        domainSize: Int
    ) throws -> MTLBuffer {
        let numCols = trace.count
        let numGates = gates.count

        let traceBuf = try packTraceColumns(trace, domainSize: domainSize)
        let gatesBuf = try packGateDescriptors(gates)
        let selectorsBuf = try packSelectorColumns(selectors, domainSize: domainSize)
        let constantsBuf = try packConstants(constants)

        var params = ConstraintParams(
            numCols: UInt32(numCols),
            domainSize: UInt32(domainSize),
            numGates: UInt32(numGates),
            numSelectors: UInt32(selectors.count),
            numConstants: UInt32(constants.count)
        )

        let outputSize = domainSize * MemoryLayout<Fr>.stride
        guard let quotientBuf = device.makeBuffer(length: max(outputSize, 32),
                                                    options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate quotient buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedQuotientPipeline)
        enc.setBuffer(traceBuf, offset: 0, index: 0)
        enc.setBuffer(quotientBuf, offset: 0, index: 1)
        enc.setBuffer(gatesBuf, offset: 0, index: 2)
        enc.setBuffer(selectorsBuf, offset: 0, index: 3)
        enc.setBuffer(constantsBuf, offset: 0, index: 4)
        enc.setBuffer(alphaPowers, offset: 0, index: 5)
        enc.setBuffer(vanishingPoly, offset: 0, index: 6)
        enc.setBytes(&params, length: MemoryLayout<ConstraintParams>.stride, index: 7)

        let tg = min(256, Int(fusedQuotientPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Fused quotient GPU error: \(error.localizedDescription)")
        }

        return quotientBuf
    }

    // MARK: - ConstraintSystem IR Path

    /// Evaluate constraints defined by the ConstraintSystem IR.
    /// Delegates to the existing ConstraintEngine for Expr-based constraint systems.
    public func evaluateConstraintSystem(
        system: ConstraintSystem,
        trace: MTLBuffer,
        numRows: Int,
        includeQuotient: Bool = false
    ) throws -> MTLBuffer {
        let compiled = try constraintEngine.compile(system: system, includeQuotient: includeQuotient)
        return try constraintEngine.evaluate(compiled: compiled, trace: trace, numRows: numRows)
    }

    // MARK: - CPU Reference Evaluation

    /// CPU reference evaluation for correctness comparison.
    /// Uses the gate descriptor format matching the GPU kernel.
    public func evaluateCPU(
        trace: [[Fr]],   // trace[col][row]
        gates: [GateDescriptor],
        constants: [Fr],
        selectors: [[Fr]] = [],
        domainSize: Int
    ) -> [Fr] {
        let numGates = gates.count
        var output = [Fr](repeating: Fr.zero, count: domainSize * numGates)

        for row in 0..<domainSize {
            for (gi, gate) in gates.enumerated() {
                // Check selector
                var selVal = Fr.one
                if gate.selIdx != 0xFFFFFFFF {
                    let si = Int(gate.selIdx)
                    if si < selectors.count && row < selectors[si].count {
                        selVal = selectors[si][row]
                    }
                }

                var eval = Fr.zero

                switch ConstraintGateType(rawValue: gate.type) {
                case .arithmetic:
                    let a = traceVal(trace, col: Int(gate.colA), row: row, domainSize: domainSize)
                    let b = traceVal(trace, col: Int(gate.colB), row: row, domainSize: domainSize)
                    let c = traceVal(trace, col: Int(gate.colC), row: row, domainSize: domainSize)
                    let qL = constants[Int(gate.aux0)]
                    let qR = constants[Int(gate.aux0) + 1]
                    let qO = constants[Int(gate.aux0) + 2]
                    let qM = constants[Int(gate.aux0) + 3]
                    let qC = constants[Int(gate.aux0) + 4]
                    // qL*a + qR*b + qO*c + qM*a*b + qC
                    var r = frMul(qL, a)
                    r = frAdd(r, frMul(qR, b))
                    r = frAdd(r, frMul(qO, c))
                    r = frAdd(r, frMul(qM, frMul(a, b)))
                    r = frAdd(r, qC)
                    eval = r

                case .mul:
                    let a = traceVal(trace, col: Int(gate.colA), row: row, domainSize: domainSize)
                    let b = traceVal(trace, col: Int(gate.colB), row: row, domainSize: domainSize)
                    let c = traceVal(trace, col: Int(gate.colC), row: row, domainSize: domainSize)
                    eval = frSub(frMul(a, b), c)

                case .bool:
                    let a = traceVal(trace, col: Int(gate.colA), row: row, domainSize: domainSize)
                    eval = frMul(a, frSub(Fr.one, a))

                case .add:
                    let a = traceVal(trace, col: Int(gate.colA), row: row, domainSize: domainSize)
                    let b = traceVal(trace, col: Int(gate.colB), row: row, domainSize: domainSize)
                    let c = traceVal(trace, col: Int(gate.colC), row: row, domainSize: domainSize)
                    eval = frSub(frAdd(a, b), c)

                case .rangeDecomp:
                    let numBits = Int(gate.aux0)
                    let value = traceVal(trace, col: Int(gate.colA), row: row, domainSize: domainSize)
                    var sum = Fr.zero
                    var pow2 = Fr.one
                    let two = frAdd(Fr.one, Fr.one)
                    for i in 0..<numBits {
                        let bit = traceVal(trace, col: Int(gate.colB) + i, row: row, domainSize: domainSize)
                        sum = frAdd(sum, frMul(pow2, bit))
                        pow2 = frMul(pow2, two)
                    }
                    eval = frSub(sum, value)

                default:
                    eval = Fr.zero
                }

                // Apply selector
                if gate.selIdx != 0xFFFFFFFF {
                    eval = frMul(eval, selVal)
                }

                output[row * numGates + gi] = eval
            }
        }

        return output
    }

    // MARK: - Buffer Helpers

    /// Create a trace column buffer from a Swift array of Fr values
    public func createColumnBuffer(_ data: [Fr]) throws -> MTLBuffer {
        let size = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: max(size, 32), options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate column buffer")
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, size)
        }
        return buf
    }

    /// Create an alpha powers buffer: [alpha^0, alpha^1, ..., alpha^(n-1)]
    public func createAlphaPowers(alpha: Fr, count: Int) throws -> MTLBuffer {
        var powers = [Fr](repeating: Fr.zero, count: count)
        powers[0] = Fr.one
        for i in 1..<count {
            powers[i] = frMul(powers[i - 1], alpha)
        }
        return try createColumnBuffer(powers)
    }

    /// Read GPU buffer contents back as an array of Fr
    public func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Fr] {
        let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - Private Helpers

    private func traceVal(_ trace: [[Fr]], col: Int, row: Int, domainSize: Int) -> Fr {
        if col < trace.count && row < trace[col].count {
            return trace[col][row]
        }
        return Fr.zero
    }

    /// Pack separate column buffers into a single column-major buffer
    private func packTraceColumns(_ columns: [MTLBuffer], domainSize: Int) throws -> MTLBuffer {
        let numCols = columns.count
        let stride = MemoryLayout<Fr>.stride
        let totalSize = numCols * domainSize * stride

        guard let buf = device.makeBuffer(length: max(totalSize, 32), options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate packed trace buffer")
        }

        let dst = buf.contents()
        for (i, col) in columns.enumerated() {
            let colSize = domainSize * stride
            memcpy(dst + i * colSize, col.contents(), min(colSize, col.length))
        }

        return buf
    }

    /// Pack selector columns into a single column-major buffer
    private func packSelectorColumns(_ selectors: [MTLBuffer], domainSize: Int) throws -> MTLBuffer {
        if selectors.isEmpty {
            // Return a tiny dummy buffer
            guard let buf = device.makeBuffer(length: 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate dummy selector buffer")
            }
            return buf
        }
        return try packTraceColumns(selectors, domainSize: domainSize)
    }

    /// Pack gate descriptors into a GPU buffer
    private func packGateDescriptors(_ gates: [GateDescriptor]) throws -> MTLBuffer {
        let size = gates.count * MemoryLayout<GateDescriptor>.stride
        guard let buf = device.makeBuffer(length: max(size, 32), options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate gate descriptor buffer")
        }
        gates.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, size)
        }
        return buf
    }

    /// Pack constants pool into a GPU buffer
    private func packConstants(_ constants: [Fr]) throws -> MTLBuffer {
        if constants.isEmpty {
            guard let buf = device.makeBuffer(length: 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate dummy constants buffer")
            }
            return buf
        }
        return try createColumnBuffer(constants)
    }
}
