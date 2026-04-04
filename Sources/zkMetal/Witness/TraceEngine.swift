// TraceEngine — GPU witness generation for execution traces
//
// An execution trace is a matrix of field elements (BN254 Fr) where:
//   - Each row is one step of the computation
//   - Each column is one register/wire value
//   - All rows execute the same instruction program (perfect GPU fit)
//
// The instruction stream approach means one compiled kernel handles any
// program — no Metal recompilation needed per circuit.

import Foundation
import Metal

// MARK: - TraceProgram (instruction builder)

/// Compiled instruction stream ready for GPU execution.
public struct CompiledProgram {
    public let instructions: [UInt32]   // Packed: [op, dst, src1, src2] x N
    public let constants: [Fr]          // Constant pool
    public let numInstructions: Int
    public let numCols: Int             // Required number of columns
    public let inputWidth: Int          // Number of input columns per row
}

/// Builder for trace evaluation programs.
/// Each operation specifies destination and source columns.
/// Column indices are 0-based. Input loading uses a separate input index space.
public class TraceProgram {
    public enum Op {
        case add(dst: Int, src1: Int, src2: Int)
        case mul(dst: Int, src1: Int, src2: Int)
        case sub(dst: Int, src1: Int, src2: Int)
        case copy(dst: Int, src: Int)
        case loadInput(dst: Int, inputIdx: Int)
        case addConst(dst: Int, src: Int, constant: Fr)
        case mulConst(dst: Int, src: Int, constant: Fr)
        case subConst(dst: Int, src: Int, constant: Fr)
        case sqr(dst: Int, src: Int)
        case double_(dst: Int, src: Int)
        case neg(dst: Int, src: Int)
        case select(dst: Int, trueCol: Int, falseCol: Int, selectorCol: Int)
    }

    private var ops: [Op] = []
    private var maxCol: Int = 0
    private var maxInput: Int = 0

    public init() {}

    private func trackCol(_ col: Int) {
        if col > maxCol { maxCol = col }
    }

    @discardableResult
    public func add(_ dst: Int, _ src1: Int, _ src2: Int) -> TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.add(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func mul(_ dst: Int, _ src1: Int, _ src2: Int) -> TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.mul(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func sub(_ dst: Int, _ src1: Int, _ src2: Int) -> TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.sub(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func copy(_ dst: Int, _ src: Int) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.copy(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func loadInput(_ dst: Int, _ inputIdx: Int) -> TraceProgram {
        trackCol(dst)
        if inputIdx > maxInput { maxInput = inputIdx }
        ops.append(.loadInput(dst: dst, inputIdx: inputIdx))
        return self
    }

    @discardableResult
    public func addConst(_ dst: Int, _ src: Int, _ constant: Fr) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.addConst(dst: dst, src: src, constant: constant))
        return self
    }

    @discardableResult
    public func mulConst(_ dst: Int, _ src: Int, _ constant: Fr) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.mulConst(dst: dst, src: src, constant: constant))
        return self
    }

    @discardableResult
    public func subConst(_ dst: Int, _ src: Int, _ constant: Fr) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.subConst(dst: dst, src: src, constant: constant))
        return self
    }

    @discardableResult
    public func sqr(_ dst: Int, _ src: Int) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.sqr(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func double_(_ dst: Int, _ src: Int) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.double_(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func neg(_ dst: Int, _ src: Int) -> TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.neg(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func select(_ dst: Int, trueCol: Int, falseCol: Int, selectorCol: Int) -> TraceProgram {
        trackCol(dst); trackCol(trueCol); trackCol(falseCol); trackCol(selectorCol)
        ops.append(.select(dst: dst, trueCol: trueCol, falseCol: falseCol, selectorCol: selectorCol))
        return self
    }

    // MARK: - Common patterns

    /// Range decomposition: decompose column `col` into `bits` bit-columns starting at `startCol`.
    /// Returns (startCol + bits) for chaining.
    @discardableResult
    public func rangeDecomposition(_ col: Int, bits: Int, startCol: Int) -> Int {
        // This is handled by a separate kernel on GPU, but we record it
        // as metadata. For the instruction stream, we just reserve columns.
        for b in 0..<bits {
            trackCol(startCol + b)
        }
        return startCol + bits
    }

    /// Addition chain: accumulate cols[0] + cols[1] + ... into dstCol.
    @discardableResult
    public func additionChain(_ dstCol: Int, cols: [Int]) -> TraceProgram {
        guard cols.count >= 2 else { return self }
        trackCol(dstCol)
        // First pair
        add(dstCol, cols[0], cols[1])
        // Accumulate rest
        for i in 2..<cols.count {
            add(dstCol, dstCol, cols[i])
        }
        return self
    }

    /// One round of Poseidon2 (t=3): apply round constants, S-box, MDS mix.
    /// stateCols: 3 columns holding the current state
    /// rcCol: 3 columns for round constants (pre-loaded)
    /// tempCol: 1 temporary column
    @discardableResult
    public func poseidon2Round(stateCols: [Int], rcCols: [Int], tempCol: Int, fullRound: Bool) -> TraceProgram {
        precondition(stateCols.count == 3 && rcCols.count == 3)
        let s0 = stateCols[0], s1 = stateCols[1], s2 = stateCols[2]
        let rc0 = rcCols[0], rc1 = rcCols[1], rc2 = rcCols[2]

        // Add round constants
        add(s0, s0, rc0)
        add(s1, s1, rc1)
        add(s2, s2, rc2)

        if fullRound {
            // Full S-box: x^5 on all state elements
            for s in stateCols {
                sqr(tempCol, s)           // t = s^2
                sqr(tempCol, tempCol)     // t = s^4
                mul(s, s, tempCol)        // s = s^5
            }
        } else {
            // Partial S-box: x^5 only on s0
            sqr(tempCol, s0)
            sqr(tempCol, tempCol)
            mul(s0, s0, tempCol)
        }

        // MDS mix (Poseidon2 M4 matrix for t=3: simple diffusion)
        // t = s0 + s1 + s2
        add(tempCol, s0, s1)
        add(tempCol, tempCol, s2)
        // s0 = t + s0, s1 = t + s1, s2 = t + s2
        add(s0, s0, tempCol)
        add(s1, s1, tempCol)
        add(s2, s2, tempCol)

        return self
    }

    /// Compile the program into a packed instruction stream for GPU.
    public func compile() -> CompiledProgram {
        var instrs = [UInt32]()
        var constants = [Fr]()
        var constMap = [String: Int]()  // Deduplicate constants

        func constIndex(_ c: Fr) -> Int {
            let key = "\(c.v.0)-\(c.v.1)-\(c.v.2)-\(c.v.3)-\(c.v.4)-\(c.v.5)-\(c.v.6)-\(c.v.7)"
            if let idx = constMap[key] { return idx }
            let idx = constants.count
            constants.append(c)
            constMap[key] = idx
            return idx
        }

        let OP_ADD: UInt32       = 0
        let OP_MUL: UInt32       = 1
        let OP_COPY: UInt32      = 2
        let OP_SUB: UInt32       = 3
        let OP_LOAD: UInt32      = 4
        let OP_SQR: UInt32       = 5
        let OP_DOUBLE: UInt32    = 6
        let OP_NEG: UInt32       = 7
        let OP_SELECT: UInt32    = 8
        let FLAG_CONST: UInt32   = 0x80

        for op in ops {
            switch op {
            case .add(let dst, let src1, let src2):
                instrs += [OP_ADD, UInt32(dst), UInt32(src1), UInt32(src2)]
            case .mul(let dst, let src1, let src2):
                instrs += [OP_MUL, UInt32(dst), UInt32(src1), UInt32(src2)]
            case .sub(let dst, let src1, let src2):
                instrs += [OP_SUB, UInt32(dst), UInt32(src1), UInt32(src2)]
            case .copy(let dst, let src):
                instrs += [OP_COPY, UInt32(dst), UInt32(src), 0]
            case .loadInput(let dst, let inputIdx):
                instrs += [OP_LOAD, UInt32(dst), UInt32(inputIdx), 0]
            case .addConst(let dst, let src, let c):
                let ci = constIndex(c)
                instrs += [OP_ADD | FLAG_CONST, UInt32(dst), UInt32(src), UInt32(ci)]
            case .mulConst(let dst, let src, let c):
                let ci = constIndex(c)
                instrs += [OP_MUL | FLAG_CONST, UInt32(dst), UInt32(src), UInt32(ci)]
            case .subConst(let dst, let src, let c):
                let ci = constIndex(c)
                instrs += [OP_SUB | FLAG_CONST, UInt32(dst), UInt32(src), UInt32(ci)]
            case .sqr(let dst, let src):
                instrs += [OP_SQR, UInt32(dst), UInt32(src), 0]
            case .double_(let dst, let src):
                instrs += [OP_DOUBLE, UInt32(dst), UInt32(src), 0]
            case .neg(let dst, let src):
                instrs += [OP_NEG, UInt32(dst), UInt32(src), 0]
            case .select(let dst, let trueCol, let falseCol, let selectorCol):
                let packed = UInt32(falseCol & 0xFFFF) | (UInt32(selectorCol & 0xFFFF) << 16)
                instrs += [OP_SELECT, UInt32(dst), UInt32(trueCol), packed]
            }
        }

        return CompiledProgram(
            instructions: instrs,
            constants: constants,
            numInstructions: ops.count,
            numCols: maxCol + 1,
            inputWidth: maxInput + 1
        )
    }
}

// MARK: - TraceEngine

public class TraceEngine {
    public static let version = Versions.witness

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let evalFunction: MTLComputePipelineState
    let rangeDecomposeFunction: MTLComputePipelineState

    // Cached buffers
    private var cachedTraceBuf: MTLBuffer?
    private var cachedTraceSize: Int = 0
    private var cachedInputBuf: MTLBuffer?
    private var cachedInputSize: Int = 0

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try TraceEngine.compileShaders(device: device)

        guard let evalFn = library.makeFunction(name: "trace_evaluate"),
              let rangeFn = library.makeFunction(name: "trace_range_decompose") else {
            throw MSMError.missingKernel
        }

        self.evalFunction = try device.makeComputePipelineState(function: evalFn)
        self.rangeDecomposeFunction = try device.makeComputePipelineState(function: rangeFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let traceSource = try String(contentsOfFile: shaderDir + "/witness/trace_eval.metal", encoding: .utf8)

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let traceClean = traceSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + traceClean

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    /// Evaluate a trace program on GPU.
    /// - Parameters:
    ///   - program: Compiled instruction stream
    ///   - inputs: Per-row input values, flattened [row0_in0, row0_in1, ..., row1_in0, ...]
    ///   - numRows: Number of rows in the trace
    /// - Returns: MTLBuffer containing numRows * numCols Fr elements (row-major)
    public func evaluate(program: CompiledProgram,
                         inputs: [Fr],
                         numRows: Int) throws -> MTLBuffer {
        let numCols = program.numCols
        let stride = MemoryLayout<Fr>.stride
        let traceBytes = numRows * numCols * stride

        // Allocate or reuse trace buffer
        let traceBuf: MTLBuffer
        if traceBytes <= cachedTraceSize, let cached = cachedTraceBuf {
            traceBuf = cached
        } else {
            guard let buf = device.makeBuffer(length: traceBytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate trace buffer (\(traceBytes) bytes)")
            }
            traceBuf = buf
            cachedTraceBuf = buf
            cachedTraceSize = traceBytes
        }

        // Zero the trace buffer
        memset(traceBuf.contents(), 0, traceBytes)

        // Input buffer
        let inputBytes = inputs.count * stride
        let inputBuf: MTLBuffer
        if inputBytes <= cachedInputSize, let cached = cachedInputBuf {
            inputBuf = cached
        } else {
            guard let buf = device.makeBuffer(length: max(inputBytes, stride), options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate input buffer")
            }
            inputBuf = buf
            cachedInputBuf = buf
            cachedInputSize = max(inputBytes, stride)
        }
        if !inputs.isEmpty {
            inputs.withUnsafeBytes { src in
                memcpy(inputBuf.contents(), src.baseAddress!, inputBytes)
            }
        }

        // Program buffer
        guard let progBuf = device.makeBuffer(
            bytes: program.instructions,
            length: program.instructions.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate program buffer")
        }

        // Constant pool buffer
        let constBuf: MTLBuffer
        if program.constants.isEmpty {
            // Need at least a valid buffer
            guard let buf = device.makeBuffer(length: stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate constant buffer")
            }
            constBuf = buf
        } else {
            guard let buf = device.makeBuffer(
                bytes: program.constants,
                length: program.constants.count * stride,
                options: .storageModeShared
            ) else {
                throw MSMError.gpuError("Failed to allocate constant buffer")
            }
            constBuf = buf
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(evalFunction)
        enc.setBuffer(traceBuf, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(progBuf, offset: 0, index: 2)
        enc.setBuffer(constBuf, offset: 0, index: 3)
        var numColsVal = UInt32(numCols)
        var numInstrsVal = UInt32(program.numInstructions)
        var inputWidthVal = UInt32(program.inputWidth)
        var numRowsVal = UInt32(numRows)
        enc.setBytes(&numColsVal, length: 4, index: 4)
        enc.setBytes(&numInstrsVal, length: 4, index: 5)
        enc.setBytes(&inputWidthVal, length: 4, index: 6)
        enc.setBytes(&numRowsVal, length: 4, index: 7)

        let tg = min(256, Int(evalFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: numRows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return traceBuf
    }

    /// Convenience: evaluate and return as Swift array.
    public func evaluateToArray(program: CompiledProgram,
                                inputs: [Fr],
                                numRows: Int) throws -> [[Fr]] {
        let buf = try evaluate(program: program, inputs: inputs, numRows: numRows)
        let numCols = program.numCols
        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: numRows * numCols)
        var result = [[Fr]]()
        result.reserveCapacity(numRows)
        for row in 0..<numRows {
            var rowData = [Fr]()
            rowData.reserveCapacity(numCols)
            for col in 0..<numCols {
                rowData.append(ptr[row * numCols + col])
            }
            result.append(rowData)
        }
        return result
    }

    /// Evaluate on a pre-allocated trace buffer (for pipelining).
    public func evaluate(program: CompiledProgram,
                         inputBuffer: MTLBuffer,
                         traceBuffer: MTLBuffer,
                         numRows: Int) throws {
        let numCols = program.numCols

        guard let progBuf = device.makeBuffer(
            bytes: program.instructions,
            length: program.instructions.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate program buffer")
        }

        let stride = MemoryLayout<Fr>.stride
        let constBuf: MTLBuffer
        if program.constants.isEmpty {
            guard let buf = device.makeBuffer(length: stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate constant buffer")
            }
            constBuf = buf
        } else {
            guard let buf = device.makeBuffer(
                bytes: program.constants,
                length: program.constants.count * stride,
                options: .storageModeShared
            ) else {
                throw MSMError.gpuError("Failed to allocate constant buffer")
            }
            constBuf = buf
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(evalFunction)
        enc.setBuffer(traceBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuffer, offset: 0, index: 1)
        enc.setBuffer(progBuf, offset: 0, index: 2)
        enc.setBuffer(constBuf, offset: 0, index: 3)
        var numColsVal = UInt32(numCols)
        var numInstrsVal = UInt32(program.numInstructions)
        var inputWidthVal = UInt32(program.inputWidth)
        var numRowsVal = UInt32(numRows)
        enc.setBytes(&numColsVal, length: 4, index: 4)
        enc.setBytes(&numInstrsVal, length: 4, index: 5)
        enc.setBytes(&inputWidthVal, length: 4, index: 6)
        enc.setBytes(&numRowsVal, length: 4, index: 7)

        let tg = min(256, Int(evalFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: numRows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }
}
