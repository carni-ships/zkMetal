// WitnessEngine — GPU-accelerated M31 witness/trace generation
//
// Accelerates the trace computation phase of Circle STARK proving.
// For many circuits, witness generation is 30-50% of total proving time.
//
// Strategies by constraint type:
//   1. Independent rows (each row computed from inputs only):
//      → One GPU thread per row, instruction-stream interpreter
//   2. Linear recurrence (Fibonacci AIR, state machines with constant matrix):
//      → Matrix-power doubling: each thread computes M^row * state0 in O(log n)
//   3. Generic sequential: falls back to CPU

import Foundation
import Metal

// MARK: - M31 Trace Program (instruction builder)

/// Compiled M31 instruction stream for GPU.
public struct CompiledM31Program {
    public let instructions: [UInt32]
    public let constants: [UInt32]   // M31.v values
    public let numInstructions: Int
    public let numCols: Int
    public let inputWidth: Int
}

/// Builder for M31 trace evaluation programs.
public class M31TraceProgram {
    public enum Op {
        case add(dst: Int, src1: Int, src2: Int)
        case mul(dst: Int, src1: Int, src2: Int)
        case sub(dst: Int, src1: Int, src2: Int)
        case copy(dst: Int, src: Int)
        case loadInput(dst: Int, inputIdx: Int)
        case addConst(dst: Int, src: Int, constant: M31)
        case mulConst(dst: Int, src: Int, constant: M31)
        case subConst(dst: Int, src: Int, constant: M31)
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
    public func add(_ dst: Int, _ src1: Int, _ src2: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.add(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func mul(_ dst: Int, _ src1: Int, _ src2: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.mul(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func sub(_ dst: Int, _ src1: Int, _ src2: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src1); trackCol(src2)
        ops.append(.sub(dst: dst, src1: src1, src2: src2))
        return self
    }

    @discardableResult
    public func copy(_ dst: Int, _ src: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.copy(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func loadInput(_ dst: Int, _ inputIdx: Int) -> M31TraceProgram {
        trackCol(dst)
        if inputIdx > maxInput { maxInput = inputIdx }
        ops.append(.loadInput(dst: dst, inputIdx: inputIdx))
        return self
    }

    @discardableResult
    public func addConst(_ dst: Int, _ src: Int, _ c: M31) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.addConst(dst: dst, src: src, constant: c))
        return self
    }

    @discardableResult
    public func mulConst(_ dst: Int, _ src: Int, _ c: M31) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.mulConst(dst: dst, src: src, constant: c))
        return self
    }

    @discardableResult
    public func subConst(_ dst: Int, _ src: Int, _ c: M31) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.subConst(dst: dst, src: src, constant: c))
        return self
    }

    @discardableResult
    public func sqr(_ dst: Int, _ src: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.sqr(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func double_(_ dst: Int, _ src: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.double_(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func neg(_ dst: Int, _ src: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(src)
        ops.append(.neg(dst: dst, src: src))
        return self
    }

    @discardableResult
    public func select(_ dst: Int, trueCol: Int, falseCol: Int, selectorCol: Int) -> M31TraceProgram {
        trackCol(dst); trackCol(trueCol); trackCol(falseCol); trackCol(selectorCol)
        ops.append(.select(dst: dst, trueCol: trueCol, falseCol: falseCol, selectorCol: selectorCol))
        return self
    }

    public func compile() -> CompiledM31Program {
        var instrs = [UInt32]()
        var constants = [UInt32]()
        var constMap = [UInt32: Int]()

        func constIndex(_ c: M31) -> Int {
            if let idx = constMap[c.v] { return idx }
            let idx = constants.count
            constants.append(c.v)
            constMap[c.v] = idx
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

        return CompiledM31Program(
            instructions: instrs,
            constants: constants,
            numInstructions: ops.count,
            numCols: maxCol + 1,
            inputWidth: maxInput + 1
        )
    }
}

// MARK: - WitnessEngine

public class WitnessEngine {
    public static let version = Versions.witness

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states
    let genericEvalPipeline: MTLComputePipelineState
    let fibonacciPipeline: MTLComputePipelineState
    let fibonacciSharedPipeline: MTLComputePipelineState
    let linearRecurrencePipeline: MTLComputePipelineState

    // Buffer cache
    private var cachedBufA: MTLBuffer?
    private var cachedBufASize: Int = 0
    private var cachedBufB: MTLBuffer?
    private var cachedBufBSize: Int = 0
    private var cachedPowersBuf: MTLBuffer?

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try WitnessEngine.compileShaders(device: device)

        guard let evalFn = library.makeFunction(name: "witness_m31_evaluate"),
              let fibFn = library.makeFunction(name: "witness_m31_fibonacci"),
              let fibSharedFn = library.makeFunction(name: "witness_m31_fibonacci_shared"),
              let linRecFn = library.makeFunction(name: "witness_m31_linear_recurrence") else {
            throw MSMError.missingKernel
        }

        self.genericEvalPipeline = try device.makeComputePipelineState(function: evalFn)
        self.fibonacciPipeline = try device.makeComputePipelineState(function: fibFn)
        self.fibonacciSharedPipeline = try device.makeComputePipelineState(function: fibSharedFn)
        self.linearRecurrencePipeline = try device.makeComputePipelineState(function: linRecFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let m31Source = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let witnessSource = try String(contentsOfFile: shaderDir + "/witness/witness_m31.metal", encoding: .utf8)

        let m31Clean = m31Source
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let witnessClean = witnessSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = m31Clean + "\n" + witnessClean

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Fibonacci trace generation

    /// Precompute Fibonacci matrix powers M^{2^j} for j=0..23 on CPU.
    /// M = [[0,1],[1,1]]. Each power is [a,b,c,d] as UInt32.
    private func getOrBuildPowersBuf() throws -> MTLBuffer {
        if let buf = cachedPowersBuf { return buf }

        // M31 2x2 matrix multiplication on CPU
        func matMul(_ x: (UInt32, UInt32, UInt32, UInt32),
                    _ y: (UInt32, UInt32, UInt32, UInt32)) -> (UInt32, UInt32, UInt32, UInt32) {
            let p = UInt64(M31.P)
            let a = UInt32((UInt64(x.0) * UInt64(y.0) + UInt64(x.1) * UInt64(y.2)) % p)
            let b = UInt32((UInt64(x.0) * UInt64(y.1) + UInt64(x.1) * UInt64(y.3)) % p)
            let c = UInt32((UInt64(x.2) * UInt64(y.0) + UInt64(x.3) * UInt64(y.2)) % p)
            let d = UInt32((UInt64(x.2) * UInt64(y.1) + UInt64(x.3) * UInt64(y.3)) % p)
            return (a, b, c, d)
        }

        var powers = [UInt32]()
        powers.reserveCapacity(24 * 4)

        // M^1 = [[0,1],[1,1]]
        var cur: (UInt32, UInt32, UInt32, UInt32) = (0, 1, 1, 1)
        for _ in 0..<24 {
            powers.append(cur.0)
            powers.append(cur.1)
            powers.append(cur.2)
            powers.append(cur.3)
            cur = matMul(cur, cur)  // M^{2^(j+1)} = (M^{2^j})^2
        }

        guard let buf = device.makeBuffer(
            bytes: powers, length: powers.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate powers buffer")
        }
        cachedPowersBuf = buf
        return buf
    }

    /// Generate Fibonacci AIR trace on GPU using matrix-power doubling.
    /// Returns (colA, colB) where a[i+1] = b[i], b[i+1] = a[i] + b[i].
    public func generateFibonacciTrace(a0: M31, b0: M31, numRows: Int) throws -> ([M31], [M31]) {
        let sz = MemoryLayout<UInt32>.stride
        let bufSize = numRows * sz

        let colABuf = try getOrAllocBuffer(cached: &cachedBufA, cachedSize: &cachedBufASize, needed: bufSize)
        let colBBuf = try getOrAllocBuffer(cached: &cachedBufB, cachedSize: &cachedBufBSize, needed: bufSize)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let logN = Int(ceil(log2(Double(max(numRows, 2)))))
        let useShared = numRows >= 256

        let enc = cmdBuf.makeComputeCommandEncoder()!
        if useShared {
            enc.setComputePipelineState(fibonacciSharedPipeline)
        } else {
            enc.setComputePipelineState(fibonacciPipeline)
        }
        enc.setBuffer(colABuf, offset: 0, index: 0)
        enc.setBuffer(colBBuf, offset: 0, index: 1)
        var a0v = a0.v
        var b0v = b0.v
        var nr = UInt32(numRows)
        enc.setBytes(&a0v, length: 4, index: 2)
        enc.setBytes(&b0v, length: 4, index: 3)
        enc.setBytes(&nr, length: 4, index: 4)
        if useShared {
            var logNv = UInt32(logN)
            enc.setBytes(&logNv, length: 4, index: 5)
            let powersBuf = try getOrBuildPowersBuf()
            enc.setBuffer(powersBuf, offset: 0, index: 6)
        }

        let pipeline = useShared ? fibonacciSharedPipeline : fibonacciPipeline
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
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

        // Read results
        let pA = colABuf.contents().bindMemory(to: UInt32.self, capacity: numRows)
        let pB = colBBuf.contents().bindMemory(to: UInt32.self, capacity: numRows)
        var colA = [M31](repeating: M31.zero, count: numRows)
        var colB = [M31](repeating: M31.zero, count: numRows)
        for i in 0..<numRows {
            colA[i] = M31(v: pA[i])
            colB[i] = M31(v: pB[i])
        }
        return (colA, colB)
    }

    /// Generate Fibonacci trace returning GPU buffers (avoids copy for pipeline use).
    public func generateFibonacciTraceGPU(a0: M31, b0: M31, numRows: Int) throws -> (MTLBuffer, MTLBuffer) {
        let sz = MemoryLayout<UInt32>.stride
        let bufSize = numRows * sz

        guard let colABuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let colBBuf = device.makeBuffer(length: bufSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate Fibonacci trace buffers")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let logN = Int(ceil(log2(Double(max(numRows, 2)))))
        let useShared = numRows >= 256

        let enc = cmdBuf.makeComputeCommandEncoder()!
        if useShared {
            enc.setComputePipelineState(fibonacciSharedPipeline)
        } else {
            enc.setComputePipelineState(fibonacciPipeline)
        }
        enc.setBuffer(colABuf, offset: 0, index: 0)
        enc.setBuffer(colBBuf, offset: 0, index: 1)
        var a0v = a0.v
        var b0v = b0.v
        var nr = UInt32(numRows)
        enc.setBytes(&a0v, length: 4, index: 2)
        enc.setBytes(&b0v, length: 4, index: 3)
        enc.setBytes(&nr, length: 4, index: 4)
        if useShared {
            var logNv = UInt32(logN)
            enc.setBytes(&logNv, length: 4, index: 5)
            let powersBuf = try getOrBuildPowersBuf()
            enc.setBuffer(powersBuf, offset: 0, index: 6)
        }

        let pipeline = useShared ? fibonacciSharedPipeline : fibonacciPipeline
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
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

        return (colABuf, colBBuf)
    }

    // MARK: - Generic M31 trace evaluation (independent rows)

    /// Evaluate a compiled M31 trace program on GPU.
    /// Each row is evaluated independently — suitable for circuits where
    /// rows do not depend on each other (given inputs).
    public func evaluate(program: CompiledM31Program,
                         inputs: [UInt32],
                         numRows: Int) throws -> MTLBuffer {
        let numCols = program.numCols
        let sz = MemoryLayout<UInt32>.stride
        let traceBytes = numRows * numCols * sz

        let traceBuf = try getOrAllocBuffer(cached: &cachedBufA, cachedSize: &cachedBufASize, needed: traceBytes)
        memset(traceBuf.contents(), 0, traceBytes)

        // Input buffer
        let inputBytes = max(inputs.count * sz, sz)
        guard let inputBuf = device.makeBuffer(length: inputBytes, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate input buffer")
        }
        if !inputs.isEmpty {
            inputs.withUnsafeBytes { src in
                memcpy(inputBuf.contents(), src.baseAddress!, inputs.count * sz)
            }
        }

        // Program buffer
        guard let progBuf = device.makeBuffer(
            bytes: program.instructions,
            length: program.instructions.count * sz,
            options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate program buffer")
        }

        // Constant pool
        let constBuf: MTLBuffer
        if program.constants.isEmpty {
            guard let buf = device.makeBuffer(length: sz, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate constant buffer")
            }
            constBuf = buf
        } else {
            guard let buf = device.makeBuffer(
                bytes: program.constants,
                length: program.constants.count * sz,
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
        enc.setComputePipelineState(genericEvalPipeline)
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

        let tg = min(256, Int(genericEvalPipeline.maxTotalThreadsPerThreadgroup))
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

    /// Evaluate and return as 2D array.
    public func evaluateToArray(program: CompiledM31Program,
                                inputs: [UInt32],
                                numRows: Int) throws -> [[M31]] {
        let buf = try evaluate(program: program, inputs: inputs, numRows: numRows)
        let numCols = program.numCols
        let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: numRows * numCols)
        var result = [[M31]]()
        result.reserveCapacity(numRows)
        for row in 0..<numRows {
            var rowData = [M31]()
            rowData.reserveCapacity(numCols)
            for col in 0..<numCols {
                rowData.append(M31(v: ptr[row * numCols + col]))
            }
            result.append(rowData)
        }
        return result
    }

    // MARK: - Linear recurrence trace generation

    /// Generate trace for a linear recurrence: state[i] = T * state[i-1]
    /// where T is a constant state_width x state_width transfer matrix.
    /// Returns trace as [row][col] of M31.
    public func generateLinearRecurrence(
        transferMatrix: [[M31]],  // state_width x state_width
        initialState: [M31],
        numRows: Int
    ) throws -> [[M31]] {
        let stateWidth = initialState.count
        precondition(transferMatrix.count == stateWidth)
        for row in transferMatrix { precondition(row.count == stateWidth) }
        precondition(stateWidth <= 8, "State width must be <= 8")

        let sz = MemoryLayout<UInt32>.stride
        let traceBytes = numRows * stateWidth * sz

        let traceBuf = try getOrAllocBuffer(cached: &cachedBufA, cachedSize: &cachedBufASize, needed: traceBytes)

        // Flatten transfer matrix
        var flatMat = [UInt32]()
        flatMat.reserveCapacity(stateWidth * stateWidth)
        for row in transferMatrix {
            for val in row { flatMat.append(val.v) }
        }

        guard let matBuf = device.makeBuffer(
            bytes: flatMat, length: flatMat.count * sz, options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate matrix buffer")
        }

        let stateVals = initialState.map { $0.v }
        guard let stateBuf = device.makeBuffer(
            bytes: stateVals, length: stateVals.count * sz, options: .storageModeShared
        ) else {
            throw MSMError.gpuError("Failed to allocate state buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(linearRecurrencePipeline)
        enc.setBuffer(traceBuf, offset: 0, index: 0)
        enc.setBuffer(matBuf, offset: 0, index: 1)
        enc.setBuffer(stateBuf, offset: 0, index: 2)
        var sw = UInt32(stateWidth)
        var nr = UInt32(numRows)
        enc.setBytes(&sw, length: 4, index: 3)
        enc.setBytes(&nr, length: 4, index: 4)

        let tg = min(256, Int(linearRecurrencePipeline.maxTotalThreadsPerThreadgroup))
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

        let ptr = traceBuf.contents().bindMemory(to: UInt32.self, capacity: numRows * stateWidth)
        var result = [[M31]]()
        result.reserveCapacity(numRows)
        for row in 0..<numRows {
            var rowData = [M31]()
            rowData.reserveCapacity(stateWidth)
            for col in 0..<stateWidth {
                rowData.append(M31(v: ptr[row * stateWidth + col]))
            }
            result.append(rowData)
        }
        return result
    }

    // MARK: - Private helpers

    private func getOrAllocBuffer(cached: inout MTLBuffer?, cachedSize: inout Int, needed: Int) throws -> MTLBuffer {
        if needed <= cachedSize, let buf = cached {
            return buf
        }
        guard let buf = device.makeBuffer(length: needed, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffer (\(needed) bytes)")
        }
        cached = buf
        cachedSize = needed
        return buf
    }
}
