// Constraint Engine — compiles constraint systems to GPU pipelines and evaluates them
// Uses runtime Metal compilation (device.makeLibrary(source:options:)) like Poseidon2Engine.

import Foundation
import Metal

/// A compiled constraint system ready for GPU evaluation
public class CompiledConstraints {
    public let evalPipeline: MTLComputePipelineState
    public let quotientPipeline: MTLComputePipelineState?
    public let system: ConstraintSystem
    public let shaderSource: String  // for debugging
    public let compileTimeMs: Double

    init(evalPipeline: MTLComputePipelineState,
         quotientPipeline: MTLComputePipelineState?,
         system: ConstraintSystem,
         shaderSource: String,
         compileTimeMs: Double) {
        self.evalPipeline = evalPipeline
        self.quotientPipeline = quotientPipeline
        self.system = system
        self.shaderSource = shaderSource
        self.compileTimeMs = compileTimeMs
    }
}

/// Engine that compiles and evaluates constraint systems on GPU
public class ConstraintEngine {
    public static let version = Versions.constraint

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let frSource: String  // BN254 Fr field arithmetic source
    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        // Load field arithmetic source (same pattern as Poseidon2Engine)
        let shaderDir = findShaderDir()
        let rawFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        self.frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        self.tuning = TuningManager.shared.config(device: device)
    }

    // MARK: - Compile

    /// Compile a constraint system into GPU pipelines.
    /// Generates Metal source, compiles at runtime, and returns a reusable compiled object.
    public func compile(system: ConstraintSystem, includeQuotient: Bool = false) throws -> CompiledConstraints {
        let t0 = CFAbsoluteTimeGetCurrent()

        let codegen = MetalCodegen()

        // Generate constraint evaluation kernel
        let evalSource = codegen.generateConstraintEval(system: system)

        // Strip #include from generated code (already have fr source)
        let cleanEval = evalSource.split(separator: "\n")
            .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
            .joined(separator: "\n")

        var fullSource = frSource + "\n" + cleanEval

        // Optionally generate quotient kernel
        var quotientPipeline: MTLComputePipelineState? = nil
        if includeQuotient {
            let quotientSource = codegen.generateQuotientEval(system: system)
            let cleanQuotient = quotientSource.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
                .joined(separator: "\n")
            fullSource += "\n" + cleanQuotient
        }

        // Compile Metal library from source
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: fullSource, options: options)
        } catch {
            // On compile failure, include the generated source for debugging
            fputs("Metal compilation failed. Generated source:\n\(fullSource)\n", stderr)
            throw MSMError.gpuError("Metal compile error: \(error.localizedDescription)")
        }

        guard let evalFn = library.makeFunction(name: "eval_constraints") else {
            throw MSMError.missingKernel
        }
        let evalPipeline = try device.makeComputePipelineState(function: evalFn)

        if includeQuotient {
            if let quotientFn = library.makeFunction(name: "eval_quotient") {
                quotientPipeline = try device.makeComputePipelineState(function: quotientFn)
            }
        }

        let compileTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        return CompiledConstraints(
            evalPipeline: evalPipeline,
            quotientPipeline: quotientPipeline,
            system: system,
            shaderSource: fullSource,
            compileTimeMs: compileTime
        )
    }

    // MARK: - Evaluate

    /// Evaluate all constraints on a trace buffer.
    /// trace: MTLBuffer containing numRows * numWires Fr values (row-major).
    /// Returns: MTLBuffer with numRows * numConstraints Fr values.
    /// Each value is the constraint evaluation at that row; zero means satisfied.
    public func evaluate(compiled: CompiledConstraints,
                         trace: MTLBuffer,
                         numRows: Int) throws -> MTLBuffer {
        let numConstraints = compiled.system.constraints.count
        let numCols = compiled.system.numWires
        let outputSize = numRows * numConstraints * MemoryLayout<Fr>.stride

        guard let outputBuf = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate constraint output buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(compiled.evalPipeline)
        enc.setBuffer(trace, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var cols = UInt32(numCols)
        var rows = UInt32(numRows)
        enc.setBytes(&cols, length: 4, index: 2)
        enc.setBytes(&rows, length: 4, index: 3)

        let tg = min(256, Int(compiled.evalPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numRows, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Constraint eval GPU error: \(error.localizedDescription)")
        }

        return outputBuf
    }

    /// Evaluate constraints and return results as a Swift array.
    public func evaluateToArray(compiled: CompiledConstraints,
                                trace: MTLBuffer,
                                numRows: Int) throws -> [Fr] {
        let outputBuf = try evaluate(compiled: compiled, trace: trace, numRows: numRows)
        let numConstraints = compiled.system.constraints.count
        let count = numRows * numConstraints
        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - Verify

    /// Check if all constraints are satisfied (all evaluations are zero).
    /// Returns true if the trace satisfies all constraints.
    public func verify(compiled: CompiledConstraints,
                       trace: MTLBuffer,
                       numRows: Int) throws -> Bool {
        let results = try evaluateToArray(compiled: compiled, trace: trace, numRows: numRows)
        for r in results {
            if !r.isZero { return false }
        }
        return true
    }

    // MARK: - Trace Helpers

    /// Create a trace buffer from a 2D array of Fr values.
    /// trace[row][col] layout, row-major.
    public func createTrace(_ data: [[Fr]]) throws -> MTLBuffer {
        guard !data.isEmpty else {
            throw MSMError.invalidInput
        }
        let numCols = data[0].count
        let numRows = data.count
        let stride = MemoryLayout<Fr>.stride
        let size = numRows * numCols * stride

        guard let buf = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate trace buffer")
        }

        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: numRows * numCols)
        for row in 0..<numRows {
            precondition(data[row].count == numCols, "All rows must have same width")
            for col in 0..<numCols {
                ptr[row * numCols + col] = data[row][col]
            }
        }
        return buf
    }

    /// Create a trace buffer from a flat array (row-major, numCols per row).
    public func createTraceFlat(_ data: [Fr], numCols: Int) throws -> MTLBuffer {
        precondition(data.count % numCols == 0, "Data length must be multiple of numCols")
        let stride = MemoryLayout<Fr>.stride
        let size = data.count * stride

        guard let buf = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate trace buffer")
        }

        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, size)
        }
        return buf
    }

    // MARK: - CPU Reference Evaluation (for benchmarking)

    /// Evaluate constraints on CPU for comparison.
    /// Returns array of constraint values, same layout as GPU output.
    public func evaluateCPU(system: ConstraintSystem, trace: [[Fr]]) -> [Fr] {
        let numRows = trace.count
        let numConstraints = system.constraints.count
        var output = [Fr](repeating: Fr.zero, count: numRows * numConstraints)

        for row in 0..<numRows {
            for (ci, constraint) in system.constraints.enumerated() {
                output[row * numConstraints + ci] = evalExprCPU(constraint.expr, trace: trace, row: row)
            }
        }
        return output
    }

    private func evalExprCPU(_ expr: Expr, trace: [[Fr]], row: Int) -> Fr {
        switch expr {
        case .wire(let w):
            let r = row + w.row
            if r < 0 || r >= trace.count { return Fr.zero }
            return trace[r][w.index]
        case .constant(let fr):
            return fr
        case .add(let a, let b):
            return frAdd(evalExprCPU(a, trace: trace, row: row),
                         evalExprCPU(b, trace: trace, row: row))
        case .mul(let a, let b):
            return frMul(evalExprCPU(a, trace: trace, row: row),
                         evalExprCPU(b, trace: trace, row: row))
        case .neg(let a):
            return frSub(Fr.zero, evalExprCPU(a, trace: trace, row: row))
        case .pow(let base, let n):
            let b = evalExprCPU(base, trace: trace, row: row)
            var result = b
            for _ in 1..<n {
                result = frMul(result, b)
            }
            return result
        }
    }
}
