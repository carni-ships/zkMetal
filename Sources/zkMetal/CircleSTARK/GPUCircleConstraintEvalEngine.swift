// GPU Circle Constraint Evaluation Engine for Circle STARK
//
// GPU-accelerated constraint evaluation following EVMetal architecture patterns.
// Uses Metal compute shaders for high-performance constraint evaluation on GPU.
//
// Architecture:
// - Thread-per-point: each thread evaluates one evaluation point
// - Column-major layout: trace[col * evalLen + i]
// - Boundary constraints handled via alpha power offsets
//
// Memory management (from EVMetal):
// - Buffer pool for reuse
// - Memory budget checking (80MB max)
// - OOM protection
//
// Usage:
//   let engine = try GPUCircleConstraintEvalEngine()
//   let result = try engine.evaluateFibonacci(
//     traceA: a, traceB: b, domainY: y,
//     alpha: alpha, bcA0: a0, bcB0: b0,
//     evalLen: evalLen, traceLen: traceLen, logTrace: logTrace
//   )

import Foundation
import Metal

/// GPU error types
public enum GPUConstraintError: Error {
    case deviceNotAvailable
    case libraryCreationFailed(String)
    case kernelCreationFailed(String)
    case bufferCreationFailed(String)
    case commandBufferCreationFailed
    case memoryBudgetExceeded(Int, Int)
    case invalidDimensions(String)
}

/// Result of GPU constraint evaluation
public struct GPUConstraintEvalResult {
    /// Composition polynomial evaluations
    public let composition: [M31]

    /// Time spent on GPU in milliseconds
    public let gpuTimeMs: Double

    /// Time for data transfer in milliseconds
    public let transferTimeMs: Double

    /// Number of evaluation points
    public let evalLen: Int

    /// Number of columns evaluated
    public let numColumns: Int

    /// Whether GPU was actually used (vs CPU fallback)
    public let usedGPU: Bool

    public var totalTimeMs: Double { gpuTimeMs + transferTimeMs }
}

/// GPU-accelerated Circle STARK constraint evaluation engine
public final class GPUCircleConstraintEvalEngine: Sendable {

    // MARK: - Constants

    /// Maximum GPU memory budget for constraint engine (80MB safety margin)
    private static let maxMemoryBudgetBytes = 80 * 1024 * 1024

    /// Maximum evaluation length to use GPU (prevents OOM)
    private static let maxEvalLenGPU = 131072  // 128K max

    /// Threads per threadgroup for constraint evaluation
    private static let threadsPerThreadgroup = 256

    // MARK: - GPU Resources

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    /// Compute pipeline for Fibonacci constraint evaluation
    private let fibonacciPipeline: MTLComputePipelineState

    /// Buffer pool for reuse (reduces allocation overhead)
    private var bufferPool: [MTLBuffer] = []

    // MARK: - Initialization

    /// Initialize the GPU constraint engine
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GPUConstraintError.deviceNotAvailable
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw GPUConstraintError.commandBufferCreationFailed
        }
        self.commandQueue = queue

        // Compile shaders from the constraint kernel file
        let shaderDir = Self.findShaderDir()

        // Load M31 field definitions
        let m31Source = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)

        // Load constraint kernel and remove the include directive (we'll inline M31)
        var constraintSource = try String(contentsOfFile: shaderDir + "/constraint/circle_stark_constraint_m31.metal", encoding: .utf8)

        // Remove the #include line since we're inlining M31
        constraintSource = constraintSource.replacingOccurrences(of: "#include \"../fields/mersenne31.metal\"\n", with: "")
        constraintSource = constraintSource.replacingOccurrences(of: "#include \"../fields/mersenne31.metal\"", with: "")

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        // Combine M31 source with constraint kernel (M31 first, then kernel)
        let combinedSource = m31Source + "\n" + constraintSource

        do {
            self.library = try device.makeLibrary(source: combinedSource, options: options)
        } catch {
            throw GPUConstraintError.libraryCreationFailed(error.localizedDescription)
        }

        // Create pipeline for Fibonacci constraint kernel
        guard let fibFn = library.makeFunction(name: "circle_fibonacci_constraint_eval") else {
            throw GPUConstraintError.kernelCreationFailed("circle_fibonacci_constraint_eval not found")
        }
        self.fibonacciPipeline = try device.makeComputePipelineState(function: fibFn)
    }

    // MARK: - Memory Management

    /// Estimate memory usage for constraint evaluation
    private static func estimateMemoryUsage(evalLen: Int, numColumns: Int = 2) -> Int {
        // Trace buffers: numColumns * evalLen * 4 bytes
        let traceBytes = numColumns * evalLen * MemoryLayout<UInt32>.stride
        // Domain buffer: evalLen * 4 bytes
        let domainBytes = evalLen * MemoryLayout<UInt32>.stride
        // Output buffer: evalLen * 4 bytes
        let outputBytes = evalLen * MemoryLayout<UInt32>.stride
        return traceBytes + domainBytes + outputBytes
    }

    /// Check if GPU can handle given dimensions
    public func canHandle(evalLen: Int, numColumns: Int = 2) -> Bool {
        if evalLen > Self.maxEvalLenGPU { return false }
        let estimated = Self.estimateMemoryUsage(evalLen: evalLen, numColumns: numColumns)
        return estimated <= Self.maxMemoryBudgetBytes
    }

    // MARK: - Fibonacci Constraint Evaluation

    /// Evaluate Fibonacci AIR constraints on GPU
    ///
    /// - Parameters:
    ///   - traceA: Column 0 LDE evaluations (evalLen elements)
    ///   - traceB: Column 1 LDE evaluations (evalLen elements)
    ///   - domainY: Y-coordinates of evaluation domain points (evalLen elements)
    ///   - alpha: Random challenge for composition polynomial
    ///   - bcA0: Boundary constraint value for A at row 0
    ///   - bcB0: Boundary constraint value for B at row 0
    ///   - evalLen: Evaluation domain size
    ///   - traceLen: Trace length
    ///   - logTrace: log2(traceLen)
    /// - Returns: GPU constraint evaluation result with composition polynomial
    public func evaluateFibonacci(
        traceA: [M31],
        traceB: [M31],
        domainY: [M31],
        alpha: M31,
        bcA0: M31,
        bcB0: M31,
        evalLen: Int,
        traceLen: Int,
        logTrace: Int
    ) throws -> GPUConstraintEvalResult {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Validate dimensions
        precondition(traceA.count == evalLen, "traceA count \(traceA.count) != evalLen \(evalLen)")
        precondition(traceB.count == evalLen, "traceB count \(traceB.count) != evalLen \(evalLen)")
        precondition(domainY.count == evalLen, "domainY count \(domainY.count) != evalLen \(evalLen)")

        // Memory check
        guard canHandle(evalLen: evalLen, numColumns: 2) else {
            throw GPUConstraintError.memoryBudgetExceeded(
                Self.estimateMemoryUsage(evalLen: evalLen, numColumns: 2),
                Self.maxMemoryBudgetBytes
            )
        }

        // Flatten inputs to UInt32 arrays
        var traceAFlat = [UInt32](repeating: 0, count: evalLen)
        var traceBFlat = [UInt32](repeating: 0, count: evalLen)
        var domainYFlat = [UInt32](repeating: 0, count: evalLen)

        for i in 0..<evalLen {
            traceAFlat[i] = traceA[i].v
            traceBFlat[i] = traceB[i].v
            domainYFlat[i] = domainY[i].v
        }

        let transferT0 = CFAbsoluteTimeGetCurrent()

        // Allocate GPU buffers
        let traceASize = evalLen * MemoryLayout<UInt32>.stride
        let traceBSize = evalLen * MemoryLayout<UInt32>.stride
        let domainYSize = evalLen * MemoryLayout<UInt32>.stride
        let outputSize = evalLen * MemoryLayout<UInt32>.stride

        guard let traceABuf = device.makeBuffer(bytes: traceAFlat, length: traceASize, options: .storageModeShared),
              let traceBBuf = device.makeBuffer(bytes: traceBFlat, length: traceBSize, options: .storageModeShared),
              let domainYBuf = device.makeBuffer(bytes: domainYFlat, length: domainYSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw GPUConstraintError.bufferCreationFailed("Failed to allocate GPU buffers")
        }

        let transferTimeMs = (CFAbsoluteTimeGetCurrent() - transferT0) * 1000

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUConstraintError.commandBufferCreationFailed
        }

        encoder.setComputePipelineState(fibonacciPipeline)
        encoder.setBuffer(traceABuf, offset: 0, index: 0)
        encoder.setBuffer(traceBBuf, offset: 0, index: 1)
        encoder.setBuffer(domainYBuf, offset: 0, index: 2)
        encoder.setBuffer(outputBuf, offset: 0, index: 3)

        // Set constants via setBytes
        var alphaVal = alpha.v
        var bcA0Val = bcA0.v
        var bcB0Val = bcB0.v
        var evalLenVal = UInt32(evalLen)
        var traceLenVal = UInt32(traceLen)
        var logTraceVal = UInt32(logTrace)

        encoder.setBytes(&alphaVal, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&bcA0Val, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&bcB0Val, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&evalLenVal, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&traceLenVal, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&logTraceVal, length: MemoryLayout<UInt32>.stride, index: 9)

        // Dispatch
        let threadsPerThreadgroup = MTLSize(width: Self.threadsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (evalLen + Self.threadsPerThreadgroup - 1) / Self.threadsPerThreadgroup, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw GPUConstraintError.commandBufferCreationFailed
        }

        // Read back results
        let gpuTimeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000 - transferTimeMs
        let outputPtr = outputBuf.contents().bindMemory(to: UInt32.self, capacity: evalLen)
        var composition = [M31](repeating: M31.zero, count: evalLen)
        for i in 0..<evalLen {
            composition[i] = M31(v: outputPtr[i])
        }

        return GPUConstraintEvalResult(
            composition: composition,
            gpuTimeMs: gpuTimeMs,
            transferTimeMs: transferTimeMs,
            evalLen: evalLen,
            numColumns: 2,
            usedGPU: true
        )
    }

    // MARK: - Helper

    private static func findShaderDir() -> String {
        // Find the shader directory relative to the module
        let possiblePaths = [
            "Sources/Shaders",
            "../Sources/Shaders",
            "../../Sources/Shaders",
            "/Users/carnation/Documents/Claude/zkMetal/Sources/Shaders"
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }

        return "Sources/Shaders"
    }
}

// MARK: - Integration Helper for GPUCircleSTARKProverEngine

/// Extension to evaluate constraints with GPU acceleration when available
extension GPUCircleSTARKProverEngine {

    /// Try GPU constraint evaluation, fall back to CPU if unavailable or too small
    func evaluateConstraintsGPUIfAvailable<A: CircleAIR>(
        air: A,
        traceLDEs: [[M31]],
        alpha: M31,
        logTrace: Int,
        logEval: Int
    ) throws -> [M31] {
        let evalLen = 1 << logEval
        let traceLen = 1 << logTrace

        // Check if we should use GPU
        if !gpuAvailable || evalLen < config.gpuConstraintThreshold {
            // Use CPU path
            return evaluateConstraints(air: air, traceLDEs: traceLDEs, alpha: alpha, logTrace: logTrace, logEval: logEval)
        }

        // Get domain y-coordinates for vanishing polynomial
        let domainY = circleCosetDomain(logN: logEval).map { $0.y }

        // For Fibonacci AIR, use GPU directly
        if let fibAir = air as? FibonacciAIR {
            let constraintEngine = try GPUCircleConstraintEvalEngine()

            guard constraintEngine.canHandle(evalLen: evalLen, numColumns: 2) else {
                // Too large for GPU, use CPU
                return evaluateConstraints(air: air, traceLDEs: traceLDEs, alpha: alpha, logTrace: logTrace, logEval: logEval)
            }

            let result = try constraintEngine.evaluateFibonacci(
                traceA: traceLDEs[0],
                traceB: traceLDEs[1],
                domainY: domainY,
                alpha: alpha,
                bcA0: fibAir.a0,
                bcB0: fibAir.b0,
                evalLen: evalLen,
                traceLen: traceLen,
                logTrace: logTrace
            )

            return result.composition
        }

        // For other AIRs, fall back to CPU (generic path)
        return evaluateConstraints(air: air, traceLDEs: traceLDEs, alpha: alpha, logTrace: logTrace, logEval: logEval)
    }
}
