// Fused NTT + Constraint Evaluation Engine
// Eliminates host round-trip between NTT and constraint evaluation by encoding
// both into a single Metal command buffer with memory barriers.
//
// Two approaches:
// 1. Small sizes (logN <= 9): Fully fused kernel — NTT in shared memory + constraint eval
//    (logN <= 9 because BN254 Fr is 32B/elem; shared_a[512]+shared_b[512] = 32KB fits in 32KB TG limit)
// 2. Large sizes: Single command buffer — encode NTT dispatches, barrier, constraint eval dispatch

import Foundation
import Metal

public class FusedNTTConstraintEngine {
    public static let version = Versions.fusedNTTConstraint

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let nttEngine: NTTEngine
    public let constraintEngine: ConstraintEngine

    // Fused small-NTT + Fibonacci constraint pipeline (compiled from Metal source)
    private let fusedFibPipeline: MTLComputePipelineState
    // Post-NTT Fibonacci constraint pipeline (phase 2 for large sizes)
    private let postNTTFibPipeline: MTLComputePipelineState

    private let tuning: TuningConfig

    // Cached buffers
    private var cachedAlphaBuf: MTLBuffer?
    private var cachedVanishingInvBuf: MTLBuffer?
    private var cachedVanishingInvSize: Int = 0

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        self.nttEngine = try NTTEngine()
        self.constraintEngine = try ConstraintEngine()
        self.tuning = TuningManager.shared.config(device: device)

        // Compile fused shaders
        let library = try FusedNTTConstraintEngine.compileShaders(device: device)

        guard let fusedFibFn = library.makeFunction(name: "fused_ntt_fib_constraint"),
              let postNTTFibFn = library.makeFunction(name: "eval_constraints_fib_post_ntt") else {
            throw MSMError.missingKernel
        }

        self.fusedFibPipeline = try device.makeComputePipelineState(function: fusedFibFn)
        self.postNTTFibPipeline = try device.makeComputePipelineState(function: postNTTFibFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fusedSource = try String(contentsOfFile: shaderDir + "/constraint/fused_ntt_constraint.metal", encoding: .utf8)

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let fusedClean = fusedSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + fusedClean

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        do {
            return try device.makeLibrary(source: combined, options: options)
        } catch {
            fputs("Fused NTT+Constraint shader compile failed:\n\(combined)\n", stderr)
            throw MSMError.gpuError("Metal compile error: \(error.localizedDescription)")
        }
    }

    // MARK: - Fused Fibonacci Constraint Evaluation

    /// Fused NTT + Fibonacci constraint quotient evaluation (small sizes, logN <= 9).
    /// Takes raw trace columns [a, b], returns quotient polynomial evaluations.
    public func evaluateFibQuotientFused(
        traceA: [Fr],
        traceB: [Fr],
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        precondition(traceA.count == n && traceB.count == n, "Trace size must be 2^logN")
        precondition(logN <= 9, "Fused kernel supports logN <= 9 (BN254 Fr 32B/elem needs shared mem limit)")

        let stride = MemoryLayout<Fr>.stride

        // Create buffers
        guard let bufA = device.makeBuffer(bytes: traceA, length: n * stride, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: traceB, length: n * stride, options: .storageModeShared),
              let quotientBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        // Twiddle factors
        let twiddles = precomputeTwiddles(logN: logN)
        guard let twiddleBuf = device.makeBuffer(bytes: twiddles, length: twiddles.count * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate twiddle buffer")
        }

        // Alpha powers: [alpha^0, alpha^1] = [1, alpha]
        let alphaPowers = [Fr.one, alpha]
        let alphaBuf = getAlphaBuffer(alphaPowers)

        // Vanishing polynomial inverse (trivial for demo: all ones)
        let vanishingInvBuf = getVanishingInvBuffer(n: n)

        // Dispatch
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedFibPipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(twiddleBuf, offset: 0, index: 2)
        enc.setBuffer(quotientBuf, offset: 0, index: 3)
        enc.setBuffer(alphaBuf, offset: 0, index: 4)
        enc.setBuffer(vanishingInvBuf, offset: 0, index: 5)
        var nVal = UInt32(n)
        var logNVal = UInt32(logN)
        enc.setBytes(&nVal, length: 4, index: 6)
        enc.setBytes(&logNVal, length: 4, index: 7)

        let tgSize = n / 2  // each thread handles one butterfly pair
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Fused NTT+constraint GPU error: \(error.localizedDescription)")
        }

        let ptr = quotientBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Single Command Buffer (large sizes)

    /// NTT + Fibonacci constraint quotient in a single command buffer.
    /// Works at any size. Encodes NTT dispatches for both columns, memory barrier,
    /// then constraint eval dispatch — all in one command buffer (no host round-trip).
    public func evaluateFibQuotientBarrier(
        traceA: [Fr],
        traceB: [Fr],
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        precondition(traceA.count == n && traceB.count == n, "Trace size must be 2^logN")

        let stride = MemoryLayout<Fr>.stride

        // Separate buffers for NTT (column-wise), then constraint eval reads both
        guard let bufA = device.makeBuffer(bytes: traceA, length: n * stride, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: traceB, length: n * stride, options: .storageModeShared),
              let quotientBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        // Alpha powers and vanishing inverse
        let alphaPowers = [Fr.one, alpha]
        let alphaBuf = getAlphaBuffer(alphaPowers)
        let vanishingInvBuf = getVanishingInvBuffer(n: n)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: NTT both columns (encoded into same command buffer)
        nttEngine.encodeNTT(data: bufA, logN: logN, cmdBuf: cmdBuf)
        nttEngine.encodeNTT(data: bufB, logN: logN, cmdBuf: cmdBuf)

        // Phase 2: Constraint evaluation on NTT'd data
        // Use a kernel that reads from separate column buffers (no interleave needed)
        try dispatchSeparateColumnConstraintEval(
            cmdBuf: cmdBuf,
            bufA: bufA, bufB: bufB,
            quotientBuf: quotientBuf,
            alphaBuf: alphaBuf,
            vanishingInvBuf: vanishingInvBuf,
            n: n
        )

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Barrier NTT+constraint GPU error: \(error.localizedDescription)")
        }

        let ptr = quotientBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Dispatch constraint evaluation that reads from separate column buffers (post-NTT).
    private func dispatchSeparateColumnConstraintEval(
        cmdBuf: MTLCommandBuffer,
        bufA: MTLBuffer, bufB: MTLBuffer,
        quotientBuf: MTLBuffer,
        alphaBuf: MTLBuffer,
        vanishingInvBuf: MTLBuffer,
        n: Int
    ) throws {
        // Compile a runtime kernel that reads from separate columns
        let pipeline = try getSeparateColumnFibPipeline()

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(quotientBuf, offset: 0, index: 2)
        enc.setBuffer(alphaBuf, offset: 0, index: 3)
        enc.setBuffer(vanishingInvBuf, offset: 0, index: 4)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 5)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    private var _separateColPipeline: MTLComputePipelineState?

    private func getSeparateColumnFibPipeline() throws -> MTLComputePipelineState {
        if let p = _separateColPipeline { return p }

        let shaderDir = findShaderDir()
        let rawFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        kernel void eval_fib_separate_cols(
            device const Fr* col_a       [[buffer(0)]],
            device const Fr* col_b       [[buffer(1)]],
            device Fr* quotient          [[buffer(2)]],
            device const Fr* alpha_powers [[buffer(3)]],
            device const Fr* vanishing_inv [[buffer(4)]],
            constant uint& num_rows      [[buffer(5)]],
            uint gid [[thread_position_in_grid]]
        ) {
            uint row = gid;
            if (row >= num_rows) return;

            Fr a = col_a[row];
            Fr b = col_b[row];

            uint next_row = (row + 1 < num_rows) ? row + 1 : 0;
            Fr a_next = col_a[next_row];
            Fr b_next = col_b[next_row];

            Fr c0 = fr_sub(a_next, b);
            Fr c1 = fr_sub(b_next, fr_add(a, b));

            Fr acc = fr_add(fr_mul(alpha_powers[0], c0), fr_mul(alpha_powers[1], c1));
            quotient[row] = fr_mul(acc, vanishing_inv[row]);
        }
        """

        let combined = frSource + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)
        guard let fn = library.makeFunction(name: "eval_fib_separate_cols") else {
            throw MSMError.missingKernel
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        _separateColPipeline = pipeline
        return pipeline
    }

    // MARK: - Separate (baseline) execution

    /// Baseline: NTT then constraint eval as two separate dispatches (with host round-trip).
    /// For benchmarking comparison against the fused approach.
    public func evaluateFibQuotientSeparate(
        traceA: [Fr],
        traceB: [Fr],
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        precondition(traceA.count == n && traceB.count == n)

        let stride = MemoryLayout<Fr>.stride

        guard let bufA = device.makeBuffer(bytes: traceA, length: n * stride, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: traceB, length: n * stride, options: .storageModeShared),
              let quotientBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        // Dispatch 1: NTT column A
        try nttEngine.ntt(data: bufA, logN: logN)

        // Dispatch 2: NTT column B
        try nttEngine.ntt(data: bufB, logN: logN)

        // Host round-trip happens here (NTT completes, CPU wakes, dispatches constraint eval)

        // Dispatch 3: Constraint evaluation
        let alphaPowers = [Fr.one, alpha]
        let alphaBuf = getAlphaBuffer(alphaPowers)
        let vanishingInvBuf = getVanishingInvBuffer(n: n)

        let pipeline = try getSeparateColumnFibPipeline()

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(quotientBuf, offset: 0, index: 2)
        enc.setBuffer(alphaBuf, offset: 0, index: 3)
        enc.setBuffer(vanishingInvBuf, offset: 0, index: 4)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 5)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Separate constraint eval GPU error: \(error.localizedDescription)")
        }

        let ptr = quotientBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - General Constraint System (fully fused in shared memory)

    /// Fully fused NTT + general constraint evaluation in a single kernel dispatch.
    /// NTT runs entirely in threadgroup shared memory, then constraints are evaluated
    /// on the NTT'd values without ever writing NTT output to global memory.
    ///
    /// Supports logN <= maxFusedLogN(numCols). For larger sizes, falls back to
    /// the barrier-based approach (separate NTT + constraint dispatches in one command buffer).
    public func evaluateQuotientFused(
        traceColumns: [[Fr]],
        system: ConstraintSystem,
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        let numCols = traceColumns.count
        precondition(numCols == system.numWires, "Column count must match numWires")
        for col in traceColumns {
            precondition(col.count == n, "All columns must have 2^logN elements")
        }

        let maxFused = MetalCodegen.maxFusedLogN(numCols: numCols)

        if logN <= maxFused {
            // Fully fused path: single kernel, NTT in shared memory + constraint eval
            return try evaluateQuotientFullyFused(
                traceColumns: traceColumns, system: system, alpha: alpha, logN: logN)
        } else {
            // Barrier path: NTT dispatches + constraint eval in one command buffer (no interleave)
            return try evaluateQuotientBarrierGeneral(
                traceColumns: traceColumns, system: system, alpha: alpha, logN: logN)
        }
    }

    /// Fully fused path: single kernel does NTT in shared memory + constraint eval.
    /// No global memory write of NTT output. This is the optimal path for small domains.
    private func evaluateQuotientFullyFused(
        traceColumns: [[Fr]],
        system: ConstraintSystem,
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        let numCols = traceColumns.count
        let stride = MemoryLayout<Fr>.stride

        // Get or compile the fused pipeline for this constraint system
        let pipeline = try getFusedGeneralPipeline(system: system)

        // Create column buffers
        var colBufs: [MTLBuffer] = []
        for col in traceColumns {
            guard let buf = device.makeBuffer(bytes: col, length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate column buffer")
            }
            colBufs.append(buf)
        }

        // Twiddle factors
        let twiddles = precomputeTwiddles(logN: logN)
        guard let twiddleBuf = device.makeBuffer(bytes: twiddles, length: twiddles.count * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate twiddle buffer")
        }

        // Alpha powers
        var alphaPowers = [Fr](repeating: Fr.zero, count: max(system.constraints.count, 1))
        alphaPowers[0] = Fr.one
        for i in 1..<alphaPowers.count {
            alphaPowers[i] = frMul(alphaPowers[i - 1], alpha)
        }
        let alphaBuf = getAlphaBuffer(alphaPowers)
        let vanishingInvBuf = getVanishingInvBuffer(n: n)

        guard let quotientBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate quotient buffer")
        }

        // Single dispatch
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)

        // Set column buffers
        for (i, buf) in colBufs.enumerated() {
            enc.setBuffer(buf, offset: 0, index: i)
        }
        enc.setBuffer(twiddleBuf, offset: 0, index: numCols)
        enc.setBuffer(quotientBuf, offset: 0, index: numCols + 1)
        enc.setBuffer(alphaBuf, offset: 0, index: numCols + 2)
        enc.setBuffer(vanishingInvBuf, offset: 0, index: numCols + 3)
        var nVal = UInt32(n)
        var logNVal = UInt32(logN)
        enc.setBytes(&nVal, length: 4, index: numCols + 4)
        enc.setBytes(&logNVal, length: 4, index: numCols + 5)

        let tgSize = n / 2  // each thread handles one butterfly pair
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Fused NTT+constraint GPU error: \(error.localizedDescription)")
        }

        let ptr = quotientBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Barrier path for larger sizes: NTT dispatches + constraint eval reading
    /// separate column buffers (no interleave step). Single command buffer, no host round-trip.
    private func evaluateQuotientBarrierGeneral(
        traceColumns: [[Fr]],
        system: ConstraintSystem,
        alpha: Fr = Fr.one,
        logN: Int
    ) throws -> [Fr] {
        let n = 1 << logN
        let numCols = traceColumns.count
        let stride = MemoryLayout<Fr>.stride

        // Create column buffers
        var colBufs: [MTLBuffer] = []
        for col in traceColumns {
            guard let buf = device.makeBuffer(bytes: col, length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate column buffer")
            }
            colBufs.append(buf)
        }

        // Alpha powers
        var alphaPowers = [Fr](repeating: Fr.zero, count: max(system.constraints.count, 1))
        alphaPowers[0] = Fr.one
        for i in 1..<alphaPowers.count {
            alphaPowers[i] = frMul(alphaPowers[i - 1], alpha)
        }
        let alphaBuf = getAlphaBuffer(alphaPowers)
        let vanishingInvBuf = getVanishingInvBuffer(n: n)

        guard let quotientBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate quotient buffer")
        }

        // Get or compile separate-column constraint eval pipeline
        let pipeline = try getSeparateColumnGeneralPipeline(system: system)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: NTT all columns (encoded into same command buffer)
        for buf in colBufs {
            nttEngine.encodeNTT(data: buf, logN: logN, cmdBuf: cmdBuf)
        }

        // Phase 2: Constraint evaluation reading from separate column buffers (no interleave)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(pipeline)
        for (i, buf) in colBufs.enumerated() {
            enc.setBuffer(buf, offset: 0, index: i)
        }
        enc.setBuffer(quotientBuf, offset: 0, index: numCols)
        enc.setBuffer(alphaBuf, offset: 0, index: numCols + 1)
        enc.setBuffer(vanishingInvBuf, offset: 0, index: numCols + 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: numCols + 3)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError("Barrier general constraint GPU error: \(error.localizedDescription)")
        }

        let ptr = quotientBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Runtime-compiled Pipeline Cache (general constraint systems)

    private var _fusedGeneralPipelineCache: [String: MTLComputePipelineState] = [:]
    private var _separateColGeneralPipelineCache: [String: MTLComputePipelineState] = [:]

    /// Cache key for a constraint system (based on structure, not instance identity)
    private func systemCacheKey(_ system: ConstraintSystem) -> String {
        return "w\(system.numWires)_c\(system.constraints.count)_n\(system.totalNodeCount)"
    }

    /// Get or compile a fully-fused NTT+constraint pipeline for the given constraint system.
    private func getFusedGeneralPipeline(system: ConstraintSystem) throws -> MTLComputePipelineState {
        let key = systemCacheKey(system)
        if let p = _fusedGeneralPipelineCache[key] { return p }

        let codegen = MetalCodegen()
        let kernelSource = codegen.generateFusedNTTConstraintEval(system: system)

        // Strip #include and using directives (we prepend Fr source)
        let cleanKernel = kernelSource.split(separator: "\n")
            .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
            .joined(separator: "\n")

        let shaderDir = findShaderDir()
        let rawFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = frSource + "\n" + cleanKernel
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        do {
            let library = try device.makeLibrary(source: combined, options: options)
            guard let fn = library.makeFunction(name: "fused_ntt_constraint_eval") else {
                throw MSMError.missingKernel
            }
            let pipeline = try device.makeComputePipelineState(function: fn)
            _fusedGeneralPipelineCache[key] = pipeline
            return pipeline
        } catch {
            throw MSMError.gpuError("Fused NTT+constraint compile error: \(error.localizedDescription)")
        }
    }

    /// Get or compile a separate-column constraint eval pipeline (no interleave needed).
    /// This reads each column from its own buffer rather than an interleaved trace.
    private func getSeparateColumnGeneralPipeline(system: ConstraintSystem) throws -> MTLComputePipelineState {
        let key = systemCacheKey(system)
        if let p = _separateColGeneralPipelineCache[key] { return p }

        let codegen = MetalCodegen()
        let kernelSource = codegen.generateSeparateColumnQuotientEval(system: system)

        let cleanKernel = kernelSource.split(separator: "\n")
            .filter { !$0.contains("#include") && !$0.contains("using namespace metal") }
            .joined(separator: "\n")

        let shaderDir = findShaderDir()
        let rawFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = frSource + "\n" + cleanKernel
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        do {
            let library = try device.makeLibrary(source: combined, options: options)
            guard let fn = library.makeFunction(name: "eval_quotient_separate_cols") else {
                throw MSMError.missingKernel
            }
            let pipeline = try device.makeComputePipelineState(function: fn)
            _separateColGeneralPipelineCache[key] = pipeline
            return pipeline
        } catch {
            throw MSMError.gpuError("Separate-column constraint compile error: \(error.localizedDescription)")
        }
    }

    // MARK: - Interleave Kernel

    private var _interleavePipeline: MTLComputePipelineState?
    private var _interleaveMaxCols: Int = 0

    private func encodeInterleave(cmdBuf: MTLCommandBuffer, colBufs: [MTLBuffer], traceBuf: MTLBuffer, n: Int, numCols: Int) throws {
        let pipeline = try getInterleavePipeline(numCols: numCols)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        for (i, buf) in colBufs.enumerated() {
            enc.setBuffer(buf, offset: 0, index: i)
        }
        enc.setBuffer(traceBuf, offset: 0, index: numCols)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: numCols + 1)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func getInterleavePipeline(numCols: Int) throws -> MTLComputePipelineState {
        if let p = _interleavePipeline, numCols <= _interleaveMaxCols { return p }

        let shaderDir = findShaderDir()
        let rawFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let frSource = rawFr
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        // Generate interleave kernel for up to numCols columns
        var bufParams = ""
        var copyLines = ""
        for i in 0..<numCols {
            bufParams += "    device const Fr* col\(i) [[buffer(\(i))]],\n"
            copyLines += "    output[row * \(numCols)u + \(i)u] = col\(i)[row];\n"
        }

        let kernelSource = """

        kernel void interleave_columns(
        \(bufParams)    device Fr* output [[buffer(\(numCols))]],
            constant uint& num_rows [[buffer(\(numCols + 1))]],
            uint gid [[thread_position_in_grid]]
        ) {
            uint row = gid;
            if (row >= num_rows) return;
        \(copyLines)}
        """

        let combined = frSource + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)
        guard let fn = library.makeFunction(name: "interleave_columns") else {
            throw MSMError.missingKernel
        }
        let pipeline = try device.makeComputePipelineState(function: fn)
        _interleavePipeline = pipeline
        _interleaveMaxCols = numCols
        return pipeline
    }

    // MARK: - Buffer Helpers

    private func getAlphaBuffer(_ powers: [Fr]) -> MTLBuffer {
        let stride = MemoryLayout<Fr>.stride
        let size = powers.count * stride
        if let buf = cachedAlphaBuf, buf.length >= size {
            powers.withUnsafeBytes { src in
                memcpy(buf.contents(), src.baseAddress!, size)
            }
            return buf
        }
        let buf = device.makeBuffer(bytes: powers, length: size, options: .storageModeShared)!
        cachedAlphaBuf = buf
        return buf
    }

    /// Get or create vanishing polynomial inverse buffer.
    /// For benchmarking purposes, fills with Fr.one (no actual vanishing polynomial division).
    /// In production, the caller would provide precomputed 1/Z_H(omega^i) values.
    private func getVanishingInvBuffer(n: Int) -> MTLBuffer {
        let stride = MemoryLayout<Fr>.stride
        let size = n * stride
        if cachedVanishingInvSize >= n, let buf = cachedVanishingInvBuf {
            return buf
        }
        var ones = [Fr](repeating: Fr.one, count: n)
        let buf = device.makeBuffer(bytes: &ones, length: size, options: .storageModeShared)!
        cachedVanishingInvBuf = buf
        cachedVanishingInvSize = n
        return buf
    }
}
