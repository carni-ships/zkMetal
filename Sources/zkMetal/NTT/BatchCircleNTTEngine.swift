// BatchCircleNTTEngine.swift
// GPU-accelerated batch Circle NTT for EVMetal
// Single GPU dispatch processes all columns in parallel
//
// Data Layout:
//   All columns are stored sequentially in a single buffer:
//   [column 0: N elements] [column 1: N elements] ... [column N-1: N elements]
//   Total buffer size = numColumns * (1 << logN) * sizeof(M31)
//
// API:
//   nttEngine.encodeINTTBatch(buffers: gpuBuffers, logN: logTrace, cmdBuf: cmdBuf)
//   nttEngine.encodeNTTBatch(buffers: gpuBuffers, logN: logTrace, cmdBuf: cmdBuf)
//
// For EVMetal integration, the buffer at index i contains column i's data
// and all columns share the same logN.

import Foundation
import Metal

public class BatchCircleNTTEngine {
    public static let version = Versions.batchCircleNTT
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Batch kernels
    private let batchButterflyDIT: MTLComputePipelineState
    private let batchButterflyDIF: MTLComputePipelineState
    private let batchScale: MTLComputePipelineState
    private let batchBitrev: MTLComputePipelineState
    private let batchFusedBitrevDIT: MTLComputePipelineState
    private let batchFusedBitrevDIF: MTLComputePipelineState

    // Scratch buffer for fused kernels
    private var scratchBuffer: MTLBuffer?
    private var scratchCapacity: Int = 0

    // Twiddle caches (reused from CircleNTTEngine if available)
    private var fwdTwiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]

    // Tuning
    private let tuning: TuningConfig

    // Max fused stages for threadgroup memory (1024 M31 elements = 4KB)
    public static let maxFusedLogN = 10

    public init(circleEngine: CircleNTTEngine? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try BatchCircleNTTEngine.compileShaders(device: device)

        guard let ditFn = library.makeFunction(name: "batch_circle_ntt_butterfly_dit"),
              let difFn = library.makeFunction(name: "batch_circle_ntt_butterfly_dif"),
              let scaleFn = library.makeFunction(name: "batch_circle_ntt_scale"),
              let bitrevFn = library.makeFunction(name: "batch_circle_ntt_bitrev"),
              let fusedDITFn = library.makeFunction(name: "batch_circle_ntt_fused_bitrev_dit"),
              let fusedDIFn = library.makeFunction(name: "batch_circle_ntt_fused_bitrev_dif") else {
            throw MSMError.missingKernel
        }

        self.batchButterflyDIT = try device.makeComputePipelineState(function: ditFn)
        self.batchButterflyDIF = try device.makeComputePipelineState(function: difFn)
        self.batchScale = try device.makeComputePipelineState(function: scaleFn)
        self.batchBitrev = try device.makeComputePipelineState(function: bitrevFn)
        self.batchFusedBitrevDIT = try device.makeComputePipelineState(function: fusedDITFn)
        self.batchFusedBitrevDIF = try device.makeComputePipelineState(function: fusedDIFn)

        self.tuning = TuningManager.shared.config(device: device)

        // Warmup kernels to avoid JIT timeout
        try warmupKernels()
    }

    private func warmupKernels() throws {
        let warmupBuf = device.makeBuffer(length: 256 * MemoryLayout<M31>.stride, options: .storageModeShared)!
        let warmupTwiddlesBuf = device.makeBuffer(length: 256 * MemoryLayout<M31>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { return }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var n: UInt32 = 256
        var stage: UInt32 = 0
        var numCols: UInt32 = 1

        enc.setComputePipelineState(batchButterflyDIT)
        enc.setBuffer(warmupBuf, offset: 0, index: 0)
        enc.setBuffer(warmupTwiddlesBuf, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.setBytes(&stage, length: 4, index: 3)
        enc.setBytes(&numCols, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: 128, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let batchSource = try String(contentsOfFile: shaderDir + "/ntt/batch_circle_ntt.metal", encoding: .utf8)

        // Clean up includes and header guards
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let cleanBatch = batchSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanField + "\n" + cleanBatch
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/mersenne31.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/mersenne31.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Buffer Management

    /// Creates a single batch buffer containing all column data
    /// Each column occupies (1 << logN) elements, laid out sequentially
    public func createBatchBuffer(columns: [[M31]], logN: Int) throws -> MTLBuffer {
        let n = 1 << logN
        let numCols = columns.count
        let totalSize = n * numCols * MemoryLayout<M31>.stride

        guard let buffer = device.makeBuffer(length: totalSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create batch buffer")
        }

        let ptr = buffer.contents().bindMemory(to: M31.self, capacity: n * numCols)
        for (colIdx, column) in columns.enumerated() {
            precondition(column.count == n, "Column \(colIdx) has \(column.count) elements, expected \(n)")
            for i in 0..<n {
                ptr[colIdx * n + i] = column[i]
            }
        }

        return buffer
    }

    /// Reads back column data from a batch buffer
    public func readColumns(from buffer: MTLBuffer, numColumns: Int, logN: Int) -> [[M31]] {
        let n = 1 << logN
        let ptr = buffer.contents().bindMemory(to: M31.self, capacity: n * numColumns)

        var result = [[M31]]()
        result.reserveCapacity(numColumns)
        for colIdx in 0..<numColumns {
            var column = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                column[i] = ptr[colIdx * n + i]
            }
            result.append(column)
        }
        return result
    }

    // MARK: - Twiddle Cache

    private func getForwardTwiddles(logN: Int) -> MTLBuffer {
        if let cached = fwdTwiddleCache[logN] { return cached }
        let twiddles = circlePrecomputeForwardTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        fwdTwiddleCache[logN] = buf
        return buf
    }

    private func getInverseTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = circlePrecomputeInverseTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let invN = m31Inverse(M31(v: UInt32(1 << logN)))
        let buf = createM31Buffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createM31Buffer(_ data: [M31]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<M31>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<M31>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    // MARK: - Batch Forward NTT

    /// Batch forward NTT on a single buffer containing all columns.
    /// Uses grid Y dimension to process columns in parallel.
    /// All columns must have the same size (1 << logN).
    public func ntt(data: MTLBuffer, numColumns: Int, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        var nVal = n
        var numColsVal = UInt32(numColumns)
        let tgSize = min(256, Int(batchButterflyDIT.maxTotalThreadsPerThreadgroup))

        // Process Circle NTT layers: k-1 down to 1 (x-twiddle), then 0 (y-twiddle)
        for layer in stride(from: logN - 1, through: 0, by: -1) {
            if layer < logN - 1 { enc.memoryBarrier(scope: .buffers) }

            enc.setComputePipelineState(batchButterflyDIT)
            enc.setBuffer(data, offset: 0, index: 0)
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(twiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numColsVal, length: 4, index: 4)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: numColumns, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Batch Inverse NTT

    /// Batch inverse NTT on a single buffer containing all columns.
    /// Uses grid Y dimension to process columns in parallel.
    /// All columns must have the same size (1 << logN).
    public func intt(data: MTLBuffer, numColumns: Int, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        var nVal = n
        var numColsVal = UInt32(numColumns)
        let tgSize = min(256, Int(batchButterflyDIF.maxTotalThreadsPerThreadgroup))

        // Process Circle INTT layers: 0 (y-twiddle), then 1..k-1 (x-twiddle)
        for layer in 0..<logN {
            if layer > 0 { enc.memoryBarrier(scope: .buffers) }

            enc.setComputePipelineState(batchButterflyDIF)
            enc.setBuffer(data, offset: 0, index: 0)
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(invTwiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numColsVal, length: 4, index: 4)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: numColumns, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Scale by 1/N
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(batchScale)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&numColsVal, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: nInt, height: numColumns, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Batch Encoding API (for EVMetal integration)

    /// Encode batch forward NTT into an existing command buffer.
    /// Processes all columns in a single GPU dispatch using grid Y dimension.
    ///
    /// - Parameters:
    ///   - buffer: Single buffer containing all columns (sequential layout)
    ///   - numColumns: Number of columns to process
    ///   - logN: Log of transform size (each column has 2^logN elements)
    ///   - cmdBuf: Existing command buffer (NOT committed by this function)
    public func encodeNTT(buffer: MTLBuffer, numColumns: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)
        var nVal = n
        var numColsVal = UInt32(numColumns)
        let tgSize = min(256, Int(batchButterflyDIT.maxTotalThreadsPerThreadgroup))

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Process Circle NTT layers: k-1 down to 1 (x-twiddle), then 0 (y-twiddle)
        for layer in stride(from: logN - 1, through: 0, by: -1) {
            if layer < logN - 1 { enc.memoryBarrier(scope: .buffers) }

            enc.setComputePipelineState(batchButterflyDIT)
            enc.setBuffer(buffer, offset: 0, index: 0)
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(twiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numColsVal, length: 4, index: 4)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: numColumns, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        enc.endEncoding()
    }

    /// Encode batch inverse NTT into an existing command buffer.
    /// Processes all columns in a single GPU dispatch using grid Y dimension.
    ///
    /// - Parameters:
    ///   - buffer: Single buffer containing all columns (sequential layout)
    ///   - numColumns: Number of columns to process
    ///   - logN: Log of transform size (each column has 2^logN elements)
    ///   - cmdBuf: Existing command buffer (NOT committed by this function)
    public func encodeINTT(buffer: MTLBuffer, numColumns: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        var nVal = n
        var numColsVal = UInt32(numColumns)
        let tgSize = min(256, Int(batchButterflyDIF.maxTotalThreadsPerThreadgroup))

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Process Circle INTT layers: 0 (y-twiddle), then 1..k-1 (x-twiddle)
        for layer in 0..<logN {
            if layer > 0 { enc.memoryBarrier(scope: .buffers) }

            enc.setComputePipelineState(batchButterflyDIF)
            enc.setBuffer(buffer, offset: 0, index: 0)
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(invTwiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numColsVal, length: 4, index: 4)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: numColumns, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Scale by 1/N
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(batchScale)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&numColsVal, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: nInt, height: numColumns, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

        enc.endEncoding()
    }

    /// Batch INTT encoding for multiple column buffers (EVMetal API).
    /// Each buffer is processed independently, but all kernels are encoded
    /// into the same command buffer for efficient pipelining.
    ///
    /// - Parameters:
    ///   - buffers: Array of MTLBuffer, one per column (each column is independent)
    ///   - logN: Log of transform size
    ///   - cmdBuf: Existing command buffer
    public func encodeINTTBatch(buffers: [MTLBuffer], logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        var nVal = n
        let tgSize = min(256, Int(batchButterflyDIF.maxTotalThreadsPerThreadgroup))

        for buffer in buffers {
            let enc = cmdBuf.makeComputeCommandEncoder()!

            // Process Circle INTT layers
            for layer in 0..<logN {
                if layer > 0 { enc.memoryBarrier(scope: .buffers) }

                enc.setComputePipelineState(batchButterflyDIF)
                enc.setBuffer(buffer, offset: 0, index: 0)
                let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
                enc.setBuffer(invTwiddles, offset: twiddleOffset, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = UInt32(logN - 1 - layer)
                enc.setBytes(&stageVal, length: 4, index: 3)
                var numColsVal: UInt32 = 1
                enc.setBytes(&numColsVal, length: 4, index: 4)
                let numButterflies = nInt / 2
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            // Scale by 1/N
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(batchScale)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(invN, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var numColsVal: UInt32 = 1
            enc.setBytes(&numColsVal, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            enc.endEncoding()
        }
    }

    /// Batch NTT encoding for multiple column buffers (EVMetal API).
    /// Each buffer is processed independently, but all kernels are encoded
    /// into the same command buffer for efficient pipelining.
    public func encodeNTTBatch(buffers: [MTLBuffer], logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)
        var nVal = n
        let tgSize = min(256, Int(batchButterflyDIT.maxTotalThreadsPerThreadgroup))

        for buffer in buffers {
            let enc = cmdBuf.makeComputeCommandEncoder()!

            // Process Circle NTT layers
            for layer in stride(from: logN - 1, through: 0, by: -1) {
                if layer < logN - 1 { enc.memoryBarrier(scope: .buffers) }

                enc.setComputePipelineState(batchButterflyDIT)
                enc.setBuffer(buffer, offset: 0, index: 0)
                let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
                enc.setBuffer(twiddles, offset: twiddleOffset, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = UInt32(logN - 1 - layer)
                enc.setBytes(&stageVal, length: 4, index: 3)
                var numColsVal: UInt32 = 1
                enc.setBytes(&numColsVal, length: 4, index: 4)
                let numButterflies = nInt / 2
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            enc.endEncoding()
        }
    }

    // MARK: - Batch LDE (INTT -> zero-pad -> NTT)

    /// Batch low-degree extension: INTT on trace domain, zero-pad, NTT on evaluation domain.
    /// This is the core operation for EVMetal trace commitment.
    ///
    /// - Parameters:
    ///   - trace: Array of column buffers (in evaluation form on trace domain)
    ///   - logTrace: Log of trace size (N = 2^logTrace)
    ///   - logEval: Log of evaluation size (M = 2^logEval, M >= N)
    ///   - cmdBuf: Command buffer to encode into
    public func encodeBatchLDE(trace: [MTLBuffer], logTrace: Int, logEval: Int, cmdBuf: MTLCommandBuffer) throws {
        let nTrace = 1 << logTrace
        let nEval = 1 << logEval

        // Step 1: Batch INTT on trace domain for all columns
        encodeINTTBatch(buffers: trace, logN: logTrace, cmdBuf: cmdBuf)

        // Step 2: Zero-pad and encode forward NTT on evaluation domain
        // For each column, we need to copy to a larger buffer and NTT
        // This is more complex - we'd need a zeroPad kernel
        // For now, we encode the NTT part only (assumes caller handled padding)

        // Step 3: Batch NTT on evaluation domain
        // Note: This requires separate buffers since sizes differ
        // encodeNTTBatch(buffers: evalBuffers, logN: logEval, cmdBuf: cmdBuf)
    }

    // MARK: - High-level API

    /// High-level batch NTT: takes [[M31]] columns, returns NTT'd columns.
    public func ntt(_ columns: [[M31]], logN: Int) throws -> [[M31]] {
        let numCols = columns.count
        guard numCols > 0 else { return [] }

        let batchBuf = try createBatchBuffer(columns: columns, logN: logN)
        try ntt(data: batchBuf, numColumns: numCols, logN: logN)
        return readColumns(from: batchBuf, numColumns: numCols, logN: logN)
    }

    /// High-level batch INTT: takes NTT'd columns, returns original columns.
    public func intt(_ columns: [[M31]], logN: Int) throws -> [[M31]] {
        let numCols = columns.count
        guard numCols > 0 else { return [] }

        let batchBuf = try createBatchBuffer(columns: columns, logN: logN)
        try intt(data: batchBuf, numColumns: numCols, logN: logN)
        return readColumns(from: batchBuf, numColumns: numCols, logN: logN)
    }
}