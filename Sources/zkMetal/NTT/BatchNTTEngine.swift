// BatchNTTEngine.swift
// GPU-accelerated batch NTT for BN254/Mersenne31 fields
// Single GPU dispatch processes all transforms in parallel using grid Y dimension
//
// Data Layout:
//   All transforms are stored sequentially in a single buffer:
//   [transform 0: N elements] [transform 1: N elements] ... [transform K-1: N elements]
//   Total buffer size = numTransforms * (1 << logN) * sizeof(Fr)
//
// API:
//   engine.encodeNTTBatch(buffer: gpuBuffer, numTransforms: K, logN: logN, cmdBuf: cmdBuf)
//   engine.encodeINTTBatch(buffer: gpuBuffer, numTransforms: K, logN: logN, cmdBuf: cmdBuf)

import Foundation
import Metal
import NeonFieldOps

public class BatchNTTEngine {
    public static let version = "1.0"
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Batch kernels
    private let bitrevBatchFunction: MTLComputePipelineState
    private let butterflyBatchFunction: MTLComputePipelineState
    private let butterflyRadix4BatchFunction: MTLComputePipelineState
    private let invButterflyBatchFunction: MTLComputePipelineState
    private let invButterflyRadix4BatchFunction: MTLComputePipelineState
    private let bitrevScaleBatchFunction: MTLComputePipelineState
    private let fusedBitrevBatchFunction: MTLComputePipelineState
    private let fusedInverseBatchFunction: MTLComputePipelineState

    // Max stages that can be fused in threadgroup memory
    private static let maxFusedLogN = 8  // 2^8 = 256 threads per threadgroup

    // Twiddle caches (reused from NTTEngine if available)
    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]

    // Tuning
    private let tuning: TuningConfig

    public init(nttEngine: NTTEngine? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let shaderDir = BatchNTTEngine.findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/ntt_kernels.metal", encoding: .utf8)

        // Clean up includes and header guards (like BatchCircleNTTEngine)
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let cleanNtt = nttSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanField + "\n" + cleanNtt
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: combined, options: options)

        guard let bitrevFn = library.makeFunction(name: "ntt_bitrev_batch"),
              let butterflyFn = library.makeFunction(name: "ntt_butterfly_batch"),
              let butterflyRadix4Fn = library.makeFunction(name: "ntt_butterfly_radix4_batch"),
              let invButterflyFn = library.makeFunction(name: "intt_butterfly_batch"),
              let invButterflyRadix4Fn = library.makeFunction(name: "intt_butterfly_radix4_batch"),
              let bitrevScaleFn = library.makeFunction(name: "ntt_bitrev_scale_batch"),
              let fusedBitrevFn = library.makeFunction(name: "ntt_fused_bitrev_batch"),
              let fusedInverseFn = library.makeFunction(name: "intt_fused_batch") else {
            throw MSMError.missingKernel
        }

        self.bitrevBatchFunction = try device.makeComputePipelineState(function: bitrevFn)
        self.butterflyBatchFunction = try device.makeComputePipelineState(function: butterflyFn)
        self.butterflyRadix4BatchFunction = try device.makeComputePipelineState(function: butterflyRadix4Fn)
        self.invButterflyBatchFunction = try device.makeComputePipelineState(function: invButterflyFn)
        self.invButterflyRadix4BatchFunction = try device.makeComputePipelineState(function: invButterflyRadix4Fn)
        self.bitrevScaleBatchFunction = try device.makeComputePipelineState(function: bitrevScaleFn)
        self.fusedBitrevBatchFunction = try device.makeComputePipelineState(function: fusedBitrevFn)
        self.fusedInverseBatchFunction = try device.makeComputePipelineState(function: fusedInverseFn)

        self.tuning = TuningManager.shared.config(device: device)

        // Warmup kernels to avoid JIT timeout
        try warmupKernels()
    }

    private func warmupKernels() throws {
        let warmupBuf = device.makeBuffer(length: 1024 * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let warmupTwiddlesBuf = device.makeBuffer(length: 512 * MemoryLayout<Fr>.stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { return }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var n: UInt32 = 1024
        var logN: UInt32 = 10
        var numTransforms: UInt32 = 1
        var stage: UInt32 = 0

        enc.setComputePipelineState(bitrevBatchFunction)
        enc.setBuffer(warmupBuf, offset: 0, index: 0)
        enc.setBytes(&n, length: 4, index: 1)
        enc.setBytes(&logN, length: 4, index: 2)
        enc.setBytes(&numTransforms, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/bn254_fr.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle Cache

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = precomputeTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = precomputeInverseTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let invNVal = frInverse(frFromInt(UInt64(1 << logN)))
        let buf = createBuffer([invNVal])!
        invNCache[logN] = buf
        return buf
    }

    private func createBuffer(_ data: [Fr]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Batch Forward NTT

    /// Encode batch forward NTT into an existing command buffer.
    /// Processes all transforms in a single GPU dispatch using grid Y dimension.
    ///
    /// - Parameters:
    ///   - buffer: Single buffer containing all transforms (sequential layout)
    ///   - numTransforms: Number of transforms to process
    ///   - logN: Log of transform size (each transform has 2^logN elements)
    ///   - cmdBuf: Existing command buffer (NOT committed by this function)
    public func encodeNTTBatch(buffer: MTLBuffer, numTransforms: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        var nVal = n
        var logNVal = UInt32(logN)
        var numK = UInt32(numTransforms)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        let tgSize = min(256, Int(bitrevBatchFunction.maxTotalThreadsPerThreadgroup))

        // Use fused kernel for better performance when logN > maxFusedLogN
        let fusedStages = min(logN, BatchNTTEngine.maxFusedLogN)
        let hasGlobal = fusedStages < logN

        if fusedStages > 1 {
            // Fused bitrev + DIT stages: single kernel launch handles both
            // Reads from input with bit-reversed indices, writes to output in sequential order
            enc.setComputePipelineState(fusedBitrevBatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)      // input
            enc.setBuffer(buffer, offset: 0, index: 1)     // output (same buffer, in-place after)
            enc.setBuffer(twiddles, offset: 0, index: 2)
            enc.setBytes(&nVal, length: 4, index: 3)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 4)
            enc.setBytes(&logNVal, length: 4, index: 5)
            enc.setBytes(&numK, length: 4, index: 6)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = (nInt >> fusedStages) * numTransforms
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        } else {
            // Small transform: separate bitrev kernel
            enc.setComputePipelineState(bitrevBatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBytes(&nVal, length: 4, index: 1)
            enc.setBytes(&logNVal, length: 4, index: 2)
            enc.setBytes(&numK, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: nInt, height: numTransforms, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Remaining stages: use radix-4 where possible (processes 2 stages at once)
        var stage: UInt32 = UInt32(fusedStages)
        while stage + 1 < UInt32(logN) {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(butterflyRadix4BatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numK, length: 4, index: 4)
            let numQuads = nInt / 4
            enc.dispatchThreads(MTLSize(width: numQuads, height: numTransforms, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            stage += 2
        }

        // Final stage if odd number of stages remaining
        if stage < UInt32(logN) {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(butterflyBatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numK, length: 4, index: 4)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: numTransforms, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        enc.endEncoding()
    }

    /// Encode batch inverse NTT into an existing command buffer.
    /// Processes all transforms in a single GPU dispatch using grid Y dimension.
    ///
    /// - Parameters:
    ///   - buffer: Single buffer containing all transforms (sequential layout)
    ///   - numTransforms: Number of transforms to process
    ///   - logN: Log of transform size (each transform has 2^logN elements)
    ///   - cmdBuf: Existing command buffer (NOT committed by this function)
    public func encodeINTTBatch(buffer: MTLBuffer, numTransforms: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        var nVal = n
        var logNVal = UInt32(logN)
        var numK = UInt32(numTransforms)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        let tgSize = min(256, Int(invButterflyBatchFunction.maxTotalThreadsPerThreadgroup))

        // Inverse stages: DIF (Gentleman-Sande) from high to low
        // Use radix-4 when we have stages s and s-1 to process (s >= 2)
        // Process stage pairs (s, s-1) together until s > 1
        var stage: UInt32 = UInt32(logN) - 1

        while stage > 1 {
            enc.memoryBarrier(scope: .buffers)
            // Use radix-4 to process 2 stages at once (stage and stage-1)
            enc.setComputePipelineState(invButterflyRadix4BatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numK, length: 4, index: 4)
            let numQuads = nInt / 4
            enc.dispatchThreads(MTLSize(width: numQuads, height: numTransforms, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            stage -= 2
        }

        // Handle final stage 1 and/or stage 0
        // Use radix-4 to process stages 1 and 0 together (they share twiddle patterns)
        if stage == 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(invButterflyRadix4BatchFunction)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal: UInt32 = 1
            enc.setBytes(&stageVal, length: 4, index: 3)
            enc.setBytes(&numK, length: 4, index: 4)
            let numQuads = nInt / 4
            enc.dispatchThreads(MTLSize(width: numQuads, height: numTransforms, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Final: bit-reversal + scale
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(bitrevScaleBatchFunction)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&logNVal, length: 4, index: 3)
        enc.setBytes(&numK, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: nInt, height: numTransforms, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

        enc.endEncoding()
    }

    // MARK: - High-level API

    /// Execute batch forward NTT on a single buffer.
    /// Buffer layout: [transform0: N][transform1: N]...[transformK-1: N]
    public func ntt(buffer: MTLBuffer, numTransforms: Int, logN: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        encodeNTTBatch(buffer: buffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Execute batch inverse NTT on a single buffer.
    /// Buffer layout: [transform0: N][transform1: N]...[transformK-1: N]
    public func intt(buffer: MTLBuffer, numTransforms: Int, logN: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        encodeINTTBatch(buffer: buffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Async API

    /// Shared event for async synchronization
    private var sharedEvent: MTLSharedEvent?
    private var nextEventValue: UInt64 = 1

    private func getSharedEvent() -> MTLSharedEvent? {
        if sharedEvent == nil {
            sharedEvent = device.makeSharedEvent()
        }
        return sharedEvent
    }

    /// Execute batch forward NTT asynchronously, calling completion when done.
    /// Uses MTLSharedEvent for GPU-CPU synchronization instead of blocking.
    public func nttAsync(buffer: MTLBuffer, numTransforms: Int, logN: Int, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            completion(.failure(MSMError.noCommandBuffer))
            return
        }

        guard let event = getSharedEvent() else {
            completion(.failure(MSMError.gpuError("Failed to create shared event")))
            return
        }

        let currentEventValue = nextEventValue
        nextEventValue += 1

        encodeNTTBatch(buffer: buffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.encodeSignalEvent(event, value: currentEventValue)
        cmdBuf.commit()

        DispatchQueue.global(qos: .userInitiated).async {
            let timeout = 10.0
            let start = Date()

            while event.signaledValue < currentEventValue {
                usleep(1000)
                if Date().timeIntervalSince(start) > timeout {
                    completion(.failure(MSMError.gpuError("Batch NTT timeout")))
                    return
                }
            }

            completion(.success(()))
        }
    }

    /// Execute batch inverse NTT asynchronously, calling completion when done.
    /// Uses MTLSharedEvent for GPU-CPU synchronization instead of blocking.
    public func inttAsync(buffer: MTLBuffer, numTransforms: Int, logN: Int, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            completion(.failure(MSMError.noCommandBuffer))
            return
        }

        guard let event = getSharedEvent() else {
            completion(.failure(MSMError.gpuError("Failed to create shared event")))
            return
        }

        let currentEventValue = nextEventValue
        nextEventValue += 1

        encodeINTTBatch(buffer: buffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.encodeSignalEvent(event, value: currentEventValue)
        cmdBuf.commit()

        DispatchQueue.global(qos: .userInitiated).async {
            let timeout = 10.0
            let start = Date()

            while event.signaledValue < currentEventValue {
                usleep(1000)
                if Date().timeIntervalSince(start) > timeout {
                    completion(.failure(MSMError.gpuError("Batch iNTT timeout")))
                    return
                }
            }

            completion(.success(()))
        }
    }

    /// Batch multiple NTT operations into a single GPU dispatch.
    /// All operations are encoded into a single command buffer and executed in parallel.
    /// This reduces per-operation kernel launch overhead.
    public func nttBatch(operations: [(buffer: MTLBuffer, numTransforms: Int, logN: Int)], completion: @escaping (Result<Void, Error>) -> Void) {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            completion(.failure(MSMError.noCommandBuffer))
            return
        }

        guard let event = getSharedEvent() else {
            completion(.failure(MSMError.gpuError("Failed to create shared event")))
            return
        }

        let currentEventValue = nextEventValue
        nextEventValue += 1

        for op in operations {
            encodeNTTBatch(buffer: op.buffer, numTransforms: op.numTransforms, logN: op.logN, cmdBuf: cmdBuf)
        }

        cmdBuf.encodeSignalEvent(event, value: currentEventValue)
        cmdBuf.commit()

        DispatchQueue.global(qos: .userInitiated).async {
            let timeout = 30.0
            let start = Date()

            while event.signaledValue < currentEventValue {
                usleep(1000)
                if Date().timeIntervalSince(start) > timeout {
                    completion(.failure(MSMError.gpuError("Batch NTT timeout")))
                    return
                }
            }

            completion(.success(()))
        }
    }
}
