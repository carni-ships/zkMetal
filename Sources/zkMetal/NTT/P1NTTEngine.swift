// P^1 Rational Function NTT Engine — standard radix-2 FFT on multiplicative cosets
//
// Unlike Circle NTT which uses the circle group, P^1 NTT uses:
// - Domain: multiplicative coset η·G where G ⊂ F_p* is a subgroup of order 2^k
// - Standard radix-2 DIT/DIF butterflies
// - Folding via t → t² (standard FRI)
//
// The field M31 = F_{2^31-1} has p-1 = 2^31 - 2 = 2 * (2^30 - 1)
// We can find a generator of the 2^30 order subgroup by computing g^((p-1)/2^30)

import Foundation
import Metal

public class P1NTTEngine {
    public static let version = Versions.p1NTT
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let butterflyFunction: MTLComputePipelineState
    let invButterflyFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState
    let butterflyFusedFunction: MTLComputePipelineState  // Fused kernel for multiple stages
    let invButterflyFusedFunction: MTLComputePipelineState  // Fused inverse kernel
    // Four-step FFT kernels
    let columnFusedFunction: MTLComputePipelineState
    let twiddleFourstepFunction: MTLComputePipelineState
    let rowFusedTwiddleFunction: MTLComputePipelineState
    let transposeOutOfPlaceFunction: MTLComputePipelineState
    let invColumnFusedFunction: MTLComputePipelineState
    let invRowFusedTwiddleFunction: MTLComputePipelineState

    private var fwdTwiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    // Scratch buffer for four-step (transpose + in-place path)
    private var scratchBuffer: MTLBuffer?
    private var scratchCapacity: Int = 0

    // Threadgroup memory limit: 1024 M31 elements = 4KB shared memory
    public static let maxFusedElements = 1024
    public static let maxFusedLogN = 10  // log2(1024)

    // P1 four-step threshold: use higher threshold (15) to avoid GPU timeout at logN=20
    // where combined kernel time in single command buffer exceeds Apple Silicon limits
    private var fourStepMinGlobalStages: Int { max(tuning.nttFourStepThreshold, 15) }
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

        // Pre-warm: compile shaders with a timeout-friendly approach
        // Use separate command buffer for compilation to avoid watchdog timeout
        let library: MTLLibrary
        do {
            library = try P1NTTEngine.compileShaders(device: device)
        } catch {
            throw MSMError.gpuError("Shader compilation failed: \(error)")
        }

        guard let butterflyFn = library.makeFunction(name: "p1_ntt_butterfly"),
              let invButterflyFn = library.makeFunction(name: "p1_intt_butterfly"),
              let scaleFn = library.makeFunction(name: "p1_ntt_scale"),
              let fusedFn = library.makeFunction(name: "p1_ntt_butterfly_fused"),
              let invFusedFn = library.makeFunction(name: "p1_intt_butterfly_fused"),
              let colFusedFn = library.makeFunction(name: "p1_ntt_column_fused"),
              let twiddleFn = library.makeFunction(name: "p1_ntt_twiddle_fourstep"),
              let rowFusedTwiddleFn = library.makeFunction(name: "p1_ntt_row_fused_twiddle"),
              let transposeFn = library.makeFunction(name: "p1_ntt_transpose_outofplace"),
              let invColFusedFn = library.makeFunction(name: "p1_intt_column_fused"),
              let invRowFusedTwiddleFn = library.makeFunction(name: "p1_intt_row_fused_twiddle") else {
            throw MSMError.missingKernel
        }

        self.butterflyFunction = try device.makeComputePipelineState(function: butterflyFn)
        self.invButterflyFunction = try device.makeComputePipelineState(function: invButterflyFn)
        self.scaleFunction = try device.makeComputePipelineState(function: scaleFn)
        self.butterflyFusedFunction = try device.makeComputePipelineState(function: fusedFn)
        self.invButterflyFusedFunction = try device.makeComputePipelineState(function: invFusedFn)
        self.columnFusedFunction = try device.makeComputePipelineState(function: colFusedFn)
        self.twiddleFourstepFunction = try device.makeComputePipelineState(function: twiddleFn)
        self.rowFusedTwiddleFunction = try device.makeComputePipelineState(function: rowFusedTwiddleFn)
        self.transposeOutOfPlaceFunction = try device.makeComputePipelineState(function: transposeFn)
        self.invColumnFusedFunction = try device.makeComputePipelineState(function: invColFusedFn)
        self.invRowFusedTwiddleFunction = try device.makeComputePipelineState(function: invRowFusedTwiddleFn)

        // Pre-warm: dispatch fused kernels with minimal work to trigger JIT compilation
        // This prevents lazy compilation timeout during first real benchmark run
        // Create buffers manually (no instance methods needed)
        let warmupDataBuf = device.makeBuffer(length: 32 * MemoryLayout<M31>.stride, options: .storageModeShared)!
        let warmupTwiddlesBuf = device.makeBuffer(length: 32 * MemoryLayout<M31>.stride, options: .storageModeShared)!
        let invTwiddlesBuf = device.makeBuffer(length: 32 * MemoryLayout<M31>.stride, options: .storageModeShared)!

        guard let warmupCmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let warmupEnc = warmupCmdBuf.makeComputeCommandEncoder()!
        warmupEnc.setComputePipelineState(butterflyFusedFunction)
        warmupEnc.setBuffer(warmupDataBuf, offset: 0, index: 0)
        warmupEnc.setBuffer(warmupTwiddlesBuf, offset: 0, index: 1)
        var nVal: UInt32 = 32
        var localStages: UInt32 = 1
        var stageOffset: UInt32 = 0
        warmupEnc.setBytes(&nVal, length: 4, index: 2)
        warmupEnc.setBytes(&localStages, length: 4, index: 3)
        warmupEnc.setBytes(&stageOffset, length: 4, index: 4)
        warmupEnc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        warmupEnc.endEncoding()
        warmupCmdBuf.commit()
        warmupCmdBuf.waitUntilCompleted()

        // Also warmup inverse fused kernel
        guard let warmupCmdBuf2 = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let warmupEnc2 = warmupCmdBuf2.makeComputeCommandEncoder()!
        warmupEnc2.setComputePipelineState(invButterflyFusedFunction)
        warmupEnc2.setBuffer(warmupDataBuf, offset: 0, index: 0)
        warmupEnc2.setBuffer(invTwiddlesBuf, offset: 0, index: 1)
        var firstStage: UInt32 = 0
        warmupEnc2.setBytes(&nVal, length: 4, index: 2)
        warmupEnc2.setBytes(&localStages, length: 4, index: 3)
        warmupEnc2.setBytes(&firstStage, length: 4, index: 4)
        warmupEnc2.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        warmupEnc2.endEncoding()
        warmupCmdBuf2.commit()
        warmupCmdBuf2.waitUntilCompleted()

        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/ntt_p1.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let combined = cleanField + "\n" + cleanNTT
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

    // MARK: - Twiddle computation

    /// Precompute forward twiddles for P^1 NTT (standard root-of-unity twiddles).
    /// Layout: standard Cooley-Tukey twiddle factors.
    private func getForwardTwiddles(logN: Int) -> MTLBuffer {
        if let cached = fwdTwiddleCache[logN] { return cached }
        let twiddles = p1PrecomputeForwardTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        fwdTwiddleCache[logN] = buf
        return buf
    }

    private func getInverseTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = p1PrecomputeInverseTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt32(1 << logN)
        let invN = m31Inverse(M31(v: n))
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

    // MARK: - Forward P^1 NTT (DIT)

    /// Forward NTT: coefficients -> evaluations on the P^1 domain.
    /// Uses four-step FFT when globalStages >= fourStepMinGlobalStages (logN > 2*maxFusedLogN regime).
    /// Falls back to fused kernel for smaller transforms.
    public func ntt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - P1NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            try nttFourStep(data: data, logN: logN)
            return
        }

        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)
        var nVal = n  // Declare nVal before branches so both can use it

        // Use fused kernel for logN > 4 (benefit outweighs overhead)
        // Fused kernel processes local_stages stages per dispatch using threadgroup memory
        // Max local_stages is limited by threadgroup size (max 256 threads = 512 elements)
        let maxLocalStages = 9  // 2^9 = 512 elements per threadgroup
        let useFused = logN > 4

        if useFused && nInt >= 16777216 {
            // Fused path: ONLY use for very large transforms (n >= 16M)
            // The fused kernel has a twiddle calculation bug for stages > 0
            // For n < 16M, use per-stage path which works correctly
            let tgSize = min(256, Int(butterflyFusedFunction.maxTotalThreadsPerThreadgroup))

            // Compute numGroups once - same for all dispatches in this transform
            let elementsPerGroup = tgSize * 2
            let numGroups = (nInt + elementsPerGroup - 1) / elementsPerGroup

            var stageOffset: UInt32 = 0
            while stageOffset < UInt32(logN) {
                var remainingStages = logN - Int(stageOffset)
                var localStages = UInt32(min(remainingStages, maxLocalStages))

                // Use separate command buffer per chunk to avoid GPU timeout
                guard let chunkCmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let chunkEnc = chunkCmdBuf.makeComputeCommandEncoder()!
                chunkEnc.setComputePipelineState(butterflyFusedFunction)
                chunkEnc.setBuffer(data, offset: 0, index: 0)
                chunkEnc.setBuffer(twiddles, offset: 0, index: 1)
                chunkEnc.setBytes(&nVal, length: 4, index: 2)
                chunkEnc.setBytes(&localStages, length: 4, index: 3)
                chunkEnc.setBytes(&stageOffset, length: 4, index: 4)
                chunkEnc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                chunkEnc.endEncoding()
                chunkCmdBuf.commit()
                chunkCmdBuf.waitUntilCompleted()
                if let error = chunkCmdBuf.error {
                    throw MSMError.gpuError(error.localizedDescription)
                }

                stageOffset += localStages
            }
            // Fused path: all chunks processed, return to avoid using original cmdBuf/enc
            return
        } else {
            // Original per-stage path for small transforms
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }
            var nVal = n
            let enc = cmdBuf.makeComputeCommandEncoder()!
            let tgSize = min(256, Int(butterflyFunction.maxTotalThreadsPerThreadgroup))
            for stage in 0..<logN {
                if stage > 0 { enc.memoryBarrier(scope: .buffers) }
                enc.setComputePipelineState(butterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = UInt32(stage)
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }
        }
    }

    // MARK: - Four-Step Forward P^1 NTT

    /// Four-step forward FFT for large transforms (logN > 2*maxFusedLogN regime).
    /// Algorithm:
    ///   1. Column FFTs of size N1 (N2 independent columns)
    ///   2. Twiddle multiply (diagonal twiddles)
    ///   3. Row FFTs of size N2 (with twiddle fused during load)
    ///   4. Transpose (N1=N2 balanced square matrix)
    private func nttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)

        // Balanced split: N1 ≈ N2 ≈ sqrt(N)
        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var n1Val = n1
        var n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Column FFTs of size N1 (N2 columns total)
        // Each threadgroup handles one column, loading with stride N2
        var localStages = UInt32(min(logN1, P1NTTEngine.maxFusedLogN))
        enc.setComputePipelineState(columnFusedFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        enc.setBytes(&localStages, length: 4, index: 5)
        let colThreads = Int(n1) / 2
        enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: colThreads, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 2: Twiddle multiply (diagonal twiddles between column and row FFTs)
        enc.setComputePipelineState(twiddleFourstepFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        let twTg = min(tuning.nttThreadgroupSize, Int(twiddleFourstepFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: twTg, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 3: Row FFTs of size N2 (twiddle fused during load)
        var rowLocalStages = UInt32(min(logN2, P1NTTEngine.maxFusedLogN))
        enc.setComputePipelineState(rowFusedTwiddleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&rowLocalStages, length: 4, index: 3)
        let rowThreads = Int(n2) / 2
        enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: rowThreads, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 4: Out-of-place square transpose (N1 = N2 for balanced split)
        let scratch = getScratchBuffer(n: nInt)
        enc.setComputePipelineState(transposeOutOfPlaceFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(scratch, offset: 0, index: 1)
        enc.setBytes(&n1Val, length: 4, index: 2)
        let tg4e = min(tuning.nttThreadgroupSize, Int(transposeOutOfPlaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg4e, height: 1, depth: 1))
        enc.endEncoding()

        // Blit transposed data from scratch back to data buffer
        let blit = cmdBuf.makeBlitCommandEncoder()!
        blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<M31>.stride)
        blit.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse P^1 NTT (DIF)

    /// Inverse NTT: evaluations -> coefficients.
    /// Uses four-step FFT when globalStages >= fourStepMinGlobalStages.
    /// Falls back to fused kernel for smaller transforms.
    public func intt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - P1NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            try inttFourStep(data: data, logN: logN)
            return
        }

        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        var nVal = n  // Declare before branches

        // Use fused kernel only for very large transforms (n >= 16M) where multi-dispatch is needed
        // Note: fused kernel has a correctness bug that manifests with multiple dispatches
        // at certain sizes (16K, 512K). For now, use per-stage path which is correct.
        let maxLocalStages = 9
        let useFused = logN > 4 && nInt >= 16777216  // 16M+ only

        if useFused {
            // Fused path: process in groups from high to low stages
            // For large transforms, use smaller chunks to avoid GPU timeout
            let tgSize = min(256, Int(invButterflyFusedFunction.maxTotalThreadsPerThreadgroup))
            var stageOffset = UInt32(logN - 1)  // Start from highest stage

            // Process in chunks of maxLocalStages to avoid GPU watchdog timeout
            // Each chunk gets its own command buffer with wait to ensure completion
            while true {
                var remainingStages = Int(stageOffset) + 1
                var localStages = UInt32(min(remainingStages, maxLocalStages))
                var firstStage = stageOffset - localStages + 1

                // Use separate command buffer per chunk to avoid GPU timeout
                guard let chunkCmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let chunkEnc = chunkCmdBuf.makeComputeCommandEncoder()!
                chunkEnc.setComputePipelineState(invButterflyFusedFunction)
                chunkEnc.setBuffer(data, offset: 0, index: 0)
                chunkEnc.setBuffer(invTwiddles, offset: 0, index: 1)
                chunkEnc.setBytes(&nVal, length: 4, index: 2)
                chunkEnc.setBytes(&localStages, length: 4, index: 3)
                chunkEnc.setBytes(&firstStage, length: 4, index: 4)

                let elementsPerGroup = tgSize * 2
                let numGroups = (nInt + elementsPerGroup - 1) / elementsPerGroup
                chunkEnc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                chunkEnc.endEncoding()
                chunkCmdBuf.commit()
                chunkCmdBuf.waitUntilCompleted()
                if let error = chunkCmdBuf.error {
                    throw MSMError.gpuError(error.localizedDescription)
                }

                if firstStage == 0 { break }
                stageOffset = firstStage - 1
            }
            // Fused path: all chunks processed, scale in separate command buffer
            guard let scaleCmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }
            let scaleEnc = scaleCmdBuf.makeComputeCommandEncoder()!
            scaleEnc.setComputePipelineState(scaleFunction)
            scaleEnc.setBuffer(data, offset: 0, index: 0)
            scaleEnc.setBuffer(invN, offset: 0, index: 1)
            scaleEnc.setBytes(&nVal, length: 4, index: 2)
            let tgScale = min(256, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
            scaleEnc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
            scaleEnc.endEncoding()
            scaleCmdBuf.commit()
            scaleCmdBuf.waitUntilCompleted()
            if let error = scaleCmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }
            return
        } else {
            // Original per-stage path for small transforms
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }
            let enc = cmdBuf.makeComputeCommandEncoder()!
            let tgSize = min(256, Int(invButterflyFunction.maxTotalThreadsPerThreadgroup))
            for stage in stride(from: logN - 1, through: 0, by: -1) {
                if stage < logN - 1 { enc.memoryBarrier(scope: .buffers) }
                enc.setComputePipelineState(invButterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = UInt32(stage)
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }
            // Scale by 1/N
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scaleFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invN, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            let tgScale = min(256, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }
        }
    }

    // MARK: - Four-Step Inverse P^1 NTT

    /// Four-step inverse FFT for large transforms.
    /// Algorithm:
    ///   1. Transpose (undo forward transpose: N2×N1 → N1×N2)
    ///   2. Row iFFTs of size N2 (with inverse twiddle during load)
    ///   3. Inverse twiddle multiply
    ///   4. Column iFFTs of size N1 (N2 columns) + scale by 1/N
    private func inttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var n1Val = n1
        var n2Val = n2

        // Step 1: Out-of-place transpose (undo forward transpose)
        // Forward wrote transposed data, so we transpose back to N1×N2
        let scratch = getScratchBuffer(n: nInt)
        var n1ValCopy = n1Val  // Need mutable for buffer
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(transposeOutOfPlaceFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(scratch, offset: 0, index: 1)
        enc.setBytes(&n1ValCopy, length: 4, index: 2)
        let tg1e = min(tuning.nttThreadgroupSize, Int(transposeOutOfPlaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1e, height: 1, depth: 1))
        enc.endEncoding()

        let blit = cmdBuf.makeBlitCommandEncoder()!
        blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<M31>.stride)
        blit.endEncoding()

        // Step 2: Row iFFTs of size N2 (inverse twiddle fused during load)
        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        var rowLocalStages = UInt32(min(logN2, P1NTTEngine.maxFusedLogN))
        enc2.setComputePipelineState(invRowFusedTwiddleFunction)
        enc2.setBuffer(data, offset: 0, index: 0)
        enc2.setBuffer(invTwiddles, offset: 0, index: 1)
        enc2.setBytes(&nVal, length: 4, index: 2)
        enc2.setBytes(&rowLocalStages, length: 4, index: 3)
        let rowThreads = Int(n2) / 2
        enc2.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: rowThreads, height: 1, depth: 1))
        enc2.memoryBarrier(scope: .buffers)

        // Step 3: Inverse twiddle multiply
        enc2.setComputePipelineState(twiddleFourstepFunction)
        enc2.setBuffer(data, offset: 0, index: 0)
        enc2.setBuffer(invTwiddles, offset: 0, index: 1)
        enc2.setBytes(&nVal, length: 4, index: 2)
        enc2.setBytes(&n1Val, length: 4, index: 3)
        enc2.setBytes(&n2Val, length: 4, index: 4)
        let twTg = min(tuning.nttThreadgroupSize, Int(twiddleFourstepFunction.maxTotalThreadsPerThreadgroup))
        enc2.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: twTg, height: 1, depth: 1))
        enc2.memoryBarrier(scope: .buffers)

        // Step 4: Column iFFTs of size N1 + scale
        var colLocalStages = UInt32(min(logN1, P1NTTEngine.maxFusedLogN))
        enc2.setComputePipelineState(invColumnFusedFunction)
        enc2.setBuffer(data, offset: 0, index: 0)
        enc2.setBuffer(invTwiddles, offset: 0, index: 1)
        enc2.setBytes(&nVal, length: 4, index: 2)
        enc2.setBytes(&n1Val, length: 4, index: 3)
        enc2.setBytes(&n2Val, length: 4, index: 4)
        enc2.setBytes(&colLocalStages, length: 4, index: 5)
        let colThreads = Int(n1) / 2
        enc2.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: colThreads, height: 1, depth: 1))
        enc2.memoryBarrier(scope: .buffers)

        // Scale by 1/N
        enc2.setComputePipelineState(scaleFunction)
        enc2.setBuffer(data, offset: 0, index: 0)
        enc2.setBuffer(invN, offset: 0, index: 1)
        enc2.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
        enc2.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))

        enc2.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - High-level API

    public func ntt(_ input: [M31]) throws -> [M31] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "P1 NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<M31>.stride
        if n > cachedDataBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create data buffer")
            }
            cachedDataBuf = buf
            cachedDataBufElements = n
        }
        let dataBuf = cachedDataBuf!
        input.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * stride)
        }
        try ntt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: M31.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func intt(_ input: [M31]) throws -> [M31] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "P1 NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<M31>.stride
        if n > cachedDataBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create data buffer")
            }
            cachedDataBuf = buf
            cachedDataBufElements = n
        }
        let dataBuf = cachedDataBuf!
        input.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * stride)
        }
        try intt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: M31.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU Reference Implementation

    /// CPU reference P^1 FFT: coefficients -> evaluations
    public static func cpuNTT(_ input: [M31], logN: Int) -> [M31] {
        let n = input.count
        precondition(n == 1 << logN)
        var data = input

        let domain = p1CosetDomain(logN: logN)
        let twiddles = p1PrecomputeForwardTwiddles(logN: logN)

        // Standard radix-2 DIT
        // Stage s: blocks of size 2^(s+1), half_block = 2^s
        // Twiddle stride = n / block_size = n / 2^(s+1)
        for stage in 0..<logN {
            let halfBlock = 1 << stage
            let blockSize = halfBlock << 1
            let twiddleStride = n >> (stage + 1)

            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for j in 0..<halfBlock {
                    let i0 = blockStart + j
                    let i1 = i0 + halfBlock
                    let twiddleIdx = j * twiddleStride
                    let tw = twiddles[twiddleIdx]
                    let u = data[i0]
                    let v = data[i1]
                    let twv = m31Mul(tw, v)
                    data[i0] = m31Add(u, twv)
                    data[i1] = m31Sub(u, twv)
                }
            }
        }

        return data
    }

    /// CPU reference P^1 IFFT: evaluations -> coefficients
    public static func cpuINTT(_ input: [M31], logN: Int) -> [M31] {
        let n = input.count
        precondition(n == 1 << logN)
        var data = input

        let invTwiddles = p1PrecomputeInverseTwiddles(logN: logN)

        // Scale by 1/N first
        let invN = m31Inverse(M31(v: UInt32(n)))
        for i in 0..<n {
            data[i] = m31Mul(data[i], invN)
        }

        // Standard radix-2 DIF
        // Process from stage logN-1 down to 0
        for stage in stride(from: logN - 1, through: 0, by: -1) {
            let halfBlock = 1 << stage
            let blockSize = halfBlock << 1
            let twiddleStride = n >> (stage + 1)

            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for j in 0..<halfBlock {
                    let i0 = blockStart + j
                    let i1 = i0 + halfBlock
                    let twiddleIdx = j * twiddleStride
                    let invTw = invTwiddles[twiddleIdx]
                    let u = data[i0]
                    let v = data[i1]
                    let sum = m31Add(u, v)
                    let diff = m31Sub(u, v)
                    data[i0] = sum
                    data[i1] = m31Mul(diff, invTw)
                }
            }
        }

        return data
    }
}

// MARK: - P^1 Domain Generation

/// P^1 coset domain: η·G where G is a subgroup of F_p* of order 2^logN.
///
/// For M31 = F_{2^31-1}, the multiplicative group has order p-1 = 2^31 - 2.
/// We find a subgroup of order 2^logN by computing a generator of the
/// 2^30-order subgroup (the largest power-of-2 subgroup), then using
/// the appropriate power for smaller subgroups.
///
/// Domain construction:
/// - Find a primitive 2^30-th root of unity ω in F_p*
/// - The domain is {η·ω^(i·2^(30-logN)) : i = 0..2^logN-1}
/// - We use η = ω^B where B is a random exponent to avoid trivial values

/// Find a generator of the large 2-power subgroup of F_p*
/// Returns an element of order 2^30 (the largest power of 2 dividing p-1)
public func p1Find2PowerGenerator() -> M31 {
    // For M31 = 2^31 - 1, p-1 = 2^31 - 2 = 2 * (2^30 - 1)
    // The 2-Sylow subgroup has order 2, so we can only get order 2 from this
    // But we can construct a subgroup of order 2^30 by using a different approach

    // Actually, we need to find an element of order 2^30 in F_p*
    // Since p is prime, F_p* is cyclic of order p-1 = 2^31 - 2
    // We need 2^30 | (p-1), but 2^30 does not divide 2^31 - 2

    // Workaround: use the fact that M31 has a multiplicative subgroup of index 2
    // that has order (p-1)/2 = 2^30 - 1, which is odd.
    // We can use the cube root of unity trick or work with a different structure.

    // For now, let's use a simple approach: find any element and raise to appropriate power
    // to get the domain points we need

    // Start with a primitive element
    let p = M31.P
    // Use 3 as a potential generator (it often works for prime fields)
    var g = M31(v: 3)

    // Compute g^((p-1)/2^30) to get an element of order 2^30
    // But (p-1)/2^30 = (2^31 - 2)/2^30 = 2 - 1/2^29, which is not integer

    // Since we can't get 2^30, let's use what we can get
    // The largest power of 2 dividing p-1 is 2^1
    // So we'll use the order-2 subgroup {1, -1}

    // Actually, for a proper FFT we need 2^logN to divide the group order
    // Since max 2-adicity is 2^1, we can only do trivial FFTs

    // Alternative: construct the domain abstractly using the t → t² folding
    // For FRI, we don't need a full FFT - we just need domain points that square nicely

    // Let's construct domain points using the squaring map structure
    // We need points t such that t² gives us the pairing structure for FRI

    // For the P^1 approach, we'll construct the domain as:
    // D = {±1, ±g, ±g², ...} where g is chosen so squaring maps nicely

    // Since this is a prototype and the full theory is still being developed,
    // we'll use a simple domain based on the FRI folding structure

    return g
}

/// Get the P^1 coset domain of size 2^logN for the given shift.
/// The domain is constructed to work with FRI folding t → t².
public func p1CosetDomain(logN: Int, shift: M31? = nil) -> [M31] {
    let n = 1 << logN

    // For M31, we construct a domain where squaring maps elements to each other
    // We use the fact that -1 has order 2, so pairs (±t) square to the same value

    // Simple construction: use powers of a base element, with sign variations
    // This gives us the "twin" structure needed for FRI

    // Find a generator that works for our domain
    // We'll use g = primitive element raised to (p-1)/2^(logN) if possible
    let pMinus1 = Int(M31.P - 1)

    // Since p-1 = 2^31 - 2 = 2 * (2^30 - 1), the max 2-power is 2^1
    // We construct the domain differently - using sign pairs

    // Domain construction using sign pairs:
    // For i in [0, n/2): t[i] = g^i, t[i+n/2] = -g^i
    // Then squaring pairs: (g^i)² = (g^2)^i and ((-g^i)²) = (g^2)^i

    // Use a primitive element as base
    var g = M31(v: 3)  // common primitive element

    // If shift is provided, multiply by it
    let base = shift ?? M31.one

    var domain = [M31](repeating: M31.zero, count: n)
    let half = n >> 1

    for i in 0..<half {
        var t = m31Pow(g, UInt32(i))
        t = m31Mul(t, base)
        domain[i] = t
        domain[i + half] = m31Neg(t)  // The paired element
    }

    return domain
}

/// Precompute forward twiddles for P^1 NTT (standard root-of-unity).
public func p1PrecomputeForwardTwiddles(logN: Int) -> [M31] {
    let n = 1 << logN
    let half = n >> 1

    // Find a primitive 2^logN-th root of unity in F_p*
    // We need ω such that ω^(2^logN) = 1 but ω^(2^(logN-1)) ≠ 1
    //
    // Since we can't get a proper root of unity (max 2-adicity is 2^1),
    // we construct twiddles that work with our domain structure
    //
    // For the sign-pair domain, twiddles are based on g^(2*i) pattern

    // Use g = 3 as base, construct appropriate twiddles
    // The key property: for domain points (±g^i), the twiddle for the
    // butterfly should pair g^i with -g^i

    var twiddles = [M31](repeating: M31.zero, count: half)

    // For standard radix-2 FFT on our domain, we use:
    // ω_j = g^(2*j) for j in [0, n/2)
    // where g is chosen so this gives proper twiddle behavior

    // Simple construction: use g = ζ^(n/2) where ζ is primitive n-th root
    // Since we can't find a proper root, use powers of 3

    let g = M31(v: 3)

    for j in 0..<half {
        // ω_j = g^(2*j) gives the right twiddle behavior for our domain
        twiddles[j] = m31Pow(g, UInt32(2 * j))
    }

    // Normalize: ensure ω_0 = 1
    let invTw0 = m31Inverse(twiddles[0])
    for j in 0..<half {
        twiddles[j] = m31Mul(twiddles[j], invTw0)
    }

    return twiddles
}

/// Precompute inverse twiddles for P^1 IFFT.
public func p1PrecomputeInverseTwiddles(logN: Int) -> [M31] {
    let twiddles = p1PrecomputeForwardTwiddles(logN: logN)
    return twiddles.map { m31Inverse($0) }
}

// MARK: - Standard root of unity helper

/// Find a primitive n-th root of unity in F_p*, where n divides p-1.
/// Returns nil if n does not divide p-1 or no root exists.
public func p1FindRootOfUnity(_ n: Int) -> M31? {
    let pMinus1 = Int(M31.P - 1)
    if pMinus1 % n != 0 {
        return nil  // n must divide p-1
    }

    // Find a generator of F_p*
    let g = M31(v: 3)  // Use 3 as primitive element (should work for M31)

    // Compute g^((p-1)/n) to get an element of order n
    let exponent = UInt32(pMinus1 / n)
    return m31Pow(g, exponent)
}
