// NTT Engine — GPU-accelerated Number Theoretic Transform on BN254 Fr
// Forward NTT: Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT: Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + scale)

import Foundation
import Metal
import NeonFieldOps

public class NTTEngine {
    public static let version = Versions.nttBN254
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let butterflyFunction: MTLComputePipelineState
    let butterflyRadix4Function: MTLComputePipelineState
    let invButterflyFunction: MTLComputePipelineState
    let invButterflyRadix4Function: MTLComputePipelineState
    let butterflyFusedFunction: MTLComputePipelineState
    let invButterflyFusedFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState
    let bitrevFunction: MTLComputePipelineState
    let bitrevInplaceFunction: MTLComputePipelineState
    let bitrevScaleFunction: MTLComputePipelineState
    let columnFusedFunction: MTLComputePipelineState
    let rowFusedFunction: MTLComputePipelineState
    let rowFusedTwiddleFunction: MTLComputePipelineState
    let rowFusedTwiddleTransposeFunction: MTLComputePipelineState
    let butterflyFusedBitrevFunction: MTLComputePipelineState
    let twiddleMultiplyFunction: MTLComputePipelineState
    let transposeFunction: MTLComputePipelineState  // in-place square transpose
    let invColumnFusedFunction: MTLComputePipelineState
    let invColumnFusedTwiddleFunction: MTLComputePipelineState
    let invRowFusedFunction: MTLComputePipelineState
    let columnFusedSubblockFunction: MTLComputePipelineState
    let columnButterflyFunction: MTLComputePipelineState
    let columnButterflyRadix4Function: MTLComputePipelineState
    let transposeRectFunction: MTLComputePipelineState
    let transposeOutOfPlaceFunction: MTLComputePipelineState
    let invColumnButterflyFunction: MTLComputePipelineState
    let invColumnButterflyRadix4Function: MTLComputePipelineState
    let invColumnFusedSubblockFunction: MTLComputePipelineState
    let invRowFusedTwiddleFunction: MTLComputePipelineState
    // Row-layout kernels for transposed column FFTs (coalesced access)
    let rowSubblockFusedFunction: MTLComputePipelineState
    let rowButterflyFunction: MTLComputePipelineState
    let rowButterflyRadix4Function: MTLComputePipelineState
    let invRowSubblockFusedFunction: MTLComputePipelineState
    let invRowButterflyFunction: MTLComputePipelineState
    let invRowButterflyRadix4Function: MTLComputePipelineState

    // Cached twiddle buffers per logN
    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]  // 1/n in Montgomery form
    private var scratchBuffer: MTLBuffer?  // scratch buffer for fused-bitrev (avoids read-write race)
    private var scratchCapacity: Int = 0

    // Cached data buffer for ntt/intt array APIs
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    // Tuning
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

        let shaderDir = findShaderDir()
        let sourceFiles = [
            shaderDir + "/fields/bn254_fr.metal",
            shaderDir + "/ntt/ntt_kernels.metal"
        ]

        let kernelNames = [
            "ntt_butterfly",
            "ntt_butterfly_radix4",
            "intt_butterfly",
            "intt_butterfly_radix4",
            "ntt_butterfly_fused",
            "intt_butterfly_fused",
            "ntt_scale",
            "ntt_bitrev",
            "ntt_bitrev_inplace",
            "ntt_bitrev_scale",
            "ntt_column_fused",
            "ntt_row_fused",
            "ntt_twiddle_multiply",
            "ntt_transpose",
            "intt_column_fused",
            "intt_column_fused_twiddle",
            "intt_row_fused",
            "ntt_row_fused_twiddle",
            "ntt_row_fused_twiddle_transpose",
            "ntt_butterfly_fused_bitrev",
            "ntt_column_fused_subblock",
            "ntt_column_butterfly",
            "ntt_column_butterfly_radix4",
            "ntt_transpose_rect",
            "ntt_transpose_outofplace",
            "intt_column_butterfly",
            "intt_column_butterfly_radix4",
            "intt_column_fused_subblock",
            "intt_row_fused_twiddle",
            "ntt_row_subblock_fused",
            "ntt_row_butterfly",
            "ntt_row_butterfly_radix4",
            "intt_row_subblock_fused",
            "intt_row_butterfly",
            "intt_row_butterfly_radix4"
        ]

        let pipelines = try ShaderCache.shared.loadOrCompile(
            module: "ntt_bn254",
            device: device,
            sourceFiles: sourceFiles,
            kernelNames: kernelNames,
            preprocessor: NTTEngine.preprocessShaders
        )

        guard let butterfly = pipelines["ntt_butterfly"],
              let butterflyRadix4 = pipelines["ntt_butterfly_radix4"],
              let invButterfly = pipelines["intt_butterfly"],
              let invButterflyRadix4 = pipelines["intt_butterfly_radix4"],
              let butterflyFused = pipelines["ntt_butterfly_fused"],
              let invButterflyFused = pipelines["intt_butterfly_fused"],
              let scale = pipelines["ntt_scale"],
              let bitrev = pipelines["ntt_bitrev"],
              let bitrevInplace = pipelines["ntt_bitrev_inplace"],
              let bitrevScale = pipelines["ntt_bitrev_scale"],
              let columnFused = pipelines["ntt_column_fused"],
              let rowFused = pipelines["ntt_row_fused"],
              let twiddleMultiply = pipelines["ntt_twiddle_multiply"],
              let transpose = pipelines["ntt_transpose"],
              let invColumnFused = pipelines["intt_column_fused"],
              let invColumnFusedTwiddle = pipelines["intt_column_fused_twiddle"],
              let invRowFused = pipelines["intt_row_fused"],
              let rowFusedTwiddle = pipelines["ntt_row_fused_twiddle"],
              let rowFusedTwiddleTranspose = pipelines["ntt_row_fused_twiddle_transpose"],
              let butterflyFusedBitrev = pipelines["ntt_butterfly_fused_bitrev"],
              let columnFusedSubblock = pipelines["ntt_column_fused_subblock"],
              let columnButterfly = pipelines["ntt_column_butterfly"],
              let columnButterflyRadix4 = pipelines["ntt_column_butterfly_radix4"],
              let transposeRect = pipelines["ntt_transpose_rect"],
              let transposeOutOfPlace = pipelines["ntt_transpose_outofplace"],
              let invColumnButterfly = pipelines["intt_column_butterfly"],
              let invColumnButterflyRadix4 = pipelines["intt_column_butterfly_radix4"],
              let invColumnFusedSubblock = pipelines["intt_column_fused_subblock"],
              let invRowFusedTwiddle = pipelines["intt_row_fused_twiddle"],
              let rowSubblockFused = pipelines["ntt_row_subblock_fused"],
              let rowButterfly = pipelines["ntt_row_butterfly"],
              let rowButterflyRadix4 = pipelines["ntt_row_butterfly_radix4"],
              let invRowSubblockFused = pipelines["intt_row_subblock_fused"],
              let invRowButterfly = pipelines["intt_row_butterfly"],
              let invRowButterflyRadix4 = pipelines["intt_row_butterfly_radix4"] else {
            throw MSMError.missingKernel
        }

        self.butterflyFunction = butterfly
        self.butterflyRadix4Function = butterflyRadix4
        self.invButterflyFunction = invButterfly
        self.invButterflyRadix4Function = invButterflyRadix4
        self.butterflyFusedFunction = butterflyFused
        self.invButterflyFusedFunction = invButterflyFused
        self.scaleFunction = scale
        self.bitrevFunction = bitrev
        self.bitrevInplaceFunction = bitrevInplace
        self.bitrevScaleFunction = bitrevScale
        self.columnFusedFunction = columnFused
        self.rowFusedFunction = rowFused
        self.twiddleMultiplyFunction = twiddleMultiply
        self.transposeFunction = transpose
        self.invColumnFusedFunction = invColumnFused
        self.invColumnFusedTwiddleFunction = invColumnFusedTwiddle
        self.invRowFusedFunction = invRowFused
        self.rowFusedTwiddleFunction = rowFusedTwiddle
        self.rowFusedTwiddleTransposeFunction = rowFusedTwiddleTranspose
        self.butterflyFusedBitrevFunction = butterflyFusedBitrev
        self.columnFusedSubblockFunction = columnFusedSubblock
        self.columnButterflyFunction = columnButterfly
        self.columnButterflyRadix4Function = columnButterflyRadix4
        self.transposeRectFunction = transposeRect
        self.transposeOutOfPlaceFunction = transposeOutOfPlace
        self.invColumnButterflyFunction = invColumnButterfly
        self.invColumnButterflyRadix4Function = invColumnButterflyRadix4
        self.invColumnFusedSubblockFunction = invColumnFusedSubblock
        self.invRowFusedTwiddleFunction = invRowFusedTwiddle
        self.rowSubblockFusedFunction = rowSubblockFused
        self.rowButterflyFunction = rowButterfly
        self.rowButterflyRadix4Function = rowButterflyRadix4
        self.invRowSubblockFusedFunction = invRowSubblockFused
        self.invRowButterflyFunction = invRowButterfly
        self.invRowButterflyRadix4Function = invRowButterflyRadix4
        self.tuning = TuningManager.shared.config(device: device)
    }

    /// Preprocessor for NTT shaders - removes #include directives and header guards.
    private static func preprocessShaders(_ combined: String) -> String {
        let lines = combined.split(separator: "\n")
            .filter { !$0.contains("#include") }
        var result = lines.joined(separator: "\n")

        // Remove duplicate header guards
        result = result
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        return result
    }

    /// Get or grow scratch buffer for fused-bitrev kernel.
    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<Fr>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    /// Get or create twiddle factor buffer for given logN.
    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = precomputeTwiddles(logN: logN)
        let buf = createFrBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = precomputeInverseTwiddles(logN: logN)
        let buf = createFrBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt64(1 << logN)
        let invN = frInverse(frFromInt(n))
        let buf = createFrBuffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    /// Create a Metal buffer from an array of Fr elements.
    private func createFrBuffer(_ data: [Fr]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // Max elements per threadgroup for fused kernel (1024 Fr * 32 bytes = 32KB shared mem)
    private static let maxFusedElements = 1024
    private static let maxFusedLogN = 10  // log2(1024)

    // Use four-step when global stages > threshold (tuned per device)
    // Also requires logN <= 2*maxFusedLogN so both sub-FFTs fit in shared memory
    private var fourStepMinGlobalStages: Int { tuning.nttFourStepThreshold }

    /// Forward NTT (in-place on GPU buffer).
    /// Uses four-step FFT for large transforms, standard fused+global otherwise.
    public func ntt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            try nttFourStep(data: data, logN: logN)
            return
        }

        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var logNVal = UInt32(logN)

        let fusedStages = min(logN, NTTEngine.maxFusedLogN)

        // Use fused-bitrev kernel when we have fused stages (reads from data, writes to scratch)
        // Then global stages operate on scratch, and we blit copy scratch→data at the end.
        let hasFused = fusedStages > 1
        let hasGlobal = (hasFused ? UInt32(fusedStages) : 0) < UInt32(logN)
        let scratch: MTLBuffer? = hasFused ? getScratchBuffer(n: nInt) : nil
        // workBuf is where global stages and final result live
        let workBuf = hasFused ? scratch! : data

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if hasFused {
            // Step 1: Fused bitrev + DIT stages (data → scratch)
            enc.setComputePipelineState(butterflyFusedBitrevFunction)
            enc.setBuffer(data, offset: 0, index: 0)      // input
            enc.setBuffer(scratch!, offset: 0, index: 1)   // output
            enc.setBuffer(twiddles, offset: 0, index: 2)
            enc.setBytes(&nVal, length: 4, index: 3)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 4)
            enc.setBytes(&logNVal, length: 4, index: 5)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        } else {
            // Fallback: separate bitrev for tiny transforms (no fused stages)
            enc.setComputePipelineState(bitrevInplaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBytes(&nVal, length: 4, index: 1)
            enc.setBytes(&logNVal, length: 4, index: 2)
            let tg0 = min(Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))
        }

        // Step 2: Remaining global stages on workBuf (radix-4 where possible)
        let startStage = hasFused ? UInt32(fusedStages) : 0
        if startStage < UInt32(logN) {
            var stage = startStage

            while stage + 1 < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyRadix4Function)
                enc.setBuffer(workBuf, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = nInt / 4
                let tg4 = min(Int(butterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                stage += 2
            }

            if stage < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyFunction)
                enc.setBuffer(workBuf, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
                let tg = min(Int(butterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }
        enc.endEncoding()

        // Copy scratch → data if we used the scratch buffer
        if hasFused {
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch!, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Inverse NTT (in-place on GPU buffer).
    /// Uses four-step inverse FFT for large transforms, standard fused+global otherwise.
    public func intt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            try inttFourStep(data: data, logN: logN)
            return
        }

        let n = UInt32(1 << logN)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n

        // DIF goes from highest stage down to 0
        // Fuse the last (lowest) stages into threadgroup-local kernel
        let fusedStages = min(logN, NTTEngine.maxFusedLogN)

        // Single encoder for all steps with memoryBarrier between
        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages (radix-4 where possible)
        let globalEnd = fusedStages > 1 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            var s: UInt32 = 0

            // Radix-4 for pairs of stages (DIF: from highest stage down)
            while s + 1 < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyRadix4Function)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = Int(n) / 4
                let tg4 = min(Int(invButterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                s += 2
            }

            // Odd remaining stage
            if s < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = Int(n) / 2
                let tg = min(Int(invButterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }

        // Step 2: Fused DIF stages (lowest stages in threadgroup memory)
        if fusedStages > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(invButterflyFusedFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 3)
            var stageOffset = UInt32(fusedStages - 1)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = Int(n) / (1 << fusedStages)
            enc.setThreadgroupMemoryLength((1 << fusedStages) * MemoryLayout<Fr>.stride, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Fused bit-reversal + scale by 1/n
        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevScaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        enc.setBuffer(invN, offset: 0, index: 3)
        let tg0 = min(tuning.nttThreadgroupSize, Int(bitrevScaleFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Four-step FFT for large NTTs (logN > maxFusedLogN).
    /// For logN <= 2*maxFusedLogN: balanced split, both sub-FFTs fit in shared memory.
    /// For logN > 2*maxFusedLogN: unbalanced split with logN2=maxFusedLogN,
    ///   column FFTs decomposed into sub-block fused + global butterfly stages.
    private func nttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        // Split: balanced when both fit in shared, otherwise force N2 to fit
        let logN2: Int
        let logN1: Int
        if logN <= 2 * NTTEngine.maxFusedLogN {
            logN1 = (logN + 1) / 2
            logN2 = logN - logN1
        } else {
            logN2 = NTTEngine.maxFusedLogN
            logN1 = logN - logN2
        }
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        let colFusedStages = min(logN1, NTTEngine.maxFusedLogN)
        let colGlobalStages = logN1 - colFusedStages
        let needsExtended = colGlobalStages > 0

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var n1Val = n1
        var n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Column FFTs of size N1
        if needsExtended {
            // Extended: sub-block fused FFTs + global butterfly stages
            let subSize = UInt32(1 << colFusedStages)
            let numSubblocks = UInt32(n1 / subSize)
            var subSizeStages = UInt32(colFusedStages)
            var numSubblocksVal = numSubblocks

            enc.setComputePipelineState(columnFusedSubblockFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            enc.setBytes(&n1Val, length: 4, index: 3)
            enc.setBytes(&n2Val, length: 4, index: 4)
            enc.setBytes(&subSizeStages, length: 4, index: 5)
            enc.setBytes(&numSubblocksVal, length: 4, index: 6)
            let subThreads = Int(subSize) / 2
            let numGroups = Int(n2) * Int(numSubblocks)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: subThreads, height: 1, depth: 1))

            var s = colFusedStages
            while s + 1 < logN1 {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(columnButterflyRadix4Function)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&n1Val, length: 4, index: 2)
                enc.setBytes(&n2Val, length: 4, index: 3)
                var stageVal = UInt32(s)
                enc.setBytes(&stageVal, length: 4, index: 4)
                let totalQuads = Int(n2) * Int(n1) / 4
                let tg = min(tuning.nttThreadgroupSize, Int(columnButterflyRadix4Function.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: totalQuads, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                s += 2
            }
            if s < logN1 {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(columnButterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&n1Val, length: 4, index: 2)
                enc.setBytes(&n2Val, length: 4, index: 3)
                var stageVal = UInt32(s)
                enc.setBytes(&stageVal, length: 4, index: 4)
                let totalPairs = Int(n2) * Int(n1) / 2
                let tg = min(tuning.nttThreadgroupSize, Int(columnButterflyFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: totalPairs, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        } else {
            // Standard column: entire column fits in shared memory
            enc.setComputePipelineState(columnFusedFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            enc.setBytes(&n1Val, length: 4, index: 3)
            enc.setBytes(&n2Val, length: 4, index: 4)
            var logN1Val = UInt32(logN1)
            enc.setBytes(&logN1Val, length: 4, index: 5)
            let colThreads = Int(n1) / 2
            enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: colThreads, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)

        if needsExtended && n1 == n2 {
            // In-place path for balanced splits (N1=N2): twiddle + extended row FFT + transpose
            // Avoids scratch buffer and blit copy by keeping everything in data buffer.

            // Step 2: Twiddle multiply (in-place)
            enc.setComputePipelineState(twiddleMultiplyFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&n2Val, length: 4, index: 2)  // number of columns
            enc.setBytes(&nVal, length: 4, index: 3)
            let twTg = min(tuning.nttThreadgroupSize, Int(twiddleMultiplyFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: twTg, height: 1, depth: 1))

            enc.memoryBarrier(scope: .buffers)

            // Step 3: Row FFTs of size N2 using subblock fused + global butterfly
            let rowFusedStages = min(logN2, NTTEngine.maxFusedLogN)
            let rowGlobalStages = logN2 - rowFusedStages

            // 3a: Row subblock fused (up to maxFusedLogN stages)
            let rowSubSize = UInt32(1 << rowFusedStages)
            let rowNumSubblocks = n2 / rowSubSize
            var rowSubStagesVal = UInt32(rowFusedStages)
            var rowNumSubblocksVal = rowNumSubblocks

            enc.setComputePipelineState(rowSubblockFusedFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            enc.setBytes(&n2Val, length: 4, index: 3)   // row size = N2
            enc.setBytes(&n1Val, length: 4, index: 4)   // num rows = N1
            enc.setBytes(&rowSubStagesVal, length: 4, index: 5)
            enc.setBytes(&rowNumSubblocksVal, length: 4, index: 6)
            let rowSubThreads = Int(rowSubSize) / 2
            let rowNumGroups = Int(n1) * Int(rowNumSubblocks)
            enc.dispatchThreadgroups(MTLSize(width: rowNumGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: rowSubThreads, height: 1, depth: 1))

            // 3b: Row global butterfly stages
            if rowGlobalStages > 0 {
                var rs = rowFusedStages
                while rs + 1 < logN2 {
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(rowButterflyRadix4Function)
                    enc.setBuffer(data, offset: 0, index: 0)
                    enc.setBuffer(twiddles, offset: 0, index: 1)
                    enc.setBytes(&n2Val, length: 4, index: 2)
                    enc.setBytes(&n1Val, length: 4, index: 3)
                    var stageVal = UInt32(rs)
                    enc.setBytes(&stageVal, length: 4, index: 4)
                    let totalQuads = Int(n1) * Int(n2) / 4
                    let r4tg = min(tuning.nttThreadgroupSize, Int(rowButterflyRadix4Function.maxTotalThreadsPerThreadgroup))
                    enc.dispatchThreads(MTLSize(width: totalQuads, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: r4tg, height: 1, depth: 1))
                    rs += 2
                }
                if rs < logN2 {
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(rowButterflyFunction)
                    enc.setBuffer(data, offset: 0, index: 0)
                    enc.setBuffer(twiddles, offset: 0, index: 1)
                    enc.setBytes(&n2Val, length: 4, index: 2)
                    enc.setBytes(&n1Val, length: 4, index: 3)
                    var stageVal = UInt32(rs)
                    enc.setBytes(&stageVal, length: 4, index: 4)
                    let totalPairs = Int(n1) * Int(n2) / 2
                    let r2tg = min(tuning.nttThreadgroupSize, Int(rowButterflyFunction.maxTotalThreadsPerThreadgroup))
                    enc.dispatchThreads(MTLSize(width: totalPairs, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: r2tg, height: 1, depth: 1))
                }
            }

            enc.memoryBarrier(scope: .buffers)

            // Step 4: Out-of-place square transpose (N1 = N2)
            // Replaces in-place transpose which wastes 50% of threads (row>=col check)
            // and has uncoalesced strided memory access.
            let scratch = getScratchBuffer(n: nInt)
            enc.setComputePipelineState(transposeOutOfPlaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            enc.setBytes(&n1Val, length: 4, index: 2)
            let trTg = min(tuning.nttThreadgroupSize, Int(transposeOutOfPlaceFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: trTg, height: 1, depth: 1))
            enc.endEncoding()

            // Blit transposed data from scratch back to data buffer
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()
        } else if needsExtended {
            // Unbalanced extended path: use scratch buffer + blit (N1 ≠ N2)
            let scratch = getScratchBuffer(n: nInt)
            enc.setComputePipelineState(rowFusedTwiddleTransposeFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            enc.setBuffer(twiddles, offset: 0, index: 2)
            enc.setBytes(&nVal, length: 4, index: 3)
            var logN2ValExt = UInt32(logN2)
            enc.setBytes(&logN2ValExt, length: 4, index: 4)
            enc.setBytes(&n1Val, length: 4, index: 5)
            let rowThreadsExt = Int(n2) / 2
            enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: rowThreadsExt, height: 1, depth: 1))
            enc.endEncoding()

            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()
        } else {
            // Steps 2+3 fused: Row FFTs with twiddle multiply during load
            enc.setComputePipelineState(rowFusedTwiddleFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var logN2Val = UInt32(logN2)
            enc.setBytes(&logN2Val, length: 4, index: 3)
            let rowThreads = Int(n2) / 2
            enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: rowThreads, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Out-of-place square transpose (N1 = N2 for balanced split)
            let scratch = getScratchBuffer(n: nInt)
            enc.setComputePipelineState(transposeOutOfPlaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            enc.setBytes(&n1Val, length: 4, index: 2)
            let tg4 = min(tuning.nttThreadgroupSize, Int(transposeOutOfPlaceFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
            enc.endEncoding()

            // Blit transposed data from scratch back to data buffer
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Four-step inverse FFT for large iNTTs.
    /// Algorithm: transpose → row DIF iFFTs → inverse twiddle → column DIF iFFTs → scale
    private func inttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        let logN2: Int
        let logN1: Int
        if logN <= 2 * NTTEngine.maxFusedLogN {
            logN1 = (logN + 1) / 2
            logN2 = logN - logN1
        } else {
            logN2 = NTTEngine.maxFusedLogN
            logN1 = logN - logN2
        }
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        let colFusedStages = min(logN1, NTTEngine.maxFusedLogN)
        let colGlobalStages = logN1 - colFusedStages
        let needsExtended = colGlobalStages > 0

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var n1Val = n1
        var n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Transpose (undo the forward's final transpose)
        if needsExtended {
            // Out-of-place rectangular transpose (data → scratch → data)
            // Forward NTT wrote output[col*N1+row], so data is in N2 rows × N1 cols.
            // Un-transpose back to N1 rows × N2 cols.
            let scratch = getScratchBuffer(n: nInt)
            enc.setComputePipelineState(transposeRectFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            enc.setBytes(&n2Val, length: 4, index: 2)  // rows of input = N2
            enc.setBytes(&n1Val, length: 4, index: 3)  // cols of input = N1
            let tg1 = min(tuning.nttThreadgroupSize, Int(transposeRectFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
            enc.endEncoding()

            // Blit scratch → data
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()

            let enc2 = cmdBuf.makeComputeCommandEncoder()!

            // Steps 2+3 fused: Row DIF iFFTs with inverse twiddle applied at writeback
            enc2.setComputePipelineState(invRowFusedTwiddleFunction)
            enc2.setBuffer(data, offset: 0, index: 0)
            enc2.setBuffer(invTwiddles, offset: 0, index: 1)
            enc2.setBytes(&nVal, length: 4, index: 2)
            var logN2Val = UInt32(logN2)
            enc2.setBytes(&logN2Val, length: 4, index: 3)
            let rowThreads = Int(n2) / 2
            enc2.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: rowThreads, height: 1, depth: 1))
            enc2.memoryBarrier(scope: .buffers)

            // Step 4: Column DIF global stages (top stages, high to low, radix-4 when possible)
            var s = 0
            while s + 1 < colGlobalStages {
                enc2.setComputePipelineState(invColumnButterflyRadix4Function)
                enc2.setBuffer(data, offset: 0, index: 0)
                enc2.setBuffer(invTwiddles, offset: 0, index: 1)
                enc2.setBytes(&n1Val, length: 4, index: 2)
                enc2.setBytes(&n2Val, length: 4, index: 3)
                var stageVal = UInt32(s)
                enc2.setBytes(&stageVal, length: 4, index: 4)
                var logN1Val = UInt32(logN1)
                enc2.setBytes(&logN1Val, length: 4, index: 5)
                let totalQuads = Int(n2) * Int(n1) / 4
                let tg = min(tuning.nttThreadgroupSize, Int(invColumnButterflyRadix4Function.maxTotalThreadsPerThreadgroup))
                enc2.dispatchThreads(MTLSize(width: totalQuads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc2.memoryBarrier(scope: .buffers)
                s += 2
            }
            if s < colGlobalStages {
                enc2.setComputePipelineState(invColumnButterflyFunction)
                enc2.setBuffer(data, offset: 0, index: 0)
                enc2.setBuffer(invTwiddles, offset: 0, index: 1)
                enc2.setBytes(&n1Val, length: 4, index: 2)
                enc2.setBytes(&n2Val, length: 4, index: 3)
                var stageVal = UInt32(s)
                enc2.setBytes(&stageVal, length: 4, index: 4)
                var logN1Val = UInt32(logN1)
                enc2.setBytes(&logN1Val, length: 4, index: 5)
                let totalPairs = Int(n2) * Int(n1) / 2
                let tg = min(tuning.nttThreadgroupSize, Int(invColumnButterflyFunction.maxTotalThreadsPerThreadgroup))
                enc2.dispatchThreads(MTLSize(width: totalPairs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc2.memoryBarrier(scope: .buffers)
            }

            // Step 5: Column DIF sub-block fused iFFTs with scale
            let subSize = UInt32(1 << colFusedStages)
            let numSubblocks = UInt32(n1 / subSize)
            var subSizeStages = UInt32(colFusedStages)
            var numSubblocksVal = numSubblocks

            enc2.setComputePipelineState(invColumnFusedSubblockFunction)
            enc2.setBuffer(data, offset: 0, index: 0)
            enc2.setBuffer(invTwiddles, offset: 0, index: 1)
            enc2.setBytes(&nVal, length: 4, index: 2)
            enc2.setBytes(&n1Val, length: 4, index: 3)
            enc2.setBytes(&n2Val, length: 4, index: 4)
            enc2.setBytes(&subSizeStages, length: 4, index: 5)
            enc2.setBytes(&numSubblocksVal, length: 4, index: 6)
            enc2.setBuffer(invN, offset: 0, index: 7)
            let subThreads = Int(subSize) / 2
            let numGroups = Int(n2) * Int(numSubblocks)
            enc2.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: subThreads, height: 1, depth: 1))
            enc2.endEncoding()
        } else {
            // Standard path: square transpose (out-of-place to avoid 50% thread waste)
            let scratch = getScratchBuffer(n: nInt)
            enc.setComputePipelineState(transposeOutOfPlaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch, offset: 0, index: 1)
            enc.setBytes(&n1Val, length: 4, index: 2)
            let tg1 = min(tuning.nttThreadgroupSize, Int(transposeOutOfPlaceFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
            enc.endEncoding()

            // Blit transposed data from scratch back to data buffer
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()

            // Step 2: N1 row DIF iFFTs of size N2
            let enc2 = cmdBuf.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(invRowFusedFunction)
            enc2.setBuffer(data, offset: 0, index: 0)
            enc2.setBuffer(invTwiddles, offset: 0, index: 1)
            enc2.setBytes(&nVal, length: 4, index: 2)
            var logN2Val = UInt32(logN2)
            enc2.setBytes(&logN2Val, length: 4, index: 3)
            let rowThreads = Int(n2) / 2
            enc2.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: rowThreads, height: 1, depth: 1))
            enc2.memoryBarrier(scope: .buffers)

            // Steps 3+4+5 fused: Column DIF iFFTs with inverse twiddle + scale
            enc2.setComputePipelineState(invColumnFusedTwiddleFunction)
            enc2.setBuffer(data, offset: 0, index: 0)
            enc2.setBuffer(invTwiddles, offset: 0, index: 1)
            enc2.setBytes(&nVal, length: 4, index: 2)
            enc2.setBytes(&n1Val, length: 4, index: 3)
            enc2.setBytes(&n2Val, length: 4, index: 4)
            var logN1Val = UInt32(logN1)
            enc2.setBytes(&logN1Val, length: 4, index: 5)
            enc2.setBuffer(invN, offset: 0, index: 6)
            let colThreads = Int(n1) / 2
            enc2.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: colThreads, height: 1, depth: 1))
            enc2.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode NTT into an existing command buffer (standard path only, for chaining).
    /// The four-step path creates its own encoders within the command buffer.
    public func encodeNTT(data: MTLBuffer, logN: Int, cmdBuf: MTLCommandBuffer) {
        let globalStages = logN - NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            // Four-step uses separate encoders which is fine within cmdBuf
            encodeNTTFourStep(data: data, logN: logN, cmdBuf: cmdBuf)
            return
        }
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)
        var nVal = n
        var logNVal = UInt32(logN)

        let fusedStages = min(logN, NTTEngine.maxFusedLogN)
        let hasFused = fusedStages > 1
        let scratch: MTLBuffer? = hasFused ? getScratchBuffer(n: nInt) : nil
        let workBuf = hasFused ? scratch! : data

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if hasFused {
            enc.setComputePipelineState(butterflyFusedBitrevFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(scratch!, offset: 0, index: 1)
            enc.setBuffer(twiddles, offset: 0, index: 2)
            enc.setBytes(&nVal, length: 4, index: 3)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 4)
            enc.setBytes(&logNVal, length: 4, index: 5)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(bitrevInplaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBytes(&nVal, length: 4, index: 1)
            enc.setBytes(&logNVal, length: 4, index: 2)
            let tg0 = min(Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))
        }

        let startStage = hasFused ? UInt32(fusedStages) : 0
        if startStage < UInt32(logN) {
            var stage = startStage
            while stage + 1 < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyRadix4Function)
                enc.setBuffer(workBuf, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = nInt / 4
                let tg4 = min(Int(butterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                stage += 2
            }
            if stage < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyFunction)
                enc.setBuffer(workBuf, offset: 0, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
                let tg = min(Int(butterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }
        enc.endEncoding()

        if hasFused {
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch!, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Fr>.stride)
            blit.endEncoding()
        }
    }

    /// Encode iNTT into an existing command buffer (standard path only).
    public func encodeINTT(data: MTLBuffer, logN: Int, cmdBuf: MTLCommandBuffer) {
        let globalStages = logN - NTTEngine.maxFusedLogN
        if globalStages >= fourStepMinGlobalStages {
            encodeINTTFourStep(data: data, logN: logN, cmdBuf: cmdBuf)
            return
        }
        let n = UInt32(1 << logN)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        var nVal = n
        let fusedStages = min(logN, NTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        let globalEnd = fusedStages > 1 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            var s: UInt32 = 0
            while s + 1 < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyRadix4Function)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = Int(n) / 4
                let tg4 = min(Int(invButterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                s += 2
            }
            if s < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = Int(n) / 2
                let tg = min(Int(invButterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }

        if fusedStages > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(invButterflyFusedFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 3)
            var stageOffset = UInt32(fusedStages - 1)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = Int(n) / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Bit-reversal
        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevInplaceFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Scale by 1/n
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Encode NTT on a sub-region of a buffer (for batched tree operations).
    /// offset is in bytes. Only supports standard path (not four-step).
    public func encodeNTT(data: MTLBuffer, offset: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let twiddles = getTwiddles(logN: logN)
        var nVal = n
        var logNVal = UInt32(logN)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bitrevInplaceFunction)
        enc.setBuffer(data, offset: offset, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tg0 = min(Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))

        let fusedStages = min(logN, NTTEngine.maxFusedLogN)
        if fusedStages > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(butterflyFusedFunction)
            enc.setBuffer(data, offset: offset, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 3)
            var stageOff: UInt32 = 0
            enc.setBytes(&stageOff, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = Int(n) / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }
        let startStage = fusedStages > 1 ? UInt32(fusedStages) : 0
        if startStage < UInt32(logN) {
            var stage = startStage
            while stage + 1 < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyRadix4Function)
                enc.setBuffer(data, offset: offset, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = Int(n) / 4
                let tg4 = min(Int(butterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                stage += 2
            }
            if stage < UInt32(logN) {
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(butterflyFunction)
                enc.setBuffer(data, offset: offset, index: 0)
                enc.setBuffer(twiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = Int(n) / 2
                let tg = min(Int(butterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
    }

    /// Encode iNTT on a sub-region of a buffer (for batched tree operations).
    public func encodeINTT(data: MTLBuffer, offset: Int, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        var nVal = n
        let fusedStages = min(logN, NTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        let globalEnd = fusedStages > 1 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            var s: UInt32 = 0
            while s + 1 < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyRadix4Function)
                enc.setBuffer(data, offset: offset, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numQuads = Int(n) / 4
                let tg4 = min(Int(invButterflyRadix4Function.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                s += 2
            }
            if s < numGlobalStages {
                if s > 0 { enc.memoryBarrier(scope: .buffers) }
                let stage = UInt32(logN) - 1 - s
                enc.setComputePipelineState(invButterflyFunction)
                enc.setBuffer(data, offset: offset, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = Int(n) / 2
                let tgB = min(Int(invButterflyFunction.maxTotalThreadsPerThreadgroup), tuning.nttThreadgroupSize)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tgB, height: 1, depth: 1))
            }
        }

        if fusedStages > 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(invButterflyFusedFunction)
            enc.setBuffer(data, offset: offset, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedStagesVal = UInt32(fusedStages)
            enc.setBytes(&fusedStagesVal, length: 4, index: 3)
            var stageOff = UInt32(fusedStages - 1)
            enc.setBytes(&stageOff, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = Int(n) / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevInplaceFunction)
        enc.setBuffer(data, offset: offset, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFunction)
        enc.setBuffer(data, offset: offset, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
        enc.endEncoding()
    }

    // Encode four-step NTT into existing command buffer
    private func encodeNTTFourStep(data: MTLBuffer, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let twiddles = getTwiddles(logN: logN)
        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)
        var nVal = n, n1Val = n1, n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(columnFusedFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        var logN1Val = UInt32(logN1)
        enc.setBytes(&logN1Val, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n1) / 2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(rowFusedTwiddleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        var logN2Val = UInt32(logN2)
        enc.setBytes(&logN2Val, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: Int(n2) / 2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(transposeFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&n1Val, length: 4, index: 1)
        let tg4e = min(tuning.nttThreadgroupSize, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg4e, height: 1, depth: 1))
        enc.endEncoding()
    }

    // Encode four-step iNTT into existing command buffer
    private func encodeINTTFourStep(data: MTLBuffer, logN: Int, cmdBuf: MTLCommandBuffer) {
        let n = UInt32(1 << logN)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)
        var nVal = n, n1Val = n1, n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(transposeFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&n1Val, length: 4, index: 1)
        let tg1e = min(tuning.nttThreadgroupSize, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1e, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(invRowFusedFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invTwiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        var logN2Val = UInt32(logN2)
        enc.setBytes(&logN2Val, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n2) / 2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(invColumnFusedTwiddleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invTwiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        var logN1Val = UInt32(logN1)
        enc.setBytes(&logN1Val, length: 4, index: 5)
        enc.setBuffer(invN, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: Int(n1) / 2, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// High-level NTT: takes Fr array, returns NTT'd array.
    /// Uses cached buffer to avoid per-call Metal allocation overhead.
    public func ntt(_ input: [Fr]) throws -> [Fr] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Fr>.stride

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

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// High-level iNTT: takes NTT'd array, returns original coefficients.
    /// Uses cached buffer to avoid per-call Metal allocation overhead.
    public func intt(_ input: [Fr]) throws -> [Fr] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Fr>.stride

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

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// CPU reference NTT for correctness verification.
    public static func cpuNTT(_ input: [Fr], logN: Int) -> [Fr] {
        let n = input.count
        var data = bitReverse(input, logN: logN)
        let omega = frRootOfUnity(logN: logN)

        for s in 0..<logN {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            // w_m = omega^(n / blockSize)
            var w_m = Fr.one
            let stepsToOmega = n / blockSize
            var temp = omega
            var k = stepsToOmega
            w_m = Fr.one
            temp = omega
            k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = frMul(w_m, temp) }
                temp = frSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Fr.one
                for j in 0..<halfBlock {
                    let u = data[block + j]
                    let v = frMul(w, data[block + j + halfBlock])
                    data[block + j] = frAdd(u, v)
                    data[block + j + halfBlock] = frSub(u, v)
                    w = frMul(w, w_m)
                }
            }
        }
        return data
    }

    /// CPU reference iNTT.
    public static func cpuINTT(_ input: [Fr], logN: Int) -> [Fr] {
        let n = input.count
        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        var data = input

        // DIF stages (top-down)
        for si in 0..<logN {
            let s = logN - 1 - si
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            var w_m = Fr.one
            let stepsToOmega = n / blockSize
            var temp = omegaInv
            var k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = frMul(w_m, temp) }
                temp = frSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Fr.one
                for j in 0..<halfBlock {
                    let a = data[block + j]
                    let b = data[block + j + halfBlock]
                    data[block + j] = frAdd(a, b)
                    data[block + j + halfBlock] = frMul(frSub(a, b), w)
                    w = frMul(w, w_m)
                }
            }
        }

        data = bitReverse(data, logN: logN)

        // Scale by 1/n
        var invN = frInverse(frFromInt(UInt64(n)))
        data.withUnsafeMutableBytes { dBuf in
            withUnsafeBytes(of: &invN) { sBuf in
                bn254_fr_batch_mul_scalar_neon(
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n))
            }
        }
        return data
    }

    /// Bit-reversal permutation.
    private static func bitReverse(_ data: [Fr], logN: Int) -> [Fr] {
        let n = data.count
        var result = data
        for i in 0..<n {
            var rev = 0
            var val = i
            for _ in 0..<logN {
                rev = (rev << 1) | (val & 1)
                val >>= 1
            }
            result[rev] = data[i]
        }
        return result
    }
}
