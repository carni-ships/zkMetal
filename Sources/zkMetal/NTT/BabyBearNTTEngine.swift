// BabyBear NTT Engine — GPU-accelerated NTT on BabyBear field (p = 0x78000001)
// Forward NTT: Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT: Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + scale)
// BabyBear elements are 4 bytes → 8x denser than BN254 Fr in threadgroup memory

import Foundation
import Metal

public class BabyBearNTTEngine {
    public static let version = Versions.nttBabyBear
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let butterflyFunction: MTLComputePipelineState
    let butterflyRadix4Function: MTLComputePipelineState
    let invButterflyFunction: MTLComputePipelineState
    let invButterflyRadix4Function: MTLComputePipelineState
    let butterflyFusedFunction: MTLComputePipelineState
    let invButterflyFusedFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState
    let bitrevInplaceFunction: MTLComputePipelineState
    // Four-step kernels
    let columnFusedFunction: MTLComputePipelineState
    let rowFusedFunction: MTLComputePipelineState
    let twiddleMultiplyFunction: MTLComputePipelineState
    let transposeFunction: MTLComputePipelineState
    let invColumnFusedFunction: MTLComputePipelineState
    let invRowFusedFunction: MTLComputePipelineState
    let butterflyFusedBitrevFunction: MTLComputePipelineState
    let rowFusedTwiddleFunction: MTLComputePipelineState
    let invRowFusedTwiddleFunction: MTLComputePipelineState
    let invColumnFusedScaleFunction: MTLComputePipelineState

    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var scratchBuffer: MTLBuffer?
    private var scratchCapacity: Int = 0
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    // 8192 Bb elements * 4 bytes = 32KB threadgroup memory
    public static let maxFusedElements = 8192
    public static let maxFusedLogN = 13  // log2(8192)
    private var fourStepMinGlobalStages: Int { tuning.nttFourStepThreshold > 10 ? tuning.nttFourStepThreshold - 4 : 6 }

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

        let library = try BabyBearNTTEngine.compileShaders(device: device)

        guard let butterflyFn = library.makeFunction(name: "bb_ntt_butterfly"),
              let butterflyRadix4Fn = library.makeFunction(name: "bb_ntt_butterfly_radix4"),
              let invButterflyFn = library.makeFunction(name: "bb_intt_butterfly"),
              let invButterflyRadix4Fn = library.makeFunction(name: "bb_intt_butterfly_radix4"),
              let butterflyFusedFn = library.makeFunction(name: "bb_ntt_butterfly_fused"),
              let invButterflyFusedFn = library.makeFunction(name: "bb_intt_butterfly_fused"),
              let scaleFn = library.makeFunction(name: "bb_ntt_scale"),
              let bitrevInplaceFn = library.makeFunction(name: "bb_ntt_bitrev_inplace"),
              let columnFusedFn = library.makeFunction(name: "bb_ntt_column_fused"),
              let rowFusedFn = library.makeFunction(name: "bb_ntt_row_fused"),
              let twiddleFn = library.makeFunction(name: "bb_ntt_twiddle_multiply"),
              let transposeFn = library.makeFunction(name: "bb_ntt_transpose"),
              let invColumnFusedFn = library.makeFunction(name: "bb_intt_column_fused"),
              let invRowFusedFn = library.makeFunction(name: "bb_intt_row_fused"),
              let butterflyFusedBitrevFn = library.makeFunction(name: "bb_ntt_butterfly_fused_bitrev"),
              let rowFusedTwiddleFn = library.makeFunction(name: "bb_ntt_row_fused_twiddle"),
              let invRowFusedTwiddleFn = library.makeFunction(name: "bb_intt_row_fused_twiddle"),
              let invColumnFusedScaleFn = library.makeFunction(name: "bb_intt_column_fused_scale") else {
            throw MSMError.missingKernel
        }

        self.butterflyFunction = try device.makeComputePipelineState(function: butterflyFn)
        self.butterflyRadix4Function = try device.makeComputePipelineState(function: butterflyRadix4Fn)
        self.invButterflyFunction = try device.makeComputePipelineState(function: invButterflyFn)
        self.invButterflyRadix4Function = try device.makeComputePipelineState(function: invButterflyRadix4Fn)
        self.butterflyFusedFunction = try device.makeComputePipelineState(function: butterflyFusedFn)
        self.invButterflyFusedFunction = try device.makeComputePipelineState(function: invButterflyFusedFn)
        self.scaleFunction = try device.makeComputePipelineState(function: scaleFn)
        self.bitrevInplaceFunction = try device.makeComputePipelineState(function: bitrevInplaceFn)
        self.columnFusedFunction = try device.makeComputePipelineState(function: columnFusedFn)
        self.rowFusedFunction = try device.makeComputePipelineState(function: rowFusedFn)
        self.twiddleMultiplyFunction = try device.makeComputePipelineState(function: twiddleFn)
        self.transposeFunction = try device.makeComputePipelineState(function: transposeFn)
        self.invColumnFusedFunction = try device.makeComputePipelineState(function: invColumnFusedFn)
        self.invRowFusedFunction = try device.makeComputePipelineState(function: invRowFusedFn)
        self.butterflyFusedBitrevFunction = try device.makeComputePipelineState(function: butterflyFusedBitrevFn)
        self.rowFusedTwiddleFunction = try device.makeComputePipelineState(function: rowFusedTwiddleFn)
        self.invRowFusedTwiddleFunction = try device.makeComputePipelineState(function: invRowFusedTwiddleFn)
        self.invColumnFusedScaleFunction = try device.makeComputePipelineState(function: invColumnFusedScaleFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<Bb>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/ntt_babybear.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

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
                let path = url.appendingPathComponent("fields/babybear.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/babybear.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle factor caching

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = bbPrecomputeTwiddles(logN: logN)
        let buf = createBbBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = bbPrecomputeInverseTwiddles(logN: logN)
        let buf = createBbBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt32(1 << logN)
        let invN = bbInverse(Bb(v: n))
        let buf = createBbBuffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createBbBuffer(_ data: [Bb]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Bb>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Forward NTT

    public func ntt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - BabyBearNTTEngine.maxFusedLogN
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

        let fusedStages = min(logN, BabyBearNTTEngine.maxFusedLogN)
        let hasFused = fusedStages > 1
        let scratch: MTLBuffer? = hasFused ? getScratchBuffer(n: nInt) : nil
        let workBuf = hasFused ? scratch! : data

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if hasFused {
            // Fused bitrev + DIT stages (data → scratch)
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
            // Fallback: separate bitrev
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
            blit.copy(from: scratch!, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Bb>.stride)
            blit.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse NTT

    public func intt(data: MTLBuffer, logN: Int) throws {
        let globalStages = logN - BabyBearNTTEngine.maxFusedLogN
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
        let fusedStages = min(logN, BabyBearNTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages with radix-4
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

        // Step 2: Fused DIF stages (lowest stages)
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
            enc.setThreadgroupMemoryLength((1 << fusedStages) * MemoryLayout<Bb>.stride, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Bit-reversal
        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevInplaceFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 4: Scale by 1/n
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Four-step FFT

    private func nttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let twiddles = getTwiddles(logN: logN)
        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n; var n1Val = n1; var n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Column FFTs
        enc.setComputePipelineState(columnFusedFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        var logN1Val = UInt32(logN1)
        enc.setBytes(&logN1Val, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n1)/2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 2+3: Fused twiddle multiply + row FFTs
        enc.setComputePipelineState(rowFusedTwiddleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(twiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        var logN2Val = UInt32(logN2)
        enc.setBytes(&logN2Val, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n2)/2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 4: Transpose
        enc.setComputePipelineState(transposeFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&n1Val, length: 4, index: 1)
        let tg4 = min(tuning.nttThreadgroupSize, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    private func inttFourStep(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)
        let logN1 = (logN + 1) / 2
        let logN2 = logN - logN1
        let n1 = UInt32(1 << logN1)
        let n2 = UInt32(1 << logN2)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n; var n1Val = n1; var n2Val = n2

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Transpose
        enc.setComputePipelineState(transposeFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&n1Val, length: 4, index: 1)
        let tg1 = min(tuning.nttThreadgroupSize, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 2+3: Fused row DIF iFFTs + inverse twiddle multiply
        enc.setComputePipelineState(invRowFusedTwiddleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invTwiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        var logN2Val = UInt32(logN2)
        enc.setBytes(&logN2Val, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: Int(n1), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n2)/2, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // Step 4+5: Fused column DIF iFFTs + 1/N scale
        enc.setComputePipelineState(invColumnFusedScaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invTwiddles, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&n1Val, length: 4, index: 3)
        enc.setBytes(&n2Val, length: 4, index: 4)
        var logN1Val = UInt32(logN1)
        enc.setBytes(&logN1Val, length: 4, index: 5)
        enc.setBuffer(invN, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: Int(n2), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Int(n1)/2, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - High-level API

    public func ntt(_ input: [Bb]) throws -> [Bb] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Bb>.stride
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
        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func intt(_ input: [Bb]) throws -> [Bb] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Bb>.stride
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
        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU reference

    public static func cpuNTT(_ input: [Bb], logN: Int) -> [Bb] {
        let n = input.count
        var data = bitReverse(input, logN: logN)
        let omega = bbRootOfUnity(logN: logN)

        for s in 0..<logN {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let stepsToOmega = n / blockSize
            var w_m = Bb.one
            var temp = omega
            var k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = bbMul(w_m, temp) }
                temp = bbSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Bb.one
                for j in 0..<halfBlock {
                    let u = data[block + j]
                    let v = bbMul(w, data[block + j + halfBlock])
                    data[block + j] = bbAdd(u, v)
                    data[block + j + halfBlock] = bbSub(u, v)
                    w = bbMul(w, w_m)
                }
            }
        }
        return data
    }

    public static func cpuINTT(_ input: [Bb], logN: Int) -> [Bb] {
        let n = input.count
        let omega = bbRootOfUnity(logN: logN)
        let omegaInv = bbInverse(omega)

        var data = input

        for si in 0..<logN {
            let s = logN - 1 - si
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let stepsToOmega = n / blockSize
            var w_m = Bb.one
            var temp = omegaInv
            var k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = bbMul(w_m, temp) }
                temp = bbSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Bb.one
                for j in 0..<halfBlock {
                    let a = data[block + j]
                    let b = data[block + j + halfBlock]
                    data[block + j] = bbAdd(a, b)
                    data[block + j + halfBlock] = bbMul(bbSub(a, b), w)
                    w = bbMul(w, w_m)
                }
            }
        }

        data = bitReverse(data, logN: logN)

        let invN = bbInverse(Bb(v: UInt32(n)))
        for i in 0..<n {
            data[i] = bbMul(data[i], invN)
        }
        return data
    }

    private static func bitReverse(_ data: [Bb], logN: Int) -> [Bb] {
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

// MARK: - Twiddle factor precomputation for BabyBear

public func bbPrecomputeTwiddles(logN: Int) -> [Bb] {
    let n = 1 << logN
    let halfN = n / 2
    let omega = bbRootOfUnity(logN: logN)
    var twiddles = [Bb](repeating: Bb.one, count: halfN)
    for i in 1..<halfN {
        twiddles[i] = bbMul(twiddles[i - 1], omega)
    }
    return twiddles
}

public func bbPrecomputeInverseTwiddles(logN: Int) -> [Bb] {
    let n = 1 << logN
    let halfN = n / 2
    let omega = bbRootOfUnity(logN: logN)
    let omegaInv = bbInverse(omega)
    var twiddles = [Bb](repeating: Bb.one, count: halfN)
    for i in 1..<halfN {
        twiddles[i] = bbMul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
