// Goldilocks NTT Engine — GPU-accelerated NTT on Goldilocks field (p = 2^64 - 2^32 + 1)
// Forward NTT: Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT: Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + scale)
// Goldilocks elements are 8 bytes → 4x denser than BN254 Fr in threadgroup memory

import Foundation
import Metal

public class GoldilocksNTTEngine {
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

    // 4096 Gl elements * 8 bytes = 32KB threadgroup memory
    public static let maxFusedElements = 4096
    public static let maxFusedLogN = 12  // log2(4096)
    private static let fourStepMinGlobalStages = 10  // enable four-step for logN >= 22 (8-byte elements need large n for strided access to pay off)

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GoldilocksNTTEngine.compileShaders(device: device)

        guard let butterflyFn = library.makeFunction(name: "gl_ntt_butterfly"),
              let butterflyRadix4Fn = library.makeFunction(name: "gl_ntt_butterfly_radix4"),
              let invButterflyFn = library.makeFunction(name: "gl_intt_butterfly"),
              let invButterflyRadix4Fn = library.makeFunction(name: "gl_intt_butterfly_radix4"),
              let butterflyFusedFn = library.makeFunction(name: "gl_ntt_butterfly_fused"),
              let invButterflyFusedFn = library.makeFunction(name: "gl_intt_butterfly_fused"),
              let scaleFn = library.makeFunction(name: "gl_ntt_scale"),
              let bitrevInplaceFn = library.makeFunction(name: "gl_ntt_bitrev_inplace"),
              let columnFusedFn = library.makeFunction(name: "gl_ntt_column_fused"),
              let rowFusedFn = library.makeFunction(name: "gl_ntt_row_fused"),
              let twiddleMultiplyFn = library.makeFunction(name: "gl_ntt_twiddle_multiply"),
              let transposeFn = library.makeFunction(name: "gl_ntt_transpose"),
              let invColumnFusedFn = library.makeFunction(name: "gl_intt_column_fused"),
              let invRowFusedFn = library.makeFunction(name: "gl_intt_row_fused"),
              let butterflyFusedBitrevFn = library.makeFunction(name: "gl_ntt_butterfly_fused_bitrev"),
              let rowFusedTwiddleFn = library.makeFunction(name: "gl_ntt_row_fused_twiddle"),
              let invRowFusedTwiddleFn = library.makeFunction(name: "gl_intt_row_fused_twiddle"),
              let invColumnFusedScaleFn = library.makeFunction(name: "gl_intt_column_fused_scale") else {
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
        self.twiddleMultiplyFunction = try device.makeComputePipelineState(function: twiddleMultiplyFn)
        self.transposeFunction = try device.makeComputePipelineState(function: transposeFn)
        self.invColumnFusedFunction = try device.makeComputePipelineState(function: invColumnFusedFn)
        self.invRowFusedFunction = try device.makeComputePipelineState(function: invRowFusedFn)
        self.butterflyFusedBitrevFunction = try device.makeComputePipelineState(function: butterflyFusedBitrevFn)
        self.rowFusedTwiddleFunction = try device.makeComputePipelineState(function: rowFusedTwiddleFn)
        self.invRowFusedTwiddleFunction = try device.makeComputePipelineState(function: invRowFusedTwiddleFn)
        self.invColumnFusedScaleFunction = try device.makeComputePipelineState(function: invColumnFusedScaleFn)
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<Gl>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/goldilocks.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/ntt_goldilocks.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#define GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#endif // GOLDILOCKS_METAL", with: "")

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
                let path = url.appendingPathComponent("fields/goldilocks.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./metal/Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/goldilocks.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle factor caching

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = glPrecomputeTwiddles(logN: logN)
        let buf = createGlBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = glPrecomputeInverseTwiddles(logN: logN)
        let buf = createGlBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt64(1 << logN)
        let invN = glInverse(Gl(v: n))
        let buf = createGlBuffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createGlBuffer(_ data: [Gl]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Gl>.stride
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
        let globalStages = logN - GoldilocksNTTEngine.maxFusedLogN
        if globalStages >= GoldilocksNTTEngine.fourStepMinGlobalStages {
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

        let fusedStages = min(logN, GoldilocksNTTEngine.maxFusedLogN)
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
            enc.setComputePipelineState(bitrevInplaceFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBytes(&nVal, length: 4, index: 1)
            enc.setBytes(&logNVal, length: 4, index: 2)
            let tg0 = min(Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup), 256)
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
                let tg4 = min(Int(butterflyRadix4Function.maxTotalThreadsPerThreadgroup), 256)
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
                let tg = min(Int(butterflyFunction.maxTotalThreadsPerThreadgroup), 256)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }
        enc.endEncoding()

        if hasFused {
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch!, sourceOffset: 0, to: data, destinationOffset: 0, size: nInt * MemoryLayout<Gl>.stride)
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
        let globalStages = logN - GoldilocksNTTEngine.maxFusedLogN
        if globalStages >= GoldilocksNTTEngine.fourStepMinGlobalStages {
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
        let fusedStages = min(logN, GoldilocksNTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages (radix-4 where possible)
        let globalEnd = fusedStages > 1 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            var s: UInt32 = 0

            // DIF goes from highest stage downward
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
                let tg4 = min(Int(invButterflyRadix4Function.maxTotalThreadsPerThreadgroup), 256)
                enc.dispatchThreads(MTLSize(width: numQuads, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg4, height: 1, depth: 1))
                s += 2
            }

            // Handle odd remaining stage with radix-2
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
                let tg = min(Int(invButterflyFunction.maxTotalThreadsPerThreadgroup), 256)
                enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            }
        }

        // Step 2: Fused DIF stages
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
            enc.setThreadgroupMemoryLength((1 << fusedStages) * MemoryLayout<Gl>.stride, index: 0)
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
        let tgBR = min(256, Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 4: Scale by 1/n
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(256, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
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
        let tg4 = min(256, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
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
        let tg1 = min(256, Int(transposeFunction.maxTotalThreadsPerThreadgroup))
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

    public func ntt(_ input: [Gl]) throws -> [Gl] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Gl>.stride
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
        let ptr = dataBuf.contents().bindMemory(to: Gl.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func intt(_ input: [Gl]) throws -> [Gl] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<Gl>.stride
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
        let ptr = dataBuf.contents().bindMemory(to: Gl.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU reference

    public static func cpuNTT(_ input: [Gl], logN: Int) -> [Gl] {
        let n = input.count
        var data = bitReverse(input, logN: logN)
        let omega = glRootOfUnity(logN: logN)

        for s in 0..<logN {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let stepsToOmega = n / blockSize
            var w_m = Gl.one
            var temp = omega
            var k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = glMul(w_m, temp) }
                temp = glSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Gl.one
                for j in 0..<halfBlock {
                    let u = data[block + j]
                    let v = glMul(w, data[block + j + halfBlock])
                    data[block + j] = glAdd(u, v)
                    data[block + j + halfBlock] = glSub(u, v)
                    w = glMul(w, w_m)
                }
            }
        }
        return data
    }

    public static func cpuINTT(_ input: [Gl], logN: Int) -> [Gl] {
        let n = input.count
        let omega = glRootOfUnity(logN: logN)
        let omegaInv = glInverse(omega)

        var data = input

        for si in 0..<logN {
            let s = logN - 1 - si
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let stepsToOmega = n / blockSize
            var w_m = Gl.one
            var temp = omegaInv
            var k = stepsToOmega
            while k > 0 {
                if k & 1 == 1 { w_m = glMul(w_m, temp) }
                temp = glSqr(temp)
                k >>= 1
            }

            for block in stride(from: 0, to: n, by: blockSize) {
                var w = Gl.one
                for j in 0..<halfBlock {
                    let a = data[block + j]
                    let b = data[block + j + halfBlock]
                    data[block + j] = glAdd(a, b)
                    data[block + j + halfBlock] = glMul(glSub(a, b), w)
                    w = glMul(w, w_m)
                }
            }
        }

        data = bitReverse(data, logN: logN)

        let invN = glInverse(Gl(v: UInt64(n)))
        for i in 0..<n {
            data[i] = glMul(data[i], invN)
        }
        return data
    }

    private static func bitReverse(_ data: [Gl], logN: Int) -> [Gl] {
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

// MARK: - Twiddle factor precomputation for Goldilocks

public func glRootOfUnity(logN: Int) -> Gl {
    precondition(logN <= Gl.TWO_ADICITY)
    var omega = Gl(v: Gl.ROOT_OF_UNITY)
    for _ in 0..<(Gl.TWO_ADICITY - logN) {
        omega = glSqr(omega)
    }
    return omega
}

public func glPrecomputeTwiddles(logN: Int) -> [Gl] {
    let n = 1 << logN
    let halfN = n / 2
    let omega = glRootOfUnity(logN: logN)
    var twiddles = [Gl](repeating: Gl.one, count: halfN)
    for i in 1..<halfN {
        twiddles[i] = glMul(twiddles[i - 1], omega)
    }
    return twiddles
}

public func glPrecomputeInverseTwiddles(logN: Int) -> [Gl] {
    let n = 1 << logN
    let halfN = n / 2
    let omega = glRootOfUnity(logN: logN)
    let omegaInv = glInverse(omega)
    var twiddles = [Gl](repeating: Gl.one, count: halfN)
    for i in 1..<halfN {
        twiddles[i] = glMul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}
