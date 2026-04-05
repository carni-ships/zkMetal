// GPU FFT Engine for BN254 Fr field
// Stockham auto-sort algorithm (no bit-reversal pass needed)
// Supports forward and inverse FFT with radix-2 and radix-4 butterflies.
// Uses shared memory for small stages, global memory for large stages.
// CPU fallback for small transforms (N <= 16).

import Foundation
import Metal

public class GPUFFTEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Standard in-place butterfly kernels (for global stages)
    private let ditButterflyFn: MTLComputePipelineState
    private let difButterflyFn: MTLComputePipelineState

    // Stockham radix-2 kernels (for reference / alternate path)
    private let stockhamR2Fn: MTLComputePipelineState
    private let stockhamInvR2Fn: MTLComputePipelineState

    // Stockham radix-4 kernel
    private let stockhamR4Fn: MTLComputePipelineState

    // Split-radix kernel
    private let splitRadixFn: MTLComputePipelineState

    // Fused shared-memory kernels
    private let fusedFn: MTLComputePipelineState
    private let fusedInvFn: MTLComputePipelineState

    // Utility kernels
    private let pointwiseMulFn: MTLComputePipelineState
    private let scaleFn: MTLComputePipelineState
    private let bitrevFn: MTLComputePipelineState

    // Caches
    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0
    private var cachedScratchBuf: MTLBuffer?
    private var cachedScratchElements: Int = 0

    private let tuning: TuningConfig

    // Max elements in shared memory: 1024 Fr * 32 bytes = 32KB (and reserve second buffer = 64KB total,
    // but Metal allows 32KB threadgroup memory, so use 512 elements with double buffering)
    // Actually the fused kernel uses two arrays of 1024 each. Metal M-series has 32KB threadgroup mem.
    // With double buffering at 1024 elements: 2 * 1024 * 32 = 64KB -- too much.
    // Use 512 elements (2 * 512 * 32 = 32KB) for double-buffered Stockham.
    // But the standard fused kernel from NTTEngine uses 1024 with single buffer -- we match that.
    // Our fused kernel has shared_a[1024] + shared_b[1024] = 64KB -- that won't fit.
    // Fix: use single-buffer in-place butterfly (not true Stockham) for fused path,
    // and Stockham only for global stages. Let me adjust the fused kernel to single-buffer.
    // The Metal shader already has this as ping-pong. Let's reduce to 512 per buffer.
    private static let maxFusedElements = 512
    private static let maxFusedLogN = 9  // log2(512)

    // CPU fallback threshold
    private static let cpuFallbackLogN = 4  // use CPU for N <= 16

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)

        let library = try GPUFFTEngine.compileShaders(device: device)

        guard let ditBf = library.makeFunction(name: "fft_dit_butterfly"),
              let difBf = library.makeFunction(name: "fft_dif_butterfly"),
              let sr2 = library.makeFunction(name: "fft_stockham_radix2"),
              let sir2 = library.makeFunction(name: "fft_stockham_inv_radix2"),
              let sr4 = library.makeFunction(name: "fft_stockham_radix4"),
              let spr = library.makeFunction(name: "fft_stockham_split_radix"),
              let ff = library.makeFunction(name: "fft_stockham_fused"),
              let fif = library.makeFunction(name: "fft_stockham_inv_fused"),
              let pmul = library.makeFunction(name: "fft_pointwise_mul"),
              let sc = library.makeFunction(name: "fft_scale"),
              let br = library.makeFunction(name: "fft_bitrev_inplace") else {
            throw MSMError.missingKernel
        }

        self.ditButterflyFn = try device.makeComputePipelineState(function: ditBf)
        self.difButterflyFn = try device.makeComputePipelineState(function: difBf)
        self.stockhamR2Fn = try device.makeComputePipelineState(function: sr2)
        self.stockhamInvR2Fn = try device.makeComputePipelineState(function: sir2)
        self.stockhamR4Fn = try device.makeComputePipelineState(function: sr4)
        self.splitRadixFn = try device.makeComputePipelineState(function: spr)
        self.fusedFn = try device.makeComputePipelineState(function: ff)
        self.fusedInvFn = try device.makeComputePipelineState(function: fif)
        self.pointwiseMulFn = try device.makeComputePipelineState(function: pmul)
        self.scaleFn = try device.makeComputePipelineState(function: sc)
        self.bitrevFn = try device.makeComputePipelineState(function: br)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fftSource = try String(contentsOfFile: shaderDir + "/ntt/fft_butterfly.metal", encoding: .utf8)

        let cleanFFT = fftSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = frClean + "\n" + cleanFFT
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Twiddle Factor Caching

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

    private func getOrCreateDataBuffer(elementCount: Int) -> MTLBuffer {
        let needed = elementCount * MemoryLayout<Fr>.stride
        if elementCount <= cachedDataBufElements, let buf = cachedDataBuf { return buf }
        let buf = device.makeBuffer(length: needed, options: .storageModeShared)!
        cachedDataBuf = buf
        cachedDataBufElements = elementCount
        return buf
    }

    private func getOrCreateScratchBuffer(elementCount: Int) -> MTLBuffer {
        let needed = elementCount * MemoryLayout<Fr>.stride
        if elementCount <= cachedScratchElements, let buf = cachedScratchBuf { return buf }
        let buf = device.makeBuffer(length: needed, options: .storageModeShared)!
        cachedScratchBuf = buf
        cachedScratchElements = elementCount
        return buf
    }

    // MARK: - Forward FFT (Stockham auto-sort)

    /// Forward FFT using Stockham auto-sort algorithm.
    /// Uses fused shared-memory kernel for small stages, global Stockham passes for large.
    /// Falls back to CPU for very small transforms.
    public func fft(data: [Fr], logN: Int, inverse: Bool = false) throws -> [Fr] {
        let n = data.count
        precondition(n == (1 << logN), "Data size must equal 2^logN")
        precondition(n > 0 && (n & (n - 1)) == 0, "FFT size must be power of 2")

        // CPU fallback for small transforms
        if logN <= GPUFFTEngine.cpuFallbackLogN {
            if inverse {
                return GPUFFTEngine.cpuIFFT(data, logN: logN)
            } else {
                return GPUFFTEngine.cpuFFT(data, logN: logN)
            }
        }

        let dataBuf = getOrCreateDataBuffer(elementCount: n)
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
        }

        if inverse {
            try performInverseFFT(data: dataBuf, logN: logN)
        } else {
            try performForwardFFT(data: dataBuf, logN: logN)
        }

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Forward FFT using standard Cooley-Tukey approach with bit-reversal + fused + global stages.
    /// This avoids the Stockham double-buffer overhead for the fused path.
    private func performForwardFFT(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var logNVal = UInt32(logN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Bit-reversal permutation
        enc.setComputePipelineState(bitrevFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 2: Fused stages in shared memory
        let fusedStages = min(logN, GPUFFTEngine.maxFusedLogN)
        if fusedStages >= 2 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(fusedFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedVal = UInt32(fusedStages)
            enc.setBytes(&fusedVal, length: 4, index: 3)
            var stageOffset = UInt32(0)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: max(numGroups, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Remaining global stages using standard in-place DIT butterfly
        var stage = UInt32(fusedStages >= 2 ? fusedStages : 0)

        while stage < UInt32(logN) {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(ditButterflyFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            let tg2 = min(tuning.nttThreadgroupSize, Int(ditButterflyFn.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
            stage += 1
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Inverse FFT using DIF (Gentleman-Sande) with bit-reversal at end + 1/N scaling.
    private func performInverseFFT(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var logNVal = UInt32(logN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages (from highest down to fused boundary)
        let fusedStages = min(logN, GPUFFTEngine.maxFusedLogN)
        let globalEnd = fusedStages >= 2 ? UInt32(fusedStages) : 0

        var stage = UInt32(logN) - 1
        while stage >= globalEnd && stage < UInt32(logN) {
            if stage < UInt32(logN) - 1 { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(difButterflyFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = stage
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            let tg2 = min(tuning.nttThreadgroupSize, Int(difButterflyFn.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
            if stage == 0 { break }
            stage -= 1
        }

        // Step 2: Fused DIF stages in shared memory
        if fusedStages >= 2 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(fusedInvFn)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var fusedVal = UInt32(fusedStages)
            enc.setBytes(&fusedVal, length: 4, index: 3)
            var stageOffset = UInt32(fusedStages - 1)
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: max(numGroups, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Bit-reversal
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(bitrevFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&nVal, length: 4, index: 1)
        enc.setBytes(&logNVal, length: 4, index: 2)
        let tgBR = min(tuning.nttThreadgroupSize, Int(bitrevFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgBR, height: 1, depth: 1))

        // Step 4: Scale by 1/N
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFn)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(tuning.nttThreadgroupSize, Int(scaleFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Pointwise Multiplication (for convolution)

    /// Pointwise multiplication on GPU: result[i] = a[i] * b[i]
    public func pointwiseMultiply(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count, "Arrays must be same length")
        let n = a.count

        guard let aBuf = createFrBuffer(a),
              let bBuf = createFrBuffer(b),
              let dstBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create buffers")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pointwiseMulFn)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(dstBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(pointwiseMulFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dstBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU Reference Implementations

    /// CPU forward FFT (Cooley-Tukey DIT with bit-reversal).
    public static func cpuFFT(_ input: [Fr], logN: Int) -> [Fr] {
        return NTTEngine.cpuNTT(input, logN: logN)
    }

    /// CPU inverse FFT.
    public static func cpuIFFT(_ input: [Fr], logN: Int) -> [Fr] {
        return NTTEngine.cpuINTT(input, logN: logN)
    }

    // MARK: - Convolution via FFT

    /// Compute polynomial multiplication via FFT: result = IFFT(FFT(a) . FFT(b))
    /// Input arrays are zero-padded to 2*max(len(a), len(b)) (next power of 2).
    public func convolve(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        let maxLen = max(a.count, b.count)
        var logN = 0
        var size = 1
        while size < 2 * maxLen {
            logN += 1
            size <<= 1
        }

        // Zero-pad inputs
        var aPad = [Fr](repeating: Fr.zero, count: size)
        var bPad = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<a.count { aPad[i] = a[i] }
        for i in 0..<b.count { bPad[i] = b[i] }

        // FFT both
        let aFFT = try fft(data: aPad, logN: logN, inverse: false)
        let bFFT = try fft(data: bPad, logN: logN, inverse: false)

        // Pointwise multiply
        let product = try pointwiseMultiply(aFFT, bFFT)

        // IFFT
        return try fft(data: product, logN: logN, inverse: true)
    }
}
