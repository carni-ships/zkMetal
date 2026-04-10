// Pallas Fr NTT Engine — GPU-accelerated NTT on Pallas scalar field
// Pallas Fr = Vesta Fp (cycle property), so uses VestaFp arithmetic.
// Forward NTT: Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT: Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + scale)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Pallas Fr field helpers (Pallas Fr = VestaFp)

// Primitive 2^32-th root of unity for Pallas Fr (in standard form).
// = 5^((p-1)/2^32) mod p, where p is the Pallas scalar field.
// Computed: pow(5, (p-1)/2^32, p)
private let _pallasFrRootOfUnity2_32: [UInt64] = [
    0xa70e2c1102b6d05f, 0x9bb97ea3c106f049,
    0x9e5c4dfd492ae26e, 0x2de6a9b8746d3f58
]
private let _pallasFrTwoAdicity = 32

/// Convert standard-form integer to Pallas Fr Montgomery form (= VestaFp).
private func pallasFrFromInt(_ val: UInt64) -> VestaFp {
    vestaFromInt(val)
}

/// Compute a^n in Pallas Fr (= VestaFp).
private func pallasFrPow(_ a: VestaFp, _ n: UInt64) -> VestaFp {
    if n == 0 { return VestaFp.one }
    if n == 1 { return a }
    var result = VestaFp.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 { result = vestaMul(result, base) }
        base = vestaSqr(base)
        k >>= 1
    }
    return result
}

/// Cached roots of unity for Pallas Fr — only 33 possible values (logN 0...32).
private let _pallasFrRootCache: [VestaFp] = {
    // Convert root from standard to Montgomery form
    let rootStd = VestaFp.from64(_pallasFrRootOfUnity2_32)
    let rootMont = vestaMul(rootStd, VestaFp.from64(VestaFp.R2_MOD_P))

    var cache = [VestaFp](repeating: VestaFp.zero, count: _pallasFrTwoAdicity + 1)
    cache[_pallasFrTwoAdicity] = rootMont
    for k in stride(from: _pallasFrTwoAdicity - 1, through: 0, by: -1) {
        cache[k] = vestaSqr(cache[k + 1])
    }
    return cache
}()

/// Get the primitive 2^k-th root of unity in Pallas Fr.
private func pallasFrRootOfUnity(logN: Int) -> VestaFp {
    precondition(logN >= 0 && logN <= _pallasFrTwoAdicity, "logN out of range for Pallas Fr")
    return _pallasFrRootCache[logN]
}

/// Precompute forward twiddle factors for Pallas Fr NTT.
private func pallasFrPrecomputeTwiddles(logN: Int) -> [VestaFp] {
    let n = 1 << logN
    let omega = pallasFrRootOfUnity(logN: logN)
    var twiddles = [VestaFp](repeating: VestaFp.one, count: n)
    for i in 1..<n {
        twiddles[i] = vestaMul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for Pallas Fr iNTT.
private func pallasFrPrecomputeInverseTwiddles(logN: Int) -> [VestaFp] {
    let n = 1 << logN
    let omega = pallasFrRootOfUnity(logN: logN)
    let omegaInv = vestaInverse(omega)
    var twiddles = [VestaFp](repeating: VestaFp.one, count: n)
    for i in 1..<n {
        twiddles[i] = vestaMul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}

// MARK: - CPU NTT fallback

/// CPU NTT for Pallas Fr (small sizes or fallback).
public func pallasFrCpuNTT(_ input: [VestaFp], logN: Int) -> [VestaFp] {
    let n = 1 << logN
    precondition(input.count == n)
    var data = input
    data.withUnsafeMutableBytes { buf in
        pallas_fr_ntt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logN))
    }
    return data
}

/// CPU iNTT for Pallas Fr.
public func pallasFrCpuINTT(_ input: [VestaFp], logN: Int) -> [VestaFp] {
    let n = 1 << logN
    precondition(input.count == n)
    var data = input
    data.withUnsafeMutableBytes { buf in
        pallas_fr_intt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logN))
    }
    return data
}

// MARK: - GPU NTT Engine

public class PallasNTTEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let butterflyFunction: MTLComputePipelineState
    let invButterflyFunction: MTLComputePipelineState
    let butterflyFusedFunction: MTLComputePipelineState
    let invButterflyFusedFunction: MTLComputePipelineState
    let butterflyFusedBitrevFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState
    let bitrevInplaceFunction: MTLComputePipelineState
    let bitrevScaleFunction: MTLComputePipelineState

    private var twiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var scratchBuffer: MTLBuffer?
    private var scratchCapacity: Int = 0

    // 1024 VestaFp elements * 32 bytes = 32KB threadgroup memory
    private static let maxFusedElements = 1024
    private static let maxFusedLogN = 10

    // CPU fallback threshold — use CPU for logN <= this
    private static let cpuFallbackLogN = 8

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try PallasNTTEngine.compileShaders(device: device)

        guard let butterflyFn = library.makeFunction(name: "pallas_ntt_butterfly"),
              let invButterflyFn = library.makeFunction(name: "pallas_intt_butterfly"),
              let butterflyFusedFn = library.makeFunction(name: "pallas_ntt_butterfly_fused"),
              let invButterflyFusedFn = library.makeFunction(name: "pallas_intt_butterfly_fused"),
              let butterflyFusedBitrevFn = library.makeFunction(name: "pallas_ntt_butterfly_fused_bitrev"),
              let scaleFn = library.makeFunction(name: "pallas_ntt_scale"),
              let bitrevInplaceFn = library.makeFunction(name: "pallas_ntt_bitrev_inplace"),
              let bitrevScaleFn = library.makeFunction(name: "pallas_ntt_bitrev_scale") else {
            throw MSMError.missingKernel
        }

        self.butterflyFunction = try device.makeComputePipelineState(function: butterflyFn)
        self.invButterflyFunction = try device.makeComputePipelineState(function: invButterflyFn)
        self.butterflyFusedFunction = try device.makeComputePipelineState(function: butterflyFusedFn)
        self.invButterflyFusedFunction = try device.makeComputePipelineState(function: invButterflyFusedFn)
        self.butterflyFusedBitrevFunction = try device.makeComputePipelineState(function: butterflyFusedBitrevFn)
        self.scaleFunction = try device.makeComputePipelineState(function: scaleFn)
        self.bitrevInplaceFunction = try device.makeComputePipelineState(function: bitrevInplaceFn)
        self.bitrevScaleFunction = try device.makeComputePipelineState(function: bitrevScaleFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/vesta_fp.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/pallas_ntt_kernels.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef VESTA_FP_METAL", with: "")
            .replacingOccurrences(of: "#define VESTA_FP_METAL", with: "")
            .replacingOccurrences(of: "#endif // VESTA_FP_METAL", with: "")

        let combined = cleanField + "\n" + cleanNTT
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<VestaFp>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    // MARK: - Twiddle caching

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = pallasFrPrecomputeTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = pallasFrPrecomputeInverseTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt64(1 << logN)
        let invN = vestaInverse(vestaFromInt(n))
        let buf = createBuffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createBuffer(_ data: [VestaFp]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<VestaFp>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Forward NTT (GPU buffer)

    public func ntt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var logNVal = UInt32(logN)
        let fusedStages = min(logN, PallasNTTEngine.maxFusedLogN)
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
            let tg0 = min(Int(bitrevInplaceFunction.maxTotalThreadsPerThreadgroup), 256)
            enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg0, height: 1, depth: 1))
        }

        // Global butterfly stages (radix-2)
        let startStage = hasFused ? UInt32(fusedStages) : 0
        var stage = startStage
        while stage < UInt32(logN) {
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
            stage += 1
        }
        enc.endEncoding()

        if hasFused {
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(from: scratch!, sourceOffset: 0, to: data, destinationOffset: 0,
                     size: nInt * MemoryLayout<VestaFp>.stride)
            blit.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse NTT (GPU buffer)

    public func intt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInvTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        let fusedStages = min(logN, PallasNTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages (top-down, radix-2)
        let globalEnd = fusedStages > 1 ? UInt32(fusedStages) : 0
        let numGlobalStages = UInt32(logN) - globalEnd
        if numGlobalStages > 0 {
            for si in 0..<Int(numGlobalStages) {
                let stage = UInt32(logN) - 1 - UInt32(si)
                if si > 0 { enc.memoryBarrier(scope: .buffers) }
                enc.setComputePipelineState(invButterflyFunction)
                enc.setBuffer(data, offset: 0, index: 0)
                enc.setBuffer(invTwiddles, offset: 0, index: 1)
                enc.setBytes(&nVal, length: 4, index: 2)
                var stageVal = stage
                enc.setBytes(&stageVal, length: 4, index: 3)
                let numButterflies = nInt / 2
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
            var stageOffset: UInt32 = 0
            enc.setBytes(&stageOffset, length: 4, index: 4)
            let tgThreads = (1 << fusedStages) / 2
            let numGroups = nInt / (1 << fusedStages)
            enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
        }

        // Step 3: Fused bit-reversal + scale by 1/n
        enc.memoryBarrier(scope: .buffers)
        var logNVal = UInt32(logN)
        enc.setComputePipelineState(bitrevScaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        enc.setBytes(&logNVal, length: 4, index: 3)
        let tg = min(Int(bitrevScaleFunction.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Array convenience APIs

    /// Forward NTT on a Swift array (copies to GPU and back).
    public func ntt(_ input: [VestaFp]) throws -> [VestaFp] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let logN = Int(log2(Double(n)))

        // CPU fallback for small sizes
        if logN <= PallasNTTEngine.cpuFallbackLogN {
            return pallasFrCpuNTT(input, logN: logN)
        }

        let byteCount = n * MemoryLayout<VestaFp>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.noGPU
        }
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }

        try ntt(data: buf, logN: logN)

        var result = [VestaFp](repeating: VestaFp.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, buf.contents(), byteCount)
        }
        return result
    }

    /// Inverse NTT on a Swift array (copies to GPU and back).
    public func intt(_ input: [VestaFp]) throws -> [VestaFp] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let logN = Int(log2(Double(n)))

        if logN <= PallasNTTEngine.cpuFallbackLogN {
            return pallasFrCpuINTT(input, logN: logN)
        }

        let byteCount = n * MemoryLayout<VestaFp>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.noGPU
        }
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }

        try intt(data: buf, logN: logN)

        var result = [VestaFp](repeating: VestaFp.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, buf.contents(), byteCount)
        }
        return result
    }

    /// CPU-only NTT (static, no GPU needed).
    public static func cpuNTT(_ input: [VestaFp], logN: Int) -> [VestaFp] {
        pallasFrCpuNTT(input, logN: logN)
    }

    /// CPU-only iNTT (static, no GPU needed).
    public static func cpuINTT(_ input: [VestaFp], logN: Int) -> [VestaFp] {
        pallasFrCpuINTT(input, logN: logN)
    }
}
