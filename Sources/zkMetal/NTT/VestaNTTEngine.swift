// Vesta Fr NTT Engine — GPU-accelerated NTT on Vesta scalar field
// Vesta Fr = Pallas Fp (cycle property), so uses PallasFp arithmetic.
// Forward NTT: Cooley-Tukey radix-2 DIT (bit-reversal + butterfly stages)
// Inverse NTT: Gentleman-Sande radix-2 DIF (butterfly stages + bit-reversal + scale)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Vesta Fr field helpers (Vesta Fr = PallasFp)

// Primitive 2^32-th root of unity for Vesta Fr (in standard form).
// = 5^((p-1)/2^32) mod p, where p is the Vesta scalar field.
// Computed: pow(5, (p-1)/2^32, p)
private let _vestaFrRootOfUnity2_32: [UInt64] = [
    0xbdad6fabd87ea32f, 0xea322bf2b7bb7584,
    0x362120830561f81a, 0x2bce74deac30ebda
]
private let _vestaFrTwoAdicity = 32

/// Compute a^n in Vesta Fr (= PallasFp).
private func vestaFrPow(_ a: PallasFp, _ n: UInt64) -> PallasFp {
    if n == 0 { return PallasFp.one }
    if n == 1 { return a }
    var result = PallasFp.one
    var base = a
    var k = n
    while k > 0 {
        if k & 1 == 1 { result = pallasMul(result, base) }
        base = pallasSqr(base)
        k >>= 1
    }
    return result
}

/// Cached roots of unity for Vesta Fr — only 33 possible values (logN 0...32).
private let _vestaFrRootCache: [PallasFp] = {
    // Convert root from standard to Montgomery form
    let rootStd = PallasFp.from64(_vestaFrRootOfUnity2_32)
    let rootMont = pallasMul(rootStd, PallasFp.from64(PallasFp.R2_MOD_P))

    var cache = [PallasFp](repeating: PallasFp.zero, count: _vestaFrTwoAdicity + 1)
    cache[_vestaFrTwoAdicity] = rootMont
    for k in stride(from: _vestaFrTwoAdicity - 1, through: 0, by: -1) {
        cache[k] = pallasSqr(cache[k + 1])
    }
    return cache
}()

/// Get the primitive 2^k-th root of unity in Vesta Fr.
private func vestaFrRootOfUnity(logN: Int) -> PallasFp {
    precondition(logN >= 0 && logN <= _vestaFrTwoAdicity, "logN out of range for Vesta Fr")
    return _vestaFrRootCache[logN]
}

/// Precompute forward twiddle factors for Vesta Fr NTT.
private func vestaFrPrecomputeTwiddles(logN: Int) -> [PallasFp] {
    let n = 1 << logN
    let omega = vestaFrRootOfUnity(logN: logN)
    var twiddles = [PallasFp](repeating: PallasFp.one, count: n)
    for i in 1..<n {
        twiddles[i] = pallasMul(twiddles[i - 1], omega)
    }
    return twiddles
}

/// Precompute inverse twiddle factors for Vesta Fr iNTT.
private func vestaFrPrecomputeInverseTwiddles(logN: Int) -> [PallasFp] {
    let n = 1 << logN
    let omega = vestaFrRootOfUnity(logN: logN)
    let omegaInv = pallasInverse(omega)
    var twiddles = [PallasFp](repeating: PallasFp.one, count: n)
    for i in 1..<n {
        twiddles[i] = pallasMul(twiddles[i - 1], omegaInv)
    }
    return twiddles
}

// MARK: - CPU NTT fallback

/// CPU NTT for Vesta Fr (small sizes or fallback).
public func vestaFrCpuNTT(_ input: [PallasFp], logN: Int) -> [PallasFp] {
    let n = 1 << logN
    precondition(input.count == n)
    var data = input
    data.withUnsafeMutableBytes { buf in
        vesta_fr_ntt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logN))
    }
    return data
}

/// CPU iNTT for Vesta Fr.
public func vestaFrCpuINTT(_ input: [PallasFp], logN: Int) -> [PallasFp] {
    let n = 1 << logN
    precondition(input.count == n)
    var data = input
    data.withUnsafeMutableBytes { buf in
        vesta_fr_intt(buf.baseAddress!.assumingMemoryBound(to: UInt64.self), Int32(logN))
    }
    return data
}

// MARK: - GPU NTT Engine

public class VestaNTTEngine {
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

    // 1024 PallasFp elements * 32 bytes = 32KB threadgroup memory
    private static let maxFusedElements = 1024
    private static let maxFusedLogN = 10

    // CPU fallback threshold
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

        let library = try VestaNTTEngine.compileShaders(device: device)

        guard let butterflyFn = library.makeFunction(name: "vesta_ntt_butterfly"),
              let invButterflyFn = library.makeFunction(name: "vesta_intt_butterfly"),
              let butterflyFusedFn = library.makeFunction(name: "vesta_ntt_butterfly_fused"),
              let invButterflyFusedFn = library.makeFunction(name: "vesta_intt_butterfly_fused"),
              let butterflyFusedBitrevFn = library.makeFunction(name: "vesta_ntt_butterfly_fused_bitrev"),
              let scaleFn = library.makeFunction(name: "vesta_ntt_scale"),
              let bitrevInplaceFn = library.makeFunction(name: "vesta_ntt_bitrev_inplace"),
              let bitrevScaleFn = library.makeFunction(name: "vesta_ntt_bitrev_scale") else {
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
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/pallas_fp.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/vesta_ntt_kernels.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef PALLAS_FP_METAL", with: "")
            .replacingOccurrences(of: "#define PALLAS_FP_METAL", with: "")
            .replacingOccurrences(of: "#endif // PALLAS_FP_METAL", with: "")

        let combined = cleanField + "\n" + cleanNTT
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<PallasFp>.stride
        if needed <= scratchCapacity, let buf = scratchBuffer { return buf }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuffer!
    }

    // MARK: - Twiddle caching

    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }
        let twiddles = vestaFrPrecomputeTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        twiddleCache[logN] = buf
        return buf
    }

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = vestaFrPrecomputeInverseTwiddles(logN: logN)
        let buf = createBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt64(1 << logN)
        let invN = pallasInverse(pallasFromInt(n))
        let buf = createBuffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createBuffer(_ data: [PallasFp]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<PallasFp>.stride
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
        let fusedStages = min(logN, VestaNTTEngine.maxFusedLogN)
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
                     size: nInt * MemoryLayout<PallasFp>.stride)
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
        let fusedStages = min(logN, VestaNTTEngine.maxFusedLogN)

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Step 1: Global DIF stages
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
    public func ntt(_ input: [PallasFp]) throws -> [PallasFp] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let logN = Int(log2(Double(n)))

        if logN <= VestaNTTEngine.cpuFallbackLogN {
            return vestaFrCpuNTT(input, logN: logN)
        }

        let byteCount = n * MemoryLayout<PallasFp>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.noGPU
        }
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }

        try ntt(data: buf, logN: logN)

        var result = [PallasFp](repeating: PallasFp.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, buf.contents(), byteCount)
        }
        return result
    }

    /// Inverse NTT on a Swift array (copies to GPU and back).
    public func intt(_ input: [PallasFp]) throws -> [PallasFp] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2")
        let logN = Int(log2(Double(n)))

        if logN <= VestaNTTEngine.cpuFallbackLogN {
            return vestaFrCpuINTT(input, logN: logN)
        }

        let byteCount = n * MemoryLayout<PallasFp>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.noGPU
        }
        input.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }

        try intt(data: buf, logN: logN)

        var result = [PallasFp](repeating: PallasFp.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, buf.contents(), byteCount)
        }
        return result
    }

    /// CPU-only NTT (static, no GPU needed).
    public static func cpuNTT(_ input: [PallasFp], logN: Int) -> [PallasFp] {
        vestaFrCpuNTT(input, logN: logN)
    }

    /// CPU-only iNTT (static, no GPU needed).
    public static func cpuINTT(_ input: [PallasFp], logN: Int) -> [PallasFp] {
        vestaFrCpuINTT(input, logN: logN)
    }
}
