// GPUCosetNTTEngine — GPU-accelerated Coset NTT for STARK provers
//
// STARK provers evaluate constraint polynomials on a coset domain
// {g * omega^i} for coset generator g. This requires NTT with a coset shift.
//
// Key optimization: fuses the coset shift multiplication with the first NTT
// butterfly stage, eliminating one full GPU pass over the data.
//
// API:
//   cosetNTT(coeffs:shift:)   — multiply coefficients by shift powers, then NTT
//   cosetINTT(evals:shift:)   — inverse coset NTT (INTT then unshift)
//   cosetLDE(evals:blowupFactor:shift:) — low-degree extension via coset
//
// Supports BN254 Fr and BabyBear fields.

import Foundation
import Metal
import NeonFieldOps

public class GPUCosetNTTEngine {
    public static let version = Versions.gpuCosetNTT

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Fused kernels
    private let cosetShiftButterflyFr: MTLComputePipelineState
    private let cosetShiftButterflyBb: MTLComputePipelineState
    private let inttUnshiftScaleFr: MTLComputePipelineState
    private let inttUnshiftScaleBb: MTLComputePipelineState
    // Standalone shift/unshift
    private let cosetShiftPowersFr: MTLComputePipelineState
    private let cosetShiftPowersBb: MTLComputePipelineState
    private let cosetUnshiftPowersFr: MTLComputePipelineState
    private let cosetUnshiftPowersBb: MTLComputePipelineState
    // Zero-pad + coset shift (reuse from coset_lde_fused)
    private let zeroPadCosetShiftFr: MTLComputePipelineState
    private let zeroPadCosetShiftBb: MTLComputePipelineState

    // NTT engines (lazily initialized)
    private var frNTTEngine: NTTEngine?
    private var bbNTTEngine: BabyBearNTTEngine?

    // Caches: key = "\(logN)_\(shiftKey)"
    private var frShiftPowersCache: [String: MTLBuffer] = [:]
    private var bbShiftPowersCache: [String: MTLBuffer] = [:]
    private var frInvShiftPowersCache: [String: MTLBuffer] = [:]
    private var bbInvShiftPowersCache: [String: MTLBuffer] = [:]
    // Coset LDE power cache
    private var frCosetPowersCache: [String: MTLBuffer] = [:]
    private var bbCosetPowersCache: [String: MTLBuffer] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        // Use ShaderCache for persistent pipeline caching
        let cache = ShaderCache.shared
        let shaderDir = GPUCosetNTTEngine.findShaderDir()
        let sourceFiles = [
            shaderDir + "/fields/bn254_fr.metal",
            shaderDir + "/fields/babybear.metal",
            shaderDir + "/ntt/coset_ntt_fused.metal",
            shaderDir + "/ntt/coset_lde_fused.metal",
        ]
        let kernelNames = [
            "coset_shift_butterfly_fr",
            "coset_shift_butterfly_bb",
            "intt_unshift_scale_fr",
            "intt_unshift_scale_bb",
            "coset_shift_powers_fr",
            "coset_shift_powers_bb",
            "coset_unshift_powers_fr",
            "coset_unshift_powers_bb",
            "lde_zero_pad_coset_shift_fr",
            "lde_zero_pad_coset_shift_bb",
        ]
        let preprocessor: ((String) -> String)? = { combined in
            combined
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { line in
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    if trimmed.contains("#include") || trimmed.hasPrefix("#ifndef") || trimmed.hasPrefix("#endif") { return false }
                    if trimmed.hasPrefix("#define") {
                        let parts = trimmed.split(separator: " ", maxSplits: 3)
                        return parts.count >= 3
                    }
                    return true
                }
                .joined(separator: "\n")
        }

        let pipelines = try cache.loadOrCompile(
            module: "coset_ntt",
            device: device,
            sourceFiles: sourceFiles,
            kernelNames: kernelNames,
            preprocessor: preprocessor
        )

        guard let csbFr = pipelines["coset_shift_butterfly_fr"],
              let csbBb = pipelines["coset_shift_butterfly_bb"],
              let iusFr = pipelines["intt_unshift_scale_fr"],
              let iusBb = pipelines["intt_unshift_scale_bb"],
              let cspFr = pipelines["coset_shift_powers_fr"],
              let cspBb = pipelines["coset_shift_powers_bb"],
              let cupFr = pipelines["coset_unshift_powers_fr"],
              let cupBb = pipelines["coset_unshift_powers_bb"],
              let zpFr = pipelines["lde_zero_pad_coset_shift_fr"],
              let zpBb = pipelines["lde_zero_pad_coset_shift_bb"] else {
            throw MSMError.missingKernel
        }

        self.cosetShiftButterflyFr = csbFr
        self.cosetShiftButterflyBb = csbBb
        self.inttUnshiftScaleFr = iusFr
        self.inttUnshiftScaleBb = iusBb
        self.cosetShiftPowersFr = cspFr
        self.cosetShiftPowersBb = cspBb
        self.cosetUnshiftPowersFr = cupFr
        self.cosetUnshiftPowersBb = cupBb
        self.zeroPadCosetShiftFr = zpFr
        self.zeroPadCosetShiftBb = zpBb
    }

    // MARK: - Shader compilation

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

    // MARK: - NTT engine accessors

    private func getFrNTTEngine() throws -> NTTEngine {
        if let e = frNTTEngine { return e }
        let e = try NTTEngine()
        frNTTEngine = e
        return e
    }

    private func getBbNTTEngine() throws -> BabyBearNTTEngine {
        if let e = bbNTTEngine { return e }
        let e = try BabyBearNTTEngine()
        bbNTTEngine = e
        return e
    }

    // MARK: - Shift power precomputation

    private func shiftKey(_ shift: Fr) -> String {
        "\(shift.v.0)_\(shift.v.1)_\(shift.v.2)_\(shift.v.3)_\(shift.v.4)_\(shift.v.5)_\(shift.v.6)_\(shift.v.7)"
    }

    private func shiftKey(_ shift: Bb) -> String {
        "\(shift.v)"
    }

    /// Precompute coset shift powers: powers[i] = shift^i (for coefficient-domain shift).
    private func getFrShiftPowers(logN: Int, shift: Fr) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = frShiftPowersCache[key] { return cached }
        let n = 1 << logN
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], shift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        frShiftPowersCache[key] = buf
        return buf
    }

    /// Precompute inverse shift powers: powers[i] = shift^(-i).
    private func getFrInvShiftPowers(logN: Int, shift: Fr) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = frInvShiftPowersCache[key] { return cached }
        let n = 1 << logN
        let shiftInv = frInverse(shift)
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], shiftInv)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        frInvShiftPowersCache[key] = buf
        return buf
    }

    private func getBbShiftPowers(logN: Int, shift: Bb) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = bbShiftPowersCache[key] { return cached }
        let n = 1 << logN
        var powers = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n {
            powers[i] = bbMul(powers[i - 1], shift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Bb>.stride,
                                    options: .storageModeShared)!
        bbShiftPowersCache[key] = buf
        return buf
    }

    private func getBbInvShiftPowers(logN: Int, shift: Bb) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = bbInvShiftPowersCache[key] { return cached }
        let n = 1 << logN
        let shiftInv = bbInverse(shift)
        var powers = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n {
            powers[i] = bbMul(powers[i - 1], shiftInv)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Bb>.stride,
                                    options: .storageModeShared)!
        bbInvShiftPowersCache[key] = buf
        return buf
    }

    // MARK: - Coset NTT (BN254 Fr)

    /// Evaluate polynomial (in coefficient form) on coset {shift * omega^i}.
    /// Algorithm: multiply coeffs[i] by shift^i on GPU, then forward NTT.
    /// For large sizes, the shift is fused with the NTT to save a pass.
    public func cosetNTT(coeffs: [Fr], shift: Fr) throws -> [Fr] {
        let n = coeffs.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        // CPU path for small inputs
        if n <= 64 {
            return cpuCosetNTTFr(coeffs: coeffs, shift: shift)
        }

        let engine = try getFrNTTEngine()

        // GPU coset shift in-place, then NTT on same buffer — no CPU roundtrip
        let powers = getFrShiftPowers(logN: logN, shift: shift)
        var data = coeffs
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetShiftPowersFr)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(powers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetShiftPowersFr.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Forward NTT directly on GPU buffer — skip CPU array roundtrip
        try engine.ntt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset NTT (BabyBear)

    /// Evaluate polynomial (in coefficient form) on coset {shift * omega^i}.
    public func cosetNTT(coeffs: [Bb], shift: Bb) throws -> [Bb] {
        let n = coeffs.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        if n <= 256 {
            return cpuCosetNTTBb(coeffs: coeffs, shift: shift)
        }

        let engine = try getBbNTTEngine()

        let powers = getBbShiftPowers(logN: logN, shift: shift)
        var data = coeffs
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetShiftPowersBb)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(powers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetShiftPowersBb.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Forward NTT directly on GPU buffer — skip CPU array roundtrip
        try engine.ntt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset INTT (BN254 Fr)

    /// Interpolate from coset evaluations back to coefficient form.
    /// Algorithm: inverse NTT, then multiply coeffs[i] by shift^(-i).
    public func cosetINTT(evals: [Fr], shift: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        if n <= 64 {
            return cpuCosetINTTFr(evals: evals, shift: shift)
        }

        let engine = try getFrNTTEngine()

        // INTT in-place on GPU buffer, then unshift on same buffer — no CPU roundtrip
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!
        try engine.intt(data: dataBuf, logN: logN)

        // GPU coset unshift
        let invPowers = getFrInvShiftPowers(logN: logN, shift: shift)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetUnshiftPowersFr)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invPowers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetUnshiftPowersFr.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset INTT (BabyBear)

    /// Interpolate from BabyBear coset evaluations back to coefficient form.
    public func cosetINTT(evals: [Bb], shift: Bb) throws -> [Bb] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        if n <= 256 {
            return cpuCosetINTTBb(evals: evals, shift: shift)
        }

        let engine = try getBbNTTEngine()

        // INTT in-place on GPU buffer, then unshift on same buffer — no CPU roundtrip
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!
        try engine.intt(data: dataBuf, logN: logN)

        let invPowers = getBbInvShiftPowers(logN: logN, shift: shift)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetUnshiftPowersBb)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invPowers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetUnshiftPowersBb.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset LDE (BN254 Fr)

    /// Low-degree extension via coset: INTT -> zero-pad -> coset shift -> NTT.
    /// Input: evaluations of size N over standard domain.
    /// Output: evaluations over coset domain of size blowupFactor * N.
    public func cosetLDE(evals: [Fr], blowupFactor: Int, shift: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be power of 2")

        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM
        precondition(logM <= Fr.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        if n <= 64 {
            return cpuCosetLDEFr(evals: evals, blowupFactor: blowupFactor, shift: shift)
        }

        let engine = try getFrNTTEngine()

        // Step 1: INTT to get coefficients — use buffer API to keep data on GPU
        var evalsData = evals
        let inttBuf = device.makeBuffer(bytes: &evalsData, length: n * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!
        try engine.intt(data: inttBuf, logN: logN)

        // Step 2+3: Fused zero-pad + coset shift on GPU
        let cosetPowers = getFrShiftPowers(logN: logM, shift: shift)
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Fr>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftFr)
        enc.setBuffer(inttBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        let tg = min(256, Int(zeroPadCosetShiftFr.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Forward NTT directly on GPU buffer — skip CPU array roundtrip
        try engine.ntt(data: outputBuf, logN: logM)
        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: m)
        return Array(UnsafeBufferPointer(start: ptr, count: m))
    }

    /// Coset LDE with default shift (multiplicative generator).
    public func cosetLDE(evals: [Fr], blowupFactor: Int) throws -> [Fr] {
        return try cosetLDE(evals: evals, blowupFactor: blowupFactor,
                           shift: frFromInt(Fr.GENERATOR))
    }

    // MARK: - Coset LDE (BabyBear)

    /// Low-degree extension via coset for BabyBear field.
    public func cosetLDE(evals: [Bb], blowupFactor: Int, shift: Bb) throws -> [Bb] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be power of 2")

        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM
        precondition(logM <= Bb.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        if n <= 256 {
            return cpuCosetLDEBb(evals: evals, blowupFactor: blowupFactor, shift: shift)
        }

        let engine = try getBbNTTEngine()

        // INTT in-place on GPU buffer — no CPU roundtrip
        var evalsData = evals
        let inttBuf = device.makeBuffer(bytes: &evalsData, length: n * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!
        try engine.intt(data: inttBuf, logN: logN)

        let cosetPowers = getBbShiftPowers(logN: logM, shift: shift)
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Bb>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftBb)
        enc.setBuffer(inttBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        let tg = min(256, Int(zeroPadCosetShiftBb.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Forward NTT directly on GPU buffer — skip CPU array roundtrip
        try engine.ntt(data: outputBuf, logN: logM)
        let ptr = outputBuf.contents().bindMemory(to: Bb.self, capacity: m)
        return Array(UnsafeBufferPointer(start: ptr, count: m))
    }

    /// Coset LDE with default shift (BabyBear multiplicative generator).
    public func cosetLDE(evals: [Bb], blowupFactor: Int) throws -> [Bb] {
        return try cosetLDE(evals: evals, blowupFactor: blowupFactor,
                           shift: Bb(v: Bb.GENERATOR))
    }

    // MARK: - CPU reference implementations

    /// CPU coset NTT for BN254 Fr.
    public func cpuCosetNTTFr(coeffs: [Fr], shift: Fr) -> [Fr] {
        let n = coeffs.count
        let logN = Int(log2(Double(n)))
        var shifted = [Fr](repeating: Fr.zero, count: n)
        var s = shift
        coeffs.withUnsafeBytes { cBuf in
            shifted.withUnsafeMutableBytes { sBuf in
                withUnsafeBytes(of: &s) { bBuf in
                    bn254_fr_batch_mul_powers(
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return NTTEngine.cpuNTT(shifted, logN: logN)
    }

    /// CPU coset INTT for BN254 Fr.
    public func cpuCosetINTTFr(evals: [Fr], shift: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let coeffs = NTTEngine.cpuINTT(evals, logN: logN)
        var shiftInv = frInverse(shift)
        var result = [Fr](repeating: Fr.zero, count: n)
        coeffs.withUnsafeBytes { cBuf in
            result.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: &shiftInv) { bBuf in
                    bn254_fr_batch_mul_powers(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    /// CPU coset NTT for BabyBear.
    public func cpuCosetNTTBb(coeffs: [Bb], shift: Bb) -> [Bb] {
        let n = coeffs.count
        let logN = Int(log2(Double(n)))
        var shifted = [Bb](repeating: Bb.zero, count: n)
        var sPow = Bb.one
        for i in 0..<n {
            shifted[i] = bbMul(coeffs[i], sPow)
            sPow = bbMul(sPow, shift)
        }
        return BabyBearNTTEngine.cpuNTT(shifted, logN: logN)
    }

    /// CPU coset INTT for BabyBear.
    public func cpuCosetINTTBb(evals: [Bb], shift: Bb) -> [Bb] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let coeffs = BabyBearNTTEngine.cpuINTT(evals, logN: logN)
        let shiftInv = bbInverse(shift)
        var result = [Bb](repeating: Bb.zero, count: n)
        var sPow = Bb.one
        for i in 0..<n {
            result[i] = bbMul(coeffs[i], sPow)
            sPow = bbMul(sPow, shiftInv)
        }
        return result
    }

    /// CPU coset LDE for BN254 Fr.
    public func cpuCosetLDEFr(evals: [Fr], blowupFactor: Int, shift: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        let coeffs = NTTEngine.cpuINTT(evals, logN: logN)
        var padded = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        var s = shift
        padded.withUnsafeMutableBytes { pBuf in
            let ptr = pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            withUnsafeBytes(of: &s) { bBuf in
                bn254_fr_batch_mul_powers(ptr, ptr,
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }
        }

        return NTTEngine.cpuNTT(padded, logN: logM)
    }

    /// CPU coset LDE for BabyBear.
    public func cpuCosetLDEBb(evals: [Bb], blowupFactor: Int, shift: Bb) -> [Bb] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        let coeffs = BabyBearNTTEngine.cpuINTT(evals, logN: logN)
        var padded = [Bb](repeating: Bb.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        var sPow = Bb.one
        for i in 0..<m {
            padded[i] = bbMul(padded[i], sPow)
            sPow = bbMul(sPow, shift)
        }

        return BabyBearNTTEngine.cpuNTT(padded, logN: logM)
    }
}
