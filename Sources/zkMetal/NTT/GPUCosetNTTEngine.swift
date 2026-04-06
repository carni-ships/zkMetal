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

        let library = try GPUCosetNTTEngine.compileShaders(device: device)

        guard let csbFr = library.makeFunction(name: "coset_shift_butterfly_fr"),
              let csbBb = library.makeFunction(name: "coset_shift_butterfly_bb"),
              let iusFr = library.makeFunction(name: "intt_unshift_scale_fr"),
              let iusBb = library.makeFunction(name: "intt_unshift_scale_bb"),
              let cspFr = library.makeFunction(name: "coset_shift_powers_fr"),
              let cspBb = library.makeFunction(name: "coset_shift_powers_bb"),
              let cupFr = library.makeFunction(name: "coset_unshift_powers_fr"),
              let cupBb = library.makeFunction(name: "coset_unshift_powers_bb"),
              let zpFr = library.makeFunction(name: "lde_zero_pad_coset_shift_fr"),
              let zpBb = library.makeFunction(name: "lde_zero_pad_coset_shift_bb") else {
            throw MSMError.missingKernel
        }

        self.cosetShiftButterflyFr = try device.makeComputePipelineState(function: csbFr)
        self.cosetShiftButterflyBb = try device.makeComputePipelineState(function: csbBb)
        self.inttUnshiftScaleFr = try device.makeComputePipelineState(function: iusFr)
        self.inttUnshiftScaleBb = try device.makeComputePipelineState(function: iusBb)
        self.cosetShiftPowersFr = try device.makeComputePipelineState(function: cspFr)
        self.cosetShiftPowersBb = try device.makeComputePipelineState(function: cspBb)
        self.cosetUnshiftPowersFr = try device.makeComputePipelineState(function: cupFr)
        self.cosetUnshiftPowersBb = try device.makeComputePipelineState(function: cupBb)
        self.zeroPadCosetShiftFr = try device.makeComputePipelineState(function: zpFr)
        self.zeroPadCosetShiftBb = try device.makeComputePipelineState(function: zpBb)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let fusedSrc = try String(contentsOfFile: shaderDir + "/ntt/coset_ntt_fused.metal", encoding: .utf8)
        let ldeFusedSrc = try String(contentsOfFile: shaderDir + "/ntt/coset_lde_fused.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldFr) + "\n" + clean(fieldBb) + "\n" +
                        clean(fusedSrc) + "\n" + clean(ldeFusedSrc)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
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

        // GPU coset shift then NTT
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

        // Read back shifted coefficients
        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: n))

        // Forward NTT
        return try engine.ntt(shifted)
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

        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: n))
        return try engine.ntt(shifted)
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

        // Inverse NTT
        let coeffs = try engine.intt(evals)

        // GPU coset unshift
        let invPowers = getFrInvShiftPowers(logN: logN, shift: shift)
        var data = coeffs
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!

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
        let coeffs = try engine.intt(evals)

        let invPowers = getBbInvShiftPowers(logN: logN, shift: shift)
        var data = coeffs
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!

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

        // Step 1: INTT to get coefficients
        let coeffs = try engine.intt(evals)

        // Step 2+3: Fused zero-pad + coset shift on GPU
        let cosetPowers = getFrShiftPowers(logN: logM, shift: shift)

        let inputBuf = device.makeBuffer(bytes: coeffs, length: n * MemoryLayout<Fr>.stride,
                                         options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Fr>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftFr)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
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

        // Step 4: Forward NTT of size M
        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
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
        let coeffs = try engine.intt(evals)

        let cosetPowers = getBbShiftPowers(logN: logM, shift: shift)

        let inputBuf = device.makeBuffer(bytes: coeffs, length: n * MemoryLayout<Bb>.stride,
                                         options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: m * MemoryLayout<Bb>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(zeroPadCosetShiftBb)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
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

        let ptr = outputBuf.contents().bindMemory(to: Bb.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
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
        var sPow = Fr.one
        for i in 0..<n {
            shifted[i] = frMul(coeffs[i], sPow)
            sPow = frMul(sPow, shift)
        }
        return NTTEngine.cpuNTT(shifted, logN: logN)
    }

    /// CPU coset INTT for BN254 Fr.
    public func cpuCosetINTTFr(evals: [Fr], shift: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let coeffs = NTTEngine.cpuINTT(evals, logN: logN)
        let shiftInv = frInverse(shift)
        var result = [Fr](repeating: Fr.zero, count: n)
        var sPow = Fr.one
        for i in 0..<n {
            result[i] = frMul(coeffs[i], sPow)
            sPow = frMul(sPow, shiftInv)
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

        var sPow = Fr.one
        for i in 0..<m {
            padded[i] = frMul(padded[i], sPow)
            sPow = frMul(sPow, shift)
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
