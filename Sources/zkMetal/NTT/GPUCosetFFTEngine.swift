// GPUCosetFFTEngine — GPU-accelerated Coset FFT/iFFT for BN254 Fr
//
// Evaluates polynomials on coset domains {g * omega^i} using GPUFFTEngine,
// with support for batch processing and low-degree extension (LDE).
//
// Forward coset FFT: multiply coeffs[i] by g^i, then FFT
// Inverse coset FFT: IFFT, then multiply coeffs[i] by g^(-i)
//
// API:
//   cosetFFT(coeffs:shift:)   — evaluate on coset {shift * omega^i}
//   cosetIFFT(evals:shift:)   — interpolate from coset evaluations
//   batchCosetFFT(columns:shift:) — batch coset FFT (multiple polynomials)
//   cosetLDE(coeffs:blowupFactor:shift:) — low-degree extension via coset
//
// Uses GPUFFTEngine (Stockham auto-sort) for the underlying transforms.

import Foundation
import Metal
import NeonFieldOps

public class GPUCosetFFTEngine {
    public static let version = Versions.gpuCosetFFT

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Coset shift/unshift kernels (reuse from coset_ntt_fused shader)
    private let cosetShiftPowersFr: MTLComputePipelineState
    private let cosetUnshiftPowersFr: MTLComputePipelineState
    // Zero-pad + coset shift (reuse from coset_lde_fused)
    private let zeroPadCosetShiftFr: MTLComputePipelineState

    // FFT engine (lazily initialized)
    private var fftEngine: GPUFFTEngine?

    // Caches: key = "\(logN)_\(shiftKey)"
    private var shiftPowersCache: [String: MTLBuffer] = [:]
    private var invShiftPowersCache: [String: MTLBuffer] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUCosetFFTEngine.compileShaders(device: device)

        guard let cspFr = library.makeFunction(name: "coset_shift_powers_fr"),
              let cupFr = library.makeFunction(name: "coset_unshift_powers_fr"),
              let zpFr = library.makeFunction(name: "lde_zero_pad_coset_shift_fr") else {
            throw MSMError.missingKernel
        }

        self.cosetShiftPowersFr = try device.makeComputePipelineState(function: cspFr)
        self.cosetUnshiftPowersFr = try device.makeComputePipelineState(function: cupFr)
        self.zeroPadCosetShiftFr = try device.makeComputePipelineState(function: zpFr)
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
                .filter { line in
                    if line.contains("#include") || line.contains("#ifndef") || line.contains("#endif") { return false }
                    if line.contains("#define") {
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        let parts = trimmed.split(separator: " ", maxSplits: 3)
                        return parts.count >= 3
                    }
                    return true
                }
                .joined(separator: "\n")
        }

        let combined = clean(fieldFr) + "\n" + clean(fieldBb) + "\n" +
                        clean(fusedSrc) + "\n" + clean(ldeFusedSrc)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - FFT engine accessor

    private func getFFTEngine() throws -> GPUFFTEngine {
        if let e = fftEngine { return e }
        let e = try GPUFFTEngine()
        fftEngine = e
        return e
    }

    // MARK: - Shift power precomputation

    private func shiftKey(_ shift: Fr) -> String {
        "\(shift.v.0)_\(shift.v.1)_\(shift.v.2)_\(shift.v.3)_\(shift.v.4)_\(shift.v.5)_\(shift.v.6)_\(shift.v.7)"
    }

    /// Precompute coset shift powers: powers[i] = shift^i.
    private func getShiftPowers(logN: Int, shift: Fr) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = shiftPowersCache[key] { return cached }
        let n = 1 << logN
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], shift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        shiftPowersCache[key] = buf
        return buf
    }

    /// Precompute inverse shift powers: powers[i] = shift^(-i).
    private func getInvShiftPowers(logN: Int, shift: Fr) -> MTLBuffer {
        let key = "\(logN)_\(shiftKey(shift))"
        if let cached = invShiftPowersCache[key] { return cached }
        let n = 1 << logN
        let shiftInv = frInverse(shift)
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], shiftInv)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        invShiftPowersCache[key] = buf
        return buf
    }

    // MARK: - Forward Coset FFT

    /// Evaluate polynomial on coset {shift * omega^i}.
    /// Algorithm: multiply coeffs[i] by shift^i on GPU, then forward FFT.
    public func cosetFFT(coeffs: [Fr], shift: Fr) throws -> [Fr] {
        let n = coeffs.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        // CPU path for small inputs
        if n <= 64 {
            return cpuCosetFFT(coeffs: coeffs, shift: shift)
        }

        let engine = try getFFTEngine()

        // GPU coset shift
        let powers = getShiftPowers(logN: logN, shift: shift)
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

        // Forward FFT
        return try engine.fft(data: shifted, logN: logN, inverse: false)
    }

    /// Coset FFT with default shift (multiplicative generator).
    public func cosetFFT(coeffs: [Fr]) throws -> [Fr] {
        return try cosetFFT(coeffs: coeffs, shift: frFromInt(Fr.GENERATOR))
    }

    // MARK: - Inverse Coset FFT

    /// Interpolate from coset evaluations back to coefficient form.
    /// Algorithm: inverse FFT, then multiply coeffs[i] by shift^(-i).
    public func cosetIFFT(evals: [Fr], shift: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        let logN = Int(log2(Double(n)))

        if n <= 64 {
            return cpuCosetIFFT(evals: evals, shift: shift)
        }

        let engine = try getFFTEngine()

        // Inverse FFT
        let coeffs = try engine.fft(data: evals, logN: logN, inverse: true)

        // GPU coset unshift
        let invPowers = getInvShiftPowers(logN: logN, shift: shift)
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

    /// Inverse coset FFT with default shift (multiplicative generator).
    public func cosetIFFT(evals: [Fr]) throws -> [Fr] {
        return try cosetIFFT(evals: evals, shift: frFromInt(Fr.GENERATOR))
    }

    // MARK: - Batch Coset FFT

    /// Batch coset FFT for multiple polynomials.
    /// All polynomials must have the same length (power of 2).
    /// Returns array of coset evaluations, one per polynomial.
    public func batchCosetFFT(columns: [[Fr]], shift: Fr) throws -> [[Fr]] {
        guard !columns.isEmpty else { return [] }
        let n = columns[0].count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        for col in columns {
            precondition(col.count == n, "All columns must have same size")
        }
        let logN = Int(log2(Double(n)))

        var results = [[Fr]]()
        results.reserveCapacity(columns.count)
        for col in columns {
            results.append(try cosetFFT(coeffs: col, shift: shift))
        }
        return results
    }

    /// Batch inverse coset FFT for multiple evaluation sets.
    public func batchCosetIFFT(columns: [[Fr]], shift: Fr) throws -> [[Fr]] {
        guard !columns.isEmpty else { return [] }
        let n = columns[0].count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        for col in columns {
            precondition(col.count == n, "All columns must have same size")
        }

        var results = [[Fr]]()
        results.reserveCapacity(columns.count)
        for col in columns {
            results.append(try cosetIFFT(evals: col, shift: shift))
        }
        return results
    }

    // MARK: - Coset LDE (Low-Degree Extension)

    /// Low-degree extension via coset FFT.
    /// Input: polynomial coefficients of size N.
    /// Output: evaluations over coset domain of size blowupFactor * N.
    /// Algorithm: zero-pad to M = blowupFactor*N, multiply by shift powers, then FFT.
    public func cosetLDE(coeffs: [Fr], blowupFactor: Int, shift: Fr) throws -> [Fr] {
        let n = coeffs.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Size must be power of 2")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be power of 2")

        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM
        precondition(logM <= Fr.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        if n <= 64 {
            return cpuCosetLDE(coeffs: coeffs, blowupFactor: blowupFactor, shift: shift)
        }

        let engine = try getFFTEngine()

        // Fused zero-pad + coset shift on GPU
        let cosetPowers = getShiftPowers(logN: logM, shift: shift)

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

        // Forward FFT of size M
        let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.fft(data: shifted, logN: logM, inverse: false)
    }

    /// Coset LDE with default shift (multiplicative generator).
    public func cosetLDE(coeffs: [Fr], blowupFactor: Int) throws -> [Fr] {
        return try cosetLDE(coeffs: coeffs, blowupFactor: blowupFactor,
                           shift: frFromInt(Fr.GENERATOR))
    }

    // MARK: - CPU reference implementations

    /// CPU coset FFT for BN254 Fr.
    public func cpuCosetFFT(coeffs: [Fr], shift: Fr) -> [Fr] {
        let n = coeffs.count
        let logN = Int(log2(Double(n)))
        var shifted = [Fr](repeating: Fr.zero, count: n)
        var s = shift
        coeffs.withUnsafeBytes { cBuf in
            shifted.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: &s) { sBuf in
                    bn254_fr_batch_mul_powers(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return NTTEngine.cpuNTT(shifted, logN: logN)
    }

    /// CPU inverse coset FFT for BN254 Fr.
    public func cpuCosetIFFT(evals: [Fr], shift: Fr) -> [Fr] {
        let n = evals.count
        let logN = Int(log2(Double(n)))
        let coeffs = NTTEngine.cpuINTT(evals, logN: logN)
        var shiftInv = frInverse(shift)
        var result = [Fr](repeating: Fr.zero, count: n)
        coeffs.withUnsafeBytes { cBuf in
            result.withUnsafeMutableBytes { rBuf in
                withUnsafeBytes(of: &shiftInv) { sBuf in
                    bn254_fr_batch_mul_powers(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    /// CPU coset LDE for BN254 Fr.
    public func cpuCosetLDE(coeffs: [Fr], blowupFactor: Int, shift: Fr) -> [Fr] {
        let n = coeffs.count
        let logN = Int(log2(Double(n)))
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        var padded = [Fr](repeating: Fr.zero, count: m)
        padded.withUnsafeMutableBytes { pBuf in
            coeffs.withUnsafeBytes { cBuf in
                memcpy(pBuf.baseAddress!, cBuf.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
        }

        var s = shift
        padded.withUnsafeMutableBytes { pBuf in
            withUnsafeBytes(of: &s) { sBuf in
                bn254_fr_batch_mul_powers(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }
        }

        return NTTEngine.cpuNTT(padded, logN: logM)
    }
}
