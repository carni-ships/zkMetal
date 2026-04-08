// GPUCosetLDEEngine — Fused GPU-accelerated Coset Low-Degree Extension
//
// Extends polynomial evaluations from domain H to coset domain g*H' where
// H' has size blowupFactor * |H|. Critical for STARK trace extension,
// composition polynomial evaluation, and FRI.
//
// Algorithm: INTT(input) -> zero-pad + coset-shift (fused GPU kernel) -> NTT(extended)
//
// Improvements over CosetLDEEngine:
//   - Custom coset shift parameter (not just multiplicative generator)
//   - Fused batch dispatch: all columns in single GPU command buffer
//   - MTLBuffer-level NTT to avoid array copies between GPU passes
//   - Cached coset power buffers keyed by (logM, cosetShift)

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPUCosetLDEEngine

public class GPUCosetLDEEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Fused zero-pad + coset shift kernels
    private let zeroPadCosetShiftFr: MTLComputePipelineState
    private let zeroPadCosetShiftBb: MTLComputePipelineState
    private let batchZeroPadCosetShiftFr: MTLComputePipelineState
    private let batchZeroPadCosetShiftBb: MTLComputePipelineState
    private let zeroPadFr: MTLComputePipelineState
    private let zeroPadBb: MTLComputePipelineState
    private let cosetShiftInplaceFr: MTLComputePipelineState
    private let cosetShiftInplaceBb: MTLComputePipelineState

    // NTT engines (lazily initialized)
    private var frNTTEngine: NTTEngine?
    private var bbNTTEngine: BabyBearNTTEngine?

    // Coset power cache: key = "\(logM)_\(shiftDescription)"
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

        let library = try GPUCosetLDEEngine.compileShaders(device: device)

        guard let zpCosetFr = library.makeFunction(name: "lde_zero_pad_coset_shift_fr"),
              let zpCosetBb = library.makeFunction(name: "lde_zero_pad_coset_shift_bb"),
              let batchZpCosetFr = library.makeFunction(name: "lde_batch_zero_pad_coset_shift_fr"),
              let batchZpCosetBb = library.makeFunction(name: "lde_batch_zero_pad_coset_shift_bb"),
              let zpFr = library.makeFunction(name: "lde_zero_pad_fr"),
              let zpBb = library.makeFunction(name: "lde_zero_pad_bb"),
              let csiFr = library.makeFunction(name: "lde_coset_shift_inplace_fr"),
              let csiBb = library.makeFunction(name: "lde_coset_shift_inplace_bb") else {
            throw MSMError.missingKernel
        }

        self.zeroPadCosetShiftFr = try device.makeComputePipelineState(function: zpCosetFr)
        self.zeroPadCosetShiftBb = try device.makeComputePipelineState(function: zpCosetBb)
        self.batchZeroPadCosetShiftFr = try device.makeComputePipelineState(function: batchZpCosetFr)
        self.batchZeroPadCosetShiftBb = try device.makeComputePipelineState(function: batchZpCosetBb)
        self.zeroPadFr = try device.makeComputePipelineState(function: zpFr)
        self.zeroPadBb = try device.makeComputePipelineState(function: zpBb)
        self.cosetShiftInplaceFr = try device.makeComputePipelineState(function: csiFr)
        self.cosetShiftInplaceBb = try device.makeComputePipelineState(function: csiBb)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let fusedSrc = try String(contentsOfFile: shaderDir + "/ntt/coset_lde_fused.metal", encoding: .utf8)

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

        let combined = clean(fieldFr) + "\n" + clean(fieldBb) + "\n" + clean(fusedSrc)
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

    // MARK: - Coset power precomputation

    /// Precompute powers of coset shift g: [g^0, g^1, ..., g^(m-1)]
    private func getFrCosetPowers(logM: Int, cosetShift: Fr) -> MTLBuffer {
        let key = "\(logM)_\(cosetShift.v.0)_\(cosetShift.v.1)_\(cosetShift.v.2)_\(cosetShift.v.3)_\(cosetShift.v.4)_\(cosetShift.v.5)_\(cosetShift.v.6)_\(cosetShift.v.7)"
        if let cached = frCosetPowersCache[key] { return cached }
        let m = 1 << logM
        var powers = [Fr](repeating: Fr.one, count: m)
        for i in 1..<m {
            powers[i] = frMul(powers[i - 1], cosetShift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Fr>.stride,
                                    options: .storageModeShared)!
        frCosetPowersCache[key] = buf
        return buf
    }

    private func getBbCosetPowers(logM: Int, cosetShift: Bb) -> MTLBuffer {
        let key = "\(logM)_\(cosetShift.v)"
        if let cached = bbCosetPowersCache[key] { return cached }
        let m = 1 << logM
        var powers = [Bb](repeating: Bb.one, count: m)
        for i in 1..<m {
            powers[i] = bbMul(powers[i - 1], cosetShift)
        }
        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<Bb>.stride,
                                    options: .storageModeShared)!
        bbCosetPowersCache[key] = buf
        return buf
    }

    // MARK: - BN254 Fr Coset LDE

    /// GPU-accelerated coset LDE for BN254 Fr.
    /// Input: polynomial evaluations of size N (power of 2) over standard domain.
    /// Output: evaluations over coset domain of size blowupFactor * N.
    /// cosetShift: the coset generator g (coefficients multiplied by g^i before NTT).
    public func extend(evals: [Fr], logN: Int, blowupFactor: Int, cosetShift: Fr) throws -> [Fr] {
        let n = 1 << logN
        precondition(evals.count == n, "Input size must equal 2^logN")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2")

        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        precondition(logM <= Fr.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        let m = 1 << logM

        // CPU path for very small sizes
        if n <= 64 {
            return try cpuExtendFr(evals: evals, logN: logN, blowupFactor: blowupFactor,
                                   cosetShift: cosetShift)
        }

        let engine = try getFrNTTEngine()

        // Step 1: INTT to get coefficients
        let coeffs = try engine.intt(evals)

        // Step 2+3: Fused zero-pad + coset shift on GPU
        let cosetPowers = getFrCosetPowers(logM: logM, cosetShift: cosetShift)

        let inputBuf = device.makeBuffer(
            bytes: coeffs, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let outputBuf = device.makeBuffer(
            length: m * MemoryLayout<Fr>.stride,
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

    // MARK: - BabyBear Coset LDE

    /// GPU-accelerated coset LDE for BabyBear.
    public func extend(evals: [Bb], logN: Int, blowupFactor: Int, cosetShift: Bb) throws -> [Bb] {
        let n = 1 << logN
        precondition(evals.count == n, "Input size must equal 2^logN")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be a power of 2")

        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        precondition(logM <= Bb.TWO_ADICITY, "Extended domain exceeds field's two-adicity")

        let m = 1 << logM

        // CPU path for small sizes
        if n <= 256 {
            return try cpuExtendBb(evals: evals, logN: logN, blowupFactor: blowupFactor,
                                   cosetShift: cosetShift)
        }

        let engine = try getBbNTTEngine()

        // Step 1: INTT to get coefficients
        let coeffs = try engine.intt(evals)

        // Step 2+3: Fused zero-pad + coset shift on GPU
        let cosetPowers = getBbCosetPowers(logM: logM, cosetShift: cosetShift)

        let inputBuf = device.makeBuffer(
            bytes: coeffs, length: n * MemoryLayout<Bb>.stride,
            options: .storageModeShared)!
        let outputBuf = device.makeBuffer(
            length: m * MemoryLayout<Bb>.stride,
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

        // Step 4: Forward NTT of size M
        let ptr = outputBuf.contents().bindMemory(to: Bb.self, capacity: m)
        let shifted = Array(UnsafeBufferPointer(start: ptr, count: m))
        return try engine.ntt(shifted)
    }

    // MARK: - Batch LDE (multiple columns, single fused dispatch)

    /// Batch coset LDE for multiple BN254 Fr columns.
    /// All columns must have the same length N = 2^logN.
    /// Returns array of extended evaluations, one per column.
    public func batchExtend(columns: [[Fr]], logN: Int, blowupFactor: Int,
                            cosetShift: Fr) throws -> [[Fr]] {
        guard !columns.isEmpty else { return [] }
        let n = 1 << logN
        let numCols = columns.count
        for col in columns {
            precondition(col.count == n, "All columns must have size 2^logN")
        }

        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        let engine = try getFrNTTEngine()

        // Step 1: INTT each column
        var allCoeffs = [[Fr]]()
        allCoeffs.reserveCapacity(numCols)
        for col in columns {
            allCoeffs.append(try engine.intt(col))
        }

        // Step 2+3: Batch fused zero-pad + coset shift in single dispatch
        let cosetPowers = getFrCosetPowers(logM: logM, cosetShift: cosetShift)

        var packed = [Fr]()
        packed.reserveCapacity(n * numCols)
        for col in allCoeffs { packed.append(contentsOf: col) }

        let packedInputBuf = device.makeBuffer(
            bytes: &packed, length: n * numCols * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let packedOutputBuf = device.makeBuffer(
            length: m * numCols * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchZeroPadCosetShiftFr)
        enc.setBuffer(packedInputBuf, offset: 0, index: 0)
        enc.setBuffer(packedOutputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        var numColsVal = UInt32(numCols)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        enc.setBytes(&numColsVal, length: 4, index: 5)
        let totalThreads = m * numCols
        let tg = min(256, Int(batchZeroPadCosetShiftFr.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Forward NTT each column
        var results = [[Fr]]()
        results.reserveCapacity(numCols)
        let outPtr = packedOutputBuf.contents().bindMemory(to: Fr.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            results.append(try engine.ntt(colSlice))
        }
        return results
    }

    /// Batch coset LDE for multiple BabyBear columns.
    public func batchExtend(columns: [[Bb]], logN: Int, blowupFactor: Int,
                            cosetShift: Bb) throws -> [[Bb]] {
        guard !columns.isEmpty else { return [] }
        let n = 1 << logN
        let numCols = columns.count
        for col in columns {
            precondition(col.count == n, "All columns must have size 2^logN")
        }

        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        let engine = try getBbNTTEngine()

        // Step 1: INTT each column
        var allCoeffs = [[Bb]]()
        allCoeffs.reserveCapacity(numCols)
        for col in columns {
            allCoeffs.append(try engine.intt(col))
        }

        // Step 2+3: Batch fused zero-pad + coset shift
        let cosetPowers = getBbCosetPowers(logM: logM, cosetShift: cosetShift)

        var packed = [Bb]()
        packed.reserveCapacity(n * numCols)
        for col in allCoeffs { packed.append(contentsOf: col) }

        let packedInputBuf = device.makeBuffer(
            bytes: &packed, length: n * numCols * MemoryLayout<Bb>.stride,
            options: .storageModeShared)!
        let packedOutputBuf = device.makeBuffer(
            length: m * numCols * MemoryLayout<Bb>.stride,
            options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchZeroPadCosetShiftBb)
        enc.setBuffer(packedInputBuf, offset: 0, index: 0)
        enc.setBuffer(packedOutputBuf, offset: 0, index: 1)
        enc.setBuffer(cosetPowers, offset: 0, index: 2)
        var nOrig = UInt32(n)
        var nExt = UInt32(m)
        var numColsVal = UInt32(numCols)
        enc.setBytes(&nOrig, length: 4, index: 3)
        enc.setBytes(&nExt, length: 4, index: 4)
        enc.setBytes(&numColsVal, length: 4, index: 5)
        let totalThreads = m * numCols
        let tg = min(256, Int(batchZeroPadCosetShiftBb.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Step 4: Forward NTT each column
        var results = [[Bb]]()
        results.reserveCapacity(numCols)
        let outPtr = packedOutputBuf.contents().bindMemory(to: Bb.self, capacity: m * numCols)
        for c in 0..<numCols {
            let colSlice = Array(UnsafeBufferPointer(start: outPtr + c * m, count: m))
            results.append(try engine.ntt(colSlice))
        }
        return results
    }

    // MARK: - CPU reference implementations

    /// CPU coset LDE for BN254 Fr (used for small sizes and correctness verification).
    public func cpuExtendFr(evals: [Fr], logN: Int, blowupFactor: Int,
                            cosetShift: Fr) throws -> [Fr] {
        let n = 1 << logN
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        // Step 1: INTT to get coefficients
        let coeffs = NTTEngine.cpuINTT(evals, logN: logN)

        // Step 2: Zero-pad to size M
        var padded = [Fr](repeating: Fr.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        // Step 3: Coset shift: padded[i] *= g^i
        padded.withUnsafeMutableBytes { rBuf in
            withUnsafeBytes(of: cosetShift) { gBuf in
                bn254_fr_batch_mul_powers(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }
        }

        // Step 4: Forward NTT
        return NTTEngine.cpuNTT(padded, logN: logM)
    }

    /// CPU coset LDE for BabyBear.
    public func cpuExtendBb(evals: [Bb], logN: Int, blowupFactor: Int,
                            cosetShift: Bb) throws -> [Bb] {
        let n = 1 << logN
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        let coeffs = BabyBearNTTEngine.cpuINTT(evals, logN: logN)

        var padded = [Bb](repeating: Bb.zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        var gPow = Bb.one
        for i in 0..<m {
            padded[i] = bbMul(padded[i], gPow)
            gPow = bbMul(gPow, cosetShift)
        }

        return BabyBearNTTEngine.cpuNTT(padded, logN: logM)
    }

    // MARK: - Convenience: extend with default coset shift (multiplicative generator)

    /// Extend BN254 Fr evaluations using the field's multiplicative generator as coset shift.
    public func extend(evals: [Fr], logN: Int, blowupFactor: Int) throws -> [Fr] {
        return try extend(evals: evals, logN: logN, blowupFactor: blowupFactor,
                         cosetShift: frFromInt(Fr.GENERATOR))
    }

    /// Extend BabyBear evaluations using the field's multiplicative generator as coset shift.
    public func extend(evals: [Bb], logN: Int, blowupFactor: Int) throws -> [Bb] {
        return try extend(evals: evals, logN: logN, blowupFactor: blowupFactor,
                         cosetShift: Bb(v: Bb.GENERATOR))
    }
}
