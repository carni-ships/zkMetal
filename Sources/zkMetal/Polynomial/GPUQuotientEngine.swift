// GPU-accelerated vanishing polynomial quotient engine
//
// Computes the quotient polynomial q(x) = p(x) / z_H(x) where z_H(x) = x^n - 1.
// This is a core STARK/Plonk operation: after evaluating constraint polynomials
// over a coset domain, the prover must divide by the vanishing polynomial.
//
// Pipeline:
//   1. Precompute coset points: g * omega^i for i in [0, domainSize)
//   2. Evaluate Z_H at each coset point (or use constant Z_H on cosets)
//   3. Batch-invert Z_H values using Montgomery's trick (1 inversion per chunk)
//   4. Element-wise multiply: q[i] = p[i] * zhInv[i]
//
// Also provides:
//   - Fused single-pass quotient (inline Z_H computation + Fermat inversion per thread)
//   - splitQuotient: split degree-bounded chunks for FRI commitment
//   - Precomputed vanishing inverse caching for repeated proving
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit Barrett).
// CPU fallback for small domains.

import Foundation
import Metal

// MARK: - GPUQuotientEngine

public class GPUQuotientEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Fused quotient kernels (single-pass: eval + invert + multiply per thread)
    private let fusedBn254: MTLComputePipelineState
    private let fusedBb: MTLComputePipelineState

    // Two-pass quotient: precomputed vanishing inverses -> element-wise multiply
    private let precomputedBn254: MTLComputePipelineState
    private let precomputedBb: MTLComputePipelineState

    // Z_H evaluation kernels
    private let zhEvalBn254: MTLComputePipelineState
    private let zhEvalBb: MTLComputePipelineState

    // Batch inverse kernels
    private let batchInvBn254: MTLComputePipelineState
    private let batchInvBb: MTLComputePipelineState

    // Chunk extraction kernels
    private let extractChunkBn254: MTLComputePipelineState
    private let extractChunkBb: MTLComputePipelineState

    /// CPU fallback threshold (domain size)
    private let cpuThreshold = 512

    /// Cache for precomputed vanishing inverses, keyed by (logDomain, logTrace, cosetGen hash)
    private var vanishingInverseCache: [String: MTLBuffer] = [:]

    /// Cache for coset point buffers
    private var cosetPointCache: [String: MTLBuffer] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUQuotientEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "quotient_fused_bn254"),
              let fn2 = library.makeFunction(name: "quotient_fused_babybear"),
              let fn3 = library.makeFunction(name: "quotient_precomputed_bn254"),
              let fn4 = library.makeFunction(name: "quotient_precomputed_babybear"),
              let fn5 = library.makeFunction(name: "quotient_zh_eval_bn254"),
              let fn6 = library.makeFunction(name: "quotient_zh_eval_babybear"),
              let fn7 = library.makeFunction(name: "quotient_batch_inverse_bn254"),
              let fn8 = library.makeFunction(name: "quotient_batch_inverse_babybear"),
              let fn9 = library.makeFunction(name: "quotient_extract_chunk_bn254"),
              let fn10 = library.makeFunction(name: "quotient_extract_chunk_babybear") else {
            throw MSMError.missingKernel
        }

        self.fusedBn254 = try device.makeComputePipelineState(function: fn1)
        self.fusedBb = try device.makeComputePipelineState(function: fn2)
        self.precomputedBn254 = try device.makeComputePipelineState(function: fn3)
        self.precomputedBb = try device.makeComputePipelineState(function: fn4)
        self.zhEvalBn254 = try device.makeComputePipelineState(function: fn5)
        self.zhEvalBb = try device.makeComputePipelineState(function: fn6)
        self.batchInvBn254 = try device.makeComputePipelineState(function: fn7)
        self.batchInvBb = try device.makeComputePipelineState(function: fn8)
        self.extractChunkBn254 = try device.makeComputePipelineState(function: fn9)
        self.extractChunkBb = try device.makeComputePipelineState(function: fn10)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let qeSource = try String(contentsOfFile: shaderDir + "/poly/quotient_engine.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define BABYBEAR") && !$0.contains("#define BN254") &&
                         !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(frSource) + "\n" + clean(bbSource) + "\n" + clean(qeSource)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Element size

    private func elementSize(for field: FieldType) -> Int {
        switch field {
        case .bn254: return 32
        case .babybear: return 4
        case .goldilocks: return 8
        }
    }

    // MARK: - Compute Quotient (fused single-pass)

    /// Compute quotient q[i] = constraintEvals[i] / Z_H(cosetPoint[i]) in a single GPU pass.
    ///
    /// Each thread computes Z_H inline via repeated squaring + Fermat inversion.
    /// Best for one-shot quotient computation where vanishing inverses are not reused.
    ///
    /// - Parameters:
    ///   - constraintEvals: MTLBuffer of constraint polynomial evaluations over coset
    ///   - cosetPoints: MTLBuffer of coset evaluation points g * omega^i
    ///   - domainSize: number of evaluation points
    ///   - logTraceLen: log2(trace length n), where Z_H(x) = x^n - 1
    ///   - field: .bn254 or .babybear
    /// - Returns: MTLBuffer containing quotient evaluations
    public func computeQuotientFused(constraintEvals: MTLBuffer, cosetPoints: MTLBuffer,
                                      domainSize: Int, logTraceLen: Int,
                                      field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)

        guard constraintEvals.length >= domainSize * elemSize,
              cosetPoints.length >= domainSize * elemSize else {
            throw MSMError.invalidInput
        }

        // CPU fallback for small domains
        if domainSize < cpuThreshold {
            return try cpuComputeQuotient(constraintEvals: constraintEvals, cosetPoints: cosetPoints,
                                           domainSize: domainSize, logTraceLen: logTraceLen, field: field)
        }

        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var domainU32 = UInt32(domainSize)
        var logTrace = UInt32(logTraceLen)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? fusedBn254 : fusedBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(constraintEvals, offset: 0, index: 0)
        enc.setBuffer(cosetPoints, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&domainU32, length: 4, index: 3)
        enc.setBytes(&logTrace, length: 4, index: 4)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outBuf
    }

    // MARK: - Compute Quotient (with precomputed vanishing inverses)

    /// Compute quotient using precomputed vanishing polynomial inverses.
    ///
    /// This is a simple element-wise multiply: out[i] = constraintEvals[i] * vanishingInverses[i].
    /// Use when proving multiple statements over the same domain (cache vanishing inverses).
    ///
    /// - Parameters:
    ///   - constraintEvals: MTLBuffer of constraint polynomial evaluations
    ///   - vanishingInverses: MTLBuffer of precomputed 1/Z_H(cosetPoint_i) values
    ///   - domainSize: number of elements
    ///   - field: .bn254 or .babybear
    /// - Returns: MTLBuffer containing quotient evaluations
    public func computeQuotient(constraintEvals: MTLBuffer, vanishingInverses: MTLBuffer,
                                 domainSize: Int, field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)

        guard constraintEvals.length >= domainSize * elemSize,
              vanishingInverses.length >= domainSize * elemSize else {
            throw MSMError.invalidInput
        }

        // CPU fallback
        if domainSize < cpuThreshold {
            return cpuComputeQuotientPrecomputed(constraintEvals: constraintEvals,
                                                  vanishingInverses: vanishingInverses,
                                                  domainSize: domainSize, field: field)
        }

        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var n = UInt32(domainSize)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        let pipeline = field == .bn254 ? precomputedBn254 : precomputedBb
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(constraintEvals, offset: 0, index: 0)
        enc.setBuffer(vanishingInverses, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outBuf
    }

    // MARK: - Precompute Vanishing Inverses

    /// Precompute 1/Z_H(cosetPoint_i) for a domain, suitable for repeated use.
    ///
    /// Pipeline:
    ///   1. Compute coset points g * omega^i
    ///   2. Evaluate Z_H at each point on GPU
    ///   3. Batch-invert via Montgomery's trick
    ///
    /// Results are cached internally for the (logDomain, logTrace, cosetGen) triple.
    ///
    /// - Parameters:
    ///   - logDomain: log2(evaluation domain size N)
    ///   - logTraceLen: log2(trace length n), Z_H(x) = x^n - 1
    ///   - cosetGen: coset generator g (in Montgomery form for BN254, raw for BabyBear)
    ///   - field: .bn254 or .babybear
    /// - Returns: MTLBuffer containing 1/Z_H(g * omega^i) for each i
    public func precomputeVanishingInverses(logDomain: Int, logTraceLen: Int,
                                             cosetGen: [UInt32],
                                             field: FieldType) throws -> MTLBuffer {
        let cacheKey = "\(logDomain)_\(logTraceLen)_\(cosetGen.hashValue)_\(field)"
        if let cached = vanishingInverseCache[cacheKey] {
            return cached
        }

        let domainSize = 1 << logDomain
        let elemSize = elementSize(for: field)

        // Step 1: Compute coset points
        let cosetPointsBuf = try getOrBuildCosetPoints(logDomain: logDomain, cosetGen: cosetGen, field: field)

        // CPU path for small domains
        if domainSize < cpuThreshold {
            let result = try cpuPrecomputeVanishingInverses(cosetPoints: cosetPointsBuf,
                                                             domainSize: domainSize,
                                                             logTraceLen: logTraceLen, field: field)
            vanishingInverseCache[cacheKey] = result
            return result
        }

        // Step 2: Evaluate Z_H on GPU
        let zhBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        // Step 3: Batch inverse on GPU
        let zhInvBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var domainU32 = UInt32(domainSize)
        var logTrace = UInt32(logTraceLen)

        // Encode Z_H evaluation
        let enc1 = cmdBuf.makeComputeCommandEncoder()!
        let zhPipeline = field == .bn254 ? zhEvalBn254 : zhEvalBb
        enc1.setComputePipelineState(zhPipeline)
        enc1.setBuffer(cosetPointsBuf, offset: 0, index: 0)
        enc1.setBuffer(zhBuf, offset: 0, index: 1)
        enc1.setBytes(&domainU32, length: 4, index: 2)
        enc1.setBytes(&logTrace, length: 4, index: 3)
        let tg1 = min(256, Int(zhPipeline.maxTotalThreadsPerThreadgroup))
        enc1.dispatchThreads(MTLSize(width: domainSize, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
        enc1.endEncoding()

        // Encode batch inverse
        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        let invPipeline = field == .bn254 ? batchInvBn254 : batchInvBb
        let chunkSize = field == .bn254 ? 256 : 1024
        let numChunks = (domainSize + chunkSize - 1) / chunkSize
        enc2.setComputePipelineState(invPipeline)
        enc2.setBuffer(zhBuf, offset: 0, index: 0)
        enc2.setBuffer(zhInvBuf, offset: 0, index: 1)
        enc2.setBytes(&domainU32, length: 4, index: 2)
        enc2.dispatchThreadgroups(MTLSize(width: numChunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc2.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        vanishingInverseCache[cacheKey] = zhInvBuf
        return zhInvBuf
    }

    // MARK: - Split Quotient into Degree-Bounded Chunks

    /// Split quotient polynomial (in coefficient form) into degree-bounded chunks for FRI.
    ///
    /// Given quotient coefficients [c_0, c_1, ..., c_{kN-1}], splits into k chunks:
    ///   chunk_j = [c_{j}, c_{j+k}, c_{j+2k}, ...] for j in [0, numChunks)
    ///
    /// So q(x) = chunk_0(x^k) + x * chunk_1(x^k) + ... + x^{k-1} * chunk_{k-1}(x^k)
    ///
    /// This is the standard coefficient interleaving used in Plonk/STARK provers
    /// to produce degree-bounded polynomials for FRI or KZG commitment.
    ///
    /// - Parameters:
    ///   - quotientCoeffs: coefficient array (ascending order)
    ///   - numChunks: number of chunks to split into
    ///   - field: .bn254 or .babybear
    /// - Returns: array of numChunks coefficient arrays, each of length quotientCoeffs.count / numChunks
    public func splitQuotient(quotientCoeffs: [UInt32], numChunks: Int,
                               field: FieldType) -> [[UInt32]] {
        let elemWords = elementSize(for: field) / 4
        let totalElems = quotientCoeffs.count / elemWords

        // Pad to multiple of numChunks if needed
        let paddedElems = ((totalElems + numChunks - 1) / numChunks) * numChunks
        var padded = quotientCoeffs
        if paddedElems > totalElems {
            padded.append(contentsOf: [UInt32](repeating: 0, count: (paddedElems - totalElems) * elemWords))
        }

        let chunkElems = paddedElems / numChunks
        var chunks = [[UInt32]]()
        chunks.reserveCapacity(numChunks)

        for j in 0..<numChunks {
            var chunk = [UInt32]()
            chunk.reserveCapacity(chunkElems * elemWords)
            for i in 0..<chunkElems {
                let srcIdx = (i * numChunks + j) * elemWords
                for w in 0..<elemWords {
                    chunk.append(padded[srcIdx + w])
                }
            }
            chunks.append(chunk)
        }

        return chunks
    }

    /// Split quotient in MTLBuffer form on GPU.
    /// Returns array of MTLBuffers, one per chunk.
    public func splitQuotientGPU(quotientBuf: MTLBuffer, totalElems: Int,
                                  numChunks: Int, field: FieldType) throws -> [MTLBuffer] {
        let elemSize = elementSize(for: field)
        let chunkElems = (totalElems + numChunks - 1) / numChunks

        guard quotientBuf.length >= totalElems * elemSize else {
            throw MSMError.invalidInput
        }

        // For small inputs, do CPU extraction
        if totalElems < cpuThreshold {
            let elemWords = elemSize / 4
            let ptr = quotientBuf.contents().bindMemory(to: UInt32.self, capacity: totalElems * elemWords)
            let data = Array(UnsafeBufferPointer(start: ptr, count: totalElems * elemWords))
            let chunks = splitQuotient(quotientCoeffs: data, numChunks: numChunks, field: field)
            return chunks.map { chunk in
                device.makeBuffer(bytes: chunk, length: chunk.count * 4, options: .storageModeShared)!
            }
        }

        var chunkBufs = [MTLBuffer]()
        chunkBufs.reserveCapacity(numChunks)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        for j in 0..<numChunks {
            let buf = device.makeBuffer(length: chunkElems * elemSize, options: .storageModeShared)!
            chunkBufs.append(buf)

            let enc = cmdBuf.makeComputeCommandEncoder()!
            let pipeline = field == .bn254 ? extractChunkBn254 : extractChunkBb
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(quotientBuf, offset: 0, index: 0)
            enc.setBuffer(buf, offset: 0, index: 1)
            var totalU32 = UInt32(totalElems)
            var chunksU32 = UInt32(numChunks)
            var idxU32 = UInt32(j)
            enc.setBytes(&totalU32, length: 4, index: 2)
            enc.setBytes(&chunksU32, length: 4, index: 3)
            enc.setBytes(&idxU32, length: 4, index: 4)
            let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: chunkElems, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return chunkBufs
    }

    // MARK: - Coset Point Management

    /// Get or build coset points buffer: g * omega^i for i in [0, 2^logDomain)
    public func getOrBuildCosetPoints(logDomain: Int, cosetGen: [UInt32],
                                       field: FieldType) throws -> MTLBuffer {
        let cacheKey = "coset_\(logDomain)_\(cosetGen.hashValue)_\(field)"
        if let cached = cosetPointCache[cacheKey] {
            return cached
        }

        let domainSize = 1 << logDomain
        let elemSize = elementSize(for: field)
        let buf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let g = Fr(v: (cosetGen[0], cosetGen[1], cosetGen[2], cosetGen[3],
                           cosetGen[4], cosetGen[5], cosetGen[6], cosetGen[7]))
            let omega = frRootOfUnity(logN: logDomain)
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            var current = g
            for i in 0..<domainSize {
                let base = i * 8
                ptr[base]   = current.v.0; ptr[base+1] = current.v.1
                ptr[base+2] = current.v.2; ptr[base+3] = current.v.3
                ptr[base+4] = current.v.4; ptr[base+5] = current.v.5
                ptr[base+6] = current.v.6; ptr[base+7] = current.v.7
                current = frMul(current, omega)
            }

        case .babybear:
            let g = Bb(v: cosetGen[0])
            let omega = bbRootOfUnity(logN: logDomain)
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            var current = g
            for i in 0..<domainSize {
                ptr[i] = current.v
                current = bbMul(current, omega)
            }

        case .goldilocks:
            throw MSMError.invalidInput
        }

        cosetPointCache[cacheKey] = buf
        return buf
    }

    // MARK: - Clear caches

    /// Clear all internal caches (vanishing inverses and coset points).
    public func clearCaches() {
        vanishingInverseCache.removeAll()
        cosetPointCache.removeAll()
    }

    // MARK: - CPU Fallback: Fused Quotient

    private func cpuComputeQuotient(constraintEvals: MTLBuffer, cosetPoints: MTLBuffer,
                                     domainSize: Int, logTraceLen: Int,
                                     field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let inPtr = constraintEvals.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let cpPtr = cosetPoints.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)

            for i in 0..<domainSize {
                let base = i * 8
                let eval = Fr(v: (inPtr[base], inPtr[base+1], inPtr[base+2], inPtr[base+3],
                                  inPtr[base+4], inPtr[base+5], inPtr[base+6], inPtr[base+7]))
                var x = Fr(v: (cpPtr[base], cpPtr[base+1], cpPtr[base+2], cpPtr[base+3],
                               cpPtr[base+4], cpPtr[base+5], cpPtr[base+6], cpPtr[base+7]))
                for _ in 0..<logTraceLen {
                    x = frSqr(x)
                }
                let zh = frSub(x, Fr.one)
                let zhInv = frInverse(zh)
                let result = frMul(eval, zhInv)
                outPtr[base]   = result.v.0; outPtr[base+1] = result.v.1
                outPtr[base+2] = result.v.2; outPtr[base+3] = result.v.3
                outPtr[base+4] = result.v.4; outPtr[base+5] = result.v.5
                outPtr[base+6] = result.v.6; outPtr[base+7] = result.v.7
            }

        case .babybear:
            let inPtr = constraintEvals.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let cpPtr = cosetPoints.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize)

            for i in 0..<domainSize {
                let eval = Bb(v: inPtr[i])
                var x = Bb(v: cpPtr[i])
                for _ in 0..<logTraceLen {
                    x = bbSqr(x)
                }
                let zh = bbSub(x, Bb.one)
                let zhInv = bbInverse(zh)
                outPtr[i] = bbMul(eval, zhInv).v
            }

        case .goldilocks:
            throw MSMError.invalidInput
        }

        return outBuf
    }

    // MARK: - CPU Fallback: Precomputed Quotient

    private func cpuComputeQuotientPrecomputed(constraintEvals: MTLBuffer,
                                                vanishingInverses: MTLBuffer,
                                                domainSize: Int,
                                                field: FieldType) -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let inPtr = constraintEvals.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let invPtr = vanishingInverses.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)

            for i in 0..<domainSize {
                let base = i * 8
                let eval = Fr(v: (inPtr[base], inPtr[base+1], inPtr[base+2], inPtr[base+3],
                                  inPtr[base+4], inPtr[base+5], inPtr[base+6], inPtr[base+7]))
                let inv = Fr(v: (invPtr[base], invPtr[base+1], invPtr[base+2], invPtr[base+3],
                                 invPtr[base+4], invPtr[base+5], invPtr[base+6], invPtr[base+7]))
                let result = frMul(eval, inv)
                outPtr[base]   = result.v.0; outPtr[base+1] = result.v.1
                outPtr[base+2] = result.v.2; outPtr[base+3] = result.v.3
                outPtr[base+4] = result.v.4; outPtr[base+5] = result.v.5
                outPtr[base+6] = result.v.6; outPtr[base+7] = result.v.7
            }

        case .babybear:
            let inPtr = constraintEvals.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let invPtr = vanishingInverses.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize)

            for i in 0..<domainSize {
                outPtr[i] = bbMul(Bb(v: inPtr[i]), Bb(v: invPtr[i])).v
            }

        case .goldilocks:
            fatalError("Goldilocks not supported")
        }

        return outBuf
    }

    // MARK: - CPU Fallback: Precompute Vanishing Inverses

    private func cpuPrecomputeVanishingInverses(cosetPoints: MTLBuffer, domainSize: Int,
                                                  logTraceLen: Int,
                                                  field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let outBuf = device.makeBuffer(length: domainSize * elemSize, options: .storageModeShared)!

        switch field {
        case .bn254:
            let cpPtr = cosetPoints.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize * 8)

            for i in 0..<domainSize {
                let base = i * 8
                var x = Fr(v: (cpPtr[base], cpPtr[base+1], cpPtr[base+2], cpPtr[base+3],
                               cpPtr[base+4], cpPtr[base+5], cpPtr[base+6], cpPtr[base+7]))
                for _ in 0..<logTraceLen {
                    x = frSqr(x)
                }
                let zh = frSub(x, Fr.one)
                let inv = frInverse(zh)
                outPtr[base]   = inv.v.0; outPtr[base+1] = inv.v.1
                outPtr[base+2] = inv.v.2; outPtr[base+3] = inv.v.3
                outPtr[base+4] = inv.v.4; outPtr[base+5] = inv.v.5
                outPtr[base+6] = inv.v.6; outPtr[base+7] = inv.v.7
            }

        case .babybear:
            let cpPtr = cosetPoints.contents().bindMemory(to: UInt32.self, capacity: domainSize)
            let outPtr = outBuf.contents().bindMemory(to: UInt32.self, capacity: domainSize)

            for i in 0..<domainSize {
                var x = Bb(v: cpPtr[i])
                for _ in 0..<logTraceLen {
                    x = bbSqr(x)
                }
                let zh = bbSub(x, Bb.one)
                outPtr[i] = bbInverse(zh).v
            }

        case .goldilocks:
            throw MSMError.invalidInput
        }

        return outBuf
    }
}
