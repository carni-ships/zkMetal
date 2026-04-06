// GPUInnerProductEngine — GPU-accelerated multi-scalar inner product for BN254 Fr
//
// Computes inner products (Sigma a_i * b_i) over field elements using Metal
// compute shaders with fused multiply-accumulate and SIMD shuffle reduction.
// Used extensively in IPA, Bulletproofs, sumcheck, and multilinear evaluation.
//
// Public API:
//   fieldInnerProduct(a:b:)           — single inner product
//   batchFieldInnerProduct(pairs:)    — multiple inner products in one dispatch
//   weightedSum(values:weights:)      — alias for inner product (different semantics)
//   multiEqInnerProduct(evals:eq:)    — inner product for sumcheck/MLE evaluation

import Foundation
import Metal

// MARK: - GPUInnerProductEngine

public class GPUInnerProductEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let mulReducePipeline: MTLComputePipelineState
    private let partialReducePipeline: MTLComputePipelineState
    private let batchPipeline: MTLComputePipelineState

    private let threadgroupSize: Int
    private let pool: GPUBufferPool

    /// Arrays smaller than this threshold are computed on CPU.
    public var cpuThreshold: Int = 1024

    /// Pre-staged input buffers for repeated calls (e.g. sumcheck rounds).
    /// Call `stageBuffers(maxSize:)` to pre-allocate.
    private var stagedA: MTLBuffer?
    private var stagedB: MTLBuffer?
    private var stagedMaxCount: Int = 0

    private let library: MTLLibrary

    public init(threadgroupSize: Int = 256) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.threadgroupSize = threadgroupSize
        self.pool = GPUBufferPool(device: device)

        self.library = try GPUInnerProductEngine.compileShaders(device: device)

        guard let mulReduceFn = library.makeFunction(name: "ip_field_mul_reduce"),
              let partialReduceFn = library.makeFunction(name: "ip_field_partial_reduce"),
              let batchFn = library.makeFunction(name: "ip_batch_field") else {
            throw MSMError.missingKernel
        }

        self.mulReducePipeline = try device.makeComputePipelineState(function: mulReduceFn)
        self.partialReducePipeline = try device.makeComputePipelineState(function: partialReduceFn)
        self.batchPipeline = try device.makeComputePipelineState(function: batchFn)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let ipSource = try String(contentsOfFile: shaderDir + "/reduce/inner_product.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanIP = ipSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanIP
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Pre-staged buffers

    /// Pre-allocate reusable GPU buffers for repeated inner product calls
    /// (e.g. sumcheck rounds where the vector size is known ahead of time).
    public func stageBuffers(maxSize: Int) {
        let bytes = maxSize * MemoryLayout<Fr>.stride
        stagedA = pool.allocate(size: bytes)
        stagedB = pool.allocate(size: bytes)
        stagedMaxCount = maxSize
    }

    /// Release pre-staged buffers.
    public func releaseStaged() {
        if let a = stagedA { pool.release(buffer: a) }
        if let b = stagedB { pool.release(buffer: b) }
        stagedA = nil
        stagedB = nil
        stagedMaxCount = 0
    }

    // MARK: - Public API

    /// Compute the field inner product: Sigma a_i * b_i over BN254 Fr.
    public func fieldInnerProduct(a: [Fr], b: [Fr]) -> Fr {
        precondition(a.count == b.count, "Inner product vectors must have equal length")
        let n = a.count
        if n == 0 { return Fr.zero }
        if n == 1 { return frMul(a[0], b[0]) }
        if n < cpuThreshold { return cpuInnerProduct(a, b) }
        return gpuInnerProduct(a, b)
    }

    /// Compute multiple inner products in a single GPU dispatch.
    /// Each pair (a_k, b_k) produces one Fr result.
    public func batchFieldInnerProduct(pairs: [([Fr], [Fr])]) -> [Fr] {
        let count = pairs.count
        if count == 0 { return [] }

        // For very small batches or small vectors, use CPU
        let totalElements = pairs.reduce(0) { $0 + $1.0.count }
        if totalElements < cpuThreshold {
            return pairs.map { cpuInnerProduct($0.0, $0.1) }
        }

        // Check that all vector pairs within each batch have matching lengths
        let maxLen = pairs.reduce(0) { max($0, $1.0.count) }

        // For batch with small individual vectors, use batch kernel
        // For batch with large individual vectors, fall back to sequential GPU
        if maxLen <= threadgroupSize {
            return gpuBatchInnerProduct(pairs: pairs)
        } else {
            // Large vectors: dispatch each individually
            return pairs.map { pair in
                let (a, b) = pair
                if a.count < cpuThreshold { return cpuInnerProduct(a, b) }
                return gpuInnerProduct(a, b)
            }
        }
    }

    /// Weighted sum: Sigma values_i * weights_i — same computation as inner product,
    /// but with semantics matching weighted sums in polynomial evaluation.
    public func weightedSum(values: [Fr], weights: [Fr]) -> Fr {
        return fieldInnerProduct(a: values, b: weights)
    }

    /// Inner product used in sumcheck/MLE evaluation:
    /// Sigma evals_i * eq_i, where eq is the multilinear equality polynomial.
    public func multiEqInnerProduct(evals: [Fr], eq: [Fr]) -> Fr {
        return fieldInnerProduct(a: evals, b: eq)
    }

    // MARK: - GPU dispatch (fused multiply-reduce)

    private func gpuInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        let n = a.count
        let frStride = MemoryLayout<Fr>.stride

        // Upload data to GPU (use pre-staged buffers if available and large enough)
        let aBuf: MTLBuffer
        let bBuf: MTLBuffer
        var releaseBufs = false

        if let sA = stagedA, let sB = stagedB, n <= stagedMaxCount {
            a.withUnsafeBytes { src in memcpy(sA.contents(), src.baseAddress!, n * frStride) }
            b.withUnsafeBytes { src in memcpy(sB.contents(), src.baseAddress!, n * frStride) }
            aBuf = sA
            bBuf = sB
        } else {
            guard let aB = pool.allocate(size: n * frStride),
                  let bB = pool.allocate(size: n * frStride) else {
                return cpuInnerProduct(a, b)
            }
            a.withUnsafeBytes { src in memcpy(aB.contents(), src.baseAddress!, n * frStride) }
            b.withUnsafeBytes { src in memcpy(bB.contents(), src.baseAddress!, n * frStride) }
            aBuf = aB
            bBuf = bB
            releaseBufs = true
        }

        // Pass 1: fused multiply-reduce
        let numGroups = (n + threadgroupSize - 1) / threadgroupSize
        let partialSize = numGroups * frStride

        guard let partialBuf = pool.allocate(size: partialSize) else {
            if releaseBufs { pool.release(buffer: aBuf); pool.release(buffer: bBuf) }
            return cpuInnerProduct(a, b)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            pool.release(buffer: partialBuf)
            if releaseBufs { pool.release(buffer: aBuf); pool.release(buffer: bBuf) }
            return cpuInnerProduct(a, b)
        }

        encoder.setComputePipelineState(mulReducePipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(partialBuf, offset: 0, index: 2)
        var count32 = UInt32(n)
        encoder.setBytes(&count32, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            pool.release(buffer: partialBuf)
            if releaseBufs { pool.release(buffer: aBuf); pool.release(buffer: bBuf) }
            return cpuInnerProduct(a, b)
        }

        // Pass 2+: reduce partials until we have a single result
        let result = reducePartials(partialBuf, count: numGroups)

        pool.release(buffer: partialBuf)
        if releaseBufs { pool.release(buffer: aBuf); pool.release(buffer: bBuf) }
        return result
    }

    /// Multi-pass reduction of partial sums to a single Fr value.
    private func reducePartials(_ inputBuffer: MTLBuffer, count: Int) -> Fr {
        if count == 1 {
            let ptr = inputBuffer.contents().bindMemory(to: Fr.self, capacity: 1)
            return ptr[0]
        }

        let frStride = MemoryLayout<Fr>.stride
        var currentBuf = inputBuffer
        var currentCount = count
        var intermediateBuffers: [MTLBuffer] = []

        while currentCount > 1 {
            let numGroups = (currentCount + threadgroupSize - 1) / threadgroupSize
            let outputSize = numGroups * frStride

            guard let outputBuf = pool.allocate(size: outputSize) else { break }

            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else {
                pool.release(buffer: outputBuf)
                break
            }

            encoder.setComputePipelineState(partialReducePipeline)
            encoder.setBuffer(currentBuf, offset: 0, index: 0)
            encoder.setBuffer(outputBuf, offset: 0, index: 1)
            var c32 = UInt32(currentCount)
            encoder.setBytes(&c32, length: 4, index: 2)
            encoder.dispatchThreadgroups(
                MTLSize(width: numGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
            )
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if cmdBuf.error != nil {
                pool.release(buffer: outputBuf)
                break
            }

            intermediateBuffers.append(outputBuf)
            currentBuf = outputBuf
            currentCount = numGroups
        }

        let ptr = currentBuf.contents().bindMemory(to: Fr.self, capacity: 1)
        let result = ptr[0]
        for buf in intermediateBuffers { pool.release(buffer: buf) }
        return result
    }

    // MARK: - Batch GPU dispatch

    private func gpuBatchInnerProduct(pairs: [([Fr], [Fr])]) -> [Fr] {
        let batchCount = pairs.count
        let frStride = MemoryLayout<Fr>.stride

        // Build concatenated a_data, b_data, offsets, lengths
        var totalLen = 0
        var offsets = [UInt32]()
        var lengths = [UInt32]()
        offsets.reserveCapacity(batchCount)
        lengths.reserveCapacity(batchCount)

        for (a, b) in pairs {
            precondition(a.count == b.count, "Inner product vectors must have equal length")
            offsets.append(UInt32(totalLen))
            lengths.append(UInt32(a.count))
            totalLen += a.count
        }

        // Concatenate all a and b vectors
        var aData = [Fr]()
        var bData = [Fr]()
        aData.reserveCapacity(totalLen)
        bData.reserveCapacity(totalLen)
        for (a, b) in pairs {
            aData.append(contentsOf: a)
            bData.append(contentsOf: b)
        }

        // Allocate GPU buffers
        guard let aBuf = pool.allocate(size: totalLen * frStride),
              let bBuf = pool.allocate(size: totalLen * frStride),
              let outBuf = pool.allocate(size: batchCount * frStride),
              let offBuf = pool.allocate(size: batchCount * MemoryLayout<UInt32>.stride),
              let lenBuf = pool.allocate(size: batchCount * MemoryLayout<UInt32>.stride) else {
            return pairs.map { cpuInnerProduct($0.0, $0.1) }
        }

        // Upload data
        aData.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, totalLen * frStride) }
        bData.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, totalLen * frStride) }
        offsets.withUnsafeBytes { src in memcpy(offBuf.contents(), src.baseAddress!, batchCount * MemoryLayout<UInt32>.stride) }
        lengths.withUnsafeBytes { src in memcpy(lenBuf.contents(), src.baseAddress!, batchCount * MemoryLayout<UInt32>.stride) }

        // Dispatch: one threadgroup per batch entry
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            pool.release(buffer: aBuf); pool.release(buffer: bBuf)
            pool.release(buffer: outBuf); pool.release(buffer: offBuf); pool.release(buffer: lenBuf)
            return pairs.map { cpuInnerProduct($0.0, $0.1) }
        }

        encoder.setComputePipelineState(batchPipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(outBuf, offset: 0, index: 2)
        encoder.setBuffer(offBuf, offset: 0, index: 3)
        encoder.setBuffer(lenBuf, offset: 0, index: 4)

        encoder.dispatchThreadgroups(
            MTLSize(width: batchCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let results: [Fr]
        if cmdBuf.error != nil {
            results = pairs.map { cpuInnerProduct($0.0, $0.1) }
        } else {
            let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: batchCount)
            results = Array(UnsafeBufferPointer(start: ptr, count: batchCount))
        }

        pool.release(buffer: aBuf); pool.release(buffer: bBuf)
        pool.release(buffer: outBuf); pool.release(buffer: offBuf); pool.release(buffer: lenBuf)
        return results
    }

    // MARK: - CPU fallback

    private func cpuInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }
}
