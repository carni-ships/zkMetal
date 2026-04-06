// GPURLCEngine — GPU-accelerated random linear combination for BN254 Fr
//
// Computes weighted sums of k vectors: result[i] = sum(j=0..k-1, vectors[j][i] * powers[j])
// Used in batch polynomial commitments, constraint evaluation with alpha powers,
// Nova/ProtoGalaxy folding, and any batched polynomial protocol.
//
// Public API:
//   combine(vectors:powers:)         — weighted sum of k Fr arrays
//   combineBuffers(buffers:powers:n:) — weighted sum of k MTLBuffers (zero-copy GPU path)
//   alphaPowers(alpha:count:)         — precompute [1, alpha, alpha^2, ...]
//   batchCommitCombine(polys:alpha:)  — combine polynomials for batch PCS opening

import Foundation
import Metal

// MARK: - GPURLCEngine

public class GPURLCEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let combineStridedPipeline: MTLComputePipelineState
    private let combinePtrsPipeline: MTLComputePipelineState
    private let alphaPowersPipeline: MTLComputePipelineState

    private let threadgroupSize: Int
    private let pool: GPUBufferPool

    /// Arrays smaller than this threshold are computed on CPU.
    public var cpuThreshold: Int = 1024

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

        self.library = try GPURLCEngine.compileShaders(device: device)

        guard let combineStridedFn = library.makeFunction(name: "rlc_combine_strided"),
              let combinePtrsFn = library.makeFunction(name: "rlc_combine_ptrs"),
              let alphaPowersFn = library.makeFunction(name: "rlc_alpha_powers") else {
            throw MSMError.missingKernel
        }

        self.combineStridedPipeline = try device.makeComputePipelineState(function: combineStridedFn)
        self.combinePtrsPipeline = try device.makeComputePipelineState(function: combinePtrsFn)
        self.alphaPowersPipeline = try device.makeComputePipelineState(function: alphaPowersFn)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let rlcSource = try String(contentsOfFile: shaderDir + "/reduce/random_linear_combination.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanRLC = rlcSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanRLC
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Public API

    /// Compute the random linear combination of k vectors with given weights.
    /// result[i] = sum(j=0..k-1, vectors[j][i] * powers[j])
    ///
    /// All vectors must have the same length. `powers` must have length >= k.
    public func combine(vectors: [[Fr]], powers: [Fr]) -> [Fr] {
        precondition(!vectors.isEmpty, "RLC requires at least one vector")
        let k = vectors.count
        let n = vectors[0].count
        precondition(powers.count >= k, "Need at least \(k) powers, got \(powers.count)")
        for j in 1..<k {
            precondition(vectors[j].count == n, "All vectors must have equal length")
        }

        if n == 0 { return [] }

        // Single vector: just scalar multiply
        if k == 1 {
            let w = powers[0]
            return vectors[0].map { frMul($0, w) }
        }

        if n < cpuThreshold {
            return cpuCombine(vectors: vectors, powers: powers)
        }

        return gpuCombineStrided(vectors: vectors, powers: powers)
    }

    /// Compute random linear combination directly from MTLBuffers (zero-copy GPU path).
    /// Each buffer must contain at least `n` Fr elements.
    /// Returns an MTLBuffer with `n` Fr results. Caller is responsible for the returned buffer.
    public func combineBuffers(buffers: [MTLBuffer], powers: [Fr], n: Int) -> MTLBuffer? {
        let k = buffers.count
        precondition(k > 0 && powers.count >= k)
        if n == 0 { return nil }

        let frStride = MemoryLayout<Fr>.stride

        // Concatenate buffer data into one buffer with offsets
        var totalSize = 0
        var offsets = [UInt32]()
        offsets.reserveCapacity(k)
        for j in 0..<k {
            offsets.append(UInt32(totalSize / frStride))
            totalSize += n * frStride
        }

        guard let concatBuf = pool.allocate(size: totalSize) else { return nil }
        for j in 0..<k {
            memcpy(concatBuf.contents() + j * n * frStride,
                   buffers[j].contents(), n * frStride)
        }

        guard let offsetBuf = pool.allocate(size: k * MemoryLayout<UInt32>.stride),
              let powerBuf = pool.allocate(size: k * frStride),
              let outputBuf = pool.allocate(size: n * frStride) else {
            pool.release(buffer: concatBuf)
            return nil
        }

        offsets.withUnsafeBytes { src in memcpy(offsetBuf.contents(), src.baseAddress!, k * MemoryLayout<UInt32>.stride) }
        powers.withUnsafeBytes { src in memcpy(powerBuf.contents(), src.baseAddress!, k * frStride) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            pool.release(buffer: concatBuf)
            pool.release(buffer: offsetBuf)
            pool.release(buffer: powerBuf)
            pool.release(buffer: outputBuf)
            return nil
        }

        encoder.setComputePipelineState(combinePtrsPipeline)
        encoder.setBuffer(concatBuf, offset: 0, index: 0)
        encoder.setBuffer(offsetBuf, offset: 0, index: 1)
        encoder.setBuffer(powerBuf, offset: 0, index: 2)
        encoder.setBuffer(outputBuf, offset: 0, index: 3)
        var n32 = UInt32(n)
        var k32 = UInt32(k)
        encoder.setBytes(&n32, length: 4, index: 4)
        encoder.setBytes(&k32, length: 4, index: 5)

        let numGroups = (n + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        pool.release(buffer: concatBuf)
        pool.release(buffer: offsetBuf)
        pool.release(buffer: powerBuf)

        if cmdBuf.error != nil {
            pool.release(buffer: outputBuf)
            return nil
        }

        return outputBuf
    }

    /// Precompute alpha powers: [1, alpha, alpha^2, ..., alpha^(count-1)]
    public func alphaPowers(alpha: Fr, count: Int) -> [Fr] {
        if count == 0 { return [] }
        if count == 1 { return [Fr.one] }

        // For small counts, CPU is faster than GPU dispatch overhead
        if count < 256 {
            return cpuAlphaPowers(alpha: alpha, count: count)
        }

        return gpuAlphaPowers(alpha: alpha, count: count)
    }

    /// Combine polynomials for batch PCS opening:
    /// result = sum(j=0..k-1, polys[j] * alpha^j)
    ///
    /// Equivalent to combine(vectors: polys, powers: alphaPowers(alpha, polys.count))
    public func batchCommitCombine(polys: [[Fr]], alpha: Fr) -> [Fr] {
        let k = polys.count
        if k == 0 { return [] }
        if k == 1 { return polys[0] }

        let powers = alphaPowers(alpha: alpha, count: k)
        return combine(vectors: polys, powers: powers)
    }

    // MARK: - GPU dispatch (strided layout)

    private func gpuCombineStrided(vectors: [[Fr]], powers: [Fr]) -> [Fr] {
        let k = vectors.count
        let n = vectors[0].count
        let frStride = MemoryLayout<Fr>.stride

        // Pack vectors into row-major layout: vectors[j][i] -> packed[j * n + i]
        let totalElements = k * n
        var packed = [Fr]()
        packed.reserveCapacity(totalElements)
        for j in 0..<k {
            packed.append(contentsOf: vectors[j])
        }

        guard let vectorBuf = pool.allocate(size: totalElements * frStride),
              let powerBuf = pool.allocate(size: k * frStride),
              let outputBuf = pool.allocate(size: n * frStride) else {
            return cpuCombine(vectors: vectors, powers: powers)
        }

        packed.withUnsafeBytes { src in memcpy(vectorBuf.contents(), src.baseAddress!, totalElements * frStride) }
        powers.withUnsafeBytes { src in memcpy(powerBuf.contents(), src.baseAddress!, k * frStride) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            pool.release(buffer: vectorBuf)
            pool.release(buffer: powerBuf)
            pool.release(buffer: outputBuf)
            return cpuCombine(vectors: vectors, powers: powers)
        }

        encoder.setComputePipelineState(combineStridedPipeline)
        encoder.setBuffer(vectorBuf, offset: 0, index: 0)
        encoder.setBuffer(powerBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)
        var n32 = UInt32(n)
        var k32 = UInt32(k)
        encoder.setBytes(&n32, length: 4, index: 3)
        encoder.setBytes(&k32, length: 4, index: 4)

        let numGroups = (n + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let results: [Fr]
        if cmdBuf.error != nil {
            results = cpuCombine(vectors: vectors, powers: powers)
        } else {
            let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: n)
            results = Array(UnsafeBufferPointer(start: ptr, count: n))
        }

        pool.release(buffer: vectorBuf)
        pool.release(buffer: powerBuf)
        pool.release(buffer: outputBuf)
        return results
    }

    // MARK: - GPU alpha powers

    private func gpuAlphaPowers(alpha: Fr, count: Int) -> [Fr] {
        let frStride = MemoryLayout<Fr>.stride

        guard let alphaBuf = pool.allocate(size: frStride),
              let outputBuf = pool.allocate(size: count * frStride) else {
            return cpuAlphaPowers(alpha: alpha, count: count)
        }

        var a = alpha
        withUnsafeBytes(of: &a) { src in
            memcpy(alphaBuf.contents(), src.baseAddress!, frStride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            pool.release(buffer: alphaBuf)
            pool.release(buffer: outputBuf)
            return cpuAlphaPowers(alpha: alpha, count: count)
        }

        encoder.setComputePipelineState(alphaPowersPipeline)
        encoder.setBuffer(alphaBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        var count32 = UInt32(count)
        encoder.setBytes(&count32, length: 4, index: 2)

        // Single thread — serial kernel
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let results: [Fr]
        if cmdBuf.error != nil {
            results = cpuAlphaPowers(alpha: alpha, count: count)
        } else {
            let ptr = outputBuf.contents().bindMemory(to: Fr.self, capacity: count)
            results = Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        pool.release(buffer: alphaBuf)
        pool.release(buffer: outputBuf)
        return results
    }

    // MARK: - CPU fallbacks

    private func cpuCombine(vectors: [[Fr]], powers: [Fr]) -> [Fr] {
        let k = vectors.count
        let n = vectors[0].count
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var acc = Fr.zero
            for j in 0..<k {
                acc = frAdd(acc, frMul(vectors[j][i], powers[j]))
            }
            result[i] = acc
        }
        return result
    }

    private func cpuAlphaPowers(alpha: Fr, count: Int) -> [Fr] {
        var powers = [Fr](repeating: Fr.zero, count: count)
        powers[0] = Fr.one
        for i in 1..<count {
            powers[i] = frMul(powers[i - 1], alpha)
        }
        return powers
    }
}
