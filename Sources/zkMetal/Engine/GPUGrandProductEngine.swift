// GPUGrandProductEngine — GPU-accelerated grand product computation
//
// Provides three core operations over BN254 Fr field elements:
//   1. grandProduct: product of all elements in a vector
//   2. partialProducts: exclusive prefix product z[i] = prod(v[0]..v[i-1])
//   3. permutationProduct: z[i] = prod(num[j]/den[j], j=0..i-1) with GPU batch inverse
//
// Architecture:
//   - Per-block sequential prefix product on GPU (field mul is order-dependent)
//   - Multi-block: scan block totals on CPU (small), propagate on GPU
//   - GPU batch inverse for denominators via GPUBatchInverseEngine
//   - CPU fallback for arrays below gpuThreshold
//
// Used by: Plonk permutation argument, Halo2 permutation, LogUp/Plookup lookup
// arguments, GKR grand product layer.

import Foundation
import Metal
import NeonFieldOps

public class GPUGrandProductEngine {
    public static let version = Versions.gpuGrandProduct

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states
    private let localPipeline: MTLComputePipelineState
    private let propagatePipeline: MTLComputePipelineState
    private let elemMulPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState

    // Batch inverse engine for denominator inversion
    private let inverseEngine: GPUBatchInverseEngine

    private let pool: GPUBufferPool

    /// Arrays smaller than this use CPU path.
    public var gpuThreshold: Int = 4096

    private let threadgroupSize: Int

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.threadgroupSize = 512

        let library = try GPUGrandProductEngine.compileShaders(device: device)

        guard let f1 = library.makeFunction(name: "grand_product_local"),
              let f2 = library.makeFunction(name: "grand_product_propagate"),
              let f3 = library.makeFunction(name: "grand_product_elem_mul"),
              let f4 = library.makeFunction(name: "grand_product_reduce") else {
            throw MSMError.missingKernel
        }

        self.localPipeline = try device.makeComputePipelineState(function: f1)
        self.propagatePipeline = try device.makeComputePipelineState(function: f2)
        self.elemMulPipeline = try device.makeComputePipelineState(function: f3)
        self.reducePipeline = try device.makeComputePipelineState(function: f4)

        self.inverseEngine = try GPUBatchInverseEngine()
        self.pool = GPUBufferPool(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let gpSource = try String(contentsOfFile: shaderDir + "/reduce/grand_product.metal", encoding: .utf8)

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let gpClean = gpSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + gpClean
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Public API

    /// Compute the product of all elements: values[0] * values[1] * ... * values[n-1].
    /// Returns Fr.one for empty input.
    public func grandProduct(values: [Fr]) -> Fr {
        let n = values.count
        guard n > 0 else { return Fr.one }

        if n < gpuThreshold {
            return cpuFullProduct(values)
        }

        if let result = gpuFullProduct(values) {
            return result
        }
        return cpuFullProduct(values)
    }

    /// Compute exclusive prefix products (partial products):
    ///   z[0] = 1
    ///   z[i] = values[0] * values[1] * ... * values[i-1]
    ///
    /// Output has the same length as input.
    public func partialProducts(values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return cpuPartialProducts(values)
        }

        if let result = gpuPartialProducts(values) {
            return result
        }
        return cpuPartialProducts(values)
    }

    /// Compute the permutation product polynomial:
    ///   z[0] = 1
    ///   z[i] = prod(numerators[j] / denominators[j], j=0..i-1)
    ///
    /// Uses GPU batch inverse for denominators, then element-wise multiply,
    /// then prefix product.
    ///
    /// - Parameters:
    ///   - numerators: Numerator field elements (same length as denominators).
    ///   - denominators: Denominator field elements (must be non-zero).
    /// - Returns: Exclusive prefix product of ratios, length == numerators.count.
    public func permutationProduct(numerators: [Fr], denominators: [Fr]) -> [Fr] {
        precondition(numerators.count == denominators.count,
                     "numerators and denominators must have equal length")
        let n = numerators.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return cpuPermutationProduct(numerators: numerators, denominators: denominators)
        }

        // Step 1: Batch invert denominators on GPU
        let invDen: [Fr]
        if let inv = try? inverseEngine.batchInverseFr(denominators) {
            invDen = inv
        } else {
            return cpuPermutationProduct(numerators: numerators, denominators: denominators)
        }

        // Step 2: Element-wise multiply numerators * inv_denominators on GPU
        let ratios: [Fr]
        if let r = gpuElemMul(numerators, invDen) {
            ratios = r
        } else {
            // CPU fallback for elem mul via batch C kernel
            var fallbackResult = [Fr](repeating: .zero, count: n)
            numerators.withUnsafeBytes { nBuf in
                invDen.withUnsafeBytes { dBuf in
                    fallbackResult.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_mul(
                            nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            dBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n)
                        )
                    }
                }
            }
            ratios = fallbackResult
        }

        // Step 3: Prefix product of ratios
        if let result = gpuPartialProducts(ratios) {
            return result
        }
        return cpuPartialProducts(ratios)
    }

    // MARK: - GPU: Full Product Reduction

    private func gpuFullProduct(_ values: [Fr]) -> Fr? {
        let n = values.count
        let frStride = MemoryLayout<Fr>.stride
        let numBlocks = (n + threadgroupSize - 1) / threadgroupSize

        let inputSize = n * frStride
        let blockSize = max(numBlocks, 1) * frStride

        guard let inputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let blockBuf = device.makeBuffer(length: blockSize, options: .storageModeShared) else {
            return nil
        }

        values.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, inputSize)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(reducePipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(blockBuf, offset: 0, index: 1)
        var count32 = UInt32(n)
        encoder.setBytes(&count32, length: 4, index: 2)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil { return nil }

        // Read block products and reduce on CPU
        let ptr = blockBuf.contents().bindMemory(to: Fr.self, capacity: numBlocks)
        var result = Fr.one
        for i in 0..<numBlocks {
            result = frMul(result, ptr[i])
        }
        return result
    }

    // MARK: - GPU: Exclusive Prefix Product

    private func gpuPartialProducts(_ values: [Fr]) -> [Fr]? {
        let n = values.count
        let frStride = MemoryLayout<Fr>.stride
        let numBlocks = (n + threadgroupSize - 1) / threadgroupSize

        let dataSize = n * frStride
        let blockSize = max(numBlocks, 1) * frStride

        guard let inputBuf = device.makeBuffer(length: dataSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: dataSize, options: .storageModeShared),
              let blockBuf = device.makeBuffer(length: blockSize, options: .storageModeShared) else {
            return nil
        }

        values.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, dataSize)
        }

        // Pass 1: Per-block exclusive prefix product
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(localPipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockBuf, offset: 0, index: 2)
        var count32 = UInt32(n)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil { return nil }

        // Multi-block: compute exclusive prefix product of block totals, propagate
        if numBlocks > 1 {
            let blockPtr = blockBuf.contents().bindMemory(to: Fr.self, capacity: numBlocks)

            // Exclusive prefix product of block totals (CPU — numBlocks is small)
            var blockPrefixes = [Fr](repeating: Fr.zero, count: numBlocks)
            blockPrefixes[0] = Fr.one
            for i in 1..<numBlocks {
                blockPrefixes[i] = frMul(blockPrefixes[i - 1], blockPtr[i - 1])
            }

            guard let prefixBuf = device.makeBuffer(length: blockSize, options: .storageModeShared) else {
                return nil
            }
            blockPrefixes.withUnsafeBytes { src in
                memcpy(prefixBuf.contents(), src.baseAddress!, numBlocks * frStride)
            }

            // Pass 2: Propagate block prefixes
            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return nil
            }

            encoder2.setComputePipelineState(propagatePipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(prefixBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()

            if cmdBuf2.error != nil { return nil }
        }

        // Read result
        var result = [Fr](repeating: Fr.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, outputBuf.contents(), dataSize)
        }
        return result
    }

    // MARK: - GPU: Element-wise Multiply

    private func gpuElemMul(_ a: [Fr], _ b: [Fr]) -> [Fr]? {
        let n = a.count
        let frStride = MemoryLayout<Fr>.stride
        let dataSize = n * frStride

        guard let aBuf = device.makeBuffer(length: dataSize, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: dataSize, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: dataSize, options: .storageModeShared) else {
            return nil
        }

        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, dataSize) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, dataSize) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(elemMulPipeline)
        encoder.setBuffer(aBuf, offset: 0, index: 0)
        encoder.setBuffer(bBuf, offset: 0, index: 1)
        encoder.setBuffer(outBuf, offset: 0, index: 2)
        var count32 = UInt32(n)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: min(256, threadgroupSize), height: 1, depth: 1)
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil { return nil }

        var result = [Fr](repeating: Fr.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, outBuf.contents(), dataSize)
        }
        return result
    }

    // MARK: - CPU Fallbacks

    private func cpuFullProduct(_ values: [Fr]) -> Fr {
        var acc = Fr.one
        for v in values {
            acc = frMul(acc, v)
        }
        return acc
    }

    private func cpuPartialProducts(_ values: [Fr]) -> [Fr] {
        let n = values.count
        var result = [Fr](repeating: Fr.zero, count: n)
        result[0] = Fr.one
        for i in 1..<n {
            result[i] = frMul(result[i - 1], values[i - 1])
        }
        return result
    }

    private func cpuPermutationProduct(numerators: [Fr], denominators: [Fr]) -> [Fr] {
        let n = numerators.count
        // Batch inverse on CPU
        var invDen = [Fr](repeating: Fr.zero, count: n)

        // Montgomery's trick
        var prefix = [Fr](repeating: Fr.zero, count: n)
        var running = Fr.one
        for i in 0..<n {
            if !denominators[i].isZero {
                running = frMul(running, denominators[i])
            }
            prefix[i] = running
        }

        var inv = frInverse(running)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if denominators[i].isZero {
                invDen[i] = Fr.zero
            } else {
                if i > 0 {
                    invDen[i] = frMul(inv, prefix[i - 1])
                } else {
                    invDen[i] = inv
                }
                inv = frMul(inv, denominators[i])
            }
        }

        // Compute ratios and prefix product
        var ratios = [Fr](repeating: Fr.zero, count: n)
        numerators.withUnsafeBytes { aBuf in
            invDen.withUnsafeBytes { bBuf in
                ratios.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return cpuPartialProducts(ratios)
    }
}
