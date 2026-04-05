// PrefixScanEngine — GPU-accelerated parallel prefix sum (scan)
//
// Provides inclusive/exclusive scan over uint32 and uint64 arrays,
// plus prefix product over BN254 Fr and BabyBear field elements.
//
// Architecture:
//   - Single-block: Blelloch work-efficient scan in one threadgroup
//   - Multi-block: scan tiles -> scan block sums -> propagate
//   - CPU fallback for small arrays (< cpuThreshold)
//
// Used by: grand product computation, Montgomery batch inverse,
// polynomial evaluation, bucket sorting in MSM.

import Foundation
import Metal

public class PrefixScanEngine {
    public static let version = Versions.prefixScan

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Scan kernels
    private let inclusiveScanU32Pipeline: MTLComputePipelineState
    private let exclusiveScanU32Pipeline: MTLComputePipelineState
    private let propagateBlockSumU32Pipeline: MTLComputePipelineState
    private let inclusiveScanU64Pipeline: MTLComputePipelineState
    private let exclusiveScanU64Pipeline: MTLComputePipelineState
    private let propagateBlockSumU64Pipeline: MTLComputePipelineState
    private let prefixProductBN254Pipeline: MTLComputePipelineState
    private let propagateBlockProductBN254Pipeline: MTLComputePipelineState
    private let prefixProductBabyBearPipeline: MTLComputePipelineState
    private let propagateBlockProductBabyBearPipeline: MTLComputePipelineState

    // Tuning
    private let threadgroupSize: Int  // threads per threadgroup for scan (elements = 2x this)
    public var cpuThreshold: Int = 2048

    public init(threadgroupSize: Int = 512) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.threadgroupSize = threadgroupSize

        let library = try PrefixScanEngine.compileShaders(device: device)

        guard let f1 = library.makeFunction(name: "inclusive_scan_u32"),
              let f2 = library.makeFunction(name: "exclusive_scan_u32"),
              let f3 = library.makeFunction(name: "propagate_block_sum_u32"),
              let f4 = library.makeFunction(name: "inclusive_scan_u64"),
              let f5 = library.makeFunction(name: "exclusive_scan_u64"),
              let f6 = library.makeFunction(name: "propagate_block_sum_u64"),
              let f7 = library.makeFunction(name: "prefix_product_bn254"),
              let f8 = library.makeFunction(name: "propagate_block_product_bn254"),
              let f9 = library.makeFunction(name: "prefix_product_babybear"),
              let f10 = library.makeFunction(name: "propagate_block_product_babybear") else {
            throw MSMError.missingKernel
        }

        self.inclusiveScanU32Pipeline = try device.makeComputePipelineState(function: f1)
        self.exclusiveScanU32Pipeline = try device.makeComputePipelineState(function: f2)
        self.propagateBlockSumU32Pipeline = try device.makeComputePipelineState(function: f3)
        self.inclusiveScanU64Pipeline = try device.makeComputePipelineState(function: f4)
        self.exclusiveScanU64Pipeline = try device.makeComputePipelineState(function: f5)
        self.propagateBlockSumU64Pipeline = try device.makeComputePipelineState(function: f6)
        self.prefixProductBN254Pipeline = try device.makeComputePipelineState(function: f7)
        self.propagateBlockProductBN254Pipeline = try device.makeComputePipelineState(function: f8)
        self.prefixProductBabyBearPipeline = try device.makeComputePipelineState(function: f9)
        self.propagateBlockProductBabyBearPipeline = try device.makeComputePipelineState(function: f10)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let scanSource = try String(contentsOfFile: shaderDir + "/scan/prefix_scan.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanBb = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")
        let cleanScan = scanSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanBb + "\n" + cleanScan
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Inclusive Scan uint32

    /// Inclusive prefix sum of uint32 array.
    /// Result[i] = input[0] + input[1] + ... + input[i]
    public func inclusiveScan(_ values: [UInt32]) -> [UInt32] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuInclusiveScan(values)
        }
        return gpuScanU32(values, inclusive: true)
    }

    // MARK: - Exclusive Scan uint32

    /// Exclusive prefix sum of uint32 array.
    /// Result[i] = input[0] + input[1] + ... + input[i-1], result[0] = 0
    public func exclusiveScan(_ values: [UInt32]) -> [UInt32] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuExclusiveScan(values)
        }
        return gpuScanU32(values, inclusive: false)
    }

    // MARK: - Inclusive Scan uint64

    /// Inclusive prefix sum of uint64 array.
    public func inclusiveScanU64(_ values: [UInt64]) -> [UInt64] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuInclusiveScanU64(values)
        }
        return gpuScanU64(values, inclusive: true)
    }

    /// Exclusive prefix sum of uint64 array.
    public func exclusiveScanU64(_ values: [UInt64]) -> [UInt64] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuExclusiveScanU64(values)
        }
        return gpuScanU64(values, inclusive: false)
    }

    // MARK: - Prefix Product BN254

    /// Running product of BN254 Fr field elements (inclusive).
    /// Elements are packed as 8x uint32 per Fr in Montgomery form.
    /// Result[i] = input[0] * input[1] * ... * input[i]
    public func prefixProductBN254(_ elements: [UInt32]) -> [UInt32] {
        let frCount = elements.count / 8
        if frCount == 0 { return [] }
        if frCount < cpuThreshold {
            // For field products, CPU threshold should be lower since GPU launch has overhead
            // but field mul is expensive enough that even small counts benefit from GPU.
            // Fall through to GPU for correctness (Montgomery mul is complex on CPU).
        }
        return gpuPrefixProductBN254(elements, count: frCount)
    }

    // MARK: - Prefix Product BabyBear

    /// Running product of BabyBear field elements (inclusive).
    /// Each element is one uint32.
    /// Result[i] = input[0] * input[1] * ... * input[i]
    public func prefixProductBabyBear(_ elements: [UInt32]) -> [UInt32] {
        let count = elements.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuPrefixProductBabyBear(elements)
        }
        return gpuPrefixProductBabyBear(elements, count: count)
    }

    // MARK: - GPU dispatch: uint32 scan

    private func gpuScanU32(_ values: [UInt32], inclusive: Bool) -> [UInt32] {
        let count = values.count
        let elementsPerBlock = threadgroupSize * 2
        let numBlocks = (count + elementsPerBlock - 1) / elementsPerBlock

        let inputSize = count * MemoryLayout<UInt32>.size
        let blockSumsSize = max(numBlocks, 1) * MemoryLayout<UInt32>.size
        guard let inputBuf = device.makeBuffer(bytes: values, length: inputSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let blockSumsBuf = device.makeBuffer(length: blockSumsSize, options: .storageModeShared) else {
            return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
        }

        let scanPipeline = inclusive ? inclusiveScanU32Pipeline : exclusiveScanU32Pipeline

        // Pass 1: scan each block
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
        }

        encoder.setComputePipelineState(scanPipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockSumsBuf, offset: 0, index: 2)
        var count32 = UInt32(count)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Multi-block: scan block sums and propagate
        if numBlocks > 1 {
            // Compute exclusive prefix sum of block sums:
            // propagation offset for block i = sum of block sums 0..i-1
            let blockSumsArray = readBufferU32(blockSumsBuf, count: numBlocks)
            let scannedBlockSums = exclusiveScan(blockSumsArray)

            // Write scanned block sums back
            guard let scannedBuf = device.makeBuffer(bytes: scannedBlockSums, length: blockSumsSize, options: .storageModeShared) else {
                return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
            }

            // Pass 2: propagate
            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
            }

            encoder2.setComputePipelineState(propagateBlockSumU32Pipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(scannedBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferU32(outputBuf, count: count)
    }

    // MARK: - GPU dispatch: uint64 scan

    private func gpuScanU64(_ values: [UInt64], inclusive: Bool) -> [UInt64] {
        let count = values.count
        let elementsPerBlock = threadgroupSize * 2
        let numBlocks = (count + elementsPerBlock - 1) / elementsPerBlock

        let inputSize = count * MemoryLayout<UInt64>.size
        let blockSumsSize = max(numBlocks, 1) * MemoryLayout<UInt64>.size
        guard let inputBuf = device.makeBuffer(bytes: values, length: inputSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let blockSumsBuf = device.makeBuffer(length: blockSumsSize, options: .storageModeShared) else {
            return inclusive ? cpuInclusiveScanU64(values) : cpuExclusiveScanU64(values)
        }

        let scanPipeline = inclusive ? inclusiveScanU64Pipeline : exclusiveScanU64Pipeline

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return inclusive ? cpuInclusiveScanU64(values) : cpuExclusiveScanU64(values)
        }

        encoder.setComputePipelineState(scanPipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockSumsBuf, offset: 0, index: 2)
        var count32 = UInt32(count)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if numBlocks > 1 {
            let blockSumsArray = readBufferU64(blockSumsBuf, count: numBlocks)
            let scannedBlockSums = exclusiveScanU64(blockSumsArray)

            guard let scannedBuf = device.makeBuffer(bytes: scannedBlockSums, length: blockSumsSize, options: .storageModeShared) else {
                return inclusive ? cpuInclusiveScanU64(values) : cpuExclusiveScanU64(values)
            }

            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return inclusive ? cpuInclusiveScanU64(values) : cpuExclusiveScanU64(values)
            }

            encoder2.setComputePipelineState(propagateBlockSumU64Pipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(scannedBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferU64(outputBuf, count: count)
    }

    // MARK: - GPU dispatch: BN254 prefix product

    private func gpuPrefixProductBN254(_ elements: [UInt32], count frCount: Int) -> [UInt32] {
        let tileSize = threadgroupSize  // elements per block (1 element per thread, thread 0 does all work)
        let numBlocks = (frCount + tileSize - 1) / tileSize

        let frStride = 8 * MemoryLayout<UInt32>.size  // 32 bytes per Fr
        let inputSize = frCount * frStride
        let blockProductsSize = max(numBlocks, 1) * frStride
        guard let inputBuf = device.makeBuffer(bytes: elements, length: inputSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let blockProductsBuf = device.makeBuffer(length: blockProductsSize, options: .storageModeShared) else {
            return elements  // fallback: return input unchanged
        }

        // Pass 1: per-block prefix product
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return elements
        }

        encoder.setComputePipelineState(prefixProductBN254Pipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockProductsBuf, offset: 0, index: 2)
        var count32 = UInt32(frCount)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Multi-block: scan block products and propagate
        if numBlocks > 1 {
            // Read block products, compute exclusive prefix product on CPU
            // (number of blocks is small, O(numBlocks) sequential muls is fine)
            let blockProducts = readBufferU32(blockProductsBuf, count: numBlocks * 8)

            // Compute inclusive prefix product of block totals on GPU (recursive)
            let scannedBlockProducts = gpuPrefixProductBN254(blockProducts, count: numBlocks)

            // We need exclusive prefix product for propagation:
            // exclusive[0] = identity, exclusive[i] = inclusive[i-1]
            var exclusivePrefix = [UInt32](repeating: 0, count: numBlocks * 8)
            // Fr one in Montgomery form
            let frOne: [UInt32] = [0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
                                   0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1]
            for j in 0..<8 { exclusivePrefix[j] = frOne[j] }
            for i in 1..<numBlocks {
                for j in 0..<8 {
                    exclusivePrefix[i * 8 + j] = scannedBlockProducts[(i - 1) * 8 + j]
                }
            }

            guard let prefixBuf = device.makeBuffer(bytes: exclusivePrefix,
                                                     length: numBlocks * frStride,
                                                     options: .storageModeShared) else {
                return elements
            }

            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return elements
            }

            encoder2.setComputePipelineState(propagateBlockProductBN254Pipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(prefixBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferU32(outputBuf, count: frCount * 8)
    }

    // MARK: - GPU dispatch: BabyBear prefix product

    private func gpuPrefixProductBabyBear(_ elements: [UInt32], count: Int) -> [UInt32] {
        let tileSize = threadgroupSize
        let numBlocks = (count + tileSize - 1) / tileSize

        let inputSize = count * MemoryLayout<UInt32>.size
        let blockProductsSize = max(numBlocks, 1) * MemoryLayout<UInt32>.size
        guard let inputBuf = device.makeBuffer(bytes: elements, length: inputSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let blockProductsBuf = device.makeBuffer(length: blockProductsSize, options: .storageModeShared) else {
            return cpuPrefixProductBabyBear(elements)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuPrefixProductBabyBear(elements)
        }

        encoder.setComputePipelineState(prefixProductBabyBearPipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockProductsBuf, offset: 0, index: 2)
        var count32 = UInt32(count)
        encoder.setBytes(&count32, length: 4, index: 3)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if numBlocks > 1 {
            let blockProducts = readBufferU32(blockProductsBuf, count: numBlocks)
            let scannedBlockProducts = gpuPrefixProductBabyBear(blockProducts, count: numBlocks)

            // Exclusive prefix: shift right, first = 1
            var exclusivePrefix = [UInt32](repeating: 0, count: numBlocks)
            exclusivePrefix[0] = 1  // BabyBear multiplicative identity
            for i in 1..<numBlocks {
                exclusivePrefix[i] = scannedBlockProducts[i - 1]
            }

            guard let prefixBuf = device.makeBuffer(bytes: exclusivePrefix,
                                                     length: numBlocks * MemoryLayout<UInt32>.size,
                                                     options: .storageModeShared) else {
                return cpuPrefixProductBabyBear(elements)
            }

            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return cpuPrefixProductBabyBear(elements)
            }

            encoder2.setComputePipelineState(propagateBlockProductBabyBearPipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(prefixBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferU32(outputBuf, count: count)
    }

    // MARK: - Buffer helpers

    private func readBufferU32(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func readBufferU64(_ buffer: MTLBuffer, count: Int) -> [UInt64] {
        let ptr = buffer.contents().bindMemory(to: UInt64.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - CPU fallbacks

    private func cpuInclusiveScan(_ values: [UInt32]) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: values.count)
        var acc: UInt32 = 0
        for i in 0..<values.count {
            acc = acc &+ values[i]
            result[i] = acc
        }
        return result
    }

    private func cpuExclusiveScan(_ values: [UInt32]) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: values.count)
        var acc: UInt32 = 0
        for i in 0..<values.count {
            result[i] = acc
            acc = acc &+ values[i]
        }
        return result
    }

    private func cpuInclusiveScanU64(_ values: [UInt64]) -> [UInt64] {
        var result = [UInt64](repeating: 0, count: values.count)
        var acc: UInt64 = 0
        for i in 0..<values.count {
            acc = acc &+ values[i]
            result[i] = acc
        }
        return result
    }

    private func cpuExclusiveScanU64(_ values: [UInt64]) -> [UInt64] {
        var result = [UInt64](repeating: 0, count: values.count)
        var acc: UInt64 = 0
        for i in 0..<values.count {
            result[i] = acc
            acc = acc &+ values[i]
        }
        return result
    }

    private func cpuPrefixProductBabyBear(_ elements: [UInt32]) -> [UInt32] {
        let p: UInt64 = 0x78000001
        var result = [UInt32](repeating: 0, count: elements.count)
        var acc: UInt64 = 1
        for i in 0..<elements.count {
            acc = (acc * UInt64(elements[i])) % p
            result[i] = UInt32(acc)
        }
        return result
    }
}
