// GPUPrefixSumEngine — GPU-accelerated parallel prefix sum for BN254 Fr field elements
//
// Provides inclusive/exclusive/segmented scan over Fr arrays using Metal
// compute shaders. Blelloch work-efficient algorithm: O(n) work, O(log n) depth.
// Multi-block: scan tiles -> scan block sums -> propagate.
// CPU fallback for small inputs.
//
// Used by: polynomial evaluation, trace generation, accumulator computation,
// grand product witness, running sums in lookup arguments.

import Foundation
import Metal

public class GPUPrefixSumEngine {
    public static let version = Versions.gpuPrefixSum

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Scan pipelines
    private let inclusiveScanFrPipeline: MTLComputePipelineState
    private let exclusiveScanFrPipeline: MTLComputePipelineState
    private let propagateBlockSumFrPipeline: MTLComputePipelineState
    private let segmentedScanFrPipeline: MTLComputePipelineState
    private let propagateBlockSumSegmentedFrPipeline: MTLComputePipelineState

    // Tuning
    private let threadgroupSize: Int  // threads per threadgroup (elements = 2x this)
    public var cpuThreshold: Int = 2048

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

        let library = try GPUPrefixSumEngine.compileShaders(device: device)

        guard let f1 = library.makeFunction(name: "inclusive_scan_fr"),
              let f2 = library.makeFunction(name: "exclusive_scan_fr"),
              let f3 = library.makeFunction(name: "propagate_block_sum_fr"),
              let f4 = library.makeFunction(name: "segmented_scan_fr"),
              let f5 = library.makeFunction(name: "propagate_block_sum_segmented_fr") else {
            throw MSMError.missingKernel
        }

        self.inclusiveScanFrPipeline = try device.makeComputePipelineState(function: f1)
        self.exclusiveScanFrPipeline = try device.makeComputePipelineState(function: f2)
        self.propagateBlockSumFrPipeline = try device.makeComputePipelineState(function: f3)
        self.segmentedScanFrPipeline = try device.makeComputePipelineState(function: f4)
        self.propagateBlockSumSegmentedFrPipeline = try device.makeComputePipelineState(function: f5)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let scanSource = try String(contentsOfFile: shaderDir + "/reduce/prefix_sum.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanScan = scanSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanScan
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Inclusive Scan Fr

    /// Inclusive prefix sum of BN254 Fr field elements.
    /// Result[i] = input[0] + input[1] + ... + input[i]  (mod r)
    public func inclusiveScan(values: [Fr]) -> [Fr] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuInclusiveScan(values)
        }
        return gpuScanFr(values, inclusive: true)
    }

    // MARK: - Exclusive Scan Fr

    /// Exclusive prefix sum of BN254 Fr field elements.
    /// Result[i] = input[0] + input[1] + ... + input[i-1], result[0] = 0
    public func exclusiveScan(values: [Fr]) -> [Fr] {
        let count = values.count
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuExclusiveScan(values)
        }
        return gpuScanFr(values, inclusive: false)
    }

    // MARK: - Segmented Scan Fr

    /// Segmented inclusive prefix sum of BN254 Fr field elements.
    /// Segments are defined by the flags array: flags[i] = true starts a new segment.
    /// The scan resets at each segment boundary.
    /// Result[i] = sum of values from the start of i's segment up to i (inclusive).
    public func segmentedScan(values: [Fr], flags: [Bool]) -> [Fr] {
        let count = values.count
        precondition(flags.count == count, "flags must have same count as values")
        if count == 0 { return [] }
        if count < cpuThreshold {
            return cpuSegmentedScan(values, flags: flags)
        }
        return gpuSegmentedScanFr(values, flags: flags)
    }

    // MARK: - GPU dispatch: Fr scan

    private func gpuScanFr(_ values: [Fr], inclusive: Bool) -> [Fr] {
        let count = values.count
        let elementsPerBlock = threadgroupSize * 2
        let numBlocks = (count + elementsPerBlock - 1) / elementsPerBlock

        let frStride = MemoryLayout<Fr>.stride
        let inputSize = count * frStride
        let blockSumsSize = max(numBlocks, 1) * frStride

        guard let inputBuf = device.makeBuffer(
            bytes: values, length: inputSize, options: .storageModeShared
        ),
              let outputBuf = device.makeBuffer(
            length: inputSize, options: .storageModeShared
        ),
              let blockSumsBuf = device.makeBuffer(
            length: blockSumsSize, options: .storageModeShared
        ) else {
            return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
        }

        let scanPipeline = inclusive ? inclusiveScanFrPipeline : exclusiveScanFrPipeline

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
            // Recursively scan the block sums (exclusive scan for propagation offsets)
            let blockSumsArray = readBufferFr(blockSumsBuf, count: numBlocks)
            let scannedBlockSums = numBlocks < cpuThreshold
                ? cpuExclusiveScan(blockSumsArray)
                : gpuScanFr(blockSumsArray, inclusive: false)

            guard let scannedBuf = device.makeBuffer(
                bytes: scannedBlockSums, length: blockSumsSize, options: .storageModeShared
            ) else {
                return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
            }

            // Pass 2: propagate block offsets
            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return inclusive ? cpuInclusiveScan(values) : cpuExclusiveScan(values)
            }

            encoder2.setComputePipelineState(propagateBlockSumFrPipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(scannedBuf, offset: 0, index: 1)
            encoder2.setBytes(&count32, length: 4, index: 2)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferFr(outputBuf, count: count)
    }

    // MARK: - GPU dispatch: segmented scan

    private func gpuSegmentedScanFr(_ values: [Fr], flags: [Bool]) -> [Fr] {
        let count = values.count
        let elementsPerBlock = threadgroupSize * 2
        let numBlocks = (count + elementsPerBlock - 1) / elementsPerBlock

        let frStride = MemoryLayout<Fr>.stride
        let inputSize = count * frStride
        let blockSumsSize = max(numBlocks, 1) * frStride
        let flagsSize = count * MemoryLayout<UInt32>.size
        let blockFlagsSize = max(numBlocks, 1) * MemoryLayout<UInt32>.size

        // Convert Bool flags to UInt32 for GPU
        let flagsU32 = flags.map { $0 ? UInt32(1) : UInt32(0) }

        guard let inputBuf = device.makeBuffer(
            bytes: values, length: inputSize, options: .storageModeShared
        ),
              let outputBuf = device.makeBuffer(
            length: inputSize, options: .storageModeShared
        ),
              let flagsBuf = device.makeBuffer(
            bytes: flagsU32, length: flagsSize, options: .storageModeShared
        ),
              let blockSumsBuf = device.makeBuffer(
            length: blockSumsSize, options: .storageModeShared
        ),
              let blockFlagsBuf = device.makeBuffer(
            length: blockFlagsSize, options: .storageModeShared
        ) else {
            return cpuSegmentedScan(values, flags: flags)
        }

        // Pass 1: segmented scan each block
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuSegmentedScan(values, flags: flags)
        }

        encoder.setComputePipelineState(segmentedScanFrPipeline)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(flagsBuf, offset: 0, index: 2)
        encoder.setBuffer(blockSumsBuf, offset: 0, index: 3)
        encoder.setBuffer(blockFlagsBuf, offset: 0, index: 4)
        var count32 = UInt32(count)
        encoder.setBytes(&count32, length: 4, index: 5)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Multi-block: propagate with segment awareness
        if numBlocks > 1 {
            // Read block sums and flags
            let blockSumsArray = readBufferFr(blockSumsBuf, count: numBlocks)
            let blockFlagsArray = readBufferU32(blockFlagsBuf, count: numBlocks)

            // Compute segmented exclusive prefix sum of block sums on CPU
            // (number of blocks is small)
            var scannedBlockSums = [Fr](repeating: Fr.zero, count: numBlocks)
            var acc = Fr.zero
            for i in 0..<numBlocks {
                if blockFlagsArray[i] != 0 {
                    acc = Fr.zero  // reset at segment boundary
                }
                scannedBlockSums[i] = acc
                acc = frAdd(acc, blockSumsArray[i])
            }

            guard let scannedBuf = device.makeBuffer(
                bytes: scannedBlockSums, length: blockSumsSize, options: .storageModeShared
            ) else {
                return cpuSegmentedScan(values, flags: flags)
            }

            // Pass 2: propagate with segment flag checking
            guard let cmdBuf2 = commandQueue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return cpuSegmentedScan(values, flags: flags)
            }

            encoder2.setComputePipelineState(propagateBlockSumSegmentedFrPipeline)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(scannedBuf, offset: 0, index: 1)
            encoder2.setBuffer(flagsBuf, offset: 0, index: 2)
            encoder2.setBytes(&count32, length: 4, index: 3)

            let propGrid = MTLSize(width: numBlocks * threadgroupSize, height: 1, depth: 1)
            encoder2.dispatchThreads(propGrid, threadsPerThreadgroup: tgSize)
            encoder2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        }

        return readBufferFr(outputBuf, count: count)
    }

    // MARK: - Buffer helpers

    private func readBufferFr(_ buffer: MTLBuffer, count: Int) -> [Fr] {
        let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func readBufferU32(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - CPU fallbacks

    private func cpuInclusiveScan(_ values: [Fr]) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: values.count)
        var acc = Fr.zero
        for i in 0..<values.count {
            acc = frAdd(acc, values[i])
            result[i] = acc
        }
        return result
    }

    private func cpuExclusiveScan(_ values: [Fr]) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: values.count)
        var acc = Fr.zero
        for i in 0..<values.count {
            result[i] = acc
            acc = frAdd(acc, values[i])
        }
        return result
    }

    private func cpuSegmentedScan(_ values: [Fr], flags: [Bool]) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: values.count)
        var acc = Fr.zero
        for i in 0..<values.count {
            if flags[i] {
                acc = Fr.zero  // reset at segment boundary
            }
            acc = frAdd(acc, values[i])
            result[i] = acc
        }
        return result
    }
}
