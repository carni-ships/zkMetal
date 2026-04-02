// GPU Radix Sort Engine — 32-bit key sort with optional values
// 4-pass LSD radix sort with radix=256 (8-bit digits)
// Used for MSM bucket accumulation, polynomial evaluation, etc.

import Foundation
import Metal

public class RadixSortEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    let histogramFunction: MTLComputePipelineState
    let prefixSumFunction: MTLComputePipelineState
    let addBlockSumsFunction: MTLComputePipelineState
    let scatterFunction: MTLComputePipelineState
    let scatterKeysOnlyFunction: MTLComputePipelineState

    // Cached buffers to avoid per-call allocation
    private var cachedHistBuf: MTLBuffer?
    private var cachedHistSize: Int = 0
    private var cachedBlockSumsBuf: MTLBuffer?
    private var cachedBlockSumsSize: Int = 0
    let tinyBuffer: MTLBuffer  // 4-byte buffer for second-level scan output

    static let RADIX = 256
    static let THREADS_PER_GROUP = 256
    static let ELEMENTS_PER_THREAD = 4
    static let ELEMENTS_PER_GROUP = THREADS_PER_GROUP * ELEMENTS_PER_THREAD  // 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try RadixSortEngine.compileShaders(device: device)

        guard let histFn = library.makeFunction(name: "radix_histogram"),
              let prefixFn = library.makeFunction(name: "radix_prefix_sum"),
              let addFn = library.makeFunction(name: "radix_add_block_sums"),
              let scatterFn = library.makeFunction(name: "radix_scatter"),
              let scatterKeysFn = library.makeFunction(name: "radix_scatter_keys_only") else {
            throw MSMError.missingKernel
        }

        self.histogramFunction = try device.makeComputePipelineState(function: histFn)
        self.prefixSumFunction = try device.makeComputePipelineState(function: prefixFn)
        self.addBlockSumsFunction = try device.makeComputePipelineState(function: addFn)
        self.scatterFunction = try device.makeComputePipelineState(function: scatterFn)
        self.scatterKeysOnlyFunction = try device.makeComputePipelineState(function: scatterKeysFn)

        guard let tiny = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create tiny buffer")
        }
        self.tinyBuffer = tiny
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/sort/radix_sort.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("sort/radix_sort.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./metal/Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/sort/radix_sort.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Public API

    /// Sort an array of UInt32 keys (keys only, no values).
    public func sortKeys(_ keys: [UInt32]) throws -> [UInt32] {
        let n = keys.count
        if n <= 1 { return keys }

        let byteCount = n * 4
        guard let keysBufA = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let keysBufB = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create sort buffers")
        }

        keys.withUnsafeBytes { src in
            memcpy(keysBufA.contents(), src.baseAddress!, byteCount)
        }

        try sortKeysInPlace(keysA: keysBufA, keysB: keysBufB, count: n)

        let ptr = keysBufA.contents().bindMemory(to: UInt32.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Sort key-value pairs by key.
    public func sortKeyValue(keys: [UInt32], values: [UInt32]) throws -> (keys: [UInt32], values: [UInt32]) {
        let n = keys.count
        precondition(values.count == n)
        if n <= 1 { return (keys, values) }

        let byteCount = n * 4
        guard let keysBufA = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let keysBufB = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let valsBufA = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let valsBufB = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create sort buffers")
        }

        keys.withUnsafeBytes { src in memcpy(keysBufA.contents(), src.baseAddress!, byteCount) }
        values.withUnsafeBytes { src in memcpy(valsBufA.contents(), src.baseAddress!, byteCount) }

        try sortKVInPlace(keysA: keysBufA, keysB: keysBufB,
                          valsA: valsBufA, valsB: valsBufB, count: n)

        let kPtr = keysBufA.contents().bindMemory(to: UInt32.self, capacity: n)
        let vPtr = valsBufA.contents().bindMemory(to: UInt32.self, capacity: n)
        return (Array(UnsafeBufferPointer(start: kPtr, count: n)),
                Array(UnsafeBufferPointer(start: vPtr, count: n)))
    }

    // MARK: - Internal sort implementation

    private func getHistBuffer(size: Int) throws -> MTLBuffer {
        if cachedHistSize >= size, let buf = cachedHistBuf { return buf }
        guard let buf = device.makeBuffer(length: size * 4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create histogram buffer")
        }
        cachedHistBuf = buf
        cachedHistSize = size
        return buf
    }

    private func getBlockSumsBuffer(size: Int) throws -> MTLBuffer {
        if cachedBlockSumsSize >= size, let buf = cachedBlockSumsBuf { return buf }
        guard let buf = device.makeBuffer(length: size * 4, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create block sums buffer")
        }
        cachedBlockSumsBuf = buf
        cachedBlockSumsSize = size
        return buf
    }

    private func sortKeysInPlace(keysA: MTLBuffer, keysB: MTLBuffer, count: Int) throws {
        let numBlocks = (count + RadixSortEngine.ELEMENTS_PER_GROUP - 1) / RadixSortEngine.ELEMENTS_PER_GROUP
        let histSize = RadixSortEngine.RADIX * numBlocks

        let histBuf = try getHistBuffer(size: histSize)
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.gpuError("Failed to create command buffer")
        }

        // Pre-compute prefix sum parameters
        let scanGroupSize = 512
        let elementsPerScanGroup = scanGroupSize * 2
        let numScanGroups = (histSize + elementsPerScanGroup - 1) / elementsPerScanGroup
        let blockSumsBuf = try getBlockSumsBuffer(size: numScanGroups)

        // Single encoder for all passes — eliminates encoder transition overhead
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var srcKeys = keysA
        var dstKeys = keysB

        for pass in 0..<4 {
            var countVal = UInt32(count)
            var shiftVal = UInt32(pass * 8)
            var histSizeVal = UInt32(histSize)

            // Phase 1: Histogram
            enc.setComputePipelineState(histogramFunction)
            enc.setBuffer(srcKeys, offset: 0, index: 0)
            enc.setBuffer(histBuf, offset: 0, index: 1)
            enc.setBytes(&countVal, length: 4, index: 2)
            enc.setBytes(&shiftVal, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: RadixSortEngine.THREADS_PER_GROUP, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Phase 2: Prefix sum (inline, no separate encoders)
            enc.setComputePipelineState(prefixSumFunction)
            enc.setBuffer(histBuf, offset: 0, index: 0)
            enc.setBuffer(blockSumsBuf, offset: 0, index: 1)
            enc.setBytes(&histSizeVal, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: numScanGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            if numScanGroups > 1 {
                var numScanGroupsVal = UInt32(numScanGroups)
                enc.setComputePipelineState(prefixSumFunction)
                enc.setBuffer(blockSumsBuf, offset: 0, index: 0)
                enc.setBuffer(tinyBuffer, offset: 0, index: 1)
                enc.setBytes(&numScanGroupsVal, length: 4, index: 2)
                let scanThreads = min(scanGroupSize, (numScanGroups + 1) / 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(addBlockSumsFunction)
                enc.setBuffer(histBuf, offset: 0, index: 0)
                enc.setBuffer(blockSumsBuf, offset: 0, index: 1)
                enc.setBytes(&histSizeVal, length: 4, index: 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: numScanGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            // Phase 3: Scatter
            enc.setComputePipelineState(scatterKeysOnlyFunction)
            enc.setBuffer(srcKeys, offset: 0, index: 0)
            enc.setBuffer(dstKeys, offset: 0, index: 1)
            enc.setBuffer(histBuf, offset: 0, index: 2)
            enc.setBytes(&countVal, length: 4, index: 3)
            enc.setBytes(&shiftVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: RadixSortEngine.THREADS_PER_GROUP, height: 1, depth: 1))

            if pass < 3 {
                enc.memoryBarrier(scope: .buffers)
            }

            swap(&srcKeys, &dstKeys)
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    private func sortKVInPlace(keysA: MTLBuffer, keysB: MTLBuffer,
                               valsA: MTLBuffer, valsB: MTLBuffer, count: Int) throws {
        let numBlocks = (count + RadixSortEngine.ELEMENTS_PER_GROUP - 1) / RadixSortEngine.ELEMENTS_PER_GROUP
        let histSize = RadixSortEngine.RADIX * numBlocks

        let histBuf = try getHistBuffer(size: histSize)

        // Pre-compute prefix sum parameters
        let scanGroupSize = 512
        let elementsPerScanGroup = scanGroupSize * 2
        let numScanGroups = (histSize + elementsPerScanGroup - 1) / elementsPerScanGroup
        let blockSumsBuf = try getBlockSumsBuffer(size: numScanGroups)

        var srcKeys = keysA
        var dstKeys = keysB
        var srcVals = valsA
        var dstVals = valsB

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Single encoder for all passes
        let enc = cmdBuf.makeComputeCommandEncoder()!

        for pass in 0..<4 {
            var countVal = UInt32(count)
            var shiftVal = UInt32(pass * 8)
            var histSizeVal = UInt32(histSize)

            // Histogram
            enc.setComputePipelineState(histogramFunction)
            enc.setBuffer(srcKeys, offset: 0, index: 0)
            enc.setBuffer(histBuf, offset: 0, index: 1)
            enc.setBytes(&countVal, length: 4, index: 2)
            enc.setBytes(&shiftVal, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: RadixSortEngine.THREADS_PER_GROUP, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Prefix sum
            enc.setComputePipelineState(prefixSumFunction)
            enc.setBuffer(histBuf, offset: 0, index: 0)
            enc.setBuffer(blockSumsBuf, offset: 0, index: 1)
            enc.setBytes(&histSizeVal, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: numScanGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            if numScanGroups > 1 {
                var numScanGroupsVal = UInt32(numScanGroups)
                enc.setComputePipelineState(prefixSumFunction)
                enc.setBuffer(blockSumsBuf, offset: 0, index: 0)
                enc.setBuffer(tinyBuffer, offset: 0, index: 1)
                enc.setBytes(&numScanGroupsVal, length: 4, index: 2)
                let scanThreads = min(scanGroupSize, (numScanGroups + 1) / 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(addBlockSumsFunction)
                enc.setBuffer(histBuf, offset: 0, index: 0)
                enc.setBuffer(blockSumsBuf, offset: 0, index: 1)
                enc.setBytes(&histSizeVal, length: 4, index: 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: numScanGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            // Scatter
            enc.setComputePipelineState(scatterFunction)
            enc.setBuffer(srcKeys, offset: 0, index: 0)
            enc.setBuffer(srcVals, offset: 0, index: 1)
            enc.setBuffer(dstKeys, offset: 0, index: 2)
            enc.setBuffer(dstVals, offset: 0, index: 3)
            enc.setBuffer(histBuf, offset: 0, index: 4)
            enc.setBytes(&countVal, length: 4, index: 5)
            enc.setBytes(&shiftVal, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: numBlocks, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: RadixSortEngine.THREADS_PER_GROUP, height: 1, depth: 1))

            if pass < 3 {
                enc.memoryBarrier(scope: .buffers)
            }

            swap(&srcKeys, &dstKeys)
            swap(&srcVals, &dstVals)
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// Encode a prefix sum on the histogram buffer using recursive Blelloch scan.
    private func encodePrefixSum(cmdBuf: MTLCommandBuffer, histBuf: MTLBuffer, totalEntries: Int) throws {
        let scanGroupSize = 512
        let elementsPerScanGroup = scanGroupSize * 2
        let numScanGroups = (totalEntries + elementsPerScanGroup - 1) / elementsPerScanGroup

        let blockSumsBuf = try getBlockSumsBuffer(size: numScanGroups)

        var totalEntriesVal = UInt32(totalEntries)

        let encScan = cmdBuf.makeComputeCommandEncoder()!
        encScan.setComputePipelineState(prefixSumFunction)
        encScan.setBuffer(histBuf, offset: 0, index: 0)
        encScan.setBuffer(blockSumsBuf, offset: 0, index: 1)
        encScan.setBytes(&totalEntriesVal, length: 4, index: 2)
        encScan.dispatchThreadgroups(
            MTLSize(width: numScanGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
        encScan.endEncoding()

        if numScanGroups > 1 {
            let blockSumsSumBuf = tinyBuffer

            var numScanGroupsVal = UInt32(numScanGroups)
            let encScanSums = cmdBuf.makeComputeCommandEncoder()!
            encScanSums.setComputePipelineState(prefixSumFunction)
            encScanSums.setBuffer(blockSumsBuf, offset: 0, index: 0)
            encScanSums.setBuffer(blockSumsSumBuf, offset: 0, index: 1)
            encScanSums.setBytes(&numScanGroupsVal, length: 4, index: 2)
            let scanThreads = min(scanGroupSize, (numScanGroups + 1) / 2)
            encScanSums.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))
            encScanSums.endEncoding()

            let encAdd = cmdBuf.makeComputeCommandEncoder()!
            encAdd.setComputePipelineState(addBlockSumsFunction)
            encAdd.setBuffer(histBuf, offset: 0, index: 0)
            encAdd.setBuffer(blockSumsBuf, offset: 0, index: 1)
            encAdd.setBytes(&totalEntriesVal, length: 4, index: 2)
            encAdd.dispatchThreadgroups(
                MTLSize(width: numScanGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: scanGroupSize, height: 1, depth: 1))
            encAdd.endEncoding()
        }
    }

    // MARK: - CPU reference

    public static func cpuRadixSort(_ keys: [UInt32]) -> [UInt32] {
        var src = keys
        var dst = [UInt32](repeating: 0, count: keys.count)
        let n = keys.count

        for pass in 0..<4 {
            let shift = pass * 8
            var count = [Int](repeating: 0, count: 256)

            // Histogram
            for key in src {
                let digit = Int((key >> shift) & 0xFF)
                count[digit] += 1
            }

            // Prefix sum
            var total = 0
            for i in 0..<256 {
                let c = count[i]
                count[i] = total
                total += c
            }

            // Scatter
            for key in src {
                let digit = Int((key >> shift) & 0xFF)
                dst[count[digit]] = key
                count[digit] += 1
            }

            swap(&src, &dst)
        }
        return src
    }
}
