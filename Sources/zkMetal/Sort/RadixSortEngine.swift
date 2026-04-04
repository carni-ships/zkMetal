// GPU Radix Sort Engine
// Sorts 32-bit unsigned integer keys using 4-pass LSD radix sort (8-bit radix).
// Three GPU kernels per pass: histogram → prefix sum → scatter.
// Supports key-only and key-value sorting.

import Foundation
import Metal

public class RadixSortEngine {
    public static let version = Versions.radixSort
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    let histogramFunction: MTLComputePipelineState
    let prefixSumFunction: MTLComputePipelineState
    let scatterFunction: MTLComputePipelineState
    let scatterKVFunction: MTLComputePipelineState

    static let radixBits = 8
    static let radixSize = 256  // 2^radixBits
    static let tileSize = 4096  // keys per threadgroup

    // Grow-only buffer cache
    private var histogramBuf: MTLBuffer?
    private var histogramBufSize: Int = 0
    private var offsetsBuf: MTLBuffer?
    private var offsetsBufSize: Int = 0
    private var tempKeysBuf: MTLBuffer?
    private var tempKeysBufSize: Int = 0
    private var tempValsBuf: MTLBuffer?
    private var tempValsBufSize: Int = 0
    private var inputKeysBuf: MTLBuffer?
    private var inputKeysBufSize: Int = 0
    private var inputValsBuf: MTLBuffer?
    private var inputValsBufSize: Int = 0

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
              let scatterFn = library.makeFunction(name: "radix_scatter"),
              let scatterKVFn = library.makeFunction(name: "radix_scatter_kv") else {
            throw MSMError.missingKernel
        }
        self.histogramFunction = try device.makeComputePipelineState(function: histFn)
        self.prefixSumFunction = try device.makeComputePipelineState(function: prefixFn)
        self.scatterFunction = try device.makeComputePipelineState(function: scatterFn)
        self.scatterKVFunction = try device.makeComputePipelineState(function: scatterKVFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/sort/radix_sort.metal", encoding: .utf8)
        let cleanSource = source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: cleanSource, options: options)
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for path in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(path)/sort/radix_sort.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    /// Sort an array of 32-bit unsigned integer keys.
    /// Returns the sorted array.
    public func sort(_ keys: [UInt32]) throws -> [UInt32] {
        let n = keys.count
        if n <= 1 { return keys }

        // For small arrays, use CPU sort
        if n < 4096 {
            return keys.sorted()
        }

        let numTiles = (n + RadixSortEngine.tileSize - 1) / RadixSortEngine.tileSize
        let histSize = numTiles * RadixSortEngine.radixSize

        // Ensure buffers are large enough
        ensureBuffers(n: n, histSize: histSize)

        // Cached input key buffer (ping-pong partner of tempKeysBuf)
        let keyBytes = n * 4
        if inputKeysBufSize < keyBytes {
            inputKeysBuf = device.makeBuffer(length: keyBytes, options: .storageModeShared)
            inputKeysBufSize = keyBytes
        }
        memcpy(inputKeysBuf!.contents(), keys, keyBytes)
        let keyBufA = inputKeysBuf!
        let keyBufB = tempKeysBuf!

        var inputBuf = keyBufA
        var outputBuf = keyBufB

        let tgSize = min(256, Int(scatterFunction.maxTotalThreadsPerThreadgroup))
        let prefixTG = min(256, Int(prefixSumFunction.maxTotalThreadsPerThreadgroup))

        // Single command buffer for all 4 passes (eliminates 11 CPU-GPU round trips)
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        for pass in 0..<4 {
            var nVal = UInt32(n)
            var shiftVal = UInt32(pass * RadixSortEngine.radixBits)
            var numTilesVal = UInt32(numTiles)

            // Kernel 1: Histogram
            let enc1 = cb.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(histogramFunction)
            enc1.setBuffer(inputBuf, offset: 0, index: 0)
            enc1.setBuffer(histogramBuf!, offset: 0, index: 1)
            enc1.setBytes(&nVal, length: 4, index: 2)
            enc1.setBytes(&shiftVal, length: 4, index: 3)
            enc1.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc1.endEncoding()

            // Kernel 2: Prefix sum (single threadgroup)
            let enc2 = cb.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(prefixSumFunction)
            enc2.setBuffer(histogramBuf!, offset: 0, index: 0)
            enc2.setBuffer(offsetsBuf!, offset: 0, index: 1)
            enc2.setBytes(&numTilesVal, length: 4, index: 2)
            enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: prefixTG, height: 1, depth: 1))
            enc2.endEncoding()

            // Kernel 3: Scatter
            let enc3 = cb.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(scatterFunction)
            enc3.setBuffer(inputBuf, offset: 0, index: 0)
            enc3.setBuffer(outputBuf, offset: 0, index: 1)
            enc3.setBuffer(offsetsBuf!, offset: 0, index: 2)
            enc3.setBytes(&nVal, length: 4, index: 3)
            enc3.setBytes(&shiftVal, length: 4, index: 4)
            enc3.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc3.endEncoding()

            // Swap ping-pong
            let tmp = inputBuf
            inputBuf = outputBuf
            outputBuf = tmp
        }

        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        // After 4 passes (even number), result is in the original inputBuf position
        let resultPtr = inputBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        return Array(UnsafeBufferPointer(start: resultPtr, count: n))
    }

    /// Sort key-value pairs by key.
    /// Returns (sorted keys, sorted values).
    public func sortKV(keys: [UInt32], values: [UInt32]) throws -> ([UInt32], [UInt32]) {
        let n = keys.count
        precondition(values.count == n)
        if n <= 1 { return (keys, values) }

        if n < 4096 {
            let indexed = keys.enumerated().sorted { $0.element < $1.element }
            return (indexed.map { $0.element }, indexed.map { values[$0.offset] })
        }

        let numTiles = (n + RadixSortEngine.tileSize - 1) / RadixSortEngine.tileSize
        let histSize = numTiles * RadixSortEngine.radixSize

        ensureBuffers(n: n, histSize: histSize)

        let keyBytes = n * 4
        if inputKeysBufSize < keyBytes {
            inputKeysBuf = device.makeBuffer(length: keyBytes, options: .storageModeShared)
            inputKeysBufSize = keyBytes
        }
        memcpy(inputKeysBuf!.contents(), keys, keyBytes)
        if inputValsBufSize < keyBytes {
            inputValsBuf = device.makeBuffer(length: keyBytes, options: .storageModeShared)
            inputValsBufSize = keyBytes
        }
        memcpy(inputValsBuf!.contents(), values, keyBytes)
        let keyBufA = inputKeysBuf!
        let keyBufB = tempKeysBuf!
        let valBufA = inputValsBuf!
        let valBufB = tempValsBuf!

        var inKeys = keyBufA, outKeys = keyBufB
        var inVals = valBufA, outVals = valBufB

        let tgSize = min(256, Int(scatterKVFunction.maxTotalThreadsPerThreadgroup))
        let prefixTG = min(256, Int(prefixSumFunction.maxTotalThreadsPerThreadgroup))

        // Single command buffer for all 4 passes
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        for pass in 0..<4 {
            var nVal = UInt32(n)
            var shiftVal = UInt32(pass * RadixSortEngine.radixBits)
            var numTilesVal = UInt32(numTiles)

            // Kernel 1: Histogram
            let enc1 = cb.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(histogramFunction)
            enc1.setBuffer(inKeys, offset: 0, index: 0)
            enc1.setBuffer(histogramBuf!, offset: 0, index: 1)
            enc1.setBytes(&nVal, length: 4, index: 2)
            enc1.setBytes(&shiftVal, length: 4, index: 3)
            enc1.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc1.endEncoding()

            // Kernel 2: Prefix sum
            let enc2 = cb.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(prefixSumFunction)
            enc2.setBuffer(histogramBuf!, offset: 0, index: 0)
            enc2.setBuffer(offsetsBuf!, offset: 0, index: 1)
            enc2.setBytes(&numTilesVal, length: 4, index: 2)
            enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: prefixTG, height: 1, depth: 1))
            enc2.endEncoding()

            // Kernel 3: Scatter KV
            let enc3 = cb.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(scatterKVFunction)
            enc3.setBuffer(inKeys, offset: 0, index: 0)
            enc3.setBuffer(outKeys, offset: 0, index: 1)
            enc3.setBuffer(inVals, offset: 0, index: 2)
            enc3.setBuffer(outVals, offset: 0, index: 3)
            enc3.setBuffer(offsetsBuf!, offset: 0, index: 4)
            enc3.setBytes(&nVal, length: 4, index: 5)
            enc3.setBytes(&shiftVal, length: 4, index: 6)
            enc3.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc3.endEncoding()

            swap(&inKeys, &outKeys)
            swap(&inVals, &outVals)
        }

        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MSMError.gpuError(error.localizedDescription) }

        let kPtr = inKeys.contents().bindMemory(to: UInt32.self, capacity: n)
        let vPtr = inVals.contents().bindMemory(to: UInt32.self, capacity: n)
        return (Array(UnsafeBufferPointer(start: kPtr, count: n)),
                Array(UnsafeBufferPointer(start: vPtr, count: n)))
    }

    // MARK: - Single pass (used by debugPass only)

    /// Debug: run one pass and return intermediate state
    public func debugPass(keys: [UInt32], shift: UInt32) throws -> (histogram: [UInt32], offsets: [UInt32], output: [UInt32]) {
        let n = keys.count
        let numTiles = (n + RadixSortEngine.tileSize - 1) / RadixSortEngine.tileSize
        let histSize = numTiles * RadixSortEngine.radixSize
        ensureBuffers(n: n, histSize: histSize)

        let inputBuf = device.makeBuffer(bytes: keys, length: n * 4, options: .storageModeShared)!
        let outputBuf = tempKeysBuf!
        var nVal = UInt32(n)
        var shiftVal = shift
        var numTilesVal = UInt32(numTiles)
        let tgSize = min(256, Int(histogramFunction.maxTotalThreadsPerThreadgroup))

        // Histogram
        let cb1 = commandQueue.makeCommandBuffer()!
        let enc1 = cb1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(histogramFunction)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(histogramBuf!, offset: 0, index: 1)
        enc1.setBytes(&nVal, length: 4, index: 2)
        enc1.setBytes(&shiftVal, length: 4, index: 3)
        enc1.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc1.endEncoding()
        cb1.commit()
        cb1.waitUntilCompleted()

        let histPtr = histogramBuf!.contents().bindMemory(to: UInt32.self, capacity: histSize)
        let histogram = Array(UnsafeBufferPointer(start: histPtr, count: histSize))

        // Prefix sum
        let cb2 = commandQueue.makeCommandBuffer()!
        let enc2 = cb2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(prefixSumFunction)
        enc2.setBuffer(histogramBuf!, offset: 0, index: 0)
        enc2.setBuffer(offsetsBuf!, offset: 0, index: 1)
        enc2.setBytes(&numTilesVal, length: 4, index: 2)
        let prefixTG = min(256, Int(prefixSumFunction.maxTotalThreadsPerThreadgroup))
        enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: prefixTG, height: 1, depth: 1))
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        let offPtr = offsetsBuf!.contents().bindMemory(to: UInt32.self, capacity: histSize)
        let offsets = Array(UnsafeBufferPointer(start: offPtr, count: histSize))

        // Scatter
        let cb3 = commandQueue.makeCommandBuffer()!
        let enc3 = cb3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(scatterFunction)
        enc3.setBuffer(inputBuf, offset: 0, index: 0)
        enc3.setBuffer(outputBuf, offset: 0, index: 1)
        enc3.setBuffer(offsetsBuf!, offset: 0, index: 2)
        enc3.setBytes(&nVal, length: 4, index: 3)
        enc3.setBytes(&shiftVal, length: 4, index: 4)
        enc3.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc3.endEncoding()
        cb3.commit()
        cb3.waitUntilCompleted()

        let outPtr = outputBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        let output = Array(UnsafeBufferPointer(start: outPtr, count: n))
        return (histogram, offsets, output)
    }

    // MARK: - Buffer management

    private func ensureBuffers(n: Int, histSize: Int) {
        let histBytes = histSize * 4
        if histogramBufSize < histBytes {
            histogramBuf = device.makeBuffer(length: histBytes, options: .storageModeShared)
            histogramBufSize = histBytes
        }
        if offsetsBufSize < histBytes {
            offsetsBuf = device.makeBuffer(length: histBytes, options: .storageModeShared)
            offsetsBufSize = histBytes
        }
        let keyBytes = n * 4
        if tempKeysBufSize < keyBytes {
            tempKeysBuf = device.makeBuffer(length: keyBytes, options: .storageModeShared)
            tempKeysBufSize = keyBytes
        }
        if tempValsBufSize < keyBytes {
            tempValsBuf = device.makeBuffer(length: keyBytes, options: .storageModeShared)
            tempValsBufSize = keyBytes
        }
    }
}
