// ParallelReduceEngine — GPU-accelerated parallel reduction
//
// Provides sum, product, min/max over large arrays of field elements or uint32.
// Uses SIMD shuffle + shared memory tree reduction in Metal shaders.
// Multi-pass: if the input exceeds one threadgroup, partial results are
// reduced in subsequent passes until a single result remains.
// Falls back to CPU for small arrays (< cpuThreshold).

import Foundation
import Metal

public class ParallelReduceEngine {
    public static let version = Versions.parallelReduce
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states for each kernel
    let reduceSumBN254: MTLComputePipelineState
    let reduceSumBabyBear: MTLComputePipelineState
    let reduceProductBN254: MTLComputePipelineState
    let reduceMinMaxU32: MTLComputePipelineState

    // Tuning
    private let threadgroupSize: Int
    /// Arrays smaller than this are reduced on CPU.
    public var cpuThreshold: Int = 1024

    // Cached intermediate buffers for multi-pass reduction
    private var scratchBufs: [MTLBuffer] = []
    private var scratchCapacity: Int = 0
    private let pool: GPUBufferPool

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

        let library = try ParallelReduceEngine.compileShaders(device: device)

        guard let sumBN254Fn = library.makeFunction(name: "reduce_sum_bn254"),
              let sumBBFn = library.makeFunction(name: "reduce_sum_babybear"),
              let prodBN254Fn = library.makeFunction(name: "reduce_product_bn254"),
              let minMaxFn = library.makeFunction(name: "reduce_min_max_u32") else {
            throw MSMError.missingKernel
        }

        self.reduceSumBN254 = try device.makeComputePipelineState(function: sumBN254Fn)
        self.reduceSumBabyBear = try device.makeComputePipelineState(function: sumBBFn)
        self.reduceProductBN254 = try device.makeComputePipelineState(function: prodBN254Fn)
        self.reduceMinMaxU32 = try device.makeComputePipelineState(function: minMaxFn)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let reduceSource = try String(contentsOfFile: shaderDir + "/reduction/parallel_reduce.metal", encoding: .utf8)

        // Strip include guards and #include directives
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanBb = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")
        let cleanReduce = reduceSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanBb + "\n" + cleanReduce
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Sum BN254

    /// Sum an array of BN254 Fr field elements on GPU. Returns a single Fr (8x uint32).
    /// For arrays smaller than cpuThreshold, computes on CPU.
    public func sumBN254(_ elements: [UInt32]) -> [UInt32] {
        // elements is packed 8x uint32 per Fr
        let frCount = elements.count / 8
        if frCount == 0 { return [UInt32](repeating: 0, count: 8) }

        if frCount < cpuThreshold {
            return cpuSumBN254(elements, count: frCount)
        }

        return gpuReduce(pipeline: reduceSumBN254, elements: elements,
                         elementCount: frCount, elementStride: 8 * MemoryLayout<UInt32>.size)
    }

    /// Sum BN254 Fr elements already in a Metal buffer.
    public func sumBN254(buffer: MTLBuffer, count: Int) -> [UInt32] {
        if count == 0 { return [UInt32](repeating: 0, count: 8) }
        if count < cpuThreshold {
            let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count * 8)
            let arr = Array(UnsafeBufferPointer(start: ptr, count: count * 8))
            return cpuSumBN254(arr, count: count)
        }
        return gpuReduceBuffer(pipeline: reduceSumBN254, inputBuffer: buffer,
                               elementCount: count, elementStride: 8 * MemoryLayout<UInt32>.size)
    }

    // MARK: - Sum BabyBear

    /// Sum an array of BabyBear field elements. Each element is one uint32.
    public func sumBabyBear(_ elements: [UInt32]) -> UInt32 {
        let count = elements.count
        if count == 0 { return 0 }
        if count < cpuThreshold {
            return cpuSumBabyBear(elements)
        }
        let result = gpuReduce(pipeline: reduceSumBabyBear, elements: elements,
                               elementCount: count, elementStride: MemoryLayout<UInt32>.size)
        return result.first ?? 0
    }

    // MARK: - Product BN254

    /// Multiply all BN254 Fr field elements. Returns a single Fr.
    public func productBN254(_ elements: [UInt32]) -> [UInt32] {
        let frCount = elements.count / 8
        if frCount == 0 {
            // Return Montgomery form of 1
            return [0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
                    0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1]
        }
        if frCount < cpuThreshold {
            return cpuProductBN254(elements, count: frCount)
        }
        return gpuReduce(pipeline: reduceProductBN254, elements: elements,
                         elementCount: frCount, elementStride: 8 * MemoryLayout<UInt32>.size)
    }

    // MARK: - Min/Max uint32

    /// Find the min and max of a uint32 array.
    public func minMaxU32(_ elements: [UInt32]) -> (min: UInt32, max: UInt32) {
        let count = elements.count
        if count == 0 { return (UInt32.max, 0) }
        if count < cpuThreshold {
            var mn: UInt32 = .max
            var mx: UInt32 = 0
            for v in elements { mn = Swift.min(mn, v); mx = Swift.max(mx, v) }
            return (mn, mx)
        }
        // MinMaxResult is {uint32, uint32} = 8 bytes
        let result = gpuReduceMinMax(elements: elements, elementCount: count)
        return result
    }

    // MARK: - GPU dispatch (generic)

    private func gpuReduce(pipeline: MTLComputePipelineState,
                           elements: [UInt32],
                           elementCount: Int,
                           elementStride: Int) -> [UInt32] {
        let inputSize = elementCount * elementStride
        guard let inputBuf = pool.allocate(size: inputSize) else {
            return [UInt32](repeating: 0, count: elementStride / MemoryLayout<UInt32>.size)
        }
        elements.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, inputSize)
        }
        let result = gpuReduceBuffer(pipeline: pipeline, inputBuffer: inputBuf,
                               elementCount: elementCount, elementStride: elementStride)
        pool.release(buffer: inputBuf)
        return result
    }

    private func gpuReduceBuffer(pipeline: MTLComputePipelineState,
                                 inputBuffer: MTLBuffer,
                                 elementCount: Int,
                                 elementStride: Int) -> [UInt32] {
        let elementsPerU32 = elementStride / MemoryLayout<UInt32>.size
        var currentBuf = inputBuffer
        var currentCount = elementCount
        var intermediateBuffers: [MTLBuffer] = []

        while currentCount > 1 {
            let numGroups = (currentCount + threadgroupSize - 1) / threadgroupSize
            let outputSize = numGroups * elementStride
            guard let outputBuf = pool.allocate(size: outputSize) else {
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return [UInt32](repeating: 0, count: elementsPerU32)
            }

            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else {
                pool.release(buffer: outputBuf)
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return [UInt32](repeating: 0, count: elementsPerU32)
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(currentBuf, offset: 0, index: 0)
            encoder.setBuffer(outputBuf, offset: 0, index: 1)
            var count32 = UInt32(currentCount)
            encoder.setBytes(&count32, length: 4, index: 2)

            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            let gridSize = MTLSize(width: numGroups * threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            intermediateBuffers.append(outputBuf)
            currentBuf = outputBuf
            currentCount = numGroups
        }

        // Read back the single result
        let ptr = currentBuf.contents().bindMemory(to: UInt32.self, capacity: elementsPerU32)
        let result = Array(UnsafeBufferPointer(start: ptr, count: elementsPerU32))
        for buf in intermediateBuffers { pool.release(buffer: buf) }
        return result
    }

    private func gpuReduceMinMax(elements: [UInt32], elementCount: Int) -> (min: UInt32, max: UInt32) {
        let inputSize = elementCount * MemoryLayout<UInt32>.size
        guard let inputBuf = pool.allocate(size: inputSize) else {
            return (.max, 0)
        }
        elements.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, inputSize)
        }

        // MinMaxResult = 2x uint32 = 8 bytes
        let resultStride = 2 * MemoryLayout<UInt32>.size
        var currentBuf = inputBuf
        var currentCount = elementCount
        var isFirstPass = true
        var intermediateBuffers: [MTLBuffer] = [inputBuf]

        while currentCount > 1 {
            let numGroups = (currentCount + threadgroupSize - 1) / threadgroupSize
            let outputSize = numGroups * resultStride
            guard let outputBuf = pool.allocate(size: outputSize) else {
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return (.max, 0)
            }

            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else {
                pool.release(buffer: outputBuf)
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return (.max, 0)
            }

            encoder.setComputePipelineState(reduceMinMaxU32)
            if isFirstPass {
                // First pass: input is uint32 array
                encoder.setBuffer(currentBuf, offset: 0, index: 0)
            } else {
                // Subsequent passes: input is MinMaxResult array; need a different approach.
                // For simplicity, re-dispatch reduce_min_max_u32 which reads uint*.
                // We need a separate second-pass kernel, or we handle it on CPU.
                // For now, if numGroups is small enough, just finish on CPU.
                let ptr = currentBuf.contents().bindMemory(to: UInt32.self, capacity: currentCount * 2)
                var mn: UInt32 = .max
                var mx: UInt32 = 0
                for i in 0..<currentCount {
                    mn = Swift.min(mn, ptr[i * 2])
                    mx = Swift.max(mx, ptr[i * 2 + 1])
                }
                encoder.endEncoding()
                pool.release(buffer: outputBuf)
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return (mn, mx)
            }
            encoder.setBuffer(outputBuf, offset: 0, index: 1)
            var count32 = UInt32(currentCount)
            encoder.setBytes(&count32, length: 4, index: 2)

            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            let gridSize = MTLSize(width: numGroups * threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            intermediateBuffers.append(outputBuf)
            currentBuf = outputBuf
            currentCount = numGroups
            isFirstPass = false
        }

        let ptr = currentBuf.contents().bindMemory(to: UInt32.self, capacity: 2)
        let result = (ptr[0], ptr[1])
        for buf in intermediateBuffers { pool.release(buffer: buf) }
        return result
    }

    // MARK: - CPU fallbacks

    private func cpuSumBN254(_ elements: [UInt32], count: Int) -> [UInt32] {
        // Simple CPU fallback: add Fr elements sequentially
        // Fr is 8x uint32 in Montgomery form, addition with modular reduction
        var acc = [UInt32](repeating: 0, count: 8)
        let p: [UInt64] = [0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72]
        for i in 0..<count {
            let base = i * 8
            // acc = acc + elements[base..<base+8] mod p
            var carry: UInt64 = 0
            var tmp = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                carry += UInt64(acc[j]) + UInt64(elements[base + j])
                tmp[j] = UInt32(carry & 0xFFFFFFFF)
                carry >>= 32
            }
            // Conditional subtract p
            var gte = false
            if carry != 0 {
                gte = true
            } else {
                for j in stride(from: 7, through: 0, by: -1) {
                    if UInt64(tmp[j]) > p[j] { gte = true; break }
                    if UInt64(tmp[j]) < p[j] { break }
                }
                // If all equal, gte = true
                if !gte {
                    var allEq = true
                    for j in 0..<8 {
                        if UInt64(tmp[j]) != p[j] { allEq = false; break }
                    }
                    if allEq { gte = true }
                }
            }
            if gte {
                var borrow: Int64 = 0
                for j in 0..<8 {
                    borrow += Int64(tmp[j]) - Int64(p[j])
                    tmp[j] = UInt32(borrow & 0xFFFFFFFF)
                    borrow >>= 32
                }
            }
            acc = tmp
        }
        return acc
    }

    private func cpuSumBabyBear(_ elements: [UInt32]) -> UInt32 {
        let p: UInt32 = 0x78000001
        var acc: UInt32 = 0
        for v in elements {
            let sum = UInt64(acc) + UInt64(v)
            acc = UInt32(sum >= UInt64(p) ? sum - UInt64(p) : sum)
        }
        return acc
    }

    private func cpuProductBN254(_ elements: [UInt32], count: Int) -> [UInt32] {
        // For CPU fallback of product, this would need full Montgomery multiplication.
        // For small counts, just dispatch to GPU anyway or return identity for count=0.
        // In practice, count < cpuThreshold means this is called for small arrays.
        // We fall through to GPU for correctness since Montgomery mul on CPU is complex.
        // Override: just use GPU for product regardless.
        return gpuReduce(pipeline: reduceProductBN254, elements: elements,
                         elementCount: count, elementStride: 8 * MemoryLayout<UInt32>.size)
    }
}
