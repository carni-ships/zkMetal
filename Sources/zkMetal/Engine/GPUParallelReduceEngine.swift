// GPUParallelReduceEngine — General-purpose GPU parallel reduction for BN254 Fr
//
// Provides sum, product, and custom reductions over large Fr arrays using Metal
// compute shaders. Multi-pass reduction for arrays exceeding one threadgroup.
// Falls back to CPU for small inputs.

import Foundation
import Metal

// MARK: - Reduction operation enum

/// The reduction operation to apply.
public enum ReduceOp: Int {
    case sum = 0
    case product = 1
    case min = 2
    case max = 3
}

// MARK: - GPUParallelReduceEngine

public class GPUParallelReduceEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Dedicated pipeline states for sum and product (no function constants, faster)
    private let sumPipeline: MTLComputePipelineState
    private let productPipeline: MTLComputePipelineState

    // Generic pipelines keyed by ReduceOp (uses function constants)
    private var genericPipelines: [Int: MTLComputePipelineState] = [:]

    private let threadgroupSize: Int
    private let pool: GPUBufferPool

    /// Arrays smaller than this are reduced on CPU.
    public var cpuThreshold: Int = 1024

    /// The compiled Metal library (retained for generic pipeline creation).
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

        self.library = try GPUParallelReduceEngine.compileShaders(device: device)

        guard let sumFn = library.makeFunction(name: "fr_reduce_sum"),
              let prodFn = library.makeFunction(name: "fr_reduce_product") else {
            throw MSMError.missingKernel
        }

        self.sumPipeline = try device.makeComputePipelineState(function: sumFn)
        self.productPipeline = try device.makeComputePipelineState(function: prodFn)

        // Pre-build generic pipelines for all ops
        for op in [ReduceOp.sum, .product, .min, .max] {
            genericPipelines[op.rawValue] = try buildGenericPipeline(op: op)
        }
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let reduceSource = try String(contentsOfFile: shaderDir + "/reduce/parallel_reduce.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanReduce = reduceSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanReduce
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private func buildGenericPipeline(op: ReduceOp) throws -> MTLComputePipelineState {
        let constants = MTLFunctionConstantValues()
        var opVal = Int32(op.rawValue)
        constants.setConstantValue(&opVal, type: .int, index: 0)

        let fn = try library.makeFunction(name: "generic_reduce", constantValues: constants)
        return try device.makeComputePipelineState(function: fn)
    }

    // MARK: - Public API

    /// Sum all Fr elements in a buffer. Returns a single Fr.
    public func sum(buffer: MTLBuffer, count: Int) -> Fr {
        if count == 0 { return Fr.zero }
        if count == 1 {
            let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: 1)
            return ptr[0]
        }
        if count < cpuThreshold {
            return cpuSum(buffer: buffer, count: count)
        }
        let result = gpuReduceBuffer(pipeline: sumPipeline, inputBuffer: buffer, elementCount: count)
        return result
    }

    /// Sum all Fr elements in an array. Returns a single Fr.
    public func sum(_ elements: [Fr]) -> Fr {
        let count = elements.count
        if count == 0 { return Fr.zero }
        if count == 1 { return elements[0] }
        if count < cpuThreshold {
            return cpuSum(elements)
        }
        let byteSize = count * MemoryLayout<Fr>.stride
        guard let buf = pool.allocate(size: byteSize) else { return cpuSum(elements) }
        elements.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, byteSize) }
        let result = gpuReduceBuffer(pipeline: sumPipeline, inputBuffer: buf, elementCount: count)
        pool.release(buffer: buf)
        return result
    }

    /// Multiply all Fr elements in a buffer. Returns a single Fr.
    public func product(buffer: MTLBuffer, count: Int) -> Fr {
        if count == 0 { return Fr.one }
        if count == 1 {
            let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: 1)
            return ptr[0]
        }
        if count < cpuThreshold {
            return cpuProduct(buffer: buffer, count: count)
        }
        return gpuReduceBuffer(pipeline: productPipeline, inputBuffer: buffer, elementCount: count)
    }

    /// Multiply all Fr elements in an array. Returns a single Fr.
    public func product(_ elements: [Fr]) -> Fr {
        let count = elements.count
        if count == 0 { return Fr.one }
        if count == 1 { return elements[0] }
        if count < cpuThreshold {
            return cpuProduct(elements)
        }
        let byteSize = count * MemoryLayout<Fr>.stride
        guard let buf = pool.allocate(size: byteSize) else { return cpuProduct(elements) }
        elements.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, byteSize) }
        let result = gpuReduceBuffer(pipeline: productPipeline, inputBuffer: buf, elementCount: count)
        pool.release(buffer: buf)
        return result
    }

    /// Generic reduce with a configurable operation.
    public func reduce(buffer: MTLBuffer, count: Int, op: ReduceOp) -> Fr {
        if count == 0 {
            switch op {
            case .sum, .max: return Fr.zero
            case .product: return Fr.one
            case .min: return Fr.zero // convention
            }
        }
        if count == 1 {
            let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: 1)
            return ptr[0]
        }
        // For sum/product with small counts, use CPU
        if count < cpuThreshold {
            switch op {
            case .sum: return cpuSum(buffer: buffer, count: count)
            case .product: return cpuProduct(buffer: buffer, count: count)
            case .min, .max:
                return cpuGeneric(buffer: buffer, count: count, op: op)
            }
        }
        guard let pipeline = genericPipelines[op.rawValue] else { return Fr.zero }
        return gpuReduceBuffer(pipeline: pipeline, inputBuffer: buffer, elementCount: count)
    }

    /// Generic reduce over an array.
    public func reduce(_ elements: [Fr], op: ReduceOp) -> Fr {
        let count = elements.count
        if count == 0 {
            switch op {
            case .sum, .max: return Fr.zero
            case .product: return Fr.one
            case .min: return Fr.zero
            }
        }
        if count == 1 { return elements[0] }
        let byteSize = count * MemoryLayout<Fr>.stride
        guard let buf = pool.allocate(size: byteSize) else {
            // Fallback: CPU
            switch op {
            case .sum: return cpuSum(elements)
            case .product: return cpuProduct(elements)
            case .min, .max: return cpuGeneric(elements, op: op)
            }
        }
        elements.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, byteSize) }
        let result = reduce(buffer: buf, count: count, op: op)
        pool.release(buffer: buf)
        return result
    }

    // MARK: - GPU dispatch (multi-pass)

    private func gpuReduceBuffer(pipeline: MTLComputePipelineState,
                                 inputBuffer: MTLBuffer,
                                 elementCount: Int) -> Fr {
        let elementStride = MemoryLayout<Fr>.stride
        var currentBuf = inputBuffer
        var currentCount = elementCount
        var intermediateBuffers: [MTLBuffer] = []

        while currentCount > 1 {
            let numGroups = (currentCount + threadgroupSize - 1) / threadgroupSize
            let outputSize = numGroups * elementStride

            guard let outputBuf = pool.allocate(size: outputSize) else {
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return Fr.zero
            }

            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else {
                pool.release(buffer: outputBuf)
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                return Fr.zero
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(currentBuf, offset: 0, index: 0)
            encoder.setBuffer(outputBuf, offset: 0, index: 1)
            var count32 = UInt32(currentCount)
            encoder.setBytes(&count32, length: 4, index: 2)

            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: numGroups, height: 1, depth: 1),
                threadsPerThreadgroup: tgSize
            )
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if cmdBuf.error != nil {
                for buf in intermediateBuffers { pool.release(buffer: buf) }
                pool.release(buffer: outputBuf)
                return Fr.zero
            }

            intermediateBuffers.append(outputBuf)
            currentBuf = outputBuf
            currentCount = numGroups
        }

        // Read back the single result
        let ptr = currentBuf.contents().bindMemory(to: Fr.self, capacity: 1)
        let result = ptr[0]
        for buf in intermediateBuffers { pool.release(buffer: buf) }
        return result
    }

    // MARK: - CPU fallbacks

    private func cpuSum(buffer: MTLBuffer, count: Int) -> Fr {
        let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: count)
        var acc = Fr.zero
        for i in 0..<count {
            acc = frAdd(acc, ptr[i])
        }
        return acc
    }

    private func cpuSum(_ elements: [Fr]) -> Fr {
        var acc = Fr.zero
        for e in elements {
            acc = frAdd(acc, e)
        }
        return acc
    }

    private func cpuProduct(buffer: MTLBuffer, count: Int) -> Fr {
        let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: count)
        var acc = Fr.one
        for i in 0..<count {
            acc = frMul(acc, ptr[i])
        }
        return acc
    }

    private func cpuProduct(_ elements: [Fr]) -> Fr {
        var acc = Fr.one
        for e in elements {
            acc = frMul(acc, e)
        }
        return acc
    }

    private func cpuGeneric(buffer: MTLBuffer, count: Int, op: ReduceOp) -> Fr {
        let ptr = buffer.contents().bindMemory(to: Fr.self, capacity: count)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: count))
        return cpuGeneric(arr, op: op)
    }

    private func cpuGeneric(_ elements: [Fr], op: ReduceOp) -> Fr {
        guard !elements.isEmpty else { return Fr.zero }
        var acc = elements[0]
        for i in 1..<elements.count {
            switch op {
            case .sum: acc = frAdd(acc, elements[i])
            case .product: acc = frMul(acc, elements[i])
            case .min: acc = frLt(elements[i], acc) ? elements[i] : acc
            case .max: acc = frLt(acc, elements[i]) ? elements[i] : acc
            }
        }
        return acc
    }

    /// Lexicographic less-than for Fr (big-endian limb comparison).
    private func frLt(_ a: Fr, _ b: Fr) -> Bool {
        let aLimbs = [a.v.7, a.v.6, a.v.5, a.v.4, a.v.3, a.v.2, a.v.1, a.v.0]
        let bLimbs = [b.v.7, b.v.6, b.v.5, b.v.4, b.v.3, b.v.2, b.v.1, b.v.0]
        for i in 0..<8 {
            if aLimbs[i] < bLimbs[i] { return true }
            if aLimbs[i] > bLimbs[i] { return false }
        }
        return false
    }
}
