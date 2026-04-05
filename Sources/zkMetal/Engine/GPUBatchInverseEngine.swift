// GPUBatchInverseEngine — GPU-accelerated batch modular inverse
//
// Montgomery's trick: compute N field inverses using 1 actual inversion + 3(N-1) multiplications.
// Supports BN254 Fr (256-bit, 8x32 Montgomery), BabyBear (32-bit), and Goldilocks (64-bit).
//
// GPU dispatch for N >= 256, CPU fallback below threshold.
// Zero handling: inverse of 0 is 0 (standard ZK convention).

import Foundation
import Metal

// MARK: - GPUBatchInverseEngine

public class GPUBatchInverseEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states for each field
    private let inverseBN254: MTLComputePipelineState
    private let inverseBB: MTLComputePipelineState
    private let inverseGL: MTLComputePipelineState

    /// CPU fallback threshold: arrays smaller than this skip the GPU.
    public var cpuThreshold: Int = 256

    private let pool: GPUBufferPool

    // Chunk sizes must match the shader #defines
    private let chunkBN254 = 512
    private let chunkBB = 2048
    private let chunkGL = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUBatchInverseEngine.compileShaders(device: device)

        guard let fnBN254 = library.makeFunction(name: "batch_inverse_bn254_safe"),
              let fnBB = library.makeFunction(name: "batch_inverse_bb_safe"),
              let fnGL = library.makeFunction(name: "batch_inverse_goldilocks") else {
            throw MSMError.missingKernel
        }

        self.inverseBN254 = try device.makeComputePipelineState(function: fnBN254)
        self.inverseBB = try device.makeComputePipelineState(function: fnBB)
        self.inverseGL = try device.makeComputePipelineState(function: fnGL)
        self.pool = GPUBufferPool(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let glSource = try String(contentsOfFile: shaderDir + "/fields/goldilocks.metal", encoding: .utf8)
        let batchSource = try String(contentsOfFile: shaderDir + "/fields/batch_inverse.metal", encoding: .utf8)

        // Strip includes and header guards for concatenation
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let bbClean = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

        let glClean = glSource
            .replacingOccurrences(of: "#ifndef GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#define GOLDILOCKS_METAL", with: "")
            .replacingOccurrences(of: "#endif // GOLDILOCKS_METAL", with: "")

        let batchClean = batchSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + bbClean + "\n" + glClean + "\n" + batchClean
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        for path in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Public API: Array-based

    /// Compute batch modular inverse for BN254 Fr elements.
    /// Returns out[i] = a[i]^(-1) mod r. inverse(0) = 0.
    public func batchInverse(_ elements: [UInt32], field: FieldType) throws -> [UInt32] {
        switch field {
        case .bn254:
            return try batchInverseBN254Array(elements)
        case .babybear:
            return try batchInverseBabyBearArray(elements)
        case .goldilocks:
            return try batchInverseGoldilocksArray(elements)
        }
    }

    /// Zero-copy MTLBuffer API: compute batch inverse directly on GPU buffers.
    /// Caller is responsible for buffer lifecycle. Output buffer is allocated from the pool.
    public func batchInverseBuffer(_ buf: MTLBuffer, count: Int, field: FieldType) throws -> MTLBuffer {
        switch field {
        case .bn254:
            return try dispatchInverseGPU(buf, count: count, pipeline: inverseBN254, chunkSize: chunkBN254,
                                          elementSize: MemoryLayout<Fr>.stride)
        case .babybear:
            return try dispatchInverseGPU(buf, count: count, pipeline: inverseBB, chunkSize: chunkBB,
                                          elementSize: MemoryLayout<Bb>.stride)
        case .goldilocks:
            return try dispatchInverseGPU(buf, count: count, pipeline: inverseGL, chunkSize: chunkGL,
                                          elementSize: MemoryLayout<Gl>.stride)
        }
    }

    // MARK: - BN254 Fr

    /// Batch inverse for BN254 Fr elements (array of UInt32, 8 limbs per element).
    private func batchInverseBN254Array(_ elements: [UInt32]) throws -> [UInt32] {
        let elemCount = elements.count / 8
        guard elemCount > 0 else { return [] }
        guard elements.count % 8 == 0 else { throw MSMError.invalidInput }

        if elemCount < cpuThreshold {
            return batchInverseCPU_BN254(elements, count: elemCount)
        }

        let byteSize = elements.count * MemoryLayout<UInt32>.stride
        let aBuf = pool.allocate(size: byteSize)!
        elements.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: elemCount, pipeline: inverseBN254,
                                            chunkSize: chunkBN254, elementSize: 8 * MemoryLayout<UInt32>.stride)

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: elements.count)
        let result = Array(UnsafeBufferPointer(start: ptr, count: elements.count))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    /// Batch inverse for BN254 Fr (typed API).
    public func batchInverseFr(_ a: [Fr]) throws -> [Fr] {
        let n = a.count
        guard n > 0 else { return [] }

        if n < cpuThreshold {
            return batchInverseCPU_FrTyped(a)
        }

        let byteSize = n * MemoryLayout<Fr>.stride
        let aBuf = pool.allocate(size: byteSize)!
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: n, pipeline: inverseBN254,
                                            chunkSize: chunkBN254, elementSize: MemoryLayout<Fr>.stride)

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: n)
        let result = Array(UnsafeBufferPointer(start: ptr, count: n))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    // MARK: - BabyBear

    /// Batch inverse for BabyBear elements (array of UInt32, 1 per element).
    private func batchInverseBabyBearArray(_ elements: [UInt32]) throws -> [UInt32] {
        let n = elements.count
        guard n > 0 else { return [] }

        if n < cpuThreshold {
            return batchInverseCPU_BB(elements)
        }

        let byteSize = n * MemoryLayout<UInt32>.stride
        let aBuf = pool.allocate(size: byteSize)!
        elements.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: n, pipeline: inverseBB,
                                            chunkSize: chunkBB, elementSize: MemoryLayout<UInt32>.stride)

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        let result = Array(UnsafeBufferPointer(start: ptr, count: n))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    /// Batch inverse for BabyBear (typed API).
    public func batchInverseBb(_ a: [Bb]) throws -> [Bb] {
        let n = a.count
        guard n > 0 else { return [] }

        if n < cpuThreshold {
            return batchInverseCPU_BbTyped(a)
        }

        let byteSize = n * MemoryLayout<Bb>.stride
        let aBuf = pool.allocate(size: byteSize)!
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: n, pipeline: inverseBB,
                                            chunkSize: chunkBB, elementSize: MemoryLayout<Bb>.stride)

        let ptr = outBuf.contents().bindMemory(to: Bb.self, capacity: n)
        let result = Array(UnsafeBufferPointer(start: ptr, count: n))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    // MARK: - Goldilocks

    /// Batch inverse for Goldilocks elements (array of UInt32, 2 per element = 1 UInt64).
    private func batchInverseGoldilocksArray(_ elements: [UInt32]) throws -> [UInt32] {
        let n = elements.count / 2
        guard n > 0 else { return [] }
        guard elements.count % 2 == 0 else { throw MSMError.invalidInput }

        if n < cpuThreshold {
            return batchInverseCPU_GL(elements, count: n)
        }

        let byteSize = elements.count * MemoryLayout<UInt32>.stride
        let aBuf = pool.allocate(size: byteSize)!
        elements.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: n, pipeline: inverseGL,
                                            chunkSize: chunkGL, elementSize: MemoryLayout<Gl>.stride)

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: elements.count)
        let result = Array(UnsafeBufferPointer(start: ptr, count: elements.count))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    /// Batch inverse for Goldilocks (typed API).
    public func batchInverseGl(_ a: [Gl]) throws -> [Gl] {
        let n = a.count
        guard n > 0 else { return [] }

        if n < cpuThreshold {
            return batchInverseCPU_GlTyped(a)
        }

        let byteSize = n * MemoryLayout<Gl>.stride
        let aBuf = pool.allocate(size: byteSize)!
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteSize) }

        let outBuf = try dispatchInverseGPU(aBuf, count: n, pipeline: inverseGL,
                                            chunkSize: chunkGL, elementSize: MemoryLayout<Gl>.stride)

        let ptr = outBuf.contents().bindMemory(to: Gl.self, capacity: n)
        let result = Array(UnsafeBufferPointer(start: ptr, count: n))
        pool.release(buffer: aBuf)
        pool.release(buffer: outBuf)
        return result
    }

    // MARK: - GPU Dispatch

    private func dispatchInverseGPU(_ aBuf: MTLBuffer, count: Int, pipeline: MTLComputePipelineState,
                                     chunkSize: Int, elementSize: Int) throws -> MTLBuffer {
        let outBuf = pool.allocate(size: count * elementSize)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            pool.release(buffer: outBuf)
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var nVal = UInt32(count)
        enc.setBytes(&nVal, length: 4, index: 2)

        let numGroups = (count + chunkSize - 1) / chunkSize
        let tg = min(64, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            pool.release(buffer: outBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        return outBuf
    }

    // MARK: - CPU Fallbacks (Montgomery's trick)

    private func batchInverseCPU_BN254(_ elements: [UInt32], count: Int) -> [UInt32] {
        // Convert to Fr array
        var a = [Fr](repeating: Fr.zero, count: count)
        elements.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Fr.self)
            for i in 0..<count { a[i] = ptr[i] }
        }
        let result = batchInverseCPU_FrTyped(a)
        return result.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt32.self))
        }
    }

    private func batchInverseCPU_FrTyped(_ a: [Fr]) -> [Fr] {
        let n = a.count
        guard n > 0 else { return [] }

        // Phase 1: prefix products (skip zeros)
        var prefix = [Fr](repeating: Fr.zero, count: n)
        var running = Fr.one
        for i in 0..<n {
            if !frIsZero(a[i]) {
                running = frMul(running, a[i])
            }
            prefix[i] = running
        }

        // Phase 2: single inverse
        var inv = frInverse(running)

        // Phase 3: backward sweep
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if frIsZero(a[i]) {
                result[i] = Fr.zero
            } else {
                if i > 0 {
                    result[i] = frMul(inv, prefix[i - 1])
                } else {
                    result[i] = inv
                }
                inv = frMul(inv, a[i])
            }
        }
        return result
    }

    private func batchInverseCPU_BB(_ elements: [UInt32]) -> [UInt32] {
        let n = elements.count
        let a = elements.map { Bb(v: $0) }
        let result = batchInverseCPU_BbTyped(a)
        return result.map { $0.v }
    }

    private func batchInverseCPU_BbTyped(_ a: [Bb]) -> [Bb] {
        let n = a.count
        guard n > 0 else { return [] }

        var prefix = [Bb](repeating: Bb.zero, count: n)
        var running = Bb.one
        for i in 0..<n {
            if !a[i].isZero {
                running = bbMul(running, a[i])
            }
            prefix[i] = running
        }

        var inv = bbInverse(running)

        var result = [Bb](repeating: Bb.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if a[i].isZero {
                result[i] = Bb.zero
            } else {
                if i > 0 {
                    result[i] = bbMul(inv, prefix[i - 1])
                } else {
                    result[i] = inv
                }
                inv = bbMul(inv, a[i])
            }
        }
        return result
    }

    private func batchInverseCPU_GL(_ elements: [UInt32], count: Int) -> [UInt32] {
        var a = [Gl](repeating: Gl.zero, count: count)
        elements.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Gl.self)
            for i in 0..<count { a[i] = ptr[i] }
        }
        let result = batchInverseCPU_GlTyped(a)
        return result.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt32.self))
        }
    }

    private func batchInverseCPU_GlTyped(_ a: [Gl]) -> [Gl] {
        let n = a.count
        guard n > 0 else { return [] }

        var prefix = [Gl](repeating: Gl.zero, count: n)
        var running = Gl.one
        for i in 0..<n {
            if !a[i].isZero {
                running = glMul(running, a[i])
            }
            prefix[i] = running
        }

        var inv = glInverse(running)

        var result = [Gl](repeating: Gl.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if a[i].isZero {
                result[i] = Gl.zero
            } else {
                if i > 0 {
                    result[i] = glMul(inv, prefix[i - 1])
                } else {
                    result[i] = inv
                }
                inv = glMul(inv, a[i])
            }
        }
        return result
    }

    // MARK: - Helpers

    private func frIsZero(_ a: Fr) -> Bool {
        return a.v.0 == 0 && a.v.1 == 0 && a.v.2 == 0 && a.v.3 == 0 &&
               a.v.4 == 0 && a.v.5 == 0 && a.v.6 == 0 && a.v.7 == 0
    }

    /// Release a buffer back to the pool (for callers using batchInverseBuffer).
    public func releaseBuffer(_ buf: MTLBuffer) {
        pool.release(buffer: buf)
    }
}
