// BatchFieldEngine — GPU-accelerated batch field operations
//
// Cross-system primitives used across multiple proof systems:
//   - Batch modular inverse (Montgomery's trick)
//   - Batch field multiply (element-wise)
//   - Batch field add (element-wise)
//   - Batch polynomial evaluation at multiple points (Horner per thread)
//
// Supports BN254 Fr (256-bit, 8x32 Montgomery) and BabyBear (32-bit).
// GPU dispatch for N >= 1024, CPU fallback below threshold.

import Foundation
import Metal
import NeonFieldOps

// MARK: - BatchFieldEngine

public class BatchFieldEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // BN254 Fr kernels
    private let batchInverseBN254: MTLComputePipelineState
    private let batchMulBN254: MTLComputePipelineState
    private let batchAddBN254: MTLComputePipelineState
    private let batchEvalBN254: MTLComputePipelineState

    // BabyBear kernels
    private let batchInverseBB: MTLComputePipelineState
    private let batchMulBB: MTLComputePipelineState
    private let batchAddBB: MTLComputePipelineState
    private let batchEvalBB: MTLComputePipelineState

    /// GPU dispatch threshold: arrays smaller than this use CPU fallback.
    public var gpuThreshold: Int = 1024

    private let tuning: TuningConfig
    private let pool: GPUBufferPool

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try BatchFieldEngine.compileShaders(device: device)

        guard let invBN254 = library.makeFunction(name: "batch_inverse_bn254"),
              let mulBN254 = library.makeFunction(name: "batch_mul_bn254"),
              let addBN254 = library.makeFunction(name: "batch_add_bn254"),
              let evalBN254 = library.makeFunction(name: "batch_eval_bn254"),
              let invBB = library.makeFunction(name: "batch_inverse_bb"),
              let mulBB = library.makeFunction(name: "batch_mul_bb"),
              let addBB = library.makeFunction(name: "batch_add_bb"),
              let evalBB = library.makeFunction(name: "batch_eval_bb") else {
            throw MSMError.missingKernel
        }

        self.batchInverseBN254 = try device.makeComputePipelineState(function: invBN254)
        self.batchMulBN254 = try device.makeComputePipelineState(function: mulBN254)
        self.batchAddBN254 = try device.makeComputePipelineState(function: addBN254)
        self.batchEvalBN254 = try device.makeComputePipelineState(function: evalBN254)
        self.batchInverseBB = try device.makeComputePipelineState(function: invBB)
        self.batchMulBB = try device.makeComputePipelineState(function: mulBB)
        self.batchAddBB = try device.makeComputePipelineState(function: addBB)
        self.batchEvalBB = try device.makeComputePipelineState(function: evalBB)
        self.tuning = TuningManager.shared.config(device: device)
        self.pool = GPUBufferPool(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let batchSource = try String(contentsOfFile: shaderDir + "/fields/batch_field_ops.metal", encoding: .utf8)

        // Strip includes and header guards for concatenation
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let bbClean = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

        let batchClean = batchSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + bbClean + "\n" + batchClean
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

    // MARK: - Buffer helpers

    private func createFrBuffer(_ data: [Fr]) -> MTLBuffer {
        let bytes = data.count * MemoryLayout<Fr>.stride
        let buf = pool.allocate(size: bytes)!
        data.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, bytes) }
        return buf
    }

    private func readFrBuffer(_ buf: MTLBuffer, count: Int) -> [Fr] {
        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func createBbBuffer(_ data: [Bb]) -> MTLBuffer {
        let bytes = data.count * MemoryLayout<Bb>.stride
        let buf = pool.allocate(size: bytes)!
        data.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, bytes) }
        return buf
    }

    private func readBbBuffer(_ buf: MTLBuffer, count: Int) -> [Bb] {
        let ptr = buf.contents().bindMemory(to: Bb.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - GPU dispatch helpers

    /// Dispatch an element-wise binary operation (a, b -> out) on the GPU.
    private func dispatchElementWise(
        _ pipeline: MTLComputePipelineState,
        _ aBuf: MTLBuffer, _ bBuf: MTLBuffer, _ outBuf: MTLBuffer,
        n: Int
    ) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    // ================================================================
    // MARK: - BN254 Fr Operations
    // ================================================================

    // MARK: Batch Inverse (BN254)

    /// Compute element-wise modular inverse: out[i] = a[i]^(-1) mod r.
    /// Uses Montgomery's trick on GPU: 1 Fermat inverse per 512-element chunk.
    /// Falls back to CPU for small arrays.
    public func batchInverse(_ a: [Fr]) throws -> [Fr] {
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchInverseCPU_BN254(a)
        }

        let aBuf = createFrBuffer(a)
        let outBuf = pool.allocate(size: n * MemoryLayout<Fr>.stride)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchInverseBN254)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 2)
        let chunkSize = 512
        let numGroups = (n + chunkSize - 1) / chunkSize
        let tg = min(64, Int(batchInverseBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            pool.release(buffer: aBuf); pool.release(buffer: outBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        let result = readFrBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: outBuf)
        return result
    }

    /// CPU fallback: Montgomery's trick for batch inverse.
    private func batchInverseCPU_BN254(_ a: [Fr]) -> [Fr] {
        let n = a.count
        guard n > 0 else { return [] }

        // Phase 1: prefix products
        var prefix = [Fr](repeating: Fr.zero, count: n)
        prefix[0] = a[0]
        for i in 1..<n {
            prefix[i] = frMul(prefix[i - 1], a[i])
        }

        // Phase 2: single inverse
        var inv = frInverse(prefix[n - 1])

        // Phase 3: backward sweep
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 1, by: -1) {
            result[i] = frMul(inv, prefix[i - 1])
            inv = frMul(inv, a[i])
        }
        result[0] = inv
        return result
    }

    // MARK: Batch Multiply (BN254)

    /// Element-wise multiply: out[i] = a[i] * b[i].
    public func batchMul(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchMulCPU_BN254(a, b)
        }

        let aBuf = createFrBuffer(a)
        let bBuf = createFrBuffer(b)
        let outBuf = pool.allocate(size: n * MemoryLayout<Fr>.stride)!
        try dispatchElementWise(batchMulBN254, aBuf, bBuf, outBuf, n: n)
        let result = readFrBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: bBuf); pool.release(buffer: outBuf)
        return result
    }

    private func batchMulCPU_BN254(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let n = a.count
        var result = [Fr](repeating: Fr.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    // MARK: Batch Add (BN254)

    /// Element-wise add: out[i] = a[i] + b[i].
    public func batchAdd(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchAddCPU_BN254(a, b)
        }

        let aBuf = createFrBuffer(a)
        let bBuf = createFrBuffer(b)
        let outBuf = pool.allocate(size: n * MemoryLayout<Fr>.stride)!
        try dispatchElementWise(batchAddBN254, aBuf, bBuf, outBuf, n: n)
        let result = readFrBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: bBuf); pool.release(buffer: outBuf)
        return result
    }

    private func batchAddCPU_BN254(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let n = a.count
        var result = [Fr](repeating: Fr.zero, count: n)
        a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_add(
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    // MARK: Batch Polynomial Evaluation (BN254)

    /// Evaluate polynomial at multiple points using Horner's method.
    /// coeffs: polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...).
    /// points: evaluation points.
    /// Returns: results[i] = poly(points[i]).
    public func batchEval(coeffs: [Fr], points: [Fr]) throws -> [Fr] {
        let degree = coeffs.count
        let numPoints = points.count
        guard degree > 0, numPoints > 0 else { return [] }

        if numPoints < gpuThreshold {
            return batchEvalCPU_BN254(coeffs: coeffs, points: points)
        }

        let coeffsBuf = createFrBuffer(coeffs)
        let pointsBuf = createFrBuffer(points)
        let resultsBuf = pool.allocate(size: numPoints * MemoryLayout<Fr>.stride)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchEvalBN254)
        enc.setBuffer(coeffsBuf, offset: 0, index: 0)
        enc.setBuffer(pointsBuf, offset: 0, index: 1)
        enc.setBuffer(resultsBuf, offset: 0, index: 2)
        var degVal = UInt32(degree)
        var npVal = UInt32(numPoints)
        enc.setBytes(&degVal, length: 4, index: 3)
        enc.setBytes(&npVal, length: 4, index: 4)
        let tg = min(256, Int(batchEvalBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            pool.release(buffer: coeffsBuf); pool.release(buffer: pointsBuf); pool.release(buffer: resultsBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        let result = readFrBuffer(resultsBuf, count: numPoints)
        pool.release(buffer: coeffsBuf); pool.release(buffer: pointsBuf); pool.release(buffer: resultsBuf)
        return result
    }

    /// CPU fallback: Horner evaluation at each point.
    private func batchEvalCPU_BN254(coeffs: [Fr], points: [Fr]) -> [Fr] {
        let degree = coeffs.count
        return points.map { x in
            var result = coeffs[degree - 1]
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = frAdd(frMul(result, x), coeffs[i])
            }
            return result
        }
    }

    // ================================================================
    // MARK: - BabyBear Operations
    // ================================================================

    // MARK: Batch Inverse (BabyBear)

    /// Compute element-wise modular inverse for BabyBear field elements.
    public func batchInverseBabyBear(_ a: [Bb]) throws -> [Bb] {
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchInverseCPU_BB(a)
        }

        let aBuf = createBbBuffer(a)
        let outBuf = pool.allocate(size: n * MemoryLayout<Bb>.stride)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchInverseBB)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 2)
        let chunkSize = 2048
        let numGroups = (n + chunkSize - 1) / chunkSize
        let tg = min(64, Int(batchInverseBB.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            pool.release(buffer: aBuf); pool.release(buffer: outBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        let result = readBbBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: outBuf)
        return result
    }

    private func batchInverseCPU_BB(_ a: [Bb]) -> [Bb] {
        let n = a.count
        guard n > 0 else { return [] }

        var prefix = [Bb](repeating: Bb.zero, count: n)
        prefix[0] = a[0]
        for i in 1..<n {
            prefix[i] = bbMul(prefix[i - 1], a[i])
        }

        var inv = bbInverse(prefix[n - 1])

        var result = [Bb](repeating: Bb.zero, count: n)
        for i in stride(from: n - 1, through: 1, by: -1) {
            result[i] = bbMul(inv, prefix[i - 1])
            inv = bbMul(inv, a[i])
        }
        result[0] = inv
        return result
    }

    // MARK: Batch Multiply (BabyBear)

    /// Element-wise multiply for BabyBear: out[i] = a[i] * b[i].
    public func batchMulBabyBear(_ a: [Bb], _ b: [Bb]) throws -> [Bb] {
        precondition(a.count == b.count)
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchMulCPU_BB(a, b)
        }

        let aBuf = createBbBuffer(a)
        let bBuf = createBbBuffer(b)
        let outBuf = pool.allocate(size: n * MemoryLayout<Bb>.stride)!
        try dispatchElementWise(batchMulBB, aBuf, bBuf, outBuf, n: n)
        let result = readBbBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: bBuf); pool.release(buffer: outBuf)
        return result
    }

    private func batchMulCPU_BB(_ a: [Bb], _ b: [Bb]) -> [Bb] {
        var result = [Bb](repeating: Bb.zero, count: a.count)
        for i in 0..<a.count {
            result[i] = bbMul(a[i], b[i])
        }
        return result
    }

    // MARK: Batch Add (BabyBear)

    /// Element-wise add for BabyBear: out[i] = a[i] + b[i].
    public func batchAddBabyBear(_ a: [Bb], _ b: [Bb]) throws -> [Bb] {
        precondition(a.count == b.count)
        let n = a.count
        guard n > 0 else { return [] }

        if n < gpuThreshold {
            return batchAddCPU_BB(a, b)
        }

        let aBuf = createBbBuffer(a)
        let bBuf = createBbBuffer(b)
        let outBuf = pool.allocate(size: n * MemoryLayout<Bb>.stride)!
        try dispatchElementWise(batchAddBB, aBuf, bBuf, outBuf, n: n)
        let result = readBbBuffer(outBuf, count: n)
        pool.release(buffer: aBuf); pool.release(buffer: bBuf); pool.release(buffer: outBuf)
        return result
    }

    private func batchAddCPU_BB(_ a: [Bb], _ b: [Bb]) -> [Bb] {
        var result = [Bb](repeating: Bb.zero, count: a.count)
        for i in 0..<a.count {
            result[i] = bbAdd(a[i], b[i])
        }
        return result
    }

    // MARK: Batch Polynomial Evaluation (BabyBear)

    /// Evaluate polynomial at multiple points (BabyBear, Horner's method).
    public func batchEvalBabyBear(coeffs: [Bb], points: [Bb]) throws -> [Bb] {
        let degree = coeffs.count
        let numPoints = points.count
        guard degree > 0, numPoints > 0 else { return [] }

        if numPoints < gpuThreshold {
            return batchEvalCPU_BB(coeffs: coeffs, points: points)
        }

        let coeffsBuf = createBbBuffer(coeffs)
        let pointsBuf = createBbBuffer(points)
        let resultsBuf = pool.allocate(size: numPoints * MemoryLayout<Bb>.stride)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchEvalBB)
        enc.setBuffer(coeffsBuf, offset: 0, index: 0)
        enc.setBuffer(pointsBuf, offset: 0, index: 1)
        enc.setBuffer(resultsBuf, offset: 0, index: 2)
        var degVal = UInt32(degree)
        var npVal = UInt32(numPoints)
        enc.setBytes(&degVal, length: 4, index: 3)
        enc.setBytes(&npVal, length: 4, index: 4)
        let tg = min(256, Int(batchEvalBB.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            pool.release(buffer: coeffsBuf); pool.release(buffer: pointsBuf); pool.release(buffer: resultsBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        let result = readBbBuffer(resultsBuf, count: numPoints)
        pool.release(buffer: coeffsBuf); pool.release(buffer: pointsBuf); pool.release(buffer: resultsBuf)
        return result
    }

    private func batchEvalCPU_BB(coeffs: [Bb], points: [Bb]) -> [Bb] {
        let degree = coeffs.count
        return points.map { x in
            var result = coeffs[degree - 1]
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = bbAdd(bbMul(result, x), coeffs[i])
            }
            return result
        }
    }
}
