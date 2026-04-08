// GPU-accelerated Horner polynomial evaluation engine
//
// Evaluates p(x) = a_0 + a_1*x + ... + a_n*x^n at multiple points on GPU.
// Each GPU thread evaluates at one point using Horner's method.
//
// API:
//   evaluate(coeffs:points:)       -- evaluate single poly at many points (GPU)
//   evaluateSingle(coeffs:point:)  -- evaluate at one point (CPU, for small inputs)
//   batchEvaluate(polys:point:)    -- evaluate many polys at one point
//
// Uses BN254 Fr field arithmetic. CPU fallback when numPoints * degree < threshold.

import Foundation
import Metal

// MARK: - GPUHornerEval

public class GPUHornerEval {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let hornerPipeline: MTLComputePipelineState
    private let cachedPipeline: MTLComputePipelineState
    private let batchPipeline: MTLComputePipelineState

    /// Minimum work (numPoints * degree) before dispatching to GPU.
    /// Below this threshold, CPU Horner is used instead.
    public var gpuWorkThreshold: Int = 4096

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUHornerEval.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "horner_eval_bn254"),
              let fn2 = library.makeFunction(name: "horner_eval_cached_bn254"),
              let fn3 = library.makeFunction(name: "horner_eval_batch_bn254") else {
            throw MSMError.missingKernel
        }

        self.hornerPipeline = try device.makeComputePipelineState(function: fn1)
        self.cachedPipeline = try device.makeComputePipelineState(function: fn2)
        self.batchPipeline = try device.makeComputePipelineState(function: fn3)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldBn254 = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let hornerShader = try String(contentsOfFile: shaderDir + "/poly/horner_eval.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define BN254") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldBn254) + "\n" + clean(hornerShader)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/bn254_fr.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - evaluate(coeffs:points:) -- single poly at many points

    /// Evaluate polynomial with given coefficients (ascending degree order) at multiple points.
    /// Returns one Fr result per point. Uses GPU when work exceeds threshold, CPU otherwise.
    public func evaluate(coeffs: [Fr], points: [Fr]) throws -> [Fr] {
        let degree = coeffs.count
        let numPoints = points.count
        guard degree >= 1 && numPoints >= 1 else { throw MSMError.invalidInput }

        let work = degree * numPoints
        if work < gpuWorkThreshold {
            return points.map { evaluateSingle(coeffs: coeffs, point: $0) }
        }

        return try gpuEvaluate(coeffs: coeffs, points: points)
    }

    // MARK: - evaluateSingle(coeffs:point:) -- CPU Horner for one point

    /// Evaluate polynomial at a single point using CPU Horner's method.
    /// Always runs on CPU -- use this for small evaluations or single-point queries.
    public func evaluateSingle(coeffs: [Fr], point: Fr) -> Fr {
        guard !coeffs.isEmpty else { return Fr.zero }
        var result = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), coeffs[i])
        }
        return result
    }

    // MARK: - batchEvaluate(polys:point:) -- many polys at one point

    /// Evaluate M polynomials (all same degree) at a single point.
    /// Returns M evaluations.
    public func batchEvaluate(polys: [[Fr]], point: Fr) throws -> [Fr] {
        let numPolys = polys.count
        guard numPolys >= 1 else { throw MSMError.invalidInput }
        let degree = polys[0].count
        guard degree >= 1 else { throw MSMError.invalidInput }

        let work = numPolys * degree
        if work < gpuWorkThreshold {
            return polys.map { evaluateSingle(coeffs: $0, point: point) }
        }

        return try gpuBatchEvaluate(polys: polys, point: point)
    }

    // MARK: - GPU dispatch: evaluate

    private func gpuEvaluate(coeffs: [Fr], points: [Fr]) throws -> [Fr] {
        let degree = coeffs.count
        let numPoints = points.count
        let elemSize = 32 // 8 x UInt32

        // Choose cached pipeline for polynomials that fit in threadgroup memory
        let pipeline = degree <= 512 ? cachedPipeline : hornerPipeline

        // Pack coefficients as raw bytes
        let coeffsBuf = coeffs.withUnsafeBytes { buf in
            device.makeBuffer(bytes: buf.baseAddress!, length: degree * elemSize,
                              options: .storageModeShared)!
        }
        let pointsBuf = points.withUnsafeBytes { buf in
            device.makeBuffer(bytes: buf.baseAddress!, length: numPoints * elemSize,
                              options: .storageModeShared)!
        }
        let resultBuf = device.makeBuffer(length: numPoints * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(coeffsBuf, offset: 0, index: 0)
        enc.setBuffer(pointsBuf, offset: 0, index: 1)
        enc.setBuffer(resultBuf, offset: 0, index: 2)
        var deg = UInt32(degree)
        var nPts = UInt32(numPoints)
        enc.setBytes(&deg, length: 4, index: 3)
        enc.setBytes(&nPts, length: 4, index: 4)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let outCount = numPoints * 8
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: outCount)
        let raw = Array(UnsafeBufferPointer(start: ptr, count: outCount))
        return stride(from: 0, to: raw.count, by: 8).map { base in
            Fr(v: (raw[base], raw[base+1], raw[base+2], raw[base+3],
                   raw[base+4], raw[base+5], raw[base+6], raw[base+7]))
        }
    }

    // MARK: - GPU dispatch: batch evaluate

    private func gpuBatchEvaluate(polys: [[Fr]], point: Fr) throws -> [Fr] {
        let numPolys = polys.count
        let degree = polys[0].count
        let elemSize = 32

        // Pack all polynomials contiguously into GPU buffer
        let totalBytes = numPolys * degree * elemSize
        let polysBuf = device.makeBuffer(length: totalBytes, options: .storageModeShared)!
        let dst = polysBuf.contents()
        for (pi, poly) in polys.enumerated() {
            poly.withUnsafeBytes { src in
                memcpy(dst + pi * degree * elemSize, src.baseAddress!, degree * elemSize)
            }
        }

        let pointBuf: MTLBuffer = withUnsafeBytes(of: point) { buf in
            device.makeBuffer(bytes: buf.baseAddress!, length: elemSize, options: .storageModeShared)!
        }
        let resultBuf = device.makeBuffer(length: numPolys * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchPipeline)
        enc.setBuffer(polysBuf, offset: 0, index: 0)
        enc.setBuffer(pointBuf, offset: 0, index: 1)
        enc.setBuffer(resultBuf, offset: 0, index: 2)
        var deg = UInt32(degree)
        var nPolys = UInt32(numPolys)
        enc.setBytes(&deg, length: 4, index: 3)
        enc.setBytes(&nPolys, length: 4, index: 4)
        let tg = min(256, Int(batchPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPolys, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let outCount = numPolys * 8
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: outCount)
        let raw = Array(UnsafeBufferPointer(start: ptr, count: outCount))
        return stride(from: 0, to: raw.count, by: 8).map { base in
            Fr(v: (raw[base], raw[base+1], raw[base+2], raw[base+3],
                   raw[base+4], raw[base+5], raw[base+6], raw[base+7]))
        }
    }

    // MARK: - Fr conversion helpers

    private func frToU32(_ f: Fr) -> [UInt32] {
        [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
    }
}
