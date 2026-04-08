// GPU-accelerated multi-point polynomial evaluation engine
//
// Evaluates one or many polynomials at many points simultaneously.
// Each GPU thread runs Horner's method for one (polynomial, point) pair.
// Critical for KZG openings, FRI, and STARK provers.
//
// Supports BN254 Fr (256-bit Montgomery), BabyBear (32-bit), Goldilocks (64-bit).
// CPU fallback for small inputs (<64 points).

import Foundation
import Metal

// FieldType is defined in GPUSumcheckEngine.swift

// MARK: - GPUPolyEvalEngine

public class GPUPolyEvalEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Single-poly kernels
    private let evalBn254: MTLComputePipelineState
    private let evalBabyBear: MTLComputePipelineState
    private let evalGoldilocks: MTLComputePipelineState

    // Batch kernels
    private let evalBatchBn254: MTLComputePipelineState
    private let evalBatchBabyBear: MTLComputePipelineState
    private let evalBatchGoldilocks: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUPolyEvalEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "poly_eval_bn254"),
              let fn2 = library.makeFunction(name: "poly_eval_babybear"),
              let fn3 = library.makeFunction(name: "poly_eval_goldilocks"),
              let fn4 = library.makeFunction(name: "poly_eval_batch_bn254"),
              let fn5 = library.makeFunction(name: "poly_eval_batch_babybear"),
              let fn6 = library.makeFunction(name: "poly_eval_batch_goldilocks") else {
            throw MSMError.missingKernel
        }

        self.evalBn254 = try device.makeComputePipelineState(function: fn1)
        self.evalBabyBear = try device.makeComputePipelineState(function: fn2)
        self.evalGoldilocks = try device.makeComputePipelineState(function: fn3)
        self.evalBatchBn254 = try device.makeComputePipelineState(function: fn4)
        self.evalBatchBabyBear = try device.makeComputePipelineState(function: fn5)
        self.evalBatchGoldilocks = try device.makeComputePipelineState(function: fn6)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldBn254 = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let fieldGl = try String(contentsOfFile: shaderDir + "/fields/goldilocks.metal", encoding: .utf8)
        let multiEval = try String(contentsOfFile: shaderDir + "/poly/multi_eval.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { line in
                    if line.contains("#include") || line.contains("#ifndef") || line.contains("#endif") { return false }
                    if line.contains("#define") {
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        let parts = trimmed.split(separator: " ", maxSplits: 3)
                        return parts.count >= 3
                    }
                    return true
                }
                .joined(separator: "\n")
        }

        let combined = clean(fieldBn254) + "\n" + clean(fieldBb) + "\n" +
                        clean(fieldGl) + "\n" + clean(multiEval)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/babybear.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/babybear.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Element size helper

    private func elementSize(for field: FieldType) -> Int {
        switch field {
        case .bn254: return 32    // 8 x UInt32
        case .babybear: return 4    // 1 x UInt32
        case .goldilocks: return 8  // 1 x UInt64
        }
    }

    // MARK: - Single polynomial evaluation

    /// Evaluate one polynomial at many points.
    /// coeffs/points/result are raw UInt32 arrays. For BN254 Fr each element is 8 x UInt32,
    /// for BabyBear each element is 1 x UInt32, for Goldilocks each element is 2 x UInt32 (one UInt64).
    /// Returns array of evaluations, one per point.
    public func evaluate(coeffs: [UInt32], points: [UInt32], field: FieldType) throws -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        let degree = coeffs.count / elemWords
        let numPoints = points.count / elemWords

        guard degree >= 1 && numPoints >= 1 else {
            throw MSMError.invalidInput
        }
        guard coeffs.count == degree * elemWords && points.count == numPoints * elemWords else {
            throw MSMError.invalidInput
        }

        // CPU fallback for small inputs
        if numPoints < 64 {
            return cpuEvaluate(coeffs: coeffs, points: points, field: field, degree: degree, numPoints: numPoints)
        }

        return try gpuEvaluateSingle(coeffs: coeffs, points: points, field: field,
                                      degree: degree, numPoints: numPoints)
    }

    /// MTLBuffer-level API for zero-copy integration.
    /// coeffsBuf contains `degree` field elements; pointsBuf contains `numPoints` field elements.
    /// Returns an MTLBuffer containing `numPoints` field elements.
    public func evaluate(coeffsBuf: MTLBuffer, pointsBuf: MTLBuffer,
                         degree: Int, numPoints: Int, field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let outSize = numPoints * elemSize
        guard let resultBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate result buffer")
        }

        let pipeline = singlePipeline(for: field)

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

        return resultBuf
    }

    // MARK: - Batch polynomial evaluation

    /// Evaluate M polynomials (all same degree) at N points.
    /// polys: array of M polynomials, each as [UInt32] of length degree*elemWords.
    /// points: [UInt32] of length numPoints*elemWords.
    /// Returns M arrays of evaluations, each of length numPoints*elemWords.
    public func evaluateBatch(polys: [[UInt32]], points: [UInt32], field: FieldType) throws -> [[UInt32]] {
        let elemWords = elementSize(for: field) / 4
        let numPolys = polys.count
        let numPoints = points.count / elemWords

        guard numPolys >= 1 && numPoints >= 1 else {
            throw MSMError.invalidInput
        }

        let degree = polys[0].count / elemWords
        guard degree >= 1 else { throw MSMError.invalidInput }

        // Validate all polys have same size
        for p in polys {
            guard p.count == degree * elemWords else { throw MSMError.invalidInput }
        }

        // CPU fallback for small inputs
        if numPoints < 64 && numPolys <= 4 {
            var results = [[UInt32]]()
            results.reserveCapacity(numPolys)
            for p in polys {
                results.append(cpuEvaluate(coeffs: p, points: points, field: field,
                                           degree: degree, numPoints: numPoints))
            }
            return results
        }

        return try gpuEvaluateBatch(polys: polys, points: points, field: field,
                                     degree: degree, numPoints: numPoints, numPolys: numPolys)
    }

    /// MTLBuffer-level batch API for zero-copy integration.
    /// polysBuf contains M*degree field elements packed contiguously.
    /// Returns an MTLBuffer containing M*numPoints field elements.
    public func evaluateBatch(polysBuf: MTLBuffer, pointsBuf: MTLBuffer,
                              degree: Int, numPoints: Int, numPolys: Int,
                              field: FieldType) throws -> MTLBuffer {
        let elemSize = elementSize(for: field)
        let outSize = numPolys * numPoints * elemSize
        guard let resultBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate result buffer")
        }

        let pipeline = batchPipeline(for: field)
        let totalThreads = numPolys * numPoints

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(polysBuf, offset: 0, index: 0)
        enc.setBuffer(pointsBuf, offset: 0, index: 1)
        enc.setBuffer(resultBuf, offset: 0, index: 2)
        var deg = UInt32(degree)
        var nPts = UInt32(numPoints)
        var nPolys = UInt32(numPolys)
        enc.setBytes(&deg, length: 4, index: 3)
        enc.setBytes(&nPts, length: 4, index: 4)
        enc.setBytes(&nPolys, length: 4, index: 5)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return resultBuf
    }

    // MARK: - Pipeline selectors

    private func singlePipeline(for field: FieldType) -> MTLComputePipelineState {
        switch field {
        case .bn254: return evalBn254
        case .babybear: return evalBabyBear
        case .goldilocks: return evalGoldilocks
        }
    }

    private func batchPipeline(for field: FieldType) -> MTLComputePipelineState {
        switch field {
        case .bn254: return evalBatchBn254
        case .babybear: return evalBatchBabyBear
        case .goldilocks: return evalBatchGoldilocks
        }
    }

    // MARK: - GPU dispatch helpers

    private func gpuEvaluateSingle(coeffs: [UInt32], points: [UInt32], field: FieldType,
                                    degree: Int, numPoints: Int) throws -> [UInt32] {
        let elemSize = elementSize(for: field)

        let coeffsBuf = device.makeBuffer(bytes: coeffs, length: degree * elemSize,
                                           options: .storageModeShared)!
        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize,
                                           options: .storageModeShared)!
        let resultBuf = try evaluate(coeffsBuf: coeffsBuf, pointsBuf: pointsBuf,
                                      degree: degree, numPoints: numPoints, field: field)

        let elemWords = elemSize / 4
        let outCount = numPoints * elemWords
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }

    private func gpuEvaluateBatch(polys: [[UInt32]], points: [UInt32], field: FieldType,
                                   degree: Int, numPoints: Int, numPolys: Int) throws -> [[UInt32]] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4

        // Pack all polynomials contiguously
        var packed = [UInt32]()
        packed.reserveCapacity(numPolys * degree * elemWords)
        for p in polys { packed.append(contentsOf: p) }

        let polysBuf = device.makeBuffer(bytes: packed, length: numPolys * degree * elemSize,
                                          options: .storageModeShared)!
        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize,
                                           options: .storageModeShared)!
        let resultBuf = try evaluateBatch(polysBuf: polysBuf, pointsBuf: pointsBuf,
                                           degree: degree, numPoints: numPoints,
                                           numPolys: numPolys, field: field)

        // Unpack results
        let totalWords = numPolys * numPoints * elemWords
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: totalWords)
        let flat = Array(UnsafeBufferPointer(start: ptr, count: totalWords))

        var results = [[UInt32]]()
        results.reserveCapacity(numPolys)
        let rowWords = numPoints * elemWords
        for i in 0..<numPolys {
            results.append(Array(flat[i * rowWords ..< (i + 1) * rowWords]))
        }
        return results
    }

    // MARK: - CPU fallback (Horner's method)

    private func cpuEvaluate(coeffs: [UInt32], points: [UInt32], field: FieldType,
                              degree: Int, numPoints: Int) -> [UInt32] {
        switch field {
        case .bn254:
            return cpuEvalBn254(coeffs: coeffs, points: points, degree: degree, numPoints: numPoints)
        case .babybear:
            return cpuEvalBabyBear(coeffs: coeffs, points: points, degree: degree, numPoints: numPoints)
        case .goldilocks:
            return cpuEvalGoldilocks(coeffs: coeffs, points: points, degree: degree, numPoints: numPoints)
        }
    }

    private func cpuEvalBn254(coeffs: [UInt32], points: [UInt32], degree: Int, numPoints: Int) -> [UInt32] {
        // Convert UInt32 arrays to Fr structs
        var coeffsFr = [Fr]()
        coeffsFr.reserveCapacity(degree)
        for i in 0..<degree {
            let base = i * 8
            coeffsFr.append(Fr(v: (coeffs[base], coeffs[base+1], coeffs[base+2], coeffs[base+3],
                                   coeffs[base+4], coeffs[base+5], coeffs[base+6], coeffs[base+7])))
        }

        var results = [UInt32]()
        results.reserveCapacity(numPoints * 8)

        for p in 0..<numPoints {
            let base = p * 8
            let x = Fr(v: (points[base], points[base+1], points[base+2], points[base+3],
                           points[base+4], points[base+5], points[base+6], points[base+7]))

            var result = coeffsFr[degree - 1]
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = frAdd(frMul(result, x), coeffsFr[i])
            }

            results.append(result.v.0); results.append(result.v.1)
            results.append(result.v.2); results.append(result.v.3)
            results.append(result.v.4); results.append(result.v.5)
            results.append(result.v.6); results.append(result.v.7)
        }
        return results
    }

    private func cpuEvalBabyBear(coeffs: [UInt32], points: [UInt32], degree: Int, numPoints: Int) -> [UInt32] {
        var results = [UInt32]()
        results.reserveCapacity(numPoints)

        for p in 0..<numPoints {
            let x = Bb(v: points[p])
            var result = Bb(v: coeffs[degree - 1])
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = bbAdd(bbMul(result, x), Bb(v: coeffs[i]))
            }
            results.append(result.v)
        }
        return results
    }

    private func cpuEvalGoldilocks(coeffs: [UInt32], points: [UInt32], degree: Int, numPoints: Int) -> [UInt32] {
        // UInt32 pairs -> UInt64
        func toU64(_ arr: [UInt32], _ idx: Int) -> UInt64 {
            UInt64(arr[idx * 2]) | (UInt64(arr[idx * 2 + 1]) << 32)
        }

        var results = [UInt32]()
        results.reserveCapacity(numPoints * 2)

        for p in 0..<numPoints {
            let x = Gl(v: toU64(points, p))
            var result = Gl(v: toU64(coeffs, degree - 1))
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = glAdd(glMul(result, x), Gl(v: toU64(coeffs, i)))
            }
            results.append(UInt32(result.v & 0xFFFFFFFF))
            results.append(UInt32(result.v >> 32))
        }
        return results
    }
}
