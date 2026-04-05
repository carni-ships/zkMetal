// GPU-accelerated multi-point polynomial evaluation engine
//
// Evaluates polynomials at sets of points with three dispatch modes:
//   1. evaluate(poly:points:)       — single poly at many points (Horner, 1 thread/point)
//   2. batchEvaluate(polys:point:)  — many polys at one point (1 thread/poly)
//   3. crossEvaluate(polys:points:) — M polys x N points matrix (MxN threads)
//
// Supports BN254 Fr (256-bit Montgomery) and BabyBear (32-bit).
// CPU fallback for small inputs below GPU dispatch threshold.

import Foundation
import Metal

// MARK: - GPUMultiPointEval

public class GPUMultiPointEval {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Horner: single poly, many points
    private let hornerBn254: MTLComputePipelineState
    private let hornerBabyBear: MTLComputePipelineState

    // Batch Horner: many polys, one point
    private let batchBn254: MTLComputePipelineState
    private let batchBabyBear: MTLComputePipelineState

    // Cross Horner: M polys x N points
    private let crossBn254: MTLComputePipelineState
    private let crossBabyBear: MTLComputePipelineState

    /// Minimum number of GPU work items before dispatching to GPU.
    /// Below this threshold, CPU Horner is used instead.
    public var gpuThreshold: Int = 64

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUMultiPointEval.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "mpe_horner_bn254"),
              let fn2 = library.makeFunction(name: "mpe_horner_babybear"),
              let fn3 = library.makeFunction(name: "mpe_batch_horner_bn254"),
              let fn4 = library.makeFunction(name: "mpe_batch_horner_babybear"),
              let fn5 = library.makeFunction(name: "mpe_cross_horner_bn254"),
              let fn6 = library.makeFunction(name: "mpe_cross_horner_babybear") else {
            throw MSMError.missingKernel
        }

        self.hornerBn254 = try device.makeComputePipelineState(function: fn1)
        self.hornerBabyBear = try device.makeComputePipelineState(function: fn2)
        self.batchBn254 = try device.makeComputePipelineState(function: fn3)
        self.batchBabyBear = try device.makeComputePipelineState(function: fn4)
        self.crossBn254 = try device.makeComputePipelineState(function: fn5)
        self.crossBabyBear = try device.makeComputePipelineState(function: fn6)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldBn254 = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let mpeShader = try String(contentsOfFile: shaderDir + "/poly/multi_point_eval.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldBn254) + "\n" + clean(fieldBb) + "\n" + clean(mpeShader)

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
        case .babybear: return 4  // 1 x UInt32
        case .goldilocks: return 8
        }
    }

    // MARK: - evaluate(poly:points:) — single poly at many points

    /// Evaluate one polynomial at many points.
    /// coeffs and points are raw UInt32 arrays. For BN254 Fr each element is 8 x UInt32,
    /// for BabyBear each element is 1 x UInt32.
    /// Returns array of evaluations, one per point.
    public func evaluate(poly coeffs: [UInt32], points: [UInt32], field: FieldType = .bn254) throws -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        let degree = coeffs.count / elemWords
        let numPoints = points.count / elemWords

        guard degree >= 1 && numPoints >= 1 else { throw MSMError.invalidInput }
        guard coeffs.count == degree * elemWords && points.count == numPoints * elemWords else {
            throw MSMError.invalidInput
        }

        if numPoints < gpuThreshold {
            return cpuEvaluate(coeffs: coeffs, points: points, field: field, degree: degree, numPoints: numPoints)
        }

        return try gpuHorner(coeffs: coeffs, points: points, field: field, degree: degree, numPoints: numPoints)
    }

    /// Convenience for Fr arrays.
    public func evaluate(poly: [Fr], points: [Fr]) throws -> [Fr] {
        let coeffsU32 = poly.flatMap { frToU32($0) }
        let pointsU32 = points.flatMap { frToU32($0) }
        let resultU32 = try evaluate(poly: coeffsU32, points: pointsU32, field: .bn254)
        return stride(from: 0, to: resultU32.count, by: 8).map { base in
            Fr(v: (resultU32[base], resultU32[base+1], resultU32[base+2], resultU32[base+3],
                   resultU32[base+4], resultU32[base+5], resultU32[base+6], resultU32[base+7]))
        }
    }

    // MARK: - batchEvaluate(polys:point:) — many polys at one point

    /// Evaluate M polynomials (all same degree) at a single point.
    /// polys: array of M polynomials, each as [UInt32].
    /// point: [UInt32] for one field element.
    /// Returns M evaluations as [UInt32].
    public func batchEvaluate(polys: [[UInt32]], point: [UInt32], field: FieldType = .bn254) throws -> [UInt32] {
        let elemWords = elementSize(for: field) / 4
        let numPolys = polys.count
        guard numPolys >= 1 else { throw MSMError.invalidInput }
        guard point.count == elemWords else { throw MSMError.invalidInput }

        let degree = polys[0].count / elemWords
        guard degree >= 1 else { throw MSMError.invalidInput }
        for p in polys {
            guard p.count == degree * elemWords else { throw MSMError.invalidInput }
        }

        if numPolys < gpuThreshold {
            return cpuBatchEvaluate(polys: polys, point: point, field: field, degree: degree, numPolys: numPolys)
        }

        return try gpuBatchHorner(polys: polys, point: point, field: field, degree: degree, numPolys: numPolys)
    }

    /// Convenience for Fr arrays.
    public func batchEvaluate(polys: [[Fr]], point: Fr) throws -> [Fr] {
        let polysU32 = polys.map { $0.flatMap { frToU32($0) } }
        let pointU32 = frToU32(point)
        let resultU32 = try batchEvaluate(polys: polysU32, point: pointU32, field: .bn254)
        return stride(from: 0, to: resultU32.count, by: 8).map { base in
            Fr(v: (resultU32[base], resultU32[base+1], resultU32[base+2], resultU32[base+3],
                   resultU32[base+4], resultU32[base+5], resultU32[base+6], resultU32[base+7]))
        }
    }

    // MARK: - crossEvaluate(polys:points:) — M x N evaluation matrix

    /// Evaluate M polynomials at N points, returning M x N matrix.
    /// All polys must have the same degree.
    /// Returns array of M arrays, each with N evaluations.
    public func crossEvaluate(polys: [[UInt32]], points: [UInt32], field: FieldType = .bn254) throws -> [[UInt32]] {
        let elemWords = elementSize(for: field) / 4
        let numPolys = polys.count
        let numPoints = points.count / elemWords

        guard numPolys >= 1 && numPoints >= 1 else { throw MSMError.invalidInput }

        let degree = polys[0].count / elemWords
        guard degree >= 1 else { throw MSMError.invalidInput }
        for p in polys {
            guard p.count == degree * elemWords else { throw MSMError.invalidInput }
        }

        let totalWork = numPolys * numPoints
        if totalWork < gpuThreshold {
            return cpuCrossEvaluate(polys: polys, points: points, field: field,
                                     degree: degree, numPoints: numPoints, numPolys: numPolys)
        }

        return try gpuCrossHorner(polys: polys, points: points, field: field,
                                   degree: degree, numPoints: numPoints, numPolys: numPolys)
    }

    /// Convenience for Fr arrays.
    public func crossEvaluate(polys: [[Fr]], points: [Fr]) throws -> [[Fr]] {
        let polysU32 = polys.map { $0.flatMap { frToU32($0) } }
        let pointsU32 = points.flatMap { frToU32($0) }
        let resultRows = try crossEvaluate(polys: polysU32, points: pointsU32, field: .bn254)
        return resultRows.map { row in
            stride(from: 0, to: row.count, by: 8).map { base in
                Fr(v: (row[base], row[base+1], row[base+2], row[base+3],
                       row[base+4], row[base+5], row[base+6], row[base+7]))
            }
        }
    }

    // MARK: - GPU dispatch: Horner (single poly, many points)

    private func gpuHorner(coeffs: [UInt32], points: [UInt32], field: FieldType,
                            degree: Int, numPoints: Int) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let pipeline: MTLComputePipelineState
        switch field {
        case .bn254: pipeline = hornerBn254
        case .babybear: pipeline = hornerBabyBear
        default: throw MSMError.gpuError("Unsupported field for GPUMultiPointEval")
        }

        let coeffsBuf = device.makeBuffer(bytes: coeffs, length: degree * elemSize,
                                            options: .storageModeShared)!
        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize,
                                            options: .storageModeShared)!
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

        let elemWords = elemSize / 4
        let outCount = numPoints * elemWords
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }

    // MARK: - GPU dispatch: Batch Horner (many polys, one point)

    private func gpuBatchHorner(polys: [[UInt32]], point: [UInt32], field: FieldType,
                                 degree: Int, numPolys: Int) throws -> [UInt32] {
        let elemSize = elementSize(for: field)
        let pipeline: MTLComputePipelineState
        switch field {
        case .bn254: pipeline = batchBn254
        case .babybear: pipeline = batchBabyBear
        default: throw MSMError.gpuError("Unsupported field for GPUMultiPointEval batch")
        }

        // Pack all polynomials contiguously
        var packed = [UInt32]()
        packed.reserveCapacity(numPolys * polys[0].count)
        for p in polys { packed.append(contentsOf: p) }

        let polysBuf = device.makeBuffer(bytes: packed, length: numPolys * degree * elemSize,
                                           options: .storageModeShared)!
        let pointBuf = device.makeBuffer(bytes: point, length: elemSize,
                                          options: .storageModeShared)!
        let resultBuf = device.makeBuffer(length: numPolys * elemSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(polysBuf, offset: 0, index: 0)
        enc.setBuffer(pointBuf, offset: 0, index: 1)
        enc.setBuffer(resultBuf, offset: 0, index: 2)
        var deg = UInt32(degree)
        var nPolys = UInt32(numPolys)
        enc.setBytes(&deg, length: 4, index: 3)
        enc.setBytes(&nPolys, length: 4, index: 4)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPolys, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let elemWords = elemSize / 4
        let outCount = numPolys * elemWords
        let ptr = resultBuf.contents().bindMemory(to: UInt32.self, capacity: outCount)
        return Array(UnsafeBufferPointer(start: ptr, count: outCount))
    }

    // MARK: - GPU dispatch: Cross Horner (M polys x N points)

    private func gpuCrossHorner(polys: [[UInt32]], points: [UInt32], field: FieldType,
                                 degree: Int, numPoints: Int, numPolys: Int) throws -> [[UInt32]] {
        let elemSize = elementSize(for: field)
        let elemWords = elemSize / 4
        let pipeline: MTLComputePipelineState
        switch field {
        case .bn254: pipeline = crossBn254
        case .babybear: pipeline = crossBabyBear
        default: throw MSMError.gpuError("Unsupported field for GPUMultiPointEval cross")
        }

        var packed = [UInt32]()
        packed.reserveCapacity(numPolys * polys[0].count)
        for p in polys { packed.append(contentsOf: p) }

        let totalThreads = numPolys * numPoints
        let polysBuf = device.makeBuffer(bytes: packed, length: numPolys * degree * elemSize,
                                           options: .storageModeShared)!
        let pointsBuf = device.makeBuffer(bytes: points, length: numPoints * elemSize,
                                            options: .storageModeShared)!
        let resultBuf = device.makeBuffer(length: totalThreads * elemSize, options: .storageModeShared)!

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

        let totalWords = totalThreads * elemWords
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

    // MARK: - CPU fallback: Horner

    private func cpuEvaluate(coeffs: [UInt32], points: [UInt32], field: FieldType,
                              degree: Int, numPoints: Int) -> [UInt32] {
        switch field {
        case .bn254:
            return cpuEvalBn254(coeffs: coeffs, points: points, degree: degree, numPoints: numPoints)
        case .babybear:
            return cpuEvalBabyBear(coeffs: coeffs, points: points, degree: degree, numPoints: numPoints)
        default:
            return [] // goldilocks not supported in this engine
        }
    }

    private func cpuEvalBn254(coeffs: [UInt32], points: [UInt32], degree: Int, numPoints: Int) -> [UInt32] {
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
        let p = UInt64(Bb.P)
        var results = [UInt32]()
        results.reserveCapacity(numPoints)

        for pt in 0..<numPoints {
            let x = UInt64(points[pt])
            var result = UInt64(coeffs[degree - 1])
            for i in stride(from: degree - 2, through: 0, by: -1) {
                result = (result * x + UInt64(coeffs[i])) % p
            }
            results.append(UInt32(result))
        }
        return results
    }

    // MARK: - CPU fallback: Batch

    private func cpuBatchEvaluate(polys: [[UInt32]], point: [UInt32], field: FieldType,
                                   degree: Int, numPolys: Int) -> [UInt32] {
        var results = [UInt32]()
        let elemWords = elementSize(for: field) / 4
        results.reserveCapacity(numPolys * elemWords)
        for poly in polys {
            let eval = cpuEvaluate(coeffs: poly, points: point, field: field, degree: degree, numPoints: 1)
            results.append(contentsOf: eval)
        }
        return results
    }

    // MARK: - CPU fallback: Cross

    private func cpuCrossEvaluate(polys: [[UInt32]], points: [UInt32], field: FieldType,
                                   degree: Int, numPoints: Int, numPolys: Int) -> [[UInt32]] {
        var results = [[UInt32]]()
        results.reserveCapacity(numPolys)
        for poly in polys {
            results.append(cpuEvaluate(coeffs: poly, points: points, field: field,
                                        degree: degree, numPoints: numPoints))
        }
        return results
    }

    // MARK: - Fr conversion helpers

    private func frToU32(_ f: Fr) -> [UInt32] {
        [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
    }
}
