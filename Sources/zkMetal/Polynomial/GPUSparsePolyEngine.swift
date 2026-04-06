// GPU-accelerated Sparse Polynomial Engine
//
// Provides GPU-parallel operations for sparse polynomials (few non-zero
// coefficients relative to degree). Uses the SparsePoly type from
// SparsePolyCommit.swift.
//
// Operations:
//   evaluate(poly:point:)          -- evaluate at single point (CPU, O(k) muls)
//   evaluateMulti(poly:points:)    -- evaluate at many points (GPU parallel)
//   toDense(poly:degree:)          -- convert sparse to dense coefficient array
//   mulDense(sparse:dense:)        -- multiply sparse by dense polynomial (GPU)
//
// Uses BN254 Fr field arithmetic. Falls back to CPU for small inputs.

import Foundation
import Metal

// MARK: - GPUSparsePolyEngine

public class GPUSparsePolyEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let evalPipeline: MTLComputePipelineState
    private let evalCachedPipeline: MTLComputePipelineState
    private let mulDensePipeline: MTLComputePipelineState

    /// Minimum work (numPoints * nnz) before dispatching to GPU.
    public var gpuWorkThreshold: Int = 2048

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUSparsePolyEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "sparse_eval_bn254"),
              let fn2 = library.makeFunction(name: "sparse_eval_cached_bn254"),
              let fn3 = library.makeFunction(name: "sparse_mul_dense_bn254") else {
            throw MSMError.missingKernel
        }

        self.evalPipeline = try device.makeComputePipelineState(function: fn1)
        self.evalCachedPipeline = try device.makeComputePipelineState(function: fn2)
        self.mulDensePipeline = try device.makeComputePipelineState(function: fn3)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSrc = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let sparseSrc = try String(contentsOfFile: shaderDir + "/poly/sparse_eval.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define BN254") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldSrc) + "\n" + clean(sparseSrc)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - evaluate(poly:point:) -- CPU single-point evaluation

    /// Evaluate a sparse polynomial at a single point using CPU.
    /// Delegates to SparsePoly.evaluate(at:) which uses incremental power computation.
    /// Cost: O(k) field multiplications where k = poly.nnz plus index gaps.
    public func evaluate(poly: SparsePoly, point: Fr) -> Fr {
        return poly.evaluate(at: point)
    }

    // MARK: - evaluateMulti(poly:points:) -- GPU multi-point evaluation

    /// Evaluate a sparse polynomial at multiple points in parallel on the GPU.
    /// Each GPU thread evaluates the polynomial at one point.
    /// Falls back to CPU when work is below threshold.
    ///
    /// - Parameters:
    ///   - poly: The sparse polynomial
    ///   - points: Array of evaluation points
    /// - Returns: Array of Fr evaluations, one per point
    public func evaluateMulti(poly: SparsePoly, points: [Fr]) throws -> [Fr] {
        let nnz = poly.nnz
        let numPoints = points.count
        guard nnz >= 1 && numPoints >= 1 else { return [Fr](repeating: Fr.zero, count: numPoints) }

        let work = nnz * numPoints
        if work < gpuWorkThreshold {
            return points.map { poly.evaluate(at: $0) }
        }

        return try gpuEvaluateMulti(poly: poly, points: points)
    }

    // MARK: - toDense(poly:degree:) -- convert to dense representation

    /// Convert a sparse polynomial to a dense coefficient array.
    /// If degree is provided, the output has that many entries; otherwise uses poly.degreeBound.
    /// This is a CPU operation (scatter into zeroed array).
    ///
    /// - Parameters:
    ///   - poly: The sparse polynomial
    ///   - degree: Output array length (defaults to poly.degreeBound)
    /// - Returns: Dense coefficient array
    public func toDense(poly: SparsePoly, degree: Int? = nil) -> [Fr] {
        let n = degree ?? poly.degreeBound
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for (idx, c) in poly.terms {
            if idx < n {
                coeffs[idx] = c
            }
        }
        return coeffs
    }

    // MARK: - mulDense(sparse:dense:) -- GPU sparse * dense multiplication

    /// Multiply a sparse polynomial by a dense polynomial on the GPU.
    /// Output degree = max_sparse_index + dense.count - 1.
    ///
    /// Each GPU thread computes one output coefficient by iterating over
    /// the sparse terms and gathering the corresponding dense coefficient.
    /// Cost: O(outputLen * nnz) total work, but parallelized across threads.
    ///
    /// Falls back to CPU for small inputs.
    ///
    /// - Parameters:
    ///   - sparse: The sparse polynomial
    ///   - dense: Dense coefficient array (ascending degree order)
    /// - Returns: Dense coefficient array of the product
    public func mulDense(sparse: SparsePoly, dense: [Fr]) throws -> [Fr] {
        let nnz = sparse.nnz
        guard nnz >= 1 && !dense.isEmpty else {
            return []
        }

        // Output length: highest sparse index + dense length
        let maxSparseIdx = sparse.terms.last!.index
        let outputLen = maxSparseIdx + dense.count

        let work = outputLen * nnz
        if work < gpuWorkThreshold {
            return cpuMulDense(sparse: sparse, dense: dense, outputLen: outputLen)
        }

        return try gpuMulDense(sparse: sparse, dense: dense, outputLen: outputLen)
    }

    // MARK: - CPU fallback for mulDense

    private func cpuMulDense(sparse: SparsePoly, dense: [Fr], outputLen: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: outputLen)
        for (sIdx, sCoeff) in sparse.terms {
            for (dIdx, dCoeff) in dense.enumerated() {
                let outIdx = sIdx + dIdx
                if outIdx < outputLen {
                    result[outIdx] = frAdd(result[outIdx], frMul(sCoeff, dCoeff))
                }
            }
        }
        return result
    }

    // MARK: - GPU dispatch: evaluateMulti

    private func gpuEvaluateMulti(poly: SparsePoly, points: [Fr]) throws -> [Fr] {
        let nnz = poly.nnz
        let numPoints = points.count
        let elemSize = 32 // 8 x UInt32

        // Pack sparse term indices as UInt32 array
        var indices = [UInt32]()
        indices.reserveCapacity(nnz)
        for (idx, _) in poly.terms {
            indices.append(UInt32(idx))
        }

        // Pack sparse term coefficients
        var coeffsU32 = [UInt32]()
        coeffsU32.reserveCapacity(nnz * 8)
        for (_, coeff) in poly.terms {
            coeffsU32.append(contentsOf: frToU32(coeff))
        }

        // Pack evaluation points
        var pointsU32 = [UInt32]()
        pointsU32.reserveCapacity(numPoints * 8)
        for p in points {
            pointsU32.append(contentsOf: frToU32(p))
        }

        let indicesBuf = device.makeBuffer(bytes: indices, length: nnz * 4,
                                            options: .storageModeShared)!
        let coeffsBuf = device.makeBuffer(bytes: coeffsU32, length: nnz * elemSize,
                                           options: .storageModeShared)!
        let pointsBuf = device.makeBuffer(bytes: pointsU32, length: numPoints * elemSize,
                                           options: .storageModeShared)!
        let resultBuf = device.makeBuffer(length: numPoints * elemSize,
                                           options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Use cached pipeline when terms fit in threadgroup memory
        let pipeline = nnz <= 256 ? evalCachedPipeline : evalPipeline

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(coeffsBuf, offset: 0, index: 1)
        enc.setBuffer(pointsBuf, offset: 0, index: 2)
        enc.setBuffer(resultBuf, offset: 0, index: 3)
        var nnzU32 = UInt32(nnz)
        var nPtsU32 = UInt32(numPoints)
        enc.setBytes(&nnzU32, length: 4, index: 4)
        enc.setBytes(&nPtsU32, length: 4, index: 5)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return readFrResults(from: resultBuf, count: numPoints)
    }

    // MARK: - GPU dispatch: mulDense

    private func gpuMulDense(sparse: SparsePoly, dense: [Fr], outputLen: Int) throws -> [Fr] {
        let nnz = sparse.nnz
        let denseLen = dense.count
        let elemSize = 32

        // Pack sparse indices
        var indices = [UInt32]()
        indices.reserveCapacity(nnz)
        for (idx, _) in sparse.terms {
            indices.append(UInt32(idx))
        }

        // Pack sparse coefficients
        var sCoeffsU32 = [UInt32]()
        sCoeffsU32.reserveCapacity(nnz * 8)
        for (_, coeff) in sparse.terms {
            sCoeffsU32.append(contentsOf: frToU32(coeff))
        }

        // Pack dense coefficients
        var dCoeffsU32 = [UInt32]()
        dCoeffsU32.reserveCapacity(denseLen * 8)
        for c in dense {
            dCoeffsU32.append(contentsOf: frToU32(c))
        }

        let indicesBuf = device.makeBuffer(bytes: indices, length: nnz * 4,
                                            options: .storageModeShared)!
        let sCoeffsBuf = device.makeBuffer(bytes: sCoeffsU32, length: nnz * elemSize,
                                            options: .storageModeShared)!
        let denseBuf = device.makeBuffer(bytes: dCoeffsU32, length: denseLen * elemSize,
                                          options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: outputLen * elemSize,
                                           options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(mulDensePipeline)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(sCoeffsBuf, offset: 0, index: 1)
        enc.setBuffer(denseBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var nnzU32 = UInt32(nnz)
        var denseLenU32 = UInt32(denseLen)
        var outputLenU32 = UInt32(outputLen)
        enc.setBytes(&nnzU32, length: 4, index: 4)
        enc.setBytes(&denseLenU32, length: 4, index: 5)
        enc.setBytes(&outputLenU32, length: 4, index: 6)
        let tg = min(256, Int(mulDensePipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: outputLen, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return readFrResults(from: outputBuf, count: outputLen)
    }

    // MARK: - Helpers

    private func readFrResults(from buffer: MTLBuffer, count: Int) -> [Fr] {
        let totalU32 = count * 8
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: totalU32)
        let raw = Array(UnsafeBufferPointer(start: ptr, count: totalU32))
        return stride(from: 0, to: raw.count, by: 8).map { base in
            Fr(v: (raw[base], raw[base+1], raw[base+2], raw[base+3],
                   raw[base+4], raw[base+5], raw[base+6], raw[base+7]))
        }
    }

    private func frToU32(_ f: Fr) -> [UInt32] {
        [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
    }
}
