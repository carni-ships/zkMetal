// GPUMatrixTransposeEngine — GPU-accelerated matrix transpose for field elements
//
// Tiled transpose using Metal threadgroup shared memory for coalesced reads/writes.
// Supports BN254 Fr (32 bytes), BabyBear (4 bytes), and Goldilocks (8 bytes).
//
// Use cases:
//   - Four-step NTT (row NTT, transpose, column NTT, transpose)
//   - Trace polynomial conversion (column-major <-> row-major)
//   - Multi-polynomial batch operations (interleaved <-> separate)
//
// Public API:
//   transpose(matrix:rows:cols:field:)          — out-of-place rectangular transpose
//   transposeInPlace(matrix:n:field:)           — in-place square matrix transpose
//   batchTranspose(matrices:rows:cols:field:)   — batch out-of-place transpose
//   transposeFr(a:rows:cols:) -> [Fr]           — typed array API for BN254 Fr

import Foundation
import Metal

// MARK: - GPUMatrixTransposeEngine

public class GPUMatrixTransposeEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states per field type
    private let transposeFrPipeline: MTLComputePipelineState
    private let transposeFrInplacePipeline: MTLComputePipelineState
    private let transposeU32Pipeline: MTLComputePipelineState
    private let transposeU64Pipeline: MTLComputePipelineState

    private let pool: GPUBufferPool

    /// CPU fallback threshold: matrices with fewer than this many elements skip GPU.
    public var cpuThreshold: Int = 256

    /// Tile dimension — must match TILE_DIM in the Metal shader.
    private let tileDim = 16

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUMatrixTransposeEngine.compileShaders(device: device)

        guard let fnFr = library.makeFunction(name: "matrix_transpose_fr"),
              let fnFrIP = library.makeFunction(name: "matrix_transpose_fr_inplace"),
              let fnU32 = library.makeFunction(name: "matrix_transpose_u32"),
              let fnU64 = library.makeFunction(name: "matrix_transpose_u64") else {
            throw MSMError.missingKernel
        }

        self.transposeFrPipeline = try device.makeComputePipelineState(function: fnFr)
        self.transposeFrInplacePipeline = try device.makeComputePipelineState(function: fnFrIP)
        self.transposeU32Pipeline = try device.makeComputePipelineState(function: fnU32)
        self.transposeU64Pipeline = try device.makeComputePipelineState(function: fnU64)
        self.pool = GPUBufferPool(device: device)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/utility/matrix_transpose.metal", encoding: .utf8)

        let cleanSource = source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: cleanSource, options: options)
    }

    // MARK: - Public API: Out-of-place transpose (MTLBuffer)

    /// Transpose a rows x cols matrix stored in an MTLBuffer.
    /// Returns a new MTLBuffer containing the cols x rows transposed result.
    /// Field type determines element size: .bn254 = 32B, .babybear = 4B, .goldilocks = 8B.
    public func transpose(matrix: MTLBuffer, rows: Int, cols: Int, field: FieldType = .bn254) throws -> MTLBuffer {
        let totalElements = rows * cols
        let elementSize = fieldElementSize(field)

        if totalElements < cpuThreshold {
            return try cpuTranspose(matrix: matrix, rows: rows, cols: cols, elementSize: elementSize)
        }

        let pipeline: MTLComputePipelineState
        switch field {
        case .bn254:     pipeline = transposeFrPipeline
        case .babybear:  pipeline = transposeU32Pipeline
        case .goldilocks: pipeline = transposeU64Pipeline
        }

        return try dispatchTranspose(src: matrix, rows: rows, cols: cols,
                                     elementSize: elementSize, pipeline: pipeline)
    }

    // MARK: - Public API: In-place square transpose (MTLBuffer)

    /// In-place transpose for a square n x n matrix. Only BN254 Fr is supported for in-place.
    /// For non-square matrices, use the out-of-place `transpose` method.
    public func transposeInPlace(matrix: MTLBuffer, n: Int, field: FieldType = .bn254) throws {
        guard field == .bn254 else {
            // For non-Fr fields, fall back to out-of-place and copy back
            let transposed = try transpose(matrix: matrix, rows: n, cols: n, field: field)
            let size = n * n * fieldElementSize(field)
            memcpy(matrix.contents(), transposed.contents(), size)
            pool.release(buffer: transposed)
            return
        }

        let totalElements = n * n
        if totalElements < cpuThreshold {
            cpuTransposeInPlace(matrix: matrix, n: n, elementSize: fieldElementSize(field))
            return
        }

        try dispatchTransposeInPlace(data: matrix, n: n)
    }

    // MARK: - Public API: Batch transpose

    /// Transpose multiple matrices of the same dimensions in one call.
    /// Returns an array of transposed MTLBuffers.
    public func batchTranspose(matrices: [MTLBuffer], rows: Int, cols: Int,
                               field: FieldType = .bn254) throws -> [MTLBuffer] {
        var results = [MTLBuffer]()
        results.reserveCapacity(matrices.count)

        for matrix in matrices {
            let transposed = try transpose(matrix: matrix, rows: rows, cols: cols, field: field)
            results.append(transposed)
        }
        return results
    }

    // MARK: - Public API: Typed array API for BN254 Fr

    /// Transpose a rows x cols matrix of Fr elements.
    /// Input: flat array of rows*cols Fr elements in row-major order.
    /// Output: flat array of cols*rows Fr elements in row-major order (transposed).
    public func transposeFr(_ a: [Fr], rows: Int, cols: Int) throws -> [Fr] {
        let n = a.count
        guard n == rows * cols else { throw MSMError.invalidInput }
        guard n > 0 else { return [] }

        if n < cpuThreshold {
            return cpuTransposeFr(a, rows: rows, cols: cols)
        }

        let byteSize = n * MemoryLayout<Fr>.stride
        guard let srcBuf = pool.allocate(size: byteSize) else {
            return cpuTransposeFr(a, rows: rows, cols: cols)
        }
        a.withUnsafeBytes { src in memcpy(srcBuf.contents(), src.baseAddress!, byteSize) }

        let dstBuf = try dispatchTranspose(src: srcBuf, rows: rows, cols: cols,
                                           elementSize: MemoryLayout<Fr>.stride,
                                           pipeline: transposeFrPipeline)

        let ptr = dstBuf.contents().bindMemory(to: Fr.self, capacity: n)
        let result = Array(UnsafeBufferPointer(start: ptr, count: n))
        pool.release(buffer: srcBuf)
        pool.release(buffer: dstBuf)
        return result
    }

    /// In-place transpose for a square n x n matrix of Fr elements.
    public func transposeFrInPlace(_ a: inout [Fr], n: Int) throws {
        guard a.count == n * n else { throw MSMError.invalidInput }

        if a.count < cpuThreshold {
            cpuTransposeFrInPlace(&a, n: n)
            return
        }

        let byteSize = a.count * MemoryLayout<Fr>.stride
        guard let buf = pool.allocate(size: byteSize) else {
            cpuTransposeFrInPlace(&a, n: n)
            return
        }
        a.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, byteSize) }

        try dispatchTransposeInPlace(data: buf, n: n)

        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: a.count)
        a.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, ptr, byteSize) }
        pool.release(buffer: buf)
    }

    /// Release a buffer back to the pool (for callers using MTLBuffer APIs).
    public func releaseBuffer(_ buf: MTLBuffer) {
        pool.release(buffer: buf)
    }

    // MARK: - GPU Dispatch: Out-of-place

    private func dispatchTranspose(src: MTLBuffer, rows: Int, cols: Int,
                                   elementSize: Int,
                                   pipeline: MTLComputePipelineState) throws -> MTLBuffer {
        let totalBytes = rows * cols * elementSize
        guard let dstBuf = pool.allocate(size: totalBytes) else {
            throw MSMError.gpuError("Failed to allocate output buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            pool.release(buffer: dstBuf)
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dstBuf, offset: 0, index: 1)
        var dims = (UInt32(rows), UInt32(cols))
        enc.setBytes(&dims, length: 8, index: 2)

        // Dispatch enough threadgroups to cover the matrix in tiles
        let tgX = (cols + tileDim - 1) / tileDim
        let tgY = (rows + tileDim - 1) / tileDim
        enc.dispatchThreadgroups(MTLSize(width: tgX, height: tgY, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tileDim, height: tileDim, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            pool.release(buffer: dstBuf)
            throw MSMError.gpuError(error.localizedDescription)
        }
        return dstBuf
    }

    // MARK: - GPU Dispatch: In-place (square, Fr only)

    private func dispatchTransposeInPlace(data: MTLBuffer, n: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(transposeFrInplacePipeline)
        enc.setBuffer(data, offset: 0, index: 0)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 1)

        // Dispatch threadgroups for the upper triangle (including diagonal).
        // The kernel itself skips blocks where blockRow > blockCol.
        let tgPerDim = (n + tileDim - 1) / tileDim
        enc.dispatchThreadgroups(MTLSize(width: tgPerDim, height: tgPerDim, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tileDim, height: tileDim, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - CPU Fallbacks

    private func cpuTranspose(matrix: MTLBuffer, rows: Int, cols: Int,
                              elementSize: Int) throws -> MTLBuffer {
        let totalBytes = rows * cols * elementSize
        guard let dstBuf = pool.allocate(size: totalBytes) else {
            throw MSMError.gpuError("Failed to allocate output buffer")
        }

        let src = matrix.contents()
        let dst = dstBuf.contents()

        for i in 0..<rows {
            for j in 0..<cols {
                let srcOffset = (i * cols + j) * elementSize
                let dstOffset = (j * rows + i) * elementSize
                memcpy(dst + dstOffset, src + srcOffset, elementSize)
            }
        }
        return dstBuf
    }

    private func cpuTransposeInPlace(matrix: MTLBuffer, n: Int, elementSize: Int) {
        let data = matrix.contents()
        var tmp = [UInt8](repeating: 0, count: elementSize)

        for i in 0..<n {
            for j in (i + 1)..<n {
                let offsetA = (i * n + j) * elementSize
                let offsetB = (j * n + i) * elementSize
                // Swap A[i][j] and A[j][i]
                memcpy(&tmp, data + offsetA, elementSize)
                memcpy(data + offsetA, data + offsetB, elementSize)
                memcpy(data + offsetB, &tmp, elementSize)
            }
        }
    }

    private func cpuTransposeFr(_ a: [Fr], rows: Int, cols: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: rows * cols)
        for i in 0..<rows {
            for j in 0..<cols {
                result[j * rows + i] = a[i * cols + j]
            }
        }
        return result
    }

    private func cpuTransposeFrInPlace(_ a: inout [Fr], n: Int) {
        for i in 0..<n {
            for j in (i + 1)..<n {
                let tmp = a[i * n + j]
                a[i * n + j] = a[j * n + i]
                a[j * n + i] = tmp
            }
        }
    }

    // MARK: - Helpers

    private func fieldElementSize(_ field: FieldType) -> Int {
        switch field {
        case .bn254:     return MemoryLayout<Fr>.stride   // 32 bytes
        case .babybear:  return MemoryLayout<Bb>.stride   // 4 bytes
        case .goldilocks: return MemoryLayout<Gl>.stride  // 8 bytes
        }
    }
}
