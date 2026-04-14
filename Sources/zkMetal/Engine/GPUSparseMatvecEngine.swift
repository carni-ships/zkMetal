// GPU-accelerated CSR Sparse Matrix-Vector Multiply for BN254 Fr
//
// Implements GPU-accelerated CSR sparse matvec for Nova, HyperNova, and Supernova
// folding schemes. Supports both single matvec and fused triple matvec when
// matrices share the same sparsity pattern.
//
// Usage:
//   let engine = try GPUSparseMatvecEngine()
//   let result = engine.matvec(rowPtr: ..., colIdx: ..., values: ..., z: ..., m: ...)
//   let (az, bz, cz) = engine.matvecTriple(rowPtr: ..., colIdx: ...,
//                                           valuesA: ..., valuesB: ..., valuesC: ...,
//                                           z: ..., m: ...)
//
// Performance notes:
//   - Best for matrices with hundreds to thousands of rows
//   - Fused triple matvec is ~3x faster than three separate matvecs
//   - GPU overhead means CPU may be faster for very small matrices (m < 64)

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Sparse Matvec Engine

public class GPUSparseMatvecEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Kernel pipelines
    private let matvecPipeline: MTLComputePipelineState
    private let matvecTriplePipeline: MTLComputePipelineState
    private let matvecBatchPipeline: MTLComputePipelineState
    private let matvecTripleBatchPipeline: MTLComputePipelineState

    private let threadgroupSize: Int
    private let pool: GPUBufferPool

    /// Threshold: matrices with fewer rows than this use CPU path
    public var cpuThreshold: Int = 64

    /// Threshold: matrices with fewer total non-zeros than this use CPU path
    public var nnzThreshold: Int = 256

    private let library: MTLLibrary

    public init(threadgroupSize: Int = 64) throws {
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

        self.library = try GPUSparseMatvecEngine.compileShaders(device: device)

        guard let matvecFn = library.makeFunction(name: "sparse_matvec_bn254"),
              let matvecTripleFn = library.makeFunction(name: "sparse_matvec_triple_bn254"),
              let matvecBatchFn = library.makeFunction(name: "sparse_matvec_batch_bn254"),
              let matvecTripleBatchFn = library.makeFunction(name: "sparse_matvec_triple_batch_bn254") else {
            throw MSMError.missingKernel
        }

        self.matvecPipeline = try device.makeComputePipelineState(function: matvecFn)
        self.matvecTriplePipeline = try device.makeComputePipelineState(function: matvecTripleFn)
        self.matvecBatchPipeline = try device.makeComputePipelineState(function: matvecBatchFn)
        self.matvecTripleBatchPipeline = try device.makeComputePipelineState(function: matvecTripleBatchFn)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let matvecSource = try String(contentsOfFile: shaderDir + "/fold/sparse_matvec.metal", encoding: .utf8)

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")
        let cleanMatvec = matvecSource
            .split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanMatvec
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Public API

    /// Compute M * z for a CSR sparse matrix.
    ///
    /// - Parameters:
    ///   - rowPtr: row pointers (length m+1)
    ///   - colIdx: column indices (length nnz)
    ///   - values: non-zero values (length nnz)
    ///   - z: input vector (length n)
    ///   - m: number of rows
    /// - Returns: result vector M*z (length m)
    public func matvec(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        z: [Fr],
        m: Int
    ) -> [Fr] {
        // CPU fallback for small matrices
        if m < cpuThreshold || values.count < nnzThreshold {
            return cpuMatvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
        }

        return gpuMatvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
    }

    /// Compute A*z, B*z, C*z for three CSR sparse matrices with shared sparsity.
    ///
    /// - Parameters:
    ///   - rowPtr: row pointers (length m+1, shared by A, B, C)
    ///   - colIdx: column indices (length nnz, shared by A, B, C)
    ///   - valuesA, valuesB, valuesC: non-zero values for each matrix
    ///   - z: input vector (length n)
    ///   - m: number of rows
    /// - Returns: (A*z, B*z, C*z) each of length m
    public func matvecTriple(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        z: [Fr],
        m: Int
    ) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
        // CPU fallback for small matrices
        if m < cpuThreshold || valuesA.count < nnzThreshold {
            return cpuMatvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                                   valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                   z: z, m: m)
        }

        return gpuMatvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                               valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                               z: z, m: m)
    }

    /// Compute M * z for K different z vectors (same sparsity pattern).
    ///
    /// - Parameters:
    ///   - rowPtr: row pointers (length m+1)
    ///   - colIdx: column indices (length nnz)
    ///   - values: non-zero values (length nnz)
    ///   - zVectors: K concatenated z vectors (each length n)
    ///   - m: number of rows
    ///   - n: vector dimension
    ///   - k: number of vectors
    /// - Returns: K result vectors (each length m)
    public func matvecBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [[Fr]] {
        if m * k < cpuThreshold || values.count < nnzThreshold {
            return cpuMatvecBatch(rowPtr: rowPtr, colIdx: colIdx, values: values,
                                  zVectors: zVectors, m: m, n: n, k: k)
        }

        return gpuMatvecBatch(rowPtr: rowPtr, colIdx: colIdx, values: values,
                              zVectors: zVectors, m: m, n: n, k: k)
    }

    /// Compute A*z, B*z, C*z for K different z vectors (shared sparsity).
    ///
    /// - Parameters:
    ///   - rowPtr: row pointers (length m+1, shared)
    ///   - colIdx: column indices (length nnz, shared)
    ///   - valuesA, valuesB, valuesC: non-zero values for each matrix
    ///   - zVectors: K concatenated z vectors (each length n)
    ///   - m: number of rows
    ///   - n: vector dimension
    ///   - k: number of vectors
    /// - Returns: ((A*z1, B*z1, C*z1), ..., (A*zK, B*zK, C*zK))
    public func matvecTripleBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [(az: [Fr], bz: [Fr], cz: [Fr])] {
        if m * k < cpuThreshold || valuesA.count < nnzThreshold {
            return cpuMatvecTripleBatch(rowPtr: rowPtr, colIdx: colIdx,
                                        valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                        zVectors: zVectors, m: m, n: n, k: k)
        }

        return gpuMatvecTripleBatch(rowPtr: rowPtr, colIdx: colIdx,
                                    valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                    zVectors: zVectors, m: m, n: n, k: k)
    }

    // MARK: - GPU dispatch (single matvec)

    private func gpuMatvec(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        z: [Fr],
        m: Int
    ) -> [Fr] {
        let frStride = MemoryLayout<Fr>.stride
        let nnz = values.count

        // Allocate GPU buffers
        guard let rowPtrBuf = pool.allocate(size: (m + 1) * 4),
              let colIdxBuf = pool.allocate(size: nnz * 4),
              let valuesBuf = pool.allocate(size: nnz * frStride),
              let zBuf = pool.allocate(size: z.count * frStride),
              let resultBuf = pool.allocate(size: m * frStride) else {
            return cpuMatvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
        }

        // Upload data
        rowPtr.withUnsafeBytes { src in memcpy(rowPtrBuf.contents(), src.baseAddress!, (m + 1) * 4) }
        colIdx.withUnsafeBytes { src in memcpy(colIdxBuf.contents(), src.baseAddress!, nnz * 4) }
        values.withUnsafeBytes { src in memcpy(valuesBuf.contents(), src.baseAddress!, nnz * frStride) }
        z.withUnsafeBytes { src in memcpy(zBuf.contents(), src.baseAddress!, z.count * frStride) }

        // Dispatch
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuMatvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
        }

        encoder.setComputePipelineState(matvecPipeline)
        encoder.setBuffer(rowPtrBuf, offset: 0, index: 0)
        encoder.setBuffer(colIdxBuf, offset: 0, index: 1)
        encoder.setBuffer(valuesBuf, offset: 0, index: 2)
        encoder.setBuffer(zBuf, offset: 0, index: 3)
        encoder.setBuffer(resultBuf, offset: 0, index: 4)
        var m32 = UInt32(m)
        encoder.setBytes(&m32, length: 4, index: 5)

        encoder.dispatchThreadgroups(
            MTLSize(width: m, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Download result
        var result = [Fr](repeating: .zero, count: m)
        if cmdBuf.error == nil {
            result.withUnsafeMutableBytes { dst in
                memcpy(dst.baseAddress!, resultBuf.contents(), m * frStride)
            }
        } else {
            return cpuMatvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
        }

        pool.release(buffer: rowPtrBuf)
        pool.release(buffer: colIdxBuf)
        pool.release(buffer: valuesBuf)
        pool.release(buffer: zBuf)
        pool.release(buffer: resultBuf)

        return result
    }

    // MARK: - GPU dispatch (fused triple matvec)

    private func gpuMatvecTriple(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        z: [Fr],
        m: Int
    ) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
        let frStride = MemoryLayout<Fr>.stride
        let nnz = valuesA.count

        guard let rowPtrBuf = pool.allocate(size: (m + 1) * 4),
              let colIdxBuf = pool.allocate(size: nnz * 4),
              let valuesABuf = pool.allocate(size: nnz * frStride),
              let valuesBBuf = pool.allocate(size: nnz * frStride),
              let valuesCBuf = pool.allocate(size: nnz * frStride),
              let zBuf = pool.allocate(size: z.count * frStride),
              let resultABuf = pool.allocate(size: m * frStride),
              let resultBBuf = pool.allocate(size: m * frStride),
              let resultCBuf = pool.allocate(size: m * frStride) else {
            return cpuMatvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                                   valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                   z: z, m: m)
        }

        // Upload data
        rowPtr.withUnsafeBytes { src in memcpy(rowPtrBuf.contents(), src.baseAddress!, (m + 1) * 4) }
        colIdx.withUnsafeBytes { src in memcpy(colIdxBuf.contents(), src.baseAddress!, nnz * 4) }
        valuesA.withUnsafeBytes { src in memcpy(valuesABuf.contents(), src.baseAddress!, nnz * frStride) }
        valuesB.withUnsafeBytes { src in memcpy(valuesBBuf.contents(), src.baseAddress!, nnz * frStride) }
        valuesC.withUnsafeBytes { src in memcpy(valuesCBuf.contents(), src.baseAddress!, nnz * frStride) }
        z.withUnsafeBytes { src in memcpy(zBuf.contents(), src.baseAddress!, z.count * frStride) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuMatvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                                   valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                   z: z, m: m)
        }

        encoder.setComputePipelineState(matvecTriplePipeline)
        encoder.setBuffer(rowPtrBuf, offset: 0, index: 0)
        encoder.setBuffer(colIdxBuf, offset: 0, index: 1)
        encoder.setBuffer(valuesABuf, offset: 0, index: 2)
        encoder.setBuffer(valuesBBuf, offset: 0, index: 3)
        encoder.setBuffer(valuesCBuf, offset: 0, index: 4)
        encoder.setBuffer(zBuf, offset: 0, index: 5)
        encoder.setBuffer(resultABuf, offset: 0, index: 6)
        encoder.setBuffer(resultBBuf, offset: 0, index: 7)
        encoder.setBuffer(resultCBuf, offset: 0, index: 8)
        var m32 = UInt32(m)
        encoder.setBytes(&m32, length: 4, index: 9)

        encoder.dispatchThreadgroups(
            MTLSize(width: m, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var az = [Fr](repeating: .zero, count: m)
        var bz = [Fr](repeating: .zero, count: m)
        var cz = [Fr](repeating: .zero, count: m)

        if cmdBuf.error == nil {
            az.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, resultABuf.contents(), m * frStride) }
            bz.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, resultBBuf.contents(), m * frStride) }
            cz.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, resultCBuf.contents(), m * frStride) }
        } else {
            return cpuMatvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                                   valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                   z: z, m: m)
        }

        pool.release(buffer: rowPtrBuf)
        pool.release(buffer: colIdxBuf)
        pool.release(buffer: valuesABuf)
        pool.release(buffer: valuesBBuf)
        pool.release(buffer: valuesCBuf)
        pool.release(buffer: zBuf)
        pool.release(buffer: resultABuf)
        pool.release(buffer: resultBBuf)
        pool.release(buffer: resultCBuf)

        return (az, bz, cz)
    }

    // MARK: - GPU dispatch (batch matvec)

    private func gpuMatvecBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [[Fr]] {
        let frStride = MemoryLayout<Fr>.stride
        let nnz = values.count
        let totalZ = k * n
        let totalResult = k * m

        guard let rowPtrBuf = pool.allocate(size: (m + 1) * 4),
              let colIdxBuf = pool.allocate(size: nnz * 4),
              let valuesBuf = pool.allocate(size: nnz * frStride),
              let zVecBuf = pool.allocate(size: totalZ * frStride),
              let resultBuf = pool.allocate(size: totalResult * frStride),
              let zOffBuf = pool.allocate(size: k * 4),
              let resOffBuf = pool.allocate(size: k * 4) else {
            return cpuMatvecBatch(rowPtr: rowPtr, colIdx: colIdx, values: values,
                                  zVectors: zVectors, m: m, n: n, k: k)
        }

        // Build offset arrays
        var zOffsets = [UInt32]()
        var resultOffsets = [UInt32]()
        zOffsets.reserveCapacity(k)
        resultOffsets.reserveCapacity(k)
        for i in 0..<k {
            zOffsets.append(UInt32(i * n))
            resultOffsets.append(UInt32(i * m))
        }

        // Upload data
        rowPtr.withUnsafeBytes { src in memcpy(rowPtrBuf.contents(), src.baseAddress!, (m + 1) * 4) }
        colIdx.withUnsafeBytes { src in memcpy(colIdxBuf.contents(), src.baseAddress!, nnz * 4) }
        values.withUnsafeBytes { src in memcpy(valuesBuf.contents(), src.baseAddress!, nnz * frStride) }
        zVectors.withUnsafeBytes { src in memcpy(zVecBuf.contents(), src.baseAddress!, totalZ * frStride) }
        zOffsets.withUnsafeBytes { src in memcpy(zOffBuf.contents(), src.baseAddress!, k * 4) }
        resultOffsets.withUnsafeBytes { src in memcpy(resOffBuf.contents(), src.baseAddress!, k * 4) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuMatvecBatch(rowPtr: rowPtr, colIdx: colIdx, values: values,
                                  zVectors: zVectors, m: m, n: n, k: k)
        }

        encoder.setComputePipelineState(matvecBatchPipeline)
        encoder.setBuffer(rowPtrBuf, offset: 0, index: 0)
        encoder.setBuffer(colIdxBuf, offset: 0, index: 1)
        encoder.setBuffer(valuesBuf, offset: 0, index: 2)
        encoder.setBuffer(zVecBuf, offset: 0, index: 3)
        encoder.setBuffer(resultBuf, offset: 0, index: 4)
        encoder.setBuffer(zOffBuf, offset: 0, index: 5)
        encoder.setBuffer(resOffBuf, offset: 0, index: 6)
        var m32 = UInt32(m)
        var n32 = UInt32(n)
        var k32 = UInt32(k)
        encoder.setBytes(&m32, length: 4, index: 7)
        encoder.setBytes(&n32, length: 4, index: 8)
        encoder.setBytes(&k32, length: 4, index: 9)

        encoder.dispatchThreadgroups(
            MTLSize(width: m * k, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var results = [[Fr]](repeating: [Fr](repeating: .zero, count: m), count: k)

        if cmdBuf.error == nil {
            for i in 0..<k {
                results[i].withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, resultBuf.contents().advanced(by: i * m * frStride), m * frStride)
                }
            }
        } else {
            return cpuMatvecBatch(rowPtr: rowPtr, colIdx: colIdx, values: values,
                                  zVectors: zVectors, m: m, n: n, k: k)
        }

        pool.release(buffer: rowPtrBuf)
        pool.release(buffer: colIdxBuf)
        pool.release(buffer: valuesBuf)
        pool.release(buffer: zVecBuf)
        pool.release(buffer: resultBuf)
        pool.release(buffer: zOffBuf)
        pool.release(buffer: resOffBuf)

        return results
    }

    // MARK: - GPU dispatch (batch triple matvec)

    private func gpuMatvecTripleBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [(az: [Fr], bz: [Fr], cz: [Fr])] {
        let frStride = MemoryLayout<Fr>.stride
        let nnz = valuesA.count
        let totalZ = k * n
        let totalResult = k * m

        guard let rowPtrBuf = pool.allocate(size: (m + 1) * 4),
              let colIdxBuf = pool.allocate(size: nnz * 4),
              let valuesABuf = pool.allocate(size: nnz * frStride),
              let valuesBBuf = pool.allocate(size: nnz * frStride),
              let valuesCBuf = pool.allocate(size: nnz * frStride),
              let zVecBuf = pool.allocate(size: totalZ * frStride),
              let resultABuf = pool.allocate(size: totalResult * frStride),
              let resultBBuf = pool.allocate(size: totalResult * frStride),
              let resultCBuf = pool.allocate(size: totalResult * frStride),
              let zOffBuf = pool.allocate(size: k * 4),
              let resOffBuf = pool.allocate(size: k * 4) else {
            return cpuMatvecTripleBatch(rowPtr: rowPtr, colIdx: colIdx,
                                        valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                        zVectors: zVectors, m: m, n: n, k: k)
        }

        var zOffsets = [UInt32]()
        var resultOffsets = [UInt32]()
        zOffsets.reserveCapacity(k)
        resultOffsets.reserveCapacity(k)
        for i in 0..<k {
            zOffsets.append(UInt32(i * n))
            resultOffsets.append(UInt32(i * m))
        }

        // Upload data
        rowPtr.withUnsafeBytes { src in memcpy(rowPtrBuf.contents(), src.baseAddress!, (m + 1) * 4) }
        colIdx.withUnsafeBytes { src in memcpy(colIdxBuf.contents(), src.baseAddress!, nnz * 4) }
        valuesA.withUnsafeBytes { src in memcpy(valuesABuf.contents(), src.baseAddress!, nnz * frStride) }
        valuesB.withUnsafeBytes { src in memcpy(valuesBBuf.contents(), src.baseAddress!, nnz * frStride) }
        valuesC.withUnsafeBytes { src in memcpy(valuesCBuf.contents(), src.baseAddress!, nnz * frStride) }
        zVectors.withUnsafeBytes { src in memcpy(zVecBuf.contents(), src.baseAddress!, totalZ * frStride) }
        zOffsets.withUnsafeBytes { src in memcpy(zOffBuf.contents(), src.baseAddress!, k * 4) }
        resultOffsets.withUnsafeBytes { src in memcpy(resOffBuf.contents(), src.baseAddress!, k * 4) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuMatvecTripleBatch(rowPtr: rowPtr, colIdx: colIdx,
                                        valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                        zVectors: zVectors, m: m, n: n, k: k)
        }

        encoder.setComputePipelineState(matvecTripleBatchPipeline)
        encoder.setBuffer(rowPtrBuf, offset: 0, index: 0)
        encoder.setBuffer(colIdxBuf, offset: 0, index: 1)
        encoder.setBuffer(valuesABuf, offset: 0, index: 2)
        encoder.setBuffer(valuesBBuf, offset: 0, index: 3)
        encoder.setBuffer(valuesCBuf, offset: 0, index: 4)
        encoder.setBuffer(zVecBuf, offset: 0, index: 5)
        encoder.setBuffer(resultABuf, offset: 0, index: 6)
        encoder.setBuffer(resultBBuf, offset: 0, index: 7)
        encoder.setBuffer(resultCBuf, offset: 0, index: 8)
        encoder.setBuffer(zOffBuf, offset: 0, index: 9)
        encoder.setBuffer(resOffBuf, offset: 0, index: 10)
        var m32 = UInt32(m)
        var n32 = UInt32(n)
        var k32 = UInt32(k)
        encoder.setBytes(&m32, length: 4, index: 11)
        encoder.setBytes(&n32, length: 4, index: 12)
        encoder.setBytes(&k32, length: 4, index: 13)

        encoder.dispatchThreadgroups(
            MTLSize(width: m * k, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var results = [(az: [Fr], bz: [Fr], cz: [Fr])]()
        results.reserveCapacity(k)

        if cmdBuf.error == nil {
            for i in 0..<k {
                var az = [Fr](repeating: .zero, count: m)
                var bz = [Fr](repeating: .zero, count: m)
                var cz = [Fr](repeating: .zero, count: m)

                az.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, resultABuf.contents().advanced(by: i * m * frStride), m * frStride)
                }
                bz.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, resultBBuf.contents().advanced(by: i * m * frStride), m * frStride)
                }
                cz.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, resultCBuf.contents().advanced(by: i * m * frStride), m * frStride)
                }
                results.append((az, bz, cz))
            }
        } else {
            return cpuMatvecTripleBatch(rowPtr: rowPtr, colIdx: colIdx,
                                        valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                        zVectors: zVectors, m: m, n: n, k: k)
        }

        pool.release(buffer: rowPtrBuf)
        pool.release(buffer: colIdxBuf)
        pool.release(buffer: valuesABuf)
        pool.release(buffer: valuesBBuf)
        pool.release(buffer: valuesCBuf)
        pool.release(buffer: zVecBuf)
        pool.release(buffer: resultABuf)
        pool.release(buffer: resultBBuf)
        pool.release(buffer: resultCBuf)
        pool.release(buffer: zOffBuf)
        pool.release(buffer: resOffBuf)

        return results
    }

    // MARK: - CPU fallback implementations

    private func cpuMatvec(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        z: [Fr],
        m: Int
    ) -> [Fr] {
        var result = [Fr](repeating: .zero, count: m)
        for i in 0..<m {
            var acc: Fr = .zero
            for k in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
                let col = Int(colIdx[k])
                acc = frAdd(acc, frMul(values[k], z[col]))
            }
            result[i] = acc
        }
        return result
    }

    private func cpuMatvecTriple(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        z: [Fr],
        m: Int
    ) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
        var az = [Fr](repeating: .zero, count: m)
        var bz = [Fr](repeating: .zero, count: m)
        var cz = [Fr](repeating: .zero, count: m)

        for i in 0..<m {
            var accA: Fr = .zero
            var accB: Fr = .zero
            var accC: Fr = .zero
            for k in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
                let col = Int(colIdx[k])
                let zval = z[col]
                accA = frAdd(accA, frMul(valuesA[k], zval))
                accB = frAdd(accB, frMul(valuesB[k], zval))
                accC = frAdd(accC, frMul(valuesC[k], zval))
            }
            az[i] = accA
            bz[i] = accB
            cz[i] = accC
        }
        return (az, bz, cz)
    }

    private func cpuMatvecBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        values: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [[Fr]] {
        var results = [[Fr]](repeating: [Fr](repeating: .zero, count: m), count: k)
        for matIdx in 0..<k {
            for i in 0..<m {
                var acc: Fr = .zero
                for nzIdx in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
                    let col = Int(colIdx[nzIdx])
                    let zIdx = matIdx * n + col
                    acc = frAdd(acc, frMul(values[nzIdx], zVectors[zIdx]))
                }
                results[matIdx][i] = acc
            }
        }
        return results
    }

    private func cpuMatvecTripleBatch(
        rowPtr: [UInt32],
        colIdx: [UInt32],
        valuesA: [Fr],
        valuesB: [Fr],
        valuesC: [Fr],
        zVectors: [Fr],
        m: Int,
        n: Int,
        k: Int
    ) -> [(az: [Fr], bz: [Fr], cz: [Fr])] {
        var results = [(az: [Fr], bz: [Fr], cz: [Fr])]()
        results.reserveCapacity(k)

        for matIdx in 0..<k {
            var az = [Fr](repeating: .zero, count: m)
            var bz = [Fr](repeating: .zero, count: m)
            var cz = [Fr](repeating: .zero, count: m)

            for i in 0..<m {
                var accA: Fr = .zero
                var accB: Fr = .zero
                var accC: Fr = .zero
                for nzIdx in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
                    let col = Int(colIdx[nzIdx])
                    let zIdx = matIdx * n + col
                    let zval = zVectors[zIdx]
                    accA = frAdd(accA, frMul(valuesA[nzIdx], zval))
                    accB = frAdd(accB, frMul(valuesB[nzIdx], zval))
                    accC = frAdd(accC, frMul(valuesC[nzIdx], zval))
                }
                az[i] = accA
                bz[i] = accB
                cz[i] = accC
            }
            results.append((az, bz, cz))
        }
        return results
    }
}

// MARK: - SparseMatrix GPU Extension

extension SparseMatrix {
    /// Matrix-vector multiply using GPU when available and beneficial.
    ///
    /// Falls back to CPU for small matrices or when GPU is unavailable.
    public func mulVecGPU(_ z: [Fr], engine: GPUSparseMatvecEngine? = nil) -> [Fr] {
        let m = rows
        let nnz = values.count

        // For very small matrices, use CPU
        if m < 64 || nnz < 256 {
            return mulVec(z)
        }

        // Try GPU if engine provided
        if let engine = engine {
            return engine.matvec(
                rowPtr: rowPtr.map { UInt32($0) },
                colIdx: colIdx.map { UInt32($0) },
                values: values,
                z: z,
                m: m
            )
        }

        // Fall back to CPU
        return mulVec(z)
    }

    /// Fused triple matrix-vector multiply using GPU when available.
    ///
    /// All three matrices must share the same sparsity pattern (rowPtr, colIdx).
    public func mulVecTripleGPU(
        _ z: [Fr],
        _ M_B: SparseMatrix,
        _ M_C: SparseMatrix,
        engine: GPUSparseMatvecEngine? = nil
    ) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
        precondition(rows == M_B.rows && rows == M_C.rows)
        precondition(cols == M_B.cols && cols == M_C.cols)

        let m = rows
        let nnz = values.count

        // For very small matrices, use CPU
        if m < 64 || nnz < 256 {
            return mulVecTriple(z, M_B, M_C)
        }

        // Verify sparsity pattern matches
        guard rowPtr.elementsEqual(M_B.rowPtr) && rowPtr.elementsEqual(M_C.rowPtr),
              colIdx.elementsEqual(M_B.colIdx) && colIdx.elementsEqual(M_C.colIdx) else {
            // Can't use fused path, fall back to separate GPU calls
            if let engine = engine {
                let az = engine.matvec(
                    rowPtr: rowPtr.map { UInt32($0) },
                    colIdx: colIdx.map { UInt32($0) },
                    values: values,
                    z: z,
                    m: m
                )
                let bz = engine.matvec(
                    rowPtr: rowPtr.map { UInt32($0) },
                    colIdx: colIdx.map { UInt32($0) },
                    values: M_B.values,
                    z: z,
                    m: m
                )
                let cz = engine.matvec(
                    rowPtr: rowPtr.map { UInt32($0) },
                    colIdx: colIdx.map { UInt32($0) },
                    values: M_C.values,
                    z: z,
                    m: m
                )
                return (az, bz, cz)
            }
            return mulVecTriple(z, M_B, M_C)
        }

        // Use GPU fused triple
        if let engine = engine {
            return engine.matvecTriple(
                rowPtr: rowPtr.map { UInt32($0) },
                colIdx: colIdx.map { UInt32($0) },
                valuesA: values,
                valuesB: M_B.values,
                valuesC: M_C.values,
                z: z,
                m: m
            )
        }

        return mulVecTriple(z, M_B, M_C)
    }
}
