// Lattice NTT Engine — GPU-accelerated NTT for Kyber (16-bit) and Dilithium (32-bit)
// Each polynomial is only 256 elements, so the entire NTT fits in shared memory.
// The GPU advantage comes from batching: process thousands of polynomials in parallel.
//
// Kyber: 256 * 2 bytes = 512 bytes per polynomial in threadgroup memory
// Dilithium: 256 * 4 bytes = 1KB per polynomial in threadgroup memory
// Both trivially fit in Metal's 32KB threadgroup memory limit.

import Foundation
import Metal

public class LatticeNTTEngine {
    public static let version = Versions.latticeNTT
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Kyber NTT kernels
    let kyberNTTFunction: MTLComputePipelineState
    let kyberINTTFunction: MTLComputePipelineState

    // Dilithium NTT kernels
    let dilithiumNTTFunction: MTLComputePipelineState
    let dilithiumINTTFunction: MTLComputePipelineState

    // Polynomial operation kernels
    let kyberPolyAddFunction: MTLComputePipelineState
    let kyberPolySubFunction: MTLComputePipelineState
    let kyberPointwiseMulFunction: MTLComputePipelineState
    let kyberMatvecFunction: MTLComputePipelineState
    let dilithiumPolyAddFunction: MTLComputePipelineState
    let dilithiumPolySubFunction: MTLComputePipelineState
    let dilithiumPointwiseMulFunction: MTLComputePipelineState
    let dilithiumMatvecFunction: MTLComputePipelineState
    let kyberCompressFunction: MTLComputePipelineState
    let kyberDecompressFunction: MTLComputePipelineState

    // Cached twiddle buffers
    private var kyberTwiddleBuf: MTLBuffer?
    private var kyberInvTwiddleBuf: MTLBuffer?
    private var kyberInvNBuf: MTLBuffer?
    private var dilithiumTwiddleBuf: MTLBuffer?
    private var dilithiumInvTwiddleBuf: MTLBuffer?
    private var dilithiumInvNBuf: MTLBuffer?

    // Threadgroup size for NTT (32 threads per polynomial)
    let nttThreadgroupSize = 32

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try LatticeNTTEngine.compileShaders(device: device)

        guard let kyberNTTFn = library.makeFunction(name: "kyber_ntt_batch"),
              let kyberINTTFn = library.makeFunction(name: "kyber_intt_batch"),
              let dilNTTFn = library.makeFunction(name: "dilithium_ntt_batch"),
              let dilINTTFn = library.makeFunction(name: "dilithium_intt_batch"),
              let kPolyAddFn = library.makeFunction(name: "kyber_poly_add"),
              let kPolySubFn = library.makeFunction(name: "kyber_poly_sub"),
              let kPwMulFn = library.makeFunction(name: "kyber_poly_pointwise_mul"),
              let kMatvecFn = library.makeFunction(name: "kyber_matvec_ntt"),
              let dPolyAddFn = library.makeFunction(name: "dilithium_poly_add"),
              let dPolySubFn = library.makeFunction(name: "dilithium_poly_sub"),
              let dPwMulFn = library.makeFunction(name: "dilithium_poly_pointwise_mul"),
              let dMatvecFn = library.makeFunction(name: "dilithium_matvec_ntt"),
              let kCompressFn = library.makeFunction(name: "kyber_compress"),
              let kDecompressFn = library.makeFunction(name: "kyber_decompress") else {
            throw MSMError.missingKernel
        }

        self.kyberNTTFunction = try device.makeComputePipelineState(function: kyberNTTFn)
        self.kyberINTTFunction = try device.makeComputePipelineState(function: kyberINTTFn)
        self.dilithiumNTTFunction = try device.makeComputePipelineState(function: dilNTTFn)
        self.dilithiumINTTFunction = try device.makeComputePipelineState(function: dilINTTFn)
        self.kyberPolyAddFunction = try device.makeComputePipelineState(function: kPolyAddFn)
        self.kyberPolySubFunction = try device.makeComputePipelineState(function: kPolySubFn)
        self.kyberPointwiseMulFunction = try device.makeComputePipelineState(function: kPwMulFn)
        self.kyberMatvecFunction = try device.makeComputePipelineState(function: kMatvecFn)
        self.dilithiumPolyAddFunction = try device.makeComputePipelineState(function: dPolyAddFn)
        self.dilithiumPolySubFunction = try device.makeComputePipelineState(function: dPolySubFn)
        self.dilithiumPointwiseMulFunction = try device.makeComputePipelineState(function: dPwMulFn)
        self.dilithiumMatvecFunction = try device.makeComputePipelineState(function: dMatvecFn)
        self.kyberCompressFunction = try device.makeComputePipelineState(function: kCompressFn)
        self.kyberDecompressFunction = try device.makeComputePipelineState(function: kDecompressFn)

        // Precompute twiddle factors
        try precomputeTwiddles()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let nttSource = try String(contentsOfFile: shaderDir + "/lattice/lattice_ntt.metal", encoding: .utf8)
        let opsSource = try String(contentsOfFile: shaderDir + "/lattice/lattice_ops.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanOps = opsSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")

        let combined = cleanNTT + "\n" + cleanOps
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/lattice/lattice_ntt.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle precomputation

    private func precomputeTwiddles() throws {
        // Kyber twiddles (128 UInt16 values)
        let kTw = kyberTwiddles()
        let kInvTw = kyberInvTwiddles()
        kyberTwiddleBuf = makeBuffer(kTw.map { $0.value })
        kyberInvTwiddleBuf = makeBuffer(kInvTw.map { $0.value })
        let kyberInvNVal = kyberInverse(KyberField(value: 128)).value
        kyberInvNBuf = makeBuffer([kyberInvNVal])

        // Dilithium twiddles (128 UInt32 values)
        let dTw = dilithiumTwiddles()
        let dInvTw = dilithiumInvTwiddles()
        dilithiumTwiddleBuf = makeBuffer(dTw.map { $0.value })
        dilithiumInvTwiddleBuf = makeBuffer(dInvTw.map { $0.value })
        let dilInvNVal = dilithiumInverse(DilithiumField(value: 128)).value
        dilithiumInvNBuf = makeBuffer([dilInvNVal])
    }

    private func makeBuffer<T>(_ data: [T]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<T>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Kyber GPU NTT

    /// Batch Kyber NTT on GPU. Input: flat array of numPolys * 256 UInt16 values.
    public func batchKyberNTT(_ polys: [UInt16], numPolys: Int) throws -> [UInt16] {
        precondition(polys.count == numPolys * 256)
        let byteCount = polys.count * MemoryLayout<UInt16>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create data buffer")
        }
        polys.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberNTTFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(kyberTwiddleBuf!, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numPolys, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt16.self, capacity: polys.count)
        return Array(UnsafeBufferPointer(start: ptr, count: polys.count))
    }

    /// Batch Kyber inverse NTT on GPU
    public func batchKyberINTT(_ polys: [UInt16], numPolys: Int) throws -> [UInt16] {
        precondition(polys.count == numPolys * 256)
        let byteCount = polys.count * MemoryLayout<UInt16>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create data buffer")
        }
        polys.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberINTTFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(kyberInvTwiddleBuf!, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.setBuffer(kyberInvNBuf!, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: numPolys, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt16.self, capacity: polys.count)
        return Array(UnsafeBufferPointer(start: ptr, count: polys.count))
    }

    // MARK: - Dilithium GPU NTT

    /// Batch Dilithium NTT on GPU
    public func batchDilithiumNTT(_ polys: [UInt32], numPolys: Int) throws -> [UInt32] {
        precondition(polys.count == numPolys * 256)
        let byteCount = polys.count * MemoryLayout<UInt32>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create data buffer")
        }
        polys.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumNTTFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(dilithiumTwiddleBuf!, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: numPolys, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: polys.count)
        return Array(UnsafeBufferPointer(start: ptr, count: polys.count))
    }

    /// Batch Dilithium inverse NTT on GPU
    public func batchDilithiumINTT(_ polys: [UInt32], numPolys: Int) throws -> [UInt32] {
        precondition(polys.count == numPolys * 256)
        let byteCount = polys.count * MemoryLayout<UInt32>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create data buffer")
        }
        polys.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumINTTFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(dilithiumInvTwiddleBuf!, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.setBuffer(dilithiumInvNBuf!, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: numPolys, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: polys.count)
        return Array(UnsafeBufferPointer(start: ptr, count: polys.count))
    }

    // MARK: - GPU Polynomial Operations

    /// GPU batch pointwise multiply (Kyber)
    public func kyberPointwiseMul(_ a: [UInt16], _ b: [UInt16]) throws -> [UInt16] {
        precondition(a.count == b.count)
        let count = a.count
        let byteCount = count * MemoryLayout<UInt16>.stride

        guard let aBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create buffers")
        }
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteCount) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberPointwiseMulFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: 4, index: 3)
        let tgSize = min(256, Int(kyberPointwiseMulFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// GPU batch pointwise multiply (Dilithium)
    public func dilithiumPointwiseMul(_ a: [UInt32], _ b: [UInt32]) throws -> [UInt32] {
        precondition(a.count == b.count)
        let count = a.count
        let byteCount = count * MemoryLayout<UInt32>.stride

        guard let aBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create buffers")
        }
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteCount) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumPointwiseMulFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: 4, index: 3)
        let tgSize = min(256, Int(dilithiumPointwiseMulFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - High-level API with KyberField/DilithiumField arrays

    /// NTT a batch of Kyber polynomials (each 256 KyberField elements)
    public func batchKyberNTT(_ polys: [[KyberField]]) throws -> [[KyberField]] {
        let numPolys = polys.count
        let flat = polys.flatMap { $0.map { $0.value } }
        let result = try batchKyberNTT(flat, numPolys: numPolys)
        return stride(from: 0, to: result.count, by: 256).map { start in
            Array(result[start..<start+256]).map { KyberField(value: $0) }
        }
    }

    /// Inverse NTT a batch of Kyber polynomials
    public func batchKyberINTT(_ polys: [[KyberField]]) throws -> [[KyberField]] {
        let numPolys = polys.count
        let flat = polys.flatMap { $0.map { $0.value } }
        let result = try batchKyberINTT(flat, numPolys: numPolys)
        return stride(from: 0, to: result.count, by: 256).map { start in
            Array(result[start..<start+256]).map { KyberField(value: $0) }
        }
    }

    /// NTT a batch of Dilithium polynomials
    public func batchDilithiumNTT(_ polys: [[DilithiumField]]) throws -> [[DilithiumField]] {
        let numPolys = polys.count
        let flat = polys.flatMap { $0.map { $0.value } }
        let result = try batchDilithiumNTT(flat, numPolys: numPolys)
        return stride(from: 0, to: result.count, by: 256).map { start in
            Array(result[start..<start+256]).map { DilithiumField(value: $0) }
        }
    }

    /// Inverse NTT a batch of Dilithium polynomials
    public func batchDilithiumINTT(_ polys: [[DilithiumField]]) throws -> [[DilithiumField]] {
        let numPolys = polys.count
        let flat = polys.flatMap { $0.map { $0.value } }
        let result = try batchDilithiumINTT(flat, numPolys: numPolys)
        return stride(from: 0, to: result.count, by: 256).map { start in
            Array(result[start..<start+256]).map { DilithiumField(value: $0) }
        }
    }
}
