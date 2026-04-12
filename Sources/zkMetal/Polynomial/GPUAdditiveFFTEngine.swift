// GPU-accelerated Additive FFT (Cantor/Lin-Chung-Han) for GF(2^8)
//
// Fuses all log₂(n) additive butterfly levels into a single Metal dispatch.
// Each GPU thread processes one GF(2^8) element through all k butterfly levels,
// achieving 1 global read + 1 global write with all intermediate data in registers.
//
// Additive FFT over GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1:
//   twist:   lo ^= s * hi    (GF(2^8) multiply by basis element)
//   propagate: hi ^= lo       (XOR — free on GPU)
//
// GPU kernel advantage over NEON: all k levels fused into one dispatch avoids
// k separate kernel launches and k separate global memory round-trips.

import Foundation
import Metal

public class GPUAdditiveFFTEngine {
    public static let version = Versions.additiveFFT

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let forwardFn: MTLComputePipelineState?
    let inverseFn: MTLComputePipelineState?
    let forwardBatchFn: MTLComputePipelineState?
    let pointwiseMulFn: MTLComputePipelineState?
    let fusedForwardThenMulFn: MTLComputePipelineState?

    /// Cached data buffer to avoid per-call allocation.
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    /// 256x256 GF(2^8) multiplication LUT (64KB), populated at init.
    var lutBuffer: MTLBuffer?

    private let tuning: TuningConfig

    /// CPU fallback threshold (use CPU for small transforms where GPU overhead dominates).
    private static let cpuFallbackLogN = 8   // n <= 256: CPU NEON faster than GPU dispatch

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.tuning = TuningManager.shared.config(device: device)

        // Populate the 256x256 GF(2^8) multiplication LUT at init time.
        self.lutBuffer = GPUAdditiveFFTEngine.createLUTBuffer(device: device)

        // Try to compile Metal shaders with USE_LUT=1; if this fails, GPU path is unavailable
        if let library = try? GPUAdditiveFFTEngine.compileShaders(device: device, defines: ["USE_LUT"]) {
            self.forwardFn = try? device.makeComputePipelineState(function: library.makeFunction(name: "additive_fft_gf8_forward")!)
            self.inverseFn = try? device.makeComputePipelineState(function: library.makeFunction(name: "additive_fft_gf8_inverse")!)
            self.forwardBatchFn = try? device.makeComputePipelineState(function: library.makeFunction(name: "additive_fft_gf8_forward_batch")!)
            self.pointwiseMulFn = try? device.makeComputePipelineState(function: library.makeFunction(name: "gf28_pointwise_mul")!)
            self.fusedForwardThenMulFn = try? device.makeComputePipelineState(function: library.makeFunction(name: "additive_fft_gf8_forward_then_pointwise_mul")!)
        } else {
            self.forwardFn = nil
            self.inverseFn = nil
            self.forwardBatchFn = nil
            self.pointwiseMulFn = nil
            self.fusedForwardThenMulFn = nil
        }
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice, defines: [String] = []) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/additive/additive_fft_gf8.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        // Add preprocessor macros (e.g. "USE_LUT=1")
        if !defines.isEmpty {
            let macros = defines.joined(separator: ",")
            // MTLCompileOptions doesn't have direct macro support, but we can prefix the source
            let prefixedSource = defines.map { "#define \($0)\n" }.joined() + source
            return try device.makeLibrary(source: prefixedSource, options: options)
        }
        return try device.makeLibrary(source: source, options: options)
    }

    // MARK: - LUT Population

    /// Creates and populates the 256x256 GF(2^8) multiplication LUT (64KB).
    private static func createLUTBuffer(device: MTLDevice) -> MTLBuffer? {
        guard let buf = device.makeBuffer(length: 65536, options: .storageModeShared) else {
            return nil
        }
        var lut = [UInt8](repeating: 0, count: 65536)
        for a: UInt8 in 0...255 {
            for b: UInt8 in 0...255 {
                lut[Int(a) * 256 + Int(b)] = GPUAdditiveFFTEngine.gf28MulCPU(a, b)
            }
        }
        lut.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, 65536)
        }
        return buf
    }

    // MARK: - Buffer Management

    private func getOrCreateDataBuffer(elementCount: Int) -> MTLBuffer {
        if elementCount <= cachedDataBufElements, let buf = cachedDataBuf { return buf }
        let byteCount = elementCount  // 1 byte per GF(2^8) element
        let buf = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        cachedDataBuf = buf
        cachedDataBufElements = elementCount
        return buf
    }

    private func createUInt8Buffer(_ data: [UInt8]) -> MTLBuffer? {
        let byteCount = data.count
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Forward Additive FFT

    /// Forward additive FFT over GF(2^8).
    /// - Parameters:
    ///   - data: input/output array of n GF(2^8) elements (as UInt8), modified in-place on GPU
    ///   - n: transform size (must be power of 2)
    ///   - k: log₂(n)
    ///   - basis: k GF(2^8) basis elements
    /// - Returns: transformed array (copy of GPU buffer)
    public func forward(data: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        precondition(data.count == n, "Data must have exactly n elements")
        precondition(basis.count == k, "Basis must have exactly k elements")
        precondition(n == (1 << k), "n must equal 2^k")

        // CPU fallback for small transforms
        if k <= GPUAdditiveFFTEngine.cpuFallbackLogN {
            return try GPUAdditiveFFTEngine.cpuForward(data: data, n: n, k: k, basis: basis)
        }

        let dataBuf = getOrCreateDataBuffer(elementCount: n)
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n)
        }

        guard let basisBuf = createUInt8Buffer(basis) else {
            throw MSMError.gpuError("Failed to allocate basis buffer")
        }

        guard let fn = forwardFn else {
            throw MSMError.gpuError("Forward pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(dataBuf, offset: 0, index: 1)
        enc.setBuffer(basisBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Forward additive FFT in-place on pre-allocated GPU buffer.
    public func forwardInPlace(dataBuffer: MTLBuffer, n: Int, k: Int, basisBuffer: MTLBuffer) throws {
        precondition(n == (1 << k))
        guard let fn = forwardFn else {
            throw MSMError.gpuError("Forward pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(dataBuffer, offset: 0, index: 1)
        enc.setBuffer(basisBuffer, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse Additive FFT

    /// Inverse additive FFT over GF(2^8).
    public func inverse(data: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        precondition(data.count == n)
        precondition(basis.count == k)
        precondition(n == (1 << k))

        if k <= GPUAdditiveFFTEngine.cpuFallbackLogN {
            return try GPUAdditiveFFTEngine.cpuInverse(data: data, n: n, k: k, basis: basis)
        }

        let dataBuf = getOrCreateDataBuffer(elementCount: n)
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n)
        }

        guard let basisBuf = createUInt8Buffer(basis) else {
            throw MSMError.gpuError("Failed to allocate basis buffer")
        }

        guard let fn = inverseFn else {
            throw MSMError.gpuError("Inverse pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(dataBuf, offset: 0, index: 1)
        enc.setBuffer(basisBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Inverse additive FFT in-place on pre-allocated GPU buffer.
    public func inverseInPlace(dataBuffer: MTLBuffer, n: Int, k: Int, basisBuffer: MTLBuffer) throws {
        precondition(n == (1 << k))
        guard let fn = inverseFn else {
            throw MSMError.gpuError("Inverse pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(dataBuffer, offset: 0, index: 1)
        enc.setBuffer(basisBuffer, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Batch Forward Additive FFT

    /// Batch forward additive FFT for multiple independent arrays.
    /// - Parameters:
    ///   - data: flat array of batch * n GF(2^8) elements
    ///   - n: size of each transform (power of 2)
    ///   - k: log₂(n)
    ///   - batch: number of independent transforms
    ///   - basis: k GF(2^8) basis elements (shared across all transforms)
    /// - Returns: transformed array (batch * n elements)
    public func forwardBatch(data: [UInt8], n: Int, k: Int, batch: Int, basis: [UInt8]) throws -> [UInt8] {
        let total = batch * n
        precondition(data.count == total)
        precondition(basis.count == k)
        precondition(n == (1 << k))

        guard let dataBuf = device.makeBuffer(length: total, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate data buffer")
        }
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, total)
        }

        guard let basisBuf = createUInt8Buffer(basis) else {
            throw MSMError.gpuError("Failed to allocate basis buffer")
        }

        guard let fn = forwardBatchFn else {
            throw MSMError.gpuError("Forward batch pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)
        var batchVal = UInt32(batch)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(dataBuf, offset: 0, index: 1)
        enc.setBuffer(basisBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        enc.setBytes(&batchVal, length: 4, index: 5)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt8.self, capacity: total)
        return Array(UnsafeBufferPointer(start: ptr, count: total))
    }

    // MARK: - Pointwise Multiply

    /// GF(2^8) pointwise multiply on GPU: out[i] = a[i] * b[i] in GF(2^8).
    public func pointwiseMultiply(a: [UInt8], b: [UInt8], n: Int) throws -> [UInt8] {
        precondition(a.count == n && b.count == n)

        guard let aBuf = createUInt8Buffer(a),
              let bBuf = createUInt8Buffer(b),
              let outBuf = device.makeBuffer(length: n, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate buffers")
        }

        guard let fn = pointwiseMulFn else {
            throw MSMError.gpuError("Pointwise multiply pipeline not available (Metal shader compilation failed)")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt8.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU Reference Implementation (for small transforms and correctness testing)

    /// GF(2^8) multiplication: shift-XOR carry-less multiply + 0x11B reduction.
    public static func gf28MulCPU(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        p ^= UInt16(a & 1)   * UInt16(b)
        p ^= UInt16(a & 2)   * UInt16(b << 1)
        p ^= UInt16(a & 4)   * UInt16(b << 2)
        p ^= UInt16(a & 8)   * UInt16(b << 3)
        p ^= UInt16(a & 16)  * UInt16(b << 4)
        p ^= UInt16(a & 32)  * UInt16(b << 5)
        p ^= UInt16(a & 64)  * UInt16(b << 6)
        p ^= UInt16(a & 128) * UInt16(b << 7)
        // Reduce by x^8 + x^4 + x^3 + x + 1 (0x11B)
        let h = p >> 8
        if h & 0x01 != 0 { p ^= 0x11B << 0 }
        if h & 0x02 != 0 { p ^= 0x11B << 1 }
        if h & 0x04 != 0 { p ^= 0x11B << 2 }
        if h & 0x08 != 0 { p ^= 0x11B << 3 }
        if h & 0x10 != 0 { p ^= 0x11B << 4 }
        if h & 0x20 != 0 { p ^= 0x11B << 5 }
        if h & 0x40 != 0 { p ^= 0x11B << 6 }
        if h & 0x80 != 0 { p ^= 0x11B << 7 }
        return UInt8(p & 0xFF)
    }

    /// CPU forward additive FFT (DIF, same algorithm as GPU kernel).
    /// For small transforms where GPU dispatch overhead isn't justified.
    public static func cpuForward(data: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        var result = data
        for depth in 0..<k {
            let blockSize = n >> depth
            let half = blockSize >> 1
            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for i in 0..<half {
                    let hiIdx = blockStart + half + i
                    let loIdx = blockStart + i
                    let s = basis[depth]
                    let loVal = result[loIdx]
                    let hiVal = result[hiIdx]
                    // Twist: lo ^= s * hi
                    let twisted = loVal ^ gf28MulCPU(s, hiVal)
                    // Propagate: hi ^= lo
                    let propagated = loVal ^ hiVal
                    result[loIdx] = twisted
                    result[hiIdx] = propagated
                }
            }
        }
        return result
    }

    /// CPU inverse additive FFT (DIT, reverse depth order).
    public static func cpuInverse(data: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        var result = data
        for depth in stride(from: k - 1, through: 0, by: -1) {
            let blockSize = n >> depth
            let half = blockSize >> 1
            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for i in 0..<half {
                    let hiIdx = blockStart + half + i
                    let loIdx = blockStart + i
                    let s = basis[depth]
                    let hiVal = result[hiIdx]
                    let loVal = result[loIdx]
                    // Un-propagate: hi ^= lo
                    let unpropagated = hiVal ^ loVal
                    // Un-twist: lo ^= s * hi_new
                    let untwisted = loVal ^ gf28MulCPU(s, unpropagated)
                    result[loIdx] = untwisted
                    result[hiIdx] = unpropagated
                }
            }
        }
        return result
    }

    // MARK: - Polynomial Multiply via Additive FFT

    /// Multiply two polynomials over GF(2^8) using additive FFT.
    /// Both polynomials must have at most n/2 coefficients.
    /// Returns the product polynomial of degree < n.
    public func multiply(_ a: [UInt8], _ b: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        precondition(a.count + b.count - 1 <= n, "Product degree exceeds transform size")
        precondition(n == (1 << k), "n must be 2^k")

        // Pad inputs to transform size
        var aData = a + [UInt8](repeating: 0, count: n - a.count)
        var bData = b + [UInt8](repeating: 0, count: n - b.count)

        // Forward FFT both polynomials
        aData = try forward(data: aData, n: n, k: k, basis: basis)
        bData = try forward(data: bData, n: n, k: k, basis: basis)

        // Pointwise multiply in frequency domain
        let product = try pointwiseMultiply(a: aData, b: bData, n: n)

        // Inverse FFT to get coefficients
        return try inverse(data: product, n: n, k: k, basis: basis)
    }

    /// Fused polynomial multiply: forward FFT + pointwise multiply in a single dispatch.
    /// Avoids an intermediate global memory round-trip between the two stages.
    /// - Parameters:
    ///   - a: first polynomial coefficients
    ///   - b: second polynomial coefficients
    ///   - n: transform size (power of 2, must be >= max(a.count, b.count) * 2)
    ///   - k: log₂(n)
    ///   - basis: k GF(2^8) basis elements
    /// - Returns: product polynomial of degree < n
    public func multiplyFused(_ a: [UInt8], _ b: [UInt8], n: Int, k: Int, basis: [UInt8]) throws -> [UInt8] {
        precondition(a.count + b.count - 1 <= n, "Product degree exceeds transform size")
        precondition(n == (1 << k), "n must be 2^k")

        var aData = a + [UInt8](repeating: 0, count: n - a.count)
        var bData = b + [UInt8](repeating: 0, count: n - b.count)

        guard let fn = fusedForwardThenMulFn else {
            // Fallback to separate kernels
            return try multiply(aData, bData, n: n, k: k, basis: basis)
        }

        let aBuf = getOrCreateDataBuffer(elementCount: n)
        aData.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, n) }
        let bBuf = createUInt8Buffer(bData)!

        guard let basisBuf = createUInt8Buffer(basis) else {
            throw MSMError.gpuError("Failed to allocate basis buffer")
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var kVal = UInt32(k)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        enc.setBuffer(basisBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&kVal, length: 4, index: 4)
        enc.setBuffer(bBuf, offset: 0, index: 5)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Now inverse FFT on aBuf in-place
        let invResult = try inverse(data: Array(UnsafeBufferPointer(start: aBuf.contents().bindMemory(to: UInt8.self, capacity: n), count: n)), n: n, k: k, basis: basis)
        return invResult
    }
}
