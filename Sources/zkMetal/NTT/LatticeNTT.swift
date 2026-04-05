// Lattice NTT Engine for Post-Quantum Cryptography (Kyber/Dilithium)
// GPU-accelerated NTT over small moduli:
//   Kyber:     q = 3329    (12-bit, elements fit in UInt16)
//   Dilithium: q = 8380417 (23-bit, elements fit in UInt32)
//
// Fixed polynomial size n=256 for both (standard ring dimension).
// GPU advantage: batch processing thousands of polynomials in parallel,
// each polynomial fits entirely in 32KB threadgroup shared memory.
//
// API uses UInt32 arrays for uniformity (Kyber values still < 3329).

import Foundation
import Metal

// MARK: - Lattice Mode Selection

public enum LatticeMode {
    case kyber       // q = 3329, n = 256
    case dilithium   // q = 8380417, n = 256

    public var modulus: UInt32 {
        switch self {
        case .kyber: return 3329
        case .dilithium: return 8380417
        }
    }

    public var polySize: Int { 256 }
}

// MARK: - Lattice NTT Engine

public class LatticeNTT {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states for each kernel
    private let kyberNTTPipeline: MTLComputePipelineState
    private let kyberINTTPipeline: MTLComputePipelineState
    private let dilithiumNTTPipeline: MTLComputePipelineState
    private let dilithiumINTTPipeline: MTLComputePipelineState
    private let kyberPointwisePipeline: MTLComputePipelineState
    private let dilithiumPointwisePipeline: MTLComputePipelineState

    // Precomputed twiddle factor buffers
    private let kyberTwiddleBuf: MTLBuffer
    private let kyberInvNBuf: MTLBuffer
    private let dilithiumTwiddleBuf: MTLBuffer
    private let dilithiumInvNBuf: MTLBuffer

    // Threadgroup configuration
    private let nttThreadgroupSize = 32

    // GPU availability flag
    private let gpuAvailable: Bool

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        // Compile shaders
        let library = try LatticeNTT.compileShaders(device: device)

        // Create pipeline states
        guard let kNTTFn = library.makeFunction(name: "lattice_ntt_kyber"),
              let kINTTFn = library.makeFunction(name: "lattice_intt_kyber"),
              let dNTTFn = library.makeFunction(name: "lattice_ntt_dilithium"),
              let dINTTFn = library.makeFunction(name: "lattice_intt_dilithium"),
              let kPwFn = library.makeFunction(name: "lattice_pointwise_kyber"),
              let dPwFn = library.makeFunction(name: "lattice_pointwise_dilithium") else {
            throw MSMError.missingKernel
        }

        self.kyberNTTPipeline = try device.makeComputePipelineState(function: kNTTFn)
        self.kyberINTTPipeline = try device.makeComputePipelineState(function: kINTTFn)
        self.dilithiumNTTPipeline = try device.makeComputePipelineState(function: dNTTFn)
        self.dilithiumINTTPipeline = try device.makeComputePipelineState(function: dINTTFn)
        self.kyberPointwisePipeline = try device.makeComputePipelineState(function: kPwFn)
        self.dilithiumPointwisePipeline = try device.makeComputePipelineState(function: dPwFn)

        // Precompute twiddle factors
        let kTw = kyberTwiddles()
        let kTwU16 = kTw.map { $0.value }
        guard let kTwBuf = device.makeBuffer(bytes: kTwU16, length: kTwU16.count * MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create Kyber twiddle buffer")
        }
        self.kyberTwiddleBuf = kTwBuf

        let kyberInvNVal = kyberInverse(KyberField(value: 128)).value
        var invNK = kyberInvNVal
        guard let kInvNBuf = device.makeBuffer(bytes: &invNK, length: MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create Kyber invN buffer")
        }
        self.kyberInvNBuf = kInvNBuf

        let dTw = dilithiumTwiddles()
        let dTwU32 = dTw.map { $0.value }
        guard let dTwBuf = device.makeBuffer(bytes: dTwU32, length: dTwU32.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create Dilithium twiddle buffer")
        }
        self.dilithiumTwiddleBuf = dTwBuf

        let dilInvNVal = dilithiumInverse(DilithiumField(value: 128)).value
        var invND = dilInvNVal
        guard let dInvNBuf = device.makeBuffer(bytes: &invND, length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create Dilithium invN buffer")
        }
        self.dilithiumInvNBuf = dInvNBuf

        self.gpuAvailable = true
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/ntt/lattice_ntt.metal", encoding: .utf8)
        let clean = source.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: clean, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/ntt/lattice_ntt.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Public API

    /// Forward NTT. Input: flat array of polynomials (each 256 elements).
    /// For Kyber, values must be < 3329. For Dilithium, values must be < 8380417.
    /// Returns NTT-domain coefficients.
    public func ntt(data: [UInt32], mode: LatticeMode) -> [UInt32] {
        let n = mode.polySize
        precondition(data.count % n == 0, "Data length must be a multiple of \(n)")
        let numPolys = data.count / n

        // CPU fallback for single polynomial or if GPU dispatch fails
        if numPolys == 0 { return [] }

        do {
            return try gpuNTT(data: data, numPolys: numPolys, mode: mode)
        } catch {
            return cpuNTT(data: data, mode: mode)
        }
    }

    /// Inverse NTT. Input: flat array of NTT-domain polynomials.
    /// Returns coefficient-domain values.
    public func intt(data: [UInt32], mode: LatticeMode) -> [UInt32] {
        let n = mode.polySize
        precondition(data.count % n == 0, "Data length must be a multiple of \(n)")
        let numPolys = data.count / n

        if numPolys == 0 { return [] }

        do {
            return try gpuINTT(data: data, numPolys: numPolys, mode: mode)
        } catch {
            return cpuINTT(data: data, mode: mode)
        }
    }

    /// Pointwise multiplication in NTT domain.
    /// out[i] = a[i] * b[i] mod q for all i.
    public func pointwiseMul(a: [UInt32], b: [UInt32], mode: LatticeMode) -> [UInt32] {
        precondition(a.count == b.count, "Arrays must have equal length")
        if a.isEmpty { return [] }

        do {
            return try gpuPointwiseMul(a: a, b: b, mode: mode)
        } catch {
            return cpuPointwiseMul(a: a, b: b, mode: mode)
        }
    }

    // MARK: - GPU NTT (Kyber)

    private func gpuNTT(data: [UInt32], numPolys: Int, mode: LatticeMode) throws -> [UInt32] {
        switch mode {
        case .kyber:
            return try gpuKyberNTT(data: data, numPolys: numPolys)
        case .dilithium:
            return try gpuDilithiumNTT(data: data, numPolys: numPolys)
        }
    }

    private func gpuINTT(data: [UInt32], numPolys: Int, mode: LatticeMode) throws -> [UInt32] {
        switch mode {
        case .kyber:
            return try gpuKyberINTT(data: data, numPolys: numPolys)
        case .dilithium:
            return try gpuDilithiumINTT(data: data, numPolys: numPolys)
        }
    }

    private func gpuKyberNTT(data: [UInt32], numPolys: Int) throws -> [UInt32] {
        // Convert UInt32 -> UInt16 for Kyber
        let u16Data = data.map { UInt16($0) }
        let byteCount = u16Data.count * MemoryLayout<UInt16>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        u16Data.withUnsafeBytes { src in memcpy(dataBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberNTTPipeline)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(kyberTwiddleBuf, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt16.self, capacity: u16Data.count)
        let result = Array(UnsafeBufferPointer(start: ptr, count: u16Data.count))
        return result.map { UInt32($0) }
    }

    private func gpuKyberINTT(data: [UInt32], numPolys: Int) throws -> [UInt32] {
        let u16Data = data.map { UInt16($0) }
        let byteCount = u16Data.count * MemoryLayout<UInt16>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        u16Data.withUnsafeBytes { src in memcpy(dataBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberINTTPipeline)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(kyberTwiddleBuf, offset: 0, index: 1)  // INTT uses forward twiddles
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.setBuffer(kyberInvNBuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt16.self, capacity: u16Data.count)
        let result = Array(UnsafeBufferPointer(start: ptr, count: u16Data.count))
        return result.map { UInt32($0) }
    }

    // MARK: - GPU NTT (Dilithium)

    private func gpuDilithiumNTT(data: [UInt32], numPolys: Int) throws -> [UInt32] {
        let byteCount = data.count * MemoryLayout<UInt32>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        data.withUnsafeBytes { src in memcpy(dataBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumNTTPipeline)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(dilithiumTwiddleBuf, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: data.count)
        return Array(UnsafeBufferPointer(start: ptr, count: data.count))
    }

    private func gpuDilithiumINTT(data: [UInt32], numPolys: Int) throws -> [UInt32] {
        let byteCount = data.count * MemoryLayout<UInt32>.stride
        guard let dataBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        data.withUnsafeBytes { src in memcpy(dataBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumINTTPipeline)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(dilithiumTwiddleBuf, offset: 0, index: 1)
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.setBuffer(dilithiumInvNBuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: data.count)
        return Array(UnsafeBufferPointer(start: ptr, count: data.count))
    }

    // MARK: - GPU Pointwise Multiply

    private func gpuPointwiseMul(a: [UInt32], b: [UInt32], mode: LatticeMode) throws -> [UInt32] {
        switch mode {
        case .kyber:
            return try gpuKyberPointwiseMul(a: a, b: b)
        case .dilithium:
            return try gpuDilithiumPointwiseMul(a: a, b: b)
        }
    }

    private func gpuKyberPointwiseMul(a: [UInt32], b: [UInt32]) throws -> [UInt32] {
        let count = a.count
        let a16 = a.map { UInt16($0) }
        let b16 = b.map { UInt16($0) }
        let byteCount = count * MemoryLayout<UInt16>.stride

        guard let aBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        a16.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteCount) }
        b16.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(kyberPointwisePipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: 4, index: 3)
        let tgSize = min(256, Int(kyberPointwisePipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: UInt16.self, capacity: count)
        let result = Array(UnsafeBufferPointer(start: ptr, count: count))
        return result.map { UInt32($0) }
    }

    private func gpuDilithiumPointwiseMul(a: [UInt32], b: [UInt32]) throws -> [UInt32] {
        let count = a.count
        let byteCount = count * MemoryLayout<UInt32>.stride

        guard let aBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, byteCount) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, byteCount) }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(dilithiumPointwisePipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: 4, index: 3)
        let tgSize = min(256, Int(dilithiumPointwisePipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
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

    // MARK: - CPU Fallback

    /// CPU NTT fallback (delegates to existing LatticeFields CPU implementations)
    public func cpuNTT(data: [UInt32], mode: LatticeMode) -> [UInt32] {
        let n = mode.polySize
        let numPolys = data.count / n
        var result = [UInt32]()
        result.reserveCapacity(data.count)

        for p in 0..<numPolys {
            let start = p * n
            let end = start + n
            let slice = Array(data[start..<end])

            switch mode {
            case .kyber:
                var poly = slice.map { KyberField(value: UInt16($0)) }
                kyberNTTCPU(&poly)
                result.append(contentsOf: poly.map { UInt32($0.value) })
            case .dilithium:
                var poly = slice.map { DilithiumField(value: $0) }
                dilithiumNTTCPU(&poly)
                result.append(contentsOf: poly.map { $0.value })
            }
        }
        return result
    }

    /// CPU INTT fallback
    public func cpuINTT(data: [UInt32], mode: LatticeMode) -> [UInt32] {
        let n = mode.polySize
        let numPolys = data.count / n
        var result = [UInt32]()
        result.reserveCapacity(data.count)

        for p in 0..<numPolys {
            let start = p * n
            let end = start + n
            let slice = Array(data[start..<end])

            switch mode {
            case .kyber:
                var poly = slice.map { KyberField(value: UInt16($0)) }
                kyberInvNTTCPU(&poly)
                result.append(contentsOf: poly.map { UInt32($0.value) })
            case .dilithium:
                var poly = slice.map { DilithiumField(value: $0) }
                dilithiumInvNTTCPU(&poly)
                result.append(contentsOf: poly.map { $0.value })
            }
        }
        return result
    }

    /// CPU pointwise multiply fallback
    public func cpuPointwiseMul(a: [UInt32], b: [UInt32], mode: LatticeMode) -> [UInt32] {
        var result = [UInt32](repeating: 0, count: a.count)
        let q = UInt64(mode.modulus)
        for i in 0..<a.count {
            result[i] = UInt32((UInt64(a[i]) * UInt64(b[i])) % q)
        }
        return result
    }

    // MARK: - MTLBuffer-based APIs for zero-copy interop

    /// NTT operating directly on MTLBuffers (avoids CPU-GPU copy overhead)
    public func nttBuffer(_ buffer: MTLBuffer, numPolys: Int, mode: LatticeMode) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        switch mode {
        case .kyber:
            enc.setComputePipelineState(kyberNTTPipeline)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(kyberTwiddleBuf, offset: 0, index: 1)
        case .dilithium:
            enc.setComputePipelineState(dilithiumNTTPipeline)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(dilithiumTwiddleBuf, offset: 0, index: 1)
        }
        var numP = UInt32(numPolys)
        enc.setBytes(&numP, length: 4, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// INTT operating directly on MTLBuffers
    public func inttBuffer(_ buffer: MTLBuffer, numPolys: Int, mode: LatticeMode) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        switch mode {
        case .kyber:
            enc.setComputePipelineState(kyberINTTPipeline)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(kyberTwiddleBuf, offset: 0, index: 1)
            var numP = UInt32(numPolys)
            enc.setBytes(&numP, length: 4, index: 2)
            enc.setBuffer(kyberInvNBuf, offset: 0, index: 3)
        case .dilithium:
            enc.setComputePipelineState(dilithiumINTTPipeline)
            enc.setBuffer(buffer, offset: 0, index: 0)
            enc.setBuffer(dilithiumTwiddleBuf, offset: 0, index: 1)
            var numP = UInt32(numPolys)
            enc.setBytes(&numP, length: 4, index: 2)
            enc.setBuffer(dilithiumInvNBuf, offset: 0, index: 3)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: numPolys, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nttThreadgroupSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }
}
