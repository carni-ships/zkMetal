// RNS NTT Engine — GPU-accelerated batch NTT for Homomorphic Encryption
// Performs NTT on all L RNS limbs of a polynomial simultaneously.
// Each (polynomial, modulus) pair is handled by one threadgroup in shared memory.
// Optimal for HE parameter sizes: N=4096-16384, L=3-15 limbs.

import Foundation
import Metal

public class RNSNTTEngine {
    public static let version = Versions.rnsNTT
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    let nttBatchFunction: MTLComputePipelineState
    let inttBatchFunction: MTLComputePipelineState
    let pointwiseMulFunction: MTLComputePipelineState
    let pointwiseAddFunction: MTLComputePipelineState
    let globalButterflyFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState

    public let moduli: [UInt32]
    public let barrettKs: [UInt32]
    public let logN: Int
    public let degree: Int

    // Cached GPU buffers
    private var moduliBuf: MTLBuffer
    private var barrettBuf: MTLBuffer
    private var fwdTwiddleCache: MTLBuffer?
    private var invTwiddleCache: MTLBuffer?
    private var invNBuf: MTLBuffer?
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufBytes: Int = 0

    /// Maximum polynomial degree that fits in threadgroup shared memory (32KB / 4 bytes)
    public static let maxSharedMemoryDegree = 8192

    public init(logN: Int, moduli: [UInt32]) throws {
        precondition(logN >= 1 && logN <= 16, "logN must be in [1, 16]")
        precondition(!moduli.isEmpty, "Need at least one modulus")

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        self.logN = logN
        self.degree = 1 << logN
        self.moduli = moduli
        self.barrettKs = moduli.map { barrettConstant($0) }

        let library = try RNSNTTEngine.compileShaders(device: device)

        guard let nttBatchFn = library.makeFunction(name: "rns_ntt_batch"),
              let inttBatchFn = library.makeFunction(name: "rns_intt_batch"),
              let pwMulFn = library.makeFunction(name: "rns_pointwise_mul"),
              let pwAddFn = library.makeFunction(name: "rns_pointwise_add"),
              let globalBfFn = library.makeFunction(name: "rns_ntt_butterfly_global"),
              let scaleFn = library.makeFunction(name: "rns_ntt_scale") else {
            throw MSMError.missingKernel
        }

        self.nttBatchFunction = try device.makeComputePipelineState(function: nttBatchFn)
        self.inttBatchFunction = try device.makeComputePipelineState(function: inttBatchFn)
        self.pointwiseMulFunction = try device.makeComputePipelineState(function: pwMulFn)
        self.pointwiseAddFunction = try device.makeComputePipelineState(function: pwAddFn)
        self.globalButterflyFunction = try device.makeComputePipelineState(function: globalBfFn)
        self.scaleFunction = try device.makeComputePipelineState(function: scaleFn)

        // Create moduli and Barrett constant buffers
        self.moduliBuf = RNSNTTEngine.createBuffer(device: device, data: moduli)!
        self.barrettBuf = RNSNTTEngine.createBuffer(device: device, data: barrettKs)!

        // Precompute twiddles and inverse N
        self.fwdTwiddleCache = precomputeAllTwiddles(forward: true)
        self.invTwiddleCache = precomputeAllTwiddles(forward: false)
        self.invNBuf = precomputeInvNs()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/he/rns_ntt.metal", encoding: .utf8)
        let cleanSource = source.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let header = """
        #include <metal_stdlib>
        using namespace metal;

        """
        let combined = header + cleanSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("he/rns_ntt.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/he/rns_ntt.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    private static func createBuffer(device: MTLDevice, data: [UInt32]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<UInt32>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Twiddle precomputation

    /// Precompute twiddles for all moduli, packed: [mod0_tw0..twN/2-1, mod1_tw0..., ...]
    private func precomputeAllTwiddles(forward: Bool) -> MTLBuffer {
        let half = degree / 2
        var allTwiddles = [UInt32]()
        allTwiddles.reserveCapacity(moduli.count * half)

        for q in moduli {
            let tw = forward
                ? precomputeTwiddles(modulus: q, logN: logN)
                : precomputeInverseTwiddles(modulus: q, logN: logN)
            allTwiddles.append(contentsOf: tw)
        }

        return RNSNTTEngine.createBuffer(device: device, data: allTwiddles)!
    }

    /// Precompute 1/N mod qi for each modulus
    private func precomputeInvNs() -> MTLBuffer {
        let n32 = UInt32(degree)
        let invNs = moduli.map { q in
            RNSLimb.inverse(RNSLimb(value: n32, modulus: q)).value
        }
        return RNSNTTEngine.createBuffer(device: device, data: invNs)!
    }

    private func getOrCreateDataBuf(byteCount: Int) -> MTLBuffer {
        if byteCount <= cachedDataBufBytes, let buf = cachedDataBuf { return buf }
        let buf = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        cachedDataBuf = buf
        cachedDataBufBytes = byteCount
        return buf
    }

    // MARK: - Forward NTT

    /// Forward NTT on all RNS limbs of one polynomial (GPU batch)
    public func forwardNTT(_ poly: inout RNSPoly) throws {
        precondition(poly.degree == degree)
        precondition(poly.moduli == moduli)

        var packed = poly.packed()
        try forwardNTTPacked(&packed, polyCount: 1)
        poly.unpack(packed)
    }

    /// Forward NTT on packed data for multiple polynomials
    public func forwardNTTPacked(_ data: inout [UInt32], polyCount: Int) throws {
        let totalElements = polyCount * moduli.count * degree
        precondition(data.count == totalElements)

        let byteCount = totalElements * MemoryLayout<UInt32>.stride
        let dataBuf = getOrCreateDataBuf(byteCount: byteCount)
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        try forwardNTTBuffer(dataBuf, polyCount: polyCount)

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: totalElements)
        data = Array(UnsafeBufferPointer(start: ptr, count: totalElements))
    }

    /// Forward NTT operating directly on a Metal buffer
    public func forwardNTTBuffer(_ dataBuf: MTLBuffer, polyCount: Int) throws {
        let numTGs = polyCount * moduli.count

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(nttBatchFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(fwdTwiddleCache!, offset: 0, index: 1)
        enc.setBuffer(moduliBuf, offset: 0, index: 2)
        enc.setBuffer(barrettBuf, offset: 0, index: 3)
        var logNVal = UInt32(logN)
        var numLimbsVal = UInt32(moduli.count)
        var polyCountVal = UInt32(polyCount)
        enc.setBytes(&logNVal, length: 4, index: 4)
        enc.setBytes(&numLimbsVal, length: 4, index: 5)
        enc.setBytes(&polyCountVal, length: 4, index: 6)

        // Each threadgroup processes one (poly, limb) pair
        // Threadgroup size = min(degree/2, 256) to handle all butterflies
        let tgSize = min(degree / 2, 256)
        let sharedMem = degree * MemoryLayout<UInt32>.stride

        enc.setThreadgroupMemoryLength(sharedMem, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: numTGs, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse NTT

    /// Inverse NTT on all RNS limbs of one polynomial
    public func inverseNTT(_ poly: inout RNSPoly) throws {
        precondition(poly.degree == degree)
        precondition(poly.moduli == moduli)

        var packed = poly.packed()
        try inverseNTTPacked(&packed, polyCount: 1)
        poly.unpack(packed)
    }

    /// Inverse NTT on packed data
    public func inverseNTTPacked(_ data: inout [UInt32], polyCount: Int) throws {
        let totalElements = polyCount * moduli.count * degree
        precondition(data.count == totalElements)

        let byteCount = totalElements * MemoryLayout<UInt32>.stride
        let dataBuf = getOrCreateDataBuf(byteCount: byteCount)
        data.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, byteCount)
        }

        try inverseNTTBuffer(dataBuf, polyCount: polyCount)

        let ptr = dataBuf.contents().bindMemory(to: UInt32.self, capacity: totalElements)
        data = Array(UnsafeBufferPointer(start: ptr, count: totalElements))
    }

    /// Inverse NTT on Metal buffer
    public func inverseNTTBuffer(_ dataBuf: MTLBuffer, polyCount: Int) throws {
        let numTGs = polyCount * moduli.count

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(inttBatchFunction)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invTwiddleCache!, offset: 0, index: 1)
        enc.setBuffer(moduliBuf, offset: 0, index: 2)
        enc.setBuffer(barrettBuf, offset: 0, index: 3)
        enc.setBuffer(invNBuf!, offset: 0, index: 4)
        var logNVal = UInt32(logN)
        var numLimbsVal = UInt32(moduli.count)
        var polyCountVal = UInt32(polyCount)
        enc.setBytes(&logNVal, length: 4, index: 5)
        enc.setBytes(&numLimbsVal, length: 4, index: 6)
        enc.setBytes(&polyCountVal, length: 4, index: 7)

        let tgSize = min(degree / 2, 256)
        let sharedMem = degree * MemoryLayout<UInt32>.stride

        enc.setThreadgroupMemoryLength(sharedMem, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: numTGs, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Pointwise operations

    /// Pointwise multiply two RNS polynomials (both must be in NTT domain)
    public func multiply(_ a: RNSPoly, _ b: RNSPoly) throws -> RNSPoly {
        precondition(a.degree == degree && b.degree == degree)
        precondition(a.moduli == moduli && b.moduli == moduli)

        let packedA = a.packed()
        let packedB = b.packed()
        let totalElements = moduli.count * degree
        let byteCount = totalElements * MemoryLayout<UInt32>.stride

        let bufA = getOrCreateDataBuf(byteCount: byteCount)
        packedA.withUnsafeBytes { src in
            memcpy(bufA.contents(), src.baseAddress!, byteCount)
        }
        let bufB = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        packedB.withUnsafeBytes { src in
            memcpy(bufB.contents(), src.baseAddress!, byteCount)
        }
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pointwiseMulFunction)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.setBuffer(moduliBuf, offset: 0, index: 3)
        enc.setBuffer(barrettBuf, offset: 0, index: 4)
        var numLimbsVal = UInt32(moduli.count)
        var degreeVal = UInt32(degree)
        enc.setBytes(&numLimbsVal, length: 4, index: 5)
        enc.setBytes(&degreeVal, length: 4, index: 6)

        let tgSize = min(256, Int(pointwiseMulFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalElements, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = RNSPoly(degree: degree, moduli: moduli)
        let ptr = bufOut.contents().bindMemory(to: UInt32.self, capacity: totalElements)
        let packed = Array(UnsafeBufferPointer(start: ptr, count: totalElements))
        result.unpack(packed)
        return result
    }

    /// Pointwise add two RNS polynomials
    public func add(_ a: RNSPoly, _ b: RNSPoly) throws -> RNSPoly {
        precondition(a.degree == degree && b.degree == degree)
        precondition(a.moduli == moduli && b.moduli == moduli)

        let packedA = a.packed()
        let packedB = b.packed()
        let totalElements = moduli.count * degree
        let byteCount = totalElements * MemoryLayout<UInt32>.stride

        let bufA = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        packedA.withUnsafeBytes { src in
            memcpy(bufA.contents(), src.baseAddress!, byteCount)
        }
        let bufB = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        packedB.withUnsafeBytes { src in
            memcpy(bufB.contents(), src.baseAddress!, byteCount)
        }
        let bufOut = device.makeBuffer(length: byteCount, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pointwiseAddFunction)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        enc.setBuffer(moduliBuf, offset: 0, index: 3)
        var numLimbsVal = UInt32(moduli.count)
        var degreeVal = UInt32(degree)
        enc.setBytes(&numLimbsVal, length: 4, index: 4)
        enc.setBytes(&degreeVal, length: 4, index: 5)

        let tgSize = min(256, Int(pointwiseAddFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalElements, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var result = RNSPoly(degree: degree, moduli: moduli)
        let ptr = bufOut.contents().bindMemory(to: UInt32.self, capacity: totalElements)
        let packed = Array(UnsafeBufferPointer(start: ptr, count: totalElements))
        result.unpack(packed)
        return result
    }

    // MARK: - Batch operations

    /// Batch forward NTT on multiple polynomials at once
    public func batchForwardNTT(_ polys: inout [RNSPoly]) throws {
        let polyCount = polys.count
        if polyCount == 0 { return }

        var packed = [UInt32]()
        packed.reserveCapacity(polyCount * moduli.count * degree)
        for p in polys {
            packed.append(contentsOf: p.packed())
        }

        try forwardNTTPacked(&packed, polyCount: polyCount)

        let elementsPerPoly = moduli.count * degree
        for i in 0..<polyCount {
            let start = i * elementsPerPoly
            let slice = Array(packed[start..<start + elementsPerPoly])
            polys[i].unpack(slice)
        }
    }

    /// Batch inverse NTT on multiple polynomials
    public func batchInverseNTT(_ polys: inout [RNSPoly]) throws {
        let polyCount = polys.count
        if polyCount == 0 { return }

        var packed = [UInt32]()
        packed.reserveCapacity(polyCount * moduli.count * degree)
        for p in polys {
            packed.append(contentsOf: p.packed())
        }

        try inverseNTTPacked(&packed, polyCount: polyCount)

        let elementsPerPoly = moduli.count * degree
        for i in 0..<polyCount {
            let start = i * elementsPerPoly
            let slice = Array(packed[start..<start + elementsPerPoly])
            polys[i].unpack(slice)
        }
    }

    // MARK: - High-level API

    /// Forward NTT a single polynomial, returning result
    public func ntt(_ poly: RNSPoly) throws -> RNSPoly {
        var p = poly
        try forwardNTT(&p)
        return p
    }

    /// Inverse NTT a single polynomial, returning result
    public func intt(_ poly: RNSPoly) throws -> RNSPoly {
        var p = poly
        try inverseNTT(&p)
        return p
    }
}
