// CosetDomainEngine — GPU-accelerated coset evaluation domain operations
//
// Coset domains are used in STARK/Plonk provers to evaluate constraint
// polynomials over a coset of the evaluation domain, avoiding division-by-zero
// at roots of unity.
//
// Operations:
//   - cosetShift: multiply evals[i] by g^i
//   - cosetUnshift: multiply evals[i] by g^(-i) (inverse of shift)
//   - evaluateVanishing: compute Z_H(x) = x^n - 1 at given points
//   - divideByVanishing: evals[i] / Z_H(coset_point[i])
//   - cosetNTT: coset shift + NTT (evaluate polynomial on coset)
//   - cosetINTT: INTT + coset unshift (interpolate from coset evaluations)

import Foundation
import Metal
import NeonFieldOps

// MARK: - CosetDomainEngine

public class CosetDomainEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // GPU pipelines
    private let cosetShiftBn254Fn: MTLComputePipelineState
    private let cosetUnshiftBn254Fn: MTLComputePipelineState
    private let cosetShiftBabyBearFn: MTLComputePipelineState
    private let cosetUnshiftBabyBearFn: MTLComputePipelineState
    private let vanishingBn254Fn: MTLComputePipelineState
    private let vanishingBabyBearFn: MTLComputePipelineState
    private let divByVanishBn254Fn: MTLComputePipelineState
    private let divByVanishBabyBearFn: MTLComputePipelineState

    // NTT engines (lazily initialized)
    private var frNTTEngine: NTTEngine?
    private var bbNTTEngine: BabyBearNTTEngine?

    // Cached coset generator powers: key = logSize
    private var frShiftPowersCache: [String: MTLBuffer] = [:]
    private var bbShiftPowersCache: [String: MTLBuffer] = [:]

    // CPU fallback threshold
    private static let cpuThresholdBn254 = 64
    private static let cpuThresholdBabyBear = 256

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try CosetDomainEngine.compileShaders(device: device)

        guard let shiftBn254 = library.makeFunction(name: "coset_shift_bn254"),
              let unshiftBn254 = library.makeFunction(name: "coset_unshift_bn254"),
              let shiftBb = library.makeFunction(name: "coset_shift_babybear"),
              let unshiftBb = library.makeFunction(name: "coset_unshift_babybear"),
              let vanishBn254 = library.makeFunction(name: "vanishing_poly_eval_bn254"),
              let vanishBb = library.makeFunction(name: "vanishing_poly_eval_babybear"),
              let divBn254 = library.makeFunction(name: "divide_by_vanishing_bn254"),
              let divBb = library.makeFunction(name: "divide_by_vanishing_babybear") else {
            throw MSMError.missingKernel
        }

        self.cosetShiftBn254Fn = try device.makeComputePipelineState(function: shiftBn254)
        self.cosetUnshiftBn254Fn = try device.makeComputePipelineState(function: unshiftBn254)
        self.cosetShiftBabyBearFn = try device.makeComputePipelineState(function: shiftBb)
        self.cosetUnshiftBabyBearFn = try device.makeComputePipelineState(function: unshiftBb)
        self.vanishingBn254Fn = try device.makeComputePipelineState(function: vanishBn254)
        self.vanishingBabyBearFn = try device.makeComputePipelineState(function: vanishBb)
        self.divByVanishBn254Fn = try device.makeComputePipelineState(function: divBn254)
        self.divByVanishBabyBearFn = try device.makeComputePipelineState(function: divBb)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldFr = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let fieldBb = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let cosetSrc = try String(contentsOfFile: shaderDir + "/ntt/coset_domain.metal", encoding: .utf8)

        func clean(_ src: String) -> String {
            src.split(separator: "\n")
                .filter { !$0.contains("#include") && !$0.contains("#ifndef") &&
                         !$0.contains("#define") && !$0.contains("#endif") }
                .joined(separator: "\n")
        }

        let combined = clean(fieldFr) + "\n" + clean(fieldBb) + "\n" + clean(cosetSrc)

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

    // MARK: - NTT engine accessors

    private func getFrNTTEngine() throws -> NTTEngine {
        if let e = frNTTEngine { return e }
        let e = try NTTEngine()
        frNTTEngine = e
        return e
    }

    private func getBbNTTEngine() throws -> BabyBearNTTEngine {
        if let e = bbNTTEngine { return e }
        let e = try BabyBearNTTEngine()
        bbNTTEngine = e
        return e
    }

    // MARK: - Generator power precomputation

    /// Get or compute powers of g: [g^0, g^1, ..., g^(n-1)]
    private func getFrPowers(generator: Fr, logSize: Int) -> MTLBuffer {
        let key = "\(logSize)_\(generator.v.0)_\(generator.v.4)"
        if let cached = frShiftPowersCache[key] { return cached }
        let n = 1 << logSize
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], generator)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                     options: .storageModeShared)!
        frShiftPowersCache[key] = buf
        return buf
    }

    /// Get or compute inverse powers: [g^0, g^(-1), g^(-2), ..., g^(-(n-1))]
    private func getFrInvPowers(generator: Fr, logSize: Int) -> MTLBuffer {
        let key = "inv_\(logSize)_\(generator.v.0)_\(generator.v.4)"
        if let cached = frShiftPowersCache[key] { return cached }
        let n = 1 << logSize
        let gInv = frInverse(generator)
        var powers = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            powers[i] = frMul(powers[i - 1], gInv)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Fr>.stride,
                                     options: .storageModeShared)!
        frShiftPowersCache[key] = buf
        return buf
    }

    private func getBbPowers(generator: Bb, logSize: Int) -> MTLBuffer {
        let key = "\(logSize)_\(generator.v)"
        if let cached = bbShiftPowersCache[key] { return cached }
        let n = 1 << logSize
        var powers = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n {
            powers[i] = bbMul(powers[i - 1], generator)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Bb>.stride,
                                     options: .storageModeShared)!
        bbShiftPowersCache[key] = buf
        return buf
    }

    private func getBbInvPowers(generator: Bb, logSize: Int) -> MTLBuffer {
        let key = "inv_\(logSize)_\(generator.v)"
        if let cached = bbShiftPowersCache[key] { return cached }
        let n = 1 << logSize
        let gInv = bbInverse(generator)
        var powers = [Bb](repeating: Bb.one, count: n)
        for i in 1..<n {
            powers[i] = bbMul(powers[i - 1], gInv)
        }
        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<Bb>.stride,
                                     options: .storageModeShared)!
        bbShiftPowersCache[key] = buf
        return buf
    }

    // MARK: - Coset Shift (BN254)

    /// Multiply each element by g^i in-place on GPU.
    /// Returns a new buffer with the shifted evaluations.
    public func cosetShift(evals: [Fr], logSize: Int, generator: Fr) throws -> [Fr] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        // CPU fallback for small inputs
        if n <= CosetDomainEngine.cpuThresholdBn254 {
            return cpuCosetShiftFr(evals: evals, generator: generator)
        }

        let powers = getFrPowers(generator: generator, logSize: logSize)
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                         options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetShiftBn254Fn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(powers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetShiftBn254Fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset Unshift (BN254)

    /// Multiply each element by g^(-i) in-place on GPU.
    public func cosetUnshift(evals: [Fr], logSize: Int, generator: Fr) throws -> [Fr] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        if n <= CosetDomainEngine.cpuThresholdBn254 {
            return cpuCosetUnshiftFr(evals: evals, generator: generator)
        }

        let invPowers = getFrInvPowers(generator: generator, logSize: logSize)
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                         options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetUnshiftBn254Fn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invPowers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetUnshiftBn254Fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset Shift / Unshift (BabyBear)

    public func cosetShift(evals: [Bb], logSize: Int, generator: Bb) throws -> [Bb] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        if n <= CosetDomainEngine.cpuThresholdBabyBear {
            return cpuCosetShiftBb(evals: evals, generator: generator)
        }

        let powers = getBbPowers(generator: generator, logSize: logSize)
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                         options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetShiftBabyBearFn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(powers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetShiftBabyBearFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func cosetUnshift(evals: [Bb], logSize: Int, generator: Bb) throws -> [Bb] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        if n <= CosetDomainEngine.cpuThresholdBabyBear {
            return cpuCosetUnshiftBb(evals: evals, generator: generator)
        }

        let invPowers = getBbInvPowers(generator: generator, logSize: logSize)
        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                         options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(cosetUnshiftBabyBearFn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invPowers, offset: 0, index: 1)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 2)
        let tg = min(256, Int(cosetUnshiftBabyBearFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = dataBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Vanishing polynomial evaluation

    /// Evaluate Z_H(x) = x^n - 1 at multiple points on GPU.
    /// domainSize must be a power of 2.
    public func evaluateVanishing(points: [Fr], domainSize: Int) throws -> [Fr] {
        precondition(domainSize > 0 && (domainSize & (domainSize - 1)) == 0, "domainSize must be power of 2")
        let numPoints = points.count
        if numPoints == 0 { return [] }

        let logDomain = Int(log2(Double(domainSize)))

        // CPU fallback
        if numPoints <= CosetDomainEngine.cpuThresholdBn254 {
            return cpuVanishingFr(points: points, logDomainSize: logDomain)
        }

        var pts = points
        let ptsBuf = device.makeBuffer(bytes: &pts, length: numPoints * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: numPoints * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(vanishingBn254Fn)
        enc.setBuffer(ptsBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var np = UInt32(numPoints)
        var ld = UInt32(logDomain)
        enc.setBytes(&np, length: 4, index: 2)
        enc.setBytes(&ld, length: 4, index: 3)
        let tg = min(256, Int(vanishingBn254Fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: numPoints)
        return Array(UnsafeBufferPointer(start: ptr, count: numPoints))
    }

    /// Evaluate Z_H(x) = x^n - 1 at multiple BabyBear points.
    public func evaluateVanishing(points: [Bb], domainSize: Int) throws -> [Bb] {
        precondition(domainSize > 0 && (domainSize & (domainSize - 1)) == 0, "domainSize must be power of 2")
        let numPoints = points.count
        if numPoints == 0 { return [] }

        let logDomain = Int(log2(Double(domainSize)))

        if numPoints <= CosetDomainEngine.cpuThresholdBabyBear {
            return cpuVanishingBb(points: points, logDomainSize: logDomain)
        }

        var pts = points
        let ptsBuf = device.makeBuffer(bytes: &pts, length: numPoints * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: numPoints * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(vanishingBabyBearFn)
        enc.setBuffer(ptsBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var np = UInt32(numPoints)
        var ld = UInt32(logDomain)
        enc.setBytes(&np, length: 4, index: 2)
        enc.setBytes(&ld, length: 4, index: 3)
        let tg = min(256, Int(vanishingBabyBearFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numPoints, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Bb.self, capacity: numPoints)
        return Array(UnsafeBufferPointer(start: ptr, count: numPoints))
    }

    // MARK: - Divide by vanishing polynomial

    /// Divide evaluations on coset by vanishing polynomial.
    /// For a coset {g * omega^i}, Z_H(g * omega^i) = g^n - 1 (constant).
    /// So this is just scalar multiplication by 1/(g^n - 1).
    public func divideByVanishing(evals: [Fr], logSize: Int, cosetGen: Fr) throws -> [Fr] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        // Compute g^n
        var gn = Fr.one
        for _ in 0..<logSize {
            gn = frSqr(gn)
        }
        // Actually g^n: need to compute cosetGen^n
        gn = cosetGen
        for _ in 0..<logSize {
            gn = frSqr(gn)
        }
        // zh = g^n - 1
        let zh = frSub(gn, Fr.one)
        let zhInv = frInverse(zh)

        if n <= CosetDomainEngine.cpuThresholdBn254 {
            var result = [Fr](repeating: Fr.zero, count: n)
            evals.withUnsafeBytes { eBuf in
                withUnsafeBytes(of: zhInv) { sBuf in
                    result.withUnsafeMutableBytes { rBuf in
                        bn254_fr_batch_mul_scalar(
                            eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n))
                    }
                }
            }
            return result
        }

        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Fr>.stride,
                                         options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride,
                                        options: .storageModeShared)!
        var zhInvVal = zhInv
        let zhInvBuf = device.makeBuffer(bytes: &zhInvVal, length: MemoryLayout<Fr>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(divByVanishBn254Fn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(zhInvBuf, offset: 0, index: 2)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 3)
        let tg = min(256, Int(divByVanishBn254Fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Divide evaluations on BabyBear coset by vanishing polynomial.
    public func divideByVanishing(evals: [Bb], logSize: Int, cosetGen: Bb) throws -> [Bb] {
        let n = 1 << logSize
        precondition(evals.count == n, "evals count must equal 2^logSize")

        // Compute cosetGen^n by repeated squaring
        var gn = cosetGen
        for _ in 0..<logSize {
            gn = bbSqr(gn)
        }
        let zh = bbSub(gn, Bb.one)
        let zhInv = bbInverse(zh)

        if n <= CosetDomainEngine.cpuThresholdBabyBear {
            var result = [Bb](repeating: Bb.zero, count: n)
            for i in 0..<n {
                result[i] = bbMul(evals[i], zhInv)
            }
            return result
        }

        var data = evals
        let dataBuf = device.makeBuffer(bytes: &data, length: n * MemoryLayout<Bb>.stride,
                                         options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: n * MemoryLayout<Bb>.stride,
                                        options: .storageModeShared)!
        var zhInvVal = zhInv
        let zhInvBuf = device.makeBuffer(bytes: &zhInvVal, length: MemoryLayout<Bb>.stride,
                                          options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(divByVanishBabyBearFn)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(zhInvBuf, offset: 0, index: 2)
        var size = UInt32(n)
        enc.setBytes(&size, length: 4, index: 3)
        let tg = min(256, Int(divByVanishBabyBearFn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outBuf.contents().bindMemory(to: Bb.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - Coset NTT (shift then forward NTT)

    /// Evaluate polynomial (in coefficient form) on coset {g * omega^i}.
    /// Algorithm: coset shift (multiply coeff[i] by g^i), then forward NTT.
    public func cosetNTT(coeffs: [Fr], logSize: Int, cosetGen: Fr) throws -> [Fr] {
        let shifted = try cosetShift(evals: coeffs, logSize: logSize, generator: cosetGen)
        let engine = try getFrNTTEngine()
        return try engine.ntt(shifted)
    }

    /// Coset NTT for BabyBear.
    public func cosetNTT(coeffs: [Bb], logSize: Int, cosetGen: Bb) throws -> [Bb] {
        let shifted = try cosetShift(evals: coeffs, logSize: logSize, generator: cosetGen)
        let engine = try getBbNTTEngine()
        return try engine.ntt(shifted)
    }

    // MARK: - Coset INTT (inverse NTT then unshift)

    /// Interpolate from coset evaluations back to coefficient form.
    /// Algorithm: inverse NTT, then coset unshift (multiply coeff[i] by g^(-i)).
    public func cosetINTT(evals: [Fr], logSize: Int, cosetGen: Fr) throws -> [Fr] {
        let engine = try getFrNTTEngine()
        let unNTT = try engine.intt(evals)
        return try cosetUnshift(evals: unNTT, logSize: logSize, generator: cosetGen)
    }

    /// Coset INTT for BabyBear.
    public func cosetINTT(evals: [Bb], logSize: Int, cosetGen: Bb) throws -> [Bb] {
        let engine = try getBbNTTEngine()
        let unNTT = try engine.intt(evals)
        return try cosetUnshift(evals: unNTT, logSize: logSize, generator: cosetGen)
    }

    // MARK: - CPU fallback implementations

    private func cpuCosetShiftFr(evals: [Fr], generator: Fr) -> [Fr] {
        let n = evals.count
        var result = [Fr](repeating: Fr.zero, count: n)
        evals.withUnsafeBytes { eBuf in
            withUnsafeBytes(of: generator) { gBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_powers(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    private func cpuCosetUnshiftFr(evals: [Fr], generator: Fr) -> [Fr] {
        let n = evals.count
        let gInv = frInverse(generator)
        var result = [Fr](repeating: Fr.zero, count: n)
        evals.withUnsafeBytes { eBuf in
            withUnsafeBytes(of: gInv) { gBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_powers(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return result
    }

    private func cpuCosetShiftBb(evals: [Bb], generator: Bb) -> [Bb] {
        let n = evals.count
        var result = [Bb](repeating: Bb.zero, count: n)
        var gPow = Bb.one
        for i in 0..<n {
            result[i] = bbMul(evals[i], gPow)
            gPow = bbMul(gPow, generator)
        }
        return result
    }

    private func cpuCosetUnshiftBb(evals: [Bb], generator: Bb) -> [Bb] {
        let n = evals.count
        let gInv = bbInverse(generator)
        var result = [Bb](repeating: Bb.zero, count: n)
        var gPow = Bb.one
        for i in 0..<n {
            result[i] = bbMul(evals[i], gPow)
            gPow = bbMul(gPow, gInv)
        }
        return result
    }

    private func cpuVanishingFr(points: [Fr], logDomainSize: Int) -> [Fr] {
        var result = [Fr](repeating: Fr.zero, count: points.count)
        for i in 0..<points.count {
            var x = points[i]
            for _ in 0..<logDomainSize {
                x = frSqr(x)
            }
            result[i] = frSub(x, Fr.one)
        }
        return result
    }

    private func cpuVanishingBb(points: [Bb], logDomainSize: Int) -> [Bb] {
        var result = [Bb](repeating: Bb.zero, count: points.count)
        for i in 0..<points.count {
            var x = points[i]
            for _ in 0..<logDomainSize {
                x = bbSqr(x)
            }
            result[i] = bbSub(x, Bb.one)
        }
        return result
    }
}
