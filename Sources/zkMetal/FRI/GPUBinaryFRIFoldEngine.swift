// GPUBinaryFRIFoldEngine — GPU-accelerated binary-native FRI folding
//
// Provides GPU-accelerated folding for binary FRI using additive domains.
// This engine wraps the Metal shaders in binary_fri_fold.metal.
//
// Key operations:
//   - Single round fold (fold-by-2)
//   - Fused fold-by-4 and fold-by-16
//   - High-arity folding (2^k elements at once)
//   - Co-curvilinearity verification
//
// All operations work over GF(2^8) with tower extension to larger fields.

import Foundation
import Metal

// MARK: - GPU Binary FRI Fold Engine

/// GPU-accelerated engine for binary FRI additive domain folding.
public final class GPUBinaryFRIFoldEngine {

    public static let version = Versions.friFold

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Kernel pipeline states
    private let foldKernel: MTLComputePipelineState
    private let foldFused2Kernel: MTLComputePipelineState
    private let foldFused4Kernel: MTLComputePipelineState
    private let foldArityKernel: MTLComputePipelineState
    private let coCurvilinearKernel: MTLComputePipelineState

    // GF(2^8) multiplication LUT (device buffer)
    private let lutBuffer: MTLBuffer

    // Ping-pong buffers for multi-round folding
    private var pingBuf: MTLBuffer?
    private var pongBuf: MTLBuffer?
    private var pingPongBytes: Int = 0

    // CPU fallback threshold
    public static let cpuFallbackThreshold = 256

    private let tuning: TuningConfig

    // MARK: - Initialization

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

        // Create GF(2^8) multiplication LUT
        let lut = GPUBinaryFRIFoldEngine.createGF8MulLUT()
        self.lutBuffer = device.makeBuffer(
            bytes: lut,
            length: 256 * 256,
            options: .storageModeShared
        )!

        let library = try GPUBinaryFRIFoldEngine.compileShaders(device: device)

        guard let foldFn = library.makeFunction(name: "binary_fri_fold_kernel"),
              let fused2Fn = library.makeFunction(name: "binary_fri_fold_fused2_kernel"),
              let fused4Fn = library.makeFunction(name: "binary_fri_fold_fused4_kernel"),
              let arityFn = library.makeFunction(name: "binary_fri_fold_arity_kernel"),
              let coCurvFn = library.makeFunction(name: "binary_fri_verify_co_curvilinear") else {
            throw MSMError.missingKernel
        }

        self.foldKernel = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Kernel = try device.makeComputePipelineState(function: fused2Fn)
        self.foldFused4Kernel = try device.makeComputePipelineState(function: fused4Fn)
        self.foldArityKernel = try device.makeComputePipelineState(function: arityFn)
        self.coCurvilinearKernel = try device.makeComputePipelineState(function: coCurvFn)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let binaryFriSource = try String(
            contentsOfFile: shaderDir + "/fri/binary_fri_fold.metal",
            encoding: .utf8
        )

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: binaryFriSource, options: options)
    }

    // MARK: - GF(2^8) LUT Generation

    /// Create the 256x256 GF(2^8) multiplication LUT.
    /// Polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
    private static func createGF8MulLUT() -> [UInt8] {
        var lut = [UInt8](repeating: 0, count: 256 * 256)

        for a in 0..<256 {
            for b in 0..<256 {
                lut[Int(a) * 256 + Int(b)] = staticGf8Mul(UInt8(a), UInt8(b))
            }
        }
        return lut
    }

    /// Static GF(2^8) multiplication with reduction by 0x11B.
    private static func staticGf8Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var a = UInt16(a)
        var b = UInt16(b)

        for _ in 0..<8 {
            if b & 1 != 0 {
                p ^= a
            }
            let hiBit = a & 0x80
            a <<= 1
            if hiBit != 0 {
                a ^= 0x1B  // 0x11B with high bit masked
            }
            b >>= 1
        }
        return UInt8(p & 0xFF)
    }

    // MARK: - Single Round Fold

    /// Perform one round of binary FRI folding on GPU.
    ///
    /// Folds evaluations at 2^k points to 2^{k-1} points using
    /// the additive domain doubling map.
    ///
    /// - Parameters:
    ///   - evals: Input evaluations (GF(2^8) elements)
    ///   - alpha: Folding challenge (GF(2^8))
    /// - Returns: Folded evaluations
    public func fold(evals: [UInt8], alpha: UInt8) throws -> [UInt8] {
        let n = evals.count
        let half = n / 2

        // CPU fallback for small inputs
        if n < GPUBinaryFRIFoldEngine.cpuFallbackThreshold {
            return try cpuFold(evals: evals, alpha: alpha)
        }

        // Upload input to GPU
        let inputBuf = device.makeBuffer(
            bytes: evals,
            length: n,
            options: .storageModeShared
        )!

        // Allocate output
        let outputBuf = device.makeBuffer(
            length: half,
            options: .storageModeShared
        )!

        // Execute kernel
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var alphaVal = alpha
        var nVal = UInt32(n)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldKernel)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBytes(&alphaVal, length: 1, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)

        let tg = min(tuning.friThreadgroupSize,
                     Int(foldKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read back result
        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: half)
        return Array(UnsafeBufferPointer(start: ptr, count: half))
    }

    // MARK: - Fused Fold Operations

    /// Fold by 4 in a single GPU dispatch (two consecutive rounds).
    public func foldBy4(evals: [UInt8], alpha0: UInt8, alpha1: UInt8) throws -> [UInt8] {
        let n = evals.count
        let quarter = n / 4

        if n < GPUBinaryFRIFoldEngine.cpuFallbackThreshold * 2 {
            let mid = try cpuFold(evals: evals, alpha: alpha0)
            return try cpuFold(evals: mid, alpha: alpha1)
        }

        let inputBuf = device.makeBuffer(bytes: evals, length: n, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: quarter, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var alpha0Val = alpha0
        var alpha1Val = alpha1
        var nVal = UInt32(n)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldFused2Kernel)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBytes(&alpha0Val, length: 1, index: 3)
        enc.setBytes(&alpha1Val, length: 1, index: 4)
        enc.setBytes(&nVal, length: 4, index: 5)

        let tg = min(tuning.friThreadgroupSize,
                     Int(foldFused2Kernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: quarter, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: quarter)
        return Array(UnsafeBufferPointer(start: ptr, count: quarter))
    }

    /// Fold by 16 in a single GPU dispatch (four consecutive rounds).
    public func foldBy16(evals: [UInt8], alphas: [UInt8]) throws -> [UInt8] {
        precondition(alphas.count >= 4, "Need 4 alphas for fold-by-16")

        let n = evals.count
        let sixteenth = n / 16

        if n < GPUBinaryFRIFoldEngine.cpuFallbackThreshold * 4 {
            var current = evals
            for i in 0..<4 {
                current = try cpuFold(evals: current, alpha: alphas[i])
            }
            return current
        }

        let inputBuf = device.makeBuffer(bytes: evals, length: n, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: sixteenth, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var alphaArray = alphas

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldFused4Kernel)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBytes(&alphaArray, length: 4, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)

        let tg = min(tuning.friThreadgroupSize,
                     Int(foldFused4Kernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: sixteenth, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: sixteenth)
        return Array(UnsafeBufferPointer(start: ptr, count: sixteenth))
    }

    // MARK: - High-Arity Fold

    /// Fold 2^arity elements at once using high-arity kernel.
    public func foldArity(evals: [UInt8], alpha: UInt8, arity: Int) throws -> [UInt8] {
        let foldFactor = 1 << arity
        let resultSize = evals.count / foldFactor

        if evals.count < GPUBinaryFRIFoldEngine.cpuFallbackThreshold {
            return try cpuFoldArity(evals: evals, alpha: alpha, arity: arity)
        }

        let inputBuf = device.makeBuffer(bytes: evals, length: evals.count, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: resultSize, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var alphaVal = alpha
        var nVal = UInt32(evals.count)
        var arityVal = UInt32(arity)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldArityKernel)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBytes(&alphaVal, length: 1, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        enc.setBytes(&arityVal, length: 4, index: 5)

        let tg = min(tuning.friThreadgroupSize,
                     Int(foldArityKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: resultSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: resultSize)
        return Array(UnsafeBufferPointer(start: ptr, count: resultSize))
    }

    // MARK: - Co-Curvilinearity Verification

    /// Verify that points lie on an affine line using GPU.
    public func verifyCoCurvilinear(points: [UInt8], numPoints: Int) throws -> Bool {
        let inputBuf = device.makeBuffer(bytes: points, length: points.count, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var numPts = UInt32(numPoints)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(coCurvilinearKernel)
        enc.setBuffer(lutBuffer, offset: 0, index: 0)
        enc.setBuffer(inputBuf, offset: 0, index: 1)
        enc.setBuffer(outputBuf, offset: 0, index: 2)
        enc.setBytes(&numPts, length: 4, index: 3)

        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
        return ptr.pointee == 1
    }

    // MARK: - CPU Fallback

    /// CPU fallback for single round fold.
    private func cpuFold(evals: [UInt8], alpha: UInt8) throws -> [UInt8] {
        let half = evals.count / 2
        var result = [UInt8](repeating: 0, count: half)

        for i in 0..<half {
            let f0 = evals[i]
            let f1 = evals[i + half]
            result[i] = gf8Add(f0, gf8MulLUT(alpha, f1))
        }

        return result
    }

    /// CPU fallback for high-arity fold.
    private func cpuFoldArity(evals: [UInt8], alpha: UInt8, arity: Int) throws -> [UInt8] {
        let foldFactor = 1 << arity
        let resultSize = evals.count / foldFactor
        var result = [UInt8](repeating: 0, count: resultSize)

        for i in 0..<resultSize {
            var acc = evals[i]
            var alphaPower = alpha

            for j in 1..<foldFactor {
                let idx = i + j * resultSize
                let term = gf8MulLUT(evals[idx], alphaPower)
                acc = gf8Add(acc, term)
                alphaPower = gf8MulLUT(alphaPower, alpha)
            }
            result[i] = acc
        }

        return result
    }

    /// GF(2^8) addition (XOR).
    private func gf8Add(_ a: UInt8, _ b: UInt8) -> UInt8 {
        return a ^ b
    }

    /// GF(2^8) multiplication via LUT.
    private func gf8MulLUT(_ a: UInt8, _ b: UInt8) -> UInt8 {
        return lutBuffer.contents().bindMemory(to: UInt8.self, capacity: 256 * 256)
            .advanced(by: Int(a) * 256)[Int(b)]
    }
}

// MARK: - GPUBinaryFRIFoldEngine Factory

extension GPUBinaryFRIFoldEngine {

    /// Create and initialize the engine, throwing on failure.
    public static func create() throws -> GPUBinaryFRIFoldEngine {
        return try GPUBinaryFRIFoldEngine()
    }
}
