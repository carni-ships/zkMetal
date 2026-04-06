// GPU-accelerated unified polynomial arithmetic engine
//
// Operations on BN254 Fr polynomials using Metal compute shaders:
//   - add(a, b, n)          -- elementwise addition (MTLBuffer API)
//   - sub(a, b, n)          -- elementwise subtraction
//   - mul(a, b, na, nb)     -- NTT-based polynomial multiplication
//   - scale(a, scalar, n)   -- scalar multiplication
//   - evaluate(a, point, n) -- Horner evaluation at a single point
//
// All buffer-level operations avoid Swift array copies on the hot path.
// NTT-based multiplication is fused into a single command buffer:
//   NTT(a) -> NTT(b) -> pointwise mul -> INTT -> result.
//
// CPU fallback is used for small inputs where GPU dispatch overhead dominates.

import Foundation
import Metal

public class GPUPolyArithEngine {
    public static let version = Versions.gpuPolyArith

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let nttEngine: NTTEngine

    private let addPipeline: MTLComputePipelineState
    private let subPipeline: MTLComputePipelineState
    private let hadamardPipeline: MTLComputePipelineState
    private let scalarMulPipeline: MTLComputePipelineState
    private let hornerPipeline: MTLComputePipelineState

    private let tuning: TuningConfig

    /// Below this element count, CPU path is used for elementwise ops.
    public var cpuThresholdEW: Int = 256
    /// Below this total output degree, CPU path is used for multiplication.
    public var cpuThresholdMul: Int = 64
    /// Below this degree, CPU Horner is used for evaluation.
    public var cpuThresholdEval: Int = 64

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
        self.nttEngine = try NTTEngine()

        let library = try GPUPolyArithEngine.compileShaders(device: device)

        guard let addFn = library.makeFunction(name: "poly_add"),
              let subFn = library.makeFunction(name: "poly_sub"),
              let hadFn = library.makeFunction(name: "poly_hadamard"),
              let smFn  = library.makeFunction(name: "poly_scalar_mul"),
              let heFn  = library.makeFunction(name: "poly_eval_horner") else {
            throw MSMError.missingKernel
        }

        self.addPipeline = try device.makeComputePipelineState(function: addFn)
        self.subPipeline = try device.makeComputePipelineState(function: subFn)
        self.hadamardPipeline = try device.makeComputePipelineState(function: hadFn)
        self.scalarMulPipeline = try device.makeComputePipelineState(function: smFn)
        self.hornerPipeline = try device.makeComputePipelineState(function: heFn)

        self.tuning = TuningManager.shared.config(device: device)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let polySource = try String(contentsOfFile: shaderDir + "/poly/poly_ops.metal", encoding: .utf8)

        let cleanPoly = polySource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: frClean + "\n" + cleanPoly, options: options)
    }

    // MARK: - Buffer helpers

    /// Create a shared MTLBuffer from a Swift Fr array.
    public func createBuffer(_ data: [Fr]) -> MTLBuffer {
        let bytes = data.count * MemoryLayout<Fr>.stride
        let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
        data.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, bytes) }
        return buf
    }

    /// Read Fr values out of an MTLBuffer.
    public func readBuffer(_ buf: MTLBuffer, count: Int) -> [Fr] {
        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Allocate an output buffer of n Fr elements.
    private func outputBuffer(n: Int) -> MTLBuffer {
        device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
    }

    // MARK: - Elementwise addition: c[i] = a[i] + b[i]

    /// GPU elementwise addition. Returns a new buffer with n elements.
    /// Both input buffers must contain at least n Fr elements.
    public func add(a: MTLBuffer, b: MTLBuffer, n: Int) throws -> MTLBuffer {
        if n <= cpuThresholdEW {
            return cpuAdd(a: a, b: b, n: n)
        }
        let out = outputBuffer(n: n)
        try dispatchEW(addPipeline, a, b, out, n: n)
        return out
    }

    /// Convenience: array-level add.
    public func add(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        let result = try add(a: createBuffer(a), b: createBuffer(b), n: n)
        return readBuffer(result, count: n)
    }

    private func cpuAdd(a: MTLBuffer, b: MTLBuffer, n: Int) -> MTLBuffer {
        let out = outputBuffer(n: n)
        let ap = a.contents().bindMemory(to: Fr.self, capacity: n)
        let bp = b.contents().bindMemory(to: Fr.self, capacity: n)
        let cp = out.contents().bindMemory(to: Fr.self, capacity: n)
        for i in 0..<n {
            cp[i] = frAdd(ap[i], bp[i])
        }
        return out
    }

    // MARK: - Elementwise subtraction: c[i] = a[i] - b[i]

    /// GPU elementwise subtraction. Returns a new buffer with n elements.
    public func sub(a: MTLBuffer, b: MTLBuffer, n: Int) throws -> MTLBuffer {
        if n <= cpuThresholdEW {
            return cpuSub(a: a, b: b, n: n)
        }
        let out = outputBuffer(n: n)
        try dispatchEW(subPipeline, a, b, out, n: n)
        return out
    }

    /// Convenience: array-level sub.
    public func sub(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        let result = try sub(a: createBuffer(a), b: createBuffer(b), n: n)
        return readBuffer(result, count: n)
    }

    private func cpuSub(a: MTLBuffer, b: MTLBuffer, n: Int) -> MTLBuffer {
        let out = outputBuffer(n: n)
        let ap = a.contents().bindMemory(to: Fr.self, capacity: n)
        let bp = b.contents().bindMemory(to: Fr.self, capacity: n)
        let cp = out.contents().bindMemory(to: Fr.self, capacity: n)
        for i in 0..<n {
            cp[i] = frSub(ap[i], bp[i])
        }
        return out
    }

    // MARK: - Scalar multiplication: out[i] = a[i] * scalar

    /// GPU scalar multiplication. Returns a new buffer with n elements.
    public func scale(a: MTLBuffer, scalar: Fr, n: Int) throws -> MTLBuffer {
        if n <= cpuThresholdEW {
            return cpuScale(a: a, scalar: scalar, n: n)
        }
        let out = outputBuffer(n: n)
        let sBuf = createBuffer([scalar])
        try dispatchEW(scalarMulPipeline, a, out, sBuf, n: n)
        return out
    }

    /// Convenience: array-level scale.
    public func scale(_ a: [Fr], _ scalar: Fr) throws -> [Fr] {
        let n = a.count
        let result = try scale(a: createBuffer(a), scalar: scalar, n: n)
        return readBuffer(result, count: n)
    }

    private func cpuScale(a: MTLBuffer, scalar: Fr, n: Int) -> MTLBuffer {
        let out = outputBuffer(n: n)
        let ap = a.contents().bindMemory(to: Fr.self, capacity: n)
        let cp = out.contents().bindMemory(to: Fr.self, capacity: n)
        for i in 0..<n {
            cp[i] = frMul(ap[i], scalar)
        }
        return out
    }

    // MARK: - NTT-based polynomial multiplication

    /// Multiply two polynomials via NTT. Returns a buffer with (na + nb - 1) coefficients.
    /// a has na coefficients, b has nb coefficients (ascending degree order).
    /// Single command buffer: NTT(a) + NTT(b) + pointwise mul + INTT.
    public func mul(a: MTLBuffer, b: MTLBuffer, na: Int, nb: Int) throws -> MTLBuffer {
        let resultLen = na + nb - 1

        if resultLen <= cpuThresholdMul {
            return cpuMul(a: a, b: b, na: na, nb: nb)
        }

        var n = 1
        while n < resultLen { n <<= 1 }
        let logN = Int(log2(Double(n)))

        let stride = MemoryLayout<Fr>.stride
        let bufBytes = n * stride

        // Create zero-padded GPU buffers
        let aBuf = device.makeBuffer(length: bufBytes, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: bufBytes, options: .storageModeShared)!
        memset(aBuf.contents(), 0, bufBytes)
        memset(bBuf.contents(), 0, bufBytes)
        memcpy(aBuf.contents(), a.contents(), na * stride)
        memcpy(bBuf.contents(), b.contents(), nb * stride)

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // NTT(a), NTT(b)
        nttEngine.encodeNTT(data: aBuf, logN: logN, cmdBuf: cmdBuf)
        nttEngine.encodeNTT(data: bBuf, logN: logN, cmdBuf: cmdBuf)

        // Pointwise multiply: aBuf = aBuf * bBuf
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hadamardPipeline)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(aBuf, offset: 0, index: 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(hadamardPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        // INTT(aBuf)
        nttEngine.encodeINTT(data: aBuf, logN: logN, cmdBuf: cmdBuf)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        // Return only the valid result range
        let outBuf = outputBuffer(n: resultLen)
        memcpy(outBuf.contents(), aBuf.contents(), resultLen * stride)
        return outBuf
    }

    /// Convenience: array-level multiplication.
    public func mul(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        let na = a.count, nb = b.count
        let result = try mul(a: createBuffer(a), b: createBuffer(b), na: na, nb: nb)
        return readBuffer(result, count: na + nb - 1)
    }

    private func cpuMul(a: MTLBuffer, b: MTLBuffer, na: Int, nb: Int) -> MTLBuffer {
        let resultLen = na + nb - 1
        let out = outputBuffer(n: resultLen)
        let ap = a.contents().bindMemory(to: Fr.self, capacity: na)
        let bp = b.contents().bindMemory(to: Fr.self, capacity: nb)
        let cp = out.contents().bindMemory(to: Fr.self, capacity: resultLen)

        // Zero-initialize output
        memset(out.contents(), 0, resultLen * MemoryLayout<Fr>.stride)

        // Schoolbook multiplication
        for i in 0..<na {
            for j in 0..<nb {
                cp[i + j] = frAdd(cp[i + j], frMul(ap[i], bp[j]))
            }
        }
        return out
    }

    // MARK: - Horner evaluation at a single point

    /// Evaluate polynomial p(x) = a[0] + a[1]*x + ... + a[n-1]*x^(n-1) at the given point.
    /// Uses GPU Horner for large polynomials, CPU for small ones.
    public func evaluate(a: MTLBuffer, point: Fr, n: Int) throws -> Fr {
        if n <= cpuThresholdEval {
            return cpuEvaluate(a: a, point: point, n: n)
        }
        return try gpuEvaluate(a: a, point: point, n: n)
    }

    /// Convenience: array-level evaluation.
    public func evaluate(_ coeffs: [Fr], at point: Fr) throws -> Fr {
        return try evaluate(a: createBuffer(coeffs), point: point, n: coeffs.count)
    }

    private func cpuEvaluate(a: MTLBuffer, point: Fr, n: Int) -> Fr {
        let ap = a.contents().bindMemory(to: Fr.self, capacity: n)
        guard n > 0 else { return Fr.zero }
        var result = ap[n - 1]
        for i in stride(from: n - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, point), ap[i])
        }
        return result
    }

    private func gpuEvaluate(a: MTLBuffer, point: Fr, n: Int) throws -> Fr {
        // Use the existing poly_eval_horner kernel with 1 point.
        let pointBuf = createBuffer([point])
        let resultBuf = outputBuffer(n: 1)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hornerPipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(pointBuf, offset: 0, index: 1)
        enc.setBuffer(resultBuf, offset: 0, index: 2)
        var degree = UInt32(n)
        var numPoints: UInt32 = 1
        enc.setBytes(&degree, length: 4, index: 3)
        enc.setBytes(&numPoints, length: 4, index: 4)
        let tg = min(tuning.nttThreadgroupSize, Int(hornerPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return readBuffer(resultBuf, count: 1)[0]
    }

    // MARK: - GPU dispatch helper

    private func dispatchEW(_ pipeline: MTLComputePipelineState,
                            _ buf0: MTLBuffer, _ buf1: MTLBuffer, _ buf2: MTLBuffer,
                            n: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buf0, offset: 0, index: 0)
        enc.setBuffer(buf1, offset: 0, index: 1)
        enc.setBuffer(buf2, offset: 0, index: 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }
}
