// GPU-accelerated multilinear extension (MLE) engine
//
// Foundation for sumcheck-based protocols: Spartan, GKR, Lasso, HyperNova.
// Provides GPU-accelerated MLE evaluation, eq polynomial computation,
// partial evaluation (bind), and tensor product.
//
// For tables >= 1024 elements, operations run on Metal GPU; below that, CPU fallback.

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Multilinear Engine

public class GPUMultilinearEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // BN254 pipelines
    private let bindBN254: MTLComputePipelineState
    private let eqBN254: MTLComputePipelineState
    private let tensorBN254: MTLComputePipelineState

    // BabyBear pipelines
    private let bindBabyBear: MTLComputePipelineState
    private let eqBabyBear: MTLComputePipelineState

    // CPU fallback threshold
    private static let gpuThreshold = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUMultilinearEngine.compileShaders(device: device)

        guard let bindBN = library.makeFunction(name: "mle_bind_bn254"),
              let eqBN = library.makeFunction(name: "mle_eq_bn254"),
              let tensorBN = library.makeFunction(name: "mle_tensor_product_bn254"),
              let bindBB = library.makeFunction(name: "mle_bind_babybear"),
              let eqBB = library.makeFunction(name: "mle_eq_babybear") else {
            throw MSMError.missingKernel
        }

        self.bindBN254 = try device.makeComputePipelineState(function: bindBN)
        self.eqBN254 = try device.makeComputePipelineState(function: eqBN)
        self.tensorBN254 = try device.makeComputePipelineState(function: tensorBN)
        self.bindBabyBear = try device.makeComputePipelineState(function: bindBB)
        self.eqBabyBear = try device.makeComputePipelineState(function: eqBB)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        // Load and inline field headers (Metal runtime compilation requires flat source)
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let cleanBB = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

        let mleSource = try String(contentsOfFile: shaderDir + "/poly/multilinear.metal", encoding: .utf8)
        let cleanMLE = mleSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")

        let combined = cleanFr + "\n" + cleanBB + "\n" + cleanMLE
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

    // MARK: - BN254 MLE Evaluation

    /// Evaluate a multilinear polynomial at an arbitrary point using GPU sequential halving.
    /// The evaluation table has 2^logSize entries (BN254 Fr). The point has logSize coordinates.
    /// Each round binds one variable, halving the table. After logSize rounds, one value remains.
    public func evaluate(evals: MTLBuffer, logSize: Int, point: [Fr]) -> Fr {
        precondition(point.count == logSize, "Point dimension \(point.count) != logSize \(logSize)")
        if logSize == 0 {
            return evals.contents().bindMemory(to: Fr.self, capacity: 1)[0]
        }

        let n = 1 << logSize
        let stride = MemoryLayout<Fr>.stride

        // CPU fallback for small inputs
        if n < GPUMultilinearEngine.gpuThreshold {
            return evaluateCPU(evals: evals, logSize: logSize, point: point)
        }

        // GPU path: iteratively bind each variable
        var currentBuf = evals
        var currentLog = logSize
        var needsCopy = true  // don't overwrite caller's buffer on first round

        for round in 0..<logSize {
            let halfN = 1 << (currentLog - 1)
            let outBytes = halfN * stride

            guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
                  let cmdBuf = commandQueue.makeCommandBuffer() else {
                // Fall back to CPU on any GPU failure
                return evaluateCPU(evals: evals, logSize: logSize, point: point)
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bindBN254)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            var chal = point[round]
            enc.setBytes(&chal, length: stride, index: 2)
            var halfNVal = UInt32(halfN)
            enc.setBytes(&halfNVal, length: 4, index: 3)

            let tgSize = min(256, Int(bindBN254.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            currentBuf = outBuf
            currentLog -= 1
            needsCopy = false
        }

        return currentBuf.contents().bindMemory(to: Fr.self, capacity: 1)[0]
    }

    /// Evaluate from a Swift array (convenience).
    public func evaluate(evals: [Fr], point: [Fr]) -> Fr {
        let logSize = point.count
        precondition(evals.count == (1 << logSize), "Table size \(evals.count) != 2^\(logSize)")

        let stride = MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: evals.count * stride, options: .storageModeShared) else {
            return evaluateCPUArray(evals: evals, point: point)
        }
        evals.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, evals.count * stride)
        }
        return evaluate(evals: buf, logSize: logSize, point: point)
    }

    // MARK: - BN254 Eq Polynomial

    /// Compute eq(r, x) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i)) for all x in {0,1}^n.
    /// Returns an MTLBuffer with 2^n Fr evaluations.
    public func eqPoly(point: [Fr]) -> MTLBuffer? {
        let n = point.count
        let size = 1 << n
        let stride = MemoryLayout<Fr>.stride
        let outBytes = size * stride

        // CPU fallback for small inputs
        if size < GPUMultilinearEngine.gpuThreshold {
            return eqPolyCPU(point: point)
        }

        guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return eqPolyCPU(point: point)
        }

        // Upload point array
        let pointBytes = n * stride
        guard let pointBuf = device.makeBuffer(length: max(pointBytes, stride), options: .storageModeShared) else {
            return eqPolyCPU(point: point)
        }
        point.withUnsafeBytes { src in
            memcpy(pointBuf.contents(), src.baseAddress!, pointBytes)
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(eqBN254)
        enc.setBuffer(pointBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var numVars = UInt32(n)
        enc.setBytes(&numVars, length: 4, index: 2)

        let tgSize = min(256, Int(eqBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: size, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return eqPolyCPU(point: point)
        }

        return outBuf
    }

    /// Eq polynomial returning a Swift array (convenience).
    public func eqPolyArray(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        if let buf = eqPoly(point: point) {
            let ptr = buf.contents().bindMemory(to: Fr.self, capacity: size)
            return Array(UnsafeBufferPointer(start: ptr, count: size))
        }
        // Fallback: compute on CPU
        return eqPolyCPUArray(point: point)
    }

    // MARK: - BN254 Bind (Partial Evaluation)

    /// Bind one variable of an MLE at a given value. Halves the table from 2^logSize to 2^(logSize-1).
    /// The bound variable is variable 0 (MSB convention).
    public func bind(evals: MTLBuffer, logSize: Int, value: Fr) -> MTLBuffer? {
        precondition(logSize > 0, "Cannot bind on 0-variable polynomial")
        let halfN = 1 << (logSize - 1)
        let stride = MemoryLayout<Fr>.stride

        if (1 << logSize) < GPUMultilinearEngine.gpuThreshold {
            return bindCPU(evals: evals, logSize: logSize, value: value)
        }

        let outBytes = halfN * stride
        guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return bindCPU(evals: evals, logSize: logSize, value: value)
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bindBN254)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var chal = value
        enc.setBytes(&chal, length: stride, index: 2)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 3)

        let tgSize = min(256, Int(bindBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return bindCPU(evals: evals, logSize: logSize, value: value)
        }

        return outBuf
    }

    /// Bind from a Swift array, returning a Swift array (convenience).
    public func bindArray(evals: [Fr], logSize: Int, value: Fr) -> [Fr] {
        let n = 1 << logSize
        precondition(evals.count == n, "Table size \(evals.count) != 2^\(logSize)")
        let stride = MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            return bindCPUArray(evals: evals, value: value)
        }
        evals.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * stride)
        }
        if let outBuf = bind(evals: buf, logSize: logSize, value: value) {
            let halfN = 1 << (logSize - 1)
            let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: halfN)
            return Array(UnsafeBufferPointer(start: ptr, count: halfN))
        }
        return bindCPUArray(evals: evals, value: value)
    }

    // MARK: - BN254 Tensor Product

    /// Compute the tensor product of two MLE evaluation tables.
    /// If a has 2^logA entries and b has 2^logB entries, the result has 2^(logA+logB) entries.
    /// result[i * 2^logB + j] = a[i] * b[j]
    public func tensorProduct(a: MTLBuffer, logA: Int, b: MTLBuffer, logB: Int) -> MTLBuffer? {
        let sizeA = 1 << logA
        let sizeB = 1 << logB
        let totalSize = sizeA * sizeB
        let stride = MemoryLayout<Fr>.stride

        if totalSize < GPUMultilinearEngine.gpuThreshold {
            return tensorProductCPU(a: a, sizeA: sizeA, b: b, sizeB: sizeB)
        }

        let outBytes = totalSize * stride
        guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return tensorProductCPU(a: a, sizeA: sizeA, b: b, sizeB: sizeB)
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(tensorBN254)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var sA = UInt32(sizeA)
        var sB = UInt32(sizeB)
        enc.setBytes(&sA, length: 4, index: 3)
        enc.setBytes(&sB, length: 4, index: 4)

        let tgSize = min(256, Int(tensorBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalSize, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return tensorProductCPU(a: a, sizeA: sizeA, b: b, sizeB: sizeB)
        }

        return outBuf
    }

    /// Tensor product from Swift arrays (convenience).
    public func tensorProductArray(a: [Fr], logA: Int, b: [Fr], logB: Int) -> [Fr] {
        let sizeA = 1 << logA
        let sizeB = 1 << logB
        precondition(a.count == sizeA && b.count == sizeB)
        let stride = MemoryLayout<Fr>.stride

        guard let aBuf = device.makeBuffer(length: sizeA * stride, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: sizeB * stride, options: .storageModeShared) else {
            return tensorProductCPUArray(a: a, b: b)
        }
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, sizeA * stride) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, sizeB * stride) }

        if let outBuf = tensorProduct(a: aBuf, logA: logA, b: bBuf, logB: logB) {
            let totalSize = sizeA * sizeB
            let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: totalSize)
            return Array(UnsafeBufferPointer(start: ptr, count: totalSize))
        }
        return tensorProductCPUArray(a: a, b: b)
    }

    // MARK: - CPU Fallbacks (BN254)

    private func evaluateCPU(evals: MTLBuffer, logSize: Int, point: [Fr]) -> Fr {
        let n = 1 << logSize
        let ptr = evals.contents().bindMemory(to: Fr.self, capacity: n)
        var table = Array(UnsafeBufferPointer(start: ptr, count: n))
        return evaluateCPUArray(evals: table, point: point)
    }

    private func evaluateCPUArray(evals: [Fr], point: [Fr]) -> Fr {
        var table = evals
        for round in 0..<point.count {
            let halfN = table.count / 2
            let r = point[round]
            var next = [Fr](repeating: Fr.zero, count: halfN)
            for i in 0..<halfN {
                let a = table[i]
                let b = table[i + halfN]
                let diff = frSub(b, a)
                next[i] = frAdd(a, frMul(r, diff))
            }
            table = next
        }
        return table[0]
    }

    private func eqPolyCPU(point: [Fr]) -> MTLBuffer? {
        let arr = eqPolyCPUArray(point: point)
        let stride = MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: arr.count * stride, options: .storageModeShared) else {
            return nil
        }
        arr.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, arr.count * stride)
        }
        return buf
    }

    private func eqPolyCPUArray(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        let one = Fr.one
        for x in 0..<size {
            var prod = Fr.one
            for i in 0..<n {
                let bit = (x >> (n - 1 - i)) & 1
                let ri = point[i]
                let factor: Fr
                if bit == 1 {
                    factor = ri
                } else {
                    factor = frSub(one, ri)
                }
                prod = frMul(prod, factor)
            }
            eq[x] = prod
        }
        return eq
    }

    private func bindCPU(evals: MTLBuffer, logSize: Int, value: Fr) -> MTLBuffer? {
        let n = 1 << logSize
        let halfN = n / 2
        let stride = MemoryLayout<Fr>.stride
        let ptr = evals.contents().bindMemory(to: Fr.self, capacity: n)
        var result = [Fr](repeating: Fr.zero, count: halfN)
        for i in 0..<halfN {
            let a = ptr[i]
            let b = ptr[i + halfN]
            let diff = frSub(b, a)
            result[i] = frAdd(a, frMul(value, diff))
        }
        guard let outBuf = device.makeBuffer(length: halfN * stride, options: .storageModeShared) else {
            return nil
        }
        result.withUnsafeBytes { src in
            memcpy(outBuf.contents(), src.baseAddress!, halfN * stride)
        }
        return outBuf
    }

    private func bindCPUArray(evals: [Fr], value: Fr) -> [Fr] {
        let halfN = evals.count / 2
        var result = [Fr](repeating: Fr.zero, count: halfN)
        for i in 0..<halfN {
            let a = evals[i]
            let b = evals[i + halfN]
            let diff = frSub(b, a)
            result[i] = frAdd(a, frMul(value, diff))
        }
        return result
    }

    private func tensorProductCPU(a: MTLBuffer, sizeA: Int, b: MTLBuffer, sizeB: Int) -> MTLBuffer? {
        let ptrA = a.contents().bindMemory(to: Fr.self, capacity: sizeA)
        let ptrB = b.contents().bindMemory(to: Fr.self, capacity: sizeB)
        let arrA = Array(UnsafeBufferPointer(start: ptrA, count: sizeA))
        let arrB = Array(UnsafeBufferPointer(start: ptrB, count: sizeB))
        let result = tensorProductCPUArray(a: arrA, b: arrB)
        let stride = MemoryLayout<Fr>.stride
        guard let outBuf = device.makeBuffer(length: result.count * stride, options: .storageModeShared) else {
            return nil
        }
        result.withUnsafeBytes { src in
            memcpy(outBuf.contents(), src.baseAddress!, result.count * stride)
        }
        return outBuf
    }

    private func tensorProductCPUArray(a: [Fr], b: [Fr]) -> [Fr] {
        let totalSize = a.count * b.count
        var result = [Fr](repeating: Fr.zero, count: totalSize)
        for i in 0..<a.count {
            if a[i].isZero { continue }
            let base = i * b.count
            for j in 0..<b.count {
                result[base + j] = frMul(a[i], b[j])
            }
        }
        return result
    }

    // MARK: - CPU Reference (static, for tests)

    /// CPU MLE evaluation reference.
    public static func cpuEvaluate(evals: [Fr], point: [Fr]) -> Fr {
        var table = evals
        for round in 0..<point.count {
            let halfN = table.count / 2
            let r = point[round]
            var next = [Fr](repeating: Fr.zero, count: halfN)
            for i in 0..<halfN {
                let a = table[i]
                let b = table[i + halfN]
                let diff = frSub(b, a)
                next[i] = frAdd(a, frMul(r, diff))
            }
            table = next
        }
        return table[0]
    }

    /// CPU eq polynomial reference.
    public static func cpuEqPoly(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        let one = Fr.one
        for x in 0..<size {
            var prod = Fr.one
            for i in 0..<n {
                let bit = (x >> (n - 1 - i)) & 1
                let ri = point[i]
                let factor: Fr
                if bit == 1 {
                    factor = ri
                } else {
                    factor = frSub(one, ri)
                }
                prod = frMul(prod, factor)
            }
            eq[x] = prod
        }
        return eq
    }
}
