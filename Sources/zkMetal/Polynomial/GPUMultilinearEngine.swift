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
    private let partialEvalBN254: MTLComputePipelineState
    private let innerProductBN254: MTLComputePipelineState

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
              let partialEvalBN = library.makeFunction(name: "mle_partial_eval_bn254"),
              let innerProductBN = library.makeFunction(name: "mle_inner_product_bn254"),
              let bindBB = library.makeFunction(name: "mle_bind_babybear"),
              let eqBB = library.makeFunction(name: "mle_eq_babybear") else {
            throw MSMError.missingKernel
        }

        self.bindBN254 = try device.makeComputePipelineState(function: bindBN)
        self.eqBN254 = try device.makeComputePipelineState(function: eqBN)
        self.tensorBN254 = try device.makeComputePipelineState(function: tensorBN)
        self.partialEvalBN254 = try device.makeComputePipelineState(function: partialEvalBN)
        self.innerProductBN254 = try device.makeComputePipelineState(function: innerProductBN)
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

        // GPU path: single command buffer, ping-pong buffers across all rounds
        let maxHalf = 1 << (logSize - 1)
        guard let bufA = device.makeBuffer(length: maxHalf * stride, options: .storageModeShared),
              let bufB = device.makeBuffer(length: maxHalf * stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return evaluateCPU(evals: evals, logSize: logSize, point: point)
        }

        var currentBuf = evals
        var currentLog = logSize
        let tgSize = min(256, Int(bindBN254.maxTotalThreadsPerThreadgroup))

        for round in 0..<logSize {
            let halfN = 1 << (currentLog - 1)
            let outBuf = (round % 2 == 0) ? bufA : bufB

            enc.setComputePipelineState(bindBN254)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            var chal = point[round]
            enc.setBytes(&chal, length: stride, index: 2)
            var halfNVal = UInt32(halfN)
            enc.setBytes(&halfNVal, length: 4, index: 3)

            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            currentBuf = outBuf
            currentLog -= 1

            if round < logSize - 1 {
                enc.memoryBarrier(scope: .buffers)
            }
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

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

    // MARK: - BN254 Partial Evaluate (arbitrary variable)

    /// Fix an arbitrary variable in an MLE at a given value. Halves the table from 2^logSize to 2^(logSize-1).
    /// Variable index uses MSB=0 convention (variable 0 is the most significant bit).
    /// This generalizes `bind` which always fixes variable 0.
    public func partialEvaluate(evals: MTLBuffer, logSize: Int, variable: Int, value: Fr) -> MTLBuffer? {
        precondition(logSize > 0, "Cannot partial-evaluate a 0-variable polynomial")
        precondition(variable >= 0 && variable < logSize, "Variable \(variable) out of range [0, \(logSize))")

        // If fixing variable 0, delegate to the simpler bind kernel
        if variable == 0 {
            return bind(evals: evals, logSize: logSize, value: value)
        }

        let halfN = 1 << (logSize - 1)
        let stride = MemoryLayout<Fr>.stride

        if (1 << logSize) < GPUMultilinearEngine.gpuThreshold {
            return partialEvaluateCPU(evals: evals, logSize: logSize, variable: variable, value: value)
        }

        let outBytes = halfN * stride
        guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return partialEvaluateCPU(evals: evals, logSize: logSize, variable: variable, value: value)
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(partialEvalBN254)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var chal = value
        enc.setBytes(&chal, length: stride, index: 2)
        var halfNVal = UInt32(halfN)
        enc.setBytes(&halfNVal, length: 4, index: 3)
        // stride_bit = n - 1 - variable (the actual bit position)
        var strideBit = UInt32(logSize - 1 - variable)
        enc.setBytes(&strideBit, length: 4, index: 4)

        let tgSize = min(256, Int(partialEvalBN254.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return partialEvaluateCPU(evals: evals, logSize: logSize, variable: variable, value: value)
        }

        return outBuf
    }

    /// Partial evaluate from a Swift array, returning a Swift array (convenience).
    public func partialEvaluateArray(evals: [Fr], logSize: Int, variable: Int, value: Fr) -> [Fr] {
        let n = 1 << logSize
        precondition(evals.count == n, "Table size \(evals.count) != 2^\(logSize)")
        let stride = MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            return partialEvaluateCPUArray(evals: evals, logSize: logSize, variable: variable, value: value)
        }
        evals.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, n * stride)
        }
        if let outBuf = partialEvaluate(evals: buf, logSize: logSize, variable: variable, value: value) {
            let halfN = 1 << (logSize - 1)
            let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: halfN)
            return Array(UnsafeBufferPointer(start: ptr, count: halfN))
        }
        return partialEvaluateCPUArray(evals: evals, logSize: logSize, variable: variable, value: value)
    }

    // MARK: - BN254 Batch Evaluate (multiple points)

    /// Evaluate an MLE at multiple points. For k points, this computes k evaluations.
    /// Uses GPU eq polynomial + inner product for each point.
    /// More efficient than k independent `evaluate` calls when k > 1 because
    /// the eq polynomial computation is embarrassingly parallel.
    public func batchEvaluate(evals: [Fr], points: [[Fr]]) -> [Fr] {
        guard !points.isEmpty else { return [] }
        let logSize = points[0].count
        precondition(evals.count == (1 << logSize), "Table size \(evals.count) != 2^\(logSize)")
        for pt in points {
            precondition(pt.count == logSize, "All points must have dimension \(logSize)")
        }

        // For a single point, just use regular evaluate
        if points.count == 1 {
            return [evaluate(evals: evals, point: points[0])]
        }

        let n = evals.count
        let stride = MemoryLayout<Fr>.stride

        // Upload evals once
        guard let evalsBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
            return points.map { evaluateCPUArray(evals: evals, point: $0) }
        }
        evals.withUnsafeBytes { src in
            memcpy(evalsBuf.contents(), src.baseAddress!, n * stride)
        }

        // For small inputs, use CPU for everything
        if n < GPUMultilinearEngine.gpuThreshold {
            return points.map { evaluateCPUArray(evals: evals, point: $0) }
        }

        var results = [Fr](repeating: Fr.zero, count: points.count)

        for (idx, point) in points.enumerated() {
            // Compute eq polynomial on GPU
            guard let eqBuf = eqPoly(point: point) else {
                results[idx] = evaluateCPUArray(evals: evals, point: point)
                continue
            }

            // GPU inner product: <evals, eq>
            if let result = gpuInnerProduct(a: evalsBuf, b: eqBuf, count: n) {
                results[idx] = result
            } else {
                results[idx] = evaluateCPUArray(evals: evals, point: point)
            }
        }

        return results
    }

    // MARK: - BN254 Eq Polynomial (renamed convenience)

    /// Compute eq(x, r) = prod(x_i*r_i + (1-x_i)*(1-r_i)) for all x in {0,1}^n.
    /// Returns a Swift array of 2^n Fr evaluations.
    /// This is a convenience alias for `eqPolyArray`.
    public func eqPolynomial(point: [Fr]) -> [Fr] {
        return eqPolyArray(point: point)
    }

    // MARK: - GPU Inner Product (helper for batch evaluate)

    /// Compute the inner product <a, b> = sum_i a[i] * b[i] using GPU reduction.
    /// Returns nil on GPU failure.
    private func gpuInnerProduct(a: MTLBuffer, b: MTLBuffer, count: Int) -> Fr? {
        let stride = MemoryLayout<Fr>.stride
        let tgSize = min(256, Int(innerProductBN254.maxTotalThreadsPerThreadgroup))
        let numGroups = (count + tgSize - 1) / tgSize

        guard let partialBuf = device.makeBuffer(length: numGroups * stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            return nil
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(innerProductBN254)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(partialBuf, offset: 0, index: 2)
        var countVal = UInt32(count)
        enc.setBytes(&countVal, length: 4, index: 3)

        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil { return nil }

        // CPU-reduce the partial sums (typically < 1024 groups)
        let ptr = partialBuf.contents().bindMemory(to: Fr.self, capacity: numGroups)
        var sum = Fr.zero
        for i in 0..<numGroups {
            sum = frAdd(sum, ptr[i])
        }
        return sum
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
            table.withUnsafeBytes { tBuf in
                withUnsafeBytes(of: r) { rBuf in
                    next.withUnsafeMutableBytes { outBuf in
                        bn254_fr_fold_halves(
                            tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(halfN))
                    }
                }
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
        let ptr = evals.contents().bindMemory(to: UInt64.self, capacity: n * 4)
        var result = [Fr](repeating: Fr.zero, count: halfN)
        withUnsafeBytes(of: value) { vBuf in
            result.withUnsafeMutableBytes { outBuf in
                bn254_fr_fold_halves(
                    ptr,
                    vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfN))
            }
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
        evals.withUnsafeBytes { eBuf in
            withUnsafeBytes(of: value) { vBuf in
                result.withUnsafeMutableBytes { outBuf in
                    bn254_fr_fold_halves(
                        eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        vBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(halfN))
                }
            }
        }
        return result
    }

    private func partialEvaluateCPU(evals: MTLBuffer, logSize: Int, variable: Int, value: Fr) -> MTLBuffer? {
        let n = 1 << logSize
        let ptr = evals.contents().bindMemory(to: Fr.self, capacity: n)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: n))
        let result = partialEvaluateCPUArray(evals: arr, logSize: logSize, variable: variable, value: value)
        let stride = MemoryLayout<Fr>.stride
        guard let outBuf = device.makeBuffer(length: result.count * stride, options: .storageModeShared) else {
            return nil
        }
        result.withUnsafeBytes { src in
            memcpy(outBuf.contents(), src.baseAddress!, result.count * stride)
        }
        return outBuf
    }

    private func partialEvaluateCPUArray(evals: [Fr], logSize: Int, variable: Int, value: Fr) -> [Fr] {
        let n = logSize
        let halfN = 1 << (n - 1)
        let oneMinusV = frSub(Fr.one, value)

        if variable == 0 {
            // Standard bind: variable 0 (MSB)
            return bindCPUArray(evals: evals, value: value)
        }

        // General case: fix variable at position `variable`
        let strideBit = n - 1 - variable
        let blockSize = 1 << strideBit

        var result = [Fr](repeating: Fr.zero, count: halfN)
        var outIdx = 0
        let numBlocks = evals.count / (2 * blockSize)
        for block in 0..<numBlocks {
            let base = block * 2 * blockSize
            for i in 0..<blockSize {
                let lo = evals[base + i]
                let hi = evals[base + blockSize + i]
                result[outIdx] = frAdd(frMul(oneMinusV, lo), frMul(value, hi))
                outIdx += 1
            }
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
        result.withUnsafeMutableBytes { rBuf in
            let rp = rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            b.withUnsafeBytes { bBuf in
                let bp = bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                for i in 0..<a.count {
                    if a[i].isZero { continue }
                    withUnsafeBytes(of: a[i]) { aBuf in
                        bn254_fr_batch_mul_scalar(
                            bp,
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rp + i * b.count * 4,
                            Int32(b.count))
                    }
                }
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
            table.withUnsafeBytes { tBuf in
                withUnsafeBytes(of: r) { rBuf in
                    next.withUnsafeMutableBytes { outBuf in
                        bn254_fr_fold_halves(
                            tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(halfN))
                    }
                }
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
