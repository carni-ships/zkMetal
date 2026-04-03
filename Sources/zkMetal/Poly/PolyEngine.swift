// Polynomial Operations GPU Engine
// Supports: add, sub, hadamard product, scalar mul, NTT-based multiply, multi-point eval
import Foundation
import Metal

public class PolyEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let nttEngine: NTTEngine

    let addFunction: MTLComputePipelineState
    let subFunction: MTLComputePipelineState
    let hadamardFunction: MTLComputePipelineState
    let scalarMulFunction: MTLComputePipelineState
    let evalHornerFunction: MTLComputePipelineState
    let evalHornerChunkedFunction: MTLComputePipelineState
    let batchInverseFunction: MTLComputePipelineState
    // Subproduct tree kernels
    let treeBuildLinearPairsFunction: MTLComputePipelineState?
    let treeBuildSchoolbookFunction: MTLComputePipelineState?
    let treeRemainderSchoolbookFunction: MTLComputePipelineState?
    let twoMinusFunction: MTLComputePipelineState?
    let tuning: TuningConfig

    // Grow-only buffer cache for SubproductTree working buffers
    // Keyed by slot name → (buffer, capacity in bytes)
    private var bufferCache: [String: (MTLBuffer, Int)] = [:]

    /// Get or grow a cached buffer for the given slot. Reuses existing buffer if large enough.
    func getCachedBuffer(slot: String, minBytes: Int) -> MTLBuffer {
        if let (buf, cap) = bufferCache[slot], cap >= minBytes {
            return buf
        }
        let buf = device.makeBuffer(length: minBytes, options: .storageModeShared)!
        bufferCache[slot] = (buf, minBytes)
        return buf
    }

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

        let library = try PolyEngine.compileShaders(device: device)
        guard let addFn = library.makeFunction(name: "poly_add"),
              let subFn = library.makeFunction(name: "poly_sub"),
              let hadamardFn = library.makeFunction(name: "poly_hadamard"),
              let scalarMulFn = library.makeFunction(name: "poly_scalar_mul"),
              let evalHornerFn = library.makeFunction(name: "poly_eval_horner"),
              let evalHornerChunkedFn = library.makeFunction(name: "poly_eval_horner_chunked"),
              let batchInvFn = library.makeFunction(name: "batch_inverse") else {
            throw MSMError.missingKernel
        }
        self.addFunction = try device.makeComputePipelineState(function: addFn)
        self.subFunction = try device.makeComputePipelineState(function: subFn)
        self.hadamardFunction = try device.makeComputePipelineState(function: hadamardFn)
        self.scalarMulFunction = try device.makeComputePipelineState(function: scalarMulFn)
        self.evalHornerFunction = try device.makeComputePipelineState(function: evalHornerFn)
        self.evalHornerChunkedFunction = try device.makeComputePipelineState(function: evalHornerChunkedFn)
        self.batchInverseFunction = try device.makeComputePipelineState(function: batchInvFn)

        // Load tree kernels (optional — may not exist in all builds)
        let treeLib = try? PolyEngine.compileTreeShaders(device: device)
        if let treeLib = treeLib {
            let lpFn = treeLib.makeFunction(name: "tree_build_linear_pairs")
            let sbFn = treeLib.makeFunction(name: "tree_build_schoolbook")
            let rsFn = treeLib.makeFunction(name: "poly_remainder_schoolbook")
            self.treeBuildLinearPairsFunction = lpFn != nil ? try device.makeComputePipelineState(function: lpFn!) : nil
            self.treeBuildSchoolbookFunction = sbFn != nil ? try device.makeComputePipelineState(function: sbFn!) : nil
            self.treeRemainderSchoolbookFunction = rsFn != nil ? try device.makeComputePipelineState(function: rsFn!) : nil
            let tmFn = treeLib.makeFunction(name: "poly_two_minus")
            self.twoMinusFunction = tmFn != nil ? try device.makeComputePipelineState(function: tmFn!) : nil
        } else {
            self.treeBuildLinearPairsFunction = nil
            self.treeBuildSchoolbookFunction = nil
            self.treeRemainderSchoolbookFunction = nil
            self.twoMinusFunction = nil
        }
        self.tuning = TuningManager.shared.config(device: device)
    }

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

    private static func compileTreeShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let treeSource = try String(contentsOfFile: shaderDir + "/poly/poly_tree.metal", encoding: .utf8)

        let cleanTree = treeSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: frClean + "\n" + cleanTree, options: options)
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        for path in ["\(execDir)/../Sources/Shaders", "./Sources/Shaders"] {
            if FileManager.default.fileExists(atPath: "\(path)/fields/bn254_fr.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    func createBuffer(_ data: [Fr]) -> MTLBuffer {
        let bytes = data.count * MemoryLayout<Fr>.stride
        let buf = device.makeBuffer(length: bytes, options: .storageModeShared)!
        data.withUnsafeBytes { src in memcpy(buf.contents(), src.baseAddress!, bytes) }
        return buf
    }

    func readBuffer(_ buf: MTLBuffer, count: Int) -> [Fr] {
        let ptr = buf.contents().bindMemory(to: Fr.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func dispatchEW(_ function: MTLComputePipelineState,
                            _ buf0: MTLBuffer, _ buf1: MTLBuffer, _ buf2: MTLBuffer, n: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(function)
        enc.setBuffer(buf0, offset: 0, index: 0)
        enc.setBuffer(buf1, offset: 0, index: 1)
        enc.setBuffer(buf2, offset: 0, index: 2)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    // MARK: - Element-wise operations

    /// c = a + b (element-wise)
    public func add(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        let aBuf = createBuffer(a), bBuf = createBuffer(b)
        let cBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        try dispatchEW(addFunction, aBuf, bBuf, cBuf, n: n)
        return readBuffer(cBuf, count: n)
    }

    /// c = a - b (element-wise)
    public func sub(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        let aBuf = createBuffer(a), bBuf = createBuffer(b)
        let cBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        try dispatchEW(subFunction, aBuf, bBuf, cBuf, n: n)
        return readBuffer(cBuf, count: n)
    }

    /// c = a * b (element-wise / Hadamard product)
    public func hadamard(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        precondition(a.count == b.count)
        let n = a.count
        let aBuf = createBuffer(a), bBuf = createBuffer(b)
        let cBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        try dispatchEW(hadamardFunction, aBuf, bBuf, cBuf, n: n)
        return readBuffer(cBuf, count: n)
    }

    /// b = a * scalar
    public func scalarMul(_ a: [Fr], _ scalar: Fr) throws -> [Fr] {
        let n = a.count
        let aBuf = createBuffer(a)
        let bBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let sBuf = createBuffer([scalar])
        try dispatchEW(scalarMulFunction, aBuf, bBuf, sBuf, n: n)
        return readBuffer(bBuf, count: n)
    }

    // MARK: - Batch inversion

    /// Compute element-wise inverse: out[i] = a[i]^(-1) mod r
    /// Uses Montgomery's trick on GPU (1 Fermat inverse per 512-element chunk).
    public func batchInverse(_ a: [Fr]) throws -> [Fr] {
        let n = a.count
        let aBuf = createBuffer(a)
        let outBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchInverseFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 2)
        // One threadgroup per 512-element chunk, only thread 0 does work
        let chunkSize = 512
        let numGroups = (n + chunkSize - 1) / chunkSize
        let tg = min(64, Int(batchInverseFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
        return readBuffer(outBuf, count: n)
    }

    // MARK: - Polynomial multiplication via NTT

    /// Multiply two polynomials using NTT: c = a * b
    /// Single command buffer: NTT(a) + NTT(b) + hadamard + iNTT → one commit/wait.
    public func multiply(_ a: [Fr], _ b: [Fr]) throws -> [Fr] {
        let resultLen = a.count + b.count - 1
        var n = 1
        while n < resultLen { n <<= 1 }
        let logN = Int(log2(Double(n)))

        // Pad and copy to GPU buffers once
        let aPad = a + [Fr](repeating: Fr.zero, count: n - a.count)
        let bPad = b + [Fr](repeating: Fr.zero, count: n - b.count)

        let aBuf = createBuffer(aPad)
        let bBuf = createBuffer(bPad)

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Encode NTT(a) and NTT(b) into same command buffer
        nttEngine.encodeNTT(data: aBuf, logN: logN, cmdBuf: cmdBuf)
        nttEngine.encodeNTT(data: bBuf, logN: logN, cmdBuf: cmdBuf)

        // Encode hadamard: aBuf = aBuf * bBuf
        let encH = cmdBuf.makeComputeCommandEncoder()!
        encH.setComputePipelineState(hadamardFunction)
        encH.setBuffer(aBuf, offset: 0, index: 0)
        encH.setBuffer(bBuf, offset: 0, index: 1)
        encH.setBuffer(aBuf, offset: 0, index: 2)
        var nVal = UInt32(n)
        encH.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(hadamardFunction.maxTotalThreadsPerThreadgroup))
        encH.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        encH.endEncoding()

        // Encode iNTT
        nttEngine.encodeINTT(data: aBuf, logN: logN, cmdBuf: cmdBuf)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        // Single copy back
        return Array(readBuffer(aBuf, count: n)[0..<resultLen])
    }

    // MARK: - Multi-point evaluation

    /// Evaluate polynomial at multiple points using GPU Horner's method.
    /// Uses chunked Horner (16 threads/point) for large polynomials.
    public func evaluate(_ coeffs: [Fr], at points: [Fr]) throws -> [Fr] {
        let n = points.count
        let coeffsBuf = createBuffer(coeffs)
        let pointsBuf = createBuffer(points)
        let resultsBuf = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        var degree = UInt32(coeffs.count)
        var numPoints = UInt32(n)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        let polyChunks = 16
        // Use chunked kernel when degree is large enough and divisible by chunk count
        if coeffs.count >= 256 && coeffs.count % polyChunks == 0 {
            enc.setComputePipelineState(evalHornerChunkedFunction)
            enc.setBuffer(coeffsBuf, offset: 0, index: 0)
            enc.setBuffer(pointsBuf, offset: 0, index: 1)
            enc.setBuffer(resultsBuf, offset: 0, index: 2)
            enc.setBytes(&degree, length: 4, index: 3)
            enc.setBytes(&numPoints, length: 4, index: 4)
            let totalThreads = n * polyChunks
            // Threadgroup size must be multiple of polyChunks
            let tg = min(tuning.nttThreadgroupSize, Int(evalHornerChunkedFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(evalHornerFunction)
            enc.setBuffer(coeffsBuf, offset: 0, index: 0)
            enc.setBuffer(pointsBuf, offset: 0, index: 1)
            enc.setBuffer(resultsBuf, offset: 0, index: 2)
            enc.setBytes(&degree, length: 4, index: 3)
            enc.setBytes(&numPoints, length: 4, index: 4)
            let tg = min(tuning.nttThreadgroupSize, Int(evalHornerFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return readBuffer(resultsBuf, count: n)
    }
}
