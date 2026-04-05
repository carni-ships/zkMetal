// GPU-accelerated FRI query phase evaluation engine
//
// Evaluates FRI fold layers on Metal GPU for the query phase of the FRI protocol.
// After the commit phase, the verifier requests evaluation proofs at random query
// points. The prover must fold through each FRI layer at those query positions.
//
// Supports BN254 Fr (256-bit Montgomery), BabyBear (32-bit), and Mersenne-31 (32-bit).
// CPU fallback for small layers (<512 elements).

import Foundation
import Metal

// MARK: - FRI field type

/// Field types supported by the GPU FRI query engine.
public enum FRIFieldType {
    case bn254
    case babybear
    case m31
}

// MARK: - GPUFRIEngine

public class GPUFRIEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Fold kernels per field
    private let foldBn254: MTLComputePipelineState
    private let foldBabyBear: MTLComputePipelineState
    private let foldM31: MTLComputePipelineState

    // Batch query kernels per field
    private let queryBn254: MTLComputePipelineState
    private let queryBabyBear: MTLComputePipelineState
    private let queryM31: MTLComputePipelineState

    // Cached inverse twiddle buffers: [field][logN] -> MTLBuffer
    private var invTwiddleCacheBn254: [Int: MTLBuffer] = [:]
    private var invTwiddleCacheBb: [Int: MTLBuffer] = [:]
    private var invTwiddleCacheM31: [Int: MTLBuffer] = [:]

    // Cached ping-pong buffers for fullFold
    private var foldBufA: MTLBuffer?
    private var foldBufB: MTLBuffer?
    private var foldBufBytes: Int = 0

    /// CPU fallback threshold: layers smaller than this use CPU fold.
    public static let cpuFallbackThreshold = 512

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUFRIEngine.compileShaders(device: device)

        guard let fn1 = library.makeFunction(name: "fri_fold_layer_bn254"),
              let fn2 = library.makeFunction(name: "fri_fold_layer_babybear"),
              let fn3 = library.makeFunction(name: "fri_fold_layer_m31"),
              let fn4 = library.makeFunction(name: "fri_batch_query_bn254"),
              let fn5 = library.makeFunction(name: "fri_batch_query_babybear"),
              let fn6 = library.makeFunction(name: "fri_batch_query_m31") else {
            throw MSMError.missingKernel
        }

        self.foldBn254 = try device.makeComputePipelineState(function: fn1)
        self.foldBabyBear = try device.makeComputePipelineState(function: fn2)
        self.foldM31 = try device.makeComputePipelineState(function: fn3)
        self.queryBn254 = try device.makeComputePipelineState(function: fn4)
        self.queryBabyBear = try device.makeComputePipelineState(function: fn5)
        self.queryM31 = try device.makeComputePipelineState(function: fn6)
    }

    // MARK: - Shader compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let friSource = try String(contentsOfFile: shaderDir + "/fri/fri_query.metal", encoding: .utf8)
        let fieldBn254 = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)

        // Strip #include directives from FRI source (we inline dependencies)
        let cleanedFRI = friSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        // Strip include guards from field source but keep #include <metal_stdlib>
        let cleanedBn254 = fieldBn254
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanedBn254 + "\n" + cleanedFRI
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

    // MARK: - Element size helper

    private func elementSize(for field: FRIFieldType) -> Int {
        switch field {
        case .bn254:    return 32   // 8 x UInt32 (Montgomery)
        case .babybear: return 4    // 1 x UInt32
        case .m31:      return 4    // 1 x UInt32
        }
    }

    // MARK: - Pipeline selectors

    private func foldPipeline(for field: FRIFieldType) -> MTLComputePipelineState {
        switch field {
        case .bn254:    return foldBn254
        case .babybear: return foldBabyBear
        case .m31:      return foldM31
        }
    }

    private func queryPipeline(for field: FRIFieldType) -> MTLComputePipelineState {
        switch field {
        case .bn254:    return queryBn254
        case .babybear: return queryBabyBear
        case .m31:      return queryM31
        }
    }

    // MARK: - Inverse twiddle generation

    private func getInvTwiddles(logN: Int, field: FRIFieldType) -> MTLBuffer {
        switch field {
        case .bn254:
            if let cached = invTwiddleCacheBn254[logN] { return cached }
            let twiddles = precomputeInverseTwiddles(logN: logN)
            let buf = device.makeBuffer(
                bytes: twiddles, length: twiddles.count * MemoryLayout<Fr>.stride,
                options: .storageModeShared)!
            invTwiddleCacheBn254[logN] = buf
            return buf

        case .babybear:
            if let cached = invTwiddleCacheBb[logN] { return cached }
            let twiddles = precomputeInvTwiddlesBb(logN: logN)
            let buf = device.makeBuffer(
                bytes: twiddles, length: twiddles.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)!
            invTwiddleCacheBb[logN] = buf
            return buf

        case .m31:
            if let cached = invTwiddleCacheM31[logN] { return cached }
            let twiddles = precomputeInvTwiddlesM31(logN: logN)
            let buf = device.makeBuffer(
                bytes: twiddles, length: twiddles.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)!
            invTwiddleCacheM31[logN] = buf
            return buf
        }
    }

    /// Precompute BabyBear inverse twiddles: omega^{-i} for i in [0, n).
    private func precomputeInvTwiddlesBb(logN: Int) -> [UInt32] {
        let n = 1 << logN
        let omega = bbRootOfUnity(logN: logN)
        let omegaInv = bbInverse(omega)
        var twiddles = [UInt32](repeating: 1, count: n)
        var w = Bb.one
        for i in 0..<n {
            twiddles[i] = w.v
            w = bbMul(w, omegaInv)
        }
        return twiddles
    }

    /// Precompute M31 inverse twiddles.
    /// M31 does not have a multiplicative subgroup of order 2^k (only the circle group does).
    /// For standard FRI over M31 we use a subgroup of M31*: order divides p-1 = 2*(2^30 - 1).
    /// The two-adicity of M31 is only 1 (p-1 = 2 * 1073741823).
    /// So standard multiplicative FRI is limited to logN=1 for M31.
    /// For larger sizes, this is a placeholder that uses sequential powers of a generator.
    /// Real M31 FRI (Circle STARKs) uses the circle group — see CircleFRIEngine.
    /// Here we support small test cases for API completeness.
    private func precomputeInvTwiddlesM31(logN: Int) -> [UInt32] {
        let n = 1 << logN
        // For M31, p-1 = 2 * (2^30 - 1). Two-adicity is 1.
        // We can still test with a mock generator for small sizes.
        // Use generator g = 3 (primitive root of M31).
        // omega_n = g^((p-1)/n) mod p
        precondition(logN <= 30, "M31 multiplicative FRI limited to logN <= 30")
        let exp = (M31.P - 1) / UInt32(n)
        let omega = m31Pow(M31(v: 3), exp)
        let omegaInv = m31Inverse(omega)
        var twiddles = [UInt32](repeating: 1, count: n)
        var w = M31.one
        for i in 0..<n {
            twiddles[i] = w.v
            w = m31Mul(w, omegaInv)
        }
        return twiddles
    }

    // MARK: - Fold Layer (GPU)

    /// Fold one FRI layer on GPU.
    /// Input: evals buffer containing 2^logSize field elements.
    /// Output: new MTLBuffer containing 2^(logSize-1) folded elements.
    /// alpha: the random folding challenge.
    public func foldLayer(evals: MTLBuffer, logSize: Int, alpha: Fr,
                          field: FRIFieldType) throws -> MTLBuffer {
        let n = 1 << logSize
        let half = n / 2
        let elemSize = elementSize(for: field)

        // CPU fallback for small layers
        if n < GPUFRIEngine.cpuFallbackThreshold {
            return try cpuFoldLayer(evals: evals, n: n, alpha: alpha, logSize: logSize, field: field)
        }

        let invTwiddles = getInvTwiddles(logN: logSize, field: field)
        let outSize = half * elemSize
        guard let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate fold output buffer")
        }

        let pipeline = foldPipeline(for: field)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(invTwiddles, offset: 0, index: 2)

        // Set alpha — pack into field-appropriate bytes
        var nVal = UInt32(n)
        switch field {
        case .bn254:
            var alphaVal = alpha
            enc.setBytes(&alphaVal, length: MemoryLayout<Fr>.stride, index: 3)
        case .babybear:
            var alphaU32 = alpha.v.0  // low 32 bits for BabyBear
            enc.setBytes(&alphaU32, length: 4, index: 3)
        case .m31:
            var alphaU32 = alpha.v.0
            enc.setBytes(&alphaU32, length: 4, index: 3)
        }
        enc.setBytes(&nVal, length: 4, index: 4)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return outBuf
    }

    // MARK: - Batch Query (GPU)

    /// Gather evaluations at query positions in one GPU dispatch.
    /// Returns array of field elements (as Fr for bn254, or Fr with only .v.0 set for 32-bit fields).
    public func batchQuery(evals: MTLBuffer, indices: [Int],
                           field: FRIFieldType) throws -> [Fr] {
        let numQueries = indices.count
        guard numQueries > 0 else { return [] }

        let elemSize = elementSize(for: field)

        // Prepare index buffer
        var idxU32 = indices.map { UInt32($0) }
        guard let idxBuf = device.makeBuffer(bytes: &idxU32, length: numQueries * 4,
                                              options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate index buffer")
        }

        let outSize = numQueries * elemSize
        guard let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate query output buffer")
        }

        let pipeline = queryPipeline(for: field)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nq = UInt32(numQueries)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(idxBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&nq, length: 4, index: 3)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numQueries, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        // Read back results
        switch field {
        case .bn254:
            let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: numQueries)
            return Array(UnsafeBufferPointer(start: ptr, count: numQueries))
        case .babybear, .m31:
            let ptr = outBuf.contents().bindMemory(to: UInt32.self, capacity: numQueries)
            return (0..<numQueries).map { i in
                Fr(v: (ptr[i], 0, 0, 0, 0, 0, 0, 0))
            }
        }
    }

    // MARK: - Full Fold (multi-layer)

    /// Fold through all FRI layers: start with evals of size 2^logSize,
    /// apply challenges[0], challenges[1], ..., reducing size by 2x each round.
    /// Returns final MTLBuffer containing 2^(logSize - challenges.count) elements.
    public func fullFold(evals: MTLBuffer, logSize: Int, challenges: [Fr],
                         field: FRIFieldType) throws -> MTLBuffer {
        precondition(challenges.count <= logSize, "Too many challenges for domain size")
        precondition(challenges.count > 0, "Need at least one challenge")

        let elemSize = elementSize(for: field)
        var currentBuf = evals
        var currentLog = logSize

        for i in 0..<challenges.count {
            let n = 1 << currentLog

            if n < GPUFRIEngine.cpuFallbackThreshold {
                // CPU fallback for small layers
                currentBuf = try cpuFoldLayer(evals: currentBuf, n: n, alpha: challenges[i],
                                               logSize: currentLog, field: field)
            } else {
                // Allocate or reuse ping-pong buffer
                let half = n / 2
                let neededBytes = half * elemSize
                let outBuf = try ensurePingPongBuffer(bytes: neededBytes)

                let invTwiddles = getInvTwiddles(logN: currentLog, field: field)
                let pipeline = foldPipeline(for: field)

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }

                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outBuf, offset: 0, index: 1)
                enc.setBuffer(invTwiddles, offset: 0, index: 2)

                var nVal = UInt32(n)
                switch field {
                case .bn254:
                    var alphaVal = challenges[i]
                    enc.setBytes(&alphaVal, length: MemoryLayout<Fr>.stride, index: 3)
                case .babybear:
                    var alphaU32 = challenges[i].v.0
                    enc.setBytes(&alphaU32, length: 4, index: 3)
                case .m31:
                    var alphaU32 = challenges[i].v.0
                    enc.setBytes(&alphaU32, length: 4, index: 3)
                }
                enc.setBytes(&nVal, length: 4, index: 4)

                let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc.endEncoding()

                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                if let error = cmdBuf.error {
                    throw MSMError.gpuError(error.localizedDescription)
                }

                currentBuf = outBuf
            }

            currentLog -= 1
        }

        return currentBuf
    }

    // MARK: - Ping-pong buffer management

    private func ensurePingPongBuffer(bytes: Int) throws -> MTLBuffer {
        // Alternate between A and B buffers to avoid aliasing
        if foldBufBytes < bytes {
            guard let a = device.makeBuffer(length: bytes, options: .storageModeShared),
                  let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate ping-pong buffers")
            }
            foldBufA = a
            foldBufB = b
            foldBufBytes = bytes
        }
        // Swap A/B each call
        let tmp = foldBufA
        foldBufA = foldBufB
        foldBufB = tmp
        return foldBufA!
    }

    // MARK: - CPU fallback

    /// CPU fold for small layers. Reads from MTLBuffer, writes to new MTLBuffer.
    private func cpuFoldLayer(evals: MTLBuffer, n: Int, alpha: Fr,
                               logSize: Int, field: FRIFieldType) throws -> MTLBuffer {
        let half = n / 2
        let elemSize = elementSize(for: field)

        switch field {
        case .bn254:
            let ptr = evals.contents().bindMemory(to: Fr.self, capacity: n)
            let invTwiddles = precomputeInverseTwiddles(logN: logSize)
            var folded = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                let a = ptr[i]
                let b = ptr[i + half]
                let sum = frAdd(a, b)
                let diff = frSub(a, b)
                let term = frMul(frMul(alpha, invTwiddles[i]), diff)
                folded[i] = frAdd(sum, term)
            }
            guard let outBuf = device.makeBuffer(bytes: folded, length: half * elemSize,
                                                  options: .storageModeShared) else {
                throw MSMError.gpuError("CPU fold: failed to allocate output")
            }
            return outBuf

        case .babybear:
            let ptr = evals.contents().bindMemory(to: UInt32.self, capacity: n)
            let invTwiddles = precomputeInvTwiddlesBb(logN: logSize)
            let alphaVal = Bb(v: alpha.v.0)
            var folded = [UInt32](repeating: 0, count: half)
            for i in 0..<half {
                let a = Bb(v: ptr[i])
                let b = Bb(v: ptr[i + half])
                let sum = bbAdd(a, b)
                let diff = bbSub(a, b)
                let wInv = Bb(v: invTwiddles[i])
                let term = bbMul(bbMul(alphaVal, wInv), diff)
                folded[i] = bbAdd(sum, term).v
            }
            guard let outBuf = device.makeBuffer(bytes: folded, length: half * elemSize,
                                                  options: .storageModeShared) else {
                throw MSMError.gpuError("CPU fold: failed to allocate output")
            }
            return outBuf

        case .m31:
            let ptr = evals.contents().bindMemory(to: UInt32.self, capacity: n)
            let invTwiddles = precomputeInvTwiddlesM31(logN: logSize)
            let alphaVal = M31(v: alpha.v.0)
            var folded = [UInt32](repeating: 0, count: half)
            for i in 0..<half {
                let a = M31(v: ptr[i])
                let b = M31(v: ptr[i + half])
                let sum = m31Add(a, b)
                let diff = m31Sub(a, b)
                let wInv = M31(v: invTwiddles[i])
                let term = m31Mul(m31Mul(alphaVal, wInv), diff)
                folded[i] = m31Add(sum, term).v
            }
            guard let outBuf = device.makeBuffer(bytes: folded, length: half * elemSize,
                                                  options: .storageModeShared) else {
                throw MSMError.gpuError("CPU fold: failed to allocate output")
            }
            return outBuf
        }
    }
}
