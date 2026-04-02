// FRI Engine — GPU-accelerated Fast Reed-Solomon IOP
// Core primitive for STARK proof systems (Plonky2, Plonky3, etc.)
// Supports iterative polynomial folding with random challenges.

import Foundation
import Metal

public class FRIEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let foldFunction: MTLComputePipelineState
    let foldFused2Function: MTLComputePipelineState
    let foldFused4Function: MTLComputePipelineState
    let queryExtractFunction: MTLComputePipelineState
    let cosetShiftFunction: MTLComputePipelineState
    let cosetUnshiftFunction: MTLComputePipelineState

    // Reuse NTT engine for LDE operations
    public let nttEngine: NTTEngine

    private var invTwiddleCache: [Int: MTLBuffer] = [:]

    // Cached ping-pong buffers for multiFold to avoid per-round allocations
    private var foldBufA: MTLBuffer?
    private var foldBufB: MTLBuffer?
    private var foldBufSize: Int = 0

    // Cached buffers for single-fold to avoid per-call allocation
    private var singleFoldInputBuf: MTLBuffer?
    private var singleFoldOutputBuf: MTLBuffer?
    private var singleFoldBufElements: Int = 0

    // Cached input buffer for multiFold to avoid per-call allocation
    private var multiFoldInputBuf: MTLBuffer?
    private var multiFoldInputBufElements: Int = 0
    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try FRIEngine.compileShaders(device: device)

        guard let foldFn = library.makeFunction(name: "fri_fold"),
              let foldFused2Fn = library.makeFunction(name: "fri_fold_fused2"),
              let foldFused4Fn = library.makeFunction(name: "fri_fold_fused4"),
              let queryFn = library.makeFunction(name: "fri_query_extract"),
              let shiftFn = library.makeFunction(name: "fri_coset_shift"),
              let unshiftFn = library.makeFunction(name: "fri_coset_unshift") else {
            throw MSMError.missingKernel
        }

        self.foldFunction = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Function = try device.makeComputePipelineState(function: foldFused2Fn)
        self.foldFused4Function = try device.makeComputePipelineState(function: foldFused4Fn)
        self.queryExtractFunction = try device.makeComputePipelineState(function: queryFn)
        self.cosetShiftFunction = try device.makeComputePipelineState(function: shiftFn)
        self.cosetUnshiftFunction = try device.makeComputePipelineState(function: unshiftFn)

        self.nttEngine = try NTTEngine()
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let friSource = try String(contentsOfFile: shaderDir + "/fri/fri_kernels.metal", encoding: .utf8)

        let cleanFRI = friSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanFr + "\n" + cleanFRI
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

    // MARK: - FRI Fold

    /// Perform one FRI folding step: reduce domain size by 2x.
    /// Input: evaluations on domain of size n
    /// Output: folded evaluations of size n/2
    public func fold(evals: MTLBuffer, folded: MTLBuffer, beta: Fr, logN: Int) throws {
        let n = UInt32(1 << logN)
        let half = Int(n) / 2
        let invTwiddles = getInvTwiddles(logN: logN)

        guard let betaBuf = createFrBuffer([beta]),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldFunction)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(folded, offset: 0, index: 1)
        enc.setBuffer(invTwiddles, offset: 0, index: 2)
        enc.setBuffer(betaBuf, offset: 0, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// High-level FRI fold: array in, array out.
    /// Uses cached buffers to avoid per-call Metal allocation overhead.
    public func fold(evals: [Fr], beta: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Domain size must be power of 2")
        let logN = Int(log2(Double(n)))
        let half = n / 2
        let stride = MemoryLayout<Fr>.stride

        // Reuse cached buffers when possible
        if n > singleFoldBufElements {
            guard let inBuf = device.makeBuffer(length: n * stride, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create buffers")
            }
            singleFoldInputBuf = inBuf
            singleFoldOutputBuf = outBuf
            singleFoldBufElements = n
        }

        let evalsBuf = singleFoldInputBuf!
        let foldedBuf = singleFoldOutputBuf!
        evals.withUnsafeBytes { src in
            memcpy(evalsBuf.contents(), src.baseAddress!, n * stride)
        }

        try fold(evals: evalsBuf, folded: foldedBuf, beta: beta, logN: logN)

        let ptr = foldedBuf.contents().bindMemory(to: Fr.self, capacity: half)
        return Array(UnsafeBufferPointer(start: ptr, count: half))
    }

    /// Ensure ping-pong fold buffers are large enough for n/4 Fr elements.
    private func ensureFoldBuffers(maxElements: Int) throws {
        let byteCount = maxElements * MemoryLayout<Fr>.stride
        if foldBufSize >= maxElements { return }
        guard let a = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let b = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create fold ping-pong buffers")
        }
        foldBufA = a
        foldBufB = b
        foldBufSize = maxElements
    }

    /// Multi-round FRI: fold repeatedly with a sequence of challenges.
    /// Uses fused 2-round kernel where possible, halving dispatch count and
    /// eliminating intermediate buffers. Single command buffer throughout.
    /// Uses cached ping-pong buffers to avoid per-round allocations.
    public func multiFold(evals: [Fr], betas: [Fr]) throws -> [Fr] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // Pre-allocate ping-pong buffers (largest output is n/16 for fused4, but input may be n/4)
        try ensureFoldBuffers(maxElements: max(n / 4, n / 16))

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Reuse cached input buffer when possible
        let stride = MemoryLayout<Fr>.stride
        if n > multiFoldInputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create multiFold input buffer")
            }
            multiFoldInputBuf = buf
            multiFoldInputBufElements = n
        }
        evals.withUnsafeBytes { src in
            memcpy(multiFoldInputBuf!.contents(), src.baseAddress!, n * stride)
        }
        var currentBuf = multiFoldInputBuf!
        var useA = true  // toggle between foldBufA and foldBufB

        let enc = cmdBuf.makeComputeCommandEncoder()!
        var i = 0
        while i < betas.count {
            let curN = 1 << (logN - i)
            let curLogN = logN - i
            let outputBuf = useA ? foldBufA! : foldBufB!

            if i + 3 < betas.count && curN >= 16 {
                // Fused 4-round fold
                let sixteenthN = curN / 16
                let invTwiddles = getInvTwiddles(logN: curLogN)

                var betaArr = (betas[i], betas[i+1], betas[i+2], betas[i+3])
                var nVal = UInt32(curN)
                enc.setComputePipelineState(foldFused4Function)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(invTwiddles, offset: 0, index: 2)
                enc.setBytes(&betaArr, length: 4 * MemoryLayout<Fr>.stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
                let tg = min(tuning.friThreadgroupSize, Int(foldFused4Function.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: sixteenthN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

                currentBuf = outputBuf
                useA = !useA
                i += 4
            } else if i + 1 < betas.count && curN >= 4 {
                // Fused 2-round fold
                let quarterN = curN / 4
                var beta0 = betas[i]
                var beta1 = betas[i + 1]

                let invTwiddles = getInvTwiddles(logN: curLogN)

                var nVal = UInt32(curN)
                enc.setComputePipelineState(foldFused2Function)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(invTwiddles, offset: 0, index: 2)
                enc.setBytes(&beta0, length: MemoryLayout<Fr>.stride, index: 3)
                enc.setBytes(&beta1, length: MemoryLayout<Fr>.stride, index: 4)
                enc.setBytes(&nVal, length: 4, index: 5)
                let tg = min(tuning.friThreadgroupSize, Int(foldFused2Function.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

                currentBuf = outputBuf
                useA = !useA
                i += 2
            } else {
                // Single-round fold
                let halfN = curN / 2
                var beta = betas[i]

                let invTwiddles = getInvTwiddles(logN: curLogN)

                var nVal = UInt32(curN)
                enc.setComputePipelineState(foldFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(invTwiddles, offset: 0, index: 2)
                enc.setBytes(&beta, length: MemoryLayout<Fr>.stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
                let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

                currentBuf = outputBuf
                useA = !useA
                i += 1
            }

            if i < betas.count {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let finalSize = 1 << (logN - betas.count)
        let ptr = currentBuf.contents().bindMemory(to: Fr.self, capacity: finalSize)
        return Array(UnsafeBufferPointer(start: ptr, count: finalSize))
    }

    // MARK: - LDE (Low-Degree Extension)

    /// Compute LDE: given coefficients of degree < d, evaluate on domain of size n = blowup * d.
    /// Steps: pad coefficients to n, NTT to get evaluations.
    public func lde(coeffs: [Fr], blowupFactor: Int) throws -> [Fr] {
        let d = coeffs.count
        let n = d * blowupFactor
        precondition(n > 0 && (n & (n - 1)) == 0, "LDE size must be power of 2")

        var padded = coeffs
        padded.append(contentsOf: [Fr](repeating: Fr.zero, count: n - d))

        return try nttEngine.ntt(padded)
    }

    // MARK: - CPU Reference

    /// CPU FRI fold for correctness verification.
    public static func cpuFold(evals: [Fr], beta: Fr, logN: Int) -> [Fr] {
        let n = evals.count
        let half = n / 2
        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        var folded = [Fr](repeating: Fr.zero, count: half)
        var w_inv = Fr.one

        for i in 0..<half {
            let a = evals[i]
            let b = evals[i + half]
            let sum = frAdd(a, b)
            let diff = frSub(a, b)
            let beta_w = frMul(beta, w_inv)
            let term = frMul(beta_w, diff)
            folded[i] = frAdd(sum, term)
            w_inv = frMul(w_inv, omegaInv)
        }

        return folded
    }

    // MARK: - Internal helpers

    private func getInvTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = precomputeInverseTwiddles(logN: logN)
        let buf = createFrBuffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func createFrBuffer(_ data: [Fr]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<Fr>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }
}
