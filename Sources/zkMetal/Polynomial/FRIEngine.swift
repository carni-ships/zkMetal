// FRI Engine — GPU-accelerated Fast Reed-Solomon IOP
// Core primitive for STARK proof systems (Plonky2, Plonky3, etc.)
// Supports iterative polynomial folding with random challenges.

import Foundation
import Metal
import NeonFieldOps

public class FRIEngine {
    public static let version = Versions.fri
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let foldFunction: MTLComputePipelineState
    let foldFused2Function: MTLComputePipelineState
    let foldFused4Function: MTLComputePipelineState
    let foldBy4Function: MTLComputePipelineState
    let foldBy8Function: MTLComputePipelineState
    let foldCascadeFunction: MTLComputePipelineState
    let queryExtractFunction: MTLComputePipelineState
    let cosetShiftFunction: MTLComputePipelineState
    let cosetUnshiftFunction: MTLComputePipelineState

    // Reuse NTT engine for LDE operations
    public let nttEngine: NTTEngine

    // Reuse Merkle engine for commit/query phases (avoid per-call shader compilation)
    private lazy var merkleEngine: Poseidon2MerkleEngine = {
        try! Poseidon2MerkleEngine()
    }()

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

    // Cached buffers for merkle phase (avoid per-call allocation)
    private var merkleRootsBuf: MTLBuffer?
    private var merkleRootsBufSize: Int = 0
    private var merkleSubtreeRootsBuf: MTLBuffer?
    private var merkleSubtreeRootsBufSize: Int = 0

    // Cached per-layer buffers for commitPhase (avoid per-call allocation)
    private var cachedLayerBufs: [MTLBuffer] = []
    private var cachedLayerBufsLogN: Int = 0

    // Cached per-layer buffers for commitPhase4 (fold-by-4, fewer layers)
    private var cachedLayer4Bufs: [MTLBuffer] = []
    private var cachedLayer4BufsLogN: Int = 0

    // Cached per-layer buffers for commitPhase8 (fold-by-8, even fewer layers)
    private var cachedLayer8Bufs: [MTLBuffer] = []
    private var cachedLayer8BufsLogN: Int = 0

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
              let foldBy4Fn = library.makeFunction(name: "fri_fold_by4"),
              let foldBy8Fn = library.makeFunction(name: "fri_fold_by8"),
              let foldCascadeFn = library.makeFunction(name: "fri_fold_cascade"),
              let queryFn = library.makeFunction(name: "fri_query_extract"),
              let shiftFn = library.makeFunction(name: "fri_coset_shift"),
              let unshiftFn = library.makeFunction(name: "fri_coset_unshift") else {
            throw MSMError.missingKernel
        }

        self.foldFunction = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Function = try device.makeComputePipelineState(function: foldFused2Fn)
        self.foldFused4Function = try device.makeComputePipelineState(function: foldFused4Fn)
        self.foldBy4Function = try device.makeComputePipelineState(function: foldBy4Fn)
        self.foldBy8Function = try device.makeComputePipelineState(function: foldBy8Fn)
        self.foldCascadeFunction = try device.makeComputePipelineState(function: foldCascadeFn)
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

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var betaVal = beta
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldFunction)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(folded, offset: 0, index: 1)
        enc.setBuffer(invTwiddles, offset: 0, index: 2)
        enc.setBytes(&betaVal, length: MemoryLayout<Fr>.stride, index: 3)
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
        // Cascade threshold: max elements that fit in 32KB threadgroup shared memory
        // Each Fr is 32 bytes, so 1024 elements = 32KB
        let cascadeMaxN = 1024
        while i < betas.count {
            let curN = 1 << (logN - i)
            let curLogN = logN - i
            let outputBuf = useA ? foldBufA! : foldBufB!

            // Cascade: when remaining data fits in shared memory, do ALL remaining
            // rounds in a single kernel dispatch
            let remainingRounds = betas.count - i
            if curN <= cascadeMaxN && remainingRounds >= 2 {
                let invTwiddles = getInvTwiddles(logN: curLogN)
                var nVal = UInt32(curN)
                var numRoundsVal = UInt32(remainingRounds)

                // Pack remaining betas into contiguous array
                var remainingBetas = Array(betas[i..<betas.count])
                enc.setComputePipelineState(foldCascadeFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(invTwiddles, offset: 0, index: 2)
                remainingBetas.withUnsafeMutableBytes { ptr in
                    enc.setBytes(ptr.baseAddress!, length: ptr.count, index: 3)
                }
                enc.setBytes(&nVal, length: 4, index: 4)
                enc.setBytes(&numRoundsVal, length: 4, index: 5)

                let halfN = curN / 2
                // Single threadgroup with n/2 threads
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: halfN, height: 1, depth: 1))

                currentBuf = outputBuf
                useA = !useA
                i = betas.count  // consumed all remaining rounds
            } else if i + 3 < betas.count && curN >= 16 {
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

    // MARK: - FRI Fold-by-4

    /// Perform one FRI fold-by-4 step: reduce domain size by 4x.
    /// Uses 4th-root-of-unity decomposition for the polynomial.
    /// Input: evaluations on domain of size n (must be divisible by 4)
    /// Output: folded evaluations of size n/4
    public func fold4(evals: MTLBuffer, folded: MTLBuffer, beta: Fr, logN: Int) throws {
        let n = UInt32(1 << logN)
        let quarter = Int(n) / 4
        let invTwiddles = getInvTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var betaVal = beta
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldBy4Function)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(folded, offset: 0, index: 1)
        enc.setBuffer(invTwiddles, offset: 0, index: 2)
        enc.setBytes(&betaVal, length: MemoryLayout<Fr>.stride, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        let tg = min(tuning.friThreadgroupSize, Int(foldBy4Function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: quarter, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// High-level FRI fold-by-4: array in, array out.
    public func fold4(evals: [Fr], beta: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n >= 4 && (n & (n - 1)) == 0, "Domain size must be power of 2 and >= 4")
        let logN = Int(log2(Double(n)))
        let quarter = n / 4
        let stride = MemoryLayout<Fr>.stride

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

        try fold4(evals: evalsBuf, folded: foldedBuf, beta: beta, logN: logN)

        let ptr = foldedBuf.contents().bindMemory(to: Fr.self, capacity: quarter)
        return Array(UnsafeBufferPointer(start: ptr, count: quarter))
    }

    /// Multi-round FRI using fold-by-4: fold repeatedly, each step divides by 4.
    /// If logN is not divisible by 2, does one fold-by-2 first, then fold-by-4 the rest.
    /// Uses a single command buffer with memory barriers.
    public func multiFold4(evals: [Fr], betas: [Fr]) throws -> [Fr] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // For fold-by-4 we consume 1 beta per fold-by-4 step (each step = 2 log-levels)
        // If logN is odd, first do one fold-by-2 consuming betas[0], then fold-by-4 with rest
        let oddStart = (logN % 2 != 0) ? 1 : 0
        let fold4Count = (logN - oddStart) / 2
        let totalBetas = oddStart + fold4Count
        precondition(betas.count >= totalBetas, "Need \(totalBetas) betas for fold-by-4, got \(betas.count)")

        try ensureFoldBuffers(maxElements: max(n / 4, 1))

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

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
        var useA = true
        var curLogN = logN
        var betaIdx = 0

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Handle odd logN: one fold-by-2 first
        if oddStart == 1 {
            let curN = 1 << curLogN
            let halfN = curN / 2
            let outputBuf = useA ? foldBufA! : foldBufB!
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA
            curLogN -= 1
            betaIdx += 1
            enc.memoryBarrier(scope: .buffers)
        }

        // Now curLogN is even; fold-by-4 until done
        while betaIdx < totalBetas {
            let curN = 1 << curLogN
            let quarterN = curN / 4
            let outputBuf = useA ? foldBufA! : foldBufB!
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldBy4Function)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA
            curLogN -= 2
            betaIdx += 1

            if betaIdx < totalBetas {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let finalSize = 1 << curLogN
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

        var padded = [Fr](repeating: Fr.zero, count: n)
        padded.withUnsafeMutableBytes { dst in
            coeffs.withUnsafeBytes { src in
                memcpy(dst.baseAddress!, src.baseAddress!, d * MemoryLayout<Fr>.stride)
            }
        }

        return try nttEngine.ntt(padded)
    }

    // MARK: - CPU Reference

    /// CPU FRI fold for correctness verification.
    public static func cpuFold(evals: [Fr], beta: Fr, logN: Int) -> [Fr] {
        let n = evals.count
        let half = n / 2
        let invTwiddles = precomputeInverseTwiddles(logN: logN)

        var result = evals
        result.withUnsafeMutableBytes { eBuf in
            withUnsafeBytes(of: beta) { bBuf in
                invTwiddles.withUnsafeBytes { tBuf in
                    bn254_fr_fri_fold_inplace(
                        eBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(half))
                }
            }
        }
        result.removeLast(half)
        return result
    }

    /// CPU FRI fold-by-4 for correctness verification.
    /// Mirrors the GPU kernel logic: 4th-root-of-unity decomposition.
    /// The /4 normalization factor is NOT applied (absorbed, same as fold-by-2's /2).
    public static func cpuFold4(evals: [Fr], beta: Fr, logN: Int) -> [Fr] {
        let n = evals.count
        precondition(n >= 4 && (n & (n - 1)) == 0)
        let quarter = n / 4

        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        // w4 = omega^{N/4} is the primitive 4th root of unity
        // w4_inv = omega^{-N/4}
        let w4_inv = frPow(omegaInv, UInt64(quarter))

        var folded = [Fr](repeating: Fr.zero, count: quarter)
        var inv_x = Fr.one  // omega^{-i}, starts at omega^0 = 1

        for i in 0..<quarter {
            let e0 = evals[i]
            let e1 = evals[i + quarter]
            let e2 = evals[i + 2 * quarter]
            let e3 = evals[i + 3 * quarter]

            let s02 = frAdd(e0, e2)
            let d02 = frSub(e0, e2)
            let s13 = frAdd(e1, e3)
            let d13 = frSub(e1, e3)

            let d13_w4inv = frMul(d13, w4_inv)

            var t0 = frAdd(s02, s13)
            var t1 = frAdd(d02, d13_w4inv)
            var t2 = frSub(s02, s13)
            var t3 = frSub(d02, d13_w4inv)

            let inv_x2 = frMul(inv_x, inv_x)
            let inv_x3 = frMul(inv_x2, inv_x)

            t1 = frMul(t1, inv_x)
            t2 = frMul(t2, inv_x2)
            t3 = frMul(t3, inv_x3)

            let r = beta
            let r2 = frMul(r, r)
            let r3 = frMul(r2, r)

            var result = frAdd(t0, frMul(r, t1))
            result = frAdd(result, frMul(r2, t2))
            result = frAdd(result, frMul(r3, t3))

            folded[i] = result
            inv_x = frMul(inv_x, omegaInv)
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

    /// CPU merkle root for small leaf arrays (avoids GPU dispatch overhead).
    private func cpuMerkleRoot(_ leaves: UnsafeBufferPointer<Fr>) -> Fr {
        let n = leaves.count
        if n == 1 { return leaves[0] }
        if n == 2 { return poseidon2Hash(leaves[0], leaves[1]) }
        // Build tree bottom-up
        var level = Array(leaves)
        while level.count > 1 {
            var next = [Fr]()
            next.reserveCapacity(level.count / 2)
            for i in stride(from: 0, to: level.count, by: 2) {
                next.append(poseidon2Hash(level[i], level[i + 1]))
            }
            level = next
        }
        return level[0]
    }

    /// Encode merkle root computation for a single layer directly from its GPU buffer.
    private func encodeMerkleForLayer(encoder: MTLComputeCommandEncoder,
                                       p2: Poseidon2Engine,
                                       layerBuf: MTLBuffer, layerN: Int,
                                       rootsBuf: MTLBuffer, rootOffset: Int,
                                       subtreeRootsBuf: MTLBuffer, subtreeSize: Int, stride: Int) {
        if layerN <= subtreeSize {
            // Small tree: single fused dispatch directly from layerBuf
            p2.encodeMerkleFused(encoder: encoder,
                                  leavesBuffer: layerBuf, leavesOffset: 0,
                                  rootsBuffer: rootsBuf, rootsOffset: rootOffset,
                                  numSubtrees: 1, subtreeSize: layerN)
        } else {
            // Large tree: fused subtrees then upper tree
            let numSubtrees = layerN / subtreeSize
            p2.encodeMerkleFused(encoder: encoder,
                                  leavesBuffer: layerBuf, leavesOffset: 0,
                                  rootsBuffer: subtreeRootsBuf, rootsOffset: 0,
                                  numSubtrees: numSubtrees)
            encoder.memoryBarrier(scope: .buffers)
            // Upper tree from subtree roots
            if numSubtrees <= subtreeSize && (numSubtrees & (numSubtrees - 1)) == 0 {
                p2.encodeMerkleFused(encoder: encoder,
                                      leavesBuffer: subtreeRootsBuf, leavesOffset: 0,
                                      rootsBuffer: rootsBuf, rootsOffset: rootOffset,
                                      numSubtrees: 1, subtreeSize: numSubtrees)
            } else {
                // Very large: level-by-level upper tree
                var levelSize = numSubtrees
                var readOff = 0
                var writeOff = numSubtrees * stride
                while levelSize > 1 {
                    let parentCount = levelSize / 2
                    p2.encodeHashPairs(encoder: encoder, buffer: subtreeRootsBuf,
                                       inputOffset: readOff,
                                       outputOffset: writeOff,
                                       count: parentCount)
                    if parentCount > 1 { encoder.memoryBarrier(scope: .buffers) }
                    readOff = writeOff
                    writeOff += parentCount * stride
                    levelSize = parentCount
                }
                // Final root is at subtreeRootsBuf readOff, need to copy to rootsBuf
                // Use a single hash pair dispatch to move it (wasteful but simple)
                // Actually, read it back on CPU after CB completes
            }
        }
    }

    // MARK: - FRI Proof Protocol

    /// Commit phase: fold iteratively, returning commitments (Poseidon2 Merkle roots)
    /// at each layer plus the final constant value.
    /// Profile flag: set to true to print commit phase timing breakdown
    public var profileCommit = false

    public func commitPhase(evals: [Fr], betas: [Fr]) throws -> FRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(betas.count <= logN)

        let merkle = merkleEngine
        let stride = MemoryLayout<Fr>.stride

        // Pre-compute all folds in one GPU pass using multiFold-style pipeline
        // Store intermediate layers for query phase
        var layers: [[Fr]] = [evals]
        var roots: [Fr] = []

        // Use buffer-level fold pipeline: single CB for all folds
        let maxElements = n
        try ensureFoldBuffers(maxElements: max(maxElements / 4, 1))

        // Allocate layer buffers on GPU
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

        // Allocate per-layer GPU buffers (cached across calls for same logN)
        var foldT0 = CFAbsoluteTimeGetCurrent()
        var layerSizes: [Int] = [n]
        for i in 0..<betas.count {
            layerSizes.append(n >> (i + 1))
        }

        if cachedLayerBufsLogN != logN || cachedLayerBufs.count != betas.count {
            cachedLayerBufs = []
            for i in 0..<betas.count {
                let layerN = layerSizes[i + 1]
                guard let buf = device.makeBuffer(length: layerN * stride, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to create layer buffer")
                }
                cachedLayerBufs.append(buf)
            }
            cachedLayerBufsLogN = logN
        }
        var layerBufs: [MTLBuffer] = [multiFoldInputBuf!]
        layerBufs.append(contentsOf: cachedLayerBufs)

        // Fold + Merkle: CHAINED into single command buffer.
        // Eliminates CPU-GPU round-trip between fold completion and merkle dispatch.
        // Fold outputs feed directly into merkle reads via encoder memory barriers.
        let subtreeSize = Poseidon2Engine.merkleSubtreeSize  // 1024
        let p2 = merkle.p2Engine
        let cpuMerkleThreshold = 16  // CPU merkle for <= 16 leaves

        // Cache roots buffer
        let numMerkleRoots = betas.count
        let rootsBufBytes = max(numMerkleRoots * stride, stride)
        if merkleRootsBuf == nil || merkleRootsBufSize < rootsBufBytes {
            merkleRootsBuf = device.makeBuffer(length: rootsBufBytes, options: .storageModeShared)
            merkleRootsBufSize = rootsBufBytes
        }
        guard let rootsBuf = merkleRootsBuf else {
            throw MSMError.gpuError("Failed to allocate merkle roots buffer")
        }

        // Cache subtree roots buffer
        let maxSubtreeRoots = max(layerSizes[1] / subtreeSize, 1)
        let subtreeRootBytes = maxSubtreeRoots * stride
        if merkleSubtreeRootsBuf == nil || merkleSubtreeRootsBufSize < subtreeRootBytes {
            merkleSubtreeRootsBuf = device.makeBuffer(length: subtreeRootBytes, options: .storageModeShared)
            merkleSubtreeRootsBufSize = subtreeRootBytes
        }
        guard let subtreeRootsBuf = merkleSubtreeRootsBuf else {
            throw MSMError.gpuError("Failed to allocate subtree roots buffer")
        }

        roots = [Fr](repeating: Fr.zero, count: betas.count)

        // Check if any layers need GPU merkle
        var hasGPUMerkle = false
        for i in 0..<betas.count {
            if layerSizes[i + 1] > cpuMerkleThreshold { hasGPUMerkle = true; break }
        }

        // Single CB: all folds + all GPU merkle dispatches
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: Encode all fold operations
        let foldEnc = cmdBuf.makeComputeCommandEncoder()!
        for i in 0..<betas.count {
            let curN = layerSizes[i]
            let curLogN = logN - i
            let halfN = curN / 2
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[i]
            var nVal = UInt32(curN)
            foldEnc.setComputePipelineState(foldFunction)
            foldEnc.setBuffer(layerBufs[i], offset: 0, index: 0)
            foldEnc.setBuffer(layerBufs[i + 1], offset: 0, index: 1)
            foldEnc.setBuffer(invTwiddles, offset: 0, index: 2)
            foldEnc.setBytes(&beta, length: stride, index: 3)
            foldEnc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            foldEnc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if i + 1 < betas.count { foldEnc.memoryBarrier(scope: .buffers) }
        }
        foldEnc.endEncoding()

        // Phase 2: Encode GPU merkle dispatches into SAME command buffer
        // New encoder acts as memory barrier — fold results are visible to merkle reads.
        if hasGPUMerkle {
            let mEnc = cmdBuf.makeComputeCommandEncoder()!
            var firstDispatch = true

            for i in 0..<betas.count {
                let layerN = layerSizes[i + 1]
                if layerN <= cpuMerkleThreshold { continue }
                if !firstDispatch { mEnc.memoryBarrier(scope: .buffers) }
                firstDispatch = false
                encodeMerkleForLayer(encoder: mEnc, p2: p2, layerBuf: layerBufs[i + 1], layerN: layerN,
                                     rootsBuf: rootsBuf, rootOffset: i * stride,
                                     subtreeRootsBuf: subtreeRootsBuf, subtreeSize: subtreeSize, stride: stride)
            }
            mEnc.endEncoding()
        }

        cmdBuf.commit()

        // While GPU runs fold+merkle, we can't read layer data yet (unified memory
        // is coherent only after completion). Wait first, then read.
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
        let foldTime = (CFAbsoluteTimeGetCurrent() - foldT0) * 1000
        let merkleT0 = CFAbsoluteTimeGetCurrent()

        // Read layer data from GPU buffers (zero-copy with unified memory, just pointer wrap)
        for i in 1...betas.count {
            let count = layerSizes[i]
            let ptr = layerBufs[i].contents().bindMemory(to: Fr.self, capacity: count)
            layers.append(Array(UnsafeBufferPointer(start: ptr, count: count)))
        }

        // Compute small merkle roots on CPU
        for i in 0..<betas.count {
            let layerN = layerSizes[i + 1]
            if layerN > cpuMerkleThreshold { continue }
            if layerN <= 1 {
                roots[i] = layers[i][0]
            } else {
                layers[i].withUnsafeBufferPointer { bp in
                    roots[i] = cpuMerkleRoot(bp)
                }
            }
        }

        // Read GPU merkle roots
        if hasGPUMerkle {
            let rootPtr = rootsBuf.contents().bindMemory(to: Fr.self, capacity: numMerkleRoots)
            for i in 0..<betas.count {
                if layerSizes[i + 1] > cpuMerkleThreshold {
                    roots[i] = rootPtr[i]
                }
            }
        }

        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000

        if profileCommit {
            fputs(String(format: "  commitPhase: fold %.1fms, merkle %.1fms, total %.1fms\n",
                        foldTime, merkleTime, foldTime + merkleTime), stderr)
        }

        let current = layers.last!
        let finalValue = current.count == 1 ? current[0] : current[0]

        return FRICommitment(
            layers: layers,
            roots: roots,
            betas: betas,
            finalValue: finalValue,
            foldMode: .foldBy2
        )
    }

    /// Query phase: given query indices, extract evaluation pairs and Merkle paths
    /// at each layer of the FRI commitment.
    public func queryPhase(commitment: FRICommitment, queryIndices: [UInt32]) throws -> [FRIQueryProof] {
        let merkle = merkleEngine
        let numQueries = queryIndices.count

        // Pre-build Merkle trees for all layers (avoid rebuilding per query)
        var layerTrees = [[Fr]]()
        layerTrees.reserveCapacity(commitment.layers.count - 1)
        for layer in 0..<commitment.layers.count - 1 {
            layerTrees.append(try merkle.buildTree(commitment.layers[layer]))
        }

        var proofs = [FRIQueryProof]()
        proofs.reserveCapacity(numQueries)

        for qi in 0..<numQueries {
            var layerEvals: [(Fr, Fr)] = []
            var merklePaths: [[[Fr]]] = []
            var idx = queryIndices[qi]

            for layer in 0..<commitment.layers.count - 1 {
                let evals = commitment.layers[layer]
                let n = evals.count
                let halfN = UInt32(n / 2)

                // Extract paired evaluations in canonical order (lower, upper)
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let upperIdx = lowerIdx + halfN
                let evalA = evals[Int(lowerIdx)]
                let evalB = evals[Int(upperIdx)]
                layerEvals.append((evalA, evalB))

                // Get Merkle path for this index
                let path = extractMerklePath(tree: layerTrees[layer], leafCount: n, index: Int(idx))
                merklePaths.append([path])

                // Derive next layer's index: fold maps to lower half
                idx = lowerIdx
            }

            proofs.append(FRIQueryProof(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    /// Verify a FRI proof: check consistency of query responses with commitments.
    /// Returns true if all queries pass.
    public func verify(commitment: FRICommitment, queries: [FRIQueryProof]) -> Bool {
        for query in queries {
            var idx = query.initialIndex

            for layer in 0..<commitment.layers.count - 1 {
                let (evalA, evalB) = query.layerEvals[layer]
                let n = commitment.layers[layer].count
                let halfN = UInt32(n / 2)
                let logN = Int(log2(Double(n)))
                let beta = commitment.betas[layer]

                // Verify fold consistency:
                // folded[lowerIdx] = (evalA + evalB) + beta * omega^(-lowerIdx) * (evalA - evalB)
                let omega = frRootOfUnity(logN: logN)
                let omegaInv = frInverse(omega)
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let w_inv = frPow(omegaInv, UInt64(lowerIdx))

                let sum = frAdd(evalA, evalB)
                let diff = frSub(evalA, evalB)
                let term = frMul(frMul(beta, w_inv), diff)
                let expected = frAdd(sum, term)

                // Check against next layer's evaluation
                let nextIdx = lowerIdx
                if layer + 1 < commitment.layers.count {
                    let nextEval = commitment.layers[layer + 1][Int(nextIdx)]
                    let expectedLimbs = frToInt(expected)
                    let actualLimbs = frToInt(nextEval)
                    if expectedLimbs != actualLimbs {
                        return false
                    }
                }

                idx = nextIdx
            }
        }
        return true
    }

    // MARK: - FRI Fold-by-4 Proof Protocol

    /// Commit phase using fold-by-4: fold iteratively by 4x, halving the number of
    /// Merkle tree commitments compared to fold-by-2.
    /// Each round consumes one beta and reduces domain by 4x.
    /// If logN is odd, one fold-by-2 is done first, then fold-by-4 for the rest.
    public func commitPhase4(evals: [Fr], betas: [Fr]) throws -> FRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        let oddStart = (logN % 2 != 0) ? 1 : 0
        let fold4Count = (logN - oddStart) / 2
        let totalBetas = oddStart + fold4Count
        precondition(betas.count >= totalBetas, "Need \(totalBetas) betas for fold-by-4 commit, got \(betas.count)")

        let merkle = merkleEngine
        let stride = MemoryLayout<Fr>.stride

        // Compute layer sizes: layer 0 = n, then each fold-by-4 reduces by 4x
        // (optionally first fold-by-2 reduces by 2x)
        var layerSizes: [Int] = [n]
        var curSize = n
        if oddStart == 1 {
            curSize /= 2
            layerSizes.append(curSize)
        }
        for _ in 0..<fold4Count {
            curSize /= 4
            layerSizes.append(curSize)
        }

        // Allocate input buffer
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

        // Allocate per-layer GPU buffers (cached across calls for same logN)
        let numFoldSteps = totalBetas
        if cachedLayer4BufsLogN != logN || cachedLayer4Bufs.count != numFoldSteps {
            cachedLayer4Bufs = []
            for i in 0..<numFoldSteps {
                let layerN = layerSizes[i + 1]
                guard let buf = device.makeBuffer(length: layerN * stride, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to create layer buffer")
                }
                cachedLayer4Bufs.append(buf)
            }
            cachedLayer4BufsLogN = logN
        }
        var layerBufs: [MTLBuffer] = [multiFoldInputBuf!]
        layerBufs.append(contentsOf: cachedLayer4Bufs)

        // Fold + Merkle: CHAINED into single command buffer.
        // Eliminates CPU-GPU round-trip between fold completion and merkle dispatch.
        var foldT0 = CFAbsoluteTimeGetCurrent()

        let subtreeSize = Poseidon2Engine.merkleSubtreeSize
        let p2 = merkle.p2Engine
        let cpuMerkleThreshold = 16

        let numMerkleRoots = totalBetas
        let rootsBufBytes = max(numMerkleRoots * stride, stride)
        if merkleRootsBuf == nil || merkleRootsBufSize < rootsBufBytes {
            merkleRootsBuf = device.makeBuffer(length: rootsBufBytes, options: .storageModeShared)
            merkleRootsBufSize = rootsBufBytes
        }
        guard let rootsBuf = merkleRootsBuf else {
            throw MSMError.gpuError("Failed to allocate merkle roots buffer")
        }

        let maxSubtreeRoots = max(layerSizes[1] / subtreeSize, 1)
        let subtreeRootBytes = maxSubtreeRoots * stride
        if merkleSubtreeRootsBuf == nil || merkleSubtreeRootsBufSize < subtreeRootBytes {
            merkleSubtreeRootsBuf = device.makeBuffer(length: subtreeRootBytes, options: .storageModeShared)
            merkleSubtreeRootsBufSize = subtreeRootBytes
        }
        guard let subtreeRootsBuf = merkleSubtreeRootsBuf else {
            throw MSMError.gpuError("Failed to allocate subtree roots buffer")
        }

        var roots = [Fr](repeating: Fr.zero, count: totalBetas)

        var hasGPUMerkle = false
        for i in 0..<totalBetas {
            if layerSizes[i + 1] > cpuMerkleThreshold { hasGPUMerkle = true; break }
        }

        // Single CB: all folds + all GPU merkle dispatches
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: Encode fold operations
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var betaIdx = 0
        var curLogN = logN

        if oddStart == 1 {
            let curN = layerSizes[0]
            let halfN = curN / 2
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)
            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(layerBufs[0], offset: 0, index: 0)
            enc.setBuffer(layerBufs[1], offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            curLogN -= 1
            betaIdx += 1
        }

        while betaIdx < totalBetas {
            let layerIdx = betaIdx
            let curN = layerSizes[layerIdx]
            let quarterN = curN / 4
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldBy4Function)
            enc.setBuffer(layerBufs[layerIdx], offset: 0, index: 0)
            enc.setBuffer(layerBufs[layerIdx + 1], offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            curLogN -= 2
            betaIdx += 1
            if betaIdx < totalBetas { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()

        // Phase 2: Encode GPU merkle into SAME command buffer
        if hasGPUMerkle {
            let mEnc = cmdBuf.makeComputeCommandEncoder()!
            var firstDispatch = true

            for i in 0..<totalBetas {
                let layerN = layerSizes[i + 1]
                if layerN <= cpuMerkleThreshold { continue }
                if !firstDispatch { mEnc.memoryBarrier(scope: .buffers) }
                firstDispatch = false
                encodeMerkleForLayer(encoder: mEnc, p2: p2, layerBuf: layerBufs[i + 1], layerN: layerN,
                                     rootsBuf: rootsBuf, rootOffset: i * stride,
                                     subtreeRootsBuf: subtreeRootsBuf, subtreeSize: subtreeSize, stride: stride)
            }
            mEnc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
        let foldTime = (CFAbsoluteTimeGetCurrent() - foldT0) * 1000
        let merkleT0 = CFAbsoluteTimeGetCurrent()

        // Read layers from GPU buffers (after completion)
        var layers: [[Fr]] = [evals]
        for i in 1..<layerSizes.count {
            let count = layerSizes[i]
            let ptr = layerBufs[i].contents().bindMemory(to: Fr.self, capacity: count)
            layers.append(Array(UnsafeBufferPointer(start: ptr, count: count)))
        }

        // CPU merkle for small layers
        for i in 0..<totalBetas {
            let layerN = layerSizes[i + 1]
            if layerN > cpuMerkleThreshold { continue }
            if layerN <= 1 {
                roots[i] = layers[i + 1][0]
            } else {
                layers[i + 1].withUnsafeBufferPointer { bp in
                    roots[i] = cpuMerkleRoot(bp)
                }
            }
        }

        // Read GPU merkle roots
        if hasGPUMerkle {
            let rootPtr = rootsBuf.contents().bindMemory(to: Fr.self, capacity: numMerkleRoots)
            for i in 0..<totalBetas {
                if layerSizes[i + 1] > cpuMerkleThreshold {
                    roots[i] = rootPtr[i]
                }
            }
        }

        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000
        if profileCommit {
            fputs(String(format: "  commitPhase4: fold %.1fms, merkle %.1fms, total %.1fms (%d layers)\n",
                        foldTime, merkleTime, foldTime + merkleTime, layerSizes.count), stderr)
        }

        let current = layers.last!
        let finalValue = current.count == 1 ? current[0] : current[0]

        return FRICommitment(
            layers: layers,
            roots: roots,
            betas: betas,
            finalValue: finalValue,
            foldMode: .foldBy4
        )
    }

    /// Query phase for fold-by-4 commitment: extract evaluation quartets and Merkle paths.
    /// For fold-by-4 layers, each query extracts 4 evaluations at stride N/4.
    /// For fold-by-2 layers (first layer when logN is odd), extracts pairs.
    public func queryPhase4(commitment: FRICommitment, queryIndices: [UInt32]) throws -> [FRIQueryProof] {
        let numQueries = queryIndices.count
        let merkle = merkleEngine

        // Pre-build Merkle trees for all layers (avoid rebuilding per query)
        var layerTrees = [[Fr]]()
        layerTrees.reserveCapacity(commitment.layers.count - 1)
        for layer in 0..<commitment.layers.count - 1 {
            layerTrees.append(try merkle.buildTree(commitment.layers[layer]))
        }

        var proofs = [FRIQueryProof]()
        proofs.reserveCapacity(numQueries)

        for qi in 0..<numQueries {
            var layerEvals: [(Fr, Fr)] = []
            var merklePaths: [[[Fr]]] = []
            var idx = queryIndices[qi]

            for layer in 0..<commitment.layers.count - 1 {
                let evals = commitment.layers[layer]
                let n = evals.count
                let nextN = commitment.layers[layer + 1].count
                let tree = layerTrees[layer]

                if nextN == n / 2 {
                    // Fold-by-2 layer
                    let halfN = UInt32(n / 2)
                    let lowerIdx = idx < halfN ? idx : idx - halfN
                    let upperIdx = lowerIdx + halfN
                    layerEvals.append((evals[Int(lowerIdx)], evals[Int(upperIdx)]))

                    let path = extractMerklePath(tree: tree, leafCount: n, index: Int(idx))
                    merklePaths.append([path])

                    idx = lowerIdx
                } else {
                    // Fold-by-4 layer: extract quartet at stride N/4
                    let quarterN = UInt32(n / 4)
                    let baseIdx = idx % quarterN
                    let e0 = evals[Int(baseIdx)]
                    let e1 = evals[Int(baseIdx + quarterN)]
                    let e2 = evals[Int(baseIdx + 2 * quarterN)]
                    let e3 = evals[Int(baseIdx + 3 * quarterN)]

                    // Store as two pairs for compatibility with FRIQueryProof
                    layerEvals.append((e0, e1))
                    layerEvals.append((e2, e3))

                    let path0 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx))
                    let path1 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + quarterN))
                    let path2 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + 2 * quarterN))
                    let path3 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + 3 * quarterN))
                    merklePaths.append([path0, path1, path2, path3])

                    idx = baseIdx
                }
            }

            proofs.append(FRIQueryProof(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    /// Verify a fold-by-4 FRI proof: check consistency of query responses.
    public func verify4(commitment: FRICommitment, queries: [FRIQueryProof]) -> Bool {
        for query in queries {
            var idx = query.initialIndex
            var evalIdx = 0  // index into layerEvals (fold-by-4 uses 2 entries per layer)

            for layer in 0..<commitment.layers.count - 1 {
                let n = commitment.layers[layer].count
                let nextN = commitment.layers[layer + 1].count

                if nextN == n / 2 {
                    // Fold-by-2 verification
                    let (evalA, evalB) = query.layerEvals[evalIdx]
                    evalIdx += 1
                    let halfN = UInt32(n / 2)
                    let logN = Int(log2(Double(n)))
                    let beta = commitment.betas[layer]

                    let omega = frRootOfUnity(logN: logN)
                    let omegaInv = frInverse(omega)
                    let lowerIdx = idx < halfN ? idx : idx - halfN
                    let w_inv = frPow(omegaInv, UInt64(lowerIdx))

                    let sum = frAdd(evalA, evalB)
                    let diff = frSub(evalA, evalB)
                    let term = frMul(frMul(beta, w_inv), diff)
                    let expected = frAdd(sum, term)

                    let nextEval = commitment.layers[layer + 1][Int(lowerIdx)]
                    if frToInt(expected) != frToInt(nextEval) { return false }
                    idx = lowerIdx
                } else {
                    // Fold-by-4 verification
                    let (e0, e1) = query.layerEvals[evalIdx]
                    let (e2, e3) = query.layerEvals[evalIdx + 1]
                    evalIdx += 2
                    let quarterN = UInt32(n / 4)
                    let logN = Int(log2(Double(n)))
                    let beta = commitment.betas[layer]

                    let baseIdx = idx % quarterN

                    // Replicate the cpuFold4 logic
                    let omega = frRootOfUnity(logN: logN)
                    let omegaInv = frInverse(omega)
                    let w4_inv = frPow(omegaInv, UInt64(quarterN))

                    let s02 = frAdd(e0, e2)
                    let d02 = frSub(e0, e2)
                    let s13 = frAdd(e1, e3)
                    let d13 = frSub(e1, e3)
                    let d13_w4inv = frMul(d13, w4_inv)

                    var t0 = frAdd(s02, s13)
                    var t1 = frAdd(d02, d13_w4inv)
                    var t2 = frSub(s02, s13)
                    var t3 = frSub(d02, d13_w4inv)

                    let inv_x = frPow(omegaInv, UInt64(baseIdx))
                    let inv_x2 = frMul(inv_x, inv_x)
                    let inv_x3 = frMul(inv_x2, inv_x)

                    t1 = frMul(t1, inv_x)
                    t2 = frMul(t2, inv_x2)
                    t3 = frMul(t3, inv_x3)

                    let r = beta
                    let r2 = frMul(r, r)
                    let r3 = frMul(r2, r)

                    var expected = frAdd(t0, frMul(r, t1))
                    expected = frAdd(expected, frMul(r2, t2))
                    expected = frAdd(expected, frMul(r3, t3))

                    let nextEval = commitment.layers[layer + 1][Int(baseIdx)]
                    if frToInt(expected) != frToInt(nextEval) { return false }
                    idx = baseIdx
                }
            }
        }
        return true
    }

    // MARK: - FRI Fold-by-8

    /// Perform one FRI fold-by-8 step: reduce domain size by 8x.
    /// Uses 8th-root-of-unity decomposition for the polynomial.
    /// Input: evaluations on domain of size n (must be divisible by 8)
    /// Output: folded evaluations of size n/8
    public func fold8(evals: MTLBuffer, folded: MTLBuffer, beta: Fr, logN: Int) throws {
        let n = UInt32(1 << logN)
        let eighth = Int(n) / 8
        let invTwiddles = getInvTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        var betaVal = beta
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldBy8Function)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(folded, offset: 0, index: 1)
        enc.setBuffer(invTwiddles, offset: 0, index: 2)
        enc.setBytes(&betaVal, length: MemoryLayout<Fr>.stride, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        let tg = min(tuning.friThreadgroupSize, Int(foldBy8Function.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: eighth, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    /// High-level FRI fold-by-8: array in, array out.
    public func fold8(evals: [Fr], beta: Fr) throws -> [Fr] {
        let n = evals.count
        precondition(n >= 8 && (n & (n - 1)) == 0, "Domain size must be power of 2 and >= 8")
        let logN = Int(log2(Double(n)))
        let eighth = n / 8
        let stride = MemoryLayout<Fr>.stride

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

        try fold8(evals: evalsBuf, folded: foldedBuf, beta: beta, logN: logN)

        let ptr = foldedBuf.contents().bindMemory(to: Fr.self, capacity: eighth)
        return Array(UnsafeBufferPointer(start: ptr, count: eighth))
    }

    /// Multi-round FRI using fold-by-8: fold repeatedly, each step divides by 8.
    /// If logN % 3 != 0, uses fold-by-4 or fold-by-2 for remainder, then fold-by-8.
    /// Uses a single command buffer with memory barriers.
    public func multiFold8(evals: [Fr], betas: [Fr]) throws -> [Fr] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        // For fold-by-8 we consume 1 beta per fold-by-8 step (each step = 3 log-levels)
        // Handle remainder: logN mod 3 can be 0, 1, or 2
        let remainder = logN % 3
        let fold8Count = (logN - remainder) / 3
        // remainder handled by fold-by-4 (2 levels) + fold-by-2 (1 level) or combinations
        let remainderBetas: Int
        switch remainder {
        case 0: remainderBetas = 0
        case 1: remainderBetas = 1   // one fold-by-2
        case 2: remainderBetas = 1   // one fold-by-4
        default: remainderBetas = 0
        }
        let totalBetas = remainderBetas + fold8Count
        precondition(betas.count >= totalBetas, "Need \(totalBetas) betas for fold-by-8, got \(betas.count)")

        try ensureFoldBuffers(maxElements: max(n / 8, 1))

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

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
        var useA = true
        var curLogN = logN
        var betaIdx = 0

        let enc = cmdBuf.makeComputeCommandEncoder()!

        // Handle remainder
        if remainder == 1 {
            // One fold-by-2
            let curN = 1 << curLogN
            let halfN = curN / 2
            let outputBuf = useA ? foldBufA! : foldBufB!
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA
            curLogN -= 1
            betaIdx += 1
            enc.memoryBarrier(scope: .buffers)
        } else if remainder == 2 {
            // One fold-by-4
            let curN = 1 << curLogN
            let quarterN = curN / 4
            let outputBuf = useA ? foldBufA! : foldBufB!
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldBy4Function)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA
            curLogN -= 2
            betaIdx += 1
            enc.memoryBarrier(scope: .buffers)
        }

        // Now curLogN is divisible by 3; fold-by-8 until done
        while betaIdx < totalBetas {
            let curN = 1 << curLogN
            let eighthN = curN / 8
            let outputBuf = useA ? foldBufA! : foldBufB!
            let invTwiddles = getInvTwiddles(logN: curLogN)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldBy8Function)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy8Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: eighthN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA
            curLogN -= 3
            betaIdx += 1

            if betaIdx < totalBetas {
                enc.memoryBarrier(scope: .buffers)
            }
        }
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let finalSize = 1 << curLogN
        let ptr = currentBuf.contents().bindMemory(to: Fr.self, capacity: finalSize)
        return Array(UnsafeBufferPointer(start: ptr, count: finalSize))
    }

    /// CPU FRI fold-by-8 for correctness verification.
    /// Mirrors the GPU kernel logic: 8th-root-of-unity decomposition.
    /// The /8 normalization factor is NOT applied (absorbed, same as fold-by-2/4).
    public static func cpuFold8(evals: [Fr], beta: Fr, logN: Int) -> [Fr] {
        let n = evals.count
        precondition(n >= 8 && (n & (n - 1)) == 0)
        let eighth = n / 8

        let omega = frRootOfUnity(logN: logN)
        let omegaInv = frInverse(omega)

        // w8 = omega^{N/8}, w8_inv = omega^{-N/8}
        // w4_inv = omega^{-N/4} = w8_inv^2
        // w8_inv3 = omega^{-3N/8}
        let w8_inv = frPow(omegaInv, UInt64(eighth))
        let w4_inv = frMul(w8_inv, w8_inv)
        let w8_inv3 = frMul(w4_inv, w8_inv)

        var folded = [Fr](repeating: Fr.zero, count: eighth)
        var inv_x = Fr.one

        for i in 0..<eighth {
            let e0 = evals[i]
            let e1 = evals[i + eighth]
            let e2 = evals[i + 2 * eighth]
            let e3 = evals[i + 3 * eighth]
            let e4 = evals[i + 4 * eighth]
            let e5 = evals[i + 5 * eighth]
            let e6 = evals[i + 6 * eighth]
            let e7 = evals[i + 7 * eighth]

            // Level 1: stride-4 butterflies
            let s0 = frAdd(e0, e4)
            let s1 = frAdd(e1, e5)
            let s2 = frAdd(e2, e6)
            let s3 = frAdd(e3, e7)
            let d0 = frSub(e0, e4)
            let d1 = frSub(e1, e5)
            let d2 = frSub(e2, e6)
            let d3 = frSub(e3, e7)

            // Level 2: even 4-point DFT
            let ss02 = frAdd(s0, s2)
            let ds02 = frSub(s0, s2)
            let ss13 = frAdd(s1, s3)
            let ds13 = frSub(s1, s3)
            let ds13_w4 = frMul(ds13, w4_inv)

            let T0 = frAdd(ss02, ss13)
            let T2 = frAdd(ds02, ds13_w4)
            let T4 = frSub(ss02, ss13)
            let T6 = frSub(ds02, ds13_w4)

            // Level 2: odd 4-point DFT with twiddles
            let d1_tw = frMul(d1, w8_inv)
            let d2_tw = frMul(d2, w4_inv)
            let d3_tw = frMul(d3, w8_inv3)

            let sd02 = frAdd(d0, d2_tw)
            let dd02 = frSub(d0, d2_tw)
            let sd13 = frAdd(d1_tw, d3_tw)
            let dd13 = frSub(d1_tw, d3_tw)
            let dd13_w4 = frMul(dd13, w4_inv)

            let T1 = frAdd(sd02, sd13)
            let T3 = frAdd(dd02, dd13_w4)
            let T5 = frSub(sd02, sd13)
            let T7 = frSub(dd02, dd13_w4)

            // Apply inv_x^k
            let inv_x2 = frMul(inv_x, inv_x)
            let inv_x3 = frMul(inv_x2, inv_x)
            let inv_x4 = frMul(inv_x2, inv_x2)
            let inv_x5 = frMul(inv_x4, inv_x)
            let inv_x6 = frMul(inv_x4, inv_x2)
            let inv_x7 = frMul(inv_x4, inv_x3)

            let t1 = frMul(T1, inv_x)
            let t2 = frMul(T2, inv_x2)
            let t3 = frMul(T3, inv_x3)
            let t4 = frMul(T4, inv_x4)
            let t5 = frMul(T5, inv_x5)
            let t6 = frMul(T6, inv_x6)
            let t7 = frMul(T7, inv_x7)

            // Horner: result = T0 + r*(t1 + r*(t2 + r*(t3 + r*(t4 + r*(t5 + r*(t6 + r*t7))))))
            let r = beta
            var result = frAdd(frMul(r, t7), t6)
            result = frAdd(frMul(r, result), t5)
            result = frAdd(frMul(r, result), t4)
            result = frAdd(frMul(r, result), t3)
            result = frAdd(frMul(r, result), t2)
            result = frAdd(frMul(r, result), t1)
            result = frAdd(frMul(r, result), T0)

            folded[i] = result
            inv_x = frMul(inv_x, omegaInv)
        }

        return folded
    }

    // MARK: - FRI Fold-by-8 Proof Protocol

    /// Commit phase using fold-by-8: fold iteratively by 8x, reducing round count by ~3x
    /// compared to fold-by-2. Each round consumes one beta and reduces domain by 8x.
    /// Remainder (logN mod 3) handled by fold-by-4 or fold-by-2.
    public func commitPhase8(evals: [Fr], betas: [Fr]) throws -> FRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))

        let remainder = logN % 3
        let fold8Count = (logN - remainder) / 3
        let remainderBetas: Int
        switch remainder {
        case 1: remainderBetas = 1   // one fold-by-2
        case 2: remainderBetas = 1   // one fold-by-4
        default: remainderBetas = 0
        }
        let totalBetas = remainderBetas + fold8Count
        precondition(betas.count >= totalBetas, "Need \(totalBetas) betas for fold-by-8 commit, got \(betas.count)")

        let merkle = merkleEngine
        let stride = MemoryLayout<Fr>.stride

        // Compute layer sizes
        var layerSizes: [Int] = [n]
        var curSize = n
        var curLog = logN
        if remainder == 1 {
            curSize /= 2
            curLog -= 1
            layerSizes.append(curSize)
        } else if remainder == 2 {
            curSize /= 4
            curLog -= 2
            layerSizes.append(curSize)
        }
        for _ in 0..<fold8Count {
            curSize /= 8
            curLog -= 3
            layerSizes.append(curSize)
        }

        // Allocate input buffer
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

        // Allocate per-layer GPU buffers (cached)
        let numFoldSteps = totalBetas
        if cachedLayer8BufsLogN != logN || cachedLayer8Bufs.count != numFoldSteps {
            cachedLayer8Bufs = []
            for i in 0..<numFoldSteps {
                let layerN = layerSizes[i + 1]
                guard let buf = device.makeBuffer(length: max(layerN * stride, stride), options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to create layer buffer")
                }
                cachedLayer8Bufs.append(buf)
            }
            cachedLayer8BufsLogN = logN
        }
        var layerBufs: [MTLBuffer] = [multiFoldInputBuf!]
        layerBufs.append(contentsOf: cachedLayer8Bufs)

        // Fold + Merkle: CHAINED into single command buffer.
        // Eliminates CPU-GPU round-trip between fold completion and merkle dispatch.
        var foldT0 = CFAbsoluteTimeGetCurrent()

        let subtreeSize = Poseidon2Engine.merkleSubtreeSize
        let p2 = merkle.p2Engine
        let cpuMerkleThreshold = 16

        let numMerkleRoots = totalBetas
        let rootsBufBytes = max(numMerkleRoots * stride, stride)
        if merkleRootsBuf == nil || merkleRootsBufSize < rootsBufBytes {
            merkleRootsBuf = device.makeBuffer(length: rootsBufBytes, options: .storageModeShared)
            merkleRootsBufSize = rootsBufBytes
        }
        guard let rootsBuf = merkleRootsBuf else {
            throw MSMError.gpuError("Failed to allocate merkle roots buffer")
        }

        let maxSubtreeRoots = max((layerSizes.count > 1 ? layerSizes[1] : 1) / subtreeSize, 1)
        let subtreeRootBytes = maxSubtreeRoots * stride
        if merkleSubtreeRootsBuf == nil || merkleSubtreeRootsBufSize < subtreeRootBytes {
            merkleSubtreeRootsBuf = device.makeBuffer(length: subtreeRootBytes, options: .storageModeShared)
            merkleSubtreeRootsBufSize = subtreeRootBytes
        }
        guard let subtreeRootsBuf = merkleSubtreeRootsBuf else {
            throw MSMError.gpuError("Failed to allocate subtree roots buffer")
        }

        var roots = [Fr](repeating: Fr.zero, count: totalBetas)

        var hasGPUMerkle = false
        for i in 0..<totalBetas {
            if layerSizes[i + 1] > cpuMerkleThreshold { hasGPUMerkle = true; break }
        }

        // Single CB: all folds + all GPU merkle dispatches
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Phase 1: Encode fold operations
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var betaIdx = 0
        curLog = logN

        if remainder == 1 {
            let curN = layerSizes[0]
            let halfN = curN / 2
            let invTwiddles = getInvTwiddles(logN: curLog)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)
            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(layerBufs[0], offset: 0, index: 0)
            enc.setBuffer(layerBufs[1], offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldFunction.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            curLog -= 1
            betaIdx += 1
        } else if remainder == 2 {
            let curN = layerSizes[0]
            let quarterN = curN / 4
            let invTwiddles = getInvTwiddles(logN: curLog)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)
            enc.setComputePipelineState(foldBy4Function)
            enc.setBuffer(layerBufs[0], offset: 0, index: 0)
            enc.setBuffer(layerBufs[1], offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy4Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: quarterN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            curLog -= 2
            betaIdx += 1
        }

        while betaIdx < totalBetas {
            let layerIdx = betaIdx
            let curN = layerSizes[layerIdx]
            let eighthN = curN / 8
            let invTwiddles = getInvTwiddles(logN: curLog)
            var beta = betas[betaIdx]
            var nVal = UInt32(curN)

            enc.setComputePipelineState(foldBy8Function)
            enc.setBuffer(layerBufs[layerIdx], offset: 0, index: 0)
            enc.setBuffer(layerBufs[layerIdx + 1], offset: 0, index: 1)
            enc.setBuffer(invTwiddles, offset: 0, index: 2)
            enc.setBytes(&beta, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
            let tg = min(tuning.friThreadgroupSize, Int(foldBy8Function.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: eighthN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            curLog -= 3
            betaIdx += 1
            if betaIdx < totalBetas { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()

        // Phase 2: Encode GPU merkle into SAME command buffer
        if hasGPUMerkle {
            let mEnc = cmdBuf.makeComputeCommandEncoder()!
            var firstDispatch = true

            for i in 0..<totalBetas {
                let layerN = layerSizes[i + 1]
                if layerN <= cpuMerkleThreshold { continue }
                if !firstDispatch { mEnc.memoryBarrier(scope: .buffers) }
                firstDispatch = false
                encodeMerkleForLayer(encoder: mEnc, p2: p2, layerBuf: layerBufs[i + 1], layerN: layerN,
                                     rootsBuf: rootsBuf, rootOffset: i * stride,
                                     subtreeRootsBuf: subtreeRootsBuf, subtreeSize: subtreeSize, stride: stride)
            }
            mEnc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
        let foldTime = (CFAbsoluteTimeGetCurrent() - foldT0) * 1000
        let merkleT0 = CFAbsoluteTimeGetCurrent()

        // Read layers from GPU buffers (after completion)
        var layers: [[Fr]] = [evals]
        for i in 1..<layerSizes.count {
            let count = layerSizes[i]
            let ptr = layerBufs[i].contents().bindMemory(to: Fr.self, capacity: count)
            layers.append(Array(UnsafeBufferPointer(start: ptr, count: count)))
        }

        // CPU merkle for small layers
        for i in 0..<totalBetas {
            let layerN = layerSizes[i + 1]
            if layerN > cpuMerkleThreshold { continue }
            if layerN <= 1 {
                roots[i] = layers[i + 1][0]
            } else {
                layers[i + 1].withUnsafeBufferPointer { bp in
                    roots[i] = cpuMerkleRoot(bp)
                }
            }
        }

        // Read GPU merkle roots
        if hasGPUMerkle {
            let rootPtr = rootsBuf.contents().bindMemory(to: Fr.self, capacity: numMerkleRoots)
            for i in 0..<totalBetas {
                if layerSizes[i + 1] > cpuMerkleThreshold {
                    roots[i] = rootPtr[i]
                }
            }
        }

        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000
        if profileCommit {
            fputs(String(format: "  commitPhase8: fold %.1fms, merkle %.1fms, total %.1fms (%d layers)\n",
                        foldTime, merkleTime, foldTime + merkleTime, layerSizes.count), stderr)
        }

        let current = layers.last!
        let finalValue = current.count >= 1 ? current[0] : Fr.zero

        return FRICommitment(
            layers: layers,
            roots: roots,
            betas: betas,
            finalValue: finalValue,
            foldMode: .foldBy8
        )
    }

    /// Query phase for fold-by-8 commitment: extract evaluation octets and Merkle paths.
    /// For fold-by-8 layers, each query extracts 8 evaluations at stride N/8.
    /// For fold-by-4 layers, extracts quartets. For fold-by-2, extracts pairs.
    public func queryPhase8(commitment: FRICommitment, queryIndices: [UInt32]) throws -> [FRIQueryProof] {
        let numQueries = queryIndices.count
        let merkle = merkleEngine

        // Pre-build Merkle trees for all layers (avoid rebuilding per query)
        var layerTrees = [[Fr]]()
        layerTrees.reserveCapacity(commitment.layers.count - 1)
        for layer in 0..<commitment.layers.count - 1 {
            layerTrees.append(try merkle.buildTree(commitment.layers[layer]))
        }

        var proofs = [FRIQueryProof]()
        proofs.reserveCapacity(numQueries)

        for qi in 0..<numQueries {
            var layerEvals: [(Fr, Fr)] = []
            var merklePaths: [[[Fr]]] = []
            var idx = queryIndices[qi]

            for layer in 0..<commitment.layers.count - 1 {
                let evals = commitment.layers[layer]
                let n = evals.count
                let nextN = commitment.layers[layer + 1].count
                let tree = layerTrees[layer]

                if nextN == n / 2 {
                    // Fold-by-2 layer
                    let halfN = UInt32(n / 2)
                    let lowerIdx = idx < halfN ? idx : idx - halfN
                    let upperIdx = lowerIdx + halfN
                    layerEvals.append((evals[Int(lowerIdx)], evals[Int(upperIdx)]))

                    let path = extractMerklePath(tree: tree, leafCount: n, index: Int(idx))
                    merklePaths.append([path])

                    idx = lowerIdx
                } else if nextN == n / 4 {
                    // Fold-by-4 layer
                    let quarterN = UInt32(n / 4)
                    let baseIdx = idx % quarterN
                    let e0 = evals[Int(baseIdx)]
                    let e1 = evals[Int(baseIdx + quarterN)]
                    let e2 = evals[Int(baseIdx + 2 * quarterN)]
                    let e3 = evals[Int(baseIdx + 3 * quarterN)]

                    layerEvals.append((e0, e1))
                    layerEvals.append((e2, e3))

                    let path0 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx))
                    let path1 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + quarterN))
                    let path2 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + 2 * quarterN))
                    let path3 = extractMerklePath(tree: tree, leafCount: n, index: Int(baseIdx + 3 * quarterN))
                    merklePaths.append([path0, path1, path2, path3])

                    idx = baseIdx
                } else {
                    // Fold-by-8 layer
                    let eighthN = UInt32(n / 8)
                    let baseIdx = idx % eighthN
                    var octEvals: [Fr] = []
                    var octPaths: [[Fr]] = []
                    for k in 0..<8 {
                        let evalIdx = Int(baseIdx + UInt32(k) * eighthN)
                        octEvals.append(evals[evalIdx])
                        octPaths.append(extractMerklePath(tree: tree, leafCount: n, index: evalIdx))
                    }

                    // Store as 4 pairs for compatibility with FRIQueryProof
                    layerEvals.append((octEvals[0], octEvals[1]))
                    layerEvals.append((octEvals[2], octEvals[3]))
                    layerEvals.append((octEvals[4], octEvals[5]))
                    layerEvals.append((octEvals[6], octEvals[7]))
                    merklePaths.append(octPaths)

                    idx = baseIdx
                }
            }

            proofs.append(FRIQueryProof(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    /// Verify a fold-by-8 FRI proof: check consistency of query responses.
    public func verify8(commitment: FRICommitment, queries: [FRIQueryProof]) -> Bool {
        for query in queries {
            var idx = query.initialIndex
            var evalIdx = 0

            for layer in 0..<commitment.layers.count - 1 {
                let n = commitment.layers[layer].count
                let nextN = commitment.layers[layer + 1].count

                if nextN == n / 2 {
                    // Fold-by-2 verification
                    let (evalA, evalB) = query.layerEvals[evalIdx]
                    evalIdx += 1
                    let halfN = UInt32(n / 2)
                    let logN = Int(log2(Double(n)))
                    let beta = commitment.betas[layer]

                    let omega = frRootOfUnity(logN: logN)
                    let omegaInv = frInverse(omega)
                    let lowerIdx = idx < halfN ? idx : idx - halfN
                    let w_inv = frPow(omegaInv, UInt64(lowerIdx))

                    let sum = frAdd(evalA, evalB)
                    let diff = frSub(evalA, evalB)
                    let term = frMul(frMul(beta, w_inv), diff)
                    let expected = frAdd(sum, term)

                    let nextEval = commitment.layers[layer + 1][Int(lowerIdx)]
                    if frToInt(expected) != frToInt(nextEval) { return false }
                    idx = lowerIdx
                } else if nextN == n / 4 {
                    // Fold-by-4 verification
                    let (e0, e1) = query.layerEvals[evalIdx]
                    let (e2, e3) = query.layerEvals[evalIdx + 1]
                    evalIdx += 2
                    let quarterN = UInt32(n / 4)
                    let logN = Int(log2(Double(n)))
                    let beta = commitment.betas[layer]
                    let baseIdx = idx % quarterN

                    let omega = frRootOfUnity(logN: logN)
                    let omegaInv = frInverse(omega)
                    let w4_inv = frPow(omegaInv, UInt64(quarterN))

                    let s02 = frAdd(e0, e2)
                    let d02 = frSub(e0, e2)
                    let s13 = frAdd(e1, e3)
                    let d13 = frSub(e1, e3)
                    let d13_w4inv = frMul(d13, w4_inv)

                    var t0 = frAdd(s02, s13)
                    var t1 = frAdd(d02, d13_w4inv)
                    var t2 = frSub(s02, s13)
                    var t3 = frSub(d02, d13_w4inv)

                    let inv_x = frPow(omegaInv, UInt64(baseIdx))
                    let inv_x2 = frMul(inv_x, inv_x)
                    let inv_x3 = frMul(inv_x2, inv_x)

                    t1 = frMul(t1, inv_x)
                    t2 = frMul(t2, inv_x2)
                    t3 = frMul(t3, inv_x3)

                    let r = beta
                    let r2 = frMul(r, r)
                    let r3 = frMul(r2, r)

                    var expected = frAdd(t0, frMul(r, t1))
                    expected = frAdd(expected, frMul(r2, t2))
                    expected = frAdd(expected, frMul(r3, t3))

                    let nextEval = commitment.layers[layer + 1][Int(baseIdx)]
                    if frToInt(expected) != frToInt(nextEval) { return false }
                    idx = baseIdx
                } else {
                    // Fold-by-8 verification
                    let e0 = query.layerEvals[evalIdx].0
                    let e1 = query.layerEvals[evalIdx].1
                    let e2 = query.layerEvals[evalIdx + 1].0
                    let e3 = query.layerEvals[evalIdx + 1].1
                    let e4 = query.layerEvals[evalIdx + 2].0
                    let e5 = query.layerEvals[evalIdx + 2].1
                    let e6 = query.layerEvals[evalIdx + 3].0
                    let e7 = query.layerEvals[evalIdx + 3].1
                    evalIdx += 4

                    let eighthN = UInt32(n / 8)
                    let logN = Int(log2(Double(n)))
                    let beta = commitment.betas[layer]
                    let baseIdx = idx % eighthN

                    let omega = frRootOfUnity(logN: logN)
                    let omegaInv = frInverse(omega)
                    let w8_inv = frPow(omegaInv, UInt64(eighthN))
                    let w4_inv = frMul(w8_inv, w8_inv)
                    let w8_inv3 = frMul(w4_inv, w8_inv)

                    // Same DFT as cpuFold8
                    let s0 = frAdd(e0, e4)
                    let s1 = frAdd(e1, e5)
                    let s2 = frAdd(e2, e6)
                    let s3 = frAdd(e3, e7)
                    let d0v = frSub(e0, e4)
                    let d1v = frSub(e1, e5)
                    let d2v = frSub(e2, e6)
                    let d3v = frSub(e3, e7)

                    let ss02 = frAdd(s0, s2)
                    let ds02 = frSub(s0, s2)
                    let ss13 = frAdd(s1, s3)
                    let ds13 = frSub(s1, s3)
                    let ds13_w4 = frMul(ds13, w4_inv)

                    let T0 = frAdd(ss02, ss13)
                    let T2 = frAdd(ds02, ds13_w4)
                    let T4 = frSub(ss02, ss13)
                    let T6 = frSub(ds02, ds13_w4)

                    let d1_tw = frMul(d1v, w8_inv)
                    let d2_tw = frMul(d2v, w4_inv)
                    let d3_tw = frMul(d3v, w8_inv3)

                    let sd02 = frAdd(d0v, d2_tw)
                    let dd02 = frSub(d0v, d2_tw)
                    let sd13 = frAdd(d1_tw, d3_tw)
                    let dd13 = frSub(d1_tw, d3_tw)
                    let dd13_w4 = frMul(dd13, w4_inv)

                    let T1 = frAdd(sd02, sd13)
                    let T3 = frAdd(dd02, dd13_w4)
                    let T5 = frSub(sd02, sd13)
                    let T7 = frSub(dd02, dd13_w4)

                    let inv_x = frPow(omegaInv, UInt64(baseIdx))
                    let inv_x2 = frMul(inv_x, inv_x)
                    let inv_x3 = frMul(inv_x2, inv_x)
                    let inv_x4 = frMul(inv_x2, inv_x2)
                    let inv_x5 = frMul(inv_x4, inv_x)
                    let inv_x6 = frMul(inv_x4, inv_x2)
                    let inv_x7 = frMul(inv_x4, inv_x3)

                    let t1 = frMul(T1, inv_x)
                    let t2 = frMul(T2, inv_x2)
                    let t3 = frMul(T3, inv_x3)
                    let t4 = frMul(T4, inv_x4)
                    let t5 = frMul(T5, inv_x5)
                    let t6 = frMul(T6, inv_x6)
                    let t7 = frMul(T7, inv_x7)

                    let r = beta
                    var expected = frAdd(frMul(r, t7), t6)
                    expected = frAdd(frMul(r, expected), t5)
                    expected = frAdd(frMul(r, expected), t4)
                    expected = frAdd(frMul(r, expected), t3)
                    expected = frAdd(frMul(r, expected), t2)
                    expected = frAdd(frMul(r, expected), t1)
                    expected = frAdd(frMul(r, expected), T0)

                    let nextEval = commitment.layers[layer + 1][Int(baseIdx)]
                    if frToInt(expected) != frToInt(nextEval) { return false }
                    idx = baseIdx
                }
            }
        }
        return true
    }

    // MARK: - Unified FRI API (mode-dispatched)

    /// Default fold mode. Set to `.foldBy8` for fewest rounds and best GPU throughput.
    /// Change to `.foldBy4` for balanced tradeoff or `.foldBy2` for maximum compatibility.
    public var defaultFoldMode: FRIFoldMode = .foldBy8

    /// Unified commit phase: dispatches to commitPhase, commitPhase4, or commitPhase8
    /// based on the current `defaultFoldMode`.
    ///
    /// Automatically computes the correct number of betas from the provided array.
    /// If `betas` has more entries than needed, only the required prefix is used.
    public func commit(evals: [Fr], betas: [Fr], mode: FRIFoldMode? = nil) throws -> FRICommitment {
        let m = mode ?? defaultFoldMode
        let logN = Int(log2(Double(evals.count)))
        let needed = m.betaCount(logN: logN)
        let useBetas = Array(betas.prefix(needed))
        switch m {
        case .foldBy2: return try commitPhase(evals: evals, betas: useBetas)
        case .foldBy4: return try commitPhase4(evals: evals, betas: useBetas)
        case .foldBy8: return try commitPhase8(evals: evals, betas: useBetas)
        }
    }

    /// Unified query phase: dispatches based on the commitment's structure.
    /// Automatically detects fold mode from layer size ratios.
    public func query(commitment: FRICommitment, queryIndices: [UInt32], mode: FRIFoldMode? = nil) throws -> [FRIQueryProof] {
        let m = mode ?? detectFoldMode(commitment: commitment)
        switch m {
        case .foldBy2: return try queryPhase(commitment: commitment, queryIndices: queryIndices)
        case .foldBy4: return try queryPhase4(commitment: commitment, queryIndices: queryIndices)
        case .foldBy8: return try queryPhase8(commitment: commitment, queryIndices: queryIndices)
        }
    }

    /// Unified verify: dispatches based on the commitment's structure.
    /// Automatically detects fold mode from layer size ratios.
    public func verifyProof(commitment: FRICommitment, queries: [FRIQueryProof], mode: FRIFoldMode? = nil) -> Bool {
        let m = mode ?? detectFoldMode(commitment: commitment)
        switch m {
        case .foldBy2: return verify(commitment: commitment, queries: queries)
        case .foldBy4: return verify4(commitment: commitment, queries: queries)
        case .foldBy8: return verify8(commitment: commitment, queries: queries)
        }
    }

    /// Detect fold mode from a commitment's stored mode or layer size progression.
    private func detectFoldMode(commitment: FRICommitment) -> FRIFoldMode {
        // Use stored mode if available (set by unified API or tagged commitPhase methods)
        if let mode = commitment.foldMode { return mode }
        // Fallback: infer from layer sizes
        let layers = commitment.layers
        guard layers.count >= 2 else { return .foldBy2 }
        for i in 0..<layers.count - 1 {
            let cur = layers[i].count
            let next = layers[i + 1].count
            if next == cur / 8 { return .foldBy8 }
            if next == cur / 4 { return .foldBy4 }
        }
        return .foldBy2
    }

    /// Batch query extraction on GPU: extract evaluation pairs for multiple query indices.
    public func batchQueryExtract(evals: [Fr], queryIndices: [UInt32]) throws -> [(Fr, Fr)] {
        let n = evals.count
        let numQueries = queryIndices.count

        guard let evalsBuf = createFrBuffer(evals),
              let idxBuf = device.makeBuffer(bytes: queryIndices, length: numQueries * 4, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: numQueries * 2 * MemoryLayout<Fr>.stride, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var qVal = UInt32(numQueries)
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(queryExtractFunction)
        enc.setBuffer(evalsBuf, offset: 0, index: 0)
        enc.setBuffer(idxBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&qVal, length: 4, index: 4)
        let tg = min(tuning.friThreadgroupSize, Int(queryExtractFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numQueries, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: Fr.self, capacity: numQueries * 2)
        var pairs = [(Fr, Fr)]()
        pairs.reserveCapacity(numQueries)
        for i in 0..<numQueries {
            pairs.append((ptr[i * 2], ptr[i * 2 + 1]))
        }
        return pairs
    }

    /// Extract a Merkle authentication path for a given leaf index.
    private func extractMerklePath(tree: [Fr], leafCount: Int, index: Int) -> [Fr] {
        let treeSize = 2 * leafCount - 1
        var path = [Fr]()
        var idx = index
        var levelStart = 0
        var levelSize = leafCount

        while levelSize > 1 {
            let siblingIdx = idx ^ 1  // flip last bit to get sibling
            if levelStart + siblingIdx < treeSize {
                path.append(tree[levelStart + siblingIdx])
            }
            idx /= 2
            levelStart += levelSize
            levelSize /= 2
        }
        return path
    }
}

// MARK: - FRI Fold Mode

/// Controls how many log-levels each FRI folding round reduces.
/// - `foldBy2`: Standard FRI, one fold per round (N -> N/2). Most rounds, most Merkle commits.
/// - `foldBy4`: Two log-levels per round (N -> N/4). Half the rounds of foldBy2. Default.
/// - `foldBy8`: Three log-levels per round (N -> N/8). Fewest rounds.
public enum FRIFoldMode: String, CaseIterable, Sendable {
    case foldBy2
    case foldBy4
    case foldBy8

    /// Number of betas needed for a given logN.
    public func betaCount(logN: Int) -> Int {
        switch self {
        case .foldBy2:
            return logN
        case .foldBy4:
            let oddStart = (logN % 2 != 0) ? 1 : 0
            return oddStart + (logN - oddStart) / 2
        case .foldBy8:
            let remainder = logN % 3
            let remainderBetas = remainder == 0 ? 0 : 1
            return remainderBetas + (logN - (remainder == 0 ? 0 : remainder)) / 3
        }
    }
}

// MARK: - FRI Proof Data Structures

/// Commitment produced during FRI commit phase.
public struct FRICommitment {
    /// Evaluations at each fold layer (layer 0 = original, layer k = after k folds)
    public let layers: [[Fr]]
    /// Poseidon2 Merkle root of each layer's evaluations
    public let roots: [Fr]
    /// Random challenges used at each fold round
    public let betas: [Fr]
    /// Final constant value after all folds
    public let finalValue: Fr
    /// Fold mode used to produce this commitment (nil for legacy commitments)
    public let foldMode: FRIFoldMode?

    public init(layers: [[Fr]], roots: [Fr], betas: [Fr], finalValue: Fr, foldMode: FRIFoldMode? = nil) {
        self.layers = layers
        self.roots = roots
        self.betas = betas
        self.finalValue = finalValue
        self.foldMode = foldMode
    }
}

/// Query proof for a single query index across all FRI layers.
public struct FRIQueryProof {
    /// The initial query index in the original domain
    public let initialIndex: UInt32
    /// Evaluation pairs (eval[idx], eval[paired_idx]) at each layer
    public let layerEvals: [(Fr, Fr)]
    /// Merkle authentication paths at each layer
    public let merklePaths: [[[Fr]]]

    public init(initialIndex: UInt32, layerEvals: [(Fr, Fr)], merklePaths: [[[Fr]]]) {
        self.initialIndex = initialIndex
        self.layerEvals = layerEvals
        self.merklePaths = merklePaths
    }
}
