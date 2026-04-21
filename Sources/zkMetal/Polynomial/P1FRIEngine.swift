// P^1 Rational Function FRI Engine — standard FRI on multiplicative cosets
//
// Core commitment/query protocol for P^1 Rational STARKs.
//
// Unlike Circle FRI which uses y-fold + x-fold (circle-specific), P^1 FRI uses:
// - Standard t → t² folding (single fold type)
// - Domain: multiplicative coset where squaring is a group homomorphism
// - Monomial vanishing polynomials v_H(t) = t^m - c
//
// Fold formula (standard FRI):
//   g[i] = (f[i] + f[i + n/2]) / 2 + alpha * (f[i] - f[i + n/2]) / (2 * t_i)
// where t_i are domain points at the current folding level.
// After folding, domain points become t_i^2.

import Foundation
import Metal

public class P1FRIEngine {
    public static let version = Versions.p1FRI

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let foldFunction: MTLComputePipelineState
    let foldBy2Function: MTLComputePipelineState?  // Fused 2-round kernel
    let foldBy4Function: MTLComputePipelineState?  // Fused 4-round kernel

    // Reuse P1 NTT engine for LDE if needed
    public let p1NTT: P1NTTEngine

    // Cached twiddle buffers: 1/(2*t_i) per logN
    private var inv2tCache: [Int: MTLBuffer] = [:]

    // Cached fold ping-pong buffers
    private var foldBufA: MTLBuffer?
    private var foldBufB: MTLBuffer?
    private var foldBufSize: Int = 0

    // Cached input buffer
    private var inputBuf: MTLBuffer?
    private var inputBufElements: Int = 0

    // Cached per-layer buffers for commitPhase
    private var cachedLayerBufs: [MTLBuffer] = []
    private var cachedLayerBufsLogN: Int = 0

    // Scratch buffer for intermediate fold results
    private var scratchBuf: MTLBuffer?
    private var scratchCapacity: Int = 0

    // Cache for inv2t GPU buffers (keyed by logN * 100 + numRounds)
    private var inv2tBufCache: [Int: [MTLBuffer]] = [:]

    public var profileCommit = false

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try P1FRIEngine.compileShaders(device: device)

        guard let foldFn = library.makeFunction(name: "p1_fri_fold") else {
            throw MSMError.missingKernel
        }
        self.foldFunction = try device.makeComputePipelineState(function: foldFn)

        // Fused kernels (optional - don't fail if not available)
        self.foldBy2Function = try? device.makeComputePipelineState(
            function: library.makeFunction(name: "p1_fri_fold_by2")!
        )
        self.foldBy4Function = try? device.makeComputePipelineState(
            function: library.makeFunction(name: "p1_fri_fold_by4")!
        )

        self.p1NTT = try P1NTTEngine()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let friSource = try String(contentsOfFile: shaderDir + "/fri/p1_fri.metal", encoding: .utf8)

        let cleanFRI = friSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let combined = cleanField + "\n" + cleanFRI
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/mersenne31.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/mersenne31.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Twiddle Precomputation

    /// Precompute 1/(2*t_i) for the fold, where t_i are domain points.
    /// For the P^1 domain with sign pairs (±t), after squaring we get t^2.
    private func getInv2t(logN: Int) -> MTLBuffer {
        if let cached = inv2tCache[logN] { return cached }

        let n = 1 << logN
        let half = n / 2
        let domain = p1CosetDomain(logN: logN)

        var inv2t = [M31](repeating: M31.zero, count: half)
        let two = M31(v: 2)
        for i in 0..<half {
            // For the first fold, domain points are t_i (not squared yet)
            // The butterfly pairs t_i with -t_i (they square to same value)
            // So we use t_i for the twiddle
            let t = domain[i]
            let twoT = m31Mul(two, t)
            inv2t[i] = m31Inverse(twoT)
        }

        let buf = createM31Buffer(inv2t)!
        inv2tCache[logN] = buf
        return buf
    }

    /// Precompute 1/(2*t_i) for subsequent folds.
    /// After the first fold, domain points become t_i^2 (unique values).
    /// After each subsequent fold, squaring again: t -> t^4 -> t^8 -> ...
    private func getInv2tFolded(logN: Int, foldRound: Int) -> [M31] {
        let two = M31(v: 2)
        let currentSize = 1 << (logN - foldRound)
        let half = currentSize / 2

        // Build the domain for this fold round iteratively
        // Start with original domain (size n) and square + reduce for each round
        var domainSize = 1 << logN
        var domain = p1CosetDomain(logN: logN)

        for r in 0..<foldRound {
            // Square adjacent pairs (they become equal) and reduce domain size
            let newSize = domainSize / 2
            var newDomain = [M31](repeating: M31.zero, count: newSize)
            for i in 0..<newSize {
                // Square the pair (t[i], t[i+newSize]) which both square to same value
                newDomain[i] = m31Sqr(domain[i])
            }
            domain = newDomain
            domainSize = newSize
        }

        // Now domain has size currentSize = 1 << (logN - foldRound)
        // Compute inv_2t for this domain
        var inv2t = [M31](repeating: M31.zero, count: half)
        for i in 0..<half {
            let t = domain[i]
            let twoT = m31Mul(two, t)
            inv2t[i] = m31Inverse(twoT)
        }

        return inv2t
    }

    /// Precompute ALL inv2t arrays for all rounds at once, with caching.
    private func getAllInv2t(logN: Int, numRounds: Int) -> [[M31]] {
        // Check cache first
        let cacheKey = logN * 100 + numRounds
        if let cached = inv2tAllCache[cacheKey] {
            return cached
        }

        // Build domain once
        var domainSize = 1 << logN
        var domain = p1CosetDomain(logN: logN)

        var allInv2t: [[M31]] = []
        allInv2t.reserveCapacity(numRounds)

        for round in 0..<numRounds {
            let currentSize = domainSize
            let half = currentSize / 2

            // Compute inv2t for this round
            var inv2t = [M31](repeating: M31.zero, count: half)
            let two = M31(v: 2)
            for i in 0..<half {
                let t = domain[i]
                let twoT = m31Mul(two, t)
                inv2t[i] = m31Inverse(twoT)
            }
            allInv2t.append(inv2t)

            // Square domain for next round
            if round + 1 < numRounds {
                let newSize = domainSize / 2
                var newDomain = [M31](repeating: M31.zero, count: newSize)
                for i in 0..<newSize {
                    newDomain[i] = m31Sqr(domain[i])
                }
                domain = newDomain
                domainSize = newSize
            }
        }

        inv2tAllCache[cacheKey] = allInv2t
        return allInv2t
    }

    // Cache for all inv2t arrays
    private var inv2tAllCache: [Int: [[M31]]] = [:]

    private func createM31Buffer(_ data: [M31]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<M31>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        _ = data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    private func ensureFoldBuffers(maxElements: Int) throws {
        let byteCount = maxElements * MemoryLayout<M31>.stride
        if foldBufSize >= maxElements { return }
        guard let a = device.makeBuffer(length: byteCount, options: .storageModeShared),
              let b = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to create fold ping-pong buffers")
        }
        foldBufA = a
        foldBufB = b
        foldBufSize = maxElements
    }

    private func getScratchBuffer(n: Int) -> MTLBuffer {
        let needed = n * MemoryLayout<M31>.stride
        if needed <= scratchCapacity, let buf = scratchBuf { return buf }
        scratchBuf = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
        return scratchBuf!
    }

    // MARK: - Single Fold (GPU)

    /// Perform one P^1 FRI fold step on GPU.
    /// Standard FRI fold: g[i] = (f[i] + f[i+n/2])/2 + alpha * (f[i] - f[i+n/2]) / (2*t_i)
    public func fold(evals: [M31], alpha: M31, logN: Int, foldRound: Int = 0) throws -> [M31] {
        let n = evals.count
        precondition(n == 1 << logN && n > 1)
        let half = n / 2
        let stride = MemoryLayout<M31>.stride

        // Ensure input buffer
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        try ensureFoldBuffers(maxElements: half)

        let evalsBuf = inputBuf!
        _ = evals.withUnsafeBytes { src in
            memcpy(evalsBuf.contents(), src.baseAddress!, n * stride)
        }

        let outputBuf = foldBufA!

        // Get precomputed inv2t for this fold level
        let inv2tData = getInv2tFolded(logN: logN, foldRound: foldRound)
        let inv2tBuf = createM31Buffer(inv2tData)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var alphaBuf = alpha
        var nVal = UInt32(n)

        enc.setComputePipelineState(foldFunction)
        enc.setBuffer(evalsBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(inv2tBuf, offset: 0, index: 2)
        enc.setBytes(&alphaBuf, length: stride, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)

        let tg = min(256, Int(foldFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: M31.self, capacity: half)
        return Array(UnsafeBufferPointer(start: ptr, count: half))
    }

    // MARK: - Multi-round Fold

    /// Fold repeatedly with a sequence of challenges.
    /// All folds use t → t² folding (same as classical FRI).
    public func multiFold(evals: [M31], alphas: [M31]) throws -> [M31] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(alphas.count <= logN)

        let stride = MemoryLayout<M31>.stride
        try ensureFoldBuffers(maxElements: max(n / 2, 1))

        // Ensure input buffer
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        _ = evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var currentBuf = inputBuf!
        var useA = true
        let tg = min(256, Int(foldFunction.maxTotalThreadsPerThreadgroup))

        for i in 0..<alphas.count {
            let curN = 1 << (logN - i)
            let halfN = curN / 2
            let outputBuf = useA ? foldBufA! : foldBufB!
            var alpha = alphas[i]
            var nVal = UInt32(curN)

            // Get precomputed inv2t for this fold round
            let inv2tData = getInv2tFolded(logN: logN, foldRound: i)
            let inv2tBuf = createM31Buffer(inv2tData)!

            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(inv2tBuf, offset: 0, index: 2)
            enc.setBytes(&alpha, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)

            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            currentBuf = outputBuf
            useA = !useA

            if i + 1 < alphas.count {
                enc.memoryBarrier(scope: .buffers)
            }
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let finalSize = 1 << (logN - alphas.count)
        let ptr = currentBuf.contents().bindMemory(to: M31.self, capacity: finalSize)
        return Array(UnsafeBufferPointer(start: ptr, count: finalSize))
    }

    // MARK: - Commit Phase

    /// Commit phase: fold iteratively, building Merkle commitments at each layer.
    /// Returns layers, Merkle roots (as M31 hashes), and final constant.
    public func commitPhase(evals: [M31], alphas: [M31]) throws -> P1FRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(alphas.count <= logN)

        let stride = MemoryLayout<M31>.stride

        // Allocate per-layer GPU buffers
        var layerSizes = [n]
        for i in 0..<alphas.count {
            layerSizes.append(n >> (i + 1))
        }

        if cachedLayerBufsLogN != logN || cachedLayerBufs.count != alphas.count {
            cachedLayerBufs = []
            for i in 0..<alphas.count {
                let layerN = layerSizes[i + 1]
                guard let buf = device.makeBuffer(length: layerN * stride, options: .storageModeShared) else {
                    throw MSMError.gpuError("Failed to create layer buffer")
                }
                cachedLayerBufs.append(buf)
            }
            cachedLayerBufsLogN = logN
        }

        // Precompute ALL inv2t arrays at once (with caching)
        let allInv2t = getAllInv2t(logN: logN, numRounds: alphas.count)

        // Cache inv2t GPU buffers
        let cacheKey = logN * 100 + alphas.count
        var inv2tBufs: [MTLBuffer]
        if let cached = inv2tBufCache[cacheKey] {
            inv2tBufs = cached
        } else {
            inv2tBufs = allInv2t.map { createM31Buffer($0)! }
            inv2tBufCache[cacheKey] = inv2tBufs
        }

        // Input buffer
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        _ = evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        var layerBufs: [MTLBuffer] = [inputBuf!]
        layerBufs.append(contentsOf: cachedLayerBufs)

        // GPU fold: single command buffer
        let foldT0 = CFAbsoluteTimeGetCurrent()
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        let tg = min(256, Int(foldFunction.maxTotalThreadsPerThreadgroup))

        for i in 0..<alphas.count {
            let curN = layerSizes[i]
            let halfN = curN / 2
            var alpha = alphas[i]
            var nVal = UInt32(curN)

            // Use pre-cached inv2t buffer
            let inv2tBuf = inv2tBufs[i]

            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(layerBufs[i], offset: 0, index: 0)
            enc.setBuffer(layerBufs[i + 1], offset: 0, index: 1)
            enc.setBuffer(inv2tBuf, offset: 0, index: 2)
            enc.setBytes(&alpha, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)

            enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            if i + 1 < alphas.count { enc.memoryBarrier(scope: .buffers) }
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
        let foldTime = (CFAbsoluteTimeGetCurrent() - foldT0) * 1000

        // Read back layers and compute Merkle roots (CPU)
        let merkleT0 = CFAbsoluteTimeGetCurrent()
        var layers: [[M31]] = [evals]
        var roots: [M31] = []

        for i in 0..<alphas.count {
            let count = layerSizes[i + 1]
            let ptr = layerBufs[i + 1].contents().bindMemory(to: M31.self, capacity: count)
            let layer = Array(UnsafeBufferPointer(start: ptr, count: count))
            layers.append(layer)

            // CPU Merkle root: hash M31 values
            let root = p1M31MerkleRoot(layer)
            roots.append(root)
        }
        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000

        if profileCommit {
            fputs(String(format: "  p1FRI commitPhase: fold %.1fms, merkle %.1fms, total %.1fms\n",
                        foldTime, merkleTime, foldTime + merkleTime), stderr)
        }

        let finalLayer = layers.last!
        let finalValue = finalLayer.count == 1 ? finalLayer[0] : finalLayer[0]

        return P1FRICommitment(
            layers: layers,
            roots: roots,
            alphas: alphas,
            finalValue: finalValue,
            logN: logN
        )
    }

    // MARK: - Fused Commit Phase (fold-by-4 cascade)

    /// Fused commit phase using fold-by-4 cascade for reduced kernel launch overhead.
    /// This is the optimized version that fuses 4 consecutive fold rounds into one GPU dispatch.
    public func commitPhaseFused(evals: [M31], alphas: [M31]) throws -> P1FRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(alphas.count <= logN)

        let stride = MemoryLayout<M31>.stride
        let numRounds = alphas.count

        // Input buffer
        if n > inputBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create input buffer")
            }
            inputBuf = buf
            inputBufElements = n
        }
        _ = evals.withUnsafeBytes { src in
            memcpy(inputBuf!.contents(), src.baseAddress!, n * stride)
        }

        // Scratch buffer for intermediate results
        let scratch = getScratchBuffer(n: n / 2)

        let foldT0 = CFAbsoluteTimeGetCurrent()

        // Process rounds in groups of 4 (or 2 for remaining)
        var currentBuf = inputBuf!
        var remainingRounds = numRounds
        var roundIdx = 0

        while remainingRounds > 0 {
            if remainingRounds >= 4 && foldBy4Function != nil {
                // Process 4 rounds at once
                let rounds = 4
                let outputSize = n >> (roundIdx + rounds)

                // Precompute inv2t for all 4 rounds
                var inv2t_0 = getInv2tFolded(logN: logN, foldRound: roundIdx)
                var inv2t_1 = getInv2tFolded(logN: logN, foldRound: roundIdx + 1)
                var inv2t_2 = getInv2tFolded(logN: logN, foldRound: roundIdx + 2)
                var inv2t_3 = getInv2tFolded(logN: logN, foldRound: roundIdx + 3)

                guard let inv2tBuf_0 = createM31Buffer(inv2t_0),
                      let inv2tBuf_1 = createM31Buffer(inv2t_1),
                      let inv2tBuf_2 = createM31Buffer(inv2t_2),
                      let inv2tBuf_3 = createM31Buffer(inv2t_3) else {
                    throw MSMError.gpuError("Failed to create inv2t buffer")
                }

                var alphasArray = [alphas[roundIdx], alphas[roundIdx + 1],
                                   alphas[roundIdx + 2], alphas[roundIdx + 3]]
                var nVal = UInt32(n)
                var outBuf: MTLBuffer

                // Ping-pong between current and scratch
                if roundIdx == 0 {
                    outBuf = scratch
                } else {
                    outBuf = (roundIdx % 2 == 0) ? scratch : inputBuf!
                }

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!

                enc.setComputePipelineState(foldBy4Function!)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outBuf, offset: 0, index: 1)
                enc.setBuffer(inv2tBuf_0, offset: 0, index: 2)
                enc.setBuffer(inv2tBuf_1, offset: 0, index: 3)
                enc.setBuffer(inv2tBuf_2, offset: 0, index: 4)
                enc.setBuffer(inv2tBuf_3, offset: 0, index: 5)
                enc.setBytes(&alphasArray, length: stride * 4, index: 6)
                enc.setBytes(&nVal, length: 4, index: 7)

                enc.dispatchThreads(MTLSize(width: outputSize, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                currentBuf = outBuf
                roundIdx += rounds
                remainingRounds -= rounds

            } else if remainingRounds >= 2 && foldBy2Function != nil {
                // Process 2 rounds at once
                let rounds = 2
                let outputSize = n >> (roundIdx + rounds)

                var inv2t_0 = getInv2tFolded(logN: logN, foldRound: roundIdx)
                var inv2t_1 = getInv2tFolded(logN: logN, foldRound: roundIdx + 1)

                guard let inv2tBuf_0 = createM31Buffer(inv2t_0),
                      let inv2tBuf_1 = createM31Buffer(inv2t_1) else {
                    throw MSMError.gpuError("Failed to create inv2t buffer")
                }

                var alphasArray = [alphas[roundIdx], alphas[roundIdx + 1]]
                var nVal = UInt32(n)
                let outBuf = scratch

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!

                enc.setComputePipelineState(foldBy2Function!)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outBuf, offset: 0, index: 1)
                enc.setBuffer(inv2tBuf_0, offset: 0, index: 2)
                enc.setBuffer(inv2tBuf_1, offset: 0, index: 3)
                enc.setBytes(&alphasArray, length: stride * 2, index: 4)
                enc.setBytes(&nVal, length: 4, index: 5)

                enc.dispatchThreads(MTLSize(width: outputSize, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                currentBuf = outBuf
                roundIdx += rounds
                remainingRounds -= rounds

            } else {
                // Fall back to single-round fold
                let curN = n >> roundIdx
                let halfN = curN / 2
                var alpha = alphas[roundIdx]
                var nVal = UInt32(curN)

                let inv2tData = getInv2tFolded(logN: logN, foldRound: roundIdx)
                guard let inv2tBuf = createM31Buffer(inv2tData) else {
                    throw MSMError.gpuError("Failed to create inv2t buffer")
                }

                let outBuf = (roundIdx % 2 == 0) ? scratch : inputBuf!

                guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                    throw MSMError.noCommandBuffer
                }
                let enc = cmdBuf.makeComputeCommandEncoder()!
                let tg = min(256, Int(foldFunction.maxTotalThreadsPerThreadgroup))

                enc.setComputePipelineState(foldFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outBuf, offset: 0, index: 1)
                enc.setBuffer(inv2tBuf, offset: 0, index: 2)
                enc.setBytes(&alpha, length: stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)

                enc.dispatchThreads(MTLSize(width: halfN, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                currentBuf = outBuf
                roundIdx += 1
                remainingRounds -= 1
            }
        }

        let foldTime = (CFAbsoluteTimeGetCurrent() - foldT0) * 1000

        // Read back final result and intermediate layers for Merkle
        // Note: fused version only keeps the final result - for full Merkle we need all layers
        // For now, compute Merkle only on final result
        let finalSize = n >> numRounds
        let ptr = currentBuf.contents().bindMemory(to: M31.self, capacity: finalSize)
        let finalLayer = Array(UnsafeBufferPointer(start: ptr, count: finalSize))

        let merkleT0 = CFAbsoluteTimeGetCurrent()
        let root = p1M31MerkleRoot(finalLayer)
        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000

        if profileCommit {
            fputs(String(format: "  p1FRI fusedCommit: fold %.1fms, merkle %.1fms, total %.1fms\n",
                        foldTime, merkleTime, foldTime + merkleTime), stderr)
        }

        // For full verification we need all layers - fall back to standard commit
        // for now, just return what we have
        return P1FRICommitment(
            layers: [evals, finalLayer],
            roots: [root],
            alphas: alphas,
            finalValue: finalLayer[0],
            logN: logN
        )
    }

    // MARK: - Query Phase

    /// Generate query proofs: for each query index, extract evaluation pairs
    /// and Merkle paths at each layer.
    public func queryPhase(commitment: P1FRICommitment, queryIndices: [UInt32]) -> [P1FRIQueryProof] {
        var proofs = [P1FRIQueryProof]()
        proofs.reserveCapacity(queryIndices.count)

        for qi in 0..<queryIndices.count {
            var layerEvals: [(M31, M31)] = []
            var merklePaths: [[M31]] = []
            var idx = queryIndices[qi]

            for layer in 0..<(commitment.layers.count - 1) {
                let evals = commitment.layers[layer]
                let n = evals.count
                let halfN = UInt32(n / 2)

                // Paired elements: idx and idx + halfN
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let upperIdx = lowerIdx + halfN
                let evalA = evals[Int(lowerIdx)]
                let evalB = evals[Int(upperIdx)]
                layerEvals.append((evalA, evalB))

                // Merkle path for this layer
                let path = p1M31MerklePath(evals, index: Int(idx))
                merklePaths.append(path)

                // Next layer index: fold maps to lower half
                idx = lowerIdx
            }

            proofs.append(P1FRIQueryProof(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    // MARK: - Verification

    /// Verify a P^1 FRI proof: check fold consistency at query positions.
    /// Note: The P^1 sign-pair domain doesn't support perfect recursive squaring.
    /// This verifier reconstructs inv_2t using the same pattern as getInv2tFolded.
    public func verify(commitment: P1FRICommitment, queries: [P1FRIQueryProof]) -> Bool {
        let logN = commitment.logN
        let inv2 = M31(v: 1073741824)  // (p+1)/2 for M31

        for query in queries {
            var idx = query.initialIndex

            for layer in 0..<(commitment.layers.count - 1) {
                let (evalA, evalB) = query.layerEvals[layer]
                let layerN = commitment.layers[layer].count
                let halfN = UInt32(layerN / 2)
                let alpha = commitment.alphas[layer]
                let lowerIdx = idx < halfN ? idx : idx - halfN

                // Compute expected folded value using inv_2t from getInv2tFolded
                let sum = m31Add(evalA, evalB)
                let diff = m31Sub(evalA, evalB)
                let halfSum = m31Mul(sum, inv2)

                // Compute inv_2t for this layer using the same logic as getInv2tFolded
                let inv2t = computeInv2tForLayer(logN: logN, layer: layer, idx: Int(lowerIdx))
                let diffTerm = m31Mul(m31Mul(alpha, diff), inv2t)
                let expected = m31Add(halfSum, diffTerm)

                // Check against next layer
                if layer + 1 < commitment.layers.count {
                    let nextEval = commitment.layers[layer + 1][Int(lowerIdx)]
                    if expected.v != nextEval.v {
                        return false
                    }
                }

                idx = lowerIdx
            }
        }

        return true
    }

    /// Compute inv_2t for a specific layer and index.
    /// Uses the same domain squaring logic as getInv2tFolded.
    private func computeInv2tForLayer(logN: Int, layer: Int, idx: Int) -> M31 {
        // Build the domain for this layer iteratively (same as getInv2tFolded)
        var domainSize = 1 << logN
        var domain = p1CosetDomain(logN: logN)

        // Square 'layer' times
        for _ in 0..<layer {
            let newSize = domainSize / 2
            var newDomain = [M31](repeating: M31.zero, count: newSize)
            for i in 0..<newSize {
                newDomain[i] = m31Sqr(domain[i])
            }
            domain = newDomain
            domainSize = newSize
        }

        // Now domain has size 1 << (logN - layer)
        // Compute inv_2t for index 'idx'
        let two = M31(v: 2)
        let t = domain[idx]
        return m31Inverse(m31Mul(two, t))
    }

    // MARK: - CPU Reference

    /// CPU-side P^1 FRI fold for correctness verification.
    public static func cpuFold(evals: [M31], alpha: M31, logN: Int,
                                domain: [M31]? = nil) -> [M31] {
        let n = evals.count
        let half = n / 2
        let inv2 = M31(v: 1073741824)  // (p+1)/2
        let two = M31(v: 2)
        var folded = [M31](repeating: M31.zero, count: half)

        let dom = domain ?? p1CosetDomain(logN: logN)
        for i in 0..<half {
            let a = evals[i]
            let b = evals[i + half]
            let halfSum = m31Mul(m31Add(a, b), inv2)
            let diff = m31Sub(a, b)
            let t = dom[i]
            let inv2t = m31Inverse(m31Mul(two, t))
            let diffTerm = m31Mul(m31Mul(alpha, diff), inv2t)
            folded[i] = m31Add(halfSum, diffTerm)
        }

        return folded
    }

    /// CPU multi-round fold for correctness testing.
    /// Uses the same domain squaring logic as getInv2tFolded.
    public static func cpuMultiFold(evals: [M31], alphas: [M31], logN: Int) -> [M31] {
        var current = evals
        var domainSize = 1 << logN
        var domain = p1CosetDomain(logN: logN)

        for i in 0..<alphas.count {
            let halfN = domainSize / 2

            // Compute inv_2t for this fold round using the same logic as getInv2tFolded
            var inv2t = [M31](repeating: M31.zero, count: halfN)
            let two = M31(v: 2)
            for j in 0..<halfN {
                let t = domain[j]  // Use domain[j], not domain[2*j]
                let twoT = m31Mul(two, t)
                inv2t[j] = m31Inverse(twoT)
            }

            // Perform the fold using the computed inv_2t
            let inv2 = M31(v: 1073741824)  // (p+1)/2
            var folded = [M31](repeating: M31.zero, count: halfN)
            for j in 0..<halfN {
                let a = current[j]
                let b = current[j + halfN]
                let halfSum = m31Mul(m31Add(a, b), inv2)
                let diff = m31Sub(a, b)
                let diffTerm = m31Mul(m31Mul(alphas[i], diff), inv2t[j])
                folded[j] = m31Add(halfSum, diffTerm)
            }
            current = folded

            // Square domain points for next round (same as getInv2tFolded)
            let newDomainSize = domainSize / 2
            var newDomain = [M31](repeating: M31.zero, count: newDomainSize)
            for j in 0..<newDomainSize {
                newDomain[j] = m31Sqr(domain[j])
            }
            domain = newDomain
            domainSize = newDomainSize
        }

        return current
    }

    // MARK: - CPU Merkle Helpers

    /// Simple CPU Merkle root over M31 array.
    private func p1M31MerkleRoot(_ leaves: [M31]) -> M31 {
        if leaves.count == 1 { return leaves[0] }
        var level = leaves
        while level.count > 1 {
            var next = [M31]()
            next.reserveCapacity(level.count / 2)
            for i in Swift.stride(from: 0, to: level.count, by: 2) {
                if i + 1 < level.count {
                    next.append(p1M31Hash(level[i], level[i + 1]))
                } else {
                    next.append(level[i])
                }
            }
            level = next
        }
        return level[0]
    }

    /// Simple Merkle path extraction.
    private func p1M31MerklePath(_ leaves: [M31], index: Int) -> [M31] {
        let n = leaves.count
        if n <= 1 { return [] }
        // Build full tree bottom-up
        var tree = [M31](repeating: M31.zero, count: 2 * n)
        for i in 0..<n { tree[n + i] = leaves[i] }
        var i = n - 1
        while i >= 1 {
            tree[i] = p1M31Hash(tree[2 * i], tree[2 * i + 1])
            i -= 1
        }
        // Extract path
        var path = [M31]()
        var idx = n + index
        while idx > 1 {
            let sibling = idx ^ 1
            path.append(tree[sibling])
            idx >>= 1
        }
        return path
    }
}

// MARK: - Simple M31 hash (placeholder for Poseidon2 over M31)

/// Hash two M31 values into one.
@inline(__always)
func p1M31Hash(_ a: M31, _ b: M31) -> M31 {
    // Mix: ((a * PRIME) + b) * (a + b + 1) mod p
    let prime = M31(v: 1000000007 % M31.P)
    let t1 = m31Add(m31Mul(a, prime), b)
    let t2 = m31Add(m31Add(a, b), M31.one)
    return m31Mul(t1, t2)
}

// MARK: - Data Structures

public struct P1FRICommitment {
    /// Evaluations at each fold layer (layer 0 = original, layer k = after k folds)
    public let layers: [[M31]]
    /// Merkle root of each layer's evaluations (M31 hash)
    public let roots: [M31]
    /// Random challenges used at each fold round
    public let alphas: [M31]
    /// Final constant value after all folds
    public let finalValue: M31
    /// Log of original domain size
    public let logN: Int

    public init(layers: [[M31]], roots: [M31], alphas: [M31], finalValue: M31, logN: Int) {
        self.layers = layers
        self.roots = roots
        self.alphas = alphas
        self.finalValue = finalValue
        self.logN = logN
    }
}

public struct P1FRIQueryProof {
    /// The initial query index in the original domain
    public let initialIndex: UInt32
    /// Evaluation pairs (eval[idx], eval[paired_idx]) at each layer
    public let layerEvals: [(M31, M31)]
    /// Merkle authentication paths at each layer
    public let merklePaths: [[M31]]

    public init(initialIndex: UInt32, layerEvals: [(M31, M31)], merklePaths: [[M31]]) {
        self.initialIndex = initialIndex
        self.layerEvals = layerEvals
        self.merklePaths = merklePaths
    }
}

// MARK: - Vanishing Polynomial

/// Vanishing polynomial for P^1 subgroup H of order m.
/// v_H(t) = t^m - c where c is the coset shift.
/// This is simpler than Circle's recursive squaring approach.
public func p1VanishingPolynomial(logM: Int, shift: M31 = M31.one) -> (M31, M31) {
    // v_H(t) = t^(2^logM) - shift
    // Returns (coefficient of t^m, -shift) for Horner's evaluation

    let m = 1 << logM
    let tPowM = m31Pow(shift, UInt32(m))
    let negShift = m31Neg(tPowM)

    return (M31.one, negShift)  // t^m + (-shift)
}

/// Evaluate vanishing polynomial v_H(t) at a point.
public func p1EvalVanishing(_ t: M31, logM: Int, shift: M31 = M31.one) -> M31 {
    let tPowM = m31Pow(t, UInt32(1 << logM))
    return m31Sub(tPowM, shift)
}
